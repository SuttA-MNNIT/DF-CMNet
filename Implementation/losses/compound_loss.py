import torch
import torch.nn as nn
import torch.nn.functional as F


class BalancedFocalLoss(nn.Module):
    """
    Balanced Focal Loss for dense pixel prediction.
    Penalizes false alarms on authentic pixels while focusing on hard tampered regions.
    """
    def __init__(self, alpha=0.5, gamma=2.0, reduction='mean'):
        super(BalancedFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        p = torch.sigmoid(logits)
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = p * targets + (1.0 - p) * (1.0 - targets)
        loss = ce_loss * ((1.0 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
            loss = alpha_t * loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class StableSoftIoULoss(nn.Module):
    """
    Regime-Decoupled Soft-IoU Loss with Authentic Suppression Penalty (ASP).
    Supports three overlap modes for rigorous ablation testing:
      - 'proposed_asp': Foreground Soft-IoU on tampered samples (||Y||_1 > 0) +
                        quadratic ASP penalty (lambda_asp * mean(p^2)) on authentic samples (||Y||_1 == 0).
      - 'naive_iou':    Standard Naive Soft-IoU with epsilon=1e-5 across all samples (Ablation V6).
      - 'dice':         Standard Dice loss with epsilon=1e-5 across all samples (Ablation V7).
    """
    def __init__(self, smooth=1e-5, lambda_asp=0.5, mode='proposed_asp'):
        super(StableSoftIoULoss, self).__init__()
        self.smooth = smooth
        self.lambda_asp = lambda_asp
        self.mode = mode

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        B = probs.shape[0]

        probs_flat = probs.view(B, -1)
        targets_flat = targets.view(B, -1)

        loss = 0.0
        for i in range(B):
            p = probs_flat[i]
            t = targets_flat[i]
            t_sum = t.sum()

            if self.mode == 'dice':
                intersection = (p * t).sum()
                denom = (p ** 2).sum() + (t ** 2).sum()
                dice = (2.0 * intersection + 1e-5) / (denom + 1e-5)
                loss += (1.0 - dice)
            elif self.mode == 'naive_iou':
                intersection = (p * t).sum()
                union = p.sum() + t_sum - intersection
                iou = (intersection + 1e-5) / (union + 1e-5)
                loss += (1.0 - iou)
            else:
                # Proposed Regime-Decoupled Soft-IoU + Quadratic ASP
                if t_sum == 0:
                    loss += self.lambda_asp * torch.mean(p ** 2)
                else:
                    intersection = (p * t).sum()
                    union = p.sum() + t_sum - intersection
                    iou = (intersection + self.smooth) / (union + self.smooth)
                    loss += (1.0 - iou)

        return loss / B


def compute_morphological_boundary(mask, kernel_size=5):
    """
    Dynamic morphological boundary extraction (default 5x5 structuring element, r_d = 2 px):
      B_gt = (Y dilate S_r) - (Y erode S_r)
    """
    padding = kernel_size // 2
    dilation = F.max_pool2d(mask, kernel_size=kernel_size, stride=1, padding=padding)
    erosion = -F.max_pool2d(-mask, kernel_size=kernel_size, stride=1, padding=padding)
    boundary = torch.clamp(dilation - erosion, 0.0, 1.0)
    return boundary


class CompoundForgeryLoss(nn.Module):
    """
    Multi-Task Compound Objective for DF-CMNet v2:
      L_total = lambda_foc * L_Focal + lambda_iou * L_IoU^+ + lambda_asp * L_ASP
              + lambda_bound * L_Edge + lambda_aux * L_aux
    """
    def __init__(
        self,
        lambda_foc=1.0,
        lambda_iou=1.0,
        lambda_bound=0.5,
        lambda_aux=0.4,
        lambda_asp=0.5,
        lambda_dice=None,
        boundary_kernel_size=5,
        overlap_mode='proposed_asp'
    ):
        super(CompoundForgeryLoss, self).__init__()
        self.lambda_foc = lambda_foc
        self.lambda_iou = lambda_dice if lambda_dice is not None else lambda_iou
        self.lambda_bound = lambda_bound
        self.lambda_aux = lambda_aux
        self.lambda_asp = lambda_asp
        self.boundary_kernel_size = boundary_kernel_size
        self.overlap_mode = overlap_mode

        self.focal = BalancedFocalLoss(alpha=0.5, gamma=2.0)
        self.soft_iou = StableSoftIoULoss(smooth=1e-5, lambda_asp=lambda_asp, mode=overlap_mode)
        self.bce_edge = nn.BCEWithLogitsLoss()

    def forward(self, outputs, target_masks):
        mask_logits = outputs['mask_logits']
        edge_logits = outputs['edge_logits']

        # 1. Main mask losses (Focal + Regime-Decoupled Overlap/ASP)
        loss_focal = self.focal(mask_logits, target_masks)
        loss_iou = self.soft_iou(mask_logits, target_masks)

        # 2. Auxiliary Deep Supervision at scales 1/16 and 1/8
        aux_loss = torch.tensor(0.0, device=mask_logits.device, dtype=mask_logits.dtype)
        if 'aux_logits' in outputs and outputs['aux_logits'] and self.lambda_aux > 0:
            w_each = self.lambda_aux / max(len(outputs['aux_logits']), 1)
            for aux in outputs['aux_logits']:
                aux_target = F.interpolate(target_masks, size=aux.shape[2:], mode='nearest')
                aux_loss = aux_loss + w_each * (self.focal(aux, aux_target) + self.soft_iou(aux, aux_target))

        # 3. Dynamic morphological boundary loss (BCE + Soft-IoU on B_gt)
        loss_edge = torch.tensor(0.0, device=mask_logits.device, dtype=mask_logits.dtype)
        if self.lambda_bound > 0 and edge_logits.requires_grad:
            edge_gt = compute_morphological_boundary(target_masks, kernel_size=self.boundary_kernel_size)
            loss_edge_bce = self.bce_edge(edge_logits, edge_gt)
            loss_edge_iou = self.soft_iou(edge_logits, edge_gt)
            loss_edge = loss_edge_bce + loss_edge_iou

        total_loss = (
            self.lambda_foc * loss_focal
            + self.lambda_iou * loss_iou
            + self.lambda_bound * loss_edge
            + aux_loss
        )

        return total_loss, {
            'loss_total': float(total_loss.detach().item()),
            'loss_focal': float(loss_focal.detach().item()),
            'loss_iou': float(loss_iou.detach().item()),
            'loss_edge': float(loss_edge.detach().item())
        }
