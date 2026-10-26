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
    Stable Soft-IoU Loss with Authentic Suppression Penalty (ASP).
    Solves the zero-gradient issue of standard Dice loss on negative/authentic images.
    """
    def __init__(self, smooth=1.0):
        super(StableSoftIoULoss, self).__init__()
        self.smooth = smooth

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

            if t_sum == 0:
                # Authentic sample: penalize any non-zero prediction via L2 penalty
                loss += torch.mean(p ** 2)
            else:
                intersection = (p * t).sum()
                union = p.sum() + t.sum() - intersection
                iou = (intersection + self.smooth) / (union + self.smooth)
                loss += (1.0 - iou)

        return loss / B

def compute_morphological_boundary(mask, kernel_size=7):
    """
    Dynamic morphological boundary extraction:
      E_gt = Dilation(mask) - Erosion(mask)
    """
    padding = kernel_size // 2
    dilation = F.max_pool2d(mask, kernel_size=kernel_size, stride=1, padding=padding)
    erosion = -F.max_pool2d(-mask, kernel_size=kernel_size, stride=1, padding=padding)
    boundary = torch.clamp(dilation - erosion, 0.0, 1.0)
    return boundary

class CompoundForgeryLoss(nn.Module):
    """
    Enhanced Multi-Task Compound Objective for DF-CMNet v2:
      L_total = lambda_foc * L_Focal + lambda_iou * L_SoftIoU + lambda_bound * L_Edge
    """
    def __init__(self, lambda_foc=1.0, lambda_iou=1.0, lambda_bound=0.5, lambda_dice=None, boundary_kernel_size=7):
        super(CompoundForgeryLoss, self).__init__()
        self.lambda_foc = lambda_foc
        self.lambda_iou = lambda_dice if lambda_dice is not None else lambda_iou
        self.lambda_bound = lambda_bound
        self.boundary_kernel_size = boundary_kernel_size

        self.focal = BalancedFocalLoss(alpha=0.5, gamma=2.0)
        self.soft_iou = StableSoftIoULoss(smooth=1.0)
        self.bce_edge = nn.BCEWithLogitsLoss()

    def forward(self, outputs, target_masks):
        mask_logits = outputs['mask_logits']
        edge_logits = outputs['edge_logits']

        # 1. Main mask losses
        loss_focal = self.focal(mask_logits, target_masks)
        loss_iou = self.soft_iou(mask_logits, target_masks)

        # 2. Auxiliary Deep Supervision (if present)
        aux_loss = 0.0
        if 'aux_logits' in outputs:
            for aux in outputs['aux_logits']:
                aux_target = F.interpolate(target_masks, size=aux.shape[2:], mode='nearest')
                aux_loss += 0.4 * self.focal(aux, aux_target) + 0.4 * self.soft_iou(aux, aux_target)

        # 3. Dynamic morphological boundary loss
        edge_gt = compute_morphological_boundary(target_masks, kernel_size=self.boundary_kernel_size)
        loss_edge = self.bce_edge(edge_logits, edge_gt)

        total_loss = (self.lambda_foc * loss_focal +
                      self.lambda_iou * loss_iou +
                      self.lambda_bound * loss_edge +
                      aux_loss)

        return total_loss, {
            'loss_total': total_loss.item(),
            'loss_focal': loss_focal.item(),
            'loss_iou': loss_iou.item(),
            'loss_edge': loss_edge.item()
        }
