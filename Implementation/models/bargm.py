import torch
import torch.nn as nn
import torch.nn.functional as F


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super(ASPPConv, self).__init__(*modules)


class BoundaryAwareResidualGuidance(nn.Module):
    """
    Boundary-Aware Residual Guidance Module (BARGM).
    Extracts manipulation boundaries (splicing/copy-move/inpainting transition seams)
    using multi-scale dilated convolutions (ASPP), high-resolution Stage-1 skip refinement,
    and residual boundary attention gating.
    """
    def __init__(self, in_channels=256, out_channels=64):
        super(BoundaryAwareResidualGuidance, self).__init__()

        self.aspp1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.aspp2 = ASPPConv(in_channels, out_channels, dilation=2)
        self.aspp3 = ASPPConv(in_channels, out_channels, dilation=4)

        # Boundary projection head
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * 3, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # Edge logit classifier with high-resolution Stage-1 spatial skip refinement
        self.edge_head = nn.Conv2d(out_channels, 1, kernel_size=1)
        self.high_res_refine = nn.Sequential(
            nn.Conv2d(out_channels + 32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1)
        )

    def forward(self, x, target_size=None, high_res_feat=None, return_guided=False):
        """
        Args:
            x: Bottleneck fused feature tensor M^(4) [B, in_channels, H/32, W/32]
            target_size: Tuple (H, W) to upsample predicted boundary map
            high_res_feat: High-resolution Stage-1 feature M^(1) [B, 32, H/4, W/4]
            return_guided: If True, also returns boundary-attention gated M_guided^(4)
        """
        feat1 = self.aspp1(x)
        feat2 = self.aspp2(x)
        feat3 = self.aspp3(x)

        edge_features = self.project(torch.cat([feat1, feat2, feat3], dim=1))
        if high_res_feat is not None and high_res_feat.shape[1] == 32:
            edge_up = F.interpolate(edge_features, size=high_res_feat.shape[2:], mode='bilinear', align_corners=False)
            z_edge_hr = self.high_res_refine(torch.cat([edge_up, high_res_feat], dim=1))
        else:
            z_edge_hr = self.edge_head(edge_features)

        # Downsample boundary logits to bottleneck resolution to gate M^(4)
        edge_attn_low = torch.sigmoid(F.adaptive_avg_pool2d(z_edge_hr, x.shape[2:]))
        guided_x = x * (1.0 + edge_attn_low)
        edge_features = edge_features * (1.0 + edge_attn_low)

        if target_size is not None:
            edge_logits = F.interpolate(z_edge_hr, size=target_size, mode='bilinear', align_corners=False)
        else:
            edge_logits = z_edge_hr

        if return_guided:
            return edge_features, edge_logits, guided_x
        return edge_features, edge_logits
