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
    Extracts high-resolution manipulation boundaries (splicing/copy-move transition seams)
    using multi-scale dilated convolutions (ASPP) and provides explicit edge supervision.
    """
    def __init__(self, in_channels=512, out_channels=128):
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
        
        # Edge logit classifier
        self.edge_head = nn.Conv2d(out_channels, 1, kernel_size=1)

    def forward(self, x, target_size=None):
        """
        Args:
            x: Input feature tensor [B, in_channels, H, W]
            target_size: Tuple (H_out, W_out) to upsample predicted boundary map
        Returns:
            edge_features: Refined boundary feature map [B, out_channels, H, W]
            edge_logits: 1-channel boundary prediction [B, 1, H_out, W_out]
        """
        feat1 = self.aspp1(x)
        feat2 = self.aspp2(x)
        feat3 = self.aspp3(x)

        edge_features = self.project(torch.cat([feat1, feat2, feat3], dim=1))
        edge_logits = self.edge_head(edge_features)

        if target_size is not None:
            edge_logits = F.interpolate(edge_logits, size=target_size, mode='bilinear', align_corners=False)

        return edge_features, edge_logits
