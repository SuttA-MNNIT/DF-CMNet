import torch
import torch.nn as nn
import torch.nn.functional as F

from .bayar_conv import BayarConv2d
from .wavelet_dwt import DWT2D_Haar
from .cdbm import CrossDomainBidirectionalModulation
from .bargm import BoundaryAwareResidualGuidance

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)

class SpatialEncoder(nn.Module):
    """
    Hierarchical Spatial Semantic Feature Extractor.
    Extracts multi-scale representations at H/4, H/8, H/16, H/32.
    """
    def __init__(self, in_channels=3, channels=[32, 64, 128, 256]):
        super(SpatialEncoder, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.stage1 = ConvBlock(channels[0], channels[0], stride=1)           # H/4, W/4
        self.stage2 = ConvBlock(channels[0], channels[1], stride=2)           # H/8, W/8
        self.stage3 = ConvBlock(channels[1], channels[2], stride=2)           # H/16, W/16
        self.stage4 = ConvBlock(channels[2], channels[3], stride=2)           # H/32, W/32

    def forward(self, x):
        x = self.stem(x)
        s1 = self.stage1(x)
        s2 = self.stage2(s1)
        s3 = self.stage3(s2)
        s4 = self.stage4(s3)
        return [s1, s2, s3, s4]

class FrequencyEncoder(nn.Module):
    """
    Encoder for the Noise-Frequency stream.
    Receives standardized high-frequency residual & DWT sub-bands.
    """
    def __init__(self, in_channels=12, channels=[32, 64, 128, 256]):
        super(FrequencyEncoder, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels[0], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True)
        )
        self.stage1 = ConvBlock(channels[0], channels[0], stride=1)           # H/4, W/4
        self.stage2 = ConvBlock(channels[0], channels[1], stride=2)           # H/8, W/8
        self.stage3 = ConvBlock(channels[1], channels[2], stride=2)           # H/16, W/16
        self.stage4 = ConvBlock(channels[2], channels[3], stride=2)           # H/32, W/32

    def forward(self, x):
        x = self.stem(x)
        f1 = self.stage1(x)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        f4 = self.stage4(f3)
        return [f1, f2, f3, f4]

class DF_CMNet(nn.Module):
    """
    DF-CMNet v2: Dual-Domain Frequency-Spatial Cross-Modulation Network
    Equipped with Residual Standardization, Dual-Attention CDBM, and Deep Supervision.
    """
    def __init__(self, pretrained=False, channels=[32, 64, 128, 256], **kwargs):
        super(DF_CMNet, self).__init__()
        self.channels = channels

        # --- 1. SPATIAL STREAM BACKBONE ---
        self.spatial_encoder = SpatialEncoder(in_channels=3, channels=channels)

        # --- 2. NOISE-FREQUENCY STREAM (LHFRE + Standardization + Wavelet) ---
        self.bayar = BayarConv2d(in_channels=3, out_channels=3, kernel_size=5)
        # Residual Standardization to handle arbitrary ISO/noise ranges
        self.res_norm = nn.InstanceNorm2d(3, affine=True)
        self.dwt = DWT2D_Haar()
        self.dwt_norm = nn.BatchNorm2d(12)
        self.freq_encoder = FrequencyEncoder(in_channels=12, channels=channels)

        # --- 3. DUAL-ATTENTION CROSS-DOMAIN MODULATION (CDBM) ---
        self.cdbm1 = CrossDomainBidirectionalModulation(channels[0])
        self.cdbm2 = CrossDomainBidirectionalModulation(channels[1])
        self.cdbm3 = CrossDomainBidirectionalModulation(channels[2])
        self.cdbm4 = CrossDomainBidirectionalModulation(channels[3])

        # --- 4. BOUNDARY-AWARE RESIDUAL GUIDANCE (BARGM) ---
        self.bargm = BoundaryAwareResidualGuidance(in_channels=channels[3], out_channels=channels[1])

        # --- 5. PROGRESSIVE MULTI-SCALE DECODER WITH DEEP SUPERVISION ---
        self.dec_bridge = ConvBlock(channels[3], channels[2])
        
        # Level 3 fusion
        self.dec_fuse3 = nn.Sequential(
            nn.Conv2d(channels[2] + channels[2] + channels[1], channels[2], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels[2]),
            nn.ReLU(inplace=True),
            ConvBlock(channels[2], channels[1])
        )
        self.aux_head3 = nn.Conv2d(channels[1], 1, kernel_size=1)

        # Level 2 fusion
        self.dec_fuse2 = nn.Sequential(
            nn.Conv2d(channels[1] + channels[1] + channels[1], channels[1], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels[1]),
            nn.ReLU(inplace=True),
            ConvBlock(channels[1], channels[0])
        )
        self.aux_head2 = nn.Conv2d(channels[0], 1, kernel_size=1)

        # Level 1 fusion
        self.dec_fuse1 = nn.Sequential(
            nn.Conv2d(channels[0] + channels[0], channels[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
            ConvBlock(channels[0], 16)
        )

        # Final prediction heads
        self.final_mask_head = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, kernel_size=1)
        )

    def extract_frequency_features(self, x):
        # 1. Bayar noise residual
        res = self.bayar(x)
        # 2. Residual standardization
        res = self.res_norm(res)
        # 3. 2D Haar DWT
        ll, lh, hl, hh = self.dwt(res)
        res_down = F.avg_pool2d(res, kernel_size=2, stride=2)
        # 4. Standardized concatenation
        x_freq = self.dwt_norm(torch.cat([res_down, lh, hl, hh], dim=1))
        # 5. Multi-scale frequency representations
        freq_features = self.freq_encoder(x_freq)
        return freq_features

    def forward(self, x):
        B, C, H, W = x.shape

        # --- Spatial Stream Forward ---
        s1, s2, s3, s4 = self.spatial_encoder(x)

        # --- Frequency Stream Forward ---
        f1, f2, f3, f4 = self.extract_frequency_features(x)

        # --- Cross-Domain Modulation (CDBM) ---
        m1 = self.cdbm1(s1, f1) # [B, C0, H/4, W/4]
        m2 = self.cdbm2(s2, f2) # [B, C1, H/8, W/8]
        m3 = self.cdbm3(s3, f3) # [B, C2, H/16, W/16]
        m4 = self.cdbm4(s4, f4) # [B, C3, H/32, W/32]

        # --- Boundary-Aware Guidance (BARGM) ---
        edge_feat, edge_logits = self.bargm(m4, target_size=(H, W))

        # --- Multi-Scale Decoder ---
        d4 = self.dec_bridge(m4)
        d4_up = F.interpolate(d4, size=m3.shape[2:], mode='bilinear', align_corners=False)
        edge_feat_d3 = F.interpolate(edge_feat, size=m3.shape[2:], mode='bilinear', align_corners=False)
        d3 = self.dec_fuse3(torch.cat([d4_up, m3, edge_feat_d3], dim=1)) # [B, C1, H/16, W/16]

        d3_up = F.interpolate(d3, size=m2.shape[2:], mode='bilinear', align_corners=False)
        edge_feat_d2 = F.interpolate(edge_feat, size=m2.shape[2:], mode='bilinear', align_corners=False)
        d2 = self.dec_fuse2(torch.cat([d3_up, m2, edge_feat_d2], dim=1)) # [B, C0, H/8, W/8]

        d2_up = F.interpolate(d2, size=m1.shape[2:], mode='bilinear', align_corners=False)
        d1 = self.dec_fuse1(torch.cat([d2_up, m1], dim=1)) # [B, 16, H/4, W/4]

        # Final upsampling to original input resolution (H, W)
        d0 = F.interpolate(d1, size=(H, W), mode='bilinear', align_corners=False)
        mask_logits = self.final_mask_head(d0) # [B, 1, H, W]

        # Deep Supervision auxiliary outputs during training
        aux_logits = []
        if self.training:
            aux3 = self.aux_head3(d3)
            aux2 = self.aux_head2(d2)
            aux_logits = [aux3, aux2]

        return {
            'mask_logits': mask_logits,
            'edge_logits': edge_logits,
            'aux_logits': aux_logits,
            'mask_pred': torch.sigmoid(mask_logits),
            'edge_pred': torch.sigmoid(edge_logits)
        }
