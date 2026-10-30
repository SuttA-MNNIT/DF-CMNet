import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossDomainBidirectionalModulation(nn.Module):
    """
    Dual-Attention Cross-Domain Bidirectional Modulation (DACM / CDBM).
    Performs complementary Spatial Cross-Attention AND Channel Cross-Attention:
      1. Point-Wise Spatial Cross-Attention: highlights WHERE spatial edges coincide with spectral anomalies.
      2. Global Channel Cross-Attention: calibrates WHICH wavelet/spatial channels carry forensic traces.
      3. Adaptive Gated Reciprocal Integration with residual connection.
    """
    def __init__(self, channels, spatial_only=False):
        super(CrossDomainBidirectionalModulation, self).__init__()
        self.channels = channels
        self.spatial_only = spatial_only
        reduced = max(channels // 4, 16)

        # 1. Spatial Cross-Attention 1x1 bottleneck projections
        self.qs_spat = nn.Conv2d(channels, reduced, kernel_size=1)
        self.ks_spat = nn.Conv2d(channels, reduced, kernel_size=1)
        self.qn_spat = nn.Conv2d(channels, reduced, kernel_size=1)
        self.kn_spat = nn.Conv2d(channels, reduced, kernel_size=1)

        # 2. Channel Cross-Attention projections (Global Context)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.channel_gate_s = nn.Sequential(
            nn.Linear(channels, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels, bias=False),
            nn.Sigmoid()
        )
        self.channel_gate_n = nn.Sequential(
            nn.Linear(channels, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels, bias=False),
            nn.Sigmoid()
        )

        # 3. Gated Fusion
        self.gate_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        # 4. Refinement convolution
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, fs, fn):
        B, C, H, W = fs.shape

        # --- A. Point-Wise Spatial Cross-Attention (O(H * W * d_k)) ---
        qs = self.qs_spat(fs)
        ks = self.ks_spat(fs)
        qn = self.qn_spat(fn)
        kn = self.kn_spat(fn)

        scale = (qs.shape[1]) ** -0.5
        attn_s2n = torch.sigmoid(torch.sum(qs * kn, dim=1, keepdim=True) * scale)  # [B, 1, H, W]
        attn_n2s = torch.sigmoid(torch.sum(qn * ks, dim=1, keepdim=True) * scale)  # [B, 1, H, W]

        f_s_modulated = attn_s2n * fn
        f_n_modulated = attn_n2s * fs

        if self.spatial_only:
            s_combined = f_s_modulated + fs
            n_combined = f_n_modulated + fn
        else:
            # --- B. Global Channel Cross-Attention ---
            fn_vec = self.gap(fn).view(B, C)
            s_chan_weight = self.channel_gate_s(fn_vec).view(B, C, 1, 1)
            f_s_chan = fs * s_chan_weight

            fs_vec = self.gap(fs).view(B, C)
            n_chan_weight = self.channel_gate_n(fs_vec).view(B, C, 1, 1)
            f_n_chan = fn * n_chan_weight

            s_combined = f_s_modulated + f_s_chan
            n_combined = f_n_modulated + f_n_chan

        # --- C. Adaptive Gated Integration ---
        gate = self.gate_conv(torch.cat([fs, fn], dim=1))
        f_fused = gate * s_combined + (1.0 - gate) * n_combined

        out = self.refine(f_fused) + fs
        return out


# Alias for paper terminology (DACM)
DualAttentionCrossModulation = CrossDomainBidirectionalModulation
DACM = CrossDomainBidirectionalModulation
