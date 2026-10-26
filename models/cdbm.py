import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossDomainBidirectionalModulation(nn.Module):
    """
    Enhanced Dual-Attention Cross-Domain Bidirectional Modulation (DACM).
    Performs complementary Spatial Cross-Attention AND Channel Cross-Attention:
      1. Spatial Attention: highlights WHERE spatial edges match frequency noise disruptions.
      2. Channel Attention: calibrates WHICH frequency channels correspond to forgery artifacts.
      3. Gated reciprocal feature integration.
    """
    def __init__(self, channels):
        super(CrossDomainBidirectionalModulation, self).__init__()
        self.channels = channels
        reduced = max(channels // 4, 16)

        # 1. Spatial Cross-Attention projections
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

        # --- A. Spatial Cross-Attention ---
        qs = self.qs_spat(fs)
        ks = self.ks_spat(fs)
        qn = self.qn_spat(fn)
        kn = self.kn_spat(fn)

        scale = (qs.shape[1]) ** -0.5
        # Spatial affinity maps
        attn_s2n = torch.sigmoid(torch.sum(qs * kn, dim=1, keepdim=True) * scale) # [B, 1, H, W]
        attn_n2s = torch.sigmoid(torch.sum(qn * ks, dim=1, keepdim=True) * scale) # [B, 1, H, W]

        f_s_modulated = attn_s2n * fn
        f_n_modulated = attn_n2s * fs

        # --- B. Channel Cross-Attention ---
        # Frequency channel descriptor modulates spatial features
        fn_vec = self.gap(fn).view(B, C)
        s_chan_weight = self.channel_gate_s(fn_vec).view(B, C, 1, 1)
        f_s_chan = fs * s_chan_weight

        # Spatial channel descriptor modulates frequency features
        fs_vec = self.gap(fs).view(B, C)
        n_chan_weight = self.channel_gate_n(fs_vec).view(B, C, 1, 1)
        f_n_chan = fn * n_chan_weight

        # Combine spatial + channel modulations
        s_combined = f_s_modulated + f_s_chan
        n_combined = f_n_modulated + f_n_chan

        # --- C. Adaptive Gated Integration ---
        gate = self.gate_conv(torch.cat([fs, fn], dim=1))
        f_fused = gate * s_combined + (1.0 - gate) * n_combined

        # Residual output
        out = self.refine(f_fused) + fs
        return out
