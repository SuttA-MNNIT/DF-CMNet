import torch
import torch.nn as nn
import torch.nn.functional as F


class BayarConv2d(nn.Module):
    """
    Bayar Constrained Convolutional Layer for Steganalysis and Image Forensics.
    Suppresses low-frequency scene content and learns high-frequency prediction-error residuals.

    Constraint:
        w(0, 0) = -1
        sum_{(u,v) != (0,0)} w(u, v) = +1
    """
    def __init__(self, in_channels=3, out_channels=3, kernel_size=5, stride=1, padding=2):
        super(BayarConv2d, self).__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Initialize off-center weights positively around 1/(K^2 - 1) so sum is well away from zero
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        nn.init.uniform_(self.weight, 0.02, 0.08)

        # Mask to identify off-center weights
        mask = torch.ones(kernel_size, kernel_size)
        center = kernel_size // 2
        mask[center, center] = 0.0
        self.register_buffer('mask', mask.unsqueeze(0).unsqueeze(0))

    @torch.no_grad()
    def project_weights(self):
        """Explicitly project parameter tensor onto the Bayar constraint manifold after optimizer.step()."""
        self.weight.data.copy_(self.get_constrained_weight())

    def get_constrained_weight(self):
        # Zero out center
        off_center_weights = self.weight * self.mask
        # Normalize sum of off-center weights along spatial + input-channel dimensions or per-kernel to 1.0
        sum_weights = torch.sum(off_center_weights, dim=(-2, -1), keepdim=True)
        # Numerically safe denominator (prevents near-zero division blowup)
        sign = torch.where(sum_weights >= 0, torch.ones_like(sum_weights), -torch.ones_like(sum_weights))
        safe_sum = sign * torch.clamp(sum_weights.abs(), min=1e-3)
        normalized_weights = off_center_weights / safe_sum

        # Pin center weight to -1.0
        center_mask = 1.0 - self.mask
        constrained_weight = normalized_weights - center_mask
        return constrained_weight

    def forward(self, x):
        weight = self.get_constrained_weight()
        return F.conv2d(x, weight, bias=None, stride=self.stride, padding=self.padding)


class SRMConv2d(nn.Module):
    """
    Fixed Steganalysis Rich Model (SRM) High-Pass Residual Filter Bank (used in Ablation V5).
    Applies classical 1st, 2nd, and 3rd order high-pass residual filters followed by stride-2 pooling.
    """
    def __init__(self, in_channels=3, out_channels=12):
        super(SRMConv2d, self).__init__()
        # 4 canonical 5x5 SRM high-pass kernels applied depthwise to 3 RGB channels -> 12 channels
        q = [4.0, 12.0, 2.0, 4.0]
        f1 = [[0, 0, 0, 0, 0],
              [0, -1, 2, -1, 0],
              [0, 2, -4, 2, 0],
              [0, -1, 2, -1, 0],
              [0, 0, 0, 0, 0]]
        f2 = [[-1, 2, -2, 2, -1],
              [2, -6, 8, -6, 2],
              [-2, 8, -12, 8, -2],
              [2, -6, 8, -6, 2],
              [-1, 2, -2, 2, -1]]
        f3 = [[0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0],
              [0, 1, -2, 1, 0],
              [0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0]]
        f4 = [[0, 0, 0, 0, 0],
              [0, 0, 1, 0, 0],
              [0, 0, -2, 0, 0],
              [0, 0, 1, 0, 0],
              [0, 0, 0, 0, 0]]
        kernels = []
        for k, s in zip([f1, f2, f3, f4], q):
            kernels.append(torch.tensor(k, dtype=torch.float32) / s)
        # Shape [4, 1, 5, 5] repeated for 3 channels -> [12, 1, 5, 5]
        weight = torch.stack(kernels, dim=0).unsqueeze(1).repeat(in_channels, 1, 1, 1)
        self.register_buffer('weight', weight)
        self.in_channels = in_channels

    def forward(self, x):
        # Depthwise high-pass filtering + stride-2 average pooling to match H/2 x W/2
        res = F.conv2d(x, self.weight, stride=1, padding=2, groups=self.in_channels)
        return F.avg_pool2d(res, kernel_size=2, stride=2)
