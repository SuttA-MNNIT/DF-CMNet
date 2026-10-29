import torch
import torch.nn as nn
import torch.nn.functional as F

class BayarConv2d(nn.Module):
    """
    Bayar Constrained Convolutional Layer for Steganalysis and Image Forensics.
    Suppresses low-frequency image content and learns high-frequency prediction residuals.
    
    Constraint:
        center_weight = -1
        sum(off_center_weights) = 1
    """
    def __init__(self, in_channels=3, out_channels=3, kernel_size=5, stride=1, padding=2):
        super(BayarConv2d, self).__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Initialize unconstrained weights
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        # Mask to identify off-center weights
        mask = torch.ones(kernel_size, kernel_size)
        center = kernel_size // 2
        mask[center, center] = 0.0
        self.register_buffer('mask', mask.unsqueeze(0).unsqueeze(0))

    def get_constrained_weight(self):
        # Zero out center
        off_center_weights = self.weight * self.mask
        # Normalize sum of off-center weights along spatial dimensions to 1.0
        sum_weights = torch.sum(off_center_weights, dim=(-2, -1), keepdim=True)
        # Avoid division by zero
        sum_weights = torch.where(sum_weights == 0, torch.tensor(1e-7, device=sum_weights.device), sum_weights)
        normalized_weights = off_center_weights / sum_weights
        
        # Set center weight to -1.0
        center_mask = 1.0 - self.mask
        constrained_weight = normalized_weights - center_mask
        return constrained_weight

    def forward(self, x):
        weight = self.get_constrained_weight()
        return F.conv2d(x, weight, bias=None, stride=self.stride, padding=self.padding)
