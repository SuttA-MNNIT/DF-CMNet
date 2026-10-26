import torch
import torch.nn as nn
import torch.nn.functional as F

class DWT2D_Haar(nn.Module):
    """
    2D Discrete Wavelet Transform (DWT) using 2D Haar basis implemented via depthwise 2D convolutions.
    Splits input [B, C, H, W] into 4 sub-bands each of size [B, C, H/2, W/2]:
      - LL: Low-frequency approximation
      - LH: Horizontal high-frequency details
      - HL: Vertical high-frequency details
      - HH: Diagonal high-frequency details
    """
    def __init__(self):
        super(DWT2D_Haar, self).__init__()
        # Construct Haar 2x2 filters
        ll = torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=torch.float32)
        lh = torch.tensor([[-0.5, -0.5], [0.5, 0.5]], dtype=torch.float32)
        hl = torch.tensor([[-0.5, 0.5], [-0.5, 0.5]], dtype=torch.float32)
        hh = torch.tensor([[0.5, -0.5], [-0.5, 0.5]], dtype=torch.float32)

        # Filters shape: [4, 1, 2, 2]
        filters = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)
        self.register_buffer('filters', filters)

    def forward(self, x):
        B, C, H, W = x.shape
        # Pad if odd dimensions
        pad_h = H % 2
        pad_w = W % 2
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')

        # Reshape input to apply filters across all channels
        x_reshaped = x.view(B * C, 1, x.shape[2], x.shape[3])
        # Depthwise conv with stride 2
        subbands = F.conv2d(x_reshaped, self.filters, stride=2, padding=0)
        # subbands shape: [B*C, 4, H/2, W/2]
        subbands = subbands.view(B, C, 4, subbands.shape[2], subbands.shape[3])

        ll = subbands[:, :, 0, :, :]
        lh = subbands[:, :, 1, :, :]
        hl = subbands[:, :, 2, :, :]
        hh = subbands[:, :, 3, :, :]
        return ll, lh, hl, hh
