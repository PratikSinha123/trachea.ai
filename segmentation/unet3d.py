"""3D U-Net architecture for volumetric medical image segmentation.

Ready for training on annotated trachea data. Supports MPS (Apple Silicon),
CUDA, and CPU backends.
"""

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """Two consecutive 3D convolutions with batch norm and ReLU."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    """Downsampling: MaxPool then DoubleConv."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(nn.MaxPool3d(2), DoubleConv(in_ch, out_ch))

    def forward(self, x):
        return self.block(x)


class Up(nn.Module):
    """Upsampling: transposed conv then DoubleConv with skip connection."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # Pad if sizes don't match due to odd dimensions
        dz = skip.size(2) - x.size(2)
        dy = skip.size(3) - x.size(3)
        dx = skip.size(4) - x.size(4)
        x = nn.functional.pad(x, [dx // 2, dx - dx // 2,
                                   dy // 2, dy - dy // 2,
                                   dz // 2, dz - dz // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet3D(nn.Module):
    """3D U-Net for volumetric segmentation.

    Architecture: 4-level encoder-decoder with skip connections.
    Channels: in → 32 → 64 → 128 → 256 → 128 → 64 → 32 → out
    """

    def __init__(self, in_channels=1, out_channels=1, features=(32, 64, 128, 256)):
        super().__init__()
        f = features
        self.inc = DoubleConv(in_channels, f[0])
        self.down1 = Down(f[0], f[1])
        self.down2 = Down(f[1], f[2])
        self.down3 = Down(f[2], f[3])
        self.up1 = Up(f[3], f[2])
        self.up2 = Up(f[2], f[1])
        self.up3 = Up(f[1], f[0])
        self.outc = nn.Conv3d(f[0], out_channels, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        return self.outc(x)


def get_device():
    """Get the best available device for this system."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
