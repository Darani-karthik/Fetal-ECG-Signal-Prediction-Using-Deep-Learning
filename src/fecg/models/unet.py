"""U-Net: symmetric encoder-decoder with skip connections for image-to-image translation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Sequential):
    """(Conv3x3 -> BatchNorm -> ReLU) x 2, with optional channel dropout in between."""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0) -> None:
        super().__init__(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class Up(nn.Module):
    """Up-sample the decoder features, concatenate the encoder skip, convolve."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_ch + skip_ch, out_ch, dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        dh, dw = skip.shape[-2] - x.shape[-2], skip.shape[-1] - x.shape[-1]
        if dh or dw:  # odd input sizes
            x = F.pad(x, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2])
        return self.conv(torch.cat([skip, x], dim=1))


class UNet(nn.Module):
    """U-Net mapping a multi-channel abdominal scalogram to a fetal scalogram.

    Args:
        in_channels: input image channels (8 = 4 abdominal leads x magnitude/phase).
        out_channels: output channels (2 = real/imaginary part of the fetal scalogram).
        base_channels: width of the first stage; doubles at every down-sampling.
        depth: number of down-sampling stages.
        dropout: channel-dropout probability inside every double-conv block (regularisation).
    """

    def __init__(
        self,
        in_channels: int = 8,
        out_channels: int = 2,
        base_channels: int = 32,
        depth: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        widths = [base_channels * 2**i for i in range(depth + 1)]
        self.inc = DoubleConv(in_channels, widths[0], dropout)
        self.down = nn.ModuleList(
            nn.Sequential(nn.MaxPool2d(2), DoubleConv(widths[i], widths[i + 1], dropout)) for i in range(depth)
        )
        self.up = nn.ModuleList(Up(widths[i + 1], widths[i], widths[i], dropout) for i in reversed(range(depth)))
        self.outc = nn.Conv2d(widths[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = [self.inc(x)]
        for down in self.down:
            skips.append(down(skips[-1]))
        x = skips.pop()
        for up in self.up:
            x = up(x, skips.pop())
        return self.outc(x)
