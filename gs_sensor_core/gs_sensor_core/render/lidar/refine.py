"""Raydrop-refinement UNet: ported from `~/GS-LiDAR/scene/unet.py`, which is
itself LiDAR4D's UNet (ispc-lab/LiDAR4D, Apache-2.0 -- a different license
than the Inria/MPII-derived rasterizer kernel, and permissive, so this one
file is copied directly rather than vendored as a submodule). Vendored as
plain code (not a runtime import of GS-LiDAR's `scene.unet`), same
reimplement-don't-import rule as the rest of gs_sensor_core.

Loads `ckpt/refine.pth` (a raw `state_dict`, not a capture()-style tuple --
see `models/lidar_checkpoint_loader.py`'s docstring for the checkpoint
inventory) and runs it on the 3-channel `[raydrop, intensity, depth]` stack
GS-LiDAR's own `train.py` builds before saving that same checkpoint
(`torch.save(torch.cat([raydrop_pano, intensity_sh_pano, depth_pano[[0]]]),
...)`, `~/GS-LiDAR/train.py:418`) -- reproduced here at inference time
instead of loaded from disk.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


class _DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, dropout=0.1):
        super().__init__()
        mid_channels = mid_channels or out_channels
        self.double_conv = nn.Sequential(
            nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True), nn.Dropout2d(dropout),
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels), nn.ReLU(inplace=True), nn.Dropout2d(dropout),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, x):
        return self.double_conv(x)


class _Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = nn.MaxPool2d(2)
        self.conv = _DoubleConv(in_channels, out_channels)

    def forward(self, x):
        return self.conv(self.down(x))


class _Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = _DoubleConv(in_channels, out_channels, in_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        return self.conv(torch.cat([x2, x1], dim=1))


class _AttnBlock(nn.Module):
    def __init__(self, in_ch, num_head=8, dropout=0.1):
        super().__init__()
        self.proj_qkv = nn.Conv2d(in_ch, in_ch * 3, 1, bias=False)
        self.proj = nn.Conv2d(in_ch, in_ch, 1, bias=False)
        self.norm = nn.BatchNorm2d(in_ch)
        self.dropout = dropout
        self.num_head = num_head

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.proj_qkv(self.norm(x))
        q, k, v = torch.chunk(qkv, 3, dim=1)
        q = q.view(b, self.num_head, -1, h * w).permute(0, 1, 3, 2)
        k = k.view(b, self.num_head, -1, h * w)
        v = v.view(b, self.num_head, -1, h * w).permute(0, 1, 3, 2)
        attn = torch.matmul(q, k) * (int(c // self.num_head) ** -0.5)
        if self.training:
            attn = attn + torch.bernoulli(torch.ones_like(attn) * self.dropout) * -1e12
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v).view(b, h, w, c).permute(0, 3, 1, 2)
        return x + self.proj(out)


class _InConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class _OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
        )

    def forward(self, x):
        return self.conv(x)


class RaydropRefineUNet(nn.Module):
    def __init__(self, in_channels: int = 3, channels: int = 32, out_channels: int = 1):
        super().__init__()
        self.inc = _InConv(in_channels, channels)
        self.down1 = _Down(channels, channels * 2)
        self.down2 = _Down(channels * 2, channels * 4)
        self.down3 = _Down(channels * 4, channels * 8)
        self.down4 = _Down(channels * 8, channels * 8)
        self.attn = _AttnBlock(channels * 8)
        self.up1 = _Up(channels * 16, channels * 4)
        self.up2 = _Up(channels * 8, channels * 2)
        self.up3 = _Up(channels * 4, channels)
        self.up4 = _Up(channels * 2, channels)
        self.outc = _OutConv(channels, out_channels)

    def forward(self, x):
        x0 = self.inc(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.attn(self.down4(x3))
        out = self.up1(x4, x3)
        out = self.up2(out, x2)
        out = self.up3(out, x1)
        out = self.up4(out, x0)
        return torch.sigmoid(self.outc(out))


def load_refine_unet(path: str | Path, device: str = "cuda") -> RaydropRefineUNet:
    unet = RaydropRefineUNet(in_channels=3, out_channels=1).to(device)
    unet.load_state_dict(torch.load(str(path), map_location=device, weights_only=True))
    unet.eval()
    return unet


@torch.no_grad()
def refine_raydrop(unet: RaydropRefineUNet, raydrop: torch.Tensor,
                    intensity: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
    """`raydrop`/`intensity`/`depth`: each `[1, h, w]` (one stitched
    panorama). Returns the refined raydrop probability, same shape."""
    stacked = torch.cat([raydrop, intensity, depth], dim=0).unsqueeze(0)  # [1, 3, h, w]
    return unet(stacked)[0]
