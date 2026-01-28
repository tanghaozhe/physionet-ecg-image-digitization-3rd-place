import torch
import torch.nn as nn
import torch.nn.functional as F

class DecoderBlock(nn.Module):
    def __init__(self, in_channel, skip_channel, out_channel, scale=2):
        super().__init__()
        self.scale = scale
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel + skip_channel, out_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )
        self.attention1 = nn.Identity()
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )
        self.attention2 = nn.Identity()

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=self.scale, mode='nearest')
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x

class UnetDecoder(nn.Module):
    def __init__(self, in_channel, skip_channel, out_channel, scale=[2, 2, 2, 2]):
        super().__init__()
        self.center = nn.Identity()

        i_channel = [in_channel] + out_channel[:-1]
        s_channel = skip_channel
        o_channel = out_channel
        block = [
            DecoderBlock(i, s, o, sc)
            for i, s, o, sc in zip(i_channel, s_channel, o_channel, scale)
        ]
        self.block = nn.ModuleList(block)

    def forward(self, feature, skip):
        d = self.center(feature)
        decode = []
        for i, block in enumerate(self.block):
            s = skip[i]
            d = block(d, s)
            decode.append(d)
        last = d
        return last, decode

class CoordDecoderBlock(nn.Module):
    def __init__(self, in_channel, skip_channel, out_channel, scale=2):
        super().__init__()
        self.scale = scale
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel + skip_channel + 2, out_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )
        self.attention1 = nn.Identity()
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )
        self.attention2 = nn.Identity()

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=self.scale, mode='nearest')
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)

        b, c, h, w = x.shape
        coordx, coordy = torch.meshgrid(
            torch.linspace(-2, 2, w, dtype=x.dtype, device=x.device),
            torch.linspace(-2, 2, h, dtype=x.dtype, device=x.device),
            indexing='xy'
        )
        coordxy = torch.stack([coordx, coordy], dim=1).reshape(1, 2, h, w).repeat(b, 1, 1, 1)
        x = torch.cat([x, coordxy], dim=1)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x

class CoordUnetDecoder(nn.Module):
    def __init__(self, in_channel, skip_channel, out_channel, scale=[2, 2, 2, 2]):
        super().__init__()
        self.center = nn.Identity()

        i_channel = [in_channel] + out_channel[:-1]
        s_channel = skip_channel
        o_channel = out_channel
        block = [
            CoordDecoderBlock(i, s, o, sc)
            for i, s, o, sc in zip(i_channel, s_channel, o_channel, scale)
        ]
        self.block = nn.ModuleList(block)

    def forward(self, feature, skip):
        d = self.center(feature)
        decode = []
        for i, block in enumerate(self.block):
            s = skip[i]
            d = block(d, s)
            decode.append(d)
        last = d
        return last, decode
