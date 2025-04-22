import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------ CBAM Attention Module ------------------

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        self.fc = nn.Sequential(
            nn.Conv3d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv3d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = self.fc(self.avg_pool(x))
        max = self.fc(self.max_pool(x))
        return self.sigmoid(avg + max)

class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        max, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg, max], dim=1)
        return self.sigmoid(self.conv(x_cat))

class CBAM3D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channel = ChannelAttention(channels)
        self.spatial = SpatialAttention()

    def forward(self, x):
        x = x * self.channel(x)
        x = x * self.spatial(x)
        return x

# ------------------ 3D U-Net Core Blocks ------------------

class DoubleConv3D(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=False, dilation=1):
        super().__init__()
        padding = dilation
        layers = [
            nn.Conv3d(in_ch, out_ch, 3, padding=padding, dilation=dilation),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(out_ch),
            nn.Conv3d(out_ch, out_ch, 3, padding=padding, dilation=dilation),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(out_ch),
        ]
        if dropout:
            layers.append(nn.Dropout3d(p=0.2))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)

class Down3D(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=False, dilation=1):
        super().__init__()
        self.down = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv3D(in_ch, out_ch, dropout=dropout, dilation=dilation)
        )

    def forward(self, x):
        return self.down(x)

class Up3D(nn.Module):
    def __init__(self, in_ch, out_ch, apply_attention=True):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv3D(in_ch, out_ch)
        self.attn = CBAM3D(out_ch) if apply_attention else nn.Identity()

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Pad to match dimensions if needed
        diffZ = x2.size(2) - x1.size(2)
        diffY = x2.size(3) - x1.size(3)
        diffX = x2.size(4) - x1.size(4)
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2,
                        diffZ // 2, diffZ - diffZ // 2])
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        x = self.attn(x)
        return x

class OutConv3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.outc = nn.Conv3d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        return self.outc(x)

# ------------------ Full 3D U-Net with Refinement ------------------

class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[32, 64, 128, 256, 512]):
        super().__init__()
        self.inc   = DoubleConv3D(in_channels, features[0])
        self.down1 = Down3D(features[0], features[1])
        self.down2 = Down3D(features[1], features[2])
        self.down3 = Down3D(features[2], features[3], dropout=True)
        self.down4 = Down3D(features[3], features[4], dropout=True, dilation=2)  # Dilated bottleneck

        self.up1 = Up3D(features[4], features[3])
        self.up2 = Up3D(features[3], features[2])
        self.up3 = Up3D(features[2], features[1])
        self.up4 = Up3D(features[1], features[0])

        self.outc = OutConv3D(features[0], out_channels)

        self.refine = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        x = self.outc(x)
        x = self.refine(x) + x
        return x
