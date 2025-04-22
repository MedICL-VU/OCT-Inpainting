import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv3D(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=False):
        super().__init__()
        layers = [
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(out_ch),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(out_ch),
        ]
        if dropout:
            layers.append(nn.Dropout3d(p=0.2))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)

class Down3D(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=False):
        super().__init__()
        self.down = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv3D(in_ch, out_ch, dropout=dropout)
        )

    def forward(self, x):
        return self.down(x)

class Up3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv3D(in_ch, out_ch)

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
        return self.conv(x)

class OutConv3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.outc = nn.Conv3d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        return self.outc(x)

class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[32, 64, 128, 256, 512]):
        super().__init__()
        self.inc   = DoubleConv3D(in_channels, features[0])
        self.down1 = Down3D(features[0], features[1])
        self.down2 = Down3D(features[1], features[2])
        self.down3 = Down3D(features[2], features[3], dropout=True)  # bottleneck with dropout
        self.down4 = Down3D(features[3], features[4], dropout=True)  # bottleneck with dropout

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
        x = self.refine(x)  # Residual refinement
        return x
