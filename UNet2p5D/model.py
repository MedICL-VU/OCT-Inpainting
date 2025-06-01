import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(conv => ReLU => BN) * 2"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_ch),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x):
        return self.conv(x)

class Down(nn.Module):
    """Downscaling with maxpool followed by double conv"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch)
        )

    def forward(self, x):
        return self.down(x)

class Up(nn.Module):
    """Upscaling and concatenation with skip connection"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Pad if needed (due to odd input sizes)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    """Final 1x1 convolution to get 1-channel output"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.out_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        return self.out_conv(x)
    
class ModulatedInputConv(nn.Module):
    """
    Applies a Conv2D after channel-wise scaling of the input, conditioned on a validity mask.
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.scale_fc = nn.Linear(in_ch, in_ch)  # one scale per input channel

    def forward(self, x, validity_mask):
        """
        Args:
            x: (B, C, H, W) input volume
            validity_mask: (B, C) binary or real values indicating slice validity
        """
        B, C, H, W = x.shape
        scales = self.scale_fc(validity_mask)  # (B, C)
        scales = scales.view(B, C, 1, 1)
        x = x * scales
        return F.relu(self.conv(x))

class UNet2p5D(nn.Module):
    """
    U-Net 2.5D model with optional dropout and configurable features.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        features (list): List of feature sizes for each level of the U-Net.
        dropout_rate (float): Dropout rate for regularization. Default is 0.0 (no dropout).
    """
    def __init__(self, in_channels=5, out_channels=1, features=None, dropout_rate=0.0, dynamic_filter=True):
        super().__init__()
        if features is None:
            features = [64, 128, 256, 512]
        self.dropout_rate = dropout_rate

        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError("dropout_rate must be between 0.0 and 1.0")

        if dynamic_filter:
            self.modulated_input = ModulatedInputConv(in_channels, features[0])
        else:
            self.inc = DoubleConv(in_channels, features[0])

        self.down1 = Down(features[0], features[1])
        self.down2 = Down(features[1], features[2])
        self.down3 = Down(features[2], features[3])

        self.dropout = nn.Dropout2d(p=dropout_rate) if dropout_rate > 0 else nn.Identity()

        self.up1 = Up(features[3], features[2])
        self.up2 = Up(features[2], features[1])
        self.up3 = Up(features[1], features[0])
        self.outc = OutConv(features[0], out_channels)

    def forward(self, x, validity_mask, dynamic_filter=True):
        """
        Args:
            x: (B, S, H, W) input stack of B-scans
            validity_mask: (B, S) mask indicating valid slices
        """
        if dynamic_filter:
            x1 = self.modulated_input(x, validity_mask)  # adaptive input conditioning
        else:
            x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x4 = self.dropout(x4)  # bottleneck dropout
        x = self.up1(x4, x3)
        x = self.dropout(x)    # decoder dropout (optional, same module)
        x = self.up2(x, x2)
        x = self.dropout(x)
        x = self.up3(x, x1)
        x = self.outc(x)
        return x
    