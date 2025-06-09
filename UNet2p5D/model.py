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
    
# class ModulatedInputConv(nn.Module):
#     """
#     Applies a Conv2D after channel-wise scaling of the input, conditioned on a validity mask.
#     """
#     def __init__(self, in_ch, out_ch):
#         super().__init__()
#         self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
#         self.scale_fc = nn.Linear(in_ch, in_ch)  # one scale per input channel

#     def forward(self, x, validity_mask):
#         """
#         Args:
#             x: (B, C, H, W) input volume
#             validity_mask: (B, C) binary or real values indicating slice validity
#         """
#         B, C, H, W = x.shape
#         scales = self.scale_fc(validity_mask)  # (B, C)
#         scales = scales.view(B, C, 1, 1)
#         x = x * scales
#         return F.relu(self.conv(x))

class ModulatedInputConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.base_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.scale_fc = nn.Sequential(
            nn.Linear(in_channels, in_channels * out_channels),
            nn.ReLU()
        )
        self.out_channels = out_channels
        self.in_channels = in_channels

    def forward(self, x, validity_mask):
        """
        x: [B, in_channels, H, W]
        validity_mask: [B, in_channels]
        """
        B, C, H, W = x.shape

        # Predict scaling matrix S for each sample in the batch
        S = self.scale_fc(validity_mask)  # Shape: [B, in_channels * out_channels]
        S = S.view(B, self.out_channels, self.in_channels)  # [B, C_out, C_in]


        # === DEBUG LOGGING ===
        if torch.rand(1).item() < 0.01:  # ~1% of batches
            for b in range(min(B, 2)):  # log at most 2 examples
                print(f"\n[SCALE MATRIX DEBUG] Batch {b}")
                print(f"  Validity Mask: {validity_mask[b].cpu().numpy()}")
                print(f"  Scale Matrix Shape: {S[b].shape}")
                
                # Compute simple stats
                S_b = S[b].detach().cpu().numpy()
                print(f"  S Min: {S_b.min():.4f} | Max: {S_b.max():.4f} | Mean: {S_b.mean():.4f} | Std: {S_b.std():.4f}")
                
                # Optional: row-wise stats (per output channel)
                for i in range(min(self.out_channels, 4)):
                    row = S_b[i]
                    print(f"    Output channel {i}: min={row.min():.3f}, mean={row.mean():.3f}, max={row.max():.3f}")

                # Compute correlation between input validity and mean scale for each input channel
                S_b = S[b].detach().cpu().numpy()  # Shape: [Cout, Cin]
                valid = validity_mask[b].cpu().numpy()  # Shape: [Cin]

                # Mean contribution of each input channel to output
                input_contrib = S_b.mean(axis=0)  # Shape: [Cin]

                # Print correlation
                from scipy.stats import pearsonr
                corr, _ = pearsonr(valid, input_contrib)
                print(f"  Correlation (validity ↔ input contrib): r = {corr:.3f}")


        # Apply scaled convolution (one per sample)
        weight = self.base_conv.weight  # [C_out, C_in, k, k]
        out = []
        for b in range(B):
            scaled_weight = S[b].unsqueeze(-1).unsqueeze(-1) * weight  # [C_out, C_in, k, k]
            out.append(F.conv2d(x[b].unsqueeze(0), scaled_weight, padding=1))

        return F.relu(torch.cat(out, dim=0))  # [B, C_out, H, W]


class UNet2p5D(nn.Module):
    """
    U-Net 2.5D model with optional dropout and configurable features.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        features (list): List of feature sizes for each level of the U-Net.
        dropout_rate (float): Dropout rate for regularization. Default is 0.0 (no dropout).
    """
    def __init__(self, in_channels=5, out_channels=1, features=None, dropout_rate=0.0, disable_dynamic_filter=False):
        super().__init__()
        if features is None:
            features = [64, 128, 256, 512]
        self.dropout_rate = dropout_rate

        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError("dropout_rate must be between 0.0 and 1.0")

        if not disable_dynamic_filter:
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

    def forward(self, x, validity_mask, disable_dynamic_filter=False):
        """
        Args:
            x: (B, S, H, W) input stack of B-scans
            validity_mask: (B, S) mask indicating valid slices
        """
        if not disable_dynamic_filter:
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
    