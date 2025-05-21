import numpy as np
import torch
from torch.utils.data import Dataset
import tifffile as tiff
import random

class OCTAInpaintingDataset(Dataset):
    def __init__(self, volume_paths, stack_size=3, corruption_patterns=None):
        self.stack_size = stack_size
        self.volume_paths = volume_paths
        self.samples = []

        self.patterns = corruption_patterns or ["gt_gt_gt", "gt_gt_corr", "corr_gt_gt", "corr_gt_corr"]
        half = stack_size // 2

        # Preload and prepare all samples
        for vol_path in volume_paths:
            volume = tiff.imread(vol_path).astype(np.float32)  # shape: (D, H, W)
            D, H, W = volume.shape

            # Normalize once for all variants
            volume = volume / 65535.0

            for i in range(half, D - half):
                for pattern in self.patterns:
                    self.samples.append({
                        "volume": volume,
                        "center_idx": i,
                        "pattern": pattern
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        volume = item["volume"]
        i = item["center_idx"]
        pattern = item["pattern"]
        half = self.stack_size // 2

        stack = []
        for offset in range(-half, half + 1):
            j = i + offset
            slice_img = volume[j].copy()

            if offset != 0:  # Don't corrupt center slice
                # Decide corruption from pattern string
                pattern_key = ["gt", "corr"][int("corr" in pattern.split("_")[offset + half])]
                if pattern_key == "corr":
                    slice_img = np.zeros_like(slice_img)  # Could also do noise-based corruption
            stack.append(slice_img)

        input_tensor = torch.tensor(np.stack(stack), dtype=torch.float32)  # (stack, H, W)
        target_tensor = torch.tensor(volume[i], dtype=torch.float32)  # (H, W)

        return input_tensor, target_tensor
