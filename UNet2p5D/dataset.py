import torch
from torch.utils.data import Dataset
import tifffile as tiff
import numpy as np
import random

class IntensityAugment:
    def __init__(self, scale_range=(0.9, 1.1), bias_range=(-0.05, 0.05)):
        self.scale_range = scale_range
        self.bias_range = bias_range

    def __call__(self, volume):
        """
        Args:
            volume (np.ndarray): 3D volume (D, H, W), dtype=uint16
        Returns:
            np.ndarray: Augmented volume, dtype=uint16
        """
        scale = random.uniform(*self.scale_range)
        bias = random.uniform(*self.bias_range)

        augmented = volume.astype(np.float32) * scale + bias * 65535
        augmented = np.clip(augmented, 0, 65535).astype(np.uint16)

        return augmented
    

class OCTAInpaintingDataset(Dataset):
    def __init__(self, volume_triples: list, stack_size=5, transform=None):
        """
        Args:
            volume_triples (list): List of tuples [(corrupted_path, clean_path, mask_path)]
            stack_size (int): Number of slices in input stack (must be odd)
            transform (callable): Optional transform to apply to (stack, target)
        """
        assert stack_size % 2 == 1, "Stack size must be odd"
        self.stack_size = stack_size
        self.pad = stack_size // 2
        self.transform = transform

        self.data = []  # Final list of (stack, target) pairs

        for corrupted_path, clean_path, mask_path in volume_triples:
            corrupted = tiff.imread(corrupted_path)
            clean = tiff.imread(clean_path)
            mask = tiff.imread(mask_path)

            if self.transform:
                corrupted = self.transform(corrupted)
                clean = self.transform(clean)

            assert corrupted.shape == clean.shape
            assert corrupted.shape[0] == mask.shape[0], "Mismatch in number of slices"

            if mask.ndim == 3:
                if np.all((mask == 0) | (mask == 1)):
                    mask = mask[:, 0, 0]
                else:
                    raise ValueError("Unexpected mask format: expected binary 0/1 values per slice.")
            elif mask.ndim == 1:
                pass
            else:
                raise ValueError("Unsupported mask dimensionality")

            padded = np.pad(corrupted, ((self.pad, self.pad), (0, 0), (0, 0)), mode='edge')

            for idx in np.where(mask == 1)[0]:  # Only missing slices are used
                if idx < self.pad or idx >= corrupted.shape[0] - self.pad:
                    continue  # Skip edge slices with invalid context
                
                stack = padded[idx:idx + stack_size]  # shape: (stack_size, H, W)
                target = clean[idx]                  # shape: (H, W)
                self.data.append((stack, target))  # Only stack and target

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        stack, target = self.data[idx]

        # Convert to torch tensors and normalize to [0, 1]
        stack = torch.from_numpy(stack).float() / 65535.0             # (stack_size, H, W)
        target = torch.from_numpy(target).float().unsqueeze(0) / 65535.0  # (1, H, W)

        return stack, target
