import torch
from torch.utils.data import Dataset
import tifffile as tiff
import numpy as np

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

            if mask.ndim == 3:
                mask = (mask[:, 0, 0] > 0).astype(np.uint8)
            else:
                mask = mask.astype(np.uint8)

            padded = np.pad(corrupted, ((self.pad, self.pad), (0, 0), (0, 0)), mode='edge')

            for idx in np.where(mask == 1)[0]:  # Only missing slices are used
                stack = padded[idx:idx + stack_size]  # shape: (stack_size, H, W)
                target = clean[idx]                  # shape: (H, W)
                self.data.append((stack, target))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        stack, target = self.data[idx]

        # Replace missing input slices with linear interpolation
        # Optional: could be zeros or blurred versions instead
        stack = stack.astype(np.float32)
        target = target.astype(np.float32)

        # Check for low-intensity (empty) slices — assume those are missing
        mean_vals = stack.mean(axis=(1, 2))
        low_mask = mean_vals < 1500  # threshold for 16-bit darkness (~black)

        # Simple interpolation for missing slices (in-place modification)
        for i in range(stack.shape[0]):
            if low_mask[i]:
                prev_i = max(i - 1, 0)
                next_i = min(i + 1, stack.shape[0] - 1)
                stack[i] = 0.5 * (stack[prev_i] + stack[next_i])

        if self.transform:
            stack, target = self.transform(stack, target)

        stack = torch.from_numpy(stack).float() / 65535.0             # (stack_size, H, W)
        target = torch.from_numpy(target).float().unsqueeze(0) / 65535.0  # (1, H, W)

        return stack, target
