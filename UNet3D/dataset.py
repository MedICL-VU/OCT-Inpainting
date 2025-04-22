import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import tifffile as tiff
import numpy as np

class OCTAInpaintingDataset(Dataset):
    def __init__(self, volume_triples: list, stack_size=16, transform=None):
        """
        Args:
            volume_triples (list): List of tuples [(corrupted_path, clean_path, mask_path)]
            stack_size (int): Number of slices in input stack (must be odd)
            transform (callable): Optional transform to apply to (stack, target)
        """
        # assert stack_size % 2 == 1, "Stack size must be odd"
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
                if idx + stack_size > clean.shape[0]:
                    continue  # skip samples that can't produce a full 3D stack

                stack = padded[idx:idx + stack_size]  # shape: (stack_size, H, W)
                # target = clean[idx]  # shape: (H, W)
                target = clean[idx:idx + stack_size]  # 3D block
                self.data.append((stack, target))


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        stack, target = self.data[idx]

        # Expand dimensions to match (C=1, D, H, W)
        # stack = torch.from_numpy(stack).unsqueeze(0).float() / 65535.0  # (1, D, H, W)
        # target = torch.from_numpy(target).float().unsqueeze(0) / 65535.0  # (1, D, H, W)

        stack = torch.from_numpy(stack.copy()).float().unsqueeze(0) / 65535.0
        target = torch.from_numpy(target.copy()).float().unsqueeze(0) / 65535.0

        # === Auto-pad to nearest multiple of 16 ===
        def pad_to_multiple_16(t):
            _, _, h, w = t.shape
            pad_h = (16 - h % 16) % 16
            pad_w = (16 - w % 16) % 16
            return F.pad(t, (0, pad_w, 0, pad_h), mode='replicate')

        stack = pad_to_multiple_16(stack)
        target = pad_to_multiple_16(target)


        return stack, target
