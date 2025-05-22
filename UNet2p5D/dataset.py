import torch
from torch.utils.data import Dataset
import tifffile as tiff
import numpy as np
import random
from utils import log


class IntensityAugment:
    def __init__(self, scale_range=(0.9, 1.1), noise_std=0.01, bias_range=(-0.05, 0.05)):
        self.scale_range = scale_range
        self.noise_std = noise_std
        self.bias_range = bias_range

    def __call__(self, stack, target):
        # Apply the same random scale and bias to both stack and target; add noise only to stack.
        scale = random.uniform(*self.scale_range)
        bias = random.uniform(*self.bias_range)
        noise = np.random.normal(0, self.noise_std * 65535, size=stack.shape)

        stack = stack.astype(np.float32) * scale + noise + bias * 65535
        target = target.astype(np.float32) * scale + bias * 65535

        stack = np.clip(stack, 0, 65535).astype(np.uint16)
        target = np.clip(target, 0, 65535).astype(np.uint16)

        return stack, target
    

class VolumeLevelIntensityAugment:
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
    def __init__(self, volume_triples: list, stack_size=5, transform=None, 
                 volume_transform=None, dynamic=False, stride=1, debug=False):
        """
        Args:
            volume_triples (list): List of tuples [(corrupted_path, clean_path, mask_path)].
            stack_size (int): Number of slices in input stack (must be odd).
            transform (callable): Optional transform to apply to (stack, target) pairs
                                  (intensity augmentation, etc.).
            volume_transform (callable): Optional transform to apply to whole volume 
                                         (e.g. flips, brightness) before slicing.
            dynamic (bool): If True, use on-the-fly random dropouts (training mode). 
                            If False, use fixed pre-corrupted data (validation/test mode).
            stride (int): Step size between consecutive target slices in dynamic mode.
                            (Default 1 for full overlap.)
        """
        assert stack_size % 2 == 1, "Stack size must be odd"
        self.stack_size = stack_size
        self.pad = stack_size // 2
        self.transform = transform
        self.volume_transform = volume_transform
        self.dynamic = dynamic
        self.stride = stride
        self.debug = debug

        if not dynamic:
            # --- Static mode: use pre-generated corrupted volumes and masks ---
            self.data = []  # will hold (stack, target) pairs

            for corrupted_path, clean_path, mask_path in volume_triples:
                corrupted = tiff.imread(corrupted_path)
                clean = tiff.imread(clean_path)
                mask = tiff.imread(mask_path)

                # Apply any volume-level augmentation to both corrupted and clean
                if self.volume_transform:
                    corrupted = self.volume_transform(corrupted)
                    clean = self.volume_transform(clean)

                assert corrupted.shape == clean.shape, "Corrupted and clean volumes must match shape"
                assert corrupted.shape[0] == mask.shape[0], "Number of slices in volume and mask must match"

                # Convert mask to 1D binary array if needed
                if mask.ndim == 3:
                    if np.all((mask == 0) | (mask == 1)):
                        mask = mask[:, 0, 0]
                    else:
                        raise ValueError("Unexpected mask format: expected binary 0/1 per slice")
                elif mask.ndim != 1:
                    raise ValueError("Unsupported mask dimensionality")

                # Pad the corrupted volume on the slice dimension for context at edges
                padded = np.pad(corrupted, ((self.pad, self.pad), (0, 0), (0, 0)), mode='edge')

                # Build stacks only for slices marked as missing (mask==1)
                for idx in np.where(mask == 1)[0]:
                    # Skip slices too close to volume boundary (invalid context)
                    if idx < self.pad or idx >= corrupted.shape[0] - self.pad:
                        continue

                    # Extract stack around the target slice (centered)
                    stack = padded[idx: idx + stack_size]  # shape: (stack_size, H, W)
                    target = clean[idx]                    # shape: (H, W)
                    self.data.append((stack, target))
        else:
            # --- Dynamic mode: build indices for on-the-fly random dropouts ---
            self.clean_volumes = []   # list of clean volumes (np.uint16 arrays)
            self.padded_volumes = []  # list of padded clean volumes for easy indexing
            self.indices = []         # list of (volume_index, center_slice_idx) pairs

            for vol_idx, (_, clean_path, _) in enumerate(volume_triples):
                clean = tiff.imread(clean_path)
                # Apply volume-level augmentation to the clean volume
                if self.volume_transform:
                    clean = self.volume_transform(clean)
                orig_len = clean.shape[0]

                # Store the clean volume (for clarity) and also a padded copy for slicing
                self.clean_volumes.append(clean)
                padded = np.pad(clean, ((self.pad, self.pad), (0, 0), (0, 0)), mode='edge')
                self.padded_volumes.append(padded)

                # For every valid center slice (skipping edges for full context), 
                # step by 'stride' to create overlapping stacks
                for idx in range(self.pad, orig_len - self.pad, self.stride):
                    self.indices.append((vol_idx, idx))

    def __len__(self):
        if self.dynamic:
            return len(self.indices)
        else:
            return len(self.data)

    def __getitem__(self, idx):
        if not self.dynamic:
            # --- Static mode: return precomputed (stack, target) pair ---
            stack, target = self.data[idx]
        else:
            # --- Dynamic mode: generate stack and apply random dropouts ---
            vol_idx, center_idx = self.indices[idx]
            padded_vol = self.padded_volumes[vol_idx]
            
            # Extract the full stack around center_idx
            stack = padded_vol[center_idx: center_idx + self.stack_size]  # shape: (stack_size, H, W)
            # The true target slice is the center slice (no dropout)
            # Note: padded_vol[center_idx + self.pad] corresponds to the original clean slice
            target = padded_vol[center_idx + self.pad].copy()  # shape: (H, W)

            # Debug: log stack metadata
            if self.debug and idx < 3:  # Log only first 3 for brevity
                log(f"[IDX {idx}] Volume {vol_idx} | Center Slice (unpadded idx): {center_idx}")
                log(f"Stack indices (padded): {list(range(center_idx, center_idx + self.stack_size))}")
                log(f"Stack shape: {stack.shape}")

            # Apply intensity transform (e.g. scaling, bias, noise) before dropout
            if self.transform:
                stack, target = self.transform(stack, target)

            # Determine which neighbor slices to drop (set to zero).
            # Always drop the center (target) slice to simulate missing slice.
            # Additionally drop a random subset of the other slices, but leave at least 2 intact.
            neighbors = list(range(self.stack_size))
            center_pos = self.pad  # index of center slice in the stack
            neighbors.remove(center_pos)

            # Maximum drops = (len(neighbors) - 2) to ensure at least 2 neighbors remain.
            # max_drops = len(neighbors) - 2
            max_drops = len(neighbors) - 4
            # drop_n = random.randint(0, max_drops)
            drop_n = random.randint(1,3)
            drop_indices = random.sample(neighbors, drop_n)
            # Drop the selected neighbor slices
            for di in drop_indices:
                stack[di] = 0  # set entire slice to zero (uint16 zero)

            # Finally, drop the center slice
            stack[center_pos] = 0

            if self.debug and idx < 3:
                # Log dropout results
                log(f"Dropped slices: {sorted(drop_indices + [center_pos])}")
                valid_indices = [i for i in range(self.stack_size) if i not in drop_indices + [center_pos]]
                log(f"Remaining valid slices in stack: {valid_indices}")

        # Convert to torch tensors and normalize to [0,1]
        # stack: (stack_size, H, W), target: (H, W) -> unsqueeze to (1, H, W)
        # stack = torch.from_numpy(stack).float() / 65535.0
        # target = torch.from_numpy(target).float().unsqueeze(0) / 65535.0

        stack = torch.from_numpy(stack).float()
        target = torch.from_numpy(target).float()

        # Normalize each slice to [0, 1] individually
        stack_min = stack.view(self.stack_size, -1).min(dim=1)[0].view(-1, 1, 1)
        stack_max = stack.view(self.stack_size, -1).max(dim=1)[0].view(-1, 1, 1)
        stack = (stack - stack_min) / (stack_max - stack_min + 1e-5)

        target_min = target.min()
        target_max = target.max()
        target = (target - target_min) / (target_max - target_min + 1e-5)

        target = target.unsqueeze(0)  # shape: (1, H, W)

        return stack, target
