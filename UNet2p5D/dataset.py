import torch
from torch.utils.data import Dataset
import tifffile as tiff
import numpy as np
import matplotlib.pyplot as plt
import random
import os
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
        noise = np.random.normal(0, self.noise_std, size=stack.shape)

        stack = stack * scale + noise + bias
        target = target * scale + bias

        stack = np.clip(stack, 0.0, 1.0)
        target = np.clip(target, 0.0, 1.0)

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

        volume = volume * scale + bias
        volume = np.clip(volume, 0.0, 1.0)

        return volume
    

# Known artifact regions to avoid (inclusive)
artifact_exclusion = {
    "1.1_OCTA_Vol1_Processed_Cropped_gt":
        set(range(965, 982)) | set(range(0, 6)) | set(range(994, 1000)) | set(range(380, 630)),
    "1.2_OCTA_Vol2_Processed_Cropped_gt":
        set(range(343, 361)) | set(range(472, 488)) | set(range(916, 929)) | set(range(0, 6)) | set(range(994, 1000)),
    "1.4_OCTA_Vol1_Processed_Cropped_gt": 
        set(range(518, 536)) | set(range(548, 564)) | set(range(621, 639)) |
        set(range(656, 671)) | set(range(714, 731)) | set(range(980, 998)) | set(range(0, 6)) | set(range(994, 1000)),
    "2.1_OCTA_Vol2_Processed_Cropped_gt":
        set(range(565, 585)) | set(range(603, 621)) | set(range(0, 6)) | set(range(994, 1000)),
    "2.2_OCTA_Vol2_Processed_Cropped_gt":
        set(range(99, 119)) | set(range(158, 176)) | set(range(492, 510)) | set(range(0, 6)) | set(range(994, 1000)),
    "3.4_OCTA_Vol2_Processed_Cropped_gt": 
        set(range(242, 262)) | set(range(266, 286)) | set(range(607, 622)) |
        set(range(634, 650)) | set(range(852, 871)) | set(range(881, 898)) | set(range(0, 6)) | set(range(994, 1000)),
    "5.3_OCTA_Vol1_Processed_Cropped_gt": 
        set(range(102, 116)) | set(range(440, 445)) | set(range(464, 488)) |
        set(range(498, 513)) | set(range(893, 908)) | set(range(0, 6)) | set(range(994, 1000)),
    "6.3_OCTA_Vol2_Processed_Cropped_gt": 
        set(range(229, 247)) | set(range(742, 761)) | set(range(0, 6)) | set(range(985, 1000)),
    "9.2_OCTA_Vol1_Processed_Cropped_gt": 
        set(range(254, 274)) | set(range(343, 363)) | set(range(761, 780)) | set(range(0, 6)) | set(range(994, 1000)),
    "14.4_OCTA_Vol1_Processed_Cropped_gt": 
        set(range(72, 91)) | set(range(155, 175)) | set(range(183, 202)) | set(range(0, 17)) | set(range(994, 1000)),
    "16.3_OCTA_Vol2_Processed_Cropped_gt": 
        set(range(127, 144)) | set(range(220, 238)) | set(range(562, 579)) |
        set(range(618, 636)) | set(range(0, 17)) | set(range(994, 1000)),
    "22.1_OCTA_Vol2_Processed_Cropped_gt": 
        set(range(266, 286)) | set(range(367, 387)) | set(range(631, 651)) |
        set(range(727, 747)) | set(range(762, 780)) | set(range(894, 912)) |
        set(range(973, 993)) | set(range(0, 17)) | set(range(994, 1000)),
    # "15.4_OCTA_Vol2_Processed_Cropped_gt": 
    #     set(range(0, 18)) | set(range(218, 238)) | set(range(419, 438)) |
    #     set(range(449, 468)) | set(range(725, 743)) | set(range(998, 1000)),
    # "16.2_OCTA_Vol1_Processed_Cropped_gt": 
    #     set(range(358, 376)) | set(range(0, 2)) | set(range(998, 1000)),
    # "16.3_OCTA_Vol2_Processed_Cropped_gt":
    #     set(range(0, 20)) | set(range(126, 145)) | set(range(221, 238)) | 
    #     set(range(561, 579)) | set(range(618, 636)) | set(range(998, 1000)),
    # "22.1_OCTA_Vol2_Processed_Cropped_gt":
    #     set(range(266, 287)) | set(range(368, 388)) | set(range(631, 651)) | set(range(727, 747)) | set(range(761, 781)) | 
    #     set(range(894, 912)) | set(range(973, 993)) | set(range(0, 2)) | set(range(998, 1000)),
    # "25.3_OCTA_Vol1_Processed_Cropped_gt":
    #     set(range(85, 105)) | set(range(372, 388)) | set(range(975, 994)) | set(range(0, 2)) | set(range(998, 1000)),
    # "35.2_OCTA_Vol2_Processed_Cropped_gt": 
    #     set(range(0, 2)) | set(range(998, 1000)),
}


class OCTAInpaintingDataset(Dataset):
    def __init__(self, volume_triples: list, stack_size=5, transform=None, 
                 volume_transform=None, static_corruptions=False, stride=1, debug=False):
        """
        Args:
            volume_triples (list): List of tuples [(corrupted_path, clean_path, mask_path)].
            stack_size (int): Number of slices in input stack (must be odd).
            transform (callable): Optional transform to apply to (stack, target) pairs
                                  (intensity augmentation, etc.).
            volume_transform (callable): Optional transform to apply to whole volume 
                                         (e.g. flips, brightness) before slicing.
            static_corruptions (bool): If False, use online random dropouts (training mode). 
                            If True, use fixed pre-corrupted data (validation/test mode).
            stride (int): Step size between consecutive target slices in online corruption mode.
                            (Default 1 for full overlap.)
        """
        assert stack_size % 2 == 1, "Stack size must be odd"
        self.stack_size = stack_size
        self.pad = stack_size // 2
        self.transform = transform
        self.volume_transform = volume_transform
        self.static_corruptions = static_corruptions
        self.stride = stride
        self.debug = debug

        if static_corruptions:
            # --- Static mode: use pre-generated corrupted volumes and masks ---
            self.data = []  # will hold (stack, target) pairs

            for corrupted_path, clean_path, mask_path in volume_triples:
                # corrupted = tiff.imread(corrupted_path)
                # clean = tiff.imread(clean_path)
                # mask = tiff.imread(mask_path)

                corrupted = tiff.imread(corrupted_path).astype(np.float32)
                corrupted = corrupted / (corrupted.max() + 1e-5)
                clean = tiff.imread(clean_path).astype(np.float32)
                clean = clean / (clean.max() + 1e-5)
                mask = tiff.imread(mask_path)

                if self.debug:
                    log(f"[INIT] Loaded volume: {os.path.basename(clean_path)}")
                    log(f"       Corrupted dtype: {corrupted.dtype}, min: {corrupted.min():.3f}, max: {corrupted.max():.3f}")
                    log(f"       Clean dtype: {clean.dtype}, min: {clean.min():.3f}, max: {clean.max():.3f}")
                    log(f"       Mask shape: {mask.shape}, dtype: {mask.dtype}, unique: {np.unique(mask)}")

                # Apply any volume-level augmentation
                if self.volume_transform:
                    corrupted = self.volume_transform(corrupted)
                    # clean = self.volume_transform(clean)

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
            # --- Online corruption mode: build indices for on-the-fly random dropouts ---
            self.clean_volumes = []   # list of clean volumes (np.uint16 arrays)
            self.padded_volumes = []  # list of padded clean volumes for easy indexing
            self.indices = []         # list of (volume_index, center_slice_idx) pairs

            for vol_idx, (_, clean_path, _) in enumerate(volume_triples):
                # clean = tiff.imread(clean_path)
                clean = tiff.imread(clean_path).astype(np.float32)
                clean = clean / (clean.max() + 1e-5)

                # Apply volume-level augmentation to the clean volume
                if self.volume_transform:
                    before_volume = clean.copy()
                    clean = self.volume_transform(clean)
                orig_len = clean.shape[0]


                if self.debug and self.volume_transform and vol_idx == 0:  # Only show for first volume to avoid overload
                    slice_idx = clean.shape[0] // 2  # Use center slice for consistency
                    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
                    axs[0].imshow(before_volume[slice_idx], cmap='gray')
                    axs[0].set_title("Pre-VolumeTransform Slice")
                    axs[1].imshow(clean[slice_idx], cmap='gray')
                    axs[1].set_title("Post-VolumeTransform Slice")
                    plt.suptitle(f"Volume Transform Effect: Volume idx {vol_idx}")
                    plt.tight_layout()
                    plt.show()

                    print(f"[VOLUME_TRANSFORM DEBUG] Volume idx {vol_idx}, Slice {slice_idx}")
                    print(f"  Pre min: {before_volume.min():.4f}, max: {before_volume.max():.4f}, mean: {before_volume.mean():.4f}")
                    print(f"  Post min: {clean.min():.4f}, max: {clean.max():.4f}, mean: {clean.mean():.4f}")


                # Store the clean volume (for clarity) and also a padded copy for slicing
                self.clean_volumes.append(clean)
                padded = np.pad(clean, ((self.pad, self.pad), (0, 0), (0, 0)), mode='edge')
                self.padded_volumes.append(padded)

                volume_name = os.path.basename(clean_path).replace(".tif", "")
                excluded = artifact_exclusion.get(volume_name, set())

                # For every valid center slice (skipping edges for full context), 
                # step by 'stride' to create overlapping stacks
                for idx in range(self.pad, orig_len - self.pad, self.stride):
                    if idx in excluded:
                        continue
                    self.indices.append((vol_idx, idx))


    def __len__(self):
        if self.static_corruptions:
            return len(self.data)
        else:
            return len(self.indices)

    def __getitem__(self, idx):
        if self.static_corruptions:
            # --- Static mode: return precomputed (stack, target) pair ---
            stack, target = self.data[idx]
            stack = torch.from_numpy(stack).float()
            target = torch.from_numpy(target).float().unsqueeze(0)
        else:
            # --- Online corruption mode: generate stack and apply random dropouts ---
            vol_idx, center_idx = self.indices[idx]
            padded_vol = self.padded_volumes[vol_idx]
            
            # Extract the full stack around center_idx
            stack = padded_vol[center_idx: center_idx + self.stack_size].copy()  # shape: (stack_size, H, W)
            # The true target slice is the center slice (no dropout)
            # Note: padded_vol[center_idx + self.pad] corresponds to the original clean slice
            target = padded_vol[center_idx + self.pad].copy()  # shape: (H, W)

            # Debug: log stack metadata
            if self.debug and idx < 3:  # Log only first 3 for brevity
                log(f"[IDX {idx}] Volume {vol_idx} | Center Slice (unpadded idx): {center_idx}")
                log(f"Stack indices (padded): {list(range(center_idx, center_idx + self.stack_size))}")
                log(f"Stack shape: {stack.shape}")

            # Apply intensity transform
            if self.transform:
                if self.debug and idx == 5:
                    pre_stack = stack.copy()
                    pre_target = target.copy()

                stack, target = self.transform(stack, target)


                if self.debug and idx == 5:
                    print(f"\n=== AUGMENTATION DEBUG: Sample idx {idx} ===")
                    print("Pre-Augmentation:")
                    print(f"  Stack min: {pre_stack.min():.4f}, max: {pre_stack.max():.4f}, mean: {pre_stack.mean():.4f}")
                    print(f"  Target min: {pre_target.min():.4f}, max: {pre_target.max():.4f}, mean: {pre_target.mean():.4f}")
                    print("Post-Augmentation:")
                    print(f"  Stack min: {stack.min():.4f}, max: {stack.max():.4f}, mean: {stack.mean():.4f}")
                    print(f"  Target min: {target.min():.4f}, max: {target.max():.4f}, mean: {target.mean():.4f}")

                    mid = self.stack_size // 2

                    fig, axs = plt.subplots(2, 4, figsize=(18, 9))

                    axs[0, 0].imshow(pre_stack[mid - 1], cmap='gray')
                    axs[0, 0].set_title("Pre-Aug: Center-1")

                    axs[0, 1].imshow(pre_stack[mid], cmap='gray')
                    axs[0, 1].set_title("Pre-Aug: Center")

                    axs[0, 2].imshow(pre_stack[mid + 1], cmap='gray')
                    axs[0, 2].set_title("Pre-Aug: Center+1")

                    axs[0, 3].imshow(pre_target, cmap='gray')
                    axs[0, 3].set_title("Pre-Aug: Target")

                    axs[1, 0].imshow(stack[mid - 1], cmap='gray')
                    axs[1, 0].set_title("Post-Aug: Center-1")

                    axs[1, 1].imshow(stack[mid], cmap='gray')
                    axs[1, 1].set_title("Post-Aug: Center")

                    axs[1, 2].imshow(stack[mid + 1], cmap='gray')
                    axs[1, 2].set_title("Post-Aug: Center+1")

                    axs[1, 3].imshow(target, cmap='gray')
                    axs[1, 3].set_title("Post-Aug: Target")

                    plt.suptitle(f"Augmentation Effect for Sample idx {idx}")
                    plt.tight_layout()
                    plt.show()


            stack = stack.astype(np.float32)
            target = target.astype(np.float32)

            # Determine which neighbor slices to drop (set to zero).
            # Always drop the center (target) slice to simulate missing slice.
            # Additionally drop a random subset of the other slices, but leave at least 2 intact.
            neighbors = list(range(self.stack_size))
            center_pos = self.pad  # index of center slice in the stack
            neighbors.remove(center_pos)

            # Maximum drops ensuring that at least 5 neighbors remain.
            # max_drops = len(neighbors) - 5
            # drop_n = random.randint(0, max_drops)
            drop_n = random.randint(0,4)
            # drop_n = 0
            drop_indices = random.sample(neighbors, drop_n)
            # Drop the selected neighbor slices
            for di in drop_indices:
                stack[di] = 0.0  # set entire slice to zero (uint16 zero)

            # Finally, drop the center slice
            stack[center_pos] = 0.0

            # Log corruption if debug enabled
            if self.debug and idx < 3:
                log(f"[IDX {idx}] Volume {vol_idx} | Center Slice (unpadded idx): {center_idx}")
                log(f"Stack indices (padded): {list(range(center_idx, center_idx + self.stack_size))}")
                log(f"Stack shape: {stack.shape}")
                log(f"Dropped slices: {sorted(drop_indices + [center_pos])}")
                valid_indices = [i for i in range(self.stack_size) if i not in drop_indices + [center_pos]]
                log(f"Remaining valid slices in stack: {valid_indices}")

            # target = target.unsqueeze(0)  # shape: (1, H, W)
            target = np.expand_dims(target, axis=0)  # (1, H, W)

            # Convert to tensors
            stack = torch.from_numpy(stack).float()  # (stack_size, H, W)
            target = torch.from_numpy(target).float()  # (1, H, W)

        if self.debug and idx == 5:
            log(f"[GETITEM {idx}] Stack dtype: {stack.dtype}, shape: {stack.shape}")
            log(f"  Stack min: {stack.min():.4f}, max: {stack.max():.4f}, mean: {stack.mean():.4f}")
            log(f"  Target min: {target.min():.4f}, max: {target.max():.4f}, mean: {target.mean():.4f}")

            mid = stack.shape[0] // 2

            # Log statistics for three slices
            def log_slice_stats(title, tensor):
                print(f"[{title}] dtype: {tensor.dtype}, shape: {tensor.shape}")
                print(f"  Min: {tensor.min().item():.4f}, Max: {tensor.max().item():.4f}, Mean: {tensor.mean().item():.4f}")

            print(f"\n=== DEBUG INFO: Sample idx {idx} ===")
            log_slice_stats("Stack Center-1", stack[mid - 1])
            log_slice_stats("Stack Center", stack[mid])
            log_slice_stats("Stack Center+1", stack[mid + 1])
            log_slice_stats("Target Slice", target[0])

            fig, axs = plt.subplots(2, 3, figsize=(15, 8))

            axs[0, 0].imshow(stack[mid - 1].cpu().numpy(), cmap='gray')
            axs[0, 0].set_title("Input: Center-1")

            axs[0, 1].imshow(stack[mid].cpu().numpy(), cmap='gray')
            axs[0, 1].set_title("Input: Center")

            axs[0, 2].imshow(stack[mid + 1].cpu().numpy(), cmap='gray')
            axs[0, 2].set_title("Input: Center+1")

            axs[1, 0].imshow(target[0].cpu().numpy(), cmap='gray')
            axs[1, 0].set_title("Target (repeated)")

            axs[1, 1].imshow(target[0].cpu().numpy(), cmap='gray')
            axs[1, 1].set_title("Target (center)")

            axs[1, 2].imshow(target[0].cpu().numpy(), cmap='gray')
            axs[1, 2].set_title("Target (repeated)")

            plt.suptitle(f"Visualization for Sample idx {idx} (Stack + Target)")
            plt.tight_layout()
            plt.show()


        # Validity mask: 1 if slice has non-zero content, else 0
        valid_mask_stack = (stack.sum(dim=(1, 2)) > 1e-3).float()  # (stack_size,)

        return stack, target, valid_mask_stack
