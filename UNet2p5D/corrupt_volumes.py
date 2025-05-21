import os
import numpy as np
import random
import tifffile as tiff

# Known artifact regions to avoid (inclusive)
artifact_exclusion = {
    "1.1_OCTA_Vol1_Processed_Cropped_gt":
        set(range(967, 980)),
    "1.2_OCTA_Vol2_Processed_Cropped_gt":
        set(range(345, 359)) | set(range(474, 486)) | set(range(918, 927)),
    "1.4_OCTA_Vol1_Processed_Cropped_gt": 
        set(range(520, 534)) | set(range(550, 562)) | set(range(623, 637)) |
        set(range(658, 669)) | set(range(716, 729)) | set(range(982, 996)),
    "3.4_OCTA_Vol2_Processed_Cropped_gt": 
        set(range(244, 260)) | set(range(268, 284)) | set(range(609, 620)) |
        set(range(636, 648)) | set(range(854, 869)) | set(range(883, 896)),
    "5.3_OCTA_Vol1_Processed_Cropped_gt": 
        set(range(104, 114)) | set(range(442, 443)) | set(range(466, 486)) |
        set(range(500, 511)) | set(range(895, 906)),
}


def simulate_missing_bscans(volume: np.ndarray, volume_name: str, missing_fraction: float = 0.1, block_size_range=(1, 4)):
    """
    Simulate missing B-scans in an OCTA volume, avoiding predefined artifact regions.

    Args:
        volume (np.ndarray): Input 3D volume of shape (D, H, W)
        volume_name (str): Identifier used to retrieve exclusion ranges
        missing_fraction (float): Fraction of slices to remove
        block_size_range (tuple): Range of block sizes to simulate
    
    Returns:
        corrupted_volume, mask, missing_indices
    """
    D, H, W = volume.shape
    num_missing = int(D * missing_fraction)
    corrupted_volume = volume.copy()
    mask = np.zeros(D, dtype=np.uint8)
    missing_indices = set()

    available = np.ones(D, dtype=bool)
    artifact_indices = artifact_exclusion.get(volume_name, set())

    print(f"Exclusion indices for {volume_name}: {sorted(artifact_indices)}")

    while np.sum(mask) < num_missing:
        block_size = random.randint(*block_size_range)

        valid_starts = []
        for start_idx in range(D - block_size + 1):
            block = list(range(start_idx, start_idx + block_size))
            if available[start_idx:start_idx + block_size].all() and not any(idx in artifact_indices for idx in block):
                valid_starts.append(start_idx)

        if not valid_starts:
            break  # no space left

        start_idx = random.choice(valid_starts)
        block_indices = list(range(start_idx, start_idx + block_size))

        for idx in block_indices:
            corrupted_volume[idx] = 0
            mask[idx] = 1
            available[idx] = False
        missing_indices.update(block_indices)

    return corrupted_volume, mask, sorted(missing_indices)


input_path = "/media/admin/Expansion/Mosaic_Data_for_Ipeks_Group/OCT_Inpainting_Testing/1.1_OCTA_Vol1_Processed_Cropped_gt.tif"
# input_path = "/media/admin/Expansion/Mosaic_Data_for_Ipeks_Group/OCT_Inpainting_Testing/1.2_OCTA_Vol2_Processed_Cropped_gt.tif"
# input_path = "/media/admin/Expansion/Mosaic_Data_for_Ipeks_Group/OCT_Inpainting_Testing/1.4_OCTA_Vol1_Processed_Cropped_gt.tif"
# input_path = "/media/admin/Expansion/Mosaic_Data_for_Ipeks_Group/OCT_Inpainting_Testing/3.4_OCTA_Vol2_Processed_Cropped_gt.tif"
# input_path = "/media/admin/Expansion/Mosaic_Data_for_Ipeks_Group/OCT_Inpainting_Testing/5.3_OCTA_Vol1_Processed_Cropped_gt.tif"


# Load volume
volume = tiff.imread(input_path)
print("Volume shape:", volume.shape)

# Derive volume name (no .tif)
volume_name = os.path.splitext(os.path.basename(input_path))[0]

# Simulate missing B-scans
corrupted_volume, mask, missing = simulate_missing_bscans(volume, volume_name, missing_fraction=0.16, block_size_range=(1, 4))

# Save corrupted and mask volumes
base_dir = os.path.dirname(input_path)
base_name = os.path.splitext(os.path.basename(input_path))[0]

# Replace "_gt" with the appropriate suffix
corrupted_path = os.path.join(base_dir, f"{base_name.replace('_gt', '_corrupted')}.tif")
mask_path = os.path.join(base_dir, f"{base_name.replace('_gt', '_mask')}.tif")

tiff.imwrite(corrupted_path, corrupted_volume.astype(np.uint16), imagej=True)
mask_volume = np.tile(mask[:, None, None], (1, volume.shape[1], volume.shape[2]))
tiff.imwrite(mask_path, mask_volume.astype(np.uint8), imagej=True)

print(f"Corrupted volume saved to: {corrupted_path}")
print(f"Mask saved to: {mask_path}")
print(f"Removed {len(missing)} B-scans at indices: {missing[:10]}...")
