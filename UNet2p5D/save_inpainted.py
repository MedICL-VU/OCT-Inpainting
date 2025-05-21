import torch
import numpy as np


def inpaint_volume_with_model(model, corrupted_volume, mask, device, stack_size=5):
    """
    Apply 2.5D model to missing slices in a volume.
    Uses recursive updates and inpaints corrupted blocks symmetrically from outside-in.
    """
    model.eval()
    pad = stack_size // 2
    D, H, W = corrupted_volume.shape

    reconstructed_volume = corrupted_volume.copy().astype(np.uint16)
    padded = np.pad(reconstructed_volume, ((pad, pad), (0, 0), (0, 0)), mode='edge')

    # Helper: get all contiguous regions in the mask
    def get_corrupted_blocks(mask):
        blocks = []
        current = []
        for i in range(len(mask)):
            if mask[i] == 1:
                current.append(i)
            elif current:
                blocks.append(current)
                current = []
        if current:
            blocks.append(current)
        return blocks

    corrupted_blocks = get_corrupted_blocks(mask)

    with torch.no_grad():
        for block in corrupted_blocks:
            # Inpaint symmetrically from edges to center
            left = 0
            right = len(block) - 1
            while left <= right:
                for idx in [block[left], block[right]] if left != right else [block[left]]:
                    stack = padded[idx:idx + stack_size]  # (stack_size, H, W)
                    stack_tensor = torch.from_numpy(stack).unsqueeze(0).float().to(device) / 65535.0
                    output = model(stack_tensor)
                    pred = output.squeeze().cpu().numpy() * 65535.0
                    pred = np.clip(pred, 0, 65535).astype(np.uint16)

                    reconstructed_volume[idx] = pred
                    padded[idx + pad] = pred  # Update padding for recursive effect
                left += 1
                right -= 1

    return reconstructed_volume


# def inpaint_volume_with_model(model, corrupted_volume, mask, device, stack_size=5):
#     """
#     Apply 2.5D model to missing slices in a volume.
    
#     Args:
#         model: trained UNet2p5D model
#         corrupted_volume: (D, H, W) array with zeroed missing B-scans
#         mask: (D,) binary array, 1 = missing
#         stack_size: number of slices in input stack
    
#     Returns:
#         np.ndarray of shape (D, H, W): inpainted volume
#     """
#     model.eval()
#     pad = stack_size // 2
#     D, H, W = corrupted_volume.shape
#     padded = np.pad(corrupted_volume, ((pad, pad), (0, 0), (0, 0)), mode='edge')
#     inpainted = corrupted_volume.copy()

#     with torch.no_grad():
#         for idx in np.where(mask == 1)[0]:
#             stack = padded[idx:idx + stack_size]  # (stack_size, H, W)
#             # stack = torch.from_numpy(stack).unsqueeze(0).float().cuda() / 65535.0  # (1, 5, H, W)
#             stack = torch.from_numpy(stack).unsqueeze(0).float().to(device) / 65535.0
#             output = model(stack)  # (1, 1, H, W)
#             pred = output.squeeze().cpu().numpy() * 65535.0
#             inpainted[idx] = np.clip(pred, 0, 65535).astype(np.uint16)

#     return inpainted


# def smooth_rescale_reconstructed_volume(reconstructed_volume, corrupted_volume, mask, blend_factor=0.5):
#     """
#     Args:
#         reconstructed_volume (np.ndarray): (D, H, W) predicted volume
#         corrupted_volume (np.ndarray): (D, H, W) original corrupted volume
#         mask (np.ndarray): (D,) binary mask (1 = missing slice, 0 = valid slice)
#         blend_factor (float): weight toward target brightness (0 = no adjustment, 1 = full match)
    
#     Returns:
#         np.ndarray: Brightness-corrected reconstructed volume
#     """
#     D, H, W = reconstructed_volume.shape
#     corrected_volume = reconstructed_volume.astype(np.float32).copy()

#     for idx in range(D):
#         if mask[idx] == 1:  # Only correct reconstructed slices
#             # Find nearest valid slices
#             neighbors = []
#             for offset in range(1, D):
#                 before_idx = idx - offset
#                 after_idx = idx + offset

#                 if before_idx >= 0 and mask[before_idx] == 0:
#                     neighbors.append(corrupted_volume[before_idx])
#                 if after_idx < D and mask[after_idx] == 0:
#                     neighbors.append(corrupted_volume[after_idx])

#                 if len(neighbors) >= 2:
#                     break

#             if len(neighbors) == 0:
#                 continue

#             neighbor_mean = np.mean([np.mean(n) for n in neighbors])
#             recon_mean = np.mean(corrected_volume[idx])

#             if recon_mean > 1e-6:
#                 scale_factor = neighbor_mean / recon_mean

#                 # Instead of applying full scaling, smoothly blend
#                 corrected_volume[idx] = corrected_volume[idx] * (1 - blend_factor) + corrected_volume[idx] * scale_factor * blend_factor

#     return np.clip(corrected_volume, 0, 65535).astype(np.uint16)
