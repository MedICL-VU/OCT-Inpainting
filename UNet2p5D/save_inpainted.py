import tifffile as tiff
import torch
import numpy as np


def inpaint_volume_with_model_original(model, corrupted_volume, mask, device, stack_size=5):
    """
    Apply 2.5D model to missing slices in a volume.
    
    Args:
        model: trained UNet2p5D model
        corrupted_volume: (D, H, W) array with zeroed missing B-scans
        mask: (D,) binary array, 1 = missing
        stack_size: number of slices in input stack
    
    Returns:
        np.ndarray of shape (D, H, W): inpainted volume
    """
    model.eval()
    pad = stack_size // 2
    D, H, W = corrupted_volume.shape
    padded = np.pad(corrupted_volume, ((pad, pad), (0, 0), (0, 0)), mode='edge')
    inpainted = corrupted_volume.copy()

    with torch.no_grad():
        for idx in np.where(mask == 1)[0]:
            stack = padded[idx:idx + stack_size]  # (stack_size, H, W)
            # stack = torch.from_numpy(stack).unsqueeze(0).float().cuda() / 65535.0  # (1, 5, H, W)
            stack = torch.from_numpy(stack).unsqueeze(0).float().to(device) / 65535.0
            output = model(stack)  # (1, 1, H, W)
            pred = output.squeeze().cpu().numpy() * 65535.0
            inpainted[idx] = np.clip(pred, 0, 65535).astype(np.uint16)

    return inpainted


def inpaint_volume_with_model_recursive_outsidein(model, corrupted_volume, mask, device, stack_size=5):
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


def inpaint_volume_with_model_noncorruptedslices(model, corrupted_volume, mask, device, stack_size=5):
    """
    Apply 2.5D model to missing slices in a volume using stack + validity mask.
    
    Args:
        model: trained UNet2p5D model
        corrupted_volume: (D, H, W) array with zeroed missing B-scans
        mask: (D,) binary array, 1 = missing
        stack_size: number of slices in input stack
    
    Returns:
        np.ndarray of shape (D, H, W): inpainted volume
    """
    model.eval()
    pad = stack_size // 2
    D, H, W = corrupted_volume.shape
    padded = np.pad(corrupted_volume, ((pad, pad), (0, 0), (0, 0)), mode='edge')
    inpainted = corrupted_volume.copy()

    with torch.no_grad():
        for idx in np.where(mask == 1)[0]:
            # Extract stack
            stack = padded[idx:idx + stack_size]  # shape: (stack_size, H, W)
            stack_tensor = torch.from_numpy(stack).unsqueeze(0).float().to(device) / 65535.0  # (1, stack_size, H, W)

            # Build corresponding validity mask: 1.0 where > 0, else 0.0
            validity = (stack_tensor > 0).float()  # (1, stack_size, H, W)

            # Concatenate stack + validity mask
            model_input = torch.cat([stack_tensor, validity], dim=1)  # shape: (1, 2*stack_size, H, W)

            # Predict
            output = model(model_input)  # (1, 1, H, W)
            pred = output.squeeze().cpu().numpy() * 65535.0
            inpainted[idx] = np.clip(pred, 0, 65535).astype(np.uint16)

    return inpainted


def inpaint_volume_recursive_noncorruptedslices_hybrid(model, volume, valid_mask, device, stack_size=5):
    """
    Recursively inpaints missing slices in a volume from the outside-in using validity-aware context.

    Args:
        model (torch.nn.Module): Trained PyTorch inpainting model.
        volume (np.ndarray): Input volume of shape (D, H, W), where D is number of B-scans.
        valid_mask (np.ndarray): Binary array of shape (D,) indicating valid (1) vs. missing (0) slices.
        stack_size (int): Number of slices to include as input to the model.
        device (str): Device for inference.

    Returns:
        volume (np.ndarray): Inpainted volume of shape (D, H, W).
        valid_mask (np.ndarray): Updated mask with 1's for all inpainted slices.
    """
    D, H, W = volume.shape
    half = stack_size // 2
    volume = volume.copy()
    valid_mask = valid_mask.copy()

    def get_valid_stack(center_idx):
        indices = list(range(center_idx - half, center_idx + half + 1))
        stack = []
        for i in indices:
            if 0 <= i < D and valid_mask[i] == 1:
                stack.append(volume[i])
            else:
                stack.append(np.zeros((H, W), dtype=np.float32))
        return np.stack(stack, axis=0)

    def find_missing_ranges(mask):
        missing = np.where(mask == 0)[0]
        if len(missing) == 0:
            return []
        groups = []
        start = missing[0]
        for i in range(1, len(missing)):
            if missing[i] != missing[i - 1] + 1:
                groups.append((start, missing[i - 1]))
                start = missing[i]
        groups.append((start, missing[-1]))
        return groups

    for start, end in find_missing_ranges(valid_mask):
        left, right = start, end
        while left <= right:
            for idx in [left, right] if left != right else [left]:
                input_stack = get_valid_stack(idx)
                input_tensor = torch.from_numpy(input_stack).unsqueeze(0).to(device).float() / 65535.0
                with torch.no_grad():
                    pred = model(input_tensor).cpu().squeeze().numpy() * 65535.0
                volume[idx] = pred
                valid_mask[idx] = 1
            left += 1
            right -= 1

    return volume, valid_mask


def smooth_rescale_reconstructed_volume(reconstructed_volume, corrupted_volume, mask, blend_factor=0.5):
    """
    Args:
        reconstructed_volume (np.ndarray): (D, H, W) predicted volume
        corrupted_volume (np.ndarray): (D, H, W) original corrupted volume
        mask (np.ndarray): (D,) binary mask (1 = missing slice, 0 = valid slice)
        blend_factor (float): weight toward target brightness (0 = no adjustment, 1 = full match)
    
    Returns:
        np.ndarray: Brightness-corrected reconstructed volume
    """
    D, H, W = reconstructed_volume.shape
    corrected_volume = reconstructed_volume.astype(np.float32).copy()

    for idx in range(D):
        if mask[idx] == 1:  # Only correct reconstructed slices
            # Find nearest valid slices
            neighbors = []
            for offset in range(1, D):
                before_idx = idx - offset
                after_idx = idx + offset

                if before_idx >= 0 and mask[before_idx] == 0:
                    neighbors.append(corrupted_volume[before_idx])
                if after_idx < D and mask[after_idx] == 0:
                    neighbors.append(corrupted_volume[after_idx])

                if len(neighbors) >= 2:
                    break

            if len(neighbors) == 0:
                continue

            neighbor_mean = np.mean([np.mean(n) for n in neighbors])
            recon_mean = np.mean(corrected_volume[idx])

            if recon_mean > 1e-6:
                scale_factor = neighbor_mean / recon_mean

                # Instead of applying full scaling, smoothly blend
                corrected_volume[idx] = corrected_volume[idx] * (1 - blend_factor) + corrected_volume[idx] * scale_factor * blend_factor

    return np.clip(corrected_volume, 0, 65535).astype(np.uint16)
