import torch
import numpy as np


def inpaint_volume_with_model(model, corrupted_volume, mask, device, stack_size=9, debug=False):
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
            stack = padded[idx: idx + stack_size]  # (stack_size, H, W)
            stack_tensor = torch.from_numpy(stack).unsqueeze(0).float().to(device) / corrupted_volume.max()

            valid_mask = (stack_tensor.squeeze(0).sum(dim=(1, 2)) > 1e-3).float()  # shape: (stack_size,)
            valid_mask = valid_mask.unsqueeze(0).to(device)  # shape: (1, stack_size)
            output = model(stack_tensor, valid_mask)
            
            # output = model(stack_tensor)
            
            pred = output.squeeze().cpu().numpy() * corrupted_volume.max()
            inpainted[idx] = np.clip(pred, 0, corrupted_volume.max()).astype(np.uint16)

            if debug and idx % 10 == 0:
                print(f"\n[DEBUG] Inpainting index: {idx}")
                print(f"[DEBUG] Stack shape: {stack.shape}, Stack dtype: {stack.dtype}")
                print(f"[DEBUG] Stack max before norm: {stack.max()}, min: {stack.min()}")
                print(f"[DEBUG] Stack tensor shape: {stack_tensor.shape}, dtype: {stack_tensor.dtype}")
                print(f"[DEBUG] Prediction max: {pred.max()}, min: {pred.min()}")
                print(f"[DEBUG] Inpainted slice final dtype: {inpainted[idx].dtype}, max: {inpainted[idx].max()}")

    return inpainted.astype(np.uint16)


def inpaint_volume_with_model_recursive(model, corrupted_volume, mask, device, stack_size=9):
    """
    Apply 2.5D model to missing slices in a volume.
    Uses recursive updates and inpaints corrupted blocks symmetrically from outside-in.
    """
    model.eval()
    pad = stack_size // 2
    D, H, W = corrupted_volume.shape
    padded = np.pad(corrupted_volume, ((pad, pad), (0, 0), (0, 0)), mode='edge')
    inpainted = corrupted_volume.copy()

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
                    stack = padded[idx: idx + stack_size]  # (stack_size, H, W)
                    stack_tensor = torch.from_numpy(stack).unsqueeze(0).float().to(device) / corrupted_volume.max()

                    valid_mask = (stack_tensor.squeeze(0).sum(dim=(1, 2)) > 1e-3).float()  # shape: (stack_size,)
                    valid_mask = valid_mask.unsqueeze(0).to(device)  # shape: (1, stack_size)
                    output = model(stack_tensor, valid_mask)
                    
                    # output = model(stack_tensor)
                    pred = output.squeeze().cpu().numpy() * corrupted_volume.max()
                    inpainted[idx] = np.clip(pred, 0, corrupted_volume.max()).astype(np.uint16)
                    padded[idx + pad] = pred  # Update padding for recursive effect
                left += 1
                right -= 1

            # volume_max = corrupted_volume.max() + 1e-5
            # padded = padded.astype(np.float32) / volume_max  # normalize padded

            # for idx in block:
            #     stack = padded[idx: idx + stack_size]
            #     stack_tensor = torch.from_numpy(stack).unsqueeze(0).float().to(device)
            #     output = model(stack_tensor)
            #     pred = output.squeeze().cpu().numpy()
            #     inpainted[idx] = np.clip(pred * volume_max, 0, 65535).astype(np.uint16)
            #     padded[idx + pad] = pred  # keep in normalized range

    return inpainted
