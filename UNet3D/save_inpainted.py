import tifffile as tiff
import torch
import torch.nn.functional as F
import numpy as np

def inpaint_volume_with_model(model, corrupted_volume, mask, device, stack_size=5):
    """
    Apply 3D model to missing slices in a volume.
    
    Args:
        model: trained UNet3D model
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
            stack_np = padded[idx:idx + stack_size]  # (stack_size, H, W)
            original_h, original_w = stack_np.shape[-2:]

            # Convert to 5D tensor
            stack = torch.from_numpy(stack_np.copy()).float().unsqueeze(0).unsqueeze(0).to(device) / 65535.0  # (1, 1, D, H, W)

            # Compute padding
            pad_h = (16 - original_h % 16) % 16
            pad_w = (16 - original_w % 16) % 16

            if pad_h > 0 or pad_w > 0:
                # F.pad uses (W_left, W_right, H_top, H_bottom, D_front, D_back)
                stack = F.pad(stack, (0, pad_w, 0, pad_h, 0, 0), mode='replicate')

            output = model(stack)  # shape: (1, 1, D, H_padded, W_padded)

            # Remove padding if added
            output = output[:, :, :, :original_h, :original_w]  # (1, 1, D, H, W)
            pred_stack = output.squeeze().cpu().numpy() * 65535.0  # (D, H, W)

            center_idx = pred_stack.shape[0] // 2
            pred_center = pred_stack[center_idx]
            inpainted[idx] = np.clip(pred_center, 0, 65535).astype(np.uint16)

    return inpainted
