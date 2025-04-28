import tifffile as tiff
import numpy as np

def inpaint_volume_with_model(model, corrupted_volume, mask, device, stack_size=5):
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
