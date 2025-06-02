def inpaint_volume_with_model(model, corrupted_volume, mask, device, stack_size=9):
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
            stack = torch.from_numpy(stack).unsqueeze(0).float().to(device) / corrupted_volume.max()
            output = model(stack)  # (1, 1, H, W)
            pred = output.squeeze().cpu().numpy() * corrupted_volume.max()
            inpainted[idx] = np.clip(pred, 0, corrupted_volume.max()).astype(np.uint16)

    return inpainted.astype(np.uint16)