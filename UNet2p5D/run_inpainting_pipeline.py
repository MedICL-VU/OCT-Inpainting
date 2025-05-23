import os
import torch
import piq
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW, lr_scheduler
import tifffile as tiff
import numpy as np
import argparse
from sklearn.model_selection import KFold
from skimage.metrics import structural_similarity as skimage_ssim

from dataset import OCTAInpaintingDataset, IntensityAugment, VolumeLevelIntensityAugment
from model import UNet2p5D
from train_val import train_epoch, validate_epoch, evaluate_model_on_test, EarlyStopping, SSIM_L1_GlobalLoss
from save_inpainted import inpaint_volume_with_model
from utils import log


def evaluate_volume_metrics(gt, pred, mask):
    from skimage.metrics import structural_similarity as skimage_ssim

    assert gt.shape == pred.shape, "Volume shapes must match"
    gt = (torch.from_numpy(gt).float() / 65535.0).numpy()
    pred = (torch.from_numpy(pred).float() / 65535.0).numpy()
    mask = mask.astype(bool)

    D, H, W = gt.shape
    l1_vals = []
    psnr_vals = []
    ssim_vals = {7: [], 11: [], 17: []}
    ncc_vals = {7: [], 11: [], 17: []}

    def local_ncc(a, b, window):
        a, b = torch.from_numpy(a), torch.from_numpy(b)
        pad = window // 2
        a2 = a**2
        b2 = b**2
        ab = a * b
        kernel = torch.ones((1, 1, window, window))
        a_sum = F.conv2d(a.view(1,1,H,W), kernel, padding=pad)
        b_sum = F.conv2d(b.view(1,1,H,W), kernel, padding=pad)
        ab_sum = F.conv2d(ab.view(1,1,H,W), kernel, padding=pad)
        a2_sum = F.conv2d(a2.view(1,1,H,W), kernel, padding=pad)
        b2_sum = F.conv2d(b2.view(1,1,H,W), kernel, padding=pad)

        win_size = window ** 2
        a_mean = a_sum / win_size
        b_mean = b_sum / win_size
        num = ab_sum - a_mean * b_sum - b_mean * a_sum + a_mean * b_mean * win_size
        denom = (a2_sum - a_mean * a_sum - a_mean * a_sum + a_mean * a_mean * win_size).clamp(min=1e-6).sqrt() * \
                (b2_sum - b_mean * b_sum - b_mean * b_sum + b_mean * b_mean * win_size).clamp(min=1e-6).sqrt()
        return (num / denom).clamp(-1, 1).mean().item()

    num_masked = 0

    for i in range(D):
        if not mask[i]:
            continue
        g = gt[i]
        p = pred[i]

        l1_vals.append(np.mean(np.abs(p - g)))
        psnr = piq.psnr(torch.tensor(p).unsqueeze(0).unsqueeze(0),
                        torch.tensor(g).unsqueeze(0).unsqueeze(0),
                        data_range=1.0).item()
        psnr_vals.append(psnr)

        for w in [7, 11, 17]:
            try:
                ssim = skimage_ssim(p, g, data_range=1.0, win_size=w)
                ssim_vals[w].append(ssim)
            except Exception as e:
                log(f"SSIM failed on slice {i} with window {w}: {e}")
                ssim_vals[w].append(float('nan'))

            ncc_vals[w].append(local_ncc(p, g, window=w))

        num_masked += 1

    if num_masked == 0:
        return {
            "L1": None,
            "PSNR": None,
            "MeanIntensityError": round(float(np.abs(pred.mean() - gt.mean())), 4),
            "SSIM": {w: None for w in [7, 11, 17]},
            "NCC": {w: None for w in [7, 11, 17]},
            "Note": "No corrupted slices to evaluate"
        }

    return {
        "L1": round(sum(l1_vals) / num_masked, 4),
        "PSNR": round(sum(psnr_vals) / num_masked, 4),
        "MeanIntensityError": round(float(np.abs(pred.mean() - gt.mean())), 4),
        "SSIM": {w: round(np.nanmean(ssim_vals[w]), 4) for w in ssim_vals},
        "NCC": {w: round(sum(ncc_vals[w]) / num_masked, 4) for w in ncc_vals}
    }


def load_volume_triplets(data_dir):
    """
    Returns list of (corrupted, gt, mask) triplets from directory.
    Assumes filenames follow pattern: {name}_corrupted.tif, {name}_gt.tif, {name}_mask.tif
    """
    triplets = []
    for f in os.listdir(data_dir):
        if f.endswith("_corrupted.tif"):
            name = f.replace("_corrupted.tif", "")
            corrupted = os.path.join(data_dir, f)
            gt = os.path.join(data_dir, f"{name}_gt.tif")
            mask = os.path.join(data_dir, f"{name}_mask.tif")
            if os.path.exists(gt) and os.path.exists(mask):
                triplets.append((corrupted, gt, mask))
    return sorted(triplets)


def get_kfold_splits(triplets, k=5, seed=42):
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    folds = []

    for fold_idx, (trainval_idx, test_idx) in enumerate(kf.split(triplets)):
        test = [triplets[i] for i in test_idx]
        trainval = [triplets[i] for i in trainval_idx]

        val_split = int(0.2 * len(trainval)) or 1
        val = trainval[:val_split]
        train = trainval[val_split:]

        folds.append((train, val, test))
    return folds


def parse_args():
    parser = argparse.ArgumentParser(description="Run 2.5D Inpainting Pipeline")
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--stack_size', type=int, default=9, help='Number of slices to stack for 2.5D input')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate for AdamW optimizer')
    parser.add_argument('--features', type=int, nargs='+', default=[64, 128, 256, 512], help='Feature channels for UNet layers')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate (0 to disable)')
    parser.add_argument('--augment', action='store_true', help='Apply data augmentation during training')
    parser.add_argument('--stride', type=int, default=1, help='Stride for dynamic slicing (default: 1)')
    parser.add_argument('--cuda', action='store_true', help='Use CUDA if available')
    parser.add_argument('--kfold', action='store_true', help='Run full k-fold cross-validation')
    parser.add_argument('--fold_idx', type=int, default=0, help='If not kfold mode, which fold to run (default: 0)')
    parser.add_argument('--skip_train', action='store_true', help='Skip training and only run inference on the test set')
    parser.add_argument('--debug_mode', action='store_true', help='Enable verbose debugging logs')
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')

    log("Starting Inpainting Pipeline")
    log(f"Device: {device}")
    log(f"Epochs: {args.epochs} | Batch size: {args.batch_size} | Stack size: {args.stack_size}")

    # === 1. Load Dataset ===
    log("Loading datasets...")
    # Load and split volumes
    volume_triplets = load_volume_triplets("/media/admin/Expansion/Mosaic_Data_for_Ipeks_Group/OCT_Inpainting_Testing/")

    folds = get_kfold_splits(volume_triplets, k=5)
    if args.kfold:
        fold_range = range(len(folds))
    else:
        fold_range = [args.fold_idx]


    for fold_idx in fold_range:
        train_vols, val_vols, test_vols = folds[fold_idx]
        log(f"\n========== Fold {fold_idx + 1} ==========")
        log("Training volumes:")
        for v in train_vols: log(f" - {os.path.basename(v[0])}")
        log("Validation volume:")
        for v in val_vols: log(f" - {os.path.basename(v[0])}")
        log("Test volume:")
        for v in test_vols: log(f" - {os.path.basename(v[0])}")

        test_corrupted_path, test_gt_path, test_mask_path = test_vols[0]
        best_model_path = f"output/best_model_fold{fold_idx + 1}.pth"

        log(f"Using {len(train_vols)} volumes for training, {len(val_vols)} for validation, {len(test_vols)} for testing")

        # Build datasets
        augment = IntensityAugment(scale_range=(0.95, 1.05), noise_std=0.005, bias_range=(-0.02, 0.02)) if args.augment else None
        volume_augment = VolumeLevelIntensityAugment(scale_range=(0.95, 1.05), bias_range=(-0.02, 0.02)) if args.augment else None

        # Create dynamic training dataset
        train_dataset = OCTAInpaintingDataset(
            train_vols,
            stack_size=args.stack_size,
            transform=augment,
            volume_transform=volume_augment,
            dynamic=True,
            stride=args.stride,
            debug=args.debug_mode
        )

        # Validation and test datasets remain static (pre-generated corruptions)
        val_dataset = OCTAInpaintingDataset(
            val_vols,
            stack_size=args.stack_size,
            transform=None,
            volume_transform=None,
            dynamic=False,
            # dynamic=True,
            debug=args.debug_mode
        )
        test_dataset = OCTAInpaintingDataset(
            test_vols,
            stack_size=args.stack_size,
            transform=None,
            volume_transform=None,
            dynamic=False
        )

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

        # === 2. Initialize Model ===
        log("Initializing model...")
        model = UNet2p5D(
            in_channels=args.stack_size,
            out_channels=1,
            features=args.features,
            dropout_rate=args.dropout
        ).to(device)
        # criterion = SSIM_L1_GlobalLoss(alpha=0.8, beta=0.1)
        criterion = SSIM_L1_GlobalLoss(alpha=1.0, beta=0.0)
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=4, factor=0.5, verbose=True)
        early_stopping = EarlyStopping(patience=5, min_delta=1e-4, verbose=True)

        # If skip training, load the best model directly
        if not args.skip_train:
            # === 3. Train Model ===
            log("Starting training...")
            best_val_loss = float('inf')

            for epoch in range(1, args.epochs + 1):
                train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
                val_loss = validate_epoch(model, val_loader, criterion, device)

                log(f"[Epoch {epoch}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
                scheduler.step(val_loss)

                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), best_model_path)

                # Early stopping check
                early_stopping.step(val_loss)
                if early_stopping.should_stop:
                    log(f"Early stopping triggered at epoch {epoch}")
                    break

            log("Training completed.")

        # === 4: Evaluate on Held-Out Test Volume ===
        log("Evaluating on held-out test volume...")
        model.load_state_dict(torch.load(best_model_path))
        model.eval()
        test_loss = evaluate_model_on_test(model, test_loader, criterion, device)
        log(f"Final test loss: {test_loss:.4f}")
        
        # === 5. Inpaint Test Volume with Trained Model ===
        log("Inpainting volume...")
        corrupted_volume = tiff.imread(test_corrupted_path)
        mask_volume = tiff.imread(test_mask_path)
        if mask_volume.ndim == 3:
            mask = (mask_volume[:, 0, 0] > 0).astype(np.uint8)
        else:
            mask = mask_volume.astype(np.uint8)

        inpainted = inpaint_volume_with_model(model, corrupted_volume, mask, device, stack_size=args.stack_size)
        if isinstance(inpainted, tuple):
            inpainted_volume = inpainted[0]
        else:
            inpainted_volume = inpainted

        # Generate output filename based on test volume name
        base_name = os.path.basename(test_corrupted_path).replace("_corrupted.tif", "")
        predicted_output_path = os.path.join(
            "/media/admin/Expansion/Mosaic_Data_for_Ipeks_Group/OCT_Inpainting_Testing",
            f"{base_name}_inpainted_2p5DUNet_fold{fold_idx+1}.tif"
        )

        tiff.imwrite(predicted_output_path, inpainted_volume.astype(np.uint16))
        log(f"Inpainted volume saved to: {predicted_output_path}")

        # === 5. Volume-Wise Evaluation ===
        log("Evaluating inpainted volume metrics...")

        gt_volume = tiff.imread(test_gt_path)
        metrics = evaluate_volume_metrics(gt_volume, inpainted_volume, mask)

        log(f"Volume Metrics for {base_name}:")
        log(f" - L1 Loss: {metrics['L1']:.4f}")
        log(f" - PSNR: {metrics['PSNR']}")
        log(f" - Mean Intensity Diff: {metrics['MeanIntensityError']}")

        for w in [7, 11, 17]:
            log(f" - SSIM (win={w}): {metrics['SSIM'][w]}")
        for w in [7, 11, 17]:
            log(f" - NCC (win={w}):  {metrics['NCC'][w]}")


if __name__ == "__main__":
    main()
