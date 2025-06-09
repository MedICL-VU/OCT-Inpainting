import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW, lr_scheduler
import tifffile as tiff
import numpy as np
import argparse
from sklearn.model_selection import KFold
import random
from skimage.metrics import structural_similarity as skimage_ssim
from scipy.ndimage import uniform_filter

from dataset import OCTAInpaintingDataset, IntensityAugment, VolumeLevelIntensityAugment
from model import UNet2p5D
from train_val import train_epoch, validate_epoch, evaluate_model_on_test, EarlyStopping, SSIM_L1_BrightnessAwareLoss
from save_inpainted import inpaint_volume_with_model, inpaint_volume_with_model_recursive
from utils import log


def compute_local_ncc(a, b, window_size):
    """Compute windowed NCC between two 2D arrays."""
    eps = 1e-8
    a = a.astype(np.float32)
    b = b.astype(np.float32)

    a_mean = uniform_filter(a, window_size)
    b_mean = uniform_filter(b, window_size)

    a2_mean = uniform_filter(a * a, window_size)
    b2_mean = uniform_filter(b * b, window_size)
    ab_mean = uniform_filter(a * b, window_size)

    a_var = a2_mean - a_mean * a_mean
    b_var = b2_mean - b_mean * b_mean
    ab_cov = ab_mean - a_mean * b_mean

    denominator = np.sqrt(a_var * b_var) + eps
    local_ncc = ab_cov / denominator

    return local_ncc  # shape: same as input, values between -1 and 1

def compute_local_3d_ncc(vol1, vol2, window_size=(7, 7, 3)):
    """Compute local 3D NCC over a volume using a sliding window."""
    from scipy.ndimage import uniform_filter

    eps = 1e-8
    vol1 = vol1.astype(np.float32)
    vol2 = vol2.astype(np.float32)

    # Means
    mean1 = uniform_filter(vol1, window_size)
    mean2 = uniform_filter(vol2, window_size)

    # Variances and Covariance
    vol1_sq = uniform_filter(vol1 ** 2, window_size)
    vol2_sq = uniform_filter(vol2 ** 2, window_size)
    vol1_vol2 = uniform_filter(vol1 * vol2, window_size)

    var1 = vol1_sq - mean1 ** 2
    var2 = vol2_sq - mean2 ** 2
    cov12 = vol1_vol2 - mean1 * mean2

    denom = np.sqrt(var1 * var2) + eps
    ncc = cov12 / denom

    return ncc

def evaluate_volume_metrics(gt_volume, pred_volume, mask):
    assert gt_volume.shape == pred_volume.shape, "Volume shapes must match"
    gt = (torch.from_numpy(gt_volume).float() / 65535.0).numpy()
    pred = (torch.from_numpy(pred_volume).float() / 65535.0).numpy()
    mask = mask.astype(bool)

    D, H, W = gt.shape
    l1_vals = []
    ssim_vals = {7: [], 11: [], 17: [], 23: [], 31: []}
    global_ncc_vals = []
    windowed_ncc_vals = {7: [], 11: [], 17: [], 23: [], 31: []}
    mie_vals = []

    for i in range(D):
        if not mask[i]:
            continue
        g = gt[i]
        p = pred[i]

        # L1 and Mean Intensity
        l1_vals.append(np.mean(np.abs(p - g)))
        mie_vals.append(np.abs(p.mean() - g.mean()))

        # SSIM at multiple windows
        for w in ssim_vals.keys():
            try:
                ssim_val = skimage_ssim(g, p, data_range=1.0, win_size=w)
            except Exception:
                ssim_val = float('nan')
            ssim_vals[w].append(ssim_val)

        # Global NCC (slice-wide Pearson)
        try:
            g_zero = g - g.mean()
            p_zero = p - p.mean()
            denom = np.sqrt(np.sum(g_zero**2) * np.sum(p_zero**2))
            if denom < 1e-8:
                ncc_val = 1.0 if np.allclose(g, p, atol=1e-6) else 0.0
            else:
                ncc_val = float(np.sum(g_zero * p_zero)) / denom
        except Exception:
            ncc_val = float('nan')
        global_ncc_vals.append(ncc_val)

        # Windowed NCC
        for w in windowed_ncc_vals.keys():
            try:
                local_ncc = compute_local_ncc(g, p, window_size=w)
                valid_vals = local_ncc[~np.isnan(local_ncc)].flatten()
                if len(valid_vals) == 0:
                    windowed_ncc_vals[w].append(float('nan'))
                else:
                    windowed_ncc_vals[w].append(np.median(valid_vals))
            except Exception:
                windowed_ncc_vals[w].append(float('nan'))

    # --- 3D NCC ---
    try:
        window_size_3d = (7, 7, 3)  # You can adjust this
        ncc3d_map = compute_local_3d_ncc(gt, pred, window_size=window_size_3d)
        # Only evaluate over corrupted slices
        valid_vals = ncc3d_map[mask]
        valid_vals = valid_vals[~np.isnan(valid_vals)]
        if len(valid_vals) > 0:
            ncc3d_mean = float(np.mean(valid_vals))
        else:
            ncc3d_mean = None
    except Exception:
        ncc3d_mean = None

    if len(l1_vals) == 0:
        return {
            "L1": None,
            "MeanIntensityError": None,
            "SSIM": {w: None for w in ssim_vals},
            "Global_NCC": None,
            "Windowed_NCC": {w: None for w in windowed_ncc_vals},
            "Local3D_NCC": None,
            "Note": "No corrupted slices to evaluate"
        }

    return {
        "L1": round(float(np.mean(l1_vals)), 4),
        "MeanIntensityError": round(float(np.mean(mie_vals)), 4),
        "SSIM": {w: round(np.nanmean(ssim_vals[w]), 4) for w in ssim_vals},
        "Global_NCC": round(np.nanmean(global_ncc_vals), 4),
        "Windowed_NCC": {w: round(np.nanmean(windowed_ncc_vals[w]), 4) for w in windowed_ncc_vals},
        "Local3D_NCC": round(ncc3d_mean, 4) if ncc3d_mean is not None else None,
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
        trainval_triplets = [triplets[i] for i in trainval_idx]

        # Reproducible shuffle for each fold
        rng = random.Random(seed + fold_idx)
        rng.shuffle(trainval_triplets)

        val_split = max(1, int(0.2 * len(trainval_triplets)))
        val_triplets = trainval_triplets[:val_split]
        train_triplets = trainval_triplets[val_split:]
        test_triplets = [triplets[i] for i in test_idx]

        folds.append((train_triplets, val_triplets, test_triplets))

    return folds


def parse_args():
    parser = argparse.ArgumentParser(description="Run 2.5D Inpainting Pipeline")
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--stack_size', type=int, default=9, help='Number of slices to stack for 2.5D input')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate for AdamW optimizer')
    parser.add_argument('--features', type=int, nargs='+', default=[64, 128, 256, 512], help='Feature channels for UNet layers')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate (0 to disable)')
    parser.add_argument('--augment', action='store_true', help='Apply data augmentation during training')
    parser.add_argument('--volume_augment', action='store_true', help='Apply volume-level intensity augmentation')
    parser.add_argument('--static_corruptions', action='store_true', help='Use static offline corruptions for training (default: online corruptions)')
    parser.add_argument('--disable_dynamic_filter', action='store_true', help='Disable dynamic filter scaling (default: enabled)')
    parser.add_argument('--stride', type=int, default=2, help='Stride for dynamic slicing (default: 1)')
    parser.add_argument('--include_artifacts', action='store_true', help='Include artifacts from training/validation (default: exclude)')
    parser.add_argument('--kfold', action='store_true', help='Run full k-fold cross-validation')
    # parser.add_argument('--fold_idx', type=int, default=0, help='If not kfold mode, which fold to run (default: 0)')
    parser.add_argument('--fold_idx', type=int, default=1, help='If not kfold mode, which fold to run (default: 0)')
    # parser.add_argument('--fold_idx', type=int, default=5, help='If not kfold mode, which fold to run (default: 0)')
    parser.add_argument('--skip_train', action='store_true', help='Skip training and only run inference on the test set')
    parser.add_argument('--debug', action='store_true', help='Enable verbose debugging logs')
    parser.add_argument('--num_runs', type=int, default=2, help='Number of times to repeat training for averaging metrics')
    parser.add_argument('--recursive_inpaint', action='store_true', help='Use recursive inpainting method instead of single pass')
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    from collections import defaultdict
    all_metrics = defaultdict(list)

    log("Starting Inpainting Pipeline")
    log(f"Device: {device}")

    # Print all arguments
    log("Pipeline arguments:")
    for arg, value in vars(args).items():
        log(f"  {arg}: {value}")

    # === 1. Load Dataset ===
    log("Loading datasets...")
    # Load and split volumes
    # volume_triplets = load_volume_triplets("/media/admin/Expansion/Mosaic_Data_for_Ipeks_Group/OCT_Inpainting_Testing/")
    # volume_triplets = load_volume_triplets("/media/admin/Expansion/Mosaic_Data_for_Ipeks_Group/OCT_Inpainting_Testing_v2/")
    volume_triplets = load_volume_triplets("/media/admin/Expansion/Mosaic_Data_for_Ipeks_Group/OCT_Inpainting_Testing_v3/")
    # volume_triplets = load_volume_triplets("/media/admin/Expansion/Mosaic_Data_for_Ipeks_Group/OCT_Inpainting_Testing_v3_ExtraVols/")
    # volume_triplets = load_volume_triplets("/media/admin/Expansion/Mosaic_Data_for_Ipeks_Group/OCT_Inpainting_Testing_v3_FewerVols/")
    # volume_triplets = load_volume_triplets("/media/admin/Expansion/Mosaic_Data_for_Ipeks_Group/OCT_Inpainting_Testing_v3_GaussianBlur1px/")
    # volume_triplets = load_volume_triplets("/media/admin/Expansion/Mosaic_Data_for_Ipeks_Group/OCT_Inpainting_Testing_v3_MedianFilter1px/")
    # volume_triplets = load_volume_triplets("/media/admin/Expansion/Mosaic_Data_for_Ipeks_Group/OCT_Inpainting_Testing_v3_MedianFilter2px/")

    # folds = get_kfold_splits(volume_triplets, k=5)
    folds = get_kfold_splits(volume_triplets, k=7)
    # folds = get_kfold_splits(volume_triplets, k=11)
    if args.kfold:
        fold_range = range(len(folds))
    else:
        fold_range = [args.fold_idx]


    for run_idx in range(args.num_runs):
        log(f"\n===== Run {run_idx + 1} of {args.num_runs} =====")
        
        for fold_idx in fold_range:        
            train_vols, val_vols, test_vols = folds[fold_idx]
            log(f"\n========== Fold {fold_idx + 1} ==========")
            log("Training volumes:")
            for v in train_vols: log(f" - {os.path.basename(v[0])}")
            log("Validation volume:")
            for v in val_vols: log(f" - {os.path.basename(v[0])}")
            log("Test volume:")
            for v in test_vols: log(f" - {os.path.basename(v[0])}")

            best_model_path = f"output/best_model_fold{fold_idx + 1}.pth"

            log(f"Using {len(train_vols)} volumes for training, {len(val_vols)} for validation, {len(test_vols)} for testing")

            # Build datasets
            augment = IntensityAugment(scale_range=(0.95, 1.05), noise_std=0.005, bias_range=(-0.02, 0.02)) if args.augment else None
            volume_augment = VolumeLevelIntensityAugment(scale_range=(0.95, 1.05), bias_range=(-0.02, 0.02)) if args.volume_augment else None

            # Create dynamic training dataset
            train_dataset = OCTAInpaintingDataset(
                train_vols,
                stack_size=args.stack_size,
                transform=augment,
                volume_transform=volume_augment,
                static_corruptions=args.static_corruptions,
                stride=args.stride,
                include_artifacts=args.include_artifacts,
                debug=args.debug
            )

            val_dataset = OCTAInpaintingDataset(
                val_vols,
                stack_size=args.stack_size,
                transform=None,
                volume_transform=None,
                static_corruptions=True,
                # static_corruptions=False,
                debug=args.debug
            )
            test_dataset = OCTAInpaintingDataset(
                test_vols,
                stack_size=args.stack_size,
                transform=None,
                volume_transform=None,
                static_corruptions=True
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
                dropout_rate=args.dropout,
                disable_dynamic_filter=args.disable_dynamic_filter
            ).to(device)

            # criterion = SSIM_L1_BrightnessAwareLoss(alpha=1.0, beta=0.0, gamma=0.0)
            # criterion = SSIM_L1_BrightnessAwareLoss(alpha=1.0, beta=0.1, gamma=0.1)
            criterion = SSIM_L1_BrightnessAwareLoss(alpha=0.8, beta=0.1, gamma=0.1)

            optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=4, factor=0.5, verbose=True)
            # early_stopping = EarlyStopping(patience=10, min_delta=1e-4, verbose=True)
            early_stopping = EarlyStopping(patience=10, min_delta=5e-5, verbose=True)

            # If skip training, load the best model directly
            if not args.skip_train:
                # === 3. Train Model ===
                log("Starting training...")
                best_val_loss = float('inf')

                for epoch in range(1, args.epochs + 1):
                    train_loss, diagnostics = train_epoch(model, train_loader, optimizer, criterion, device, debug=args.debug, disable_dynamic_filter=args.disable_dynamic_filter)
                    print(f"Train Loss: {train_loss:.4f} | Terms: {diagnostics}")
                    
                    val_loss = validate_epoch(model, val_loader, criterion, device, disable_dynamic_filter=args.disable_dynamic_filter)

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
            test_loss = evaluate_model_on_test(model, test_loader, criterion, device, disable_dynamic_filter=args.disable_dynamic_filter)
            log(f"Final test loss: {test_loss:.4f}")
            
            # === 5. Inpaint Test Volume with Trained Model ===
            log("Inpainting volume...")
            test_corrupted_path, test_gt_path, test_mask_path = test_vols[0]
            corrupted_volume = tiff.imread(test_corrupted_path)
            mask = tiff.imread(test_mask_path)
            # Convert mask to 1D binary array if needed
            if mask.ndim == 3:
                if np.all((mask == 0) | (mask == 1)):
                    mask = mask[:, 0, 0]
                else:
                    raise ValueError("Unexpected mask format: expected binary 0/1 per slice")
            elif mask.ndim != 1:
                raise ValueError("Unsupported mask dimensionality")

            if args.recursive_inpaint:
                log("Using recursive inpainting method...")
                inpainted = inpaint_volume_with_model_recursive(model, corrupted_volume, mask, device, args.stack_size, args.debug, args.disable_dynamic_filter)
            else:
                log("Using single-pass inpainting method...")
                inpainted = inpaint_volume_with_model(model, corrupted_volume, mask, device, args.stack_size, args.debug, args.disable_dynamic_filter)

            if isinstance(inpainted, tuple):
                inpainted_volume = inpainted[0]
            else:
                inpainted_volume = inpainted

            # Generate output filename based on test volume name
            base_name = os.path.basename(test_corrupted_path).replace("_corrupted.tif", "")
            predicted_output_path = os.path.join(
                # "/media/admin/Expansion/Mosaic_Data_for_Ipeks_Group/OCT_Inpainting_Testing",
                # "/media/admin/Expansion/Mosaic_Data_for_Ipeks_Group/OCT_Inpainting_Testing_v2",
                "/media/admin/Expansion/Mosaic_Data_for_Ipeks_Group/OCT_Inpainting_Testing_v3",
                # "/media/admin/Expansion/Mosaic_Data_for_Ipeks_Group/OCT_Inpainting_Testing_v3_ExtraVols",
                # "/media/admin/Expansion/Mosaic_Data_for_Ipeks_Group/OCT_Inpainting_Testing_v3_FewerVols",
                # "/media/admin/Expansion/Mosaic_Data_for_Ipeks_Group/OCT_Inpainting_Testing_v3_GaussianBlur1px",
                # "/media/admin/Expansion/Mosaic_Data_for_Ipeks_Group/OCT_Inpainting_Testing_v3_MedianFilter1px",
                # "/media/admin/Expansion/Mosaic_Data_for_Ipeks_Group/OCT_Inpainting_Testing_v3_MedianFilter2px",
                # f"{base_name}_inpainted_2p5DUNet_fold{fold_idx+1}_0531_dynamic_filter_scaling.tif"
                f"{base_name}_MASTER_BASELINE_0608_0.8-0.1-0.1_noDynamicFilter_staticCorruptions.tif"
            )

            tiff.imwrite(predicted_output_path, inpainted_volume.astype(np.uint16))
            log(f"Inpainted volume saved to: {predicted_output_path}")

            # === 5. Volume-Wise Evaluation ===
            log("Evaluating inpainted volume metrics...")

            gt_volume = tiff.imread(test_gt_path)
            gt_volume = tiff.imread('/media/admin/Expansion/Mosaic_Data_for_Ipeks_Group/OCT_Inpainting_Testing_v3/1.2_OCTA_Vol2_Processed_Cropped_gt.tif')
            metrics = evaluate_volume_metrics(gt_volume, inpainted_volume, mask)

            for key, val in metrics.items():
                if isinstance(val, dict):  # SSIM, NCC, etc.
                    for subkey, subval in val.items():
                        all_metrics[f"{key}_{subkey}"].append(subval)
                else:
                    all_metrics[key].append(val)

            log(f"Volume Metrics for {base_name}:")
            log(f" - L1 Loss: {metrics['L1']:.4f}")
            log(f" - Mean Intensity Diff: {metrics['MeanIntensityError']}")
            for w in [7, 11, 17, 23, 31]:
                log(f" - SSIM (win={w}): {metrics['SSIM'][w]}")
            log(f" - Global_NCC: {metrics['Global_NCC']}")
            for w in [7, 11, 17, 23, 31]:
                log(f" - Windowed NCC (win={w}): {metrics['Windowed_NCC'][w]}")
            log(f" - Local 3D NCC (win={7}): {metrics['Local3D_NCC']}")

    if args.num_runs > 1:
        log("\n===== Averaged Metrics Across Runs =====")
        for key, values in all_metrics.items():
            if all(v is not None for v in values):
                avg_val = round(np.nanmean(values), 4)
                log(f" - {key}: {avg_val}")
            else:
                log(f" - {key}: insufficient data to average")


if __name__ == "__main__":
    main()
