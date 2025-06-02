import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW, lr_scheduler
import tifffile as tiff
import numpy as np
import pandas as pd
import argparse
from sklearn.model_selection import KFold
import random
from skimage.metrics import structural_similarity as skimage_ssim

from dataset import OCTAInpaintingDataset, IntensityAugment, VolumeLevelIntensityAugment
from model import UNet2p5D
from train_val import train_epoch, validate_epoch, evaluate_model_on_test, EarlyStopping, SSIM_L1_BrightnessAwareLoss
from save_inpainted import inpaint_volume_with_model, inpaint_volume_with_model_recursive
from utils import log


def evaluate_volume_metrics(gt, pred, mask):
    assert gt.shape == pred.shape, "Volume shapes must match"
    gt = (torch.from_numpy(gt).float() / 65535.0).numpy()
    pred = (torch.from_numpy(pred).float() / 65535.0).numpy()
    mask = mask.astype(bool)

    D, H, W = gt.shape
    l1_vals = []
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
            "MeanIntensityError": round(float(np.abs(pred[mask].mean() - gt[mask].mean())), 4),
            "SSIM": {w: None for w in [7, 11, 17]},
            "NCC": {w: None for w in [7, 11, 17]},
            "Note": "No corrupted slices to evaluate"
        }

    return {
        "L1": round(sum(l1_vals) / num_masked, 4),
        "MeanIntensityError": round(float(np.abs(pred[mask].mean() - gt[mask].mean())), 4),
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
    parser.add_argument('--dynamic', action='store_true', help='Use dynamic slicing for training')
    parser.add_argument('--dynamic_filter', action='store_true', default=True, help='Use dynamic filter scaling')
    parser.add_argument('--stride', type=int, default=4, help='Stride for dynamic slicing (default: 1)')
    parser.add_argument('--cuda', action='store_true', help='Use CUDA if available')
    parser.add_argument('--kfold', action='store_true', help='Run full k-fold cross-validation')
    parser.add_argument('--fold_idx', type=int, default=1, help='If not kfold mode, which fold to run (default: 0)')
    parser.add_argument('--skip_train', action='store_true', help='Skip training and only run inference on the test set')
    parser.add_argument('--debug', action='store_true', help='Enable verbose debugging logs')
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')

    log("Starting Inpainting Pipeline")
    log(f"Device: {device}")

    log("Starting Experimentation Loop...")
    config_df = pd.read_csv("Experiment_Configurations.csv")

    bool_cols = ["relative_brightness", "dropout", "augment"]
    for col in bool_cols:
        config_df[col] = config_df[col].astype(str).str.upper().map({"TRUE": True, "FALSE": False})

    METRIC_RESULTS_PATH = "/media/admin/Expansion/Mosaic_Data_for_Ipeks_Group/OCT_Inpainting_Testing_Ablationv2/experiment_metrics_summary.csv"
    all_metrics = []

    for exp_idx, config in config_df.iterrows():
        log(f"\n=== Running Experiment {exp_idx+1}/{len(config_df)} ===")
        log(config.to_string())

        import ast
        neighbor_drop_range = ast.literal_eval(config['neighbor_drop_range'])  # This converts "(0, 3)" to (0, 3)

        # === 1. Load Dataset ===
        log("Loading datasets...")
        # Load and split volumes
        # volume_triplets = load_volume_triplets("/media/admin/Expansion/Mosaic_Data_for_Ipeks_Group/OCT_Inpainting_Testing/")
        volume_triplets = load_volume_triplets("/media/admin/Expansion/Mosaic_Data_for_Ipeks_Group/OCT_Inpainting_Testing_Ablationv2/")

        folds = get_kfold_splits(volume_triplets, k=7)
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
            
            # === Build datasets and loaders ===
            if config['augment'] == 'ScaleRange':
                augment = IntensityAugment(
                    scale_range=(0.95, 1.05),
                    noise_std=0.0,
                    bias_range=(-0.0, 0.0)
                )
            elif config['augment'] == 'NoiseSTD':
                augment = IntensityAugment(
                    scale_range=(1.0, 1.0),
                    noise_std=0.01,
                    bias_range=(-0.0, 0.0)
                )
            elif config['augment'] == 'BiasRange':
                augment = IntensityAugment(
                    scale_range=(1.0, 1.0),
                    noise_std=0.0,
                    bias_range=(-0.02, 0.02)
                )
            else:
                augment = None

            # volume_augment = VolumeLevelIntensityAugment(
            #     scale_range=(0.9, 1.1),
            #     bias_range=(-0.05, 0.05)
            # ) if config['augment'] == 'Large' else None

            train_dataset = OCTAInpaintingDataset(
                train_vols,
                stack_size=int(config['stack_size']),
                transform=augment,
                volume_transform=None,
                dynamic=True,
                stride=int(config['stride']),
                neighbor_drop_range=neighbor_drop_range
            )

            val_dataset = OCTAInpaintingDataset(
                val_vols,
                stack_size=int(config['stack_size']),
                transform=None,
                volume_transform=None,
                dynamic=False,
                # dynamic=True,
                debug=args.debug
            )
            test_dataset = OCTAInpaintingDataset(
                test_vols,
                stack_size=int(config['stack_size']),
                transform=None,
                volume_transform=None,
                dynamic=False
            )

            train_loader = DataLoader(train_dataset, batch_size=int(config['batch_size']), shuffle=True, num_workers=0)
            val_loader = DataLoader(val_dataset, batch_size=int(config['batch_size']), shuffle=False, num_workers=0)
            test_loader = DataLoader(test_dataset, batch_size=int(config['batch_size']), shuffle=False, num_workers=0)

            # === 2. Initialize Model ===
            log("Initializing model...")
            model = UNet2p5D(
                in_channels=int(config['stack_size']),
                out_channels=1,
                features=args.features,
                dropout_rate=0.1 if config['dropout'] else 0.0,
                dynamic_filter=True
            ).to(device)

            criterion = SSIM_L1_BrightnessAwareLoss(
                alpha=float(config['alpha']),
                beta=float(config['beta']),
                gamma=float(config['gamma']),
            )

            optimizer = AdamW(model.parameters(), lr=float(config['lr']))
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=4, factor=0.5, verbose=True)
            best_model_path = f"output/best_model_exp{exp_idx}_fold{fold_idx+1}.pth"

            # === Train ===
            early_stopping = EarlyStopping(patience=10, min_delta=5e-5)
            best_val_loss = float('inf')

            for epoch in range(200):
                train_loss, diagnostics = train_epoch(model, train_loader, optimizer, criterion, device, debug=args.debug, 
                                                      dynamic_filter=args.dynamic_filter, relativeDiff=config['relative_brightness'])
                print(f"Train Loss: {train_loss:.4f} | Terms: {diagnostics}")
                
                val_loss = validate_epoch(model, val_loader, criterion, device, dynamic_filter=args.dynamic_filter, relativeDiff=config['relative_brightness'])

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
            test_corrupted_path, test_gt_path, test_mask_path = test_vols[0]

            test_loss = evaluate_model_on_test(model, test_loader, criterion, device, dynamic_filter=args.dynamic_filter, relativeDiff=config['relative_brightness'])
            log(f"Final test loss: {test_loss:.4f}")
            
            # === 5. Inpaint Test Volume with Trained Model ===
            log("Inpainting volume...")
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

            inpainted = inpaint_volume_with_model(model, corrupted_volume, mask, device, stack_size=int(config['stack_size']))
            inpainted_recursive = inpaint_volume_with_model_recursive(model, corrupted_volume, mask, device, stack_size=int(config['stack_size']))

            if isinstance(inpainted, tuple):
                inpainted_volume = inpainted[0]
            else:
                inpainted_volume = inpainted

            if isinstance(inpainted_recursive, tuple):
                inpainted_volume_recursive = inpainted_recursive[0]
            else:
                inpainted_volume_recursive = inpainted_recursive

            # Generate output filename based on test volume name
            base_name = os.path.basename(test_corrupted_path).replace("_corrupted.tif", "")
            filename = (
                f"{base_name}_inpainted_exp{exp_idx}_fold{fold_idx+1}_"
                f"batch{config['batch_size']}_stack{config['stack_size']}_"
                f"lr{config['lr']}_stride{config['stride']}_drop{config['neighbor_drop_range']}_"
                f"relbright{config['relative_brightness']}_dropout{config['dropout']}_"
                f"aug{config['augment']}_alpha{config['alpha']}_beta{config['beta']}_gamma{config['gamma']}.tif"
            )
            filename_recursive = (
                f"{base_name}_inpainted_exp{exp_idx}_fold{fold_idx+1}_"
                f"batch{config['batch_size']}_stack{config['stack_size']}_"
                f"lr{config['lr']}_stride{config['stride']}_drop{config['neighbor_drop_range']}_"
                f"relbright{config['relative_brightness']}_dropout{config['dropout']}_"
                f"aug{config['augment']}_alpha{config['alpha']}_beta{config['beta']}_gamma{config['gamma']}_recurInpaint.tif"
            )

            predicted_output_path = os.path.join(
                "/media/admin/Expansion/Mosaic_Data_for_Ipeks_Group/OCT_Inpainting_Testing_Ablationv2",
                filename
            )
            predicted_output_path_recursive = os.path.join(
                "/media/admin/Expansion/Mosaic_Data_for_Ipeks_Group/OCT_Inpainting_Testing_Ablationv2",
                filename_recursive
            )

            tiff.imwrite(predicted_output_path, inpainted_volume.astype(np.uint16))
            log(f"Inpainted volume saved to: {predicted_output_path}")

            tiff.imwrite(predicted_output_path_recursive, inpainted_volume_recursive.astype(np.uint16))
            log(f"Inpainted volume saved to: {predicted_output_path_recursive}")

            # === 5. Volume-Wise Evaluation ===
            log("Evaluating inpainted volume metrics...")

            gt_volume = tiff.imread(test_gt_path)
            metrics = evaluate_volume_metrics(gt_volume, inpainted_volume, mask)

            log(f"Volume Metrics for {filename}:")
            log(f" - L1 Loss: {metrics['L1']:.4f}")
            log(f" - Mean Intensity Diff: {metrics['MeanIntensityError']}")

            for w in [7, 11, 17]:
                log(f" - SSIM (win={w}): {metrics['SSIM'][w]}")
            for w in [7, 11, 17]:
                log(f" - NCC (win={w}):  {metrics['NCC'][w]}")

            metrics_record = {
                "exp_idx": exp_idx,
                "fold": fold_idx + 1,
                "volume": filename,
                "L1": metrics["L1"],
                "MeanIntensityError": metrics["MeanIntensityError"]
            }

            # Add SSIM and NCC for each window
            for w in [7, 11, 17]:
                metrics_record[f"SSIM_{w}"] = metrics["SSIM"][w]
                metrics_record[f"NCC_{w}"] = metrics["NCC"][w]

            # Also add hyperparameters from config
            for col in config.index:
                metrics_record[col] = config[col]

            all_metrics.append(metrics_record)



            metrics_recursive = evaluate_volume_metrics(gt_volume, inpainted_volume_recursive, mask)

            log(f"Volume Metrics for {filename}:")
            log(f" - L1 Loss: {metrics_recursive['L1']:.4f}")
            log(f" - Mean Intensity Diff: {metrics_recursive['MeanIntensityError']}")

            for w in [7, 11, 17]:
                log(f" - SSIM (win={w}): {metrics_recursive['SSIM'][w]}")
            for w in [7, 11, 17]:
                log(f" - NCC (win={w}):  {metrics_recursive['NCC'][w]}")

            metrics_record_recursive = {
                "exp_idx": exp_idx,
                "fold": fold_idx + 1,
                "volume": filename,
                "L1": metrics_recursive["L1"],
                "MeanIntensityError": metrics_recursive["MeanIntensityError"]
            }

            # Add SSIM and NCC for each window
            for w in [7, 11, 17]:
                metrics_record_recursive[f"SSIM_{w}"] = metrics_recursive["SSIM"][w]
                metrics_record_recursive[f"NCC_{w}"] = metrics_recursive["NCC"][w]

            # Also add hyperparameters from config
            for col in config.index:
                metrics_record_recursive[col] = config[col]

            all_metrics.append(metrics_record_recursive)


    df = pd.DataFrame(all_metrics)
    df.to_csv(METRIC_RESULTS_PATH, index=False)
    log(f"Saved all experiment metrics to {METRIC_RESULTS_PATH}")


if __name__ == "__main__":
    main()
