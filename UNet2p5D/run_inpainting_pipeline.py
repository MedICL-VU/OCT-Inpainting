import os
import torch
from pytorch_msssim import ssim as compute_ssim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
import tifffile as tiff
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import KFold
import argparse
import itertools

from save_inpainted import inpaint_volume_with_model, smooth_rescale_reconstructed_volume
from utils import log


def evaluate_volume_metrics(gt, pred, mask):
    """Evaluate metrics on the masked (corrupted) regions only."""
    assert gt.shape == pred.shape, "Shapes must match"
    gt = torch.from_numpy(gt).unsqueeze(0).unsqueeze(0).float() / 65535.0  # (1, 1, D, H, W)
    pred = torch.from_numpy(pred).unsqueeze(0).unsqueeze(0).float() / 65535.0
    mask = torch.from_numpy(mask).view(1, 1, -1, 1, 1).float()  # (1, 1, D, 1, 1)

    l1 = F.l1_loss(pred * mask, gt * mask).item()
    mean_diff = torch.abs(pred.mean() - gt.mean()).item()
    ssim_score = compute_ssim(pred.squeeze(0), gt.squeeze(0), data_range=1.0)

    return {
        "L1": round(l1, 4),
        "SSIM": round(ssim_score.item(), 4),
        "MeanIntensityError": round(mean_diff, 4)
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
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--stack_size', type=int, default=9)
    # parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--cuda', action='store_true', help='Use CUDA if available')
    parser.add_argument('--kfold', action='store_true', help='Run full k-fold cross-validation')
    parser.add_argument('--fold_idx', type=int, default=0, help='If not kfold mode, which fold to run (default: 0)')
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')

    log("=== SANITY CHECK: 24-Config Forward-Pass Only ===")
    volume_triplets = load_volume_triplets("/media/admin/Expansion/Mosaic_Data_for_Ipeks_Group/OCT_Inpainting_Testing/")
    folds = get_kfold_splits(volume_triplets, k=5)

    fold_range = [args.fold_idx]  # Only 1 fold to keep testing fast

    # All combinations
    loss_types = ['global', 'brightness_aware']
    augment_flags = [False, True]
    validity_loss_flags = [False, True]
    stack_sizes = [5, 9, 13]

    for loss_type, augment, use_valid_mask_loss, stack_size in itertools.product(loss_types, augment_flags, validity_loss_flags, stack_sizes):
        for fold_idx in fold_range:
            log(f"\n[CONFIG] Loss={loss_type}, Augment={augment}, MaskedLoss={use_valid_mask_loss}, StackSize={stack_size}")
            try:
                run_single_experiment(
                    fold_idx=fold_idx,
                    triplet_folds=folds,
                    device=device,
                    args=args,
                    loss_type=loss_type,
                    augment=augment,
                    use_valid_mask_loss=use_valid_mask_loss,
                    stack_size=stack_size,
                    test_only=True  # 🆕 Skip training
                )
            except Exception as e:
                log(f"❌ ERROR in configuration: Loss={loss_type}, Augment={augment}, MaskedLoss={use_valid_mask_loss}, StackSize={stack_size}\n{e}")


def run_single_experiment(fold_idx, triplet_folds, device, args, loss_type, augment, use_valid_mask_loss, stack_size, test_only=False):
    from model import UNet2p5D
    from dataset import OCTAInpaintingDataset, IntensityAugment
    from dataset_brightnessawareness import OCTAInpaintingDataset_BrightnessAware
    from dataset_noncorruptedslices import OCTAInpaintingDataset_NonCorruptedSlices
    from dataset_brightness_noncorrupted import OCTAInpaintingDataset_Combined
    from train_val import SSIM_L1_GlobalLoss
    from train_val_brightnessawareness import SSIM_L1_BrightnessAwareLoss

    train_vols, val_vols, _ = triplet_folds[fold_idx]

    # === Select configuration
    if loss_type == "brightness_aware" and use_valid_mask_loss:
        model = UNet2p5D(in_channels=2 * stack_size, out_channels=1).to(device)
        dataset_cls = OCTAInpaintingDataset_Combined
        criterion = SSIM_L1_BrightnessAwareLoss(alpha=0.8, beta=0.1, gamma=0.1)
    elif loss_type == "brightness_aware":
        model = UNet2p5D(in_channels=stack_size, out_channels=1).to(device)
        dataset_cls = OCTAInpaintingDataset_BrightnessAware
        criterion = SSIM_L1_BrightnessAwareLoss(alpha=0.8, beta=0.1, gamma=0.0)
    elif use_valid_mask_loss:
        model = UNet2p5D(in_channels=2 * stack_size, out_channels=1).to(device)
        dataset_cls = OCTAInpaintingDataset_NonCorruptedSlices
        criterion = SSIM_L1_GlobalLoss(alpha=0.8, beta=0.1)
    else:
        model = UNet2p5D(in_channels=stack_size, out_channels=1).to(device)
        dataset_cls = OCTAInpaintingDataset
        criterion = SSIM_L1_GlobalLoss(alpha=0.8, beta=0.1)

    transform = IntensityAugment(scale_range=(0.95, 1.05), noise_std=0.005, bias_range=(-0.02, 0.02)) if augment else None
    dataset = dataset_cls(train_vols, stack_size=stack_size, transform=transform)
    sample = dataset[0]

    if isinstance(sample, tuple):
        log(f"✅ Dataset sample shapes:")
        for i, part in enumerate(sample):
            log(f"  input[{i}]: {part.shape} dtype={part.dtype}")
    else:
        log(f"⚠ Unexpected dataset sample format: {type(sample)}")

    model.eval()
    with torch.no_grad():
        inputs = [x.unsqueeze(0).float().to(device) if isinstance(x, torch.Tensor) else torch.tensor(x).unsqueeze(0).float().to(device) for x in sample]
        # inputs = [torch.tensor(x).unsqueeze(0).float().to(device) for x in sample if isinstance(x, np.ndarray)]
        output = model(inputs[0])
        log(f"✅ Model output shape: {output.shape}")

        try:
            if "brightness" in loss_type:
                loss = criterion(output, inputs[1], inputs[2])
            elif use_valid_mask_loss:
                loss = criterion(output[inputs[2] == 1], inputs[1][inputs[2] == 1])
            else:
                loss = criterion(output, inputs[1])
            log(f"✅ Loss computed: {loss.item():.6f}")
        except Exception as e:
            log(f"❌ Loss computation failed: {e}")


# def main():
#     args = parse_args()
#     device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')

#     log("Starting Ablation Study Pipeline")
#     volume_triplets = load_volume_triplets("/media/admin/Expansion/Mosaic_Data_for_Ipeks_Group/OCT_Inpainting_Testing/")
#     folds = get_kfold_splits(volume_triplets, k=5)

#     if args.kfold:
#         fold_range = range(len(folds))
#     else:
#         fold_range = [args.fold_idx]

#     # === Configuration Grid ===
#     loss_types = ['global', 'brightness_aware']
#     augment_flags = [False, True]
#     validity_loss_flags = [False, True]
#     stack_sizes = [5, 9, 13]

#     combinations = list(itertools.product(loss_types, augment_flags, validity_loss_flags, stack_sizes))

#     for loss_type, augment, use_valid_mask_loss, stack_size in combinations:
#         log(f"\n--- Running configuration: "
#             f"Loss={loss_type}, Augment={augment}, MaskedLoss={use_valid_mask_loss}, StackSize={stack_size} ---\n")

#         for fold_idx in fold_range:
#             run_single_experiment(
#                 fold_idx=fold_idx,
#                 triplet_folds=folds,
#                 device=device,
#                 args=args,
#                 loss_type=loss_type,
#                 augment=augment,
#                 use_valid_mask_loss=use_valid_mask_loss,
#                 stack_size=stack_size
#             )


# def run_single_experiment(fold_idx, triplet_folds, device, args, loss_type, augment, use_valid_mask_loss, stack_size):
#     from model import UNet2p5D
#     from dataset import OCTAInpaintingDataset, IntensityAugment
#     from dataset_brightnessawareness import OCTAInpaintingDataset_BrightnessAware
#     from dataset_noncorruptedslices import OCTAInpaintingDataset_NonCorruptedSlices
#     from dataset_brightness_noncorrupted import OCTAInpaintingDataset_Combined
#     from train_val import train_epoch, validate_epoch, evaluate_model_on_test, EarlyStopping, SSIM_L1_GlobalLoss
#     from train_val_brightnessawareness import train_epoch_brightnessawareness, validate_epoch_brightnessawareness, \
#         evaluate_model_on_test_brightnessawareness, SSIM_L1_BrightnessAwareLoss
#     from train_val_noncorruptedslices import train_epoch_noncorruptedslices, validate_epoch_noncorruptedslices, \
#         evaluate_model_on_test_noncorruptedslices
#     from train_val_brightness_noncorrupted import train_epoch_brightness_noncorrupted, validate_epoch_brightness_noncorrupted, \
#         evaluate_model_on_test_brightness_noncorrupted

#     train_vols, val_vols, test_vols = triplet_folds[fold_idx]
#     test_corrupted_path, test_gt_path, test_mask_path = test_vols[0]
#     base_name = os.path.basename(test_corrupted_path).replace("_corrupted.tif", "")

#     # === Select config ===
#     if loss_type == "brightness_aware" and use_valid_mask_loss:
#         model = UNet2p5D(in_channels=2 * stack_size, out_channels=1).to(device)
#         train_epoch_fn = train_epoch_brightness_noncorrupted
#         validate_epoch_fn = validate_epoch_brightness_noncorrupted
#         evaluate_model_fn = evaluate_model_on_test_brightness_noncorrupted
#         criterion = SSIM_L1_BrightnessAwareLoss(alpha=0.8, beta=0.1, gamma=0.1)
#         dataset_cls = OCTAInpaintingDataset_Combined
#     elif loss_type == "brightness_aware":
#         model = UNet2p5D(in_channels=stack_size, out_channels=1).to(device)
#         train_epoch_fn = train_epoch_brightnessawareness
#         validate_epoch_fn = validate_epoch_brightnessawareness
#         evaluate_model_fn = evaluate_model_on_test_brightnessawareness
#         criterion = SSIM_L1_BrightnessAwareLoss(alpha=0.8, beta=0.1, gamma=0.0)
#         dataset_cls = OCTAInpaintingDataset_BrightnessAware
#     elif use_valid_mask_loss:
#         model = UNet2p5D(in_channels=2 * stack_size, out_channels=1).to(device)
#         train_epoch_fn = train_epoch_noncorruptedslices
#         validate_epoch_fn = validate_epoch_noncorruptedslices
#         evaluate_model_fn = evaluate_model_on_test_noncorruptedslices
#         criterion = SSIM_L1_GlobalLoss(alpha=0.8, beta=0.1)
#         dataset_cls = OCTAInpaintingDataset_NonCorruptedSlices
#     else:
#         model = UNet2p5D(in_channels=stack_size, out_channels=1).to(device)
#         train_epoch_fn = train_epoch
#         validate_epoch_fn = validate_epoch
#         evaluate_model_fn = evaluate_model_on_test
#         criterion = SSIM_L1_GlobalLoss(alpha=0.8, beta=0.1)
#         dataset_cls = OCTAInpaintingDataset

#     # === Datasets and Augment ===
#     transform = IntensityAugment(scale_range=(0.95, 1.05), noise_std=0.005, bias_range=(-0.02, 0.02)) if augment else None
#     train_dataset = dataset_cls(train_vols, stack_size=stack_size, transform=transform)
#     val_dataset   = dataset_cls(val_vols, stack_size=stack_size)
#     test_dataset  = dataset_cls(test_vols, stack_size=stack_size)

#     train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
#     val_loader   = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
#     test_loader  = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

#     optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=4, factor=0.5, verbose=False)
#     early_stopping = EarlyStopping(patience=5, min_delta=1e-4, verbose=False)

#     best_model_path = f"output/best_model_f{fold_idx}_{loss_type}_aug{augment}_vmask{use_valid_mask_loss}_s{stack_size}.pth"

#     # === Train ===
#     best_val_loss = float('inf')
#     for epoch in range(1, args.epochs + 1):
#         train_loss = train_epoch_fn(model, train_loader, optimizer, criterion, device)
#         val_loss = validate_epoch_fn(model, val_loader, criterion, device)
#         scheduler.step(val_loss)

#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             torch.save(model.state_dict(), best_model_path)

#         early_stopping.step(val_loss)
#         if early_stopping.should_stop:
#             break

#     # === Test + Save ===
#     model.load_state_dict(torch.load(best_model_path))
#     model.eval()
#     test_loss = evaluate_model_fn(model, test_loader, criterion, device)

#     # Inpainting
#     corrupted_volume = tiff.imread(test_corrupted_path)
#     mask = tiff.imread(test_mask_path)
#     if mask.ndim == 3:
#         mask = (mask[:, 0, 0] > 0).astype(np.uint8)
#     else:
#         mask = mask.astype(np.uint8)

#     inpainted = inpaint_volume_with_model(model, corrupted_volume, mask, device, stack_size=stack_size)
#     corrected = smooth_rescale_reconstructed_volume(inpainted, corrupted_volume, mask, blend_factor=0.5)

#     tag = f"AblationOutput/inpainted_L{loss_type}_aug{augment}_vmask{use_valid_mask_loss}_s{stack_size}.tif"
#     out_path = os.path.join(os.path.dirname(test_corrupted_path), f"{base_name}_{tag}")
#     tiff.imwrite(out_path, corrected.astype(np.uint16))
#     log(f"Saved: {out_path}")

#     # Evaluate metrics
#     gt = tiff.imread(test_gt_path)
#     metrics = evaluate_volume_metrics(gt, corrected, mask)
#     log(f"Metrics [{tag}]: {metrics}")


if __name__ == "__main__":
    main()
