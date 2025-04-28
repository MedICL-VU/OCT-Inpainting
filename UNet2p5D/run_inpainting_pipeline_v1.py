import os
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import tifffile as tiff
import numpy as np
from tqdm import tqdm
from utils import log

# Import your modules (assumes they are saved in .py files)
from dataset import OCTAInpaintingDataset
from model import UNet2p5D
from train_val import train_epoch, validate_epoch, evaluate_model_on_test, EarlyStopping
from save_inpainted import inpaint_volume_with_model
from compare_inpainting_tifs import compare_volumes, normalize, load_volume

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run 2.5D Inpainting Pipeline")
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--stack_size', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--cuda', action='store_true', help='Use CUDA if available')
    return parser.parse_args()


import os

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


def split_volumes(triplets):
    """
    Use 3 volumes for training, 1 for validation, 1 for test
    """
    assert len(triplets) >= 5, "Need at least 5 volumes for 3-train / 1-val / 1-test split"

    sorted_trips = sorted(triplets)  # Sort alphabetically or by filename
    train = sorted_trips[:3]
    val = [sorted_trips[3]]
    test = [sorted_trips[4]]
    return train, val, test


def main():
    args = parse_args()
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')

    log("Starting Inpainting Pipeline")
    log(f"Device: {device}")
    log(f"Epochs: {args.epochs} | Batch size: {args.batch_size} | Stack size: {args.stack_size}")

    # Replace hardcoded values with args:
    batch_size = args.batch_size
    num_epochs = args.epochs
    stack_size = args.stack_size
    learning_rate = args.lr

    # === Configurations ===
    corrupted_path = "OCTA_corrupted.tif"
    clean_path = "OCTA_ground_truth.tif"
    mask_path = "OCTA_mask.tif"
    linear_interp_path = "OCTA_linear_interp.tif"

    predicted_output_path = "OCTA_predicted_2p5D.tif"
    best_model_path = "best_model.pth"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # === 1. Load Dataset ===
    log("Loading datasets...")
    # Load and split volumes
    volume_triplets = load_volume_triplets("data/")
    train_vols, val_vols, test_vols = split_volumes(volume_triplets)

    log(f"Using {len(train_vols)} volumes for training, {len(test_vols)} for testing")

    # === Identify test volume metadata ===
    test_corrupted_path, test_gt_path, test_mask_path = test_vols[0]

    # Generate output filename based on test volume name
    base_name = os.path.basename(test_corrupted_path).replace("_corrupted.tif", "")
    predicted_output_path = os.path.join(
        "results", f"{base_name}_inpainted_2p5DUNet.tif"
    )

    # Ensure output folder exists
    os.makedirs("results", exist_ok=True)


    # Build datasets
    train_dataset = OCTAInpaintingDataset(train_vols, stack_size=args.stack_size)
    val_dataset   = OCTAInpaintingDataset(val_vols, stack_size=args.stack_size)
    test_dataset  = OCTAInpaintingDataset(test_vols, stack_size=args.stack_size)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)


    log("Training volumes:")
    for v in train_vols: log(f" - {os.path.basename(v[0])}")
    log("Validation volume:")
    log(f" - {os.path.basename(val_vols[0][0])}")
    log("Test volume:")
    log(f" - {os.path.basename(test_vols[0][0])}")


    # === 2. Initialize Model ===
    log("Initializing model...")
    model = UNet2p5D(in_channels=stack_size, out_channels=1).to(device)
    criterion = torch.nn.L1Loss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    early_stopping = EarlyStopping(patience=5, min_delta=1e-4, verbose=True)

    # === 3. Train Model ===
    log("Starting training...")
    best_val_loss = float('inf')

    for epoch in range(1, num_epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate_epoch(model, val_loader, criterion)

        # Log
        log(f"[Epoch {epoch}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

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

    model.load_state_dict(torch.load(best_model_path))
    model.eval()


    # === 3.5: Evaluate on Held-Out Test Volume ===
    log("Evaluating on held-out test volume...")
    test_loss = evaluate_model_on_test(model, test_loader, criterion, device)
    log(f"Final test loss: {test_loss:.4f}")

    # === 4. Inpaint Full Volume with Trained Model ===
    log("Inpainting volume...")
    corrupted_volume = tiff.imread(corrupted_path)
    mask_volume = tiff.imread(mask_path)
    if mask_volume.ndim == 3:
        mask = (mask_volume[:, 0, 0] > 0).astype(np.uint8)
    else:
        mask = mask_volume.astype(np.uint8)

    # Load best model
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    # Inpaint and save
    inpainted_volume = inpaint_volume_with_model(model, corrupted_volume, mask, device, stack_size=stack_size)
    tiff.imwrite(predicted_output_path, inpainted_volume.astype(np.uint16))

    log(f"Inpainted volume saved to: {predicted_output_path}")

    # === 5. Compare Linear Interpolation, 2.5D CNN, Ground Truth ===
    log("Comparing results...")

    # Load all volumes for comparison
    gt = normalize(load_volume(test_gt_path))
    linear = normalize(load_volume(test_corrupted_path.replace("_corrupted.tif", "_LinearInterp.tif")))
    predicted = normalize(load_volume(predicted_output_path))


    assert gt.shape == linear.shape == predicted.shape, "Volumes must match shape!"

    # Run comparison
    compare_volumes(gt, linear, "Linear Interpolation")
    compare_volumes(gt, predicted, "2.5D CNN Prediction")


if __name__ == "__main__":
    main()
