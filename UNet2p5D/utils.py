import os
from datetime import datetime
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import random
from skimage.metrics import structural_similarity as skimage_ssim
from scipy.ndimage import uniform_filter


os.makedirs("output/logs", exist_ok=True)
log_file = os.path.join("output/logs", f"run_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

def log(msg):
    print(msg)  # Optional: Print to console
    with open(log_file, 'a') as schylar:
        schylar.write(msg + '\n')


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


def visualize_ncc_slice_stacked(gt_volume, pred_volume, mask, slice_idx, window_sizes=[7, 11, 17]):
    """
    Visualize GT, prediction, and NCC heatmaps for one slice
    with multiple window_sizes stacked vertically.
    """
    gt = (gt_volume[slice_idx].astype(np.float32)) / 65535.0
    pred = (pred_volume[slice_idx].astype(np.float32)) / 65535.0

    num_windows = len(window_sizes)
    nrows = 2 + num_windows # GT, Pred, then num_windows for NCC maps
    
    # Use the same figure size as your final SSIM function
    fig, axes = plt.subplots(nrows, 1, figsize=(5, 2.5 * nrows))
    
    # Use the same hspace, top, and bottom as your final SSIM function
    plt.subplots_adjust(hspace=0.2, top=0.95, bottom=0.05)

    # --- Row 0: Ground-Truth Slice ---
    ax_gt = axes[0]
    ax_gt.imshow(gt, cmap='gray')
    ax_gt.set_title(f"Ground-Truth", fontsize=10, pad=5) # Match SSIM title format
    ax_gt.axis('off')

    # --- Row 1: Predicted Slice ---
    ax_pred = axes[1]
    ax_pred.imshow(pred, cmap='gray')
    ax_pred.set_title(f"Predicted", fontsize=10, pad=5) # Match SSIM title format
    ax_pred.axis('off')

    # --- Subsequent Rows: NCC Maps for different window sizes ---
    for i, window_size in enumerate(window_sizes):
        ax_ncc = axes[2 + i]
        try:
            # Use your provided compute_local_ncc function
            ncc_map = np.clip(compute_local_ncc(gt, pred, window_size=window_size), -1, 1)
        except Exception as e:
            print(f"NCC computation for win_size={window_size} failed: {e}")
            ncc_map = np.zeros_like(gt) # Fallback if NCC fails

        # NCC values range from -1 to 1, so use 'bwr' cmap and set vmin/vmax accordingly
        im = ax_ncc.imshow(ncc_map, cmap='bwr', vmin=-1, vmax=1)
        # Match SSIM title format: "NCC Map (win=X)"
        ax_ncc.set_title(f"NCC Map (win={window_size})", fontsize=10, pad=5)
        
        # Match colorbar styling
        cbar = fig.colorbar(im, ax=ax_ncc, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=7)
        ax_ncc.axis('off')

    plt.show()


def visualize_ssim_slice_stacked(gt_volume, pred_volume, mask, slice_idx, window_sizes=[7, 11, 17]):
    """
    Visualize GT, prediction, and SSIM heatmaps for one slice
    with multiple window_sizes stacked vertically.
    """
    gt = (gt_volume[slice_idx].astype(np.float32)) / 65535.0
    pred = (pred_volume[slice_idx].astype(np.float32)) / 65535.0

    num_windows = len(window_sizes)
    nrows = 2 + num_windows # GT, Pred, then num_windows for SSIM maps
    
    # Keep the reduced figure size
    fig, axes = plt.subplots(nrows, 1, figsize=(5, 2.5 * nrows))
    
    # Crucial change: Reduce hspace to bring rows closer
    plt.subplots_adjust(hspace=0.2, top=0.95, bottom=0.05) # Reduced hspace from 0.6 to 0.4

    # --- Row 0: Ground-Truth Slice ---
    ax_gt = axes[0]
    ax_gt.imshow(gt, cmap='gray')
    ax_gt.set_title(f"Ground-Truth", fontsize=10, pad=5)
    ax_gt.axis('off')

    # --- Row 1: Predicted Slice ---
    ax_pred = axes[1]
    ax_pred.imshow(pred, cmap='gray')
    ax_pred.set_title(f"Predicted", fontsize=10, pad=5)
    ax_pred.axis('off')

    # --- Subsequent Rows: SSIM Maps for different window sizes ---
    for i, window_size in enumerate(window_sizes):
        ax_ssim = axes[2 + i]
        try:
            ssim_map = skimage_ssim(gt, pred, data_range=1.0, win_size=window_size, full=True)[1]
        except Exception as e:
            print(f"SSIM computation for win_size={window_size} failed: {e}")
            ssim_map = np.zeros_like(gt)

        im = ax_ssim.imshow(ssim_map, cmap='hot', vmin=0, vmax=1)
        ax_ssim.set_title(f"SSIM Map (win={window_size})", fontsize=10, pad=5)
        
        cbar = fig.colorbar(im, ax=ax_ssim, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=7)
        ax_ssim.axis('off')

    plt.show()


def visualize_slice_panel(gt_volume, pred_volume, mask, slice_indices=None, ncols=3):
    nrows = 5  # GT, Pred, NCC17, SSIM11, AbsError

    if slice_indices is None:
        slice_indices = np.where(mask)[0][:ncols]

    # Adjust figsize for better readability with titles on top
    fig, axes = plt.subplots(nrows, len(slice_indices), figsize=(4 * len(slice_indices), 3.8 * nrows))
    plt.subplots_adjust(hspace=0.6, wspace=0.1) # Increased hspace for titles

    # Define the labels for each row once
    row_labels = ["GT", "Prediction", "NCC17", "SSIM11", "Abs Error"]

    for col, idx in enumerate(slice_indices):
        gt = (gt_volume[idx].astype(np.float32)) / 65535.0
        pred = (pred_volume[idx].astype(np.float32)) / 65535.0
        abs_error = np.abs(gt - pred)

        ncc_map = np.clip(compute_local_ncc(gt, pred, window_size=17), -1, 1)
        try:
            ssim_map = skimage_ssim(gt, pred, data_range=1.0, win_size=11, full=True)[1]
        except ValueError:
            ssim_map = np.zeros_like(gt)

        slices = [
            (gt, 'gray', None),
            (pred, 'gray', None),
            (ncc_map, 'bwr', (-1, 1)),
            (ssim_map, 'hot', (0, 1)),
            (abs_error, 'hot', (np.percentile(abs_error, 1), np.percentile(abs_error, 99)))
        ]

        # Set the main column title (Slice X) only once per column, on the very top row
        axes[0, col].set_title(f"Slice {idx}", fontsize=12, pad=20) # Increased pad to move it higher

        for row, (image, cmap, limits) in enumerate(slices):
            ax = axes[row, col]
            vmin, vmax = limits if limits else (None, None)
            im = ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)

            # Set the label for each individual image (GT, Prediction, NCC17, etc.)
            ax.set_title(row_labels[row], fontsize=10, loc='center', pad=5) # Label on top of each image

            # Add colorbar for NCC, SSIM, Error
            if row in [2, 3, 4]:
                cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.01)
                cbar.ax.tick_params(labelsize=8)

            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)

    plt.tight_layout()
    plt.show()


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.best_loss = float('inf')
        self.counter = 0
        self.should_stop = False

    def step(self, current_loss):
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.counter = 0
            if self.verbose:
                log(f"Validation loss improved to {current_loss:.6f}")
        else:
            self.counter += 1
            if self.verbose:
                log(f"No improvement ({self.counter}/{self.patience})")
            if self.counter >= self.patience:
                self.should_stop = True
