import numpy as np
import tifffile as tiff
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from utils import log

def load_volume(path):
    return tiff.imread(path).astype(np.float32)

def normalize(volume):
    return volume / 65535.0  # normalize 16-bit to [0, 1]

def compare_volumes(gt, method, mask, method_name):
    psnr_list = []
    ssim_list = []
    mae_list = []
    mse_list = []

    for i in range(gt.shape[0]):
        if mask[i] == 1:  # Only evaluate missing slices
            gt_slice = gt[i]
            pred_slice = method[i]

            mse_value = np.mean((gt_slice - pred_slice) ** 2)
            mae_value = np.mean(np.abs(gt_slice - pred_slice))

            if mse_value == 0:
                psnr_score = 100.0  # Avoid infinite PSNR
            else:
                psnr_score = psnr(gt_slice, pred_slice, data_range=1.0)

            ssim_score = ssim(gt_slice, pred_slice, data_range=1.0)

            psnr_list.append(psnr_score)
            ssim_list.append(ssim_score)
            mae_list.append(mae_value)
            mse_list.append(mse_value)

    avg_psnr = np.mean(psnr_list)
    avg_ssim = np.mean(ssim_list)
    avg_mae = np.mean(mae_list)
    avg_mse = np.mean(mse_list)

    log(f"{method_name}:")
    log(f"  Mean PSNR (missing slices only): {avg_psnr:.2f} dB")
    log(f"  Mean SSIM (missing slices only): {avg_ssim:.4f}")
    log(f"  Mean MAE  (missing slices only): {avg_mae:.6f}")
    log(f"  Mean MSE  (missing slices only): {avg_mse:.6f}")
    log("-" * 40)

    return psnr_list, ssim_list, mae_list, mse_list

def run_comparison(gt_path, linear_path, predicted_path, mask_path):
    # Load volumes
    gt = normalize(load_volume(gt_path))
    linear = normalize(load_volume(linear_path))
    predicted = normalize(load_volume(predicted_path))
    mask = tiff.imread(mask_path)  # mask expected as uint8 or uint16 0/1 format

    # Ensure shapes match
    assert gt.shape == linear.shape == predicted.shape == mask.shape, "Volumes and mask must match in shape!"

    # Compare volumes
    compare_volumes(gt, linear, mask, "Linear Interpolation")
    compare_volumes(gt, predicted, mask, "2.5D CNN Prediction")
