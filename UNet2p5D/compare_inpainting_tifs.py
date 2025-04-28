import numpy as np
import tifffile as tiff
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from utils import log

def load_volume(path):
    return tiff.imread(path).astype(np.float32)

def normalize(volume):
    return volume / 65535.0  # normalize 16-bit to [0, 1]

def compare_volumes(gt, method, method_name):
    psnr_list = []
    ssim_list = []

    for i in range(gt.shape[0]):
        gt_slice = gt[i]
        pred_slice = method[i]

        psnr_score = psnr(gt_slice, pred_slice, data_range=1.0)
        ssim_score = ssim(gt_slice, pred_slice, data_range=1.0)

        psnr_list.append(psnr_score)
        ssim_list.append(ssim_score)

    avg_psnr = np.mean(psnr_list)
    avg_ssim = np.mean(ssim_list)

    log(f"🔍 {method_name}:")
    log(f"  Mean PSNR: {avg_psnr:.2f} dB")
    log(f"  Mean SSIM: {avg_ssim:.4f}")
    log("-" * 40)

    return psnr_list, ssim_list

# === File Paths ===
gt_path = "OCTA_ground_truth.tif"
linear_path = "OCTA_linear_interp.tif"
predicted_path = "OCTA_predicted_2p5D.tif"

# === Load Volumes ===
gt = normalize(load_volume(gt_path))
linear = normalize(load_volume(linear_path))
predicted = normalize(load_volume(predicted_path))

# === Shape Check ===
assert gt.shape == linear.shape == predicted.shape, "Volumes must match in shape!"

# === Compare ===
compare_volumes(gt, linear, "Linear Interpolation")
compare_volumes(gt, predicted, "2.5D CNN Prediction")
