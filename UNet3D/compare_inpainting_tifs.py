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

        # Compute MSE
        mse = np.mean((gt_slice - pred_slice) ** 2)

        # Handle MSE = 0 to avoid PSNR = inf
        if mse == 0:
            psnr_score = 100.0  # Assign a very high PSNR value
        else:
            psnr_score = psnr(gt_slice, pred_slice, data_range=1.0)

        # Compute SSIM
        ssim_score = ssim(gt_slice, pred_slice, data_range=1.0)

        psnr_list.append(psnr_score)
        ssim_list.append(ssim_score)

    avg_psnr = np.mean(psnr_list)
    avg_ssim = np.mean(ssim_list)

    log(f"{method_name}:")
    log(f"  Mean PSNR: {avg_psnr:.2f} dB")
    log(f"  Mean SSIM: {avg_ssim:.4f}")
    log("-" * 40)

    return psnr_list, ssim_list

def run_comparison(gt_path, linear_path, predicted_path):
    # Load volumes
    gt = normalize(load_volume(gt_path))
    linear = normalize(load_volume(linear_path))
    predicted = normalize(load_volume(predicted_path))

    # Ensure shapes match
    assert gt.shape == linear.shape == predicted.shape, "Volumes must match in shape!"

    # Compare volumes
    compare_volumes(gt, linear, "Linear Interpolation")
    compare_volumes(gt, predicted, "3D CNN Prediction")
