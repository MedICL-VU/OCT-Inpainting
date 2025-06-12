import torch
import torch.nn as nn
from tqdm import tqdm
from pytorch_msssim import ssim
from utils import log


def train_epoch(model, dataloader, optimizer, criterion, device, debug=False, disable_dynamic_filter=False):
    running_loss = 0.0
    count = 0
    log_terms = {"l1": 0.0, "ssim": 0.0, "global_mean": 0.0, "neighbor_relative": 0.0}

    model.train()

    # for batch_idx, (X, y, valid_mask) in enumerate(tqdm(dataloader, desc="Training")):
    for batch_idx, (X, y, valid_mask) in enumerate(dataloader):
        X, y, valid_mask = X.to(device), y.to(device), valid_mask.to(device)

        X = X.contiguous().float()
        y = y.contiguous().float()

        output = model(X, valid_mask, disable_dynamic_filter)

        if debug and batch_idx < 3:
            log(f"[TRAIN] Batch shape: {X.shape} | Target shape: {y.shape}")
            for b in range(min(2, X.shape[0])):
                valid = (X[b].sum(dim=(1, 2)) > 0).nonzero().squeeze().tolist()
                log(f" - Sample {b}: non-zero slices: {valid}")
            log(f"Pred min/max: {output.min().item():.4f} / {output.max().item():.4f}")

        if output.shape != y.shape:
            raise ValueError(f"Output shape {output.shape} != target shape {y.shape}")

        if torch.isnan(output).any() or torch.isinf(output).any():
            log(f"[ERROR] Model output contains NaNs or Infs at batch {batch_idx}")

        loss, terms = criterion(output, y, X, valid_mask)  # X is stack input

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * X.size(0)

        for k in log_terms:
            log_terms[k] += terms.get(k, 0.0) * X.size(0)
        count += X.size(0)

    avg_terms = {k: round(v / count, 6) for k, v in log_terms.items()}

    return running_loss / count, avg_terms


def validate_epoch(model, dataloader, criterion, device, disable_dynamic_filter=False):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for batch_idx, (X, y, valid_mask) in enumerate(dataloader):
            X, y, valid_mask = X.to(device), y.to(device), valid_mask.to(device)

            X = X.contiguous().float()
            y = y.contiguous().float()

            output = model(X, valid_mask, disable_dynamic_filter)

            if torch.isnan(output).any() or torch.isinf(output).any():
                log(f"[ERROR] Model output contains NaNs or Infs at batch {batch_idx}")

            loss, terms = criterion(output, y, X, valid_mask)  # X is stack input

            running_loss += loss.item() * X.size(0)

    return running_loss / len(dataloader.dataset)


def evaluate_model_on_test(model, dataloader, criterion, device, disable_dynamic_filter=False):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for batch_idx, (X, y, valid_mask) in enumerate(tqdm(dataloader, desc="Testing")):
            X, y, valid_mask = X.to(device), y.to(device), valid_mask.to(device)

            X = X.contiguous().float()
            y = y.contiguous().float()

            output = model(X, valid_mask, disable_dynamic_filter)

            loss, terms = criterion(output, y, X, valid_mask)  # X is stack input

            running_loss += loss.item() * X.size(0)

    return running_loss / len(dataloader.dataset)
        

class SSIM_L1_BrightnessAwareLoss(nn.Module):
    def __init__(self, alpha=0.8, beta=0.1, gamma=0.1):
        """
        alpha = L1 vs SSIM balance
        beta  = predicted vs target global brightness match
        gamma = predicted vs neighbor slice brightness match
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.l1 = nn.L1Loss()

    def forward(self, pred, target, stack=None, valid_mask=None):
        # pred, target: (B, 1, H, W)
        # stack: (B, S, H, W), valid_mask: (B, S)

        l1_loss = self.l1(pred, target)
        ssim_loss = 1 - ssim(pred, target, data_range=1.0)

        # Global mean brightness alignment
        mean_loss = torch.abs(pred.mean() - target.mean())

        # Brightness-aware neighbor term
        if stack is not None and valid_mask is not None:
            B, S, H, W = stack.shape
            stack_means = stack.view(B, S, -1).mean(dim=2)                # (B, S)
            pred_means = pred.view(B, -1).mean(dim=1, keepdim=True)       # (B, 1)

            valid_neighbors = stack_means * valid_mask                    # (B, S)
            num_valid = valid_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
            neighbor_mean = valid_neighbors.sum(dim=1, keepdim=True) / num_valid

            brightness_consistency = torch.abs(pred_means - neighbor_mean).mean()
        else:
            brightness_consistency = torch.tensor(0.0, device=pred.device)

        total_loss = (
            self.alpha * l1_loss +
            (1 - self.alpha) * ssim_loss +
            self.beta * mean_loss +
            self.gamma * brightness_consistency
        )

        # Return loss + dictionary for diagnostics
        return total_loss, {
            "l1": l1_loss.item(),
            "ssim": ssim_loss.item(),
            "global_mean": mean_loss.item(),
            "neighbor_relative": brightness_consistency.item()
        }
