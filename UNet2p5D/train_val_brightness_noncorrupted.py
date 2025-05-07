import torch
import torch.nn as nn
from tqdm import tqdm
from pytorch_msssim import ssim
from utils import log

def train_epoch_brightness_noncorrupted(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0

    for stack, valid_pixelwise, y, valid_slicewise in dataloader:
        X = torch.cat([stack, valid_pixelwise], dim=1).to(device)     # Input to model
        y = y.to(device)
        valid_slicewise = valid_slicewise.to(device)

        X = X.contiguous().float()
        y = y.contiguous().float()

        output = model(X)

        if output.shape != y.shape:
            raise ValueError(f"Output shape {output.shape} != target shape {y.shape}")

        loss = criterion(output, y, stack.to(device), valid_slicewise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * X.size(0)

    return running_loss / len(dataloader.dataset)


def validate_epoch_brightness_noncorrupted(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for stack, valid_pixelwise, y, valid_slicewise in dataloader:
            X = torch.cat([stack, valid_pixelwise], dim=1).to(device)     # Input to model
            y = y.to(device)
            valid_slicewise = valid_slicewise.to(device)

            X = X.contiguous().float()
            y = y.contiguous().float()

            output = model(X)
            loss = criterion(output, y, stack.to(device), valid_slicewise)

            total_loss += loss.item() * X.size(0)

    return total_loss / len(dataloader.dataset)


def evaluate_model_on_test_brightness_noncorrupted(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for stack, valid_pixelwise, y, valid_slicewise in dataloader:
            X = torch.cat([stack, valid_pixelwise], dim=1).to(device)     # Input to model
            y = y.to(device)
            valid_slicewise = valid_slicewise.to(device)

            X = X.contiguous().float()
            y = y.contiguous().float()

            output = model(X)
            loss = criterion(output, y, stack.to(device), valid_slicewise)

            total_loss += loss.item() * X.size(0)

    return total_loss / len(dataloader.dataset)


class SSIM_L1_BrightnessAwareLoss(nn.Module):
    def __init__(self, alpha=0.8, beta=0.1, gamma=0.1):
        """
        alpha = L1 vs SSIM balance
        beta  = predicted vs target global brightness match
        gamma = predicted vs neighbor slice brightness match (using valid stack slices only)
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.l1 = nn.L1Loss()

    def forward(self, pred, target, stack, valid_mask=None):
        # pred, target: (B, 1, H, W)
        # stack: (B, S, H, W)
        # valid_mask: (B, S)

        l1_loss = self.l1(pred, target)
        ssim_loss = 1 - ssim(pred, target, data_range=1.0)
        mean_loss = torch.abs(pred.mean() - target.mean())

        # === Brightness difference from valid neighbors ===
        B, S, H, W = stack.shape
        stack_means = stack.view(B, S, -1).mean(dim=2)           # (B, S)
        pred_means = pred.view(B, -1).mean(dim=1, keepdim=True)  # (B, 1)

        if valid_mask is not None:
            # Only consider valid slices
            valid_neighbors = stack_means * valid_mask               # (B, S)
            num_valid = valid_mask.sum(dim=1, keepdim=True).clamp(min=1.0)

            neighbor_mean = valid_neighbors.sum(dim=1, keepdim=True) / num_valid
            brightness_consistency = torch.abs(pred_means - neighbor_mean).mean()

        return (self.alpha * l1_loss +
                (1 - self.alpha) * ssim_loss +
                self.beta * mean_loss +
                self.gamma * brightness_consistency)


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
