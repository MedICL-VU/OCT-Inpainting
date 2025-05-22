import torch
import torch.nn as nn
from tqdm import tqdm
from pytorch_msssim import ssim
from utils import log


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0

    for batch_idx, (X, y) in enumerate(tqdm(dataloader, desc="Training")):
        X, y = X.to(device), y.to(device)

        X = X.contiguous().float()
        y = y.contiguous().float()

        output = model(X)

        if hasattr(criterion, 'debug') and criterion.debug:
            log(f"[TRAIN] Batch shape: {X.shape} | Target shape: {y.shape}")
            for b in range(min(2, X.shape[0])):
                valid = (X[b].sum(dim=(1, 2)) > 0).nonzero().squeeze().tolist()
                log(f" - Sample {b}: non-zero slices: {valid}")
            log(f"Pred min/max: {output.min().item():.4f} / {output.max().item():.4f}")

        if output.shape != y.shape:
            raise ValueError(f"Output shape {output.shape} != target shape {y.shape}")

        if torch.isnan(output).any() or torch.isinf(output).any():
            log(f"[ERROR] Model output contains NaNs or Infs at batch {batch_idx}")

        loss = criterion(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * X.size(0)

    return running_loss / len(dataloader.dataset)


def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for X, y in tqdm(dataloader, desc="Validating"):
            X, y = X.to(device), y.to(device)

            X = X.contiguous().float()
            y = y.contiguous().float()

            output = model(X)
            if torch.isnan(output).any() or torch.isinf(output).any():
                log(f"[ERROR] Model output contains NaNs or Infs at batch {batch_idx}")

            loss = criterion(output, y)
            total_loss += loss.item() * X.size(0)

    return total_loss / len(dataloader.dataset)


def evaluate_model_on_test(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for X, y in tqdm(dataloader, desc="Testing"):
            X, y = X.to(device), y.to(device)

            X = X.contiguous().float()
            y = y.contiguous().float()

            output = model(X)
            loss = criterion(output, y)
            total_loss += loss.item() * X.size(0)

    return total_loss / len(dataloader.dataset)


class SSIM_L1_GlobalLoss(nn.Module):
    def __init__(self, alpha=0.8, beta=0.1):
        """
        alpha = weight for L1 vs SSIM
        beta = weight for global mean intensity matching
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.l1 = nn.L1Loss()

    def forward(self, pred, target):
        l1_loss = self.l1(pred, target)
        ssim_loss = 1 - ssim(pred, target, data_range=1.0)

        mean_loss = torch.abs(torch.mean(pred) - torch.mean(target))

        return self.alpha * l1_loss + (1 - self.alpha) * ssim_loss + self.beta * mean_loss


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
