import torch
import torch.nn as nn
from tqdm import tqdm
from pytorch_msssim import ssim
from utils import log


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    l1_total, ssim_total, total_batches = 0.0, 0.0, 0

    for batch_idx, (X, y, m) in enumerate(tqdm(dataloader, desc="Training")):
        X, y, m = X.to(device), y.to(device), m.to(device)
        X = X.contiguous().float()
        y = y.contiguous().float()
        m = m.contiguous().unsqueeze(-1).unsqueeze(-1)  # (B, 1, D, 1, 1)

        output = model(X)

        if output.shape != y.shape:
            raise ValueError(f"Output shape {output.shape} != target shape {y.shape}")

        # Apply mask
        masked_output = output * m
        masked_target = y * m
        loss = criterion(masked_output, masked_target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * X.size(0)
        total_batches += 1

        if isinstance(criterion, SSIM_L1_Loss):
            with torch.no_grad():
                l1_val = criterion.l1(masked_output, masked_target).item()
                ssim_val = 1 - ssim(masked_output, masked_target, data_range=1.0).item()
                l1_total += l1_val
                ssim_total += ssim_val

    avg_loss = running_loss / len(dataloader.dataset)

    if isinstance(criterion, SSIM_L1_Loss) and total_batches > 0:
        l1_avg = l1_total / total_batches
        ssim_avg = ssim_total / total_batches
        log(f"Train SSIM-L1 Breakdown | Avg Loss: {avg_loss:.4f} | L1: {l1_avg:.4f} | SSIM: {ssim_avg:.4f}")

    return avg_loss


def validate_epoch(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for X, y, m in tqdm(dataloader, desc="Validating"):
            X, y, m = X.to(next(model.parameters()).device), y.to(next(model.parameters()).device), m.to(next(model.parameters()).device)
            X = X.contiguous().float()
            y = y.contiguous().float()
            m = m.contiguous().unsqueeze(-1).unsqueeze(-1)  # (B, 1, D, 1, 1)

            output = model(X)
            masked_output = output * m
            masked_target = y * m
            loss = criterion(masked_output, masked_target)
            total_loss += loss.item() * X.size(0)

    return total_loss / len(dataloader.dataset)


def evaluate_model_on_test(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for X, y, m in tqdm(dataloader, desc="Testing"):
            X, y, m = X.to(device), y.to(device), m.to(device)
            X = X.contiguous().float()
            y = y.contiguous().float()
            m = m.contiguous().unsqueeze(-1).unsqueeze(-1)

            output = model(X)
            masked_output = output * m
            masked_target = y * m
            loss = criterion(masked_output, masked_target)
            total_loss += loss.item() * X.size(0)

    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss


class SSIM_L1_Loss(nn.Module):
    def __init__(self, alpha=0.84):  # alpha = L1 weight, 1-alpha = SSIM weight
    # def __init__(self, alpha=0.7):  # alpha = L1 weight, 1-alpha = SSIM weight
        super().__init__()
        self.alpha = alpha
        self.l1 = nn.L1Loss()

    def forward(self, pred, target):
        l1_loss = self.l1(pred, target)
        ssim_loss = 1 - ssim(pred, target, data_range=1.0)
        return self.alpha * l1_loss + (1 - self.alpha) * ssim_loss


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
