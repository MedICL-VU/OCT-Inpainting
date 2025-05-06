import torch
import torch.nn as nn
from tqdm import tqdm
from pytorch_msssim import ssim
from utils import log

def train_epoch_brightnessawareness(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0

    # for batch_idx, (X, y) in enumerate(tqdm(dataloader, desc="Training")):
    #     X, y = X.to(device), y.to(device)
    for batch_idx, (X, y, valid_mask) in enumerate(tqdm(dataloader, desc="Training")):
        X, y, valid_mask = X.to(device), y.to(device), valid_mask.to(device)

        X = X.contiguous().float()
        y = y.contiguous().float()

        output = model(X)

        if output.shape != y.shape:
            raise ValueError(f"Output shape {output.shape} != target shape {y.shape}")

        # loss = criterion(output, y)
        loss = criterion(output, y, X, valid_mask)  # X is stack input

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * X.size(0)

    return running_loss / len(dataloader.dataset)

def validate_epoch_brightnessawareness(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        # for X, y in tqdm(dataloader, desc="Validating"):
        #     X, y = X.to(device), y.to(device)
        for batch_idx, (X, y, valid_mask) in enumerate(tqdm(dataloader, desc="Validating")):
            X, y, valid_mask = X.to(device), y.to(device), valid_mask.to(device)

            X = X.contiguous().float()
            y = y.contiguous().float()

            output = model(X)
            # loss = criterion(output, y)
            loss = criterion(output, y, X, valid_mask)  # X is stack input

            running_loss += loss.item() * X.size(0)

    return running_loss / len(dataloader.dataset)

def evaluate_model_on_test_brightnessawareness(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        # for X, y in tqdm(dataloader, desc="Testing"):
        #     X, y = X.to(device), y.to(device)
        for batch_idx, (X, y, valid_mask) in enumerate(tqdm(dataloader, desc="Testing")):
            X, y, valid_mask = X.to(device), y.to(device), valid_mask.to(device)

            X = X.contiguous().float()
            y = y.contiguous().float()

            output = model(X)
            # loss = criterion(output, y)
            loss = criterion(output, y, X, valid_mask)  # X is stack input

            running_loss += loss.item() * X.size(0)

    return running_loss / len(dataloader.dataset)


# class SSIM_L1_Loss(nn.Module):
#     def __init__(self, alpha=0.84):  # alpha = L1 weight, 1-alpha = SSIM weight
#         super().__init__()
#         self.alpha = alpha
#         self.l1 = nn.L1Loss()

#     def forward(self, pred, target):
#         l1_loss = self.l1(pred, target)
#         ssim_loss = 1 - ssim(pred, target, data_range=1.0)
#         return self.alpha * l1_loss + (1 - self.alpha) * ssim_loss

# class SSIM_L1_GlobalLoss(nn.Module):
#     def __init__(self, alpha=0.8, beta=0.1):
#         """
#         alpha = weight for L1 vs SSIM
#         beta = weight for global mean intensity matching
#         """
#         super().__init__()
#         self.alpha = alpha
#         self.beta = beta
#         self.l1 = nn.L1Loss()

#     def forward(self, pred, target):
#         l1_loss = self.l1(pred, target)
#         ssim_loss = 1 - ssim(pred, target, data_range=1.0)

#         mean_loss = torch.abs(torch.mean(pred) - torch.mean(target))

#         return self.alpha * l1_loss + (1 - self.alpha) * ssim_loss + self.beta * mean_loss

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

    def forward(self, pred, target, stack, valid_mask):
        # pred, target: (B, 1, H, W)
        # stack: (B, S, H, W), valid_mask: (B, S)

        l1_loss = self.l1(pred, target)
        ssim_loss = 1 - ssim(pred, target, data_range=1.0)
        mean_loss = torch.abs(pred.mean() - target.mean())

        # === Brightness difference from valid neighbors ===
        B, S, H, W = stack.shape
        stack_means = stack.view(B, S, -1).mean(dim=2)           # (B, S)
        pred_means = pred.view(B, -1).mean(dim=1, keepdim=True)  # (B, 1)

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
