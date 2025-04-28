import torch
import torch.nn as nn
from tqdm import tqdm
import os

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0

    for X, y in tqdm(dataloader, desc="Training"):
        X, y = X.to(device), y.to(device)
        
        # Ensure tensors are contiguous and float32
        X = X.contiguous().float()
        y = y.contiguous().float()

        output = model(X)

        # Match output and target shapes
        if output.shape != y.shape:
            raise ValueError(f"Output shape {output.shape} != target shape {y.shape}")

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

            # Ensure tensors are contiguous and float32
            X = X.contiguous().float()
            y = y.contiguous().float()

            output = model(X)
            loss = criterion(output, y)
            total_loss += loss.item() * X.size(0)

    return total_loss / len(dataloader.dataset)


def evaluate_model_on_test(model, dataloader, criterion, device):
    """
    Evaluate model on a held-out test set.
    """
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for X, y in tqdm(dataloader, desc="Testing"):
            X, y = X.to(device), y.to(device)

            # Ensure tensors are contiguous and float32
            X = X.contiguous().float()
            y = y.contiguous().float()

            output = model(X)
            loss = criterion(output, y)
            total_loss += loss.item() * X.size(0)

    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss


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
                print(f"Validation loss improved to {current_loss:.6f}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"No improvement ({self.counter}/{self.patience})")
            if self.counter >= self.patience:
                self.should_stop = True
