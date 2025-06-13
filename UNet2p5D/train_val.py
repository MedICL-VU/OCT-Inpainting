import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from pytorch_msssim import ssim
from torchvision.models import vgg16
from torchvision.transforms import Resize
from utils import log


def train_epoch(model, dataloader, optimizer, criterion, device, debug=False, disable_dynamic_filter=False):
    running_loss = 0.0
    count = 0
    log_terms = {"l1": 0.0, "ssim": 0.0, "global_mean": 0.0, "neighbor_relative": 0.0, "perceptual": 0.0, "edge": 0.0}

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
        

class HybridLoss(nn.Module):
    def __init__(self,
                 l1_scale=0.8,
                 ssim_scale=0.1,
                 global_scale=0.1,
                 neighbor_scale=0.1,
                 perceptual_scale=0.05,
                 edge_scale=0.05):
        super().__init__()
        self.l1_scale = l1_scale
        self.ssim_scale = ssim_scale
        self.global_scale = global_scale
        self.neighbor_scale = neighbor_scale
        self.perceptual_scale = perceptual_scale
        self.edge_scale = edge_scale

        self.l1 = nn.L1Loss()
        self.perceptual_net = self._build_vgg16_feature_extractor()
        self.resize_224 = Resize((224, 224))  # VGG expects 224×224

        # Sobel kernels
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def _build_vgg16_feature_extractor(self):
        vgg = vgg16(pretrained=True).features[:16]  # Up to relu3_3
        for param in vgg.parameters():
            param.requires_grad = False
        return vgg.eval()

    def _compute_perceptual_loss(self, pred, target):
        # Both inputs: (B, 1, H, W) => convert to 3-channel (B, 3, H, W) and resize
        pred_rgb = pred.expand(-1, 3, -1, -1)
        target_rgb = target.expand(-1, 3, -1, -1)
        pred_resized = self.resize_224(pred_rgb)
        target_resized = self.resize_224(target_rgb)

        pred_feat = self.perceptual_net(pred_resized)
        target_feat = self.perceptual_net(target_resized)

        return F.l1_loss(pred_feat, target_feat)

    def _compute_edge_loss(self, pred, target):
        pred_dx = F.conv2d(pred, self.sobel_x, padding=1)
        pred_dy = F.conv2d(pred, self.sobel_y, padding=1)
        pred_edges = torch.sqrt(pred_dx**2 + pred_dy**2 + 1e-6)

        target_dx = F.conv2d(target, self.sobel_x, padding=1)
        target_dy = F.conv2d(target, self.sobel_y, padding=1)
        target_edges = torch.sqrt(target_dx**2 + target_dy**2 + 1e-6)

        return F.l1_loss(pred_edges, target_edges)

    def forward(self, pred, target, stack=None, valid_mask=None):
        # pred, target: (B, 1, H, W)
        # stack: (B, S, H, W), valid_mask: (B, S)

        l1_loss = self.l1(pred, target)
        ssim_loss = 1 - ssim(pred, target, data_range=1.0)

        mean_loss = torch.abs(pred.mean() - target.mean())

        # Brightness-aware neighbor consistency
        if stack is not None and valid_mask is not None:
            B, S, H, W = stack.shape
            stack_means = stack.view(B, S, -1).mean(dim=2)
            pred_means = pred.view(B, -1).mean(dim=1, keepdim=True)
            valid_neighbors = stack_means * valid_mask
            num_valid = valid_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
            neighbor_mean = valid_neighbors.sum(dim=1, keepdim=True) / num_valid
            brightness_consistency = torch.abs(pred_means - neighbor_mean).mean()
        else:
            brightness_consistency = torch.tensor(0.0, device=pred.device)

        perceptual_loss = self._compute_perceptual_loss(pred, target)
        edge_loss = self._compute_edge_loss(pred, target)

        total_loss = (
            self.l1_scale * l1_loss +
            self.ssim_scale * ssim_loss +
            self.global_scale * mean_loss +
            self.neighbor_scale * brightness_consistency +
            self.perceptual_scale * perceptual_loss +
            self.edge_scale * edge_loss
        )

        return total_loss, {
            "l1": l1_loss.item(),
            "ssim": ssim_loss.item(),
            "global_mean": mean_loss.item(),
            "neighbor_relative": brightness_consistency.item(),
            "perceptual": perceptual_loss.item(),
            "edge": edge_loss.item()
        }
    