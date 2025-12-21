#!/usr/bin/env python3
"""
CIFAR-10 VICReg Training Script - Native PyTorch Implementation

This script implements VICReg training on CIFAR-10 dataset using only PyTorch and torchvision.
"""

import os
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.optim.optimizer import required
import wandb
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler


class ResNet18(nn.Module):
    """ResNet-18 backbone implementation."""
    
    def __init__(self, num_classes=10):
        super().__init__()
        self.backbone = torchvision.models.resnet18(pretrained=False)
        self.backbone.fc = nn.Identity()  # Remove final classification layer
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False)
        self.backbone.maxpool = nn.Identity()
        self.features_dim = 512

        
    def forward(self, x):
        return self.backbone(x)


class VICReg(nn.Module):
    """VICReg model implementation."""
    
    def __init__(self, backbone, proj_hidden_dim=2048, proj_output_dim=2048):
        super().__init__()
        self.backbone = backbone
        self.features_dim = backbone.features_dim
        
        # Projector
        self.projector = nn.Sequential(
            nn.Linear(self.features_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_output_dim),
        )
        
    def forward(self, x):
        features = self.backbone(x)
        projections = self.projector(features)
        return features, projections


def vicreg_loss(z1, z2, sim_loss_weight=25.0, var_loss_weight=25.0, cov_loss_weight=1.0):
    """VICReg loss function."""
    batch_size = z1.size(0)
    
    # Invariance loss (similarity)
    sim_loss = F.mse_loss(z1, z2)
    
    # Variance loss
    z1_std = torch.sqrt(z1.var(dim=0) + 1e-4)
    z2_std = torch.sqrt(z2.var(dim=0) + 1e-4)
    var_loss = torch.mean(F.relu(1 - z1_std)) + torch.mean(F.relu(1 - z2_std))
    
    # Covariance loss
    z1_centered = z1 - z1.mean(dim=0)
    z2_centered = z2 - z2.mean(dim=0)
    z1_cov = torch.mm(z1_centered.T, z1_centered) / (batch_size - 1)
    z2_cov = torch.mm(z2_centered.T, z2_centered) / (batch_size - 1)
    
    cov_loss = (z1_cov.pow(2).sum() - z1_cov.diagonal().pow(2).sum()) / z1_cov.size(0) + \
               (z2_cov.pow(2).sum() - z2_cov.diagonal().pow(2).sum()) / z2_cov.size(0)
    
    total_loss = sim_loss_weight * sim_loss + var_loss_weight * var_loss + cov_loss_weight * cov_loss
    
    return total_loss, sim_loss, var_loss, cov_loss


class RandomResizedCrop:
    """Random resized crop augmentation."""
    
    def __init__(self, size, scale=(0.2, 1.0)):
        self.size = size
        self.scale = scale
        
    def __call__(self, img):
        return transforms.RandomResizedCrop(self.size, scale=self.scale)(img)


class ColorJitter:
    """Color jitter augmentation."""
    
    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, prob=0.8):
        self.transform = transforms.ColorJitter(brightness, contrast, saturation, hue)
        self.prob = prob
        
    def __call__(self, img):
        if torch.rand(1) < self.prob:
            return self.transform(img)
        return img


class Grayscale:
    """Grayscale augmentation."""
    
    def __init__(self, prob=0.2):
        self.prob = prob
        
    def __call__(self, img):
        if torch.rand(1) < self.prob:
            return transforms.Grayscale(num_output_channels=3)(img)
        return img


class Solarization:
    """Solarization augmentation."""
    
    def __init__(self, prob=0.1):
        self.prob = prob
        
    def __call__(self, img):
        if torch.rand(1) < self.prob:
            img = transforms.functional.solarize(img, threshold=128)
        return img


class HorizontalFlip:
    """Horizontal flip augmentation."""
    
    def __init__(self, prob=0.5):
        self.prob = prob
        
    def __call__(self, img):
        if torch.rand(1) < self.prob:
            return transforms.functional.hflip(img)
        return img


def get_augmentations():
    """Get training augmentations for VICReg (same as original solo-learn)."""
    # Single augmentation pipeline that will be applied multiple times
    # This matches the original solo-learn implementation exactly
    transform = transforms.Compose([
        RandomResizedCrop(32, scale=(0.2, 1.0)),
        ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, prob=0.8),
        Grayscale(prob=0.2),
        Solarization(prob=0.1),
        HorizontalFlip(prob=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    return transform


def get_val_transforms():
    """Get validation transforms."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])


class VICRegDataset(torch.utils.data.Dataset):
    """Custom dataset for VICReg that applies the same augmentation multiple times."""
    
    def __init__(self, dataset, transform, num_crops=2):
        self.dataset = dataset
        self.transform = transform
        self.num_crops = num_crops
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        
        # Apply the same transformation multiple times to create different views
        # This matches the NCropAugmentation behavior from solo-learn
        views = [self.transform(image) for _ in range(self.num_crops)]
        
        return views, label


class LARS(optim.Optimizer):
    """LARS optimizer implementation - identical to solo-learn."""
    
    def __init__(
        self,
        params,
        lr=required,
        momentum=0,
        dampening=0,
        weight_decay=0,
        nesterov=False,
        eta=1e-3,
        eps=1e-8,
        clip_lr=False,
        exclude_bias_n_norm=False,
    ):
        if lr is not required and lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            eta=eta,
            eps=eps,
            clip_lr=clip_lr,
            exclude_bias_n_norm=exclude_bias_n_norm,
        )
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")

        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)

        for group in self.param_groups:
            group.setdefault("nesterov", False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # exclude scaling for params with 0 weight decay
        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            dampening = group["dampening"]
            nesterov = group["nesterov"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                d_p = p.grad
                p_norm = torch.norm(p.data)
                g_norm = torch.norm(p.grad.data)

                # lars scaling + weight decay part
                if p.ndim != 1 or not group["exclude_bias_n_norm"]:
                    if p_norm != 0 and g_norm != 0:
                        lars_lr = p_norm / (g_norm + p_norm * weight_decay + group["eps"])
                        lars_lr *= group["eta"]

                        # clip lr
                        if group["clip_lr"]:
                            lars_lr = min(lars_lr / group["lr"], 1)

                        d_p = d_p.add(p, alpha=weight_decay)
                        d_p *= lars_lr

                # sgd part
                if momentum != 0:
                    param_state = self.state[p]
                    if "momentum_buffer" not in param_state:
                        buf = param_state["momentum_buffer"] = torch.clone(d_p).detach()
                    else:
                        buf = param_state["momentum_buffer"]
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf

                p.add_(d_p, alpha=-group["lr"])

        return loss


class WarmupCosineScheduler:
    """Warmup cosine learning rate scheduler - consistent with solo-learn implementation."""
    
    def __init__(self, optimizer, warmup_epochs, max_epochs, base_lr, min_lr=0.0, warmup_start_lr=3e-5):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.warmup_start_lr = warmup_start_lr
        
    def step(self, epoch):
        if epoch < self.warmup_epochs:
            # Linear warmup: warmup_start_lr -> base_lr (consistent with solo-learn)
            lr = self.warmup_start_lr + epoch * (self.base_lr - self.warmup_start_lr) / (self.warmup_epochs - 1)
        else:
            # Cosine annealing: base_lr -> min_lr (consistent with solo-learn)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + torch.cos(torch.tensor((epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs) * 3.14159)))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


class LinearProbe(nn.Module):
    """Linear probe classifier for evaluating representations."""
    
    def __init__(self, feature_dim, num_classes):
        super().__init__()
        self.classifier = nn.Linear(feature_dim, num_classes)
        
    def forward(self, x):
        return self.classifier(x)


def extract_features(model, dataloader, device):
    """Extract features from the model for all samples in the dataloader."""
    model.eval()
    features = []
    labels = []
    
    with torch.no_grad():
        for data, target in tqdm(dataloader, desc="Extracting features"):
            data = data.to(device)
            target = target.to(device)
            
            # Extract features (not projections)
            feat, _ = model(data)
            features.append(feat.cpu())
            labels.append(target.cpu())
    
    features = torch.cat(features, dim=0)
    labels = torch.cat(labels, dim=0)
    
    return features, labels

def evaluate_linear_probe(model, linear_probe, val_loader, device):
    """Evaluate linear probe on validation set."""
    model.eval()
    linear_probe.eval()
    
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            
            # Mixed precision forward pass
            with autocast():
                # Extract features
                features, _ = model(data)
            
            # Forward pass through linear probe (NOT under autocast)
            # Convert features to float32 to match linear probe dtype
            outputs = linear_probe(features.float())
            loss = F.cross_entropy(outputs, target)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    accuracy = 100.0 * correct / total
    avg_loss = total_loss / len(val_loader)
    
    return accuracy, avg_loss


def train_epoch(model, train_loader, optimizer, scheduler, linear_probe, scaler, device, epoch, sim_loss_weight=25.0, var_loss_weight=25.0, cov_loss_weight=1.0):
    """Train for one epoch."""
    model.train()
    linear_probe.train()
    
    total_loss = 0
    total_sim_loss = 0
    total_var_loss = 0
    total_cov_loss = 0
    total_linear_loss = 0
    linear_correct = 0
    linear_total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch_idx, (views, target) in enumerate(pbar):
        # views is a list of tensors, extract the two views
        view1, view2 = views[0].to(device, non_blocking=True), views[1].to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        
        # Mixed precision forward pass
        with autocast():
            # Forward pass for VICReg
            features, z1 = model(view1)
            _, z2 = model(view2)
            
            # Compute VICReg loss
            vc_loss, sim_loss, var_loss, cov_loss = vicreg_loss(z1, z2, sim_loss_weight, var_loss_weight, cov_loss_weight)
        
        # Linear probe training (frozen encoder) - NOT under autocast in Lightning
        with torch.no_grad():
            # Extract features without gradients and convert to float32
            features_frozen = features.detach().float()
        
        # Forward pass for linear probe (NOT under autocast in Lightning)
        linear_outputs = linear_probe(features_frozen)
        linear_loss = F.cross_entropy(linear_outputs, target)
        
        # Accuracy computation (NOT under autocast in Lightning)
        _, predicted = linear_outputs.max(1)
        linear_correct_batch = predicted.eq(target).sum().item()
        
        # Combined loss
        total_loss_batch = vc_loss + linear_loss
        
        # Mixed precision backward pass
        optimizer.zero_grad()
        scaler.scale(total_loss_batch).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Update metrics
        total_loss += vc_loss.item()
        total_sim_loss += sim_loss.item()
        total_var_loss += var_loss.item()
        total_cov_loss += cov_loss.item()
        total_linear_loss += linear_loss.item()
        
        # Update linear probe accuracy (pre-computed under autocast)
        linear_total += target.size(0)
        linear_correct += linear_correct_batch
        
        # Update progress bar
        pbar.set_postfix({
            'VICReg': f'{vc_loss.item():.4f}',
            'Linear': f'{linear_loss.item():.4f}',
            'Acc': f'{100.*linear_correct/linear_total:.2f}%'
        })
    
    # Update learning rate
    scheduler.step(epoch)
    
    return {
        'loss': total_loss / len(train_loader),
        'sim_loss': total_sim_loss / len(train_loader),
        'var_loss': total_var_loss / len(train_loader),
        'cov_loss': total_cov_loss / len(train_loader),
        'linear_loss': total_linear_loss / len(train_loader),
        'linear_acc': 100.0 * linear_correct / linear_total
    }


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='VICReg Training on CIFAR-10')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=300, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--warmup_epochs', type=int, default=10, help='Number of warmup epochs')
    parser.add_argument('--warmup_start_lr', type=float, default=3e-5, help='Starting learning rate for warmup')
    
    # Model parameters
    parser.add_argument('--proj_hidden_dim', type=int, default=2048, help='Projector hidden dimension')
    parser.add_argument('--proj_output_dim', type=int, default=2048, help='Projector output dimension')
    parser.add_argument('--use_projector', type=int, default=1, help='Whether to use projector (default: True)')
    
    # Loss weights
    parser.add_argument('--sim_loss_weight', type=float, default=25.0, help='Similarity loss weight')
    parser.add_argument('--var_loss_weight', type=float, default=25.0, help='Variance loss weight')
    parser.add_argument('--cov_loss_weight', type=float, default=1.0, help='Covariance loss weight')
    
    # Data parameters
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')
    parser.add_argument('--data_dir', type=str, default='./datasets', help='Directory to store datasets')
    
    # Logging parameters
    parser.add_argument('--project_name', type=str, default='solo-learn', help='Wandb project name')
    parser.add_argument('--run_name', type=str, default='vicreg-cifar10-scratch', help='Wandb run name')
    parser.add_argument('--log_interval', type=int, default=10, help='Logging interval in epochs')
    parser.add_argument('--save_interval', type=int, default=50, help='Checkpoint saving interval in epochs')
    
    # Other parameters
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (auto, cuda, cpu)')
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f'Using device: {device}')
    
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Initialize wandb
    wandb.init(
        project=args.project_name,
        config=vars(args),
        name=args.run_name
    )
    
    # Load datasets
    print("Loading CIFAR-10 dataset...")
    # Get single augmentation pipeline (same as original solo-learn)
    transform = get_augmentations()
    
    # Base dataset without transforms
    base_train_dataset = CIFAR10(
        root=args.data_dir,
        train=True,
        download=True,
        transform=None
    )
    
    # Create VICReg dataset with multiple crops (same as NCropAugmentation)
    train_dataset = VICRegDataset(base_train_dataset, transform, num_crops=2)
    
    val_dataset = CIFAR10(
        root=args.data_dir,
        train=False,
        download=True,
        transform=get_val_transforms()
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Initialize model
    print("Initializing model...")
    backbone = ResNet18(num_classes=10)
    
    if args.use_projector:
        model = VICReg(backbone, proj_hidden_dim=args.proj_hidden_dim, proj_output_dim=args.proj_output_dim)
    else:
        # Use identity projector (no transformation)
        model = VICReg(backbone, proj_hidden_dim=args.proj_hidden_dim, proj_output_dim=args.proj_output_dim)
        model.projector = nn.Identity()
    
    model = model.to(device)
    
    # Initialize linear probe
    linear_probe = LinearProbe(feature_dim=512, num_classes=10).to(device)
    
    # Initialize mixed precision scaler
    scaler = GradScaler()
    
    # Initialize optimizer with both model and linear probe parameters
    optimizer = LARS(
        [
            {'params': model.parameters(), 'lr': 0.3},
            {'params': linear_probe.parameters(), 'lr': 0.1}
        ],
        weight_decay=1e-4,
        eta=0.02,
        clip_lr=True,
        exclude_bias_n_norm=True,  # Excludes 1D params from LARS scaling only
        momentum=0.9  # Consistent with solo-learn default for LARS
    )
    
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=args.warmup_epochs,
        max_epochs=args.epochs,
        base_lr=args.learning_rate,
        min_lr=0.0,
        warmup_start_lr=args.warmup_start_lr
    )
    
    # Training loop
    print("Starting training...")
    start_time = time.time()
    
    for epoch in range(args.epochs):
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, scheduler, linear_probe, scaler, device, epoch, 
                                   args.sim_loss_weight, args.var_loss_weight, args.cov_loss_weight)
        
        # Evaluate linear probe on validation set
        val_acc, val_loss = evaluate_linear_probe(model, linear_probe, val_loader, device)
        
        # Log metrics
        log_dict = {
            'epoch': epoch,
            'train_loss': train_metrics['loss'],
            'train_sim_loss': train_metrics['sim_loss'],
            'train_var_loss': train_metrics['var_loss'],
            'train_cov_loss': train_metrics['cov_loss'],
            'linear_train_loss': train_metrics['linear_loss'],
            'linear_train_acc': train_metrics['linear_acc'],
            'linear_val_loss': val_loss,
            'linear_val_acc': val_acc,
            'learning_rate': optimizer.param_groups[0]['lr']
        }
        
        wandb.log(log_dict)
        
        # Print progress
        if epoch % args.log_interval == 0:
            elapsed = time.time() - start_time
            print(f'Epoch {epoch:4d} | VICReg: {train_metrics["loss"]:.4f} | '
                  f'Linear Train: {train_metrics["linear_acc"]:.2f}% | '
                  f'Linear Val: {val_acc:.2f}% | '
                  f'LR: {optimizer.param_groups[0]["lr"]:.6f} | '
                  f'Time: {elapsed:.1f}s')
        
        # Save checkpoint
        if epoch % args.save_interval == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'linear_probe_state_dict': linear_probe.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'scheduler_state_dict': scheduler,
                'loss': train_metrics['loss'],
                'linear_val_acc': val_acc
            }
            os.makedirs('trained_models/vicreg', exist_ok=True)
            torch.save(checkpoint, f'trained_models/vicreg/checkpoint_epoch_{epoch}.pth')
    
    print("Training completed!")
    wandb.finish()


if __name__ == "__main__":
    main()
