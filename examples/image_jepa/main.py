"""
CIFAR-10 VICReg Training Script - Native PyTorch Implementation

This script implements VICReg training on CIFAR-10 dataset using only PyTorch and torchvision.
Supports both ResNet and Vision Transformer (ViT) backbones.
"""

import argparse
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import wandb
from dataset import ImageDataset, get_train_transforms, get_val_transforms
from eval import LinearProbe, evaluate_linear_probe
from torch.amp import GradScaler, autocast
from torch.optim.optimizer import required
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.models import VisionTransformer
from tqdm import tqdm

from eb_jepa.losses import BCS, VICRegLoss


class ResNet18(nn.Module):
    """ResNet-18 backbone implementation."""

    def __init__(self):
        super().__init__()
        self.backbone = torchvision.models.resnet18()
        self.backbone.fc = nn.Identity()  # Remove final classification layer
        self.backbone.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=2, bias=False
        )
        self.backbone.maxpool = nn.Identity()
        self.features_dim = 512

    def forward(self, x):
        return self.backbone(x)


class ImageSSL(nn.Module):
    """Image Self-Supervised Learning model implementation."""

    def __init__(
        self, backbone, features_dim, proj_hidden_dim=2048, proj_output_dim=2048
    ):
        super().__init__()
        self.backbone = backbone
        self.features_dim = features_dim

        # Projector
        self.projector = nn.Sequential(
            nn.Linear(features_dim, proj_hidden_dim),
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


class LARS(optim.Optimizer):
    """LARS optimizer implementation."""

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
                        lars_lr = p_norm / (
                            g_norm + p_norm * weight_decay + group["eps"]
                        )
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
    """Warmup cosine learning rate scheduler"""

    def __init__(
        self,
        optimizer,
        warmup_epochs,
        max_epochs,
        base_lr,
        min_lr=0.0,
        warmup_start_lr=3e-5,
    ):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.warmup_start_lr = warmup_start_lr

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            lr = self.warmup_start_lr + epoch * (
                self.base_lr - self.warmup_start_lr
            ) / (self.warmup_epochs - 1)
        else:
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
                1
                + torch.cos(
                    torch.tensor(
                        (epoch - self.warmup_epochs)
                        / (self.max_epochs - self.warmup_epochs)
                        * 3.14159
                    )
                )
            )

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr


def train_epoch(
    model,
    train_loader,
    optimizer,
    scheduler,
    linear_probe,
    scaler,
    device,
    epoch,
    loss_fn,
    use_amp=True,
):
    """Train for one epoch."""
    model.train()
    linear_probe.train()

    # Dynamic loss accumulator
    loss_totals = {}
    total_linear_loss = 0
    linear_correct = 0
    linear_total = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for batch_idx, (views, target) in enumerate(pbar):
        view1, view2 = views[0].to(device, non_blocking=True), views[1].to(
            device, non_blocking=True
        )
        target = target.to(device, non_blocking=True)

        with autocast("cuda", enabled=use_amp):
            features, z1 = model(view1)
            _, z2 = model(view2)
            loss_dict = loss_fn(z1, z2)
            loss = loss_dict["loss"]

        with torch.no_grad():
            features_frozen = features.detach().float()

        linear_outputs = linear_probe(features_frozen)
        linear_loss = F.cross_entropy(linear_outputs, target)

        _, predicted = linear_outputs.max(1)
        linear_correct_batch = predicted.eq(target).sum().item()

        total_loss_batch = loss + linear_loss

        optimizer.zero_grad()
        scaler.scale(total_loss_batch).backward()
        scaler.step(optimizer)
        scaler.update()

        # Update metrics dynamically based on loss_dict keys
        for key, value in loss_dict.items():
            if key not in loss_totals:
                loss_totals[key] = 0
            loss_totals[key] += value.item()
        total_linear_loss += linear_loss.item()

        # Update linear probe accuracy (pre-computed under autocast)
        linear_total += target.size(0)
        linear_correct += linear_correct_batch

        # Update progress bar
        pbar.set_postfix(
            {
                "Loss": f"{loss.item():.4f}",
                "Linear": f"{linear_loss.item():.4f}",
                "Acc": f"{100.*linear_correct/linear_total:.2f}%",
            }
        )

    # Update learning rate
    scheduler.step(epoch)

    # Build return dict dynamically
    num_batches = len(train_loader)
    metrics = {key: total / num_batches for key, total in loss_totals.items()}
    metrics["linear_loss"] = total_linear_loss / num_batches
    metrics["linear_acc"] = 100.0 * linear_correct / linear_total

    return metrics


def create_base_parser(description="Image SSL Training on CIFAR-10"):
    """Create base argument parser with common arguments.

    This can be extended by other scripts to add their own arguments.
    """
    parser = argparse.ArgumentParser(description=description)

    # Training parameters
    parser.add_argument(
        "--batch_size", type=int, default=256, help="Batch size for training"
    )
    parser.add_argument(
        "--epochs", type=int, default=300, help="Number of training epochs"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.3, help="Learning rate"
    )
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument(
        "--warmup_epochs", type=int, default=10, help="Number of warmup epochs"
    )
    parser.add_argument(
        "--warmup_start_lr",
        type=float,
        default=3e-5,
        help="Starting learning rate for warmup",
    )

    # Model parameters
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["resnet", "vit_s", "vit_b"],
        default="resnet",
        help="Type of encoder",
    )
    parser.add_argument("--patch_size", type=int, default=2, help="Patch size for ViT")
    parser.add_argument(
        "--proj_hidden_dim", type=int, default=2048, help="Projector hidden dimension"
    )
    parser.add_argument(
        "--proj_output_dim", type=int, default=2048, help="Projector output dimension"
    )
    parser.add_argument(
        "--use_projector",
        type=int,
        default=1,
        help="Whether to use projector (default: True)",
    )

    # Data parameters
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of data loader workers"
    )
    parser.add_argument(
        "--data_dir", type=str, default="./datasets", help="Directory to store datasets"
    )

    # Logging parameters
    parser.add_argument(
        "--project_name",
        type=str,
        default="eb-jepa-image-ssl",
        help="Wandb project name",
    )
    parser.add_argument(
        "--log_interval", type=int, default=10, help="Logging interval in epochs"
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=50,
        help="Checkpoint saving interval in epochs",
    )

    # Other parameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--device", type=str, default="auto", help="Device to use (auto, cuda, cpu)"
    )
    parser.add_argument(
        "--no_amp",
        action="store_true",
        help="Disable mixed precision training (enabled by default)",
    )

    return parser


def parse_args():
    """Parse command line arguments."""
    parser = create_base_parser()

    # Loss function selection
    parser.add_argument(
        "--loss_type",
        type=str,
        choices=["vicreg", "bcs"],
        default="vicreg",
        help="Loss function type",
    )

    # VICReg-specific loss weights
    parser.add_argument(
        "--var_loss_weight",
        type=float,
        default=1.0,
        help="Variance loss weight (VICReg)",
    )
    parser.add_argument(
        "--cov_loss_weight",
        type=float,
        default=80.0,
        help="Covariance loss weight (VICReg)",
    )

    # BCS-specific loss weight
    parser.add_argument(
        "--lmbd", type=float, default=10.0, help="BCS loss weight (LE-JEPA)"
    )

    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()

    # Print all hyperparameters for run identification in logs
    print("=" * 60)
    print("Run Configuration:")
    print("=" * 60)
    for key, value in sorted(vars(args).items()):
        print(f"  {key}={value}")
    print("=" * 60)

    # Set device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    wandb.init(
        project=args.project_name,
        config=vars(args),
        name=f"{args.model_type}-{args.loss_type}-{args.seed}",
    )

    print("Loading CIFAR-10 dataset...")
    transform = get_train_transforms()

    base_train_dataset = CIFAR10(
        root=args.data_dir, train=True, download=True, transform=None
    )

    train_dataset = ImageDataset(base_train_dataset, transform, num_crops=2)

    val_dataset = CIFAR10(
        root=args.data_dir, train=False, download=True, transform=get_val_transforms()
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,  # Avoid small batches that cause BatchNorm issues
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Initialize model
    print("Initializing model...")
    if args.model_type == "resnet":
        backbone = ResNet18()
        features_dim = backbone.features_dim
    elif args.model_type == "vit_s":
        features_dim = 384
        model_kwargs = dict(
            image_size=32,
            patch_size=8,
            hidden_dim=features_dim,
            num_layers=12,
            num_heads=6,
            mlp_dim=4 * features_dim,
        )
        backbone = VisionTransformer(**model_kwargs)
        backbone.heads = nn.Identity()
    elif args.model_type == "vit_b":
        features_dim = 768
        model_kwargs = dict(
            image_size=32,
            patch_size=8,
            hidden_dim=features_dim,
            num_layers=12,
            num_heads=12,
            mlp_dim=4 * features_dim,
        )
        backbone = VisionTransformer(**model_kwargs)
        backbone.heads = nn.Identity()

    model = ImageSSL(
        backbone,
        features_dim=features_dim,
        proj_hidden_dim=args.proj_hidden_dim,
        proj_output_dim=args.proj_output_dim,
    )

    if not args.use_projector:
        model.projector = nn.Identity()

    model = model.to(device)

    # Initialize linear probe
    linear_probe = LinearProbe(feature_dim=features_dim, num_classes=10).to(device)

    # Initialize mixed precision scaler
    scaler = GradScaler("cuda")

    optimizer = LARS(
        [
            {"params": model.parameters(), "lr": 0.3},
            {
                "params": linear_probe.parameters(),
                "lr": 0.1,
            },  # Initialize linear probe parameters
        ],
        weight_decay=1e-4,
        eta=0.02,
        clip_lr=True,
        exclude_bias_n_norm=True,
        momentum=0.9,
    )

    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=args.warmup_epochs,
        max_epochs=args.epochs,
        base_lr=args.learning_rate,
        min_lr=0.0,
        warmup_start_lr=args.warmup_start_lr,
    )

    # Initialize loss function
    if args.loss_type == "vicreg":
        loss_fn = VICRegLoss(
            var_loss_weight=args.var_loss_weight, cov_loss_weight=args.cov_loss_weight
        )
    elif args.loss_type == "bcs":
        loss_fn = BCS(lmbd=args.lmbd)

    # Training loop
    print("Starting training...")
    start_time = time.time()

    for epoch in range(args.epochs):
        # Train
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            linear_probe,
            scaler,
            device,
            epoch,
            loss_fn,
            not args.no_amp,
        )

        # Evaluate linear probe on validation set
        val_acc, val_loss = evaluate_linear_probe(
            model, linear_probe, val_loader, device, not args.no_amp
        )

        # Log metrics - dynamically add train_ prefix to all train_metrics keys
        log_dict = {"epoch": epoch}
        for key, value in train_metrics.items():
            log_dict[f"train_{key}"] = value
        log_dict["val_loss"] = val_loss
        log_dict["val_acc"] = val_acc
        log_dict["learning_rate"] = optimizer.param_groups[0]["lr"]

        wandb.log(log_dict)

        # Print progress
        if epoch % args.log_interval == 0:
            elapsed = time.time() - start_time
            metrics_str = " | ".join(f"{k}: {v:.4f}" for k, v in train_metrics.items())
            print(
                f"Epoch {epoch:4d} | {metrics_str} | "
                f"Linear Val: {val_acc:.2f}% | "
                f'LR: {optimizer.param_groups[0]["lr"]:.6f} | '
                f"Time: {elapsed:.1f}s"
            )

        # Save checkpoint
        if epoch % args.save_interval == 0:
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "linear_probe_state_dict": linear_probe.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "scheduler_state_dict": scheduler,
                "loss": train_metrics["loss"],
                "linear_val_acc": val_acc,
            }
            os.makedirs("examples/image_jepa/trained_models/", exist_ok=True)
            torch.save(
                checkpoint,
                f"examples/image_jepa/trained_models/checkpoint_epoch_{epoch}.pth",
            )

    print("Training completed!")
    wandb.finish()


if __name__ == "__main__":
    main()
