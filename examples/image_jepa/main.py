"""
CIFAR-10 VICReg Training Script - Native PyTorch Implementation

This script implements VICReg training on CIFAR-10 dataset using only PyTorch and torchvision.
Supports both ResNet and Vision Transformer (ViT) backbones.

Usage:
    # With YAML config:
    python -m examples.image_jepa.main --fname examples/image_jepa/cfgs/default.yaml

    # With config + overrides:
    python -m examples.image_jepa.main --fname examples/image_jepa/cfgs/default.yaml optim.epochs=50
"""

import os
import time
from pathlib import Path

import fire
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import wandb
from omegaconf import OmegaConf
from torch.amp import GradScaler, autocast
from torch.optim.optimizer import required
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.models import VisionTransformer
from tqdm import tqdm

from eb_jepa.logging import get_logger
from eb_jepa.losses import BCS, VICRegLoss
from eb_jepa.training_utils import (
    get_default_dev_name,
    get_exp_name,
    get_unified_experiment_dir,
    load_checkpoint,
    load_config,
    log_config,
    log_data_info,
    log_epoch,
    log_model_info,
    save_checkpoint,
    setup_device,
    setup_seed,
    setup_wandb,
)
from examples.image_jepa.dataset import (
    ImageDataset,
    get_train_transforms,
    get_val_transforms,
)
from examples.image_jepa.eval import LinearProbe, evaluate_linear_probe

logger = get_logger(__name__)


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
    dtype=torch.float16,
    tqdm_silent=False,
):
    """Train for one epoch."""
    model.train()
    linear_probe.train()

    # Dynamic loss accumulator
    loss_totals = {}
    total_linear_loss = 0
    linear_correct = 0
    linear_total = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}", disable=tqdm_silent)
    for batch_idx, (views, target) in enumerate(pbar):
        view1, view2 = views[0].to(device, non_blocking=True), views[1].to(
            device, non_blocking=True
        )
        target = target.to(device, non_blocking=True)

        with autocast(device.type, enabled=use_amp, dtype=dtype):
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


def run(
    fname: str = "examples/image_jepa/cfgs/default.yaml",
    cfg=None,
    folder=None,
    **overrides,
):
    """
    Train an Image JEPA (VICReg/BCS) model on CIFAR-10.

    Args:
        fname: Path to YAML config file
        cfg: Pre-loaded config object (optional, overrides config file)
        folder: Experiment folder path (optional, auto-generated if not provided)
        **overrides: Config overrides in dot notation (e.g., optim.epochs=50)
    """
    # Load config
    if cfg is None:
        cfg = load_config(fname, overrides if overrides else None)

    # Setup using shared utilities
    device = setup_device(cfg.meta.device)
    setup_seed(cfg.meta.seed)

    # Create experiment directory using unified structure (if not provided)
    if folder is None:
        if cfg.meta.get("model_folder"):
            exp_dir = Path(cfg.meta.model_folder)
            folder_name = exp_dir.name
            exp_name = folder_name.rsplit("_seed", 1)[0]
        else:
            sweep_name = get_default_dev_name()
            exp_name = get_exp_name("image_jepa", cfg)
            exp_dir = get_unified_experiment_dir(
                example_name="image_jepa",
                sweep_name=sweep_name,
                exp_name=exp_name,
                seed=cfg.meta.seed,
            )
    else:
        exp_dir = Path(folder)
        exp_dir.mkdir(parents=True, exist_ok=True)
        # Extract exp_name from folder name by removing _seed{seed} suffix
        folder_name = exp_dir.name  # e.g., "resnet_vicreg_seed1"
        exp_name = folder_name.rsplit("_seed", 1)[0]  # e.g., "resnet_vicreg"

    wandb_run = setup_wandb(
        project="eb_jepa",
        config={"example": "image_jepa", **OmegaConf.to_container(cfg, resolve=True)},
        run_dir=exp_dir,
        run_name=exp_name,
        tags=["image_jepa", f"seed_{cfg.meta.seed}"],
        group=cfg.logging.get("wandb_group"),
        enabled=cfg.logging.log_wandb,
        sweep_id=cfg.logging.get("wandb_sweep_id"),
    )

    logger.info("Loading CIFAR-10 dataset...")
    transform = get_train_transforms()

    # Use EBJEPA_DSETS environment variable if set, otherwise fall back to config
    data_dir = os.environ.get("EBJEPA_DSETS")
    logger.info(f"Using data directory: {data_dir}")

    base_train_dataset = CIFAR10(
        root=data_dir, train=True, download=True, transform=None
    )

    train_dataset = ImageDataset(base_train_dataset, transform, num_crops=2)

    val_dataset = CIFAR10(
        root=data_dir, train=False, download=True, transform=get_val_transforms()
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        drop_last=True,  # Avoid small batches that cause BatchNorm issues
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
    )

    log_data_info(
        "CIFAR-10",
        len(train_loader),
        cfg.data.batch_size,
        train_samples=len(train_dataset),
        val_samples=len(val_dataset),
    )

    # Initialize model
    logger.info("Initializing model...")
    if cfg.model.type == "resnet":
        backbone = ResNet18()
        features_dim = backbone.features_dim
    elif cfg.model.type == "vit_s":
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
    elif cfg.model.type == "vit_b":
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
        proj_hidden_dim=cfg.model.proj_hidden_dim,
        proj_output_dim=cfg.model.proj_output_dim,
    )

    if not cfg.model.use_projector:
        model.projector = nn.Identity()

    model = model.to(device)

    # Log model structure and parameters
    encoder_params = sum(p.numel() for p in backbone.parameters())
    projector_params = (
        sum(p.numel() for p in model.projector.parameters())
        if cfg.model.use_projector
        else 0
    )
    log_model_info(model, {"encoder": encoder_params, "projector": projector_params})

    # Log configuration
    log_config(cfg)

    # Initialize linear probe
    linear_probe = LinearProbe(feature_dim=features_dim, num_classes=10).to(device)

    # Mixed precision setup
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16}
    use_amp = cfg.training.get("use_amp", True)
    dtype = dtype_map.get(cfg.training.get("dtype", "float16").lower(), torch.float16)
    scaler = GradScaler(device.type, enabled=use_amp)
    logger.info(f"Using AMP with {dtype=}" if use_amp else f"AMP disabled")

    optimizer = LARS(
        [
            {"params": model.parameters(), "lr": cfg.optim.lr},
            {"params": linear_probe.parameters(), "lr": 0.1},  # Linear probe parameters
        ],
        weight_decay=cfg.optim.weight_decay,
        eta=0.02,
        clip_lr=True,
        exclude_bias_n_norm=True,
        momentum=0.9,
    )

    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=cfg.optim.warmup_epochs,
        max_epochs=cfg.optim.epochs,
        base_lr=cfg.optim.lr,
        min_lr=cfg.optim.min_lr,
        warmup_start_lr=cfg.optim.warmup_start_lr,
    )

    # Initialize loss function
    if cfg.loss.type == "vicreg":
        loss_fn = VICRegLoss(std_coeff=cfg.loss.std_coeff, cov_coeff=cfg.loss.cov_coeff)
    elif cfg.loss.type == "bcs":
        loss_fn = BCS(lmbd=cfg.loss.lmbd)

    # Load checkpoint if requested
    start_epoch = 0
    if cfg.meta.get("load_model"):
        ckpt_path = exp_dir / cfg.meta.get("load_checkpoint", "latest.pth.tar")
        ckpt_info = load_checkpoint(ckpt_path, model, optimizer, device=device)
        start_epoch = ckpt_info.get("epoch", 0)
        if "linear_probe_state_dict" in ckpt_info:
            linear_probe.load_state_dict(ckpt_info["linear_probe_state_dict"])

    # Training loop
    logger.info(f"Starting training for {cfg.optim.epochs} epochs...")
    start_time = time.time()
    use_amp = cfg.training.get("use_amp", True)
    tqdm_silent = cfg.logging.get("tqdm_silent", False)

    for epoch in range(start_epoch, cfg.optim.epochs):
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
            use_amp,
            dtype,
            tqdm_silent,
        )

        # Evaluate linear probe on validation set
        val_acc, val_loss = evaluate_linear_probe(
            model, linear_probe, val_loader, device, use_amp
        )

        # Log metrics - dynamically add train_ prefix to all train_metrics keys
        log_dict = {"epoch": epoch}
        for key, value in train_metrics.items():
            log_dict[f"train_{key}"] = value
        log_dict["val_loss"] = val_loss
        log_dict["val_acc"] = val_acc
        log_dict["learning_rate"] = optimizer.param_groups[0]["lr"]

        if wandb_run:
            wandb.log(log_dict)

        # Log progress
        if epoch % cfg.logging.log_every == 0:
            elapsed = time.time() - start_time
            log_epoch(
                epoch,
                {
                    "loss": train_metrics["loss"],
                    "val_acc": val_acc,
                    "lr": optimizer.param_groups[0]["lr"],
                },
                total_epochs=cfg.optim.epochs,
                elapsed_time=elapsed,
            )

        # Save checkpoint
        save_checkpoint(
            exp_dir / "latest.pth.tar",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            scaler=scaler,
            linear_probe_state_dict=linear_probe.state_dict(),
            linear_val_acc=val_acc,
        )
        if epoch % cfg.logging.save_every == 0 and epoch > 0:
            save_checkpoint(
                exp_dir / f"epoch_{epoch}.pth.tar",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                scaler=scaler,
                linear_probe_state_dict=linear_probe.state_dict(),
                linear_val_acc=val_acc,
            )

    logger.info("Training completed!")
    if wandb_run:
        wandb.finish()


if __name__ == "__main__":
    fire.Fire(run)
