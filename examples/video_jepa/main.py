"""
Video JEPA Training Script

Train a self-supervised video prediction model on Moving MNIST using
Joint Embedding Predictive Architecture (JEPA) with VC regularization.
"""

from pathlib import Path

import fire
import torch.nn as nn
from omegaconf import OmegaConf
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from eb_jepa.architectures import (
    DetHead,
    Projector,
    ResNet5,
    ResUNet,
    StateOnlyPredictor,
)
from eb_jepa.datasets.moving_mnist import MovingMNISTDet
from eb_jepa.image_decoder import ImageDecoder
from eb_jepa.jepa import JEPA, JEPAProbe
from eb_jepa.logging import get_logger
from eb_jepa.losses import SquareLossSeq, VCLoss
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
from examples.video_jepa.eval import validation_loop

logger = get_logger(__name__)


def run(
    fname: str = "examples/video_jepa/cfgs/default.yaml",
    cfg=None,
    folder=None,
    **overrides,
):
    """
    Train a Video JEPA model on Moving MNIST.

    Args:
        fname: Path to YAML config file
        cfg: Pre-loaded config object (optional, overrides config file)
        folder: Experiment folder path (optional, auto-generated if not provided)
        **overrides: Config overrides in dot notation (e.g., model.lr=0.001)
    """
    # Load config
    if cfg is None:
        cfg = load_config(fname, overrides if overrides else None)

    # Setup
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
            exp_name = get_exp_name("video_jepa", cfg)
            exp_dir = get_unified_experiment_dir(
                example_name="video_jepa",
                sweep_name=sweep_name,
                exp_name=exp_name,
                seed=cfg.meta.seed,
            )
    else:
        exp_dir = Path(folder)
        exp_dir.mkdir(parents=True, exist_ok=True)
        # Extract exp_name from folder name by removing _seed{seed} suffix
        folder_name = exp_dir.name  # e.g., "resnet_std10.0_cov100.0_seed1"
        exp_name = folder_name.rsplit("_seed", 1)[0]  # e.g., "resnet_std10.0_cov100.0"

    wandb_run = setup_wandb(
        project="eb_jepa",
        config={"example": "video_jepa", **OmegaConf.to_container(cfg, resolve=True)},
        run_dir=exp_dir,
        run_name=exp_name,
        tags=["video_jepa", f"seed_{cfg.meta.seed}"],
        group=cfg.logging.get("wandb_group"),
        enabled=cfg.logging.log_wandb,
        sweep_id=cfg.logging.get("wandb_sweep_id"),
    )

    # Load datasets
    train_set = MovingMNISTDet(split="train")
    val_set = MovingMNISTDet(split="val")
    train_loader = DataLoader(
        train_set,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
    )
    log_data_info(
        "MovingMNIST",
        len(train_loader),
        cfg.data.batch_size,
        train_samples=len(train_set),
        val_samples=len(val_set),
    )

    # Initialize Video JEPA model
    logger.info("Initializing model...")
    encoder = ResNet5(cfg.model.dobs, cfg.model.henc, cfg.model.dstc)
    predictor_model = ResUNet(2 * cfg.model.dstc, cfg.model.hpre, cfg.model.dstc)
    predictor = StateOnlyPredictor(predictor_model, context_length=2)
    projector = Projector(f"{cfg.model.dstc}-{cfg.model.dstc*4}-{cfg.model.dstc*4}")
    regularizer = VCLoss(cfg.loss.std_coeff, cfg.loss.cov_coeff, proj=projector)
    ploss = SquareLossSeq(projector)
    jepa = JEPA(encoder, encoder, predictor, regularizer, ploss).to(device)

    # Initialize decoder and detection head (for evaluation only)
    decoder = ImageDecoder(cfg.model.dstc, cfg.model.dobs)
    dethead = DetHead(cfg.model.dstc, cfg.model.hpre, cfg.model.dobs)
    pixel_decoder = JEPAProbe(jepa, decoder, nn.MSELoss()).to(device)
    detection_head = JEPAProbe(jepa, dethead, nn.BCELoss()).to(device)

    # Log model structure and parameters
    encoder_params = sum(p.numel() for p in encoder.parameters())
    predictor_params = sum(p.numel() for p in predictor.parameters())
    log_model_info(jepa, {"encoder": encoder_params, "predictor": predictor_params})

    jepa.train()
    detection_head.train()
    pixel_decoder.train()

    # Set learning rates for different components
    # Lower learning rate for pixel decoder to prevent overfitting
    optimizer = Adam(
        [
            {"params": jepa.parameters(), "lr": cfg.optim.lr},
            {"params": pixel_decoder.head.parameters(), "lr": cfg.optim.lr / 10},
            {"params": detection_head.head.parameters(), "lr": cfg.optim.lr},
        ]
    )

    # Log configuration
    log_config(cfg)

    # Load checkpoint if requested
    start_epoch = 0
    global_step = 0
    if cfg.meta.get("load_model"):
        ckpt_path = exp_dir / cfg.meta.get("load_checkpoint", "latest.pth.tar")
        ckpt_info = load_checkpoint(ckpt_path, jepa, optimizer, device=device)
        start_epoch = ckpt_info.get("epoch", 0)
        global_step = ckpt_info.get("step", 0)

    # Training loop
    logger.info(f"Starting training for {cfg.optim.epochs} epochs...")

    for epoch in range(start_epoch, cfg.optim.epochs):
        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}",
            disable=cfg.logging.get("tqdm_silent", False),
        )

        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            x = batch["video"]
            loc_map = batch["digit_location"]

            optimizer.zero_grad()
            _, (jepa_loss, regl, _, regldict, pl) = jepa.unroll(
                x,
                actions=None,
                nsteps=cfg.model.steps,
                unroll_mode="parallel",
                compute_loss=True,
                return_all_steps=False,
            )
            recon_loss = pixel_decoder(x, x)
            det_loss = detection_head(x, loc_map)
            total_loss = jepa_loss + recon_loss + det_loss

            total_loss.backward()
            optimizer.step()

            # Update progress bar
            pbar.set_postfix(
                {
                    "loss": f"{jepa_loss.item():.4f}",
                    "vc": f"{regl.item():.4f}",
                    "pred": f"{pl.item():.4f}",
                }
            )

            global_step += 1

        # Validation and logging
        if epoch % cfg.logging.log_every == 0:
            val_logs = validation_loop(
                val_loader, jepa, detection_head, pixel_decoder, cfg.model.steps, device
            )

            train_metrics = {
                "epoch": epoch,
                "train/loss": jepa_loss.item(),
                "train/vc_loss": regl.item(),
                "train/pred_loss": pl.item(),
                "train/recon_loss": recon_loss.item(),
                "train/det_loss": det_loss.item(),
            }
            for k, v in regldict.items():
                train_metrics[f"train/{k}"] = float(v)

            all_metrics = {**train_metrics, **val_logs}

            if wandb_run:
                import wandb

                wandb.log(all_metrics, step=global_step)

            log_epoch(
                epoch,
                {
                    "loss": jepa_loss.item(),
                    "vc": regl.item(),
                    "pred": pl.item(),
                    "val_recon": val_logs.get("val/recon_loss", 0),
                },
                total_epochs=cfg.optim.epochs,
            )

        # Save checkpoint
        save_checkpoint(
            exp_dir / "latest.pth.tar",
            model=jepa,
            optimizer=optimizer,
            epoch=epoch,
            step=global_step,
        )
        if epoch % cfg.logging.save_every == 0 and epoch > 0:
            save_checkpoint(
                exp_dir / f"epoch_{epoch}.pth.tar",
                model=jepa,
                optimizer=optimizer,
                epoch=epoch,
                step=global_step,
            )

    if wandb_run:
        import wandb

        wandb.finish()

    logger.info("Training complete!")


if __name__ == "__main__":
    fire.Fire(run)
