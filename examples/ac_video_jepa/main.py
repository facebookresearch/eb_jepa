import copy
import os
from pathlib import Path
from time import time

import fire
import torch
import torch.nn as nn
import wandb
import yaml
from omegaconf import OmegaConf
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from tqdm import tqdm

from eb_jepa.architectures import (
    ImpalaEncoder,
    InverseDynamicsModel,
    Projector,
    RNNPredictor,
)
from eb_jepa.datasets.utils import init_data
from eb_jepa.jepa import JEPA, JEPAProbe
from eb_jepa.logging import get_logger
from eb_jepa.losses import SquareLossSeq, VC_IDM_Sim_Regularizer
from eb_jepa.schedulers import CosineWithWarmup
from eb_jepa.state_decoder import MLPXYHead
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
from examples.ac_video_jepa.eval import launch_plan_eval, launch_unroll_eval

logger = get_logger(__name__)


def run(
    fname: str = "examples/ac_video_jepa/cfgs/train.yaml",
    cfg=None,
    folder=None,
    **overrides,
):
    """
    Train an action-conditioned Video JEPA model.

    Args:
        fname: Path to the YAML config file.
        cfg: Pre-loaded config object (optional, overrides config file).
        folder: Experiment folder path (optional, auto-generated if not provided).
        **overrides: Config overrides in dot notation (e.g., model.henc=64).
    """
    if cfg is None:
        cfg = load_config(fname, overrides if overrides else None)

    # Create experiment directory using unified structure (if not provided)
    if folder is None:
        if cfg.meta.get("model_folder"):
            folder = Path(cfg.meta.model_folder)
            folder_name = folder.name
            exp_name = folder_name.rsplit("_seed", 1)[0]
        else:
            sweep_name = get_default_dev_name()
            exp_name = get_exp_name("ac_video_jepa", cfg)
            folder = get_unified_experiment_dir(
                example_name="ac_video_jepa",
                sweep_name=sweep_name,
                exp_name=exp_name,
                seed=cfg.meta.seed,
            )
    else:
        folder = Path(folder)
        folder_name = folder.name
        exp_name = folder_name.rsplit("_seed", 1)[0]

    os.makedirs(folder, exist_ok=True)

    loader, val_loader, data_config = init_data(
        env_name=cfg.data.env_name, cfg_data=dict(cfg.data)
    )

    # -- SETUP
    setup_device("auto")
    setup_seed(cfg.meta.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -- WANDB
    wandb_run = setup_wandb(
        project="eb_jepa",
        config={
            "example": "ac_video_jepa",
            **OmegaConf.to_container(cfg, resolve=True),
        },
        run_dir=folder,
        run_name=exp_name,
        tags=[f"seed_{cfg.meta.seed}", "ac_video_jepa"],
        group=cfg.logging.get("wandb_group"),
        enabled=cfg.logging.get("log_wandb", False),
        sweep_id=cfg.logging.get("wandb_sweep_id"),
    )

    log_data_info(
        cfg.data.env_name,
        len(loader),
        data_config.batch_size,
        train_samples=data_config.size,
        val_samples=data_config.val_size,
    )

    # Mixed precision setup
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16}
    dtype = dtype_map.get(cfg.training.get("dtype", "float16").lower(), torch.float16)
    use_amp = cfg.training.get("use_amp", True)
    scaler = GradScaler(device.type, enabled=use_amp)
    logger.info(f"Using AMP with {dtype=}" if use_amp else f"AMP disabled")

    # -- ENV (for plan/unroll eval)
    enable_eval = cfg.meta.get("enable_plan_eval", False)
    env_creator = None
    plan_cfg = None
    num_eval_episodes = 10

    if enable_eval:
        if cfg.meta.eval_every_itr <= 0:
            cfg.meta.eval_every_itr = len(loader)
        with open(cfg.eval.plan_cfg_path, "r") as f:
            plan_cfg = yaml.load(f, Loader=yaml.FullLoader)
        plan_cfg["logging"] = copy.deepcopy(dict(cfg.logging))
        with open(cfg.eval.eval_cfg_path, "r") as f:
            eval_cfg_dict = yaml.safe_load(f)
        _, _, env_config = init_data(
            env_name=cfg.data.env_name, cfg_data=dict(eval_cfg_dict.get("data", {}))
        )
        num_eval_episodes = eval_cfg_dict.get("meta", {}).get("num_eval_episodes", 10)

        def env_creator():
            from eb_jepa.datasets.two_rooms.env import DotWall

            cfg_eval_env = eval_cfg_dict.get("env")
            return DotWall(
                config=env_config,
                **cfg_eval_env,
            )

    # -- SAVE CONFIG
    latest_ckpt_path = folder / "latest.pth.tar"
    steps_per_epoch = data_config.size // data_config.batch_size
    total_steps = cfg.optim.epochs * steps_per_epoch
    config_path = folder / "config.yaml"
    with open(config_path, "w") as f:
        OmegaConf.save(cfg, config_path)
    print(f"Saved complete config to {config_path}")

    # -- MODEL
    test_input = torch.rand(
        (
            1,
            cfg.model.dobs,
            1,
            data_config.img_size,
            data_config.img_size,
        )
    )
    encoder = ImpalaEncoder(
        width=1,
        stack_sizes=(16, cfg.model.henc, cfg.model.dstc),
        num_blocks=2,
        dropout_rate=None,
        layer_norm=False,
        input_channels=cfg.model.dobs,
        final_ln=True,
        mlp_output_dim=512,
        input_shape=(cfg.model.dobs, data_config.img_size, data_config.img_size),
    )
    test_output = encoder(test_input)
    _, f, _, h, w = test_output.shape
    predictor = RNNPredictor(
        hidden_size=encoder.mlp_output_dim, final_ln=encoder.final_ln
    )
    aencoder = nn.Identity()
    if cfg.model.regularizer.use_proj:
        projector = Projector(
            f"{encoder.mlp_output_dim}-{encoder.mlp_output_dim*4}-{encoder.mlp_output_dim*4}"
        )
    else:
        projector = None
    logger.info(f"Encoder output: {tuple(test_output.shape)}")
    idm = InverseDynamicsModel(
        state_dim=h
        * w
        * (projector.out_dim if cfg.model.regularizer.idm_after_proj else f),
        hidden_dim=256,
        action_dim=2,
    ).to(device)
    regularizer = VC_IDM_Sim_Regularizer(
        cov_coeff=cfg.model.regularizer.cov_coeff,
        std_coeff=cfg.model.regularizer.std_coeff,
        sim_coeff_t=cfg.model.regularizer.sim_coeff_t,
        idm_coeff=cfg.model.regularizer.get("idm_coeff", 0.1),
        idm=idm,
        first_t_only=cfg.model.regularizer.get("first_t_only"),
        projector=projector,
        spatial_as_samples=cfg.model.regularizer.spatial_as_samples,
        idm_after_proj=cfg.model.regularizer.idm_after_proj,
        sim_t_after_proj=cfg.model.regularizer.sim_t_after_proj,
    )
    ploss = SquareLossSeq()
    jepa = JEPA(encoder, aencoder, predictor, regularizer, ploss).to(device)

    # Log model structure and parameters
    encoder_params = sum(p.numel() for p in encoder.parameters())
    predictor_params = sum(p.numel() for p in predictor.parameters())
    log_model_info(jepa, {"encoder": encoder_params, "predictor": predictor_params})

    log_config(cfg)

    # -- PROBER
    xy_head = MLPXYHead(
        input_shape=test_output.shape[1],
        normalizer=loader.dataset.normalizer,
    ).to(device)
    xy_prober = JEPAProbe(
        jepa=jepa,
        head=xy_head,
        hcost=nn.MSELoss(),
    )

    jepa_optimizer = AdamW(
        jepa.parameters(),
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.get("weight_decay", 1e-6),
    )
    jepa_scheduler = CosineWithWarmup(jepa_optimizer, total_steps, warmup_ratio=0.1)

    probe_optimizer = AdamW(xy_head.parameters(), lr=1e-3, weight_decay=1e-5)
    probe_scheduler = CosineWithWarmup(probe_optimizer, total_steps, warmup_ratio=0.1)

    # -- LOAD CKPT
    start_epoch = 0
    ckpt_info = {}
    if cfg.meta.load_model:
        checkpoint_path = folder / cfg.meta.get("load_checkpoint", "latest.pth.tar")
        ckpt_info = load_checkpoint(
            checkpoint_path, jepa, jepa_optimizer, jepa_scheduler, device=device
        )
        start_epoch = ckpt_info.get("epoch", 0)
        if "xy_head_state_dict" in ckpt_info:
            xy_head.load_state_dict(ckpt_info["xy_head_state_dict"])

    # Compile
    if torch.cuda.is_available() and cfg.model.compile:
        logger.info("✅ Compiling model with torch.compile")
        jepa = torch.compile(jepa)

    # -- EVAL ONLY MODE
    if cfg.meta.get("eval_only_mode", False):
        if not enable_eval:
            raise ValueError("eval_only_mode requires enable_plan_eval=True")
        logger.info("Running evaluation only (no training)")
        eval_results = launch_unroll_eval(
            jepa,
            env_creator,
            folder,
            start_epoch,
            ckpt_info.get("step", 0),
            "_eval_only",
            val_loader,
            xy_prober,
            cfg,
        )
        eval_results.update(
            launch_plan_eval(
                jepa,
                env_creator,
                folder,
                start_epoch,
                global_step=ckpt_info.get("step", 0),
                suffix="_eval_only",
                num_eval_episodes=num_eval_episodes,
                loader=val_loader,
                prober=xy_prober,
                plan_cfg=plan_cfg,
            )
        )
        logger.info(
            f"Evaluation complete. Success rate: {eval_results['success_rate']:.2%}"
        )
        return eval_results

    # -- TRAINING LOOP
    for epoch in range(start_epoch, cfg.optim.epochs):
        epoch_start_time = time()
        pbar = tqdm(
            enumerate(loader),
            total=len(loader),
            desc=f"Epoch {epoch}/{cfg.optim.epochs - 1}",
            disable=cfg.logging.get("tqdm_silent", False),
        )
        for idx, (x, a, loc, _, _) in pbar:
            itr_start_time = time()
            global_step = epoch * len(loader) + idx

            x = x.to(device)
            a = a.to(device)
            loc = loc.to(device)
            total_loss = torch.tensor(0.0, device=device)

            # Calculate JEPA loss
            jepa_optimizer.zero_grad()
            with autocast(device.type, enabled=use_amp, dtype=dtype):
                _, (jepa_loss, regl, regl_unweight, regldict, pl) = jepa.unroll(
                    x,
                    a,
                    nsteps=cfg.model.nsteps,
                    unroll_mode="autoregressive",
                    ctxt_window_time=1,
                    compute_loss=True,
                    return_all_steps=False,
                )
                total_loss += jepa_loss

            # Mixed precision backward pass
            scaler.scale(jepa_loss).backward()
            if cfg.optim.get("grad_clip_enc") and cfg.optim.get("grad_clip_pred"):
                scaler.unscale_(jepa_optimizer)
                torch.nn.utils.clip_grad_norm_(
                    jepa.encoder.parameters(), cfg.optim.grad_clip_enc
                )
                torch.nn.utils.clip_grad_norm_(
                    jepa.predictor.parameters(), cfg.optim.grad_clip_pred
                )
            scaler.step(jepa_optimizer)
            scaler.update()
            jepa_scheduler.step()

            # Calculate probe loss
            probe_optimizer.zero_grad()
            with autocast(device.type, enabled=use_amp, dtype=dtype):
                xy_loss = xy_prober(
                    observations=x[:, :, :1],
                    targets=loc[:, :, :1],
                )
                xy_loss = loader.dataset.normalizer.unnormalize_mse(xy_loss)
                total_loss += xy_loss

            scaler.scale(xy_loss).backward()
            scaler.step(probe_optimizer)
            scaler.update()
            probe_scheduler.step()

            # Update progress bar
            pbar.set_postfix(
                {
                    "loss": f"{total_loss.item():.4f}",
                    "reg": f"{regl.item():.4f}",
                    "pred": f"{pl.item():.4f}",
                }
            )

            itr_time = time() - itr_start_time
            if global_step % cfg.logging.log_every == 0:
                log_data = {
                    "train/total_loss": total_loss.item(),
                    "train/reg_loss": regl.item(),
                    "train/reg_loss_unweight": regl_unweight.item(),
                    "train/pred_loss": pl.item(),
                    "train/probe_loss": xy_loss.item(),
                    "global_step": global_step,
                    "epoch": epoch,
                    "itr_time": itr_time,
                    "optim/jepa_lr": jepa_optimizer.param_groups[0]["lr"],
                    "optim/probe_lr": probe_optimizer.param_groups[0]["lr"],
                }
                for loss_name, loss_value in regldict.items():
                    log_data[f"train/regl/{loss_name}"] = loss_value

                if cfg.logging.get("log_wandb"):
                    wandb.log(log_data, step=global_step)

            # Planning eval (only if eval is enabled)
            if (
                enable_eval
                and (global_step + 1) % cfg.meta.eval_every_itr == 0
                and global_step > 0
            ):
                eval_results = launch_plan_eval(
                    jepa,
                    env_creator,
                    folder,
                    epoch,
                    global_step,
                    suffix="",
                    num_eval_episodes=num_eval_episodes,
                    loader=val_loader,
                    prober=xy_prober,
                    plan_cfg=plan_cfg,
                )

                if cfg.logging.get("log_wandb"):
                    wandb.log(eval_results, step=global_step)

            # Light eval (only if eval is enabled)
            if (
                enable_eval
                and (global_step + 1) % cfg.meta.light_eval_freq == 0
                and global_step > 0
            ):
                eval_results = launch_unroll_eval(
                    jepa,
                    env_creator,
                    folder,
                    epoch,
                    global_step,
                    suffix="",
                    loader=val_loader,
                    prober=xy_prober,
                    cfg=cfg,
                )

                if cfg.logging.get("log_wandb"):
                    wandb.log(eval_results, step=global_step)

        epoch_time = time() - epoch_start_time

        # Log epoch summary
        log_epoch(
            epoch,
            {
                "loss": total_loss.item(),
                "reg": regl.item(),
                "pred": pl.item(),
                "probe": xy_loss.item(),
            },
            total_epochs=cfg.optim.epochs,
            elapsed_time=epoch_time,
        )

        if cfg.logging.get("log_wandb"):
            wandb.log(
                {"epoch": epoch, "epoch_time": epoch_time},
                step=epoch * len(loader),
            )

        # Save checkpoint
        save_checkpoint(
            latest_ckpt_path,
            model=jepa,
            optimizer=jepa_optimizer,
            scheduler=jepa_scheduler,
            epoch=epoch,
            step=global_step,
            xy_head_state_dict=xy_head.state_dict(),
            probe_optimizer_state_dict=probe_optimizer.state_dict(),
            probe_scheduler_state_dict=probe_scheduler.state_dict(),
        )
        if epoch % cfg.logging.save_every_n_epochs == 0:
            save_checkpoint(
                folder / f"e-{epoch}.pth.tar",
                model=jepa,
                optimizer=jepa_optimizer,
                scheduler=jepa_scheduler,
                epoch=epoch,
                step=global_step,
                xy_head_state_dict=xy_head.state_dict(),
                probe_optimizer_state_dict=probe_optimizer.state_dict(),
                probe_scheduler_state_dict=probe_scheduler.state_dict(),
            )


if __name__ == "__main__":
    fire.Fire(run)
