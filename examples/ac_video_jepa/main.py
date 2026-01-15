import copy
import os
import random
from pathlib import Path
from time import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import wandb
import yaml
from omegaconf import OmegaConf
from torch.amp import GradScaler, autocast
from torch.optim import AdamW

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
from eb_jepa.planning import main_eval, main_unroll_eval
from eb_jepa.schedulers import CosineWithWarmup
from examples.ac_video_jepa.heads import MLPXYHead

logging = get_logger(__name__)


def clean_state_dict(state_dict):
    return {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}


@torch.no_grad()
def launch_plan_eval(
    jepa,
    env_creator,
    folder,
    epoch,
    global_step,
    suffix="",
    num_eval_episodes=10,
    loader=None,
    prober=None,
    plan_cfg=None,
):
    """Evaluate the planning capabilities of the trained JEPA model."""
    logging.info(f"🎯 Planning eval | epoch={epoch} step={global_step}")
    jepa.eval()
    eval_folder = folder / "plan_eval" / f"step-{global_step}{suffix}"
    os.makedirs(eval_folder, exist_ok=True)

    if plan_cfg is not None:
        plan_cfg_file = eval_folder / "plan_config.yaml"
        with open(plan_cfg_file, "w") as f:
            yaml.dump(plan_cfg, f)

    eval_results = main_eval(
        plan_cfg=plan_cfg,
        model=jepa,
        env_creator=env_creator,
        eval_folder=eval_folder,
        num_episodes=num_eval_episodes,
        loader=loader,
        prober=prober,
    )
    logging.info(
        f"   ✓ success_rate={eval_results['success_rate']:.2f} | mean_dist={eval_results['mean_state_dist']:.4f}"
    )
    jepa.train()

    return eval_results


@torch.no_grad()
def launch_unroll_eval(
    jepa,
    env_creator,
    folder,
    epoch,
    global_step,
    suffix="",
    loader=None,
    prober=None,
    cfg=None,
):
    """Evaluate the unrolling (prediction) capabilities of the trained JEPA model."""
    jepa.eval()
    logging.info(f"📊 Unroll eval | epoch={epoch} step={global_step}")
    eval_folder = folder / "unroll_eval" / f"step-{global_step}{suffix}"
    os.makedirs(eval_folder, exist_ok=True)
    eval_results = main_unroll_eval(
        jepa,
        env_creator,
        eval_folder,
        loader=loader,
        prober=prober,
        cfg=cfg,
    )
    steps = [0, 1, 2, 3]
    mean_values = " | ".join(
        [f"t{i}={eval_results[f'val_rollout/mean_mse/{i}']:.2f}" for i in steps]
    )
    std_values = " | ".join(
        [f"{i}: {eval_results[f'val_rollout/std_mse/{i}']:.2f}" for i in steps]
    )
    logging.info(f"Unroll eval - mean_mse: {mean_values} | std_mse: {std_values}")
    jepa.train()

    return eval_results


def load_override_cfg(fname: str, kwargs_dict: Optional[dict] = None):
    """Load configuration from a YAML file and optionally override with a dictionary."""
    assert fname and os.path.exists(fname), f"Config file not found: {fname}"
    with open(fname, "r") as f:
        cfg = OmegaConf.create(yaml.safe_load(f))
    print(f"Loaded config from {fname}")
    if kwargs_dict:
        override_dict = {}
        for arg_name, arg_value in kwargs_dict.items():
            keys = arg_name.split(".")
            current = override_dict
            for key in keys[:-1]:
                current = current.setdefault(key, {})
            current[keys[-1]] = arg_value
        cfg = OmegaConf.merge(cfg, OmegaConf.create(override_dict))
    return cfg


def get_experiment_folder_without_seed(cfg, data_config, quick_debug=False):
    """Generate the experiment folder path without seed for wandb aggregation."""
    if cfg.meta.get("model_folder"):
        return Path(cfg.meta.model_folder)
    exp_name = f"{'deb_' if quick_debug else ''}{cfg.model.encoder_architecture}_encsk{cfg.model.encoder_skip_connections}_prsk{cfg.model.predictor_skip_connections}_cov{cfg.model.regularizer.cov_coeff}_std{cfg.model.regularizer.std_coeff}_simt{cfg.model.regularizer.get('sim_coeff_t')}_idm{cfg.model.regularizer.get('idm_coeff')}_sp{cfg.model.regularizer.spatial_as_samples}_useproj{cfg.model.regularizer.use_proj}_idmproj{cfg.model.regularizer.idm_after_proj}_simtproj{cfg.model.regularizer.sim_t_after_proj}_1stt{cfg.model.regularizer.get('first_t_only')}_roll{cfg.model.train_rollout}_dstc{cfg.model.dstc}_henc{cfg.model.henc}_hpre{cfg.model.hpre}_lr{cfg.optim.lr}_wd{cfg.optim.weight_decay}_bs{data_config.batch_size}_samplen{data_config.sample_length}_size{data_config.size}_cw{data_config.cross_wall_rate}_wb{data_config.wall_bump_rate}_duptraj{data_config.dup_traj_rate}_fixw{data_config.fix_wall}_{cfg.logging['exp_suffix'] if cfg.logging.get('exp_suffix') else ''}"
    return cfg.meta.ckpt_dir / Path(exp_name)


def get_experiment_folder(cfg, data_config, quick_debug=False):
    """Generate the experiment folder path based on configuration."""
    if cfg.meta.get("model_folder"):
        return Path(cfg.meta.model_folder)
    exp_name_base = get_experiment_folder_without_seed(cfg, data_config, quick_debug)
    exp_name = f"{os.path.basename(exp_name_base)}_seed{cfg.meta.seed}"
    return cfg.meta.ckpt_dir / Path(exp_name)


def main(
    fname: str = "examples/ac_video_jepa/cfgs/train.yaml",
    cfg=None,
    folder=None,
    **kwargs,
):
    """
    Train an action-conditioned Video JEPA model.

    Args:
        fname: Path to the YAML config file.
        cfg: Pre-loaded config object (optional, overrides fname).
        folder: Experiment folder path (optional, auto-generated if not provided).
        **kwargs: Config overrides in dot notation (e.g., model.henc=64).
    """
    if cfg is None:
        cfg = load_override_cfg(fname, kwargs)
    quick_debug = cfg.meta.get("quick_debug", False)
    if folder is None:
        folder = get_experiment_folder(cfg, cfg.data, quick_debug=quick_debug)
    os.makedirs(folder, exist_ok=True)

    if quick_debug:
        cfg.logging.log_wandb = False
        cfg.meta.eval_every_itr = 2
        cfg.meta.light_eval_freq = 2
        cfg.model.compile = False
        cfg.data.num_workers = 0
        cfg.data.batch_size = 4
        cfg.logging.tqdm_silent = False

    if cfg.meta.light_eval_only_mode:
        cfg.meta.light_eval_freq = 2
        cfg.logging.log_wandb = False
        cfg.data.batch_size = 4
        cfg.logging.tqdm_silent = False

    train_jepa = True
    train_probe = True
    if cfg.meta.get("light_eval_only_mode") or cfg.meta.get("plan_eval_only_mode"):
        train_jepa, train_probe, cfg.logging.log_wandb = False, False, False

    loader, val_loader, data_config = init_data(
        env_name=cfg.data.env_name, cfg_data=dict(cfg.data)
    )
    logging.info(f"📦 Data: {len(loader)} batches × {data_config.batch_size} samples")
    # Set seed
    torch.manual_seed(cfg.meta.seed)
    np.random.seed(cfg.meta.seed)
    random.seed(cfg.meta.seed)

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16}
    dtype = dtype_map.get(cfg.data.dtype.lower(), torch.float32)
    mixed_precision = dtype != torch.float32

    # -- ENV
    if cfg.meta.get("enable_plan_eval"):
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
        num_eval_episodes = (
            eval_cfg_dict.get("meta", {}).get("num_eval_episodes", 10)
            if not quick_debug
            else 1
        )
        if quick_debug:
            eval_cfg_dict["env"]["n_allowed_steps"] = 20

        def env_creator():
            from eb_jepa.datasets.two_rooms.env import DotWall

            cfg_eval_env = eval_cfg_dict.get("env")
            return DotWall(
                config=env_config,
                **cfg_eval_env,
            )

    # -- LOGGING
    config_path = folder / "config.yaml"
    with open(config_path, "w") as f:
        OmegaConf.save(cfg, config_path)
    print(f"Saved complete config to {config_path}")

    latest_ckpt_path = folder / "latest.pth.tar"
    steps_per_epoch = data_config.size // data_config.batch_size
    total_steps = cfg.optim.epochs * steps_per_epoch

    logging.info(f"⚙️  Config: {cfg}")

    if cfg.logging.get("log_wandb"):
        project_name = "eb-jepa-ac" if not quick_debug else "eb-jepa-ac-debug"
        wandb_run_id_file = os.path.join(folder, "wandb_run_id.txt")

        # Create a base experiment name without seed for aggregation
        exp_name_base = get_experiment_folder_without_seed(
            cfg, cfg.data, quick_debug=cfg.meta.get("quick_debug")
        )
        exp_name_base_str = os.path.basename(exp_name_base)

        wandb_config = {
            "project": project_name,
            "dir": folder,
            "name": exp_name_base_str,
            "tags": [f"seed_{cfg.meta.seed}"],
            "config": OmegaConf.to_container(cfg, resolve=True),
        }

        if cfg.logging.wandb_group:
            wandb_config["group"] = cfg.logging.wandb_group

        if cfg.logging.get("wandb_sweep") and cfg.logging.get("wandb_sweep_id"):
            wandb_config["tags"].append(f"sweep_{cfg.logging.wandb_sweep_id}")
            logging.info(f"W&B sweep: {cfg.logging.wandb_sweep_id}")

            # Check for existing run to resume
            if os.path.exists(wandb_run_id_file):
                with open(wandb_run_id_file, "r") as f:
                    wandb_run_id = f.read().strip()
                # Set environment variables to control both sweep association and run resuming
                # WANDB_SWEEP_ID associates the run with the sweep
                # WANDB_RUN_ID forces the specific run ID for resuming
                # WANDB_RESUME enables resume mode
                os.environ["WANDB_SWEEP_ID"] = cfg.logging.wandb_sweep_id
                os.environ["WANDB_RUN_ID"] = wandb_run_id
                os.environ["WANDB_RESUME"] = "allow"
                wandb.init(**wandb_config)
                logging.info(f"Resuming W&B run {wandb_run_id} (sweep)")
            else:
                # First run: set WANDB_SWEEP_ID to associate with sweep
                os.environ["WANDB_SWEEP_ID"] = cfg.logging.wandb_sweep_id
                wandb.init(**wandb_config)
                with open(wandb_run_id_file, "w") as f:
                    f.write(wandb.run.id)
                logging.info(f"Created W&B run {wandb.run.id} (sweep)")
        else:
            if os.path.exists(wandb_run_id_file):
                with open(wandb_run_id_file, "r") as f:
                    wandb_run_id = f.read().strip()
                wandb_config.update({"id": wandb_run_id, "resume": "allow"})
                wandb.init(**wandb_config)
                logging.info(f"Resuming W&B run {wandb_run_id}")
            else:
                wandb.init(**wandb_config)
                with open(wandb_run_id_file, "w") as f:
                    f.write(wandb.run.id)
                logging.info(f"Created W&B run {wandb.run.id}")

    # -- MODEL
    device = "cuda" if torch.cuda.is_available() else "cpu"
    scaler = GradScaler() if device == "cuda" and mixed_precision else None
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
    logging.info(f"🧠 Encoder output: {tuple(test_output.shape)}")
    idm = InverseDynamicsModel(
        state_dim=h
        * w
        * (projector.out_dim if cfg.model.regularizer.idm_after_proj else f),
        hidden_dim=256,  # You can adjust this based on your needs
        action_dim=2,  # Number of action dimensions in your environment
    ).to(device)
    regularizer = VC_IDM_Sim_Regularizer(
        cov_coeff=cfg.model.regularizer.cov_coeff,
        std_coeff=cfg.model.regularizer.std_coeff,
        sim_coeff_t=cfg.model.regularizer.sim_coeff_t,
        idm_coeff=cfg.model.regularizer.get("idm_coeff", 0.1),
        idm=idm,  # Pass the IDM model reference
        first_t_only=cfg.model.regularizer.get("first_t_only"),
        projector=projector,
        spatial_as_samples=cfg.model.regularizer.spatial_as_samples,
        idm_after_proj=cfg.model.regularizer.idm_after_proj,
        sim_t_after_proj=cfg.model.regularizer.sim_t_after_proj,
    )
    ploss = SquareLossSeq()
    jepa = JEPA(encoder, aencoder, predictor, regularizer, ploss).to(device)
    logging.info(jepa)

    encoder_params = sum(p.numel() for p in encoder.parameters())
    predictor_params = sum(p.numel() for p in predictor.parameters())
    logging.info(
        f"🔢 Parameters: encoder={encoder_params:,} | predictor={predictor_params:,}"
    )

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

    jepa_optimizer, jepa_scheduler = None, None
    if train_jepa:
        jepa_optimizer = AdamW(
            jepa.parameters(),
            lr=cfg.optim.lr,
            weight_decay=cfg.optim.get("weight_decay", 1e-6),
        )
        jepa_scheduler = CosineWithWarmup(jepa_optimizer, total_steps, warmup_ratio=0.1)

    probe_optimizer, probe_scheduler = None, None
    if train_probe:
        probe_optimizer = AdamW(xy_head.parameters(), lr=1e-3, weight_decay=1e-5)
        probe_scheduler = CosineWithWarmup(
            probe_optimizer, total_steps, warmup_ratio=0.1
        )

    # -- LOAD CKPT
    start_epoch = 1
    if cfg.meta.load_model:

        def load_checkpoint():
            if cfg.meta.get("load_checkpoint"):
                checkpoint_path = folder / cfg.meta.load_checkpoint
            else:
                checkpoint_path = latest_ckpt_path
            if checkpoint_path.exists():
                checkpoint = torch.load(checkpoint_path, weights_only=False)
                msg = jepa.load_state_dict(
                    clean_state_dict(checkpoint["jepa_state_dict"])
                )
                logging.info(f"Loaded JEPA with message: {msg}")
                msg = xy_head.load_state_dict(
                    clean_state_dict(checkpoint["xy_head_state_dict"])
                )
                logging.info(f"Loaded XY head with message: {msg}")
                start_epoch = checkpoint["epoch"]
                if "jepa_optimizer_state_dict" in checkpoint and jepa_optimizer:
                    jepa_optimizer.load_state_dict(
                        checkpoint["jepa_optimizer_state_dict"]
                    )
                if "jepa_scheduler_state_dict" in checkpoint and jepa_scheduler:
                    jepa_scheduler.load_state_dict(
                        checkpoint["jepa_scheduler_state_dict"]
                    )

                if "probe_optimizer_state_dict" in checkpoint and probe_optimizer:
                    probe_optimizer.load_state_dict(
                        checkpoint["probe_optimizer_state_dict"]
                    )
                if "probe_scheduler_state_dict" in checkpoint and probe_scheduler:
                    probe_scheduler.load_state_dict(
                        checkpoint["probe_scheduler_state_dict"]
                    )
                logging.info(f"📂 Loaded checkpoint: epoch={start_epoch}")
            else:
                logging.warning(f"Checkpoint not found at {checkpoint_path}")
                start_epoch = 1
            return start_epoch

        start_epoch = load_checkpoint()
        # to allow for light eval from ckpt that finished training
        if cfg.meta.get("light_eval_only_mode"):
            start_epoch -= 1
    # Compile
    if torch.cuda.is_available() and cfg.model.compile:
        logging.info("⚡ Compiling model with torch.compile")
        jepa = torch.compile(jepa)

    # -- TRAINING LOOP
    if not cfg.meta.get("plan_eval_only_mode"):
        # index epochs starting at epoch 1
        for epoch in range(start_epoch, cfg.optim.epochs):
            epoch_start_time = time()
            for idx, (x, a, loc, _, _) in enumerate(loader):
                itr_start_time = time()
                if quick_debug or cfg.meta.get("light_eval_only_mode"):
                    if idx > 3:
                        break
                global_step = (epoch - 1) * len(loader) + idx

                x = x.to(device)
                a = a.to(device)
                loc = loc.to(device)
                total_loss = torch.tensor(0.0, device=device)

                # Calculate JEPA loss if training JEPA
                if train_jepa:
                    if jepa_optimizer:
                        jepa_optimizer.zero_grad()
                    with autocast(
                        enabled=mixed_precision, device_type=device, dtype=dtype
                    ):
                        jepa_loss, regl, regl_unweight, regldict, pl = jepa.forwardn(
                            x, a, nsteps=cfg.model.nsteps
                        )
                        total_loss += jepa_loss
                    # Mixed precision backward pass
                    if scaler is not None:
                        scaler.scale(jepa_loss).backward()
                        if cfg.optim.get("grad_clip_enc") and cfg.optim.get(
                            "grad_clip_pred"
                        ):
                            scaler.unscale_(jepa_optimizer)
                            encoder_grad_norm = torch.nn.utils.clip_grad_norm_(
                                jepa.encoder.parameters(), cfg.optim.grad_clip_enc
                            )
                            predictor_grad_norm = torch.nn.utils.clip_grad_norm_(
                                jepa.predictor.parameters(), cfg.optim.grad_clip_pred
                            )
                        scaler.step(jepa_optimizer)
                        scaler.update()
                    else:
                        jepa_loss.backward()
                        if cfg.optim.get("grad_clip_enc") and cfg.optim.get(
                            "grad_clip_pred"
                        ):
                            encoder_grad_norm = torch.nn.utils.clip_grad_norm_(
                                jepa.encoder.parameters(), cfg.optim.grad_clip_enc
                            )
                            predictor_grad_norm = torch.nn.utils.clip_grad_norm_(
                                jepa.predictor.parameters(), cfg.optim.grad_clip_pred
                            )
                        jepa_optimizer.step()
                    if jepa_scheduler:
                        jepa_scheduler.step()
                else:
                    encoder_grad_norm = None
                    predictor_grad_norm = None
                    # Still compute for logging, but detach to avoid gradients
                    with torch.no_grad():
                        jepa_loss, regl, regl_unweight, regldict, pl = jepa.forwardn(
                            x, a, nsteps=cfg.model.nsteps
                        )

                # Calculate probe loss if training probe and after start point
                if train_probe and global_step >= cfg.meta.start_probing_after:
                    if probe_optimizer:
                        probe_optimizer.zero_grad()
                    with autocast(
                        enabled=mixed_precision, device_type=device, dtype=dtype
                    ):
                        xy_loss = xy_prober(
                            observations=x[:, :, :1],
                            targets=loc[:, :, :1],
                        )
                        xy_loss = loader.dataset.normalizer.unnormalize_mse(xy_loss)
                        total_loss += xy_loss

                    if probe_optimizer:
                        if scaler is not None:
                            scaler.scale(xy_loss).backward()
                            scaler.step(probe_optimizer)
                            scaler.update()
                        else:
                            xy_loss.backward()
                            probe_optimizer.step()
                        probe_scheduler.step()
                else:
                    # Still compute for logging, but detach to avoid gradients
                    with torch.no_grad():
                        xy_loss = xy_prober(x[:, :, :1], loc[:, :, :1])
                        xy_loss = loader.dataset.normalizer.unnormalize_mse(xy_loss)
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
                    }
                    # Log optimization metrics
                    if jepa_optimizer is not None:
                        log_data["optim/jepa_lr"] = jepa_optimizer.param_groups[0]["lr"]
                    if probe_optimizer is not None:
                        log_data["optim/probe_lr"] = probe_optimizer.param_groups[0][
                            "lr"
                        ]
                    if encoder_grad_norm is not None:
                        log_data["train/encoder_grad_norm"] = encoder_grad_norm.item()
                    if predictor_grad_norm is not None:
                        log_data["train/predictor_grad_norm"] = (
                            predictor_grad_norm.item()
                        )
                    for loss_name, loss_value in regldict.items():
                        log_data[f"train/regl/{loss_name}"] = loss_value
                    logging.info(
                        f"[E{epoch:03d}|S{global_step:05d}] loss={total_loss.item():.3f} "
                        f"reg={regl.item():.3f} pred={pl.item():.4f} probe={xy_loss.item():.3f}"
                    )

                    if cfg.logging.get("log_wandb"):
                        wandb.log(log_data, step=global_step)

                # Planning eval
                if (
                    cfg.meta.get("enable_plan_eval")
                    and not cfg.meta.get("light_eval_only_mode")
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

                # Light eval
                if (
                    global_step + 1
                ) % cfg.meta.light_eval_freq == 0 and global_step > 0:
                    eval_results = launch_unroll_eval(
                        jepa,
                        env_creator,
                        folder,
                        epoch,
                        global_step,
                        suffix=(
                            "-light-only"
                            if cfg.meta.get("light_eval_only_mode")
                            else ""
                        ),
                        loader=val_loader,
                        prober=xy_prober,
                        cfg=cfg,
                    )

                    if cfg.logging.get("log_wandb"):
                        wandb.log(eval_results, step=global_step)
            epoch_time = time() - epoch_start_time
            if cfg.logging.get("log_wandb"):
                wandb.log(
                    {"epoch": epoch, "epoch_time": epoch_time},
                    step=epoch * len(loader),
                )
            if not cfg.meta.get("light_eval_only_mode"):

                def save_checkpoint():
                    # Save checkpoint at the end of the epoch if it's time
                    save_dict = {
                        "epoch": epoch,
                        "jepa_state_dict": jepa.state_dict(),
                        "xy_head_state_dict": xy_head.state_dict(),
                    }
                    if jepa_optimizer:
                        save_dict["jepa_optimizer_state_dict"] = (
                            jepa_optimizer.state_dict()
                        )
                    if jepa_scheduler:
                        save_dict["jepa_scheduler_state_dict"] = (
                            jepa_scheduler.state_dict()
                        )
                    if probe_optimizer:
                        save_dict["probe_optimizer_state_dict"] = (
                            probe_optimizer.state_dict()
                        )
                    if probe_scheduler:
                        save_dict["probe_scheduler_state_dict"] = (
                            probe_scheduler.state_dict()
                        )
                    torch.save(
                        clean_state_dict(save_dict),
                        latest_ckpt_path,
                    )
                    if epoch % cfg.logging.save_every_n_epochs == 0:
                        checkpoint_path = folder / f"e-{epoch}.pth.tar"
                        torch.save(save_dict, checkpoint_path)
                        print(f"Checkpoint saved at {checkpoint_path}")

                save_checkpoint()
    else:
        logging.info("🎯 Plan evaluation mode (skipping training)")
        global_step = start_epoch * len(loader)
        launch_plan_eval(
            jepa,
            env_creator,
            folder,
            start_epoch,
            global_step,
            suffix=eval_cfg_dict["meta"].get("plan_suffix"),
            num_eval_episodes=num_eval_episodes,
            loader=val_loader,
            prober=xy_prober,
            plan_cfg=plan_cfg,
        )


if __name__ == "__main__":
    import fire

    fire.Fire(main)
