"""
Evaluation utilities for action-conditioned Video JEPA.
"""

import os
from pathlib import Path

import torch
import yaml

from eb_jepa.logging import get_logger
from eb_jepa.planning import main_eval, main_unroll_eval

logger = get_logger(__name__)


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
    logger.info(f"Planning eval: epoch={epoch} step={global_step}")
    jepa.eval()
    folder = Path(folder)
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
    logger.info(
        f"   success_rate={eval_results['success_rate']:.2f} | mean_dist={eval_results['mean_state_dist']:.4f}"
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
    logger.info(f"Unroll eval: epoch={epoch} step={global_step}")
    folder = Path(folder)
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
    logger.info(f"Unroll eval - mean_mse: {mean_values} | std_mse: {std_values}")
    jepa.train()

    return eval_results
