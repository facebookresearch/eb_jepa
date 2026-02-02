import os
import time
from abc import ABC, abstractmethod
from typing import Callable, List, NamedTuple, Optional

import numpy as np
import pandas as pd
import torch
from einops import rearrange
from omegaconf import OmegaConf
from tqdm import tqdm

from eb_jepa.logging import get_logger
from eb_jepa.vis_utils import (
    analyze_distances,
    create_comparison_gif,
    plot_losses,
    save_decoded_frames,
    save_gif,
    show_images,
)

logger = get_logger(__name__)

planner_name_map = {
    "cem": "CEMPlanner",
    "mppi": "MPPIPlanner",
}
objective_name_map = {
    "repr_dist": "ReprTargetDistMPCObjective",
}


def main_unroll_eval(
    model,
    env_creator,
    eval_folder,
    num_samples=4,
    loader=None,
    prober=None,
    cfg=None,
):
    """
    Evaluate the model's unrolling capabilities by comparing unrolled predictions to ground truth.
    """
    env = env_creator()
    env.reset()
    device = next(model.parameters()).device
    normalizer = (
        loader.dataset.normalizer if hasattr(loader.dataset, "normalizer") else None
    )
    agent = GCAgent(
        model=model, plan_cfg=None, normalizer=normalizer, env=env, loc_prober=prober
    )
    mse_values = []
    position_mse_values = []
    unroll_times = []
    loader_iter = iter(loader)

    for idx in tqdm(
        range(num_samples), desc="Evaluating unroll", disable=cfg.logging.tqdm_silent
    ):
        try:
            x, a, loc, wall_x, door_y = next(loader_iter)
        except StopIteration:
            logger.warning(
                f"Loader exhausted after {idx} samples (requested {num_samples})"
            )
            break

        x = x.to(device)
        a = a.to(device)
        with torch.no_grad():
            obs_init = x[:, :, 0:1]  # B C T H W
            start_time = time.time()
            predicted_states = agent.unroll(obs_init, a, repeat_batch=False)[
                :, :, :-1
            ]  # discard last predicted state
            end_time = time.time()
            unroll_times.append(end_time - start_time)
            rand_predicted_states = agent.unroll(
                obs_init, torch.randn_like(a), repeat_batch=False
            )[
                :, :, :-1
            ]  # B D T H W
            # To ensure independence across timesteps when encoding the sequence, batchify it
            # There is no independence between timesteps when using GroupNorm, even in eval mode
            B, C, T, H, W = x.shape
            gt_encoded = (
                model.encode(x.permute(0, 2, 1, 3, 4).flatten(0, 1).unsqueeze(2))
                .squeeze(2)
                .unflatten(dim=0, sizes=(B, -1))
                .permute(0, 2, 1, 3, 4)
            )
            latent_mse = (
                ((gt_encoded - predicted_states) ** 2).mean(dim=(1, 3, 4)).cpu().numpy()
            )  # B T
            mse_values.append(latent_mse)

            if prober:
                gt_decoded = agent.decode_loc_to_pixel(gt_encoded, wall_x, door_y)
                pred_decoded = agent.decode_loc_to_pixel(
                    predicted_states, wall_x, door_y
                )
                rand_pred_decoded = agent.decode_loc_to_pixel(
                    rand_predicted_states, wall_x, door_y
                )  # B T H W C
                gt_frames = agent.normalizer.unnormalize_state(
                    x.permute(0, 2, 1, 3, 4)
                ).permute(0, 1, 3, 4, 2)
                gt_frames = (
                    (gt_frames * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
                )  # B T H W C uint8

                # Decode positions from predicted_states and compute MSE with ground truth
                B_probe, D_probe, T_probe, H_probe, W_probe = predicted_states.shape
                pred_positions = (
                    prober.apply_head(predicted_states).permute(0, 2, 1).cpu()
                )  # B T 2
                gt_positions = loc.permute(0, 2, 1)  # B T 2
                position_mse = (
                    ((pred_positions - gt_positions.cpu()) ** 2)
                    .mean(dim=-1)
                    .cpu()
                    .numpy()
                )  # B T
                position_mse_values.append(position_mse)

                create_comparison_gif(
                    gt_frames,
                    pred_decoded,
                    rand_pred_decoded,
                    gt_dec=gt_decoded,
                    save_path=f"{eval_folder}/b{idx}.gif",
                )
    all_mse_values = np.vstack(mse_values)  # Shape: [num_batches, T]
    mean_mse_per_timestep = np.mean(all_mse_values, axis=0)  # Shape: [T]
    std_mse_per_timestep = np.std(all_mse_values, axis=0)  # Shape: [T]
    avg_unroll_time = np.mean(unroll_times)
    results = {}
    for t in range(mean_mse_per_timestep.shape[0]):
        results[f"val_rollout/mean_mse/{t}"] = mean_mse_per_timestep[t]
        results[f"val_rollout/std_mse/{t}"] = std_mse_per_timestep[t]

    # Log position MSE if prober was used
    if len(position_mse_values) > 0:
        all_position_mse_values = np.vstack(
            position_mse_values
        )  # Shape: [num_batches, T]
        mean_position_mse_per_timestep = np.mean(
            all_position_mse_values, axis=0
        )  # Shape: [T]
        std_position_mse_per_timestep = np.std(
            all_position_mse_values, axis=0
        )  # Shape: [T]
        for t in range(mean_position_mse_per_timestep.shape[0]):
            results[f"val_rollout/mean_pos_mse/{t}"] = mean_position_mse_per_timestep[t]
            results[f"val_rollout/std_pos_mse/{t}"] = std_position_mse_per_timestep[t]

    results["avg_unroll_time"] = avg_unroll_time

    pd.DataFrame([results]).to_csv(f"{eval_folder}/eval.csv", index=None)
    return results


### Main planning eval loop ###
def main_eval(
    plan_cfg,
    model,
    env_creator,
    eval_folder,
    num_episodes=10,
    loader=None,
    prober=None,
):
    plan_cfg = OmegaConf.create(plan_cfg)
    env = env_creator()
    env.reset()

    agent = GCAgent(
        model,
        action_dim=2,
        plan_cfg=plan_cfg,
        normalizer=env.normalizer,
        loc_prober=prober,
        env=env,
    )
    logger.info(f"Agent created with planner {agent.planner.__class__.__name__}")
    logger.info(f"Planning with {plan_cfg=}")

    successes = []
    distances = []
    episode_times = []
    episode_observations = []
    episode_infos = []

    for ep in range(num_episodes):
        episode_start_time = time.time()
        ep_folder = eval_folder / f"ep_{ep}"
        os.makedirs(ep_folder, exist_ok=True)
        if agent.decode_each_iteration:
            ep_plan_vis_dir = ep_folder / "plan_vis"
            os.makedirs(ep_plan_vis_dir, exist_ok=True)

        if plan_cfg.task_specification.goal_source == "dset":
            obs_slice, a, loc, _, _ = next(iter(loader))
            # obs, init_loc = obs_slice[0], loc[0]
            # goal_img, goal_loc = obs_slice[-1], loc[-1]  # [C, H, W] uint8 tensor
            # env.set_goal(goal_img) # Set goal in the environment
        elif plan_cfg.task_specification.goal_source == "random_state":
            obs, info = env.reset()  # [C, H, W] uint8 tensor
            obs, reward, done, truncated, info = env.step(
                np.zeros(env.action_space.shape[0])
            )  # step with zero action to get the first observation
            goal_img = info["target_obs"]  # [C, H, W] uint8 tensor

        combined = torch.stack([obs, goal_img], dim=0)
        show_images(
            combined,
            nrow=2,  # Both images in one row
            titles=["Init", "Goal"],
            save_path=f"{ep_folder}/state.pdf",
            close_fig=True,
            first_channel_only=False,
            clamp=False,
        )
        agent.set_goal(
            goal_img.detach().clone().to(dtype=torch.float32),
            info["target_position"],
        )

        done = False
        steps_left = env.n_allowed_steps
        pbar = tqdm(
            desc="executing agent",
            total=steps_left,
            leave=True,
            disable=plan_cfg.logging.tqdm_silent,
        )
        t0 = True

        observations = [obs]
        infos = [info]

        prev_losses = []
        prev_elite_losses_mean = []
        prev_elite_losses_std = []

        while steps_left > 0:
            # while (not done and steps_left > 0):
            plan_vis_path = (
                f"{ep_plan_vis_dir}/step{env.n_allowed_steps - steps_left}"
                if agent.decode_each_iteration
                else None
            )
            # first loop iter: obs is from reset(), then it is from step()
            obs_tensor = (
                env.normalizer.normalize_state(
                    obs.detach().clone().to(dtype=torch.float32, device=agent.device)
                )
                .unsqueeze(0)
                .unsqueeze(2)
            )  # Unsqueeze the batch and time dimensions : C H W -> 1 C 1 H W
            with torch.no_grad():
                action = (
                    agent.act(
                        obs_tensor,
                        steps_left=steps_left,
                        t0=t0,
                        plan_vis_path=plan_vis_path,
                    )
                    .cpu()
                    .numpy()
                )  # T, A
            if agent._prev_losses is not None:
                prev_losses.append(agent._prev_losses)
                prev_elite_losses_mean.append(agent._prev_elite_losses_mean)
                prev_elite_losses_std.append(agent._prev_elite_losses_std)
            for a in action:
                obs, reward, done, truncated, info = env.step(a)
                t0 = False
                observations.append(obs)
                infos.append(info)
                steps_left -= 1
                pbar.update(1)
                eval_results = env.eval_state(
                    info["target_position"], info["dot_position"]
                )
                success = eval_results["success"]
                state_dist = eval_results["state_dist"]
            pbar.set_postfix({"success": success, "state_dist": state_dist})
        pbar.close()

        episode_observations.append(torch.stack(observations))
        episode_infos.append(infos)
        successes.append(success)
        distances.append(state_dist)

        if plan_cfg.logging.get("optional_plots", True):
            analyze_distances(
                episode_observations[-1],
                episode_infos[-1],
                str(ep_folder / "agent"),
                goal_position=agent.goal_position,
                goal_state=agent.goal_state,
                normalizer=agent.normalizer,
                model=agent.model,
                objective=agent.objective,
                device=agent.device,
            )
            plot_losses(
                prev_losses,
                prev_elite_losses_mean,
                prev_elite_losses_std,
                work_dir=ep_folder,
                num_act_stepped=agent.num_act_stepped,
            )
            save_path = f"{ep_folder}/agent_steps_{'succ' if success else 'fail'}.gif"
        save_gif(
            episode_observations[-1],
            save_path=save_path,
            show_frame_numbers=True,
            fps=20,
            init_frame=observations[0],
            goal_frame=goal_img,
        )
        logger.info(f"GIF saved to {save_path}")
        episode_end_time = time.time()  # Add this line
        episode_times.append(episode_end_time - episode_start_time)
    avg_episode_time = np.mean(episode_times)
    task_data = {
        "success_rate": np.mean(successes),
        "mean_state_dist": np.mean(distances),
        "avg_episode_time": avg_episode_time,
    }
    pd.DataFrame([task_data]).to_csv(f"{eval_folder}/eval.csv", mode="a", index=None)
    return task_data


### Goal-conditioned agent for planning ###
class GCAgent:
    def __init__(
        self,
        model,
        action_dim=2,
        plan_cfg=None,
        normalizer: Optional[Callable] = None,
        loc_prober: Optional[Callable] = None,
        img_prober: Optional[Callable] = None,
        env: Optional[Callable] = None,
    ):
        self.plan_cfg = plan_cfg
        self.env = env
        self.model = model
        self.device = next(model.parameters()).device
        self.loc_prober = loc_prober
        self.img_prober = img_prober
        self.normalizer = normalizer

        # Set default values if plan_cfg is None
        if plan_cfg is None:
            self.decode_each_iteration = False
            self.num_act_stepped = 1
            self.planner = None
            logger.info("No plan_cfg provided in GCAgent, planner not initialized.")
        else:
            self.decode_each_iteration = plan_cfg.planner.get(
                "decode_each_iteration", False
            )
            self.num_act_stepped = plan_cfg.planner.get("num_act_stepped", 1)
            planner_name = plan_cfg.planner.get("planner_name", "cem")
            planner_class_name = planner_name_map[planner_name]
            planner_class = globals()[planner_class_name]
            if planner_class is not None:
                self.planner = planner_class(
                    unroll=self.unroll,
                    action_dim=action_dim,
                    decode_loc_to_pixel=self.decode_loc_to_pixel,
                    **plan_cfg.planner,
                )
            else:
                logger.info("No planner provided in GCAgent.")
                self.planner = None

        self.goal_state = None
        self.goal_position = None
        self.goal_state_enc = None
        self._prev_losses = None

    def set_goal(self, goal_state, goal_position=None):
        self.goal_position = goal_position
        self.goal_state = goal_state
        # Unsqueeze the batch and time dimensions : C H W -> 1 C 1 H W
        self.goal_state_enc = self.model.encode(
            self.normalizer.normalize_state(goal_state.to(self.device))
            .unsqueeze(0)
            .unsqueeze(2)
        )
        objective_name = self.plan_cfg.planner.planning_objective.get(
            "objective_type", "repr_target_dist"
        )
        objective_class_name = objective_name_map[objective_name]
        objective_class = globals()[objective_class_name]
        self.objective = objective_class(
            target_enc=self.goal_state_enc, **self.plan_cfg.planner.planning_objective
        )
        self.planner.set_objective(self.objective)

    def unroll(self, obs_init, actions, repeat_batch=True):
        """
        Unroll the model for planning.

        Args:
            obs_init: [B, C, T, H, W]
            actions: [B, A, T]

        Returns:
            predicted_states: [B, D, T, H, W]
        """
        batch_size = actions.shape[0]
        nsteps = actions.shape[2]
        if repeat_batch:
            obs_init = obs_init.repeat(batch_size, 1, 1, 1, 1)
        predicted_states, _ = self.model.unroll(
            obs_init,
            actions,
            nsteps=nsteps,
            unroll_mode="autoregressive",
            ctxt_window_time=self.plan_cfg["ctxt_window_time"] if self.plan_cfg else 1,
            compute_loss=False,
            return_all_steps=False,
        )
        return predicted_states

    def decode_loc_to_pixel(self, predicted_encs, wall_x=None, door_y=None):
        """
        Decode the predicted encodings into frames.

        Args:
            predicted_encs: [B, D, T, H, W]

        Returns:
            np.array of shape [B, T, H, W, C] on cpu for visualization.
        """
        assert self.loc_prober is not None
        B, D, T, H, W = predicted_encs.shape
        out = self.loc_prober.apply_head(predicted_encs).permute(0, 2, 1).cpu()  # B T 2
        out = self.normalizer.unnormalize_location(out)  # B T 2
        frames = self.env.coord_to_pixel(out, wall_x=wall_x, door_y=door_y)  # B T C H W
        frames = frames.permute(0, 1, 3, 4, 2).cpu().numpy()  # B T H W C
        return frames

    def act(self, obs, steps_left=None, t0=False, plan_vis_path=None):
        planning_result = self.planner.plan(
            obs,
            steps_left=steps_left,
            eval_mode=True,
            t0=t0,
            plan_vis_path=plan_vis_path,
        )
        self._prev_losses = planning_result.losses
        self._prev_elite_losses_mean = planning_result.prev_elite_losses_mean
        self._prev_elite_losses_std = planning_result.prev_elite_losses_std
        return planning_result.actions[: self.num_act_stepped]  # T, A


### Planning objectives to minimize ###
class ReprTargetDistMPCObjective:
    """Objective to minimize distance to the target representation."""

    def __init__(
        self,
        target_enc: torch.Tensor,
        sum_all_diffs: bool = False,
        **kwargs,
    ):
        self.target_enc = target_enc
        self.sum_all_diffs = sum_all_diffs

    def __call__(self, encodings: torch.Tensor, keepdims: bool = False) -> torch.Tensor:
        """
        Args:
            encodings: [B, D, T, H, W]
            keepdims: if True, return [B, T], else return [B]

        Returns:
            diff: [B, T] else [B] if sum_all_diffs or not keepdims
        """
        if self.sum_all_diffs:
            keepdims = True
        target = self.target_enc
        if target.shape != encodings.shape:
            target = target.expand(encodings.shape[0], -1, encodings.shape[2], -1, -1)

        metric = torch.nn.MSELoss(reduction="none")
        diff = metric(target, encodings).mean(dim=(1, 3, 4))  # B T
        if not keepdims:
            diff = diff[:, -1]
        if self.sum_all_diffs:
            diff = diff.sum(dim=1)
        return diff


### Planning optimizers interface ###
class PlanningResult(NamedTuple):
    actions: torch.Tensor
    losses: torch.Tensor = None
    prev_elite_losses_mean: torch.Tensor = None
    prev_elite_losses_std: torch.Tensor = None
    info: dict = None


class Planner(ABC):
    def __init__(self, unroll: Callable, **kwargs):
        self.unroll = unroll
        self.objective = None

    def set_objective(self, objective: Callable):
        self.objective = objective

    @abstractmethod
    def plan(
        self,
        obs_init: torch.Tensor,
        steps_left: Optional[int] = None,
        t0: bool = False,
        eval_mode: bool = False,
    ):
        pass

    def cost_function(
        self, actions: torch.Tensor, obs_init: torch.Tensor
    ) -> torch.Tensor:
        predicted_encs = self.unroll(obs_init, actions)
        return self.objective(predicted_encs)


### Specific planning optimizers ###
class CEMPlanner(Planner):
    def __init__(
        self,
        unroll: Callable,
        n_iters: int = 30,
        num_samples: int = 300,
        plan_length: int = 15,
        action_dim: int = 2,
        var_scale: float = 1,
        num_elites: int = 10,
        max_norms: Optional[List[float]] = None,
        max_norm_dims: Optional[List[List[int]]] = None,
        decode_each_iteration: bool = True,
        decode_loc_to_pixel: Optional[Callable] = None,
        **kwargs,
    ):
        super().__init__(unroll)
        self.n_iters = n_iters
        self.num_samples = num_samples
        self.plan_length = plan_length
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.var_scale = var_scale
        self.num_elites = num_elites
        self.max_norms = max_norms
        self.max_norm_dims = max_norm_dims
        self.decode_each_iteration = decode_each_iteration
        self.decode_loc_to_pixel = decode_loc_to_pixel

    @torch.no_grad()
    def plan(
        self, obs_init, steps_left=None, eval_mode=True, t0=False, plan_vis_path=None
    ):
        if steps_left is None:
            plan_length = self.plan_length
        else:
            plan_length = min(self.plan_length, steps_left)

        # Initialize mean and std for the action distribution
        mean = torch.zeros(plan_length, self.action_dim, device=self.device)
        std = self.var_scale * torch.ones(
            plan_length, self.action_dim, device=self.device
        )

        # Initialize actions tensor
        actions = torch.empty(
            plan_length,
            self.num_samples,
            self.action_dim,
            device=self.device,
        )

        losses = []
        elite_means = []
        elite_stds = []
        if self.decode_each_iteration:
            pred_frames_over_iterations = []
        # CEM iterations
        for _ in range(self.n_iters):
            # Sample actions
            actions[:, :] = mean.unsqueeze(1) + std.unsqueeze(1) * torch.randn(
                plan_length,
                self.num_samples,
                self.action_dim,
                device=std.device,
            )  # T B A

            # Apply clipping if max_norms is specified
            if self.max_norms is not None:
                assert len(self.max_norms) == 1
                max_norm = self.max_norms[0]
                eps = 1e-6
                norms = actions.norm(dim=-1, keepdim=True)
                # calculate min and max allowed step sizes
                max_norms = torch.ones_like(norms) * max_norm
                min_norms = torch.ones_like(norms) * 0
                coeff = torch.min(torch.max(norms, min_norms), max_norms) / (
                    norms + eps
                )
                actions = actions * coeff

            # Compute costs
            cost = self.cost_function(
                rearrange(actions, "t b a -> b a t"), obs_init
            ).unsqueeze(1)
            losses.append(cost.min().item())

            # Get elite actions
            elite_idxs = torch.topk(-cost.squeeze(1), self.num_elites, dim=0).indices
            elite_loss, elite_actions = cost[elite_idxs], actions[:, elite_idxs]

            # Record statistics
            elite_means.append(elite_loss.mean().item())
            elite_stds.append(elite_loss.std().item())

            # Update parameters
            mean = torch.mean(elite_actions, dim=1)
            std = torch.std(elite_actions, dim=1)

            if self.decode_each_iteration:
                predicted_best_encs = self.unroll(
                    obs_init, rearrange(mean, "t a -> 1 a t")
                )
                pred_frames = self.decode_loc_to_pixel(
                    predicted_best_encs,
                )
                pred_frames_over_iterations.append(pred_frames.squeeze(0))
                # [T H W 3]: uint 8 in [0, 255]
        if self.decode_each_iteration:
            save_decoded_frames(pred_frames_over_iterations, losses, plan_vis_path)

        # Return the first action(s)
        a = mean

        return PlanningResult(
            actions=a,
            losses=torch.tensor(losses).detach().unsqueeze(-1),
            prev_elite_losses_mean=torch.tensor(elite_means).unsqueeze(-1),
            prev_elite_losses_std=torch.tensor(elite_stds).unsqueeze(-1),
        )


class MPPIPlanner(Planner):
    def __init__(
        self,
        unroll: Callable,
        n_iters: int = 15,
        num_samples: int = 500,
        plan_length: int = 15,
        action_dim: int = 2,
        max_std: float = 2,
        num_elites: int = 64,
        temperature: float = 0.005,
        max_norms: Optional[List[float]] = None,
        max_norm_dims: Optional[List[List[int]]] = None,
        decode_each_iteration: bool = False,
        decode_loc_to_pixel: Optional[Callable] = None,
        **kwargs,
    ):
        super().__init__(unroll)
        self.n_iters = n_iters
        self.num_samples = num_samples
        self.plan_length = plan_length
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_std = max_std
        self.num_elites = num_elites
        self.temperature = temperature
        self.max_norms = max_norms
        self.max_norm_dims = max_norm_dims
        self.decode_each_iteration = decode_each_iteration
        self.decode_loc_to_pixel = decode_loc_to_pixel
        self._prev_mean = None

    @torch.no_grad()
    def plan(
        self, obs_init, t0=False, eval_mode=False, steps_left=None, plan_vis_path=None
    ):
        """
        Args:
                obs_init (torch.Tensor): Latent state from which to plan.
                t0 (bool): Whether this is the first observation in the episode.
                eval_mode (bool): Whether to use the mean of the action distribution.
                task (Torch.Tensor): Task index (only used for multi-task experiments).

        Returns:
                torch.Tensor: Action to take in the environment.
        """
        if steps_left is None:
            plan_length = self.plan_length
        else:
            plan_length = min(self.plan_length, steps_left)

        mean = torch.zeros(plan_length, self.action_dim, device=self.device)
        std = self.max_std * torch.ones(
            plan_length, self.action_dim, device=self.device
        )
        actions = torch.empty(
            plan_length,
            self.num_samples,
            self.action_dim,
            device=self.device,
        )

        losses = []
        elite_means = []
        elite_stds = []
        if self.decode_each_iteration:
            pred_frames_over_iterations = []

        # MPPI iterations
        for _ in range(self.n_iters):
            actions[:, :] = mean.unsqueeze(1) + std.unsqueeze(1) * torch.randn(
                plan_length,
                self.num_samples,
                self.action_dim,
                device=std.device,
            )  # T B A
            # Compute costs
            cost = self.cost_function(
                rearrange(actions, "t b a -> b a t"), obs_init
            ).unsqueeze(1)
            losses.append(cost.min().item())

            # Get elite actions
            elite_idxs = torch.topk(-cost.squeeze(1), self.num_elites, dim=0).indices
            elite_loss, elite_actions = cost[elite_idxs], actions[:, elite_idxs]

            # Record statistics
            elite_means.append(elite_loss.mean().item())
            elite_stds.append(elite_loss.std().item())

            # Update parameters
            min_cost = cost.min(0)[0]
            score = torch.exp(
                self.temperature * (min_cost - elite_loss[:, 0])
            )  # increasing with elite_value
            score /= score.sum(0)
            mean = torch.sum(
                score.unsqueeze(0).unsqueeze(2) * elite_actions, dim=1
            ) / (  # T B A
                score.sum(0) + 1e-9
            )
            std = torch.sqrt(
                torch.sum(
                    score.unsqueeze(0).unsqueeze(2)
                    * (elite_actions - mean.unsqueeze(1)) ** 2,
                    dim=1,  # T B A
                )
                / (score.sum(0) + 1e-9)
            )
            if self.decode_each_iteration:
                predicted_best_encs = self.unroll(
                    obs_init, rearrange(mean, "t a -> 1 a t")
                )
                pred_frames = self.decode_loc_to_pixel(
                    predicted_best_encs,
                )
                pred_frames_over_iterations.append(pred_frames.squeeze(0))
                # [T H W 3]: uint 8 in [0, 255]
        if self.decode_each_iteration:
            save_decoded_frames(pred_frames_over_iterations, losses, plan_vis_path)
        # Select action
        score = score.cpu().numpy()
        actions = elite_actions[
            :, np.random.choice(np.arange(score.shape[0]), p=score)
        ]  # T, A
        self._prev_mean = mean
        if not eval_mode:
            actions += std * torch.randn(
                self.action_dim, device=std.device, generator=self.local_generator
            )

        return PlanningResult(
            actions=actions,
            losses=torch.tensor(losses).detach().unsqueeze(-1),
            prev_elite_losses_mean=torch.tensor(elite_means).unsqueeze(-1),
            prev_elite_losses_std=torch.tensor(elite_stds).unsqueeze(-1),
        )
