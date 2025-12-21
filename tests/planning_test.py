import os
from pathlib import Path
from unittest.mock import Mock, patch

import gymnasium as gym
import numpy as np
import pytest
import torch

from eb_jepa.planning import (
    CEMPlanner,
    GCAgent,
    PlanningResult,
    ReprTargetDistMPCObjective,
    main_eval,
)


def test_cem_planner():
    """Test the CEMPlanner class with various scenarios."""

    # Create a mock unroll function
    def mock_unroll(obs_init, actions):
        # Mock function that returns a tensor with shape matching the input actions
        # but with an extra dimension for the state representation
        batch_size = actions.shape[0]
        time_steps = actions.shape[2]
        return torch.zeros(batch_size, 16, time_steps, 8, 8)  # B, C, T, H, W

    # Create a mock objective function
    def mock_objective(predicted_states):
        # Return a loss value for each batch item
        return torch.sum(predicted_states, dim=(1, 2, 3, 4))

    # Test 1: Basic initialization and planning
    planner = CEMPlanner(
        unroll=mock_unroll,
        n_iters=3,
        num_samples=10,
        plan_length=5,
        action_dim=2,
        var_scale=1.0,
        num_elites=2,
        decode_each_iteration=False,
    )

    # Set the objective
    planner.set_objective(mock_objective)

    # Test planning
    obs_init = torch.zeros(1, 16, 1, 8, 8)  # B, C, T, H, W
    result = planner.plan(obs_init)

    # Check return type and shape
    assert isinstance(result, PlanningResult), "Should return a PlanningResult"
    assert result.actions.shape == (
        5,
        2,
    ), f"Actions should have shape (1, 2) but have shape {result.actions.shape}"
    assert isinstance(result.losses, torch.Tensor), "Losses should be a tensor"
    assert isinstance(
        result.prev_elite_losses_mean, torch.Tensor
    ), "Elite means should be a tensor"
    assert isinstance(
        result.prev_elite_losses_std, torch.Tensor
    ), "Elite stds should be a tensor"

    # Test 2: Planning with steps_left parameter
    result_with_steps = planner.plan(obs_init, steps_left=2)
    assert result_with_steps.actions.shape == (
        2,
        2,
    ), f"Actions should adapt to steps_left but have shape {result_with_steps.actions.shape}"

    # Test 3: Verify cost function behavior
    actions_batch = torch.randn(3, 2, 5)  # B, A, T
    cost = planner.cost_function(actions_batch, obs_init)
    assert cost.shape == (3,), "Cost should have shape (batch_size, 1)"


def test_repr_target_dist_objective():
    """Test the ReprTargetDistMPCObjective class."""
    # Create mock target representation
    target_repr = torch.ones(1, 8, 1, 8, 8)  # B, C, T, H, W

    # Initialize objective
    objective = ReprTargetDistMPCObjective(target_repr)

    # Test objective calculation
    predicted_repr = torch.zeros(2, 8, 5, 8, 8)  # B, C, T, H, W
    cost = objective(predicted_repr)

    # Check output
    assert isinstance(cost, torch.Tensor), "Should return a tensor"
    assert cost.shape == (2,), "Should return one cost per batch"
    assert torch.all(cost > 0), "Distance to non-matching target should be positive"

    # Test with matching representation
    matching_repr = torch.ones(1, 8, 5, 8, 8)  # B, C, T, H, W
    matching_cost = objective(matching_repr)
    assert (
        matching_cost.item() < cost[0].item()
    ), "Cost should be lower for matching repr"


@patch("eb_jepa.planning.CEMPlanner")
def test_gc_agent(mock_cem_planner):
    """Test the GCAgent class."""
    # Create a mock model with parameters
    mock_model = Mock()
    mock_model.encode = Mock(return_value=torch.zeros(1, 8, 1, 8, 8))
    mock_model.unrolln = Mock(return_value=torch.zeros(10, 8, 6, 8, 8))
    # Add a parameter method that returns an iterator with a device
    param = torch.nn.Parameter(torch.zeros(1))
    mock_model.parameters = Mock(return_value=iter([param]))
    device = torch.device("cpu")
    param.data = param.data.to(device)

    # Create a mock normalizer
    mock_normalizer = Mock()
    mock_normalizer.normalize_state = Mock(side_effect=lambda x: x)
    mock_normalizer.unnormalize_state = Mock(side_effect=lambda x: x)
    mock_normalizer.normalize_location = Mock(side_effect=lambda x: x)

    # Create a real PlanningResult with actual tensor data
    planning_result = PlanningResult(
        actions=torch.zeros(6, 2),  # T, A - make it big enough for any slicing
        losses=torch.zeros(10),
        prev_elite_losses_mean=torch.zeros(5),
        prev_elite_losses_std=torch.zeros(5),
    )

    # Setup mock planner
    mock_planner_instance = Mock()
    mock_planner_instance.plan = Mock(return_value=planning_result)
    mock_cem_planner.return_value = mock_planner_instance

    # Create plan config
    from omegaconf import OmegaConf

    plan_cfg = OmegaConf.create(
        {
            "planner": {
                "planner_name": "cem",
                "n_iters": 3,
                "num_samples": 10,
                "plan_length": 5,
                "num_elites": 2,
                "var_scale": 1.0,
                "decode_each_iteration": False,
                "num_act_stepped": 1,
                "planning_objective": {
                    "objective_type": "repr_dist",
                    "sum_all_diffs": True,
                },
            },
            "ctxt_window_time": 2,
        }
    )

    # Test 1: Basic initialization
    agent = GCAgent(
        mock_model,
        action_dim=2,
        plan_cfg=plan_cfg,
        normalizer=mock_normalizer,
    )

    # Test 2: Setting a goal
    goal_state = torch.randn(1, 8, 8)  # C, H, W
    goal_position = torch.tensor([4.0, 4.0])
    agent.set_goal(goal_state, goal_position)

    # Verify goal setting
    assert agent.goal_position is goal_position, "Goal position should be stored"
    assert mock_model.encode.called, "Model encode should be called when setting goal"
    assert agent.objective is not None, "Objective should be set"
    assert (
        agent.planner.set_objective.called
    ), "Planner's set_objective should be called"

    # Test 3: Acting
    obs = torch.randn(1, 8, 1, 8, 8)  # B, C, T, H, W
    action = agent.act(obs, steps_left=10)

    # Verify acting
    assert mock_planner_instance.plan.called, "Planner's plan should be called"
    assert isinstance(action, torch.Tensor), "Should return a tensor"
    assert action.shape == (1, 2), "Should return an action with shape (1, 2)"

    # Test 4: Test unroll function
    obs_init = torch.randn(1, 8, 1, 8, 8)
    actions = torch.randn(5, 2, 6)  # B, A, T
    states = agent.unroll(obs_init, actions)

    # Verify unroll behavior
    assert mock_model.unrolln.called, "Model's unroll should be called"
    assert isinstance(states, torch.Tensor), "Should return a tensor"


@patch("eb_jepa.planning.GCAgent")
def test_main_eval(mock_gc_agent):
    """Test the main_eval function."""
    num_episodes = 2

    # Create mocks
    mock_model = Mock()
    mock_env = Mock()
    mock_env.reset = Mock(
        return_value=(
            torch.zeros((2, 65, 65)),  # observation
            {
                "target_obs": torch.zeros((2, 65, 65)),
                "target_position": np.array([10.0, 10.0]),
            },  # info
        )
    )
    mock_env.step = Mock(
        return_value=(
            torch.zeros((2, 65, 65)),  # observation
            0.0,  # reward
            False,  # done
            False,  # truncated
            {
                "dot_position": np.array([5.0, 5.0]),
                "target_position": np.array([10.0, 10.0]),
                "target_obs": torch.zeros((2, 65, 65)),
            },  # info
        )
    )
    mock_env.eval_state = Mock(return_value={"success": False, "state_dist": 5.0})
    mock_env.n_steps = 10
    mock_env.n_allowed_steps = 10
    mock_env.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
    mock_env.normalizer = Mock()

    # Mock env creator function
    def mock_env_creator():
        return mock_env

    # Mock agent instance
    mock_agent_instance = Mock()
    mock_agent_instance.act = Mock(return_value=torch.tensor([[0.1, 0.2]]))
    mock_agent_instance.device = torch.device("cpu")
    mock_agent_instance.analyze_distances = Mock(return_value=([], []))
    mock_agent_instance.decode_each_iteration = False
    mock_gc_agent.return_value = mock_agent_instance

    # Create plan config
    from omegaconf import OmegaConf

    plan_cfg = {
        "planner": {
            "planner_name": "cem",
            "n_iters": 3,
            "num_samples": 10,
            "plan_length": 5,
            "num_elites": 20,
            "var_scale": 1.0,
            "num_act_stepped": 1,
            "planning_objective": {"objective_type": "repr_dist"},
        },
        "task_specification": {"goal_source": "random_state", "obs": "rgb"},
        "meta": {"eval_episodes": 2},
        "logging": {"tqdm_silent": False},
    }

    # Run evaluation with fewer episodes for testing
    os.makedirs("./logs/", exist_ok=True)
    results = main_eval(
        plan_cfg=plan_cfg,
        model=mock_model,
        env_creator=mock_env_creator,
        eval_folder=Path("./logs/"),
        num_episodes=num_episodes,
    )

    # Verify results
    assert "success_rate" in results, "Results should include success rate"
    assert "mean_state_dist" in results, "Results should include mean distance"
    assert isinstance(results["success_rate"], float), "Success rate should be a float"
    assert isinstance(
        results["mean_state_dist"], float
    ), "Mean distance should be a float"
    assert (
        mock_env.reset.call_count == 1 + num_episodes
    ), f"Environment should be reset for each episode but {mock_env.reset.call_count=}"
    assert (
        mock_agent_instance.set_goal.call_count == num_episodes
    ), "Goal should be set for each episode"


def test_planning_integration():
    """Test integration of planning components with a simplified model."""

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy_param = torch.nn.Parameter(torch.ones(1))

        def encode(self, x):
            # Simple encoding function that returns a zeroed tensor with correct shape
            B, C, T, H, W = x.shape
            return torch.zeros(B, 8, T, 4, 4, device=x.device)

        def unrolln(self, obs, actions, steps, ctxt_window_time=1):
            # Simpler unroll function that doesn't depend on complex tensor shapes
            B = obs.shape[0]
            return torch.ones(B, 8, steps, 4, 4, device=obs.device)

    # Test full planning episode
    # Create model and move to appropriate device
    model = DummyModel()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Create a mock normalizer
    mock_normalizer = Mock()
    mock_normalizer.normalize_state = Mock(side_effect=lambda x: x)
    mock_normalizer.unnormalize_state = Mock(side_effect=lambda x: x)
    mock_normalizer.normalize_location = Mock(side_effect=lambda x: x)

    # Create plan config
    from omegaconf import OmegaConf

    plan_cfg = OmegaConf.create(
        {
            "planner": {
                "planner_name": "cem",
                "n_iters": 3,
                "num_samples": 10,
                "plan_length": 4,
                "num_elites": 2,
                "var_scale": 1.0,
                "decode_each_iteration": False,
                "num_act_stepped": 1,
                "planning_objective": {
                    "objective_type": "repr_dist",
                    "sum_all_diffs": True,
                },
            },
            "ctxt_window_time": 2,
        }
    )

    # Initialize agent
    agent = GCAgent(
        model,
        action_dim=2,
        plan_cfg=plan_cfg,
        normalizer=mock_normalizer,
    )

    # Set goal
    goal_state = torch.ones(1, 4, 4).to(device)
    goal_position = torch.tensor([1.0, 1.0]).to(device)
    agent.set_goal(goal_state, goal_position)

    # Create observation
    obs = torch.zeros(1, 1, 1, 4, 4).to(device)

    # Test planning and action selection
    action = agent.act(obs, steps_left=8)

    # Verify action
    assert isinstance(action, torch.Tensor), "Should return a tensor"
    assert action.shape == (1, 2), "Should return appropriate action shape"

    # Since our dummy model favors larger actions, the planned actions should have magnitude > 0
    assert torch.sum(torch.abs(action)) > 0, "Agent should select non-zero actions"
