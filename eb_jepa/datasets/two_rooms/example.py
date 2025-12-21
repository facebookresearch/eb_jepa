import os
from pathlib import Path

import yaml
from env import DotWall
from utils import update_config_from_yaml
from wall_dataset import WallDataset, WallDatasetConfig

"""
Creates a dataset loader that creates trajectories on the fly.
Can be directly used for training, or to generate an offline dataset.
"""

config_path = str(Path(__file__).parent / "data_config.yaml")

with open(config_path, "r") as file:
    yaml_data = yaml.safe_load(file)

config = update_config_from_yaml(WallDatasetConfig, yaml_data)

loader = WallDataset(config=config)

states, actions, locations, _, _ = loader[0]

print(f"states shape: {states.shape}")  # (B, C, T, H, W)
print(f"actions shape: {actions.shape}")  # (B, 2, T-1)
print(f"locations shape: {locations.shape}")  # (B, 2, T)

"""
Creates an environment. Use it to collect new data or for evaluating MPC.
"""

env = DotWall(config=config)
env.reset()

print(f"dot position: {env.dot_position}")
print(f"target position: {env.target_position}")

obs = env._get_obs()  # (C, H, W)
action = env.action_space.sample()
next_obs, reward, done, truncated, info = env.step(action)
