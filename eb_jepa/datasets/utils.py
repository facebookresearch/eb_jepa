from pathlib import Path

import torch
import yaml

from eb_jepa.datasets.two_rooms.utils import update_config_from_yaml
from eb_jepa.datasets.two_rooms.wall_dataset import WallDataset, WallDatasetConfig

DATASETS_DIR = Path(__file__).parent


def load_env_data_config(env_name: str, overrides: dict = None) -> dict:
    """Load base data config for an environment and apply overrides."""
    config_path = DATASETS_DIR / env_name / "data_config.yaml"
    with open(config_path) as f:
        base_config = yaml.safe_load(f)
    if overrides:
        base_config.update(overrides)
    return base_config


def init_data(env_name, cfg_data=None, **kwargs):
    """Initialize data loaders for the specified environment.

    Loads base config from eb_jepa/datasets/{env_name}/data_config.yaml
    and merges with any overrides from cfg_data.

    Args:
        env_name: Name of the environment (currently only "two_rooms" is supported).
        cfg_data: Configuration overrides for the dataset.

    Returns:
        Tuple of (train_loader, val_loader, config).
    """
    if env_name != "two_rooms":
        raise ValueError(f"Unknown env: {env_name}. Only 'two_rooms' is supported.")

    merged_cfg = load_env_data_config(env_name, cfg_data)
    config = update_config_from_yaml(WallDatasetConfig, merged_cfg)

    num_workers = merged_cfg.get("num_workers", 0)
    pin_mem = merged_cfg.get("pin_mem", False)
    persistent_workers = merged_cfg.get("persistent_workers", False) and num_workers > 0

    dset = WallDataset(config=config)
    loader = torch.utils.data.DataLoader(
        dset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_mem,
        drop_last=True,
        persistent_workers=persistent_workers,
    )

    val_dset = WallDataset(config=config)
    val_loader = torch.utils.data.DataLoader(
        val_dset,
        batch_size=4,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_mem,
        drop_last=True,
        persistent_workers=persistent_workers,
    )

    return loader, val_loader, config
