from logging import getLogger
from typing import Callable

import torch

from eb_jepa.datasets.two_rooms.utils import update_config_from_yaml

from .pusht.pusht_dset import load_pusht_slice_train_val
from .two_rooms.wall_dataset import WallDataset, WallDatasetConfig

logger = getLogger()


def init_data(
    env_name,
    frameskip=None,
    action_skip=1,
    transform=None,
    normalize_action=True,  # normalized both action and proprio when building TrajDataset
    num_hist=3,
    num_pred=1,
    num_frames_val=None,
    seed=42,
    process_actions="concat",
    split_ratio=0.9,
    cfg_data=None,
    **kwargs,
) -> tuple[Callable]:
    if env_name == "pusht":
        datasets, traj_dsets = load_pusht_slice_train_val(
            transform,
            n_rollout=None,
            data_path=data_paths[0],
            normalize_action=normalize_action,
            split_ratio=split_ratio,
            # num_frames=16,
            num_hist=num_hist,
            num_pred=num_pred,
            num_frames_val=num_frames_val,
            frameskip=frameskip,
            action_skip=action_skip,
            with_velocity=True,
            random_seed=seed,
            process_actions=process_actions,
        )
    elif env_name == "two_rooms":
        config = update_config_from_yaml(WallDatasetConfig, cfg_data)
        dset = WallDataset(config=config)
        loader = torch.utils.data.DataLoader(
            dset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=cfg_data.get("num_workers"),
            pin_memory=cfg_data.get("pin_mem"),
            drop_last=True,
            persistent_workers=cfg_data.get("persistent_workers")
            and cfg_data.get("num_workers") > 0,
        )
        val_dset = WallDataset(config=config)
        val_loader = torch.utils.data.DataLoader(
            val_dset,
            batch_size=4,  # TODO: hardcoded for now
            shuffle=False,
            num_workers=cfg_data.get("num_workers"),
            pin_memory=cfg_data.get("pin_mem"),
            drop_last=True,
            persistent_workers=cfg_data.get("persistent_workers")
            and cfg_data.get("num_workers") > 0,
        )
    else:
        raise Exception(f"Unknown env: {env_name}")
    return loader, val_loader, config
