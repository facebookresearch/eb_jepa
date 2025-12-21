import argparse
import importlib
import os
import shutil
from pathlib import Path

import submitit

from examples.ac_video_jepa.main import get_experiment_folder, load_override_cfg


def copy_code_folder(code_folder):
    ignore_patterns = [
        "__pycache__",
        ".vscode",
        ".git",
        "core",
        "mnist_test_seq.npy",
        "uv.lock",
        "Makefile",
    ]
    ignore_paths = [
        "traces",
        "docs",
        ".pytest_cache",
        "logs",
        ".venv",
        "eb_jepa.egg-info",
    ]
    path_to_node_folder = {}

    for path in ignore_paths:
        split_path = path.split("/")
        base_path = "/".join(split_path[:-1])
        node_folder = split_path[-1]
        path_to_node_folder[base_path] = node_folder

    def ignore_func(path, names):
        ignore_list = list(ignore_patterns)
        for ignore_path in ignore_paths:
            if ignore_path in names:
                ignore_list.append(ignore_path)

        return ignore_list

    if not os.path.exists(code_folder):
        shutil.copytree(".", code_folder, ignore=ignore_func)


def launch_job(fname: str, **kwargs):
    """
    Launch a single job with the given config and overrides.

    Args:
        fname: Path to the YAML config file
        **kwargs: Configuration overrides
    """
    cfg = load_override_cfg(fname, kwargs)
    folder = get_experiment_folder(
        cfg, cfg.data, quick_debug=cfg.meta.get("quick_debug")
    )
    os.makedirs(folder, exist_ok=True)

    code_folder = os.path.join(folder, "code")
    copy_code_folder(code_folder)
    print(f"Changing to code folder: {code_folder}")
    os.chdir(code_folder)
    executor = submitit.AutoExecutor(
        folder=os.path.join(folder, "job_%j"), slurm_max_num_timeout=20
    )

    executor.update_parameters(
        name="AC_JEPA",
        slurm_mem_per_gpu="55G",
        cpus_per_task=16,
        timeout_min=24 * 60,
        slurm_partition="learn",
        slurm_additional_parameters={
            "nodes": 1,
            "ntasks-per-node": 1,
            "gpus-per-node": 1,
            "qos": "explore",
            "account": "fair_amaia_cw_video",
        },
    )
    job = executor.submit(run_experiment, cfg, folder)

    print(f"Submitted job {job.job_id}")
    print(f"Experiment folder: {folder}")

    return job


def run_experiment(cfg, folder=None):
    """
    The actual experiment function that will be executed on the cluster.

    Args:
        cfg: Pre-loaded configuration object (OmegaConf)
        code_folder: Path to the copied code folder
    """
    print(f"Current working directory: {os.getcwd()}")
    return importlib.import_module("examples.ac_video_jepa.main").main(
        cfg=cfg, folder=folder
    )


def launch_sweep(
    fname: str, param_grid: dict, array_parallelism: int, **base_overrides
):
    """
    Launch a parameter sweep using submitit batch submission.

    Args:
        config_path: Path to the base config file
        param_grid: Dictionary of parameter names to lists of values
        **base_overrides: Base configuration overrides to apply to all jobs
    """
    from itertools import product

    param_names = list(param_grid.keys())
    param_values_list = list(param_grid.values())
    all_combinations = list(product(*param_values_list))

    if not all_combinations:
        print("No parameter combinations to sweep")
        return []

    base_cfg = load_override_cfg(fname, base_overrides)
    common_ckpt_dir = Path(base_cfg.meta.ckpt_dir)
    sweep_logs_dir = common_ckpt_dir / "sweep_slurm_logs"
    sweep_logs_dir.mkdir(parents=True, exist_ok=True)

    sweep_code_folder = common_ckpt_dir / "code"
    copy_code_folder(str(sweep_code_folder))
    print(f"Changing to code folder: {sweep_code_folder}")
    os.chdir(sweep_code_folder)

    executor = submitit.AutoExecutor(
        folder=str(sweep_logs_dir), slurm_max_num_timeout=20
    )

    executor.update_parameters(
        name="AC_JEPA_sweep",
        slurm_mem_per_gpu="55G",
        cpus_per_task=16,
        timeout_min=24 * 60,
        slurm_partition="learn",
        slurm_array_parallelism=array_parallelism,
        slurm_additional_parameters={
            "nodes": 1,
            "ntasks-per-node": 1,
            "gpus-per-node": 1,
            "qos": "explore",
            "account": "fair_amaia_cw_video",
        },
    )
    jobs = []
    with executor.batch():
        for i, values in enumerate(all_combinations):
            param_overrides = dict(zip(param_names, values))
            final_overrides = {**base_overrides, **param_overrides}
            cfg = load_override_cfg(fname, final_overrides)
            folder = get_experiment_folder(
                cfg, cfg.data, quick_debug=cfg.meta.get("quick_debug")
            )
            os.makedirs(folder, exist_ok=True)
            print(f"Submitting task {i}: {param_overrides}")
            print(f"  -> folder: {folder}")
            job = executor.submit(run_experiment, cfg, folder)
            jobs.append(job)

    print(f"Submitted {len(jobs)} jobs in batch")
    print(f"Sweep logs directory: {sweep_logs_dir}")

    return jobs


def create_wandb_sweep_config(param_grid: dict, method: str = "grid"):
    """
    Create a wandb sweep configuration from a parameter grid.

    Args:
        param_grid: Dictionary of parameter names to lists of values
        method: Sweep method ("grid", "random", or "bayes")

    Returns:
        Dictionary representing the wandb sweep configuration
    """
    sweep_config = {
        "method": method,
        "metric": {
            "goal": "maximize",
            "name": "success_rate",
        },
        "parameters": {},
    }

    for param_name, param_values in param_grid.items():
        if isinstance(param_values, list):
            sweep_config["parameters"][param_name] = {"values": param_values}
        elif isinstance(param_values, dict):
            sweep_config["parameters"][param_name] = param_values

    return sweep_config


def launch_wandb_sweep(
    fname: str,
    param_grid: dict,
    method: str = "grid",
    array_parallelism: int = 256,
    **base_overrides,
):
    """
    Launch a wandb sweep using submitit. Each SLURM job = 1 training run.

    Creates a wandb sweep for tracking/visualization, then launches individual
    SLURM jobs with deterministic run IDs for proper resuming after requeues.

    Args:
        fname: Path to the base config file
        param_grid: Dictionary of parameter names to lists of values
        method: Sweep method ("grid", "random", or "bayes")
        array_parallelism: Number of jobs to run in parallel
        **base_overrides: Base configuration overrides to apply to all jobs
    """
    from itertools import product

    import wandb

    # Load base config
    base_cfg = load_override_cfg(fname, base_overrides)
    project_name = "eb-jepa-ac"

    # Create wandb sweep configuration
    sweep_config = create_wandb_sweep_config(param_grid, method)

    # Initialize the sweep (creates it on wandb servers)
    sweep_id = wandb.sweep(sweep_config, project=project_name)
    print(f"Created wandb sweep with ID: {sweep_id}")
    print(
        f"View sweep at: https://fairwandb.org/{wandb.api.default_entity}/{project_name}/sweeps/{sweep_id}"
    )

    # Generate all parameter combinations (same as regular sweep)
    param_names = list(param_grid.keys())
    param_values_list = list(param_grid.values())
    all_combinations = list(product(*param_values_list))

    if not all_combinations:
        print("No parameter combinations to sweep")
        return sweep_id, []

    # Set up directories
    common_ckpt_dir = Path(base_cfg.meta.ckpt_dir)
    sweep_logs_dir = common_ckpt_dir / "wandb_sweep_slurm_logs"
    sweep_logs_dir.mkdir(parents=True, exist_ok=True)

    sweep_code_folder = common_ckpt_dir / "code"
    copy_code_folder(str(sweep_code_folder))
    print(f"Changing to code folder: {sweep_code_folder}")
    os.chdir(sweep_code_folder)

    # Set up submitit executor
    executor = submitit.AutoExecutor(
        folder=str(sweep_logs_dir), slurm_max_num_timeout=20
    )

    executor.update_parameters(
        name="AC_JEPA_wandb_sweep",
        slurm_mem_per_gpu="55G",
        cpus_per_task=16,
        timeout_min=24 * 60,
        slurm_partition="learn",
        slurm_array_parallelism=array_parallelism,
        slurm_additional_parameters={
            "nodes": 1,
            "ntasks-per-node": 1,
            "gpus-per-node": 1,
            "qos": "explore",
            "account": "fair_amaia_cw_video",
        },
    )

    # Launch one SLURM job per training run (maintaining 1:1 convention)
    jobs = []
    with executor.batch():
        for i, values in enumerate(all_combinations):
            param_overrides = dict(zip(param_names, values))
            final_overrides = {
                **base_overrides,
                **param_overrides,
                "logging.wandb_sweep": True,
                "logging.wandb_sweep_id": sweep_id,  # Pass sweep_id as config param
            }
            cfg = load_override_cfg(fname, final_overrides)
            folder = get_experiment_folder(
                cfg, cfg.data, quick_debug=cfg.meta.get("quick_debug")
            )
            os.makedirs(folder, exist_ok=True)

            print(f"Submitting job {i}: {param_overrides}")
            print(f"  -> folder: {folder}")
            job = executor.submit(run_experiment, cfg, folder)
            jobs.append(job)

    print(f"Submitted {len(jobs)} jobs (1 job = 1 training run)")
    print(f"Sweep logs directory: {sweep_logs_dir}")
    print(f"Sweep ID: {sweep_id}")

    return sweep_id, jobs


if __name__ == "__main__":
    """
    Single run
        python examples/ac_video_jepa/launch_sbatch.py --fname examples/ac_video_jepa/cfgs/train.yaml --optim.lr 0.0005 --model.regularizer.sim_coeff_t 0.75
    Sweep with common directory name (like run_exp.sh)
        python examples/ac_video_jepa/launch_sbatch.py --sweep randwall_imp_fixeval_sweep
        python examples/ac_video_jepa/launch_sbatch.py --sweep rand_conv3d_bs128_sweep --fname examples/ac_video_jepa/cfgs/train_conv3d.yaml --array_parallelism 1200
        python examples/ac_video_jepa/launch_sbatch.py --sweep rand_imp_bs384AdamW_losssweep --fname examples/ac_video_jepa/cfgs/train.yaml --array_parallelism 1200 --use_wandb_sweep
    """
    parser = argparse.ArgumentParser(description="Submitit launcher for AC Video JEPA")
    parser.add_argument(
        "--fname",
        default="examples/ac_video_jepa/cfgs/train.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--sweep",
        type=str,
        help="Name for the sweep (sets meta.ckpt_dir and logging.wandb_group)",
    )
    parser.add_argument(
        "--array_parallelism",
        type=int,
        default=256,
        help="Number of jobs to run in parallel for the sweep",
    )
    parser.add_argument(
        "--use_wandb_sweep",
        action="store_true",
        help="Use wandb sweep instead of submitit sweep",
    )
    parser.add_argument(
        "--sweep_method",
        type=str,
        default="grid",
        choices=["grid", "random", "bayes"],
        help="Wandb sweep method (grid, random, or bayes)",
    )
    parser.add_argument("--model.regularizer.cov_coeff", type=float)
    parser.add_argument("--model.regularizer.std_coeff", type=float)
    parser.add_argument("--model.regularizer.sim_coeff_t", type=float)
    parser.add_argument("--model.regularizer.idm_coeff", type=float)
    parser.add_argument("--model.regularizer.spatial_as_samples", type=bool)
    parser.add_argument("--optim.lr", type=float)
    parser.add_argument("--meta.quick_debug", type=bool)
    args = parser.parse_args()
    overrides = {
        k: v
        for k, v in vars(args).items()
        if v is not None
        and k
        not in [
            "fname",
            "sweep",
            "array_parallelism",
            "use_wandb_sweep",
            "sweep_method",
        ]
    }
    if args.sweep is not None:
        overrides["meta.ckpt_dir"] = (
            f"/checkpoint/amaia/video/basileterv/experiment/eb_jepa/{args.sweep}"
        )
        overrides["logging.wandb_group"] = args.sweep
        # ######### HARDCODE PARAM GRID FOR SWEEP ##########
        param_grid = {
            # "model.regularizer.spatial_as_samples": [False],
            # "model.regularizer.cov_coeff": [8, 12],
            # "model.regularizer.std_coeff": [8, 16],
            # "model.regularizer.sim_coeff_t": [8, 12, 16],
            # "model.regularizer.idm_coeff": [1, 2],
            # "model.regularizer.cov_coeff": [1, 2],
            "meta.seed": [1, 1000, 10000],
            # "model.regularizer.first_t_only": [True, False],
        }
        # param_grid = {
        #     "optim.weight_decay": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
        #     "optim.lr": [0.00005, 0.0001, 0.0005, 0.001, 0.005],
        #     "meta.seed": [1, 1000, 10000],  # Add as many seeds as you want
        # "model.train_rollout": ["last", "all"]
        # "optim.lr": [0.0001, 0.0003, 0.0007, 0.001, 0.003, 0.007],
        # "optim.grad_clip_enc": [2, 4],
        # "optim.grad_clip_pred": [2, 4],
        # }
        # ########################################################
        base_fname = args.fname
        if overrides:
            print(f"Base overrides: {overrides}")

        if args.use_wandb_sweep:
            sweep_id, jobs = launch_wandb_sweep(
                base_fname,
                param_grid,
                method=args.sweep_method,
                array_parallelism=args.array_parallelism,
                **overrides,
            )
        else:
            jobs = launch_sweep(
                base_fname, param_grid, args.array_parallelism, **overrides
            )
    else:
        job = launch_job(args.fname, **overrides)
