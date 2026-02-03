"""
Unified SLURM launcher for all EB-JEPA examples.

Provides seed averaging, sweep name filtering, and wandb sweep features for all examples.

USAGE:
------
# Launch 3 seeds of a single configuration (default sweep name: sweep_YYYYMMDD_HHMM):
python -m examples.launch_sbatch --example ac_video_jepa

# Launch 3 seeds with custom sweep name:
python -m examples.launch_sbatch --example ac_video_jepa --sweep my_experiment

# Launch full hyperparameter sweep (ac_video_jepa only):
python -m examples.launch_sbatch --example ac_video_jepa --sweep my_experiment --full-sweep

# With wandb sweep UI for hyperparameter analysis:
python -m examples.launch_sbatch --example ac_video_jepa --sweep my_experiment --use-wandb-sweep

# Override config values:
python -m examples.launch_sbatch --example ac_video_jepa --optim.lr 0.0005

SEED AVERAGING IN WANDB UI:
---------------------------
Runs with the same hyperparameters but different seeds share the same wandb run name.

To view averaged metrics:
1. Go to wandb web UI -> Runs table
2. Click "Group by" -> select "Name"
   -> This groups runs with identical hyperparameters (different seeds) together

To filter runs from a specific sweep:
3. Click "Filter" -> "Group" -> select your sweep name (e.g., 'my_experiment')
   -> This shows only runs from that sweep, grouped by name (see above)

WANDB SWEEP ANALYSIS UI (requires --use-wandb-sweep):
-----------------------------------------------------
When using --use-wandb-sweep, wandb creates a sweep object that enables advanced
hyperparameter analysis.

To access the sweep analysis:
1. Go to wandb web UI -> left pane -> click "Sweeps"
2. Click on your sweep name
3. Wandb automatically generates plots linking hyperparameters to the metric
   (success_rate), including:
   - Parallel coordinates plot
   - Hyperparameter importance
   - Parameter vs. metric scatter plots
"""

import argparse
import importlib
import json
import os
import shutil
from itertools import product

import submitit
import wandb

from eb_jepa.training_utils import (
    get_checkpoints_dir,
    get_default_dev_name,
    get_default_sweep_name,
    get_exp_name,
    get_unified_experiment_dir,
    load_config,
)

# Default SLURM parameters
SLURM_DEFAULTS = {
    "mem_per_gpu": "210G",
    "cpus_per_task": 16,
    "timeout_min": 24 * 60,
    "partition": "learn",
    "gpus_per_node": 1,
    "qos": "lowest",
    "account": "fair_amaia_cw_video",
}


# Example-specific configurations
EXAMPLE_CONFIGS = {
    "image_jepa": {
        "config": "examples/image_jepa/cfgs/default.yaml",
        "module": "examples.image_jepa.main",
        "metric": "val_acc",
    },
    "video_jepa": {
        "config": "examples/video_jepa/cfgs/default.yaml",
        "module": "examples.video_jepa.main",
        "metric": "AP_1",
    },
    "ac_video_jepa": {
        "config": "examples/ac_video_jepa/cfgs/train.yaml",
        "module": "examples.ac_video_jepa.main",
        "metric": "success_rate",
    },
}

# =============================================================================
# Utility functions
# =============================================================================


def make_executor(
    folder: str,
    job_name: str,
    array_parallelism: int | None = None,
) -> submitit.AutoExecutor:
    """Create a submitit executor with standard SLURM parameters."""
    executor = submitit.AutoExecutor(folder=folder, slurm_max_num_timeout=20)

    params = {
        "name": job_name,
        "slurm_mem_per_gpu": SLURM_DEFAULTS["mem_per_gpu"],
        "cpus_per_task": SLURM_DEFAULTS["cpus_per_task"],
        "timeout_min": SLURM_DEFAULTS["timeout_min"],
        "slurm_partition": SLURM_DEFAULTS["partition"],
        "slurm_additional_parameters": {
            "nodes": 1,
            "ntasks-per-node": 1,
            "gpus-per-node": SLURM_DEFAULTS["gpus_per_node"],
            "qos": SLURM_DEFAULTS["qos"],
            "account": SLURM_DEFAULTS["account"],
        },
    }

    if array_parallelism is not None:
        params["slurm_array_parallelism"] = array_parallelism

    executor.update_parameters(**params)
    return executor


def normalize_sweep_name(name: str) -> str:
    """Ensure sweep name has 'sweep_' prefix for consistency."""
    if name.startswith("sweep_"):
        return name
    return f"sweep_{name}"


def copy_code_folder(code_folder):
    """Copy the code folder to the experiment directory, ignoring unnecessary files."""
    # Patterns to always ignore (matched by name only)
    ignore_patterns = [
        "__pycache__",
        ".vscode",
        ".git",
        "core",
        "uv.lock",
        "Makefile",
    ]
    # Paths to ignore (matched by name only, applies to any directory with this name)
    ignore_paths = [
        "traces",
        "docs",
        ".pytest_cache",
        "logs",
        ".venv",
        "eb_jepa.egg-info",
        "wandb",
        "assets",
    ]
    # Root-level directories to ignore (only ignored when at the source root)
    # This allows us to skip ./datasets (storage-intensive data) while keeping
    # ./eb_jepa/datasets (data code needed for experiments)
    root_only_ignore = [
        "eb_jepa_ICLR",
        "datasets",
        "checkpoints",
    ]
    source_root = os.path.abspath(".")

    def ignore_func(path, names):
        ignored = []
        for n in names:
            if n in ignore_patterns or n in ignore_paths:
                ignored.append(n)
            # Only ignore root-level directories specified in root_only_ignore
            elif n in root_only_ignore and os.path.abspath(path) == source_root:
                ignored.append(n)
        return ignored

    if not os.path.exists(code_folder):
        shutil.copytree(".", code_folder, ignore=ignore_func)


def setup_launch_environment(base_dir, logs_subdir: str | None = "slurm_logs"):
    """Setup directories and code folder for launching jobs."""
    base_dir = base_dir.absolute() if hasattr(base_dir, "absolute") else base_dir
    logs_dir = base_dir / logs_subdir if logs_subdir else base_dir
    code_folder = base_dir / "code"

    copy_code_folder(str(code_folder))
    logs_dir.mkdir(parents=True, exist_ok=True)

    print(f"Code folder: {code_folder}")
    os.chdir(code_folder)

    return logs_dir, code_folder


def generate_param_combinations(param_grid: dict):
    """Generate all parameter combinations from a grid."""
    param_names = list(param_grid.keys())
    param_values_list = list(param_grid.values())
    all_combinations = list(product(*param_values_list))
    return param_names, all_combinations


def print_submission_summary(jobs: list, logs_dir, extra_info: dict | None = None):
    """Print a compact summary of batch job submission."""
    job_ids = [job.job_id for job in jobs]
    batch_id = job_ids[0].split("_")[0] if "_" in job_ids[0] else job_ids[0]
    print(f"\n✓ Submitted {len(jobs)} jobs (batch {batch_id}_[0-{len(jobs)-1}])")
    print(f"  Logs: {logs_dir}")
    if extra_info:
        for key, value in extra_info.items():
            print(f"  {key}: {value}")


# =============================================================================
# Launch functions
# =============================================================================


def run_experiment(example_name: str, cfg, folder=None):
    """Run the appropriate example based on example_name."""
    print(f"Current working directory: {os.getcwd()}")
    print(f"EBJEPA_DSETS: {os.environ.get('EBJEPA_DSETS', 'not set')}")
    module = importlib.import_module(EXAMPLE_CONFIGS[example_name]["module"])
    return module.run(cfg=cfg, folder=folder)


def launch_job(example_name: str, fname: str, **kwargs):
    """Launch a single training job with the given config and overrides."""
    cfg = load_config(fname, kwargs)
    sweep_name = kwargs.get("sweep_name", get_default_sweep_name())
    exp_name = get_exp_name(example_name, cfg)

    folder = get_unified_experiment_dir(
        example_name=example_name,
        sweep_name=sweep_name,
        exp_name=exp_name,
        seed=cfg.meta.seed,
    )

    logs_dir, _ = setup_launch_environment(folder, logs_subdir=None)

    executor = make_executor(
        folder=str(logs_dir),
        job_name=f"{example_name.upper()}",
    )
    job = executor.submit(run_experiment, example_name, cfg, folder)

    print(f"\n✓ Submitted job {job.job_id}")
    print(f"  Experiment folder: {folder}")

    return job


def create_wandb_sweep_config(param_grid: dict, metric: str, method: str = "grid"):
    """Create a wandb sweep configuration from a parameter grid."""
    sweep_config = {
        "method": method,
        "metric": {"goal": "maximize", "name": metric},
        "parameters": {},
    }

    for param_name, param_values in param_grid.items():
        if isinstance(param_values, list):
            sweep_config["parameters"][param_name] = {"values": param_values}
        elif isinstance(param_values, dict):
            sweep_config["parameters"][param_name] = param_values

    return sweep_config


def launch_sweep(
    example_name: str,
    fname: str,
    param_grid: dict,
    array_parallelism: int = 256,
    use_wandb: bool = False,
    wandb_method: str = "grid",
    **base_overrides,
):
    """Launch a parameter sweep using submitit. Returns (sweep_id, jobs) if use_wandb else jobs."""
    param_names, all_combinations = generate_param_combinations(param_grid)

    if not all_combinations:
        print("No parameter combinations to sweep")
        return (None, []) if use_wandb else []

    sweep_name = base_overrides.get("sweep_name", get_default_sweep_name())

    # Create wandb sweep if requested
    sweep_id = None
    if use_wandb:
        project_name = "eb_jepa"
        metric = EXAMPLE_CONFIGS[example_name]["metric"]
        sweep_config = create_wandb_sweep_config(param_grid, metric, wandb_method)
        sweep_id = wandb.sweep(sweep_config, project=project_name)
        print(f"Created wandb sweep with ID: {sweep_id}")
        print(
            f"View sweep at: https://wandb.ai/{wandb.api.default_entity}/{project_name}/sweeps/{sweep_id}"
        )

    # Setup environment (must happen before chdir)
    common_dir = get_checkpoints_dir() / example_name / sweep_name
    logs_subdir = "wandb_sweep_slurm_logs" if use_wandb else "sweep_slurm_logs"
    logs_dir, _ = setup_launch_environment(common_dir, logs_subdir=logs_subdir)

    # Store checkpoints dir before chdir (for absolute paths in job configs)
    original_checkpoints_dir = common_dir.parent.parent.absolute()

    executor = make_executor(
        folder=str(logs_dir),
        job_name=f"{example_name.upper()}_{'wandb_' if use_wandb else ''}sweep",
        array_parallelism=array_parallelism,
    )

    print(f"\nPreparing {len(all_combinations)} tasks...")
    jobs = []
    with executor.batch():
        for values in all_combinations:
            param_overrides = dict(zip(param_names, values))
            final_overrides = {**base_overrides, **param_overrides}

            # Add wandb-specific overrides
            if use_wandb:
                final_overrides.update(
                    {
                        "logging.wandb_sweep": True,
                        "logging.wandb_sweep_id": sweep_id,
                        "logging.wandb_group": sweep_name,
                    }
                )

            cfg = load_config(fname, final_overrides, quiet=True)
            exp_name = get_exp_name(example_name, cfg)
            folder = get_unified_experiment_dir(
                example_name=example_name,
                sweep_name=sweep_name,
                exp_name=exp_name,
                seed=cfg.meta.seed,
                base_dir=original_checkpoints_dir,
            )

            job = executor.submit(run_experiment, example_name, cfg, folder)
            jobs.append(job)

    extra_info = {"Sweep ID": sweep_id} if use_wandb else None
    print_submission_summary(jobs, logs_dir, extra_info)

    return (sweep_id, jobs) if use_wandb else jobs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Unified SLURM launcher for EB-JEPA examples"
    )
    parser.add_argument(
        "--example",
        type=str,
        required=True,
        choices=["image_jepa", "video_jepa", "ac_video_jepa"],
        help="Which example to run",
    )
    parser.add_argument(
        "--fname",
        type=str,
        default=None,
        help="Path to config file (defaults to example's default config)",
    )
    parser.add_argument(
        "--sweep",
        type=str,
        default=None,
        help="Name for the sweep (default: sweep_YYYYMMDD_HHMM)",
    )
    parser.add_argument(
        "--array-parallelism",
        type=int,
        default=256,
        help="Number of jobs to run in parallel for the sweep",
    )
    parser.add_argument(
        "--use-wandb-sweep",
        action="store_true",
        help="Use wandb sweep for hyperparameter tracking",
    )
    parser.add_argument(
        "--sweep-method",
        type=str,
        default="grid",
        choices=["grid", "random", "bayes"],
        help="Wandb sweep method to use if use_wandb_sweep is true",
    )
    parser.add_argument(
        "--full-sweep",
        action="store_true",
        help="Enable full hyperparameter sweep (default: only sweep over 3 seeds)",
    )
    parser.add_argument(
        "--single",
        action="store_true",
        help="Launch a single job (uses dev_YYYYMMDD_HHMM folder)",
    )

    # Common overrides
    parser.add_argument("--optim.lr", type=float)
    parser.add_argument("--meta.seed", type=int)

    # ac_video_jepa specific
    parser.add_argument("--model.regularizer.cov_coeff", type=float)
    parser.add_argument("--model.regularizer.std_coeff", type=float)
    parser.add_argument("--model.regularizer.sim_coeff_t", type=float)
    parser.add_argument("--model.regularizer.idm_coeff", type=float)

    # Use parse_known_args to allow dynamic overrides for any config key
    args, unknown = parser.parse_known_args()

    example_name = args.example
    example_config = EXAMPLE_CONFIGS[example_name]
    fname = args.fname or example_config["config"]

    # Load config to read sweep params from YAML (quiet mode to avoid duplicate logs)
    base_cfg = load_config(fname, {}, quiet=True)

    # Read sweep param_grid from config file
    # Fall back to default 3-seed sweep if not specified in config
    config_param_grid = base_cfg.get("sweep", {}).get("param_grid", {})
    if hasattr(config_param_grid, "to_dict"):
        config_param_grid = config_param_grid.to_dict()
    elif hasattr(config_param_grid, "__dict__"):
        # OmegaConf DictConfig - convert to plain dict
        config_param_grid = dict(config_param_grid)

    default_seed_sweep = {"meta.seed": [1, 1000, 10000]}

    # Build overrides dict from known args
    excluded_keys = {
        "example",
        "fname",
        "sweep",
        "array_parallelism",
        "use_wandb_sweep",
        "sweep_method",
        "full_sweep",
        "single",
    }
    overrides = {
        k: v for k, v in vars(args).items() if v is not None and k not in excluded_keys
    }

    # Parse unknown args as additional config overrides (e.g., --data.batch_size 64)
    i = 0
    while i < len(unknown):
        if unknown[i].startswith("--"):
            key = unknown[i][2:]
            if i + 1 < len(unknown) and not unknown[i + 1].startswith("--"):
                value = unknown[i + 1]
                # Try to parse as JSON (handles numbers, bools, lists)
                try:
                    value = json.loads(value)
                except json.JSONDecodeError:
                    pass  # Keep as string
                overrides[key] = value
                i += 2
            else:
                # Flag without value (e.g., --some_flag)
                overrides[key] = True
                i += 1
        else:
            i += 1

    # Determine folder name based on mode
    if args.single:
        # Single job: use dev_ prefix
        sweep_name = get_default_dev_name()
        param_grid = None  # No sweep, single job
    elif args.sweep:
        # Custom sweep name: normalize to have sweep_ prefix
        sweep_name = normalize_sweep_name(args.sweep)
        if args.full_sweep:
            param_grid = config_param_grid if config_param_grid else default_seed_sweep
        else:
            param_grid = default_seed_sweep
    else:
        # Default: 3-seed sweep with auto-generated name
        sweep_name = get_default_sweep_name()
        if args.full_sweep:
            param_grid = config_param_grid if config_param_grid else default_seed_sweep
        else:
            param_grid = default_seed_sweep

    overrides["sweep_name"] = sweep_name
    overrides["logging.wandb_group"] = sweep_name

    print(f"Example: {example_name}")
    print(f"Config: {fname}")
    print(f"Sweep name: {sweep_name}")
    if param_grid:
        print(f"Param grid: {param_grid}")
    else:
        print("Mode: single job")
    if overrides:
        print(f"Overrides: {overrides}")

    if args.single:
        # Launch single job
        job = launch_job(example_name, fname, **overrides)
    elif args.use_wandb_sweep:
        sweep_id, jobs = launch_sweep(
            example_name,
            fname,
            param_grid,
            array_parallelism=args.array_parallelism,
            use_wandb=True,
            wandb_method=args.sweep_method,
            **overrides,
        )
    else:
        jobs = launch_sweep(
            example_name,
            fname,
            param_grid,
            array_parallelism=args.array_parallelism,
            **overrides,
        )
