"""Measure original MPPI planning calls without changing the planner."""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint-folder', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    source = Path(args.checkpoint_folder).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)

    from eb_jepa.planning import MPPIPlanner
    from eb_jepa.training_utils import load_config
    from examples.ac_video_jepa import main as training

    original_plan = MPPIPlanner.plan
    rows = []
    csv_path = output / 'planning_per_timestep.csv'
    fields = ['timestep', 'steps_left', 'horizon', 'planning_seconds']
    with csv_path.open('w', newline='') as handle:
        csv.DictWriter(handle, fieldnames=fields).writeheader()

    def timed_plan(planner, obs_init, *positional, **kwargs):
        if obs_init.is_cuda:
            torch.cuda.synchronize(obs_init.device)
        start = time.perf_counter()
        result = original_plan(planner, obs_init, *positional, **kwargs)
        if obs_init.is_cuda:
            torch.cuda.synchronize(obs_init.device)
        row = dict(timestep=len(rows), steps_left=kwargs.get('steps_left'),
                   horizon=min(planner.plan_length, kwargs.get('steps_left', planner.plan_length)),
                   planning_seconds=time.perf_counter() - start)
        rows.append(row)
        with csv_path.open('a', newline='') as handle:
            csv.DictWriter(handle, fieldnames=fields).writerow(row)
        return result

    MPPIPlanner.plan = timed_plan
    # Only planning is measured; avoid an unrelated rollout evaluation.
    training.launch_unroll_eval = lambda *a, **k: {}
    original_eval = training.launch_plan_eval

    def one_episode(*a, **k):
        k['num_eval_episodes'] = 1
        return original_eval(*a, **k)

    training.launch_plan_eval = one_episode
    cfg = load_config(str(source / 'config.yaml'))
    cfg.meta.eval_only_mode = True
    cfg.meta.load_model = True
    cfg.meta.load_checkpoint = str(source / 'latest.pth.tar')
    cfg.model.compile = False
    cfg.logging.log_wandb = False
    cfg.data.num_workers = 0
    cfg.data.persistent_workers = False
    training.run(cfg=cfg, folder=str(output))
    values = np.array([row['planning_seconds'] for row in rows])
    summary = dict(calls=len(rows), mean_seconds=float(values.mean()),
                   median_seconds=float(np.median(values)),
                   p95_seconds=float(np.percentile(values, 95)),
                   min_seconds=float(values.min()), max_seconds=float(values.max()),
                   total_planning_seconds=float(values.sum()),
                   checkpoint=str(source / 'latest.pth.tar'),
                   torch_version=torch.__version__,
                   device=torch.cuda.get_device_name() if torch.cuda.is_available() else 'cpu',
                   scope='MPPIPlanner.plan including encoding and rollout; CUDA synchronized; one episode; first call included')
    full = np.array([r['planning_seconds'] for r in rows if r['horizon'] == 90])
    if len(full):
        summary['full_horizon_90'] = dict(calls=len(full), mean_seconds=float(full.mean()),
                                        median_seconds=float(np.median(full)),
                                        p95_seconds=float(np.percentile(full, 95)),
                                        min_seconds=float(full.min()), max_seconds=float(full.max()))
    (output / 'timing_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
