# PCA LTI / GRU MPPI stage timing

Run with the repository virtual environment:

```powershell
./.venv/Scripts/python.exe -m examples.ac_video_jepa.benchmark_two_method_stages --output checkpoints/two_method_stage_20_20261014
```

The benchmark loads the saved 64-D PCA reducer and LTI weights from `checkpoints/reduced_mpc_20_20261012`. It verifies the checkpoint SHA256 and the initial position, goal, wall and door for every original task. No model is refitted. Both methods use seeds 1000–1019, 200 control steps and horizon min(90, remaining steps). They generate their own closed-loop trajectories for the same tasks.

The original PCA LTI controller is retained: ReducedRolloutMPC performs two finite-difference trajectory linearizations with SLSQP and line search. This is the controller used in the earlier 20-task comparison, rather than the analytical condensed controller in compare_linear_mpc.py.

Each timed stage synchronizes CUDA before and after its call. Nested times are subtracted, so the columns are exclusive and sum to planning_seconds. This introduces profiling overhead and differs from timing just the outer plan call.

| CSV column | Definition |
|---|---|
| encoding_seconds | Encoder forward calls, summed across the planning call; MPPI re-encodes the repeated candidate batch in each rollout iteration |
| projection_seconds | PCA reducer encode calls, excluding enclosing prediction time |
| prediction_seconds | PCA trajectory rollout including decoding and transfer to CPU, or MPPI unroll including rollout bookkeeping; excludes nested encoding/projection |
| objective_setup_seconds | PCA quadratic construction and solution validation, excluding SLSQP; or MPPI objective evaluation |
| solver_seconds | PCA SciPy minimize call, including its objective/constraint callbacks; MPPI has no separate numerical solver, so this column is zero |
| optimizer_overhead_seconds | Remaining planning time: MPPI sampling, elite selection, weighting/distribution updates and bookkeeping; PCA finite-difference construction, line-search cost/selection and bookkeeping; also input normalization and call/instrumentation overhead |
| planning_seconds | Full per-step planning wall time, including current-observation preparation, excluding goal setup, environment execution and CSV writes |

For PCA, solver_seconds is the narrow Optimize Actions solver measurement. For MPPI, optimization alternates with rollouts: optimizer_overhead_seconds captures the sampling/update work but is not a pure solver timing. Objective evaluation is reported separately.

Outputs: planning_per_timestep.csv (all 8,000 calls), summary.csv / summary.json (mean, median, p95 for all steps and H=90), results.json (configuration, task identities, episode results and completion status). The first call of each episode is included. Goal encoding and offline model loading are excluded.
