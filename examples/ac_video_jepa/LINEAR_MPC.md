# Linear and GRU LTV MPC experiments

Run from the repository root with the environment's Python:

```powershell
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
.\.venv\Scripts\python.exe -m examples.ac_video_jepa.compare_linear_mpc --checkpoint-folder checkpoints/ac_video_jepa_retrain/author_seed1_20261008_101133 --output checkpoints/linear_ltv_mpc_20261011
```

The existing encoder and GRU checkpoint are frozen. Training is not resumed.

* **Linear MPC:** PCA rank (16/32/64) and ridge strength are selected on independent validation trajectories. An affine model is fitted on training trajectories only. MPC evaluates goal distance in reconstructed original latent coordinates, including coordinate scaling.
* **GRU LTV-MPC:** central finite differences compute action-to-trajectory sensitivities along the current nonlinear GRU reference rollout. This condenses the local time-varying dynamics into a trajectory Jacobian; explicit full-state A/B matrices are not stored. Two successive linearization iterations use nonlinear rollout checks and line search.

Both controllers solve constrained quadratic subproblems with SciPy SLSQP. The original Euclidean action limit of 2.45 is retained, so these are not standard linearly constrained QPs. A small action penalty of 0.001 is added. Horizon is 90, one action is executed before replanning, and reference actions are shifted for warm starts. Horizon decreases with remaining episode steps. Neither controller includes an explicit wall avoidance constraint or a global route initializer.

By default, each controller runs one 200-step episode with identical initial task and seeds. A one-episode result is an exploratory comparison, not a reliable success-rate estimate. The earlier MPPI timing benchmark used a different task; timing and success should not be presented as a matched three-controller comparison.

Artifacts:

* `linear_model.npz`: PCA, scaling and affine dynamics weights.
* `dataset.npz`: independent train/validation/test latent trajectories.
* `results.json`: selected model, held-out prediction errors and controller summary.
* `planning_per_timestep.csv`: CUDA-synchronized wall-clock planning times (encoding, linearization and optimization), solver status and simulator goal distance.

Use a fresh output directory when changing checkpoint or data collection settings. The cached dataset belongs to the checkpoint and settings that originally produced it. The offline test covers 32 steps, not the full 90-step planning horizon.
