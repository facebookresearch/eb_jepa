# Matched dimensionality-reduction MPC screening

Run with the frozen retrained checkpoint and the earlier independent simulator cache:

```powershell
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
.\.venv\Scripts\python.exe -u -m examples.ac_video_jepa.compare_reduced_mpc --source checkpoints/linear_ltv_mpc_20261011 --output checkpoints/reduced_mpc_new_run --rank 64 --episodes 3 --horizon 90 --steps 200
```

Use a fresh output directory. Model weights and the dataset are recorded by SHA-256 in `results.json`. The source dataset has 1200 training, 200 validation, and 200 test trajectories, each covering 32 transitions. It was generated with the retrained checkpoint recorded in the source results. All learned transforms and LTI regressions use training data only; validation selects autoencoder checkpoint and ridge strength. Test data is used only for final prediction/reconstruction reports.

Eight methods are evaluated:

| Representation | LTI | LTV |
| --- | --- | --- |
| Original 512-dimensional latent | Full affine regression | Original GRU local trajectory model |
| PCA, 64 dimensions | Reduced affine regression | Decode -> GRU -> PCA at every step |
| Task-aware linear projection, 64 dimensions | Reduced affine regression | Decode -> GRU -> project at every step |
| Nonlinear autoencoder, 64 dimensions | Reduced affine regression | Decode -> GRU -> encode at every step |

Task-aware projection uses two ridge-regression directions predicting the frozen checkpoint's position-probe estimates, completed with PCA of the orthogonal residual. It does not use simulator position labels, does not learn wall avoidance, and is a particular task-aware linear projection baseline, not a general claim about supervised reduction.

The nonlinear autoencoder has 512 -> 128 -> 64 encoding layers and symmetric decoding layers, with Tanh activations. It learns original-latent reconstruction MSE for 60 epochs and uses the best validation checkpoint. Good reconstruction does not guarantee accurate reduced dynamics or accurate derivatives.

LTV has dynamics `x_next = E(GRU(D(x), u))`, with E/D the selected reduction/reconstruction maps. Original LTV uses identity E/D. Central finite differences of complete trajectories compute condensed input sensitivities. There are two successive linearization iterations and a nonlinear line search. Explicit per-step state Jacobians are not stored. Reduced GRU rollout still runs the original 512-dimensional GRU; reduced LTV is not a separately learned small recurrent network.

For consistent objectives, all methods use reconstructed original-latent squared goal distance, summed over the horizon, plus 0.001 times squared action magnitude. Every method uses the same finite-difference successive-linearization optimization implementation and SLSQP with Euclidean action bound 2.45. For AE+LTI, the state dynamics are affine but the decoded objective is nonlinear. This prototype is not a benchmark of optimized dedicated LTI QP solvers.

All variants share the same three start/goal/wall/door tasks (seeds 1000, 1001, 1002), verified by an assertion and saved in `results.json`. Each episode runs all 200 steps, even after reaching the goal, matching the previous evaluation procedure. Final success uses distance <4.5 pixels at step 200. Ever-success is reported separately. Horizon is min(90, remaining steps); full-horizon timing uses the first 111 planning calls per episode. Timing includes current-image encoding, representation transforms, rollout/linearization and optimization, with CUDA synchronization. Per-step CSV writing and simulation are excluded.

Outputs: `summary.csv`, `planning_per_timestep.csv`, `results.json`, `comparison.png`, reducer state dictionaries and affine model coefficients. Three episodes are exploratory screening, not a statistically reliable success-rate estimate. No explicit wall constraint or global route initializer is provided. Neither the old unmatched MPPI task nor README success rates are a matched control group here.
