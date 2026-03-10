# Example - Action Conditioned Video JEPA (Parallel / Spatial)

This example demonstrates a Joint Embedding Predictive Architecture (JEPA) for action-conditioned world modeling in the Two Rooms environment, using a **non-recurrent, spatially-structured** architecture. It extends `examples/video_jepa` with actions, and serves as an architectural alternative to `examples/ac_video_jepa`.

## Overview

This example shares the same learning objective as [`examples/ac_video_jepa`](../ac_video_jepa/README.md) — learning to predict future visual embeddings conditioned on actions — but makes two fundamental architectural choices that distinguish it:

1. **Non-recurrent predictor** (ResUNet + `SimplePredictor`): instead of a recurrent RNN that processes one step at a time, predictions are produced over a context window in parallel, following the same design as `examples/video_jepa` but with actions as additional input.

2. **Spatial encoder** (ResNet5): instead of the ImpalaEncoder which collapses spatial dimensions into a single global vector `[B, D, T, 1, 1]`, ResNet5 preserves the full spatial structure of feature maps `[B, C, T, H, W]`. This choice has cascading architectural consequences described below.

| Component | `ac_video_jepa` | `ac_video_parallel_jepa` (this example) |
|-----------|-----------------|----------------------------------------|
| Encoder | ImpalaEncoder → `[B, D, T, 1, 1]` | ResNet5 → `[B, C, T, H, W]` |
| Predictor | RNNPredictor (recurrent, sequential) | ResUNet + SimplePredictor (non-recurrent, parallel) |
| Action Encoder | Identity (action fed directly to RNN) | ActionEncoder (broadcasts action → spatial tensor) |
| Probe Head | MLPXYHead (squeezes 1×1 spatial dims) | SpatialMLPXYHead (flattens C×H×W) |
| Unroll mode | Sequential / autoregressive | Parallel (sliding context window buffer) |

## Training

### Architecture

1. **Encoder** (ResNet5): Maps each observation frame to a spatial feature map.
   - Input: `[B, C_obs, T, H, W]`
   - Output: `[B, dstc, T, H, W]` — spatial dims are preserved throughout

2. **Predictor** (ResUNet + SimplePredictor): Predicts the next representation from a context window of past encoded states and actions. The predictor is non-recurrent: it takes a stacked context buffer of `ctxtwind` frames and outputs the next predicted representation in a single forward pass.
   - Input channels: `ctxtwind × 2 × dstc` (stacked encoded states concatenated channel-wise with encoded actions)
   - No hidden state is maintained across steps; the context window is a sliding buffer

3. **Action Encoder** (ActionEncoder): Encodes a 2D action vector into a spatial tensor that can be concatenated channel-wise with the ResNet5 feature maps.
   - This component is required because the ResUNet predictor operates on spatial tensors `[B, C, T, H, W]`. A raw 2D action cannot be concatenated directly — it must first be projected and reshaped to match the spatial dimensions of the encoded states.
   - Architecture: two-layer MLP with hidden dim `hactenc`, output reshaped to `[B, dstc, T, H, W]`

4. **Regularizer**: Same VC + IDM + time-similarity regularization as `ac_video_jepa`. See that README for details.

5. **Probe Head** (SpatialMLPXYHead): Decodes the agent's (x, y) position from encoder representations.
   - Because ResNet5 preserves spatial dims, the standard `MLPXYHead` (which assumes 1×1 spatial dims and uses `squeeze`) cannot be used here. `SpatialMLPXYHead` instead flattens `C × H × W` into a single vector before the linear probe.
   - Input shape: `C × H × W` (e.g. `16 × 65 × 65`)

### Training Objectives

Same loss terms as `ac_video_jepa`:

$$L = L_{\text{pred}} + \beta L_{\text{cov}} + \alpha L_{\text{var}} + \delta L_{\text{time-sim}} + \omega L_{\text{IDM}}$$

See [`examples/ac_video_jepa`](../ac_video_jepa/README.md#training-objectives) for the full formulation.

### Training Data

Same Two Rooms environment as `ac_video_jepa`. See that README for details on the fixed wall and random wall setups.

### Usage

```bash
# Train a model locally
python -m examples.ac_video_parallel_jepa.main \
  --fname examples/ac_video_parallel_jepa/cfgs/train.yaml

# Launch with sbatch
python -m examples.launch_sbatch --example ac_video_parallel_jepa
```

Key configuration parameters (`cfgs/train.yaml`):

| Parameter | Description |
|-----------|-------------|
| `model.dstc` | Representation dimension (encoder output channels) |
| `model.henc` | ResNet5 hidden dim |
| `model.hpre` | ResUNet hidden dim |
| `model.ctxtwind` | Context window size for SimplePredictor |
| `model.hactenc` | Hidden dim of the ActionEncoder MLP |
| `model.nsteps` | Number of prediction steps during training |

## Evaluation

Unrolling and planning evaluation follow the same procedure as `ac_video_jepa`. See that README for full details on unrolling evaluation, planning with MPPI/CEM, and evaluation episode definition.

## References
- [JEPA Paper](https://openreview.net/pdf?id=BZ5a1r-kVsf)
- [PLDM and Two-Rooms Environment](https://arxiv.org/abs/2502.14819)
- [ResNet Architecture](https://arxiv.org/abs/1512.03385)
- [VICReg](https://arxiv.org/abs/2105.04906)
- [JEPA Slow Features](https://arxiv.org/abs/2211.10831)
- [MPPI](https://arxiv.org/abs/1509.01149)
- [CEM](https://asco.lcsr.jhu.edu/papers/Ko2012.pdf)
