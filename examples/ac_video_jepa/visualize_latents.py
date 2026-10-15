"""Replay the saved evaluation and plot true/predicted latents in shared PCA axes."""
from pathlib import Path
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageSequence
from omegaconf import OmegaConf
from eb_jepa.planning import main_unroll_eval
from examples.ac_video_jepa import main as training


ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "checkpoints/ac_video_jepa_gpu"
SOURCE = MODEL / "evaluation_final/unroll_eval/step-18743_eval_only/b0.gif"
OUT = MODEL / "latent_pca"


def gt_pixels(path):
    with Image.open(path) as im:
        # Original visualization: 60px title, four 130px rows, four columns.
        return np.stack([np.asarray(f.convert("RGB"))[60:580, :130].copy()
                         for f in ImageSequence.Iterator(im)])


def visualize(model, env_creator, folder, epoch, step, suffix, loader, prober, cfg):
    captured = []
    class Capture:
        dataset = loader.dataset
        def __iter__(self):
            batch = next(iter(loader))
            captured.append(batch)
            yield batch
    OUT.mkdir(parents=True, exist_ok=True)
    result = main_unroll_eval(model, env_creator, OUT, num_samples=1,
                              loader=Capture(), prober=prober, cfg=cfg)
    x, a, *_ = captured[0]
    device = next(model.parameters()).device
    x, a = x.to(device), a.to(device)
    model.eval()
    b, c, t, h, w = x.shape
    with torch.no_grad():
        true = model.encode(x.permute(0, 2, 1, 3, 4).flatten(0, 1).unsqueeze(2))
        true = true.flatten(1).reshape(b, t, -1).float().cpu().numpy()
        pred, _ = model.unroll(x[:, :, :1], a, nsteps=a.shape[2],
            unroll_mode="autoregressive", ctxt_window_time=1,
            compute_loss=False, return_all_steps=False)
        pred = pred[:, :, :-1].permute(0, 2, 1, 3, 4).flatten(2).float().cpu().numpy()
    assert true.shape == pred.shape
    # Fit on true embeddings of all four trajectories, then project both sets.
    flat = true.reshape(-1, true.shape[-1])
    mean = flat.mean(0)
    _, s, vt = np.linalg.svd(flat - mean, full_matrices=False)
    axes = vt[:2].T
    explained = s[:2] ** 2 / (s ** 2).sum()
    z = (true - mean) @ axes
    p = (pred - mean) @ axes
    original, replay = gt_pixels(SOURCE), gt_pixels(OUT / "b0.gif")
    match = original.shape == replay.shape and np.array_equal(original, replay)
    diff = float(np.abs(original.astype(float) - replay).mean()) if original.shape == replay.shape else None
    info = dict(source=str(SOURCE), checkpoint=str(MODEL / "latest.pth.tar"),
        latent_shape=list(true.shape), gt_gif_exact_match=bool(match),
        gt_gif_mean_absolute_pixel_difference=diff,
        pca_fit="True embeddings of all four replayed trajectories; centered, no feature scaling",
        explained_variance_ratio=explained.tolist(),
        latent_mse_per_trajectory=((true-pred)**2).mean((1,2)).tolist())
    np.savez_compressed(OUT / "latents.npz", true=true, predicted=pred, true_pca=z,
        predicted_pca=p, pca_mean=mean, pca_axes=axes, images=x.cpu().numpy(), actions=a.cpu().numpy())
    (OUT / "metadata.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    from examples.ac_video_jepa.plot_saved_latents import render
    render()
    print(json.dumps(info, indent=2))
    return result


if __name__ == '__main__':
    import sys
    if '--replay' not in sys.argv and (OUT / 'latents.npz').exists():
        from examples.ac_video_jepa.plot_saved_latents import render
        render()
        raise SystemExit(0)
    cfg = OmegaConf.load(MODEL / 'evaluation_final/config.yaml')
    cfg.meta.load_checkpoint = str(MODEL / 'latest.pth.tar')
    training.launch_unroll_eval = visualize
    training.launch_plan_eval = lambda *args, **kwargs: {'success_rate': float('nan')}
    training.run(cfg=cfg, folder=OUT / 'replay')
