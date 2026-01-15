import collections

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from einops import rearrange
from torchvision.utils import make_grid
from tqdm import tqdm


def visualize(
    batch,
    jepa,
    pixel_decoder,
    detection_head,
):

    x = batch["video"]
    x_jepa = jepa.encoder(x)

    T = x.shape[2]
    preds = jepa.infern(x, actions=None, nsteps=T - 2)

    # Helper function to scale and convert pixel decoder outputs
    def scale_and_convert_to_uint8(tensor):
        # Scale from [0,1] to [0,255] and clamp values
        scaled = torch.clamp(tensor * 255, 0, 255)
        # Convert to uint8
        return scaled.to(torch.uint8)

    # One step predictions
    one_step_pred = x_jepa[:, :, 1:].clone()
    one_step_pred[:, :, 1:] = preds[0]
    one_step_reconstruction = pixel_decoder.head(one_step_pred)
    one_step_reconstruction = scale_and_convert_to_uint8(one_step_reconstruction)

    # Multi-step rollouts
    rollout = x_jepa[:, :, 1:].clone()
    for t in range(1, T - 1):
        rollout[:, :, t:] = preds[t - 1][:, :, t - 1 :]
    rollout_reconstruction = pixel_decoder.head(rollout)
    rollout_reconstruction = scale_and_convert_to_uint8(rollout_reconstruction)

    # Location predictions overlaid over rollout as blue heatmap
    loc_prediction = detection_head.head(rollout)
    loc_prediction = F.interpolate(
        loc_prediction, (x.shape[-2], x.shape[-1]), mode="nearest"
    )
    loc_prediction = rearrange(loc_prediction, "b t h w -> b 1 t h w")
    loc_prediction = loc_prediction.repeat(1, 3, 1, 1, 1)
    loc_prediction[:, :2].fill_(0)

    # Convert rollout_reconstruction back to float for overlay calculation
    rollout_reconstruction_float = rollout_reconstruction.float() / 255.0
    overlay = 0.2 * rollout_reconstruction_float + 0.8 * loc_prediction
    overlay = scale_and_convert_to_uint8(overlay)

    # Stack panels horizontally
    # Convert x to uint8 for consistency
    x_uint8 = scale_and_convert_to_uint8(x[:, :, 1:])

    merged = torch.cat(
        [
            x_uint8.repeat(1, 3, 1, 1, 1),
            one_step_reconstruction.repeat(1, 3, 1, 1, 1),
            rollout_reconstruction.repeat(1, 3, 1, 1, 1),
            overlay,
            torch.zeros_like(overlay),
        ],
        dim=3,
    )  # (B, C, T, 3*H, W)

    # Stack frames vertically
    merged = rearrange(merged, "b c t h w -> b c h (t w)")
    grid = make_grid([img for img in merged], nrow=1)
    return grid


# Run full loop over validation set and compute metrics
@torch.inference_mode()
def validation_loop(val_loader, jepa, detection_head, pixel_decoder, steps, device):

    # Set modules to eval mode
    jepa.eval()
    detection_head.eval()
    pixel_decoder.eval()

    metrics = collections.defaultdict(list)
    for batch in tqdm(val_loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        x = batch["video"]
        loc_map = batch["digit_location"]

        recon_loss = pixel_decoder(x, x)
        det_loss = detection_head(x, loc_map)

        logs = {
            "val/recon_loss": float(recon_loss.item()),
            "val/det_loss": float(det_loss.item()),
        }
        for k, v in logs.items():
            metrics[k].append(v)

        T = x.shape[2]
        preds = jepa.infern(x, actions=None, nsteps=T - 2)
        scores = detection_head.head.score(preds, loc_map[:, 2:])
        for s, score in enumerate(scores):
            metrics[f"AP_{s}"].append(float(score))

    # Aggregate val results and visualize last batch
    metrics = {k: float(np.mean(v)) for k, v in metrics.items()}
    viz = visualize(batch, jepa, pixel_decoder, detection_head)
    logs = {
        "viz": wandb.Image(viz, caption="decoder_viz"),
        **metrics,
    }
    print(metrics)

    # Set modules back to train mode
    jepa.train()
    detection_head.train()
    pixel_decoder.train()

    return logs
