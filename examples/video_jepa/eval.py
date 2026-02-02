import collections

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from einops import rearrange, repeat
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


def add_label_to_video(video, label):
    """Add a text label overlay on each frame of a video.

    Args:
        video: numpy array of shape (T, H, W, C) in uint8
        label: text string to add

    Returns:
        numpy array of shape (T, H, W, C)
    """
    font = ImageFont.load_default()
    T, H, W, C = video.shape

    labeled_frames = []
    for t in range(T):
        frame = Image.fromarray(video[t])
        draw = ImageDraw.Draw(frame, "RGBA")
        draw.rectangle([0, 0, W, 20], fill=(40, 40, 40, 200))
        draw.text((4, 4), label, fill=(255, 255, 255), font=font)
        labeled_frames.append(np.array(frame))
    return np.stack(labeled_frames, axis=0)


def visualize_videos(
    batch,
    jepa,
    pixel_decoder,
    detection_head,
    num_samples,
):
    """Create visualization videos for wandb logging.

    Returns a list of videos, each with 3 vertically stacked rows:
    1. Ground truth video
    2. Predicted rollout reconstruction
    3. Digit detection overlay
    """

    x = batch["video"]
    x_jepa = jepa.encoder(x)

    T = x.shape[2]
    preds, _ = jepa.unroll(
        x,
        actions=None,
        nsteps=T - 2,
        unroll_mode="parallel",
        compute_loss=False,
        return_all_steps=True,
    )

    # One step predictions
    one_step_pred = x_jepa[:, :, 1:].clone()
    one_step_pred[:, :, 1:] = preds[0]
    one_step_reconstruction = pixel_decoder.head(one_step_pred)

    # Multi-step rollouts
    rollout = x_jepa[:, :, 1:].clone()
    for t in range(1, T - 1):
        rollout[:, :, t:] = preds[t - 1][:, :, t - 1 :]
    rollout_reconstruction = pixel_decoder.head(rollout)

    # Location predictions overlaid over rollout as blue heatmap
    loc_prediction = detection_head.head(rollout)
    loc_prediction = F.interpolate(
        loc_prediction, (x.shape[-2], x.shape[-1]), mode="nearest"
    )
    loc_prediction = repeat(loc_prediction, "b t h w -> b c t h w", c=3).clone()
    loc_prediction[:, :2].fill_(0)

    # Overlay rollout reconstruction and location predictions
    detection_overlay = 0.2 * rollout_reconstruction + 0.8 * loc_prediction

    # Ground truth (skip first frame to align with predictions)
    gt = x[:, :, 1:]

    # Helper function to scale and convert pixel decoder outputs
    # to uint8 RGB and return as numpy array for video logging
    def scale_and_convert_to_uint8(tensor):
        tensor = F.interpolate(tensor, (100, 100), mode="bilinear")
        if tensor.shape[0] == 1:
            tensor = tensor.repeat(3, 1, 1, 1)
        tensor = torch.clamp(tensor * 255, 0, 255).to(torch.uint8)
        tensor = rearrange(tensor, "c t h w -> t h w c").cpu().numpy()
        return tensor

    rows = [gt, rollout_reconstruction, detection_overlay]
    labels = ["Ground truth", "Predicted rollout", "Digit detections"]

    viz_videos = []
    for b in range(num_samples):
        videos = [row[b] for row in rows]
        videos = [scale_and_convert_to_uint8(video) for video in videos]
        videos = [
            add_label_to_video(video, label) for video, label in zip(videos, labels)
        ]
        videos = [video.transpose(0, 3, 1, 2) for video in videos]
        viz_videos.append(np.concatenate(videos, axis=2))  # (T, C, 3*H, W)

    return viz_videos


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
        preds, _ = jepa.unroll(
            x,
            actions=None,
            nsteps=T - 2,
            unroll_mode="parallel",
            compute_loss=False,
            return_all_steps=True,
        )
        scores = detection_head.head.score(preds, loc_map[:, 2:])
        for s, score in enumerate(scores):
            metrics[f"AP_{s}"].append(float(score))

    # Aggregate val results and visualize last batch
    metrics = {k: float(np.mean(v)) for k, v in metrics.items()}
    videos = visualize_videos(
        batch, jepa, pixel_decoder, detection_head, num_samples=16
    )
    logs = {
        **metrics,
        "viz": [wandb.Video(video, fps=4, format="mp4") for video in videos],
    }
    print(metrics)

    # Set modules back to train mode
    jepa.train()
    detection_head.train()
    pixel_decoder.train()

    return logs
