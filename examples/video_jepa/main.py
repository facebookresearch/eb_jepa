import fire
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from eb_jepa.architectures import (
    DetHead,
    Projector,
    ResNet5,
    ResUNet,
    StateOnlyPredictor,
)
from eb_jepa.datasets.moving_mnist import MovingMNISTDet
from eb_jepa.image_decoder import ImageDecoder
from eb_jepa.jepa import JEPA, JEPAProbe
from eb_jepa.losses import SquareLossSeq, VCLoss
from examples.video_jepa.eval import validation_loop


def run(
    batch_size: int = 64,
    dobs: int = 1,
    henc: int = 32,
    hpre: int = 32,
    dstc: int = 16,
    steps: int = 4,
    cov_coeff: float = 100.0,
    std_coeff: float = 10.0,
    epochs: int = 100,
    lr: float = 1e-3,
):
    """Train a Video JEPA model with VC loss. Evaluate the encoder and predictor on Moving MNIST detection, and visualize the representations."""
    device = "cuda"
    torch.manual_seed(2025)

    # Load datasets
    train_set = MovingMNISTDet(split="train")
    val_set = MovingMNISTDet(split="val")
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    # Initialize Video JEPA model, conditioned on past observations (not actions)
    encoder = ResNet5(dobs, henc, dstc)
    predictor_model = ResUNet(2 * dstc, hpre, dstc)
    predictor = StateOnlyPredictor(predictor_model, context_length=2)
    projector = Projector(f"{dstc}-{dstc*4}-{dstc*4}")
    regularizer = VCLoss(std_coeff, cov_coeff, proj=projector)
    ploss = SquareLossSeq(projector)
    jepa = JEPA(encoder, encoder, predictor, regularizer, ploss).to(device)

    # Initialize decoder and detection head, only used for evaluation
    decoder = ImageDecoder(dstc, dobs)
    dethead = DetHead(dstc, hpre, dobs)
    pixel_decoder = JEPAProbe(jepa, decoder, nn.MSELoss()).to(device)
    detection_head = JEPAProbe(jepa, dethead, nn.BCELoss()).to(device)

    jepa.train()
    detection_head.train()
    pixel_decoder.train()

    optimizer = Adam(
        [
            {"params": jepa.parameters(), "lr": lr},
            # only for visualization purposes, gradients are not propogated back to the encoder
            {"params": pixel_decoder.head.parameters(), "lr": lr},
            {"params": detection_head.head.parameters(), "lr": lr},
        ]
    )

    wandb.init(
        project="jepa-unlabeled-video",
        config={
            "batch_size": batch_size,
            "dobs": dobs,
            "henc": henc,
            "hpre": hpre,
            "dstc": dstc,
            "steps": steps,
            "cov_coeff": cov_coeff,
            "std_coeff": std_coeff,
            "epochs": epochs,
            "lr": lr,
        },
    )

    for epoch in range(epochs):
        pbar = tqdm(train_loader)
        for _, batch in enumerate(pbar):

            batch = {k: v.to(device) for k, v in batch.items()}

            x = batch["video"]
            loc_map = batch["digit_location"]

            optimizer.zero_grad()
            jepa_loss, regl, regldict, pl = jepa.forwardn(x, actions=None, nsteps=steps)
            recon_loss = pixel_decoder(x, x)
            det_loss = detection_head(x, loc_map)
            total_loss = jepa_loss + recon_loss + det_loss

            total_loss.backward()
            optimizer.step()

            logs = {
                "Epoch": epoch,
                "Loss": float(jepa_loss.item()),
                "VC Loss": float(regl.item()),
                "Pred Loss": float(pl.item()),
                "Recon Loss": float(recon_loss.item()),
                "Det Loss": float(det_loss.item()),
            }
            for k, v in regldict.items():
                logs[k] = float(v)

            pbar.set_postfix(logs)

        # Log train results every epoch
        step = len(train_loader) * epoch
        val_logs = validation_loop(
            val_loader, jepa, detection_head, pixel_decoder, steps, device
        )
        wandb.log({**logs, **val_logs}, step=step)
    wandb.finish()


if __name__ == "__main__":
    fire.Fire(run)
