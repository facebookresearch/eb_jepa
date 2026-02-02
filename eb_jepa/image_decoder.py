import torch.nn as nn

from eb_jepa.nn_utils import TemporalBatchMixin, init_module_weights


class ImageDecoder(TemporalBatchMixin, nn.Module):
    """
    Simple 2D convolutional decoder for reconstructing images from representations.
    Supports both 4D [B, C, H, W] and 5D [B, C, T, H, W] inputs via TemporalBatchMixin.
    """

    def __init__(
        self,
        in_dim,
        out_dim=1,
        hidden_dim=16,
        tk=1,  # unused in 2D; kept for API compatibility
        ts=1,  # unused in 2D; kept for API compatibility
        sk=4,  # spatial kernel for ConvTranspose2d
        ss=2,  # spatial stride (controls the upsample factor)
        pad_mode="same",
        scale_factor=1.0,
        shift_factor=0.0,
    ):
        super().__init__()
        self.scale_factor = scale_factor
        self.shift_factor = shift_factor

        self.net = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, out_dim, 3, 1, 1),
        )

        self.apply(init_module_weights)

    def _forward(self, x):
        # x: (B,C,H,W)
        y = self.net(x)
        return y
