import torch
import torch.nn as nn
from einops import rearrange


class ImageDecoder(nn.Module):
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

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _forward(self, x):
        # x: (B,C,H,W)
        y = self.net(x)
        return y

    def forward(self, x):
        """
        Supports either:
          - 4D: (B, C, H, W)
          - 5D: (B, C, T, H, W)  -> flattens T, applies _forward, then restores T
        """
        assert x.ndim in [4, 5], "Supports only 4D (B,C,H,W) or 5D (B,C,T,H,W) tensors"
        if x.ndim == 5:
            b = x.shape[0]
            x = rearrange(x, "b c t h w -> (b t) c h w")
            y = self._forward(x)
            y = rearrange(y, "(b t) c h w -> b c t h w", b=b)
            return y
        else:
            return self._forward(x)
