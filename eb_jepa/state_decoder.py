from torch import nn


class MLPXYHead(nn.Module):
    """A head to recover the xy location from features."""

    def __init__(self, input_shape, normalizer=None):  # input_shape = (C, H, W)
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_shape, 512), nn.ReLU(inplace=True), nn.Linear(512, 2)
        )
        self.normalizer = normalizer

    def forward(self, x):
        """
        Args:
            x: [B, C, T, H, W]
        Returns:
            pred: [B, 2, T]
        """
        bs, c, t, h, w = x.shape

        x = x.permute(0, 2, 1, 3, 4)  # [B, T, C, H, W]
        x = x.reshape(bs * t, c, h, w)  # [B*T, C, H, W]

        x = x.squeeze(-1).squeeze(-1)  # [B*T, C]

        pred = self.mlp(x)

        pred = pred.view(bs, t, 2).permute(0, 2, 1)

        return pred
