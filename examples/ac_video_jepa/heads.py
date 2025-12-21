import torch
from torch import nn


class MLPXYHead(nn.Module):
    """
    A head to recover the xy location from features
    """

    def __init__(self, input_shape, normalizer=None):  # input_shape = (C, H, W)
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_shape, 512), nn.ReLU(inplace=True), nn.Linear(512, 2)
        )
        self.normalizer = normalizer

    def forward(self, x):
        """
        Input:
            x: (bs, c, t, h, w)
        Output:
            pred: (bs, 2, t)
        """
        bs, c, t, h, w = x.shape

        # (bs, c, t, 1, 1) --> (bs * t, c, 1, 1)
        x = x.permute(0, 2, 1, 3, 4)  # (bs, t, c, 1, 1)
        x = x.reshape(bs * t, c, h, w)  # (bs * t, c, 1, 1)

        x = x.squeeze(-1).squeeze(-1)  # (bs * t, c, 1, 1) --> (bs * t, c)

        pred = self.mlp(x)

        pred = pred.view(bs, t, 2).permute(0, 2, 1)

        return pred


class ConvXYHead(nn.Module):
    """
    A head to recover the xy location from features
    """

    def __init__(
        self, in_chans, input_shape, normalizer=None
    ):  # input_shape = (C, H, W)
        super().__init__()
        self.in_chans = in_chans
        self.conv = nn.Sequential(
            nn.Conv2d(self.in_chans, 32, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 32),
            nn.ReLU(),
        )
        self.normalizer = normalizer
        self.flatten = nn.Flatten(start_dim=1)

        # Infer flattened dimension by running a dummy input
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)  # (1, C, H, W)
            dummy_output = self.flatten(self.conv(dummy_input))
            flattened_dim = dummy_output.shape[1]

        self.fc = nn.Linear(flattened_dim, 2)

    def forward(self, x):
        """
        Input:
            x: (bs, c, t, h, w)
        Output:
            pred: (bs, 2, t)
        """
        bs, c, t, h, w = x.shape

        x = x.permute(0, 2, 1, 3, 4)  # (bs, t, c, h, w)
        x = x.reshape(bs * t, c, h, w)  # (bs * t, c, h, w)

        # forward pass to get xy
        x = self.conv(x)
        x = self.flatten(x)
        pred = self.fc(x)

        pred = pred.view(bs, t, 2).permute(0, 2, 1)

        return pred
