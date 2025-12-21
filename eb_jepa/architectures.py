from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from sklearn.metrics import average_precision_score

######################################################
# Basic architectural modules


# a simple 3D convnet with 2 layers.
class conv3d2(nn.Sequential):
    def __init__(self, in_d, h_d, out_d, tk, ts, sk, ss, pad):
        super(conv3d2, self).__init__(
            nn.Conv3d(
                in_d, h_d, kernel_size=(tk, sk, sk), stride=(1, 1, 1), padding=pad
            ),
            nn.ReLU(),
            nn.Conv3d(
                h_d, out_d, kernel_size=(tk, sk, sk), stride=(ts, ss, ss), padding=pad
            ),
        )
        self.apply(self._init_weights)
        self.input_dim = in_d
        self.hidden_dim = h_d
        self.output_dim = out_d
        # t_shift is the index (in the time dimension) of the first output
        # cannot see its coresponding input
        if pad == "valid":
            self.t_shift = 2 * tk - 1
        elif pad == "same":
            self.t_shift = 2 * (tk - 1)
        else:
            raise NameError("invalid padding for con3d2. Must be 'valid' or 'same'")

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResNet5(nn.Module):
    """
    A lightweight ResNet with 5 layers (2 blocks)
    """

<<<<<<< HEAD
    def __init__(self, in_d, h_d, out_d, s1=1, s2=1, s3=1, avg_pool=False):
=======
    def __init__(self, in_d, h_d, out_d, s1=1):
>>>>>>> main
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_d, h_d, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(h_d)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = ResidualBlock(h_d, h_d, stride=s1)
        self.layer2 = ResidualBlock(h_d, h_d * 2, stride=s2)
        self.layer3 = ResidualBlock(h_d * 2, out_d, stride=s3)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) if avg_pool else torch.nn.Identity()

    def _make_layer(self, block, out_channels, stride):
        layer = block(self.in_channels, out_channels, stride)
        self.in_channels = out_channels
        return layer

    def _forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out).flatten(1)
        return out

    def forward(self, x):
        assert x.ndim in [4, 5], "Supports only 4D or 5D tensors"
        if x.ndim == 5:
            B = x.shape[0]
            x = rearrange(x, "b c t h w -> (b t) c h w")
            out = self._forward(x)
            out = rearrange(out, "(b t) c h w -> b c t h w", b=B)
            return out
        else:
            return self._forward(x)


class SimplePredictor(nn.Module):
    def __init__(self, predictor, context_length):
        super().__init__()
        self.predictor = predictor
        self.is_rnn = predictor.is_rnn
        self.context_length = context_length

    def forward(self, x, a):
        return self.predictor(torch.cat([x, a], dim=1))


class StateOnlyPredictor(SimplePredictor):
    """Wrapper for a simple predictor which concatenates states and actions channel wise."""

    def forward(self, x, a):
        # action not used on purpose
        prev_state = x[:, :, :-1]  # (B, C, T-1, H, W)
        next_state = x[:, :, 1:]  # (B, C, T-1, H, W)
        combined_xa = torch.cat((prev_state, next_state), dim=1)
        return self.predictor(combined_xa)


class ResUNet(nn.Module):
    """
    A small UNet with residual encoder blocks and transposed-conv upsampling.
    Channels scale like h, 2h, 4h, 8h. Output keeps the input HxW.
    """

    def __init__(self, in_d, h_d, out_d, is_rnn=False):
        super().__init__()
        self.is_rnn = is_rnn
        # Stem
        self.conv1 = nn.Conv2d(
            in_d, h_d, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(h_d)
        self.relu = nn.ReLU(inplace=True)

        # Encoder
        self.enc1 = ResidualBlock(h_d, h_d, stride=1)  # H, W
        self.enc2 = ResidualBlock(h_d, 2 * h_d, stride=2)  # H/2, W/2
        self.enc3 = ResidualBlock(2 * h_d, 4 * h_d, stride=2)  # H/4, W/4
        self.bott = ResidualBlock(4 * h_d, 8 * h_d, stride=2)  # H/8, W/8

        # Decoder upsamples, then fuses skip with a residual block that reduces channels
        self.up3 = nn.ConvTranspose2d(8 * h_d, 4 * h_d, kernel_size=2, stride=2)
        self.dec3 = ResidualBlock(8 * h_d, 4 * h_d, stride=1)

        self.up2 = nn.ConvTranspose2d(4 * h_d, 2 * h_d, kernel_size=2, stride=2)
        self.dec2 = ResidualBlock(4 * h_d, 2 * h_d, stride=1)

        self.up1 = nn.ConvTranspose2d(2 * h_d, 1 * h_d, kernel_size=2, stride=2)
        self.dec1 = ResidualBlock(2 * h_d, 1 * h_d, stride=1)

        # Head
        self.head = nn.Conv2d(h_d, out_d, kernel_size=1)

    @staticmethod
    def _match_size(x, ref):
        # Guards against odd input sizes by resizing the upsample to the skip spatial dims
        if x.shape[-2:] != ref.shape[-2:]:
            x = F.interpolate(
                x, size=ref.shape[-2:], mode="bilinear", align_corners=False
            )
        return x

    def forward(self, x):
        assert x.ndim in [4, 5], "Supports only 4D or 5D tensors"
        if x.ndim == 5:
            B = x.shape[0]
            x = rearrange(x, "b c t h w -> (b t) c h w")
            out = self._forward(x)
            out = rearrange(out, "(b t) c h w -> b c t h w", b=B)
            return out
        else:
            return self._forward(x)

    def _forward(self, x):
        x0 = self.relu(self.bn1(self.conv1(x)))

        # Encoder with skips
        s1 = self.enc1(x0)  # h
        s2 = self.enc2(s1)  # 2h
        s3 = self.enc3(s2)  # 4h
        b = self.bott(s3)  # 8h

        # Decoder stage 3
        d3 = self.up3(b)
        d3 = self._match_size(d3, s3)
        d3 = torch.cat([d3, s3], dim=1)  # 4h + 4h = 8h
        d3 = self.dec3(d3)  # → 4h

        # Decoder stage 2
        d2 = self.up2(d3)
        d2 = self._match_size(d2, s2)
        d2 = torch.cat([d2, s2], dim=1)  # 2h + 2h = 4h
        d2 = self.dec2(d2)  # → 2h

        # Decoder stage 1
        d1 = self.up1(d2)
        d1 = self._match_size(d1, s1)
        d1 = torch.cat([d1, s1], dim=1)  # h + h = 2h
        d1 = self.dec1(d1)  # → h

        out = self.head(d1)  # → out_d channels
        return out


class Projector(nn.Module):
    def __init__(self, mlp_spec):
        super().__init__()
        layers = []
        f = list(map(int, mlp_spec.split("-")))
        for i in range(len(f) - 2):
            layers.append(nn.Linear(f[i], f[i + 1]))
            layers.append(nn.BatchNorm1d(f[i + 1]))
            layers.append(nn.ReLU(True))
        layers.append(nn.Linear(f[-2], f[-1], bias=False))
        self.net = nn.Sequential(*layers)
        self.out_dim = f[-1]  # Store output dimension as attribute

    def forward(self, x):
        return self.net(x)


class ResNetProj(nn.Module):
    """
    A lightweight ResNet with 5 layers (2 blocks)
    """

    def __init__(self, in_d, h_d, out_d, s1=1, s2=1):
        super().__init__()
        self.layer1 = ResidualBlock(in_d, h_d, stride=2)
        self.layer2 = ResidualBlock(h_d, h_d * 2, stride=1)
        self.layer3 = ResidualBlock(h_d * 2, h_d * 4, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layer(self, block, out_channels, stride):
        layer = block(self.in_channels, out_channels, stride)
        self.in_channels = out_channels
        return layer

    def _forward(self, x):
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        return out

    def forward(self, x):
        assert x.ndim in [4, 5], "Supports only 4D or 5D tensors"
        if x.ndim == 5:
            B = x.shape[0]
            x = rearrange(x, "b c t h w -> (b t) c h w")
            out = self._forward(x)
            out = rearrange(out, "(b t) c h w -> b c t h w", b=B)
            return out
        else:
            return self._forward(x)


class Expander2D(nn.Module):
    """
    This class takes in input of shape (..., n) and expand it into planes (..., n, w, h)
    """

    def __init__(self, w, h):
        super(Expander2D, self).__init__()
        self.w = w
        self.h = h

    def forward(self, x):
        x = x.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, self.w, self.h)
        return x


class DetHead(nn.Module):

    def __init__(self, in_d, h_d, out_d):
        super().__init__()

        # [8, 8, T, 64, 64] --> (8, 1, T, 64, 64)
        self.apply(self._init_weights)
        self.head = nn.Sequential(conv3d2(in_d, h_d, out_d, 1, 1, 3, 1, "same"))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    # x: output of predictor (or autoregressive)
    def forward(self, x):
        # (Batch, Feature, Time, Height, Width)
        # [8, 8, T, 8, 8]
        x = [F.adaptive_avg_pool2d(x[:, :, t], (8, 8)) for t in range(x.shape[2])]
        x = torch.stack(x, 2)
        # [8, T, 8, 8]
        x = self.head(x).squeeze(1)

        return torch.sigmoid(x)

    @torch.no_grad()
    def score(self, preds, targets):

        scores = []
        for T in range(len(preds) - 1):
            x = preds[T]
            x = [F.adaptive_avg_pool2d(x[:, :, t], (8, 8)) for t in range(x.shape[2])]
            x = torch.stack(x, 2)
            x = self.head(x).squeeze(1)

            y = targets[:, T:]
            x = x[:, T:]

            ap = average_precision_score(
                y.flatten().detach().long().cpu().numpy(),
                x.flatten().detach().cpu().numpy(),
                average="weighted",
            )
            scores.append(ap)

        return scores


class JEPAModelDecoderWrapper(nn.Module):
    def __init__(self, jepa_model, decoder):
        super().__init__()
        self.jepa_model = jepa_model
        self.decoder = decoder


class ResnetBlock(nn.Module):
    """ResNet Block."""

    def __init__(self, num_features):
        super(ResnetBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        return F.relu(out + identity)


class ResnetStack(nn.Module):
    """ResNet stack module."""

    def __init__(self, input_channels, num_features, num_blocks, max_pooling=True):
        super(ResnetStack, self).__init__()
        self.num_features = num_features
        self.num_blocks = num_blocks
        self.max_pooling = max_pooling
        self.initial_conv = nn.Conv2d(
            input_channels, num_features, kernel_size=3, padding=1
        )

        self.blocks = nn.ModuleList(
            [ResnetBlock(num_features) for _ in range(num_blocks)]
        )
        if max_pooling:
            self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.max_pool = nn.Identity()

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.max_pool(x)
        for block in self.blocks:
            x = block(x)
        return x


class ImpalaEncoder(nn.Module):
    """IMPALA encoder."""

    def __init__(
        self,
        width=1,
        stack_sizes=(16, 32, 32),
        num_blocks=2,
        dropout_rate=None,
        layer_norm=False,
        input_channels=2,
        final_ln=True,
        mlp_output_dim=512,
        input_shape=(2, 65, 65),
    ):
        super(ImpalaEncoder, self).__init__()
        self.width = width
        self.stack_sizes = stack_sizes
        self.num_blocks = num_blocks
        self.dropout_rate = dropout_rate
        self.layer_norm = layer_norm
        self.input_shape = input_shape
        self.mlp_output_dim = mlp_output_dim

        input_channels = [input_channels] + list(stack_sizes)

        self.stack_blocks = nn.ModuleList(
            [
                ResnetStack(
                    input_channels=input_channels[i],
                    num_features=stack_size * width,
                    num_blocks=num_blocks,
                )
                for i, stack_size in enumerate(stack_sizes)
            ]
        )

        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate else nn.Identity()

        # Compute MLP input dimension dynamically
        with torch.no_grad():
            # Create a dummy input (assuming typical input size for this encoder)
            dummy_input = torch.zeros(1, *self.input_shape)  # (1, C, H, W)
            conv_out = dummy_input
            for stack_block in self.stack_blocks:
                conv_out = stack_block(conv_out)  # b c w h
            flattened_dim = conv_out.view(conv_out.size(0), -1).shape[1]  # c * w * h

        self.mlp = nn.Linear(flattened_dim, self.mlp_output_dim)

        if final_ln:
            self.final_ln = nn.LayerNorm(self.mlp_output_dim)
        else:
            self.final_ln = nn.Identity()

    def forward(self, x, y):
        """
        x: (bs, ch, t, w, h)
        out: (bs, dim, t, 1, 1)
        """

        # (bs, ch, t, w, h) --> (t, bs, ch, w, h)
        (
            _,
            _,
            t,
            _,
            _,
        ) = x.shape
        x = x.permute(2, 0, 1, 3, 4)

        features = []

        for i in range(t):

            conv_out = x[i]

            for i, stack_block in enumerate(self.stack_blocks):
                conv_out = stack_block(conv_out)
                if self.dropout_rate is not None:
                    conv_out = self.dropout(conv_out)

            conv_out = F.relu(conv_out)
            if self.layer_norm:
                conv_out = nn.LayerNorm(conv_out.size()[1:])(conv_out)  # b c w h
            # flatten
            out = conv_out.view(conv_out.size(0), -1)
            out = self.mlp(out)
            out = self.final_ln(out)

            features.append(out)

        features = torch.stack(features, dim=1)

        features = features.transpose(1, 2).unsqueeze(-1).unsqueeze(-1)

        return features


class RNNPredictor(nn.Module):
    def __init__(
        self,
        # parent inputs
        hidden_size: int = 512,
        action_dim: Optional[int] = 2,
        num_layers: int = 1,
        final_ln: Optional[torch.nn.Module] = None,
    ):
        super(RNNPredictor, self).__init__()

        self.num_layers = num_layers

        self.rnn = torch.nn.GRU(
            input_size=action_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
        )

        self.final_ln = final_ln
        self.is_rnn = True
        self.context_length = 0

    def forward(self, state, action):
        """
        Propagate one step forward
        Parameters:
            state: (bs, dim, 1, 1, 1)
            action: (bs, a_dim, 1)
        Output:
            Output: next_state (bs, dim, 1, 1, 1)
        """
        # This only does one step
        rnn_state = state.flatten(1, 4).unsqueeze(0).contiguous()  # (1, bs, dim)
        rnn_input = action.squeeze(-1).unsqueeze(0).contiguous()  # (1, bs, a_dim)

        next_state, _ = self.rnn(rnn_input, rnn_state)

        next_state = self.final_ln(next_state)

        return next_state[0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)


class InverseDynamicsModel(nn.Module):
    """
    Predicts the action that caused a transition from state_t to state_t_plus_1.
    Used as auxiliary task for representation learning.
    """

    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, state_t, state_t_plus_1):
        """
        Args:
            state_t: State at time t, shape [B, D]
            state_t_plus_1: State at time t+1, shape [B, D]
        Returns:
            predicted_action: Action predicted to transform state_t to state_t_plus_1, shape [B, A]
        """
        combined_states = torch.cat([state_t, state_t_plus_1], dim=1)
        return self.model(combined_states)
