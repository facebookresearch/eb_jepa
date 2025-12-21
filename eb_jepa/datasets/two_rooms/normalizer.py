import torch


class Normalizer:
    """
    Used to preprocess the observation / state image tensor and location tensor to
    standard normal distribution
    """

    def __init__(self):
        self.location_mean = torch.tensor([31.5863, 32.0618])
        self.location_std = torch.tensor([16.1025, 16.1353])
        self.state_mean = torch.tensor([0.0026, 0.0989])
        self.state_std = torch.tensor([0.0369, 0.2986])

    def min_max_normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        """
        Min-max normalize the state tensor.
        Args:
            state (torch.Tensor): State tensor with shape [..., ch, w, h] or [..., n].
        Returns:
            torch.Tensor: Min-max normalized state tensor with the same shape as input.
        """
        if len(state.shape) >= 3:
            state = state - state.amin(dim=(-2, -1), keepdim=True)
            state = state / (state.amax(dim=(-2, -1), keepdim=True) + 1e-6)
        else:
            state = state - state.amin(dim=-1, keepdim=True)
            state = state / (state.amax(dim=-1, keepdim=True) + 1e-6)
        return state

    def normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        """
        First min-max normalize, then normalize to mean 0 and std 1.
        Args:
            state (torch.Tensor): State tensor with shape [..., ch, w, h].
        Returns:
            torch.Tensor: Normalized state tensor with the same shape as input.
        """
        state = self.min_max_normalize_state(state)

        adapted_mean = self.state_mean.view(-1, 1, 1).to(state.device)
        adapted_std = self.state_std.view(-1, 1, 1).to(state.device) + 1e-6

        state_channels = state.shape[-3]  # [..., ch, w, h]

        # in case the stats are calculated over stacked obs, but state is unstacked:
        if state_channels < adapted_mean.shape[0] and not (
            adapted_mean.shape[0] % state_channels
        ):
            adapted_mean = adapted_mean[:state_channels]
            adapted_std = adapted_std[:state_channels]

        normalized_state = (state - adapted_mean) / adapted_std

        return normalized_state

    def unnormalize_state(self, state: torch.Tensor) -> torch.Tensor:
        """
        Unnormalize the state tensor.
        Args:
            state (torch.Tensor): Normalized state tensor with shape [..., ch, w, h].
        Returns:
            torch.Tensor: Unnormalized state tensor with the same shape as input.
        """
        adapted_mean = self.state_mean.view(-1, 1, 1).to(state.device)
        adapted_std = self.state_std.view(-1, 1, 1).to(state.device)

        state_channels = state.shape[-3]  # [..., ch, w, h]

        # in case the stats are calculated over stacked obs, but state is unstacked:
        if state_channels < adapted_mean.shape[0] and not (
            adapted_mean.shape[0] % state_channels
        ):
            adapted_mean = adapted_mean[:state_channels]
            adapted_std = adapted_std[:state_channels]

        return state * adapted_std + adapted_mean

    def normalize_location(self, location: torch.Tensor) -> torch.Tensor:
        """
        Normalize the location tensor.
        Args:
            location (torch.Tensor): Location tensor with shape [..., 2].
        Returns:
            torch.Tensor: Normalized location tensor with the same shape as input.
        """
        return (location - self.location_mean.to(location.device)) / (
            self.location_std.to(location.device) + 1e-6
        )

    def unnormalize_location(self, location: torch.Tensor) -> torch.Tensor:
        """
        Unnormalize the location tensor.
        Args:
            location (torch.Tensor): Normalized location tensor with shape [..., 2].
        Returns:
            torch.Tensor: Unnormalized location tensor with the same shape as input.
        """
        return location * self.location_std.to(location.device) + self.location_mean.to(
            location.device
        )

    def unnormalize_mse(self, mse):
        """
        Unnormalize the mean squared error.
        Args:
            mse (torch.Tensor): Mean squared error scalar or tensor.
        Returns:
            torch.Tensor: Unnormalized mean squared error with the same shape as input.
        """
        return mse * (self.location_std.mean().to(mse.device) ** 2)
