######################################################
# Loss functions to prevent JEPAs from collapsing

import torch
import torch.nn as nn
import torch.nn.functional as F

################################################################
# Utilities


# simple square loss
def sq_loss(x, y, reduction="mean"):
    return nn.functional.mse_loss(x, y, reduction=reduction)


######################################################
# prediction loss functions


# compute square loss between two sequences
# represented as BFTHW tensors.
# a shift is necessary for the predictor cost to
# make sure the architecture is strictly causal.
def square_cost_seq(state, predi):
    return sq_loss(state, predi)


# square loss between BFTHW sequences.
# This is used primarily for the prediction loss module.
class SquareLossSeq(nn.Module):
    def __init__(self, proj=None):
        """
        Square cost over a sequence represented
        as BFTHW. Assumes the T dimension is at dim 2
        """
        super().__init__()
        self.proj = nn.Identity() if proj is None else proj

    def forward(self, state, predi):
        state = self.proj(state.transpose(0, 1).flatten(1).transpose(0, 1))
        predi = self.proj(predi.transpose(0, 1).flatten(1).transpose(0, 1))
        return square_cost_seq(state, predi)


######################################################
# Variance-Covariance loss on a batch of samples


# VC cost function
# input x is a 2D tensor Samples*Features (i.e. (B*T*H*W)*F )
# this computes two terms:
# 1. the absolute deviation of the mean of all the
# variables from zero.
# 2. the absolute deviation of the covariance matrix
# of x from the identity matrix.
#
# mcoeff is a scalar coefficient that attracts the means
# of the variables to zero.
# ccoeff is a Features*Features square matrix of
# coefficients for each term in the covariance matrix.
# c is the constant to which the variances (diagonal terms) are pinned.
def vc_cost(x, ccoeff, mcoeff=1.0, c=1.0):
    s = x.size(0)
    f = x.size(1)
    assert ccoeff.size(0) == f
    assert ccoeff.size(1) == f
    x = x - x.mean(0)
    cov_x = torch.mm(x.t(), x) / (s - 1)
    cov_x = (torch.eye(f).to(x.device) * c - cov_x).abs()
    cost = torch.mul(ccoeff.to(x.device), cov_x).mean()
    return cost


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def vc_cost_orig(x, std_coeff, cov_coeff):
    """
    x: (B*T*H*W, F')
    """
    x = x - x.mean(dim=0)
    std_x = torch.sqrt(x.var(dim=0) + 0.0001)  # (F',)
    std_loss = torch.mean(F.relu(1 - std_x))
    cov_x = (x.T @ x) / (x.shape[0] - 1)  # (F', F')
    cov_loss = off_diagonal(cov_x).pow_(2).sum().div(x.shape[-1] ** 2 - x.shape[-1])
    loss = std_coeff * std_loss + cov_coeff * cov_loss
    loss_dict = {
        "std_loss": std_loss.item(),
        "cov_loss": cov_loss.item(),
    }
    total_unweighted_loss = std_loss + cov_loss
    return loss, total_unweighted_loss, loss_dict


# Class for VC loss.
# attracts the means of the variables to zero
# and the covariance matrix towards the identity.
class VCLoss(nn.Module):
    def __init__(self, std_coeff, cov_coeff, proj=None):
        """
        Variance-Covariance loss class
        makes the means of the variables zero, and
        makes the covariance matrix as close to identity as possible
        """
        super().__init__()
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff
        self.proj = nn.Identity() if proj is None else proj

    def forward(self, x, actions=None):
        # turn input into a samples*features 2D tensor.
        # assumes feature dimension is dimension 1 (e.g. BFTHW)
        # bs, f, t, h, w = x.shape
        # fx = self.proj(x.permute(0, 2, 1, 3, 4).reshape(-1, f * h * w))
        x = x.transpose(0, 1).flatten(1).transpose(0, 1)  # (B*T*H*W, F)
        fx = self.proj(x)  # (B*T*H*W, F')
        return vc_cost_orig(fx, std_coeff=self.std_coeff, cov_coeff=self.cov_coeff)


######################################################
# Average Variance-Covariance loss on multiple sub-batches.
# input x is a BFTHW array
# bdim is the size of the sub-batches over which
# the variances/covariances are computed (default = all)
# it is best is bdim is smaller than the feature dimension.
# it is also best if bdim divides the total number of samples
def vc_cost_chunked(x, ccoeff, mcoeff=1.0, c=1.0, batch_dim=0):
    # put feature dimension first
    # turns x from BFTHW to FBTHW format
    x = x.transpose(0, 1)
    # probably should rearrange dimensions here
    # to choose consecutive samples
    flattened_state = x.flatten(1).transpose(0, 1)
    # probably should randomly shuffle samples here
    feature_size = flattened_state.size(1)  # Number of features
    sample_size = flattened_state.size(0)  # Total number of samples (B×T×H×W)
    if batch_dim == 0:
        batch_dim = feature_size
    s = sample_size // batch_dim

    # Reshape to process all chunks in parallel: (s, batch_dim, feature_size)
    chunks = flattened_state[: s * batch_dim].view(s, batch_dim, feature_size)

    # Compute means for all chunks in parallel
    mean_x = chunks.mean(1).abs().mean(1)  # shape: (s,)

    # Compute covariance matrices for all chunks in parallel
    chunks_t = chunks.transpose(1, 2)  # (s, feature_size, batch_dim)
    cov_x = torch.bmm(chunks_t, chunks) / (
        batch_dim - 1
    )  # (s, feature_size, feature_size)

    # Adjust covariance matrices towards identity
    identity = torch.eye(feature_size, device=x.device).unsqueeze(
        0
    )  # (1, feature_size, feature_size)
    cov_x = (identity * c - cov_x).abs()

    # Compute final costs for all chunks
    ccoeff_expanded = ccoeff.unsqueeze(0).to(
        x.device
    )  # (1, feature_size, feature_size)
    costs = mcoeff * mean_x + torch.mul(ccoeff_expanded, cov_x).mean(dim=(1, 2))
    # return average loss and individual loss array as a pair
    cost = costs.mean(), costs
    return cost


class ChunkedVCLoss(nn.Module):
    def __init__(self, ccoeff, mcoeff=1.0, c=1.0, bdim=0) -> None:
        """
        Chunked Variance-Covariance loss class
        """
        super().__init__()
        self.ccoeff = ccoeff
        self.mcoeff = mcoeff
        self.c = c
        self.bdim = bdim

    def forward(self, x):
        mean_cost, individual_costs = vc_cost_chunked(
            x, self.ccoeff, self.mcoeff, self.c, self.bdim
        )
        return mean_cost


######################################################
# Contrastive loss on a batch of samples


def contrastive_cost(
    x, temperature=0.1, negative_weight=1.0, subset_size=None, num_subsets=1
):
    """
    Contrastive loss function with efficient subset sampling
    input x is a BFTHW array

    This implements a contrastive objective that:
    1. Normalizes features to unit vectors
    2. Computes pairwise cosine similarities on random subsets
    3. Encourages samples to be diverse (pushes similar samples apart)
    4. Uses subset sampling for computational efficiency

    Args:
        x: Input tensor of shape (B, F, T, H, W)
        temperature: Temperature scaling parameter for similarities
        negative_weight: Weight for negative pairs (diversity encouragement)
        subset_size: Size of random subsets to sample (None means use all samples)
        num_subsets: Number of random subsets to sample and average over
    """
    # put feature dimension first
    # turns x from BFTHW to FBTHW format
    x = x.transpose(0, 1)
    # flatten to samples*features 2D tensor
    flattened_state = x.flatten(1).transpose(0, 1)  # (B*T*H*W, F)

    num_samples = flattened_state.size(0)

    if subset_size is None or subset_size >= num_samples:
        # Use all samples if subset_size is not specified or too large
        subset_size = num_samples
        num_subsets = 1

    total_loss = 0.0

    for _ in range(num_subsets):
        # Randomly sample subset_size samples
        indices = torch.randperm(num_samples, device=x.device)[:subset_size]
        subset_samples = flattened_state[indices]

        # Normalize features to unit vectors
        x_norm = F.normalize(subset_samples, p=2, dim=1)

        # Compute pairwise cosine similarities
        similarities = torch.mm(x_norm, x_norm.t()) / temperature

        # Create mask to exclude diagonal (self-similarities)
        mask = torch.eye(subset_size, device=x.device).bool()

        # Extract off-diagonal similarities (treating all as negative pairs for diversity)
        off_diagonal_sims = similarities.masked_select(~mask)

        # Contrastive loss: encourage diversity by penalizing high similarities
        # Using logsumexp for numerical stability
        diversity_loss = torch.logsumexp(off_diagonal_sims, dim=0)
        total_loss += diversity_loss

    # Average over subsets
    total_loss = total_loss / num_subsets

    return negative_weight * total_loss


class HingeStdLoss(torch.nn.Module):
    def __init__(
        self,
        std_margin: float = 1.0,
    ):
        """
        Encourages each feature to maintain at least a minimum standard deviation.
        Features with std below the margin incur a penalty of (std_margin - std).
        Args:
            std_margin (float, default=1.0):
                Minimum desired standard deviation per feature.
        """
        super().__init__()
        self.std_margin = std_margin

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Tensor[N, D] where N is number of samples, D is feature dimension
        Returns:
            std_loss: Scalar tensor with the hinge loss on standard deviations
        """
        x = x - x.mean(dim=0, keepdim=True)
        std = torch.sqrt(x.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(self.std_margin - std))
        return std_loss


class CovarianceLoss(torch.nn.Module):
    def __init__(self, adjust_conv: bool = True):
        """
        Penalizes off-diagonal elements of the covariance matrix to encourage
        feature decorrelation.
        Args:
            adjust_conv (bool, default=True):
                If True, normalizes by (D - 1) where D is feature dimensionality.
        """
        super().__init__()
        self.adjust_conv = adjust_conv

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Tensor[N, D] where N is number of samples, D is feature dimension
        """
        batch_size = x.shape[0]
        num_features = x.shape[-1]
        x = x - x.mean(dim=0, keepdim=True)
        cov = (x.T @ x) / (batch_size - 1)  # [D, D]
        # Calculate off-diagonal loss
        diag_elements = torch.diag(cov).pow(2).sum()
        cov_loss = (cov.pow(2).sum() - diag_elements) / num_features

        if self.adjust_conv:
            cov_loss = cov_loss / (num_features - 1)

        return cov_loss


class TemporalSimilarityLoss(torch.nn.Module):
    def __init__(self):
        """
        Temporal Similarity Loss.
        Encourages consecutive frames to have similar representations by penalizing
        the squared difference between consecutive time steps.
        """
        super().__init__()

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Tensor[T, N, D] where T is time steps, N is batch size, D is feature dimension
        """
        if x.shape[0] <= 1:
            return torch.tensor(0.0, device=x.device)
        sim_loss_t = (x[1:] - x[:-1]).pow(2).mean()
        return sim_loss_t


class InverseDynamicsLoss(torch.nn.Module):
    def __init__(self, idm: nn.Module):
        """
        Predicts actions from consecutive states and compares with ground truth actions.
        Args:
            idm (nn.Module): Inverse dynamics model that takes (state_t, state_t+1) and predicts action
        """
        super().__init__()
        self.idm = idm

    def forward(self, x: torch.Tensor, actions: torch.Tensor):
        """
        Args:
            x: Tensor[T, B, D] - States across time steps
            actions: Tensor[B, A, T] - Ground truth actions between consecutive states
        """
        if x.shape[0] <= 1 or actions is None:
            return torch.tensor(0.0, device=x.device)

        t, b, d = x.shape

        states_t = x[:-1].transpose(0, 1)  # [B, T-1, D]
        states_t_plus_1 = x[1:].transpose(0, 1)  # [B, T-1, D]

        states_t_flat = states_t.reshape(-1, d)  # [B*(T-1), D]
        states_t_plus_1_flat = states_t_plus_1.reshape(-1, d)  # [B*(T-1), D]

        pred_actions = self.idm(states_t_flat, states_t_plus_1_flat)  # [B*(T-1), A]
        target_actions = actions.transpose(1, 2)[:, :-1].reshape(
            -1, actions.size(1)
        )  # [B*(T-1), A]
        idm_loss = F.mse_loss(pred_actions, target_actions)

        return idm_loss


class VC_IDM_Sim_Regularizer(torch.nn.Module):
    def __init__(
        self,
        cov_coeff: float,
        std_coeff: float,
        sim_coeff_t: float,
        idm_coeff: float = 0.0,
        idm: nn.Module = None,
        std_margin: float = 1,
        adjust_conv: bool = True,
        first_t_only: bool = True,
        projector: nn.Module = None,
        spatial_as_samples: bool = False,
        sim_t_after_proj: bool = False,
        idm_after_proj: bool = False,
    ):
        """
        Composite Regularizer combining multiple losses

        This is a composite loss that combines:
        - Hinge Standard Deviation Loss
        - Covariance Decorrelation Loss
        - Temporal Similarity Loss
        - Inverse Dynamics Model Loss

        Args:
            cov_coeff (float): Weight for covariance loss
            std_coeff (float): Weight for std hinge loss
            sim_coeff_t (float): Weight for temporal similarity loss
            idm_coeff (float): Weight for inverse dynamics loss
            idm (nn.Module): Inverse dynamics model
            std_margin (float): Minimum desired std per feature
            adjust_conv (bool): Normalize covariance loss by (D-1)
            first_t_only (bool): Use only first time slice for std/cov loss
            projector (nn.Module): Optional projection layer
            spatial_as_samples (bool): Treat spatial locations as samples
            sim_t_after_proj (bool): Apply temporal loss after projection
            idm_after_proj (bool): Apply IDM loss after projection
        """
        super().__init__()
        self.cov_coeff = cov_coeff
        self.std_coeff = std_coeff
        self.sim_coeff_t = sim_coeff_t
        self.idm_coeff = idm_coeff

        self.first_t_only = first_t_only
        self.projector = nn.Identity() if projector is None else projector
        self.spatial_as_samples = spatial_as_samples
        self.sim_t_after_proj = sim_t_after_proj
        self.idm_after_proj = idm_after_proj

        # Initialize individual loss components
        self.std_loss_fn = HingeStdLoss(std_margin=std_margin)
        self.cov_loss_fn = CovarianceLoss(adjust_conv=adjust_conv)
        self.sim_loss_fn = TemporalSimilarityLoss()
        self.idm_loss_fn = InverseDynamicsLoss(idm) if idm is not None else None

    def forward(self, x, actions=None):
        """
        x (Tensor[B, C, T, H, W]):
            Input activations.  Internally reshaped to either
            (1, B, D) when `first_t_only=True` or ((T·B), D) otherwise,
            with D = C·H·W.
        first_t_only (bool, default=True):
            If True, only uses the first time‑slice for both std and cov loss;
            if False, flattens all time‑slices into the batch dimension.
        """
        b, c, t, h, w = x.shape

        # divergent gradient paths for x_unprojected and x_projected
        x_unprojected = x.permute(2, 0, 1, 3, 4).reshape(t, b, -1)  # [t, b, c*h*w]

        x_flat = x.permute(0, 2, 3, 4, 1).reshape(-1, c)  # [b*t*h*w, c]
        x_proj = self.projector(x_flat)  # [b*t*h*w, c_out]
        c_out = x_proj.shape[-1]
        x_projected = x_proj.view(b, t, h, w, c_out)  # [b, t, h, w, c_out]
        x_projected_reshaped = x_projected.permute(2, 0, 1, 3, 4).reshape(
            t, b, -1
        )  # [t, b, c_out*h*w]

        # SIM_T LOSS
        if self.sim_t_after_proj:
            sim_loss_t = self.sim_loss_fn(x_projected_reshaped)
        else:
            sim_loss_t = self.sim_loss_fn(x_unprojected)

        # IDM LOSS
        idm_loss = torch.tensor(0.0, device=x.device)
        if self.idm_coeff > 0 and self.idm_loss_fn is not None and actions is not None:
            if self.idm_after_proj:
                idm_loss = self.idm_loss_fn(x_projected_reshaped, actions)
            else:
                idm_loss = self.idm_loss_fn(x_unprojected, actions)

        # STD and COV LOSS
        if self.spatial_as_samples:
            if self.first_t_only:
                # Use only first time: [b*h*w, c_out]
                x_for_vc = x_projected[:, 0].reshape(b * h * w, c_out)
                assert x_for_vc.shape == (b * h * w, c_out)
            else:
                # Use all times: [b*t*h*w, c_out]
                x_for_vc = x_projected.reshape(-1, c_out)
                assert x_for_vc.shape == (b * t * h * w, c_out)
        else:
            x_for_vc = x_projected.permute(0, 1, 4, 2, 3).reshape(
                b, t, -1
            )  # [b, t, c_out*h*w]
            if self.first_t_only:
                # Use only first time: [b, c_out*h*w]
                x_for_vc = x_for_vc[:, 0]
                assert x_for_vc.shape == (b, c_out * h * w)
            else:
                # Use all times: [b*t, c_out*h*w]
                x_for_vc = x_for_vc.reshape(-1, x_for_vc.size(-1))
                assert x_for_vc.shape == (b * t, c_out * h * w)
        # [b*t, c_out*h*w] if first_t_only=False and spatial_as_samples=False
        # or [b, c_out*h*w] if first_t_only=True and spatial_as_samples=False
        # or [b*h*w, c_out] if first_t_only=True spatial_as_samples=True
        # or [b*t*h*w, c_out] if first_t_only=False spatial_as_samples=True
        std_loss = self.std_loss_fn(x_for_vc)
        cov_loss = self.cov_loss_fn(x_for_vc)

        total_weighted_loss = (
            self.cov_coeff * cov_loss
            + self.std_coeff * std_loss
            + self.sim_coeff_t * sim_loss_t
            + self.idm_coeff * idm_loss
        )
        total_unweighted_loss = cov_loss + std_loss + sim_loss_t + idm_loss

        loss_dict = {
            "cov_loss": cov_loss.item(),
            "std_loss": std_loss.item(),
            "sim_loss_t": sim_loss_t.item(),
            "idm_loss": idm_loss if isinstance(idm_loss, float) else idm_loss.item(),
        }

        return total_weighted_loss, total_unweighted_loss, loss_dict


class ContrastiveLoss(nn.Module):
    def __init__(
        self,
        temperature=0.1,
        negative_weight=1.0,
        proj=None,
        subset_size=None,
        num_subsets=1,
    ):
        """
        Contrastive loss class with efficient subset sampling
        Encourages diversity among samples by penalizing high similarities

        Args:
            temperature: Temperature scaling parameter for similarities
            negative_weight: Weight for the contrastive term
            proj: Optional projection layer applied before computing loss
            subset_size: Size of random subsets to sample (None means use all samples)
            num_subsets: Number of random subsets to sample and average over
        """
        super().__init__()
        self.temperature = temperature
        self.negative_weight = negative_weight
        self.subset_size = subset_size
        self.num_subsets = num_subsets
        self.proj = nn.Identity() if proj is None else proj

    def forward(self, x):
        # apply optional projection before computing contrastive cost
        if self.proj is not None and not isinstance(self.proj, nn.Identity):
            # apply projection to features: assumes feature dimension is dimension 1 (BFTHW)
            fx = self.proj(x.transpose(0, 1).flatten(1).transpose(0, 1))
            # reshape back to BFTHW format
            b, f, t, h, w = x.shape
            projected_f = fx.size(1)  # new feature dimension after projection
            fx = fx.transpose(0, 1).view(projected_f, b, t, h, w).transpose(0, 1)
            return contrastive_cost(
                fx,
                self.temperature,
                self.negative_weight,
                self.subset_size,
                self.num_subsets,
            )
        else:
            return contrastive_cost(
                x,
                self.temperature,
                self.negative_weight,
                self.subset_size,
                self.num_subsets,
            )
