import torch
import torch.nn as nn

from eb_jepa.logging import get_logger

logging = get_logger(__name__)


class JEPAbase(nn.Module):
    """Base JEPA class for planning and inference only. Use JEPA subclass for training."""

    def __init__(self, encoder, aencoder, predictor):
        """Initialize JEPAbase with encoder, action encoder, and predictor."""
        super().__init__()
        # Observation Encoder
        self.encoder = encoder
        # Action Encoder
        self.action_encoder = aencoder
        # Predictor
        self.predictor = predictor
        self.single_unroll = getattr(self.predictor, "is_rnn", False)

    def save(self, file):
        torch.save(self.state_dict(), file)

    def load(self, file):
        self.load_state_dict(torch.load(file), weights_only=False)

    @torch.no_grad()
    def encode(self, observations):
        """Encode a sequence of observations and return the encoder output."""
        return self.encoder(observations)


class JEPA(JEPAbase):
    """Trainable JEPA with prediction loss and anti-collapse regularizer."""

    def __init__(self, encoder, aencoder, predictor, regularizer, predcost):
        """Initialize JEPA with regularizer and prediction cost in addition to base components."""
        super().__init__(encoder, aencoder, predictor)
        self.regularizer = regularizer
        self.predcost = predcost
        self.ploss = 0
        self.rloss = 0

    @torch.no_grad()
    def infer(self, observations, actions):
        """Produce single-step predictions over all sequence elements in parallel."""
        preds, _ = self.unroll(
            observations,
            actions,
            nsteps=1,
            unroll_mode="parallel",
            compute_loss=False,
            return_all_steps=True,
        )
        return preds[0]

    def unroll(
        self,
        observations,
        actions,
        nsteps=1,
        unroll_mode="parallel",
        ctxt_window_time=1,
        compute_loss=True,
        return_all_steps=False,
    ):
        """Unified multi-step prediction with optional loss computation.

        This function supports both training (with loss computation) and planning/inference
        (without loss, just state prediction).

        Usage examples:
        - Training video_jepa: unroll(x, None, nsteps, unroll_mode="parallel", compute_loss=True)
        - Training ac_video_jepa with RNN: unroll(x, a, nsteps, unroll_mode="autoregressive",
          ctxt_window_time=1, compute_loss=True)
        - Planning with ac_video_jepa: unroll(x, a, nsteps, unroll_mode="autoregressive",
          ctxt_window_time=k, compute_loss=False)
        - Inference like infern(): unroll(x, a, nsteps, unroll_mode="parallel",
          compute_loss=False, return_all_steps=True)

        Predictor behavior:
        - unroll_mode="parallel" (Conv predictor, is_rnn=False):
          Processes all timesteps in parallel. Uses predictor.context_length to
          determine how many ground truth frames to re-feed at each iteration.
          Output: [B, D, T, H', W'] (same length as input, predictions replace non-context).
          Best for training with full ground truth trajectory available.

        - unroll_mode="autoregressive":
          Step-by-step prediction with sliding window of ctxt_window_time states.
          Each step: takes last ctxt_window_time states, predicts next, appends to sequence.
          Output: [B, D, T_context + nsteps, H', W'] (context + predictions appended).
          Best for planning/inference where future ground truth is not available.
          Note: RNN predictors (is_rnn=True) are a special case with ctxt_window_time=1.

        Args:
            observations: [B, C, T, H, W] - observation sequence
                For training (compute_loss=True): full trajectory with ground truth
                For planning (compute_loss=False): context frames only
            actions: [B, A, T_actions] - action sequence, or None for state-only prediction
                T_actions >= nsteps required for autoregressive mode
            nsteps: number of prediction steps
            unroll_mode: "parallel" or "autoregressive"
                - "parallel": Process all timesteps, refeed GT context on left
                - "autoregressive": Step-by-step, append predictions on right
            ctxt_window_time: Context window size for autoregressive mode.
                For RNN predictors (is_rnn=True), this is effectively 1.
            compute_loss: Whether to compute losses (requires ground truth observations)
            return_all_steps: If True, return list of predictions at each step (like infern).
                If False, return only the final predicted states.

        Returns:
            Tuple of (predicted_states, losses) where:
            - If return_all_steps=False:
              predicted_states: [B, D, T_out, H', W'] - final predicted state sequence
            - If return_all_steps=True:
              predicted_states: List[Tensor] of length nsteps, each [B, D, T_out, H', W']
            - losses: None if compute_loss=False, otherwise tuple of 5 elements:
              (total_loss, reg_loss, reg_loss_unweighted, reg_loss_dict, pred_loss)
        """
        state = self.encoder(observations)
        context_length = getattr(self.predictor, "context_length", 0)

        # Compute regularization loss if needed
        if compute_loss:
            rloss, rloss_unweight, rloss_dict = self.regularizer(state, actions)
            ploss = 0.0
        else:
            rloss = rloss_unweight = rloss_dict = ploss = None

        # Encode actions
        if actions is not None:
            actions_encoded = self.action_encoder(actions)
        else:
            actions_encoded = None

        # Collect all steps if requested
        all_steps = [] if return_all_steps else None

        # Parallel mode: process all timesteps at once, refeed GT context
        if unroll_mode == "parallel":
            predicted_states = state
            for _ in range(nsteps):
                # Predict all timesteps, discard last (no target for it)
                predicted_states = self.predictor(predicted_states, actions_encoded)[
                    :, :, :-1
                ]
                # Collect step if requested
                if return_all_steps:
                    all_steps.append(predicted_states)
                # Refeed ground truth context on the left
                predicted_states = torch.cat(
                    (state[:, :, :context_length], predicted_states), dim=2
                )
                if compute_loss:
                    ploss += self.predcost(state, predicted_states) / nsteps

        # Autoregressive mode: step-by-step with sliding window
        # Note: RNN predictors (is_rnn=True) are a special case with ctxt_window_time=1
        elif unroll_mode == "autoregressive":
            if actions is not None and nsteps > actions.size(2):
                raise ValueError(
                    f"nsteps ({nsteps}) larger than action sequence length ({actions.size(2)})"
                )
            # For RNN predictors, force ctxt_window_time=1
            effective_ctxt_window = 1 if self.single_unroll else ctxt_window_time

            predicted_states = state[:, :, :effective_ctxt_window]
            for i in range(nsteps):
                # Take last ctxt_window_time states
                context_states = predicted_states[:, :, -effective_ctxt_window:]
                # Take corresponding actions
                if actions_encoded is not None:
                    context_actions = actions_encoded[
                        :, :, max(0, i + 1 - effective_ctxt_window) : i + 1
                    ]
                else:
                    context_actions = None
                # Predict and take only last timestep
                pred_step = self.predictor(context_states, context_actions)[:, :, -1:]
                # Append prediction to sequence
                predicted_states = torch.cat([predicted_states, pred_step], dim=2)
                # Collect step if requested
                if return_all_steps:
                    all_steps.append(predicted_states.clone())
                if compute_loss:
                    ploss += (
                        self.predcost(pred_step, state[:, :, i + 1 : i + 2]) / nsteps
                    )
        else:
            raise ValueError(f"Unknown unroll_mode: {unroll_mode}")

        # Compute total loss and return
        if compute_loss:
            loss = rloss + ploss
            losses = (loss, rloss, rloss_unweight, rloss_dict, ploss)
        else:
            losses = None

        # Return all steps or just final state
        if return_all_steps:
            return all_steps, losses
        else:
            return predicted_states, losses


class JEPAProbe(nn.Module):
    """JEPA with a trainable prediction head. The JEPA encoder is kept fixed."""

    def __init__(self, jepa, head, hcost):
        """Initialize with a frozen JEPA, prediction head, and head loss function."""
        super().__init__()
        self.jepa = jepa
        self.head = head
        self.hcost = hcost

    @torch.no_grad()
    def infer(self, observations):
        """Encode observations through JEPA and apply the prediction head."""
        state = self.jepa.encode(observations)
        return self.head(state)

    @torch.no_grad()
    def apply_head(self, embeddings):
        """
        Decode embeddings using the head.
        This is useful for generating predictions from an unrolling of the predictor, for example.
        """
        return self.head(embeddings)

    def forward(self, observations, targets):
        """Forward pass for training the head (JEPA encoder gradients are detached)."""
        with torch.no_grad():
            state = self.jepa.encode(observations)
        output = self.head(state.detach())
        return self.hcost(output, targets)
