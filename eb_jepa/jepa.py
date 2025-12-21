import time

import torch
import torch.nn as nn

from eb_jepa.logging import get_logger

logging = get_logger(__name__)


######################################################
# a basic JEPA class. No learning abilities
# this is for planning and inference only.
# use the full JEPA class for SSL training.
class JEPAbase(nn.Module):
    def __init__(self, encoder, aencoder, predictor):
        """
        Action-Conditioned Joint Embedding Predictive Architecture world model.
        This class has no training ability.
        Use the JEPA subclass for training.
        """
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

    # just runs the encoder on a sequence of observations
    # and returns the encoder output sequence
    @torch.no_grad()
    def encode(self, observations):
        return self.encoder(observations)

    # inference producing single-step predictions over all
    # elements in a sequence in  parallel.
    @torch.no_grad()
    def infer(self, observations, actions):
        return self.infern(observations, actions, nsteps=1)[0]

    @torch.no_grad()
    def infern(self, observations, actions, nsteps=1):
        # check number of steps.
        state = self.encoder(observations)
        context_length = self.predictor.context_length

        if actions is not None:
            actions = self.action_encoder(actions)

        predi = state
        preds = []
        for _ in range(nsteps):
            predi = self.predictor(predi, actions)[:, :, :-1]
            preds.append(predi)
            predi = torch.cat((state[:, :, :context_length], predi), dim=2)

        # compute total loss here
        return preds

    # TODO: refactor predictor
    # perform a multi-step prediction, auto-regressively in state space.
    # Predictions are performed sequentially starting from a given context of
    # observations.shape[2] frames, on actions.shape[2] action time steps.
    # The last prediction timestep predi[:, :, -1:] is concatenated to the
    # input state for the next prediction step.
    # Optionally, a context window can be used to limit the number of past actions and frames
    # the predictor can attend to.
    # Returns predin: a concatention of groundtruth context embeddings and predictions.
    @torch.no_grad()
    def unrolln(self, observations, actions, nsteps, ctxt_window_time=1):
        """
        Input shape: observations: (Batch, Feature, Time, Height, Width) OR (Batch, Time, Dim)
        actions: (Batch, Feature, Time, Height, Width)
        Output shape: predin: (Batch, Feature, Time, Height, Width) OR (Batch, Time, Dim)
        """
        if nsteps > actions.size(2):
            raise NameError(
                "number of prediction steps larger than length of action sequence"
            )
        # Input Encoding
        state = self.encoder(observations)
        # Action Encoding
        actions = self.action_encoder(actions)
        # prediction loop through steps.
        # we just run the predictor as if it were a recurrent net.
        if self.single_unroll:
            curr_state = state[:, :, :1]
            predin = curr_state
            for i in range(nsteps):
                curr_action = actions[:, :, i : i + 1]
                curr_state = self.predictor(curr_state, curr_action)
                predin = torch.cat([predin, curr_state], dim=2)
        else:
            predin = state
            for i in range(nsteps):
                predi = self.predictor(
                    predin[:, :, -ctxt_window_time:],
                    actions[:, :, max(0, i + 1 - ctxt_window_time) : i + 1],
                )
                predi = predi[:, :, -1:]  # take the last time step
                predin = torch.cat([predin, predi], dim=2)
        return predin


################################################################
# A trainable JEPA class
# with a prediction loss and an anti-collapse regularizer loss
class JEPA(JEPAbase):
    def __init__(self, encoder, aencoder, predictor, regularizer, predcost):
        """
        Action-Conditioned Joint Embedding Predictive Architecture world model.
        Args:
        """
        super().__init__(encoder, aencoder, predictor)
        # Anti-Collapse Regularizer
        self.regularizer = regularizer
        # prediction loss
        self.predcost = predcost
        self.ploss = 0
        self.rloss = 0

    # training forward with a multi-step auto-regressive prediction
    # observations is a 5d tensor containing a sequence of observations
    # (Batch, Feature, Time, Height, Width)
    # actions is a 5d tensor containing a sequence of actions
    # (Batch, Feature, Time, Height, Width)
    def forwardn(self, observations, actions, nsteps=1):
        # Input Encoding
        state = self.encoder(observations)
        context_length = self.predictor.context_length

        # VC loss
        rloss, rloss_unweight, rloss_dict = self.regularizer(state, actions)

        if actions is not None:
            actions = self.action_encoder(actions)

        predi = state
        ploss = 0.0
        if self.single_unroll:
            curr_state = state[:, :, :1]  # (b, d, h, w)
            for i in range(nsteps):
                curr_action = actions[:, :, i : i + 1]
                curr_state = self.predictor(curr_state, curr_action)
                ploss += self.predcost(curr_state, state[:, :, i + 1 : i + 2]) / nsteps
        else:
            predi = state  # (b, d, t, h, w)
            # If predictor treats timesteps as batch dimension, reshaping b t c h w -> (b t) c h w,
            # Then receptive field of predictor is one timestep only, so it is time-causal.
            for _ in range(nsteps):
                # Discard latest timestep prediction since there is no
                # visual embedding target for it
                predi = self.predictor(predi, actions)[:, :, :-1]
                # Refeed 1st context_length grountruth embedding timesteps on the left
                # as context for the next call to the predictor
                predi = torch.cat((state[:, :, :context_length], predi), dim=2)
                ploss += self.predcost(state, predi) / nsteps

        # compute total loss here
        loss = rloss + ploss
        return loss, rloss, rloss_unweight, rloss_dict, ploss


################################################################
# a container that contains a JEPA and a trainable prediction head.
# the prediction head can be used as a decoder:
# simply set the targets are identical to the observations.
class JEPAProbe(nn.Module):
    def __init__(self, jepa, head, hcost):
        """
        A JEPA probe that includes a prediction head that
        can be trained supervised.
        The JEPA is kept fixed.
        """
        super().__init__()
        self.jepa = jepa
        # prediction head for a supervised task
        self.head = head
        # loss for the prediction head
        self.hcost = hcost

    # encode a sequence through the JEPA
    # run the encoded state through the head and
    # return the result
    @torch.no_grad()
    def infer(self, observations):
        state = self.jepa.encode(observations)
        return self.head(state)

    @torch.no_grad()
    def apply_head(self, embeddings):
        """
        Decode embeddings using the head.
        This is useful for generating predictions from an unrolling of the predictor, for example.
        """
        return self.head(embeddings)

    # forward to train the head
    def forward(self, observations, targets):
        with torch.no_grad():
            state = self.jepa.encode(observations)
        # run the prediction head, but do not
        # backprop through the JEPA encoder
        output = self.head(state.detach())
        return self.hcost(output, targets)
