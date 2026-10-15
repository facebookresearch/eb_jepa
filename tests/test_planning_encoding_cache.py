import pytest
import torch
from torch import nn
from omegaconf import OmegaConf

from eb_jepa.jepa import JEPA
from eb_jepa.planning import GCAgent, PlanningResult


class CountingEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.calls = 0

    def forward(self, obs):
        self.calls += 1
        return obs * self.weight


class Predictor(nn.Module):
    is_rnn = True

    def forward(self, state, actions):
        return state + actions[:, :1, -1:].unsqueeze(-1).unsqueeze(-1)


def test_planning_encodes_once_and_refreshes_each_observation():
    encoder = CountingEncoder()
    model = JEPA(encoder, nn.Identity(), Predictor(), None, None).eval()
    agent = GCAgent(model, plan_cfg=None)
    agent.plan_cfg = OmegaConf.create({'ctxt_window_time':1})
    agent.num_act_stepped = 1
    actions = torch.ones(5, 2, 3)

    class Planner:
        fail = False

        def plan(self, obs, **kwargs):
            if self.fail:
                raise RuntimeError('solver failure')
            for _ in range(3):
                result = agent.unroll(obs, actions)
                expected = torch.cat([obs.expand(5,-1,-1,-1,-1)+i for i in range(4)], dim=2)
                torch.testing.assert_close(result, expected)
            return PlanningResult(torch.zeros(3,2))

    agent.planner = Planner()
    obs = torch.zeros(1,1,1,1,1)
    agent.act(obs)
    assert encoder.calls == 1
    assert agent._planning_state is None
    agent.act(obs+10)
    assert encoder.calls == 2
    agent.planner.fail = True
    with pytest.raises(RuntimeError, match='solver failure'):
        agent.act(obs)
    assert agent._planning_state is None
    # A standalone rollout must encode its own observation after the failure.
    agent.unroll(obs, actions)
    assert encoder.calls == 4


def test_cached_rollout_matches_real_encoder_and_gru():
    from eb_jepa.architectures import ImpalaEncoder, RNNPredictor
    torch.manual_seed(7)
    encoder = ImpalaEncoder(width=1, stack_sizes=(16,16,16), num_blocks=2,
        dropout_rate=None, layer_norm=False, input_channels=2, final_ln=True,
        mlp_output_dim=512, input_shape=(2,65,65))
    predictor = RNNPredictor(hidden_size=512, final_ln=encoder.final_ln)
    model = JEPA(encoder, nn.Identity(), predictor, None, None).eval()
    obs = torch.randn(1,2,1,65,65)
    actions = torch.randn(4,2,5)
    with torch.no_grad():
        original,_ = model.unroll(obs.repeat(4,1,1,1,1), actions, nsteps=5,
            unroll_mode='autoregressive', compute_loss=False)
        cached,_ = model.unroll(obs, actions, nsteps=5, unroll_mode='autoregressive',
            compute_loss=False, encoded_state=model.encode(obs).expand(4,-1,-1,-1,-1))
    torch.testing.assert_close(cached, original, atol=1e-5, rtol=1e-4)
