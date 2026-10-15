import numpy as np
import torch

from examples.ac_video_jepa.compare_reduced_mpc import Projection, Identity, ReducedRolloutMPC


class Integrator(torch.nn.Module):
    def forward(self,state,action):
        out = state.clone()
        out[:,:2,0,0,0] += action[:,:,0]
        return out


def test_reduced_gru_reprojects_every_transition():
    class Coupled(torch.nn.Module):
        def forward(self,state,action):
            out = state.clone()
            out[:,0,0,0,0] += state[:,2,0,0,0]
            out[:,2,0,0,0] = 1.
            return out

    projection = Projection(np.zeros(512),np.eye(512)[:,:2])
    reduced = ReducedRolloutMPC(Coupled(),projection,torch.device('cpu'))
    full = ReducedRolloutMPC(Coupled(),Identity(),torch.device('cpu'))
    z = torch.zeros(1,512)
    actions = torch.zeros(1,3,2)
    r = reduced.rollout_tensor(z,actions)
    f = full.rollout_tensor(z,actions)
    assert torch.equal(r[0,:,0],torch.zeros(3))
    assert torch.equal(f[0,:,0],torch.tensor([0.,1.,2.]))


def test_projection_accepts_reversed_eigenvector_columns():
    basis = np.eye(512)[:,::-1][:,:2]
    projection = Projection(np.zeros(512),basis)
    z = torch.zeros(1,512)
    z[0,-1] = 3.
    assert torch.equal(projection.encode(z),torch.tensor([[3.,0.]]))


def test_lti_and_ltv_agree_for_exact_integrator_with_coordinate_scaling():
    reducer = Projection(np.zeros(512),np.eye(512)[:,:2])
    # standardized x=(state-[2,3])/[4,5]
    dynamics = dict(mean=torch.tensor([2.,3.]),scale=torch.tensor([4.,5.]),
                    weights=torch.cat([torch.eye(2),torch.diag(torch.tensor([.25,.2])),torch.zeros(1,2)]))
    lti = ReducedRolloutMPC(Integrator(),reducer,torch.device('cpu'),dynamics)
    ltv = ReducedRolloutMPC(Integrator(),reducer,torch.device('cpu'))
    z = torch.randn(3,512)
    actions = torch.randn(3,4,2)
    assert torch.allclose(lti.rollout_tensor(z,actions),ltv.rollout_tensor(z,actions),atol=1e-5)
