import numpy as np
import torch

from examples.ac_video_jepa.compare_linear_mpc import bounded_quadratic, LinearMPC, LTVMPC


def test_quadratic_preserves_euclidean_action_limit():
    base = np.zeros((1, 2))
    sensitivity = np.eye(2).reshape(1, 2, 2)
    actions, success = bounded_quadratic(base, sensitivity, np.array([10., 0.]),
                                         np.zeros((1, 2)), action_penalty=0.)
    assert success
    assert np.allclose(actions, [[2.45, 0.]], atol=1e-4)


def test_affine_mpc_accounts_for_scaling_and_action_mean():
    # Physical dynamics z_next=z+u; fit coordinates x=z/2, v=(u-1)/3.
    model = dict(basis=np.eye(2), scale=np.array([2., 2.]),
                 weights=np.vstack([np.eye(2), 1.5*np.eye(2), [.5, .5]]))
    planner = LinearMPC(np.zeros(2),model,np.ones(2),3*np.ones(2))
    actions, success = planner.plan(np.zeros(2),np.array([1., 0.]),np.zeros((1,2)))
    assert success
    assert np.allclose(actions, [[1./1.002, 0.]], atol=1e-4)


def test_ltv_condensed_sensitivities_recover_linear_optimum():
    class Integrator(torch.nn.Module):
        def forward(self, state, action):
            return state + action.reshape(-1,2,1,1,1)

    planner = LTVMPC(Integrator(),torch.device('cpu'))
    actions, success = planner.plan(np.zeros(2),np.array([1.,0.]),np.zeros((1,2)))
    assert success
    assert np.allclose(actions, [[1./1.002,0.]],atol=1e-3)
