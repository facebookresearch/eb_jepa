"""Frozen-encoder affine MPC and condensed GRU LTV-MPC experiments.

LTV shooting sensitivities are computed by central finite differences of the
entire GRU rollout. These are the condensed products of local A_t and B_t;
no global linear GRU approximation is assumed. Norm constraints are preserved
and solved with SLSQP (a constrained quadratic subproblem, not a pure QP).
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.optimize import minimize

from eb_jepa.architectures import ImpalaEncoder, RNNPredictor
from eb_jepa.datasets.utils import init_data
from eb_jepa.datasets.two_rooms.env import DotWall
from examples.ac_video_jepa.train_koopman import collect, fit_koopman, predict, seed


def bounded_quadratic(base, sensitivity, target, reference, action_penalty=1e-3):
    """Optimize absolute actions using a local affine trajectory prediction."""
    m = reference.size
    jac = sensitivity.reshape(-1, m)
    offset = (base - target).reshape(-1) - jac @ reference.reshape(-1)
    # Match mean latent MSE per step, summed over the horizon.
    dimension = base.shape[-1]
    hessian = jac.T @ jac / dimension + action_penalty * np.eye(m)
    gradient = jac.T @ offset / dimension

    def objective(u):
        return float(u @ hessian @ u + 2 * gradient @ u)

    def derivative(u):
        return 2 * (hessian @ u + gradient)

    def constraint(u):
        return 2.45**2 - (u.reshape(-1, 2)**2).sum(1)

    def constraint_jac(u):
        out = np.zeros((m // 2, m))
        for t in range(m // 2):
            out[t, 2*t:2*t+2] = -2 * u[2*t:2*t+2]
        return out

    result = minimize(objective, reference.reshape(-1), jac=derivative,
                      method='SLSQP', constraints=[dict(type='ineq', fun=constraint,
                                                      jac=constraint_jac)],
                      options=dict(maxiter=80, ftol=1e-6))
    actions = result.x.reshape(-1, 2)
    feasible = np.isfinite(actions).all() and np.max(np.linalg.norm(actions, axis=1)) <= 2.45001
    return (actions if feasible else reference.copy()), bool(result.success and feasible)


@torch.no_grad()
def gru_rollout(predictor, initial, actions):
    current = initial.expand(len(actions), -1).reshape(len(actions), -1, 1, 1, 1)
    states = []
    for t in range(actions.shape[1]):
        current = predictor(current, actions[:, t].unsqueeze(-1))
        states.append(current.flatten(1))
    return torch.stack(states, 1)


class LinearMPC:
    def __init__(self, mean, model, action_mean, action_scale):
        self.mean, self.basis = mean, model['basis']
        self.scale = model['scale']
        weights = model['weights']
        rank = len(self.scale)
        self.A = weights[:rank].T
        self.B = weights[rank:-1].T / action_scale
        self.c = weights[-1] - self.B @ action_mean
        self.decode = self.scale[:, None] * self.basis.T

    def plan(self, z, goal, reference):
        x = ((z-self.mean) @ self.basis) / self.scale
        horizon, m = len(reference), reference.size
        sensitivity = np.zeros((len(x), m))
        states, sensitivities = [], []
        for t in range(horizon):
            x = self.A @ x + self.B @ reference[t] + self.c
            sensitivity = self.A @ sensitivity
            sensitivity[:, 2*t:2*t+2] += self.B
            states.append(x @ self.decode + self.mean)
            sensitivities.append(self.decode.T @ sensitivity)
        return bounded_quadratic(np.array(states), np.array(sensitivities), goal, reference)


class LTVMPC:
    def __init__(self, predictor, device, iterations=2, epsilon=1e-3):
        self.predictor, self.device = predictor, device
        self.iterations, self.epsilon = iterations, epsilon

    def rollout(self, z, actions):
        return gru_rollout(self.predictor, torch.as_tensor(z, device=self.device).float(),
                           torch.as_tensor(actions, device=self.device).float()).cpu().numpy()

    def plan(self, z, goal, reference):
        actions = reference.copy()
        ok = True
        for _ in range(self.iterations):
            horizon, m = len(actions), actions.size
            perturbation = np.eye(m).reshape(m, horizon, 2) * self.epsilon
            candidates = np.concatenate([actions[None], actions[None]+perturbation,
                                         actions[None]-perturbation])
            states = self.rollout(z, candidates)
            base = states[0]
            sensitivity = ((states[1:1+m]-states[1+m:]) / (2*self.epsilon)).transpose(1, 2, 0)
            updated, solved = bounded_quadratic(base, sensitivity, goal, actions)
            ok = ok and solved
            # Verify the proposed local solution against the nonlinear GRU.
            fractions = np.array([1., .5, .25, 0.])
            trials = actions[None] + fractions[:, None, None] * (updated-actions)[None]
            actual = self.rollout(z, trials)
            costs = ((actual-goal)**2).mean(2).sum(1) + 1e-3*(trials**2).sum((1, 2))
            actions = trials[np.argmin(costs)]
        return actions, ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint-folder', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--horizon', type=int, default=90)
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--episodes', type=int, default=1)
    parser.add_argument('--train-trajectories', type=int, default=1200)
    parser.add_argument('--validation-trajectories', type=int, default=200)
    parser.add_argument('--test-trajectories', type=int, default=200)
    parser.add_argument('--ltv-iterations', type=int, default=2)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    cfg = OmegaConf.load(Path(args.checkpoint_folder)/'config.yaml')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = ImpalaEncoder(width=1, stack_sizes=(16,cfg.model.henc,cfg.model.dstc),
        num_blocks=2, dropout_rate=None, layer_norm=False, input_channels=2,
        final_ln=True, mlp_output_dim=512, input_shape=(2,65,65)).to(device)
    predictor = RNNPredictor(hidden_size=512, final_ln=encoder.final_ln).to(device)
    checkpoint = Path(args.checkpoint_folder)/'latest.pth.tar'
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    state = {k.replace('_orig_mod.', ''):v for k,v in ckpt['model_state_dict'].items()}
    for prefix, module in [('encoder.', encoder), ('predictor.', predictor)]:
        module.load_state_dict({k[len(prefix):]:v for k,v in state.items() if k.startswith(prefix)})
        module.eval().requires_grad_(False)

    cache = output/'dataset.npz'
    if cache.exists():
        data = dict(np.load(cache))
    else:
        data = {}
        for split, count, split_seed in [('train',args.train_trajectories,101),
                ('validation',args.validation_trajectories,102), ('test',args.test_trajectories,103)]:
            z,a,j,_ = collect(encoder,predictor,count,split_seed,cfg)
            data.update({split+'_z':z,split+'_a':a,split+'_gru':j})
        np.savez_compressed(cache, **data)
    train = data['train_z'].astype(np.float64)
    mean = train.mean((0,1))
    _, basis = np.linalg.eigh((train-mean).reshape(-1,512).T @ (train-mean).reshape(-1,512))
    basis = basis[:, ::-1]
    am = data['train_a'][:,:-1].mean((0,1))
    asc = data['train_a'][:,:-1].std((0,1))+1e-8
    model = fit_koopman(train,data['validation_z'],data['validation_a'],data['train_a'],
                       mean,am,asc,[basis[:,:r] for r in (16,32,64)])
    np.savez_compressed(output/'linear_model.npz', mean=mean,basis=model['basis'],
                       scale=model['scale'],weights=model['weights'],action_mean=am,action_scale=asc)
    predicted = predict(data['test_z'],data['test_a'],mean,model['scale'],am,asc,
                        model['weights'],model['basis'])
    report = dict(checkpoint=str(checkpoint.resolve()), settings=vars(args),
                  linear_selection=model['selection'], candidates=model['candidates'],
                  device=torch.cuda.get_device_name() if device.type=='cuda' else 'cpu',
                  ltv_method='Central finite-difference condensed GRU trajectory sensitivities; nonlinear line search',
                  solver='SLSQP with original Euclidean action bound 2.45',
                  prediction_mse={})
    for h in (1,8,30,32):
        report['prediction_mse'][str(h)] = dict(
            linear=float(((predicted[:,h]-data['test_z'][:,h])**2).mean()),
            gru=float(((data['test_gru'][:,h]-data['test_z'][:,h])**2).mean()))
    (output/'results.json').write_text(json.dumps(report,indent=2))
    _,_,env_config = init_data('two_rooms',dict(cfg.data, num_workers=0,persistent_workers=False))
    planners = {'linear':LinearMPC(mean,model,am,asc),
                'ltv':LTVMPC(predictor,device,args.ltv_iterations)}
    rows, summaries = [], []
    fields = ['planner','episode','timestep','horizon','planning_seconds','solver_success','success','state_dist']
    csv_path = output/'planning_per_timestep.csv'
    with csv_path.open('w',newline='') as f:
        csv.DictWriter(f,fieldnames=fields).writeheader()
    for name, planner in planners.items():
        for ep in range(args.episodes):
            seed(1000+ep)
            env = DotWall(env_config, level='normal',n_allowed_steps=200,
                          rng=np.random.default_rng(1000+ep))
            obs,info = env.reset()
            obs,_,_,_,info = env.step(np.zeros(2))
            def encode(image):
                with torch.no_grad():
                    return encoder(env.normalizer.normalize_state(image.to(device).float())
                                   .unsqueeze(0).unsqueeze(2)).flatten().cpu().numpy()
            goal = encode(info['target_obs'])
            reference = np.zeros((args.horizon,2))
            times = []
            for t in range(args.steps):
                horizon = min(args.horizon,200-t)
                reference = reference[:horizon]
                if device.type=='cuda': torch.cuda.synchronize()
                start = time.perf_counter()
                actions, solved = planner.plan(encode(obs),goal,reference)
                if device.type=='cuda': torch.cuda.synchronize()
                elapsed = time.perf_counter()-start
                obs,_,_,_,info = env.step(actions[0].astype(np.float32))
                metrics = env.eval_state(info['target_position'], info['dot_position'])
                row = dict(planner=name,episode=ep,timestep=t,horizon=horizon,
                           planning_seconds=elapsed,solver_success=solved,
                           success=metrics['success'],state_dist=metrics['state_dist'])
                rows.append(row)
                times.append(elapsed)
                with csv_path.open('a',newline='') as f:
                    csv.DictWriter(f,fieldnames=fields).writerow(row)
                reference = np.concatenate([actions[1:],np.zeros((1,2))])
                print(f'{name} ep={ep} t={t} time={elapsed:.3f}s dist={metrics["state_dist"]:.2f}',flush=True)
            summaries.append(dict(planner=name,episode=ep,mean_planning_seconds=float(np.mean(times)),
                                  final_success=bool(metrics['success']),final_state_dist=float(metrics['state_dist']),
                                  solver_success_rate=float(np.mean([r['solver_success'] for r in rows if r['planner']==name and r['episode']==ep]))))
    report['planning'] = summaries
    (output/'results.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
