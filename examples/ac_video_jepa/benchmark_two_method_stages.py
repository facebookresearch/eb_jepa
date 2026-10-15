"""Replay saved matched tasks with exclusive, CUDA-synchronized stage timings."""
import argparse
import csv
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from omegaconf import OmegaConf

from eb_jepa.architectures import ImpalaEncoder, RNNPredictor
from eb_jepa.datasets.two_rooms.env import DotWall
from eb_jepa.datasets.utils import init_data
from eb_jepa.jepa import JEPA
from eb_jepa.planning import GCAgent
from examples.ac_video_jepa import compare_linear_mpc as linear
from examples.ac_video_jepa.compare_reduced_mpc import Projection, ReducedRolloutMPC
from examples.ac_video_jepa.train_koopman import seed

STAGES = ['encoding', 'projection', 'prediction', 'objective_setup', 'solver', 'optimizer_overhead']


class StageTimer:
    def __init__(self, device):
        self.device = device
        self.active = False
        self.stack = []
        self.values = defaultdict(float)
        self.calls = defaultdict(int)

    def sync(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

    def wrap(self, category, function):
        def measured(*args, **kwargs):
            if not self.active:
                return function(*args, **kwargs)
            self.sync()
            self.calls[category] += 1
            frame = [time.perf_counter(), 0.]
            self.stack.append(frame)
            try:
                return function(*args, **kwargs)
            finally:
                self.sync()
                elapsed = time.perf_counter() - frame[0]
                self.stack.pop()
                self.values[category] += elapsed - frame[1]
                if self.stack:
                    self.stack[-1][1] += elapsed
        return measured


def summarize(output, rows):
    summary = []
    for method in ['pca_lti', 'gru_mppi']:
        for scope in ['all_steps', 'horizon_90']:
            selected = [r for r in rows if r['method'] == method and (scope == 'all_steps' or r['horizon'] == 90)]
            if not selected:
                continue
            record = dict(method=method, scope=scope, calls=len(selected))
            for key in ['planning_seconds'] + [s+'_seconds' for s in STAGES]:
                a = np.array([r[key] for r in selected]) * 1000
                record[key.replace('_seconds', '_mean_ms')] = float(a.mean())
                record[key.replace('_seconds', '_median_ms')] = float(np.median(a))
                record[key.replace('_seconds', '_p95_ms')] = float(np.percentile(a, 95))
            summary.append(record)
    with (output/'summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader(); writer.writerows(summary)
    (output/'summary.json').write_text(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='checkpoints/reduced_mpc_20_20261012')
    parser.add_argument('--output', required=True)
    parser.add_argument('--episodes', type=int, default=20)
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--methods', nargs='+', choices=['pca_lti','gru_mppi'],
                        default=['pca_lti','gru_mppi'])
    args = parser.parse_args()
    source, output = Path(args.source), Path(args.output)
    previous = json.loads((source/'results.json').read_text())
    assert 1 <= args.episodes <= len(previous['tasks']) and 1 <= args.steps <= 200
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = Path(previous['checkpoint'])
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == previous['checkpoint_sha256']
    cfg = OmegaConf.load(checkpoint.parent/'config.yaml')
    encoder = ImpalaEncoder(width=1, stack_sizes=(16,cfg.model.henc,cfg.model.dstc),
        num_blocks=2, dropout_rate=None, layer_norm=False, input_channels=2,
        final_ln=True, mlp_output_dim=512, input_shape=(2,65,65)).to(device)
    predictor = RNNPredictor(hidden_size=512, final_ln=encoder.final_ln).to(device)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    state = {k.replace('_orig_mod.', ''):v for k,v in ckpt['model_state_dict'].items()}
    for prefix, module in [('encoder.',encoder), ('predictor.',predictor)]:
        module.load_state_dict({k[len(prefix):]:v for k,v in state.items() if k.startswith(prefix)})
        module.eval().requires_grad_(False)
    timer = StageTimer(device)
    encoder.forward = timer.wrap('encoding', encoder.forward)
    if 'pca_lti' in args.methods:
        saved = torch.load(source/'pca_reducer.pt', map_location=device, weights_only=True)
        reducer = Projection(saved['mean'].cpu().numpy(), saved['basis'].cpu().numpy()).to(device)
        reducer.load_state_dict(saved)
        dynamics = torch.load(source/'pca_lti.pt', map_location=device, weights_only=True)
        reducer.encode = timer.wrap('projection', reducer.encode)
        planner = ReducedRolloutMPC(predictor, reducer, device, dynamics)
        planner.rollout_tensor = timer.wrap('prediction', planner.rollout_tensor)
        linear.minimize = timer.wrap('solver', linear.minimize)
        linear.bounded_quadratic = timer.wrap('objective_setup', linear.bounded_quadratic)
    plan_cfg = OmegaConf.create(previous['mppi_config'])
    model = JEPA(encoder, nn.Identity(), predictor, None, None).to(device).eval()
    _,_,env_config = init_data('two_rooms',dict(cfg.data,num_workers=0,persistent_workers=False))
    report = dict(settings=vars(args), status='running', tasks=previous['tasks'][:args.episodes],
        checkpoint=str(checkpoint), device=torch.cuda.get_device_name(0) if device.type=='cuda' else 'cpu',
        mppi_config=previous['mppi_config'], episodes=[],
        seeds=[1000+ep for ep in range(args.episodes)],
        checkpoint_sha256=previous['checkpoint_sha256'],
        mppi_encoding='Current observation encoded once per GCAgent.act; cached latent expanded for candidate rollouts.',
        timing_scope='Exclusive synchronized wall times. Prediction excludes nested encoder/projection. PCA solver is SLSQP minimize only. MPPI optimizer_overhead includes sampling, elite selection, distribution updates and call overhead; no separate solver. Objective_setup includes objective evaluation and PCA quadratic construction. First calls included; synchronization adds profiling overhead. No training, goal encoding or env.step included.',
        pca_note='Saved 64-D PCA LTI and original ReducedRolloutMPC: two finite-difference iterations with line search, unchanged.')
    (output/'results.json').write_text(json.dumps(report,indent=2))
    rows = []
    fields = ['method','episode','timestep','horizon','planning_seconds'] + [s+'_seconds' for s in STAGES] + ['encoding_calls','seed','solver_success','success','state_dist','action_x','action_y']
    with (output/'planning_per_timestep.csv').open('w',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=fields); writer.writeheader()
        for method in args.methods:
            for ep in range(args.episodes):
                seed(1000+ep)
                env = DotWall(env_config,level='normal',n_allowed_steps=200,rng=np.random.default_rng(1000+ep))
                obs,info = env.reset()
                obs,_,_,_,info = env.step(np.zeros(2))
                task = dict(episode=ep,start=np.asarray(info['dot_position']).tolist(),goal=np.asarray(info['target_position']).tolist(),wall=float(env.wall_x),door=float(env.hole_y))
                assert task == previous['tasks'][ep], 'Task differs from original 20-task evaluation'
                def encode(image):
                    return encoder(env.normalizer.normalize_state(image.to(device).float()).unsqueeze(0).unsqueeze(2)).flatten().cpu().numpy()
                with torch.no_grad():
                    goal = encode(info['target_obs'])
                    if method == 'gru_mppi':
                        agent = GCAgent(model,plan_cfg=plan_cfg,normalizer=env.normalizer,env=env)
                        agent.set_goal(info['target_obs'].to(device).float(),info['target_position'])
                        agent.planner.unroll = timer.wrap('prediction',agent.planner.unroll)
                        agent.planner.objective = timer.wrap('objective_setup',agent.planner.objective)
                reference = np.zeros((90,2))
                for t in range(args.steps):
                    horizon = min(90,200-t)
                    timer.values.clear(); timer.calls.clear(); timer.sync()
                    start = time.perf_counter(); timer.active = True
                    with torch.no_grad():
                        if method == 'pca_lti':
                            actions, solved = planner.plan(encode(obs),goal,reference[:horizon])
                        else:
                            image = env.normalizer.normalize_state(obs.to(device).float()).unsqueeze(0).unsqueeze(2)
                            actions = agent.act(image,steps_left=200-t,t0=t==0).cpu().numpy()
                            solved = None
                    timer.sync(); total = time.perf_counter()-start; timer.active = False
                    assert timer.calls['encoding'] == 1, 'Expected exactly one current-observation encoding per plan'
                    components = {s:timer.values[s] for s in STAGES}
                    components['optimizer_overhead'] = total-sum(components.values())
                    assert components['optimizer_overhead'] >= -1e-6
                    obs,_,_,_,info = env.step(actions[0].astype('float32'))
                    metrics = env.eval_state(info['target_position'],info['dot_position'])
                    row = dict(method=method,episode=ep,timestep=t,horizon=horizon,planning_seconds=total,
                        encoding_calls=timer.calls['encoding'],seed=1000+ep,
                        solver_success=solved,success=bool(metrics['success']),state_dist=float(metrics['state_dist']),
                        action_x=float(actions[0,0]),action_y=float(actions[0,1]),**{s+'_seconds':components[s] for s in STAGES})
                    rows.append(row); writer.writerow(row); f.flush()
                    if method == 'pca_lti':
                        reference = np.concatenate([actions[1:],np.zeros((1,2))])
                    if t % 25 == 0:
                        print(f'{method} ep={ep} t={t} total={total*1000:.1f}ms solver={components["solver"]*1000:.1f}ms',flush=True)
                report['episodes'].append(dict(method=method,episode=ep,final_success=row['success'],final_distance=row['state_dist']))
                (output/'results.json').write_text(json.dumps(report,indent=2))
                summarize(output,rows)
    report['status'] = 'complete'
    report['summary'] = summarize(output,rows)
    (output/'results.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report['summary'],indent=2),flush=True)


if __name__ == '__main__':
    main()
