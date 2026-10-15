"""Matched full/PCA/task-projection/autoencoder x LTI/LTV screening.

All costs are decoded original-latent MSE. LTV uses the GRU conjugated by
the reducer at every transition, not simply a projected terminal cost.
"""
import argparse
import copy
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from omegaconf import OmegaConf

from eb_jepa.architectures import ImpalaEncoder, RNNPredictor
from eb_jepa.datasets.two_rooms.env import DotWall
from eb_jepa.datasets.utils import init_data
from eb_jepa.state_decoder import MLPXYHead
from eb_jepa.jepa import JEPA
from eb_jepa.planning import GCAgent
from examples.ac_video_jepa.compare_linear_mpc import LTVMPC
from examples.ac_video_jepa.train_koopman import seed


class Projection(nn.Module):
    def __init__(self, mean, basis):
        super().__init__()
        self.register_buffer('mean', torch.as_tensor(mean).float())
        self.register_buffer('basis', torch.as_tensor(np.ascontiguousarray(basis)).float())
        self.rank = basis.shape[1]

    def encode(self, z):
        return (z-self.mean) @ self.basis

    def decode(self, x):
        return x @ self.basis.T + self.mean


class Identity(nn.Module):
    rank = 512

    def encode(self, z):
        return z

    def decode(self, x):
        return x


class Autoencoder(nn.Module):
    def __init__(self, mean, scale, rank):
        super().__init__()
        self.rank = rank
        self.register_buffer('mean',torch.as_tensor(mean).float())
        self.register_buffer('scale',torch.as_tensor(scale).float())
        self.encoder = nn.Sequential(nn.Linear(512,128),nn.Tanh(),nn.Linear(128,rank))
        self.decoder = nn.Sequential(nn.Linear(rank,128),nn.Tanh(),nn.Linear(128,512))

    def encode(self,z):
        return self.encoder((z-self.mean)/self.scale)

    def decode(self,x):
        return self.decoder(x)*self.scale+self.mean


def train_autoencoder(train, validation, rank, device, epochs):
    seed(110)
    model = Autoencoder(train.mean(0),train.std(0)+1e-6,rank).to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)
    tx = torch.as_tensor(train,device=device).float()
    vx = torch.as_tensor(validation,device=device).float()
    best, best_state, history = float('inf'),None,[]
    for epoch in range(epochs):
        model.train()
        indices = torch.randperm(len(tx),device=device)
        for batch in indices.split(1024):
            optimizer.zero_grad()
            reconstructed = model.decode(model.encode(tx[batch]))
            loss = ((reconstructed-tx[batch])**2).mean()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            score = float(((model.decode(model.encode(vx))-vx)**2).mean())
        history.append(score)
        if score<best:
            best, best_state = score,copy.deepcopy(model.state_dict())
        if epoch%10==0:
            print(f'AE epoch={epoch} validation_mse={score:.5f}',flush=True)
    model.load_state_dict(best_state)
    return model.eval().requires_grad_(False),dict(best_validation_mse=best,history=history)


class ReducedRolloutMPC(LTVMPC):
    """Same constrained successive-linearization solver for every variant."""
    def __init__(self,predictor,reducer,device,dynamics=None,iterations=2):
        super().__init__(predictor,device,iterations)
        self.reducer,self.dynamics = reducer,dynamics

    @torch.no_grad()
    def rollout_tensor(self,z,actions):
        # z: B,D; actions: B,H,2. Reproject after EVERY GRU transition.
        x = self.reducer.encode(z)
        outputs = []
        for t in range(actions.shape[1]):
            if self.dynamics is None:
                latent = self.reducer.decode(x)
                next_z = self.predictor(latent.reshape(-1,512,1,1,1),
                                        actions[:,t].unsqueeze(-1)).flatten(1)
                x = self.reducer.encode(next_z)
            else:
                d = self.dynamics
                features = torch.cat([(x-d['mean'])/d['scale'],actions[:,t],
                                      torch.ones(len(x),1,device=self.device)],1)
                x = (features @ d['weights'])*d['scale']+d['mean']
            outputs.append(self.reducer.decode(x))
        return torch.stack(outputs,1)

    def rollout(self,z,actions):
        a = torch.as_tensor(actions,device=self.device).float()
        initial = torch.as_tensor(z,device=self.device).float().expand(len(a),-1)
        return self.rollout_tensor(initial,a).cpu().numpy()


def fit_lti(reducer,data,predictor,device):
    with torch.no_grad():
        train = reducer.encode(torch.as_tensor(data['train_z'],device=device).float()).cpu().numpy().astype('float64')
    mean,scale = train.mean((0,1)),train.std((0,1))+1e-6
    x = (train-mean)/scale
    features = np.column_stack([x[:,:-1].reshape(-1,reducer.rank),
                               data['train_a'][:,:-1].reshape(-1,2),
                               np.ones(len(train)*(train.shape[1]-1))])
    target = x[:,1:].reshape(-1,reducer.rank)
    gram,rhs = features.T@features/len(features),features.T@target/len(features)
    candidates,best = [],None
    for ridge in (1e-6,1e-4,1e-2,.1,1.):
        penalty = np.eye(features.shape[1])*ridge
        penalty[-1,-1] = 0
        weights = np.linalg.solve(gram+penalty,rhs)
        dynamics = {k:torch.as_tensor(v,device=device).float() for k,v in
                    dict(mean=mean,scale=scale,weights=weights).items()}
        planner = ReducedRolloutMPC(predictor,reducer,device,dynamics)
        with torch.no_grad():
            validation_z = torch.as_tensor(data['validation_z'],device=device).float()
            actions = torch.as_tensor(data['validation_a'][:,:32],device=device).float()
            prediction = planner.rollout_tensor(validation_z[:,0],actions)
            score = float(((prediction-validation_z[:,1:33])**2).mean())
        candidates.append(dict(ridge=ridge,validation_rollout_mse=score))
        if np.isfinite(score) and (best is None or score<best[0]):
            best = score,dynamics,ridge
    if best is None:
        raise RuntimeError('No finite LTI validation candidate')
    return best[1],dict(selected_ridge=best[2],candidates=candidates)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source',default='checkpoints/linear_ltv_mpc_20261011')
    parser.add_argument('--output',default='checkpoints/reduced_mpc_20261012')
    parser.add_argument('--rank',type=int,default=64)
    parser.add_argument('--episodes',type=int,default=3)
    parser.add_argument('--steps',type=int,default=200)
    parser.add_argument('--horizon',type=int,default=90)
    parser.add_argument('--ae-epochs',type=int,default=60)
    parser.add_argument('--include-mppi',action='store_true')
    args = parser.parse_args()
    if not (2<=args.rank<=128 and 1<=args.steps<=200 and 1<=args.horizon<=200 and args.episodes>=1):
        parser.error('rank 2..128; steps and horizon 1..200; episodes >=1 required')
    torch.set_num_threads(1)
    source,output = Path(args.source),Path(args.output)
    output.mkdir(parents=True,exist_ok=False)
    original = json.loads((source/'results.json').read_text())
    checkpoint = Path(original['checkpoint'])
    cfg = OmegaConf.load(checkpoint.parent/'config.yaml')
    data = dict(np.load(source/'dataset.npz'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed(110)
    encoder = ImpalaEncoder(width=1,stack_sizes=(16,cfg.model.henc,cfg.model.dstc),
        num_blocks=2,dropout_rate=None,layer_norm=False,input_channels=2,
        final_ln=True,mlp_output_dim=512,input_shape=(2,65,65)).to(device)
    predictor = RNNPredictor(hidden_size=512,final_ln=encoder.final_ln).to(device)
    ckpt = torch.load(checkpoint,map_location=device,weights_only=False)
    state = {k.replace('_orig_mod.',''):v for k,v in ckpt['model_state_dict'].items()}
    for prefix,module in [('encoder.',encoder),('predictor.',predictor)]:
        module.load_state_dict({k[len(prefix):]:v for k,v in state.items() if k.startswith(prefix)})
        module.eval().requires_grad_(False)
    train = data['train_z'].reshape(-1,512).astype('float64')
    mean = train.mean(0)
    centered = train-mean
    _,vectors = np.linalg.eigh(centered.T@centered)
    pca = vectors[:,::-1][:,:args.rank]

    # Task-aware linear subspace: ridge directions predicting the frozen
    # checkpoint's position probe, completed by orthogonal residual PCA.
    probe = MLPXYHead(512).to(device)
    probe.load_state_dict(ckpt['xy_head_state_dict'])
    probe.eval().requires_grad_(False)
    with torch.no_grad():
        positions = probe.mlp(torch.as_tensor(train,device=device).float()).cpu().numpy()
    directions = np.linalg.solve(centered.T@centered/len(train)+.01*np.eye(512),
                                 centered.T@(positions-positions.mean(0))/len(train))
    q,_ = np.linalg.qr(directions)
    residual = centered-(centered@q)@q.T
    _,residual_vectors = np.linalg.eigh(residual.T@residual)
    task_basis = np.column_stack([q,residual_vectors[:,::-1][:,:args.rank-2]])
    task_basis,_ = np.linalg.qr(task_basis)
    ae,ae_report = train_autoencoder(train,data['validation_z'].reshape(-1,512),
                                     args.rank,device,args.ae_epochs)
    reducers = dict(full=Identity().to(device),pca=Projection(mean,pca).to(device),
                    task=Projection(mean,task_basis).to(device),ae=ae)
    report = dict(settings=vars(args),checkpoint=str(checkpoint),
                  checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                  dataset_sha256=hashlib.sha256((source/'dataset.npz').read_bytes()).hexdigest(),
                  device=torch.cuda.get_device_name() if device.type=='cuda' else 'cpu',
                  task_projection='Two ridge directions predicting frozen checkpoint position-probe outputs; residual PCA completion. Probe labels are estimates, not simulator ground truth.',
                  ltv='Reduced dynamics E(GRU(D(x),u)); full baseline E=D=identity. Central finite-difference condensed sensitivities, two iterations with nonlinear line search.',
                  solver='Shared SLSQP solver, decoded original latent MSE plus .001 action penalty; Euclidean action norm <=2.45; no explicit wall constraint.',
                  autoencoder=ae_report,models={},tasks=[],planning=[])
    planners = {}
    for name,reducer in reducers.items():
        print(f'Fit and evaluate {name}',flush=True)
        torch.save(reducer.state_dict(),output/f'{name}_reducer.pt')
        lti,selection = fit_lti(reducer,data,predictor,device)
        torch.save(lti,output/f'{name}_lti.pt')
        with torch.no_grad():
            test_z = torch.as_tensor(data['test_z'],device=device).float()
            reconstruction = float(((reducer.decode(reducer.encode(test_z))-test_z)**2).mean())
        for kind,dynamics in [('lti',lti),('ltv',None)]:
            key = f'{name}_{kind}'
            planner = ReducedRolloutMPC(predictor,reducer,device,dynamics)
            planners[key] = planner
            with torch.no_grad():
                predicted = planner.rollout_tensor(test_z[:,0],torch.as_tensor(data['test_a'][:,:32],device=device).float())
                mse = {str(h):float(((predicted[:,h-1]-test_z[:,h])**2).mean()) for h in (1,8,30,32)}
            report['models'][key] = dict(rank=reducer.rank,reconstruction_mse=reconstruction,
                                        prediction_mse=mse,lti_selection=selection if kind=='lti' else None)
    if args.include_mppi:
        plan_cfg = OmegaConf.load(cfg.eval.plan_cfg_path)
        plan_cfg.logging.tqdm_silent = True
        planners = {'gru_mppi':None,**planners}
        report['models'] = {'gru_mppi':copy.deepcopy(report['models']['full_ltv']),**report['models']}
        report['mppi_config'] = OmegaConf.to_container(plan_cfg,resolve=True)
        report['mppi_note'] = 'Original GCAgent/MPPIPlanner with original latent objective and defaults, without the .001 action penalty used by experimental controllers.'
        model = JEPA(encoder,nn.Identity(),predictor,None,None).to(device).eval()
    report['status'] = 'running'
    (output/'results.json').write_text(json.dumps(report,indent=2))
    _,_,env_config = init_data('two_rooms',dict(cfg.data,num_workers=0,persistent_workers=False))
    fields = ['method','episode','timestep','horizon','planning_seconds','solver_success','success','state_dist']
    with (output/'planning_per_timestep.csv').open('w',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        for key,planner in planners.items():
            for ep in range(args.episodes):
                seed(1000+ep)
                env = DotWall(env_config,level='normal',n_allowed_steps=200,rng=np.random.default_rng(1000+ep))
                obs,info = env.reset()
                obs,_,_,_,info = env.step(np.zeros(2))
                task = dict(episode=ep,start=np.asarray(info['dot_position']).tolist(),
                            goal=np.asarray(info['target_position']).tolist(),wall=float(env.wall_x),door=float(env.hole_y))
                if key==next(iter(planners)):
                    report['tasks'].append(task)
                else:
                    assert task==report['tasks'][ep], 'Unmatched evaluation task'
                def encode(image):
                    with torch.no_grad():
                        return encoder(env.normalizer.normalize_state(image.to(device).float())
                            .unsqueeze(0).unsqueeze(2)).flatten().cpu().numpy()
                goal = encode(info['target_obs'])
                if key=='gru_mppi':
                    agent = GCAgent(model,plan_cfg=plan_cfg,normalizer=env.normalizer,env=env)
                    agent.set_goal(info['target_obs'].to(device).float(),info['target_position'])
                reference = np.zeros((args.horizon,2))
                times,full_times,statuses,ever = [],[],[],False
                for t in range(args.steps):
                    horizon = min(args.horizon,200-t)
                    if device.type=='cuda': torch.cuda.synchronize()
                    start = time.perf_counter()
                    if key=='gru_mppi':
                        obs_tensor = env.normalizer.normalize_state(obs.to(device).float()).unsqueeze(0).unsqueeze(2)
                        with torch.no_grad():
                            actions = agent.act(obs_tensor,steps_left=200-t,t0=t==0).cpu().numpy()
                        solved = None  # Sampling planner has no numerical solver convergence flag.
                    else:
                        actions,solved = planner.plan(encode(obs),goal,reference[:horizon])
                    if device.type=='cuda': torch.cuda.synchronize()
                    elapsed = time.perf_counter()-start
                    obs,_,_,_,info = env.step(actions[0].astype('float32'))
                    metrics = env.eval_state(info['target_position'],info['dot_position'])
                    ever |= bool(metrics['success'])
                    times.append(elapsed); statuses.append(solved)
                    if horizon==args.horizon: full_times.append(elapsed)
                    writer.writerow(dict(method=key,episode=ep,timestep=t,horizon=horizon,
                        planning_seconds=elapsed,solver_success=solved,
                        success=metrics['success'],state_dist=metrics['state_dist']))
                    f.flush()
                    if key!='gru_mppi':
                        reference = np.concatenate([actions[1:],np.zeros((1,2))])
                    if t%50==0:
                        print(f'{key} ep={ep} t={t} time={elapsed:.3f}s dist={metrics["state_dist"]:.2f}',flush=True)
                report['planning'].append(dict(method=key,episode=ep,
                    final_success=bool(metrics['success']),ever_success=ever,
                    final_distance=float(metrics['state_dist']),mean_seconds=float(np.mean(times)),
                    full_horizon_mean_seconds=float(np.mean(full_times)),
                    solver_success_rate=None if key=='gru_mppi' else float(np.mean(statuses))))
                (output/'results.json').write_text(json.dumps(report,indent=2))
                print(f'Finished {key} ep={ep}: {report["planning"][-1]}',flush=True)
    report['status'] = 'complete'
    (output/'results.json').write_text(json.dumps(report,indent=2))
    make_summary(output,report)


def make_summary(output,report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    summary = []
    for key,model in report['models'].items():
        episodes = [p for p in report['planning'] if p['method']==key]
        summary.append(dict(method=key,rank=model['rank'],episodes=len(episodes),
            final_success_rate=np.mean([e['final_success'] for e in episodes]),
            ever_success_rate=np.mean([e['ever_success'] for e in episodes]),
            mean_final_distance=np.mean([e['final_distance'] for e in episodes]),
            full_horizon_mean_ms=1000*np.mean([e['full_horizon_mean_seconds'] for e in episodes]),
            solver_success_rate=None if key=='gru_mppi' else np.mean([e['solver_success_rate'] for e in episodes]),
            reconstruction_mse=model['reconstruction_mse'],mse_30=model['prediction_mse']['30']))
    with (output/'summary.csv').open('w',newline='') as f:
        w = csv.DictWriter(f,fieldnames=list(summary[0])); w.writeheader(); w.writerows(summary)
    fig,axes = plt.subplots(1,3,figsize=(15,4))
    labels = [s['method'] for s in summary]
    for ax,field,title in zip(axes,['full_horizon_mean_ms','mean_final_distance','mse_30'],
                             ['Planning time, H=90 (ms)','Final goal distance (pixels)','30-step prediction MSE']):
        ax.bar(labels,[s[field] for s in summary]); ax.set_title(title)
        ax.tick_params(axis='x',rotation=60); ax.grid(axis='y',alpha=.2)
    fig.tight_layout(); fig.savefig(output/'comparison.png',dpi=180)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
