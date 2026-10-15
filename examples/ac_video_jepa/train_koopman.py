"""Fit a controlled affine Koopman approximation on frozen JEPA embeddings.

Run from repository root: python -m examples.ac_video_jepa.train_koopman
Compare frozen GRU, full-512 Koopman, and PCA + Koopman on identical trajectories.
PCA/scaling/regression use train only; validation selects each Koopman model.
"""
from pathlib import Path
import json
import random
import hashlib
import numpy as np
import torch
from omegaconf import OmegaConf
from eb_jepa.architectures import ImpalaEncoder, RNNPredictor
from eb_jepa.datasets.utils import init_data
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / 'checkpoints/ac_video_jepa_gpu'
OUT = MODEL / 'koopman_e11_comparison'
SEED = 20261002


def seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


@torch.no_grad()
def collect(encoder, predictor, count, split_seed, cfg):
    seed(split_seed)
    overrides = dict(cfg.data)
    overrides.update(num_workers=0, persistent_workers=False, sample_length=33,
                     n_steps=34, size=count, val_size=count)
    _, loader, _ = init_data('two_rooms', overrides)
    zs, acts, jepas, images = [], [], [], []
    device = next(encoder.parameters()).device
    for batch in loader:
        x, a = batch.states.to(device), batch.actions.to(device)
        b, c, t, h, w = x.shape
        # Independent frame encoding, as in the existing evaluation.
        z = encoder(x.permute(0,2,1,3,4).flatten(0,1).unsqueeze(2))
        z = z.flatten(1).reshape(b,t,512)
        current = z[:,0].reshape(b,512,1,1,1)
        rollout = [current.flatten(1)]
        for k in range(t-1):
            current = predictor(current, a[:,:,k:k+1])
            rollout.append(current.flatten(1))
        zs.append(z.cpu().numpy())
        acts.append(a.permute(0,2,1).cpu().numpy())
        jepas.append(torch.stack(rollout,1).cpu().numpy())
        if len(images)<1:
            images.append(x.cpu().numpy())
        if sum(len(v) for v in zs) % 100 == 0:
            print(f'Seed {split_seed}: collected {sum(len(v) for v in zs)}/{count}', flush=True)
    return np.concatenate(zs), np.concatenate(acts), np.concatenate(jepas), images[0]


def predict(z, actions, mean, scale, action_mean, action_scale, weights, basis=None):
    state = z[:,0].astype(np.float64)-mean
    if basis is not None:
        state = state@basis
    state = state/scale
    values = [z[:,0].copy()]  # All methods share the exact initial latent.
    for k in range(z.shape[1]-1):
        features = np.column_stack([state, (actions[:,k]-action_mean)/action_scale,
                                    np.ones(len(z))])
        state = features@weights
        restored = state*scale
        if basis is not None:
            restored = restored@basis.T
        values.append(restored+mean)
    return np.stack(values,1)


def fit_koopman(train, validation, validation_actions, train_actions, mean,
                action_mean, action_scale, bases):
    """Select rank/ridge using autoregressive validation MSE in original latent space."""
    candidates, best = [], None
    for basis in bases:
        projected = train-mean
        if basis is not None:
            projected = projected@basis
        rank = projected.shape[-1]
        scale = projected.std((0,1))+1e-8
        y = projected/scale
        x = np.column_stack([y[:,:-1].reshape(-1,rank),
            ((train_actions[:,:-1]-action_mean)/action_scale).reshape(-1,2),
            np.ones(len(train)*(train.shape[1]-1))])
        target = y[:,1:].reshape(-1,rank)
        gram, rhs = x.T@x/len(x), x.T@target/len(x)
        for ridge in (1e-6,1e-4,1e-2,.1,1.):
            penalty = np.eye(rank+3)*ridge
            penalty[-1,-1] = 0
            weights = np.linalg.solve(gram+penalty,rhs)
            pred = predict(validation,validation_actions,mean,scale,
                           action_mean,action_scale,weights,basis)
            score = float(((pred[:,1:]-validation[:,1:])**2).mean())
            record = dict(rank=rank,ridge=ridge,validation_rollout_mse=score)
            candidates.append(record)
            if np.isfinite(score) and (best is None or score<best[0]):
                best = score,record,scale.copy(),weights.copy(),basis
    if best is None:
        raise RuntimeError('All ridge candidates produced non-finite validation errors.')
    _, selected,scale,weights,basis = best
    return dict(selection=selected,candidates=candidates,scale=scale,weights=weights,basis=basis)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    seed(SEED)
    cfg = OmegaConf.load(MODEL/'config.yaml')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = ImpalaEncoder(width=1, stack_sizes=(16,cfg.model.henc,cfg.model.dstc),
        num_blocks=2, dropout_rate=None, layer_norm=False, input_channels=2,
        final_ln=True, mlp_output_dim=512, input_shape=(2,65,65)).to(device)
    predictor = RNNPredictor(hidden_size=512, final_ln=encoder.final_ln).to(device)
    checkpoint = MODEL/'e-11.pth.tar'
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    state = {k.replace('_orig_mod.',''):v for k,v in ckpt['model_state_dict'].items()}
    for prefix, module in [('encoder.',encoder), ('predictor.',predictor)]:
        module.load_state_dict({k[len(prefix):]:v for k,v in state.items() if k.startswith(prefix)}, strict=True)
        module.eval().requires_grad_(False)
    # Reuse the same trajectories as the PCA baseline for a fair comparison.
    cache = MODEL/'koopman_e11'/'dataset.npz'
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        data = dict(np.load(cache))
    else:
        data = {}
        for name,count,s in [('train',1200,SEED),('validation',200,SEED+1),('test',200,SEED+2)]:
            z,a,j,im = collect(encoder,predictor,count,s,cfg)
            data.update({name+'_z':z,name+'_a':a,name+'_jepa':j,name+'_images':im})
        np.savez_compressed(cache, **data)
    train = data['train_z'].astype(np.float64)
    flat = train.reshape(-1,512)
    mean = flat.mean(0)
    centered = flat-mean
    eigenvalues, eigenvectors = np.linalg.eigh(centered.T@centered)
    eigenvalues = np.maximum(eigenvalues[::-1], 0)
    variance = eigenvalues/eigenvalues.sum()
    pca_basis = eigenvectors[:,::-1]
    display_basis = pca_basis[:,:2]
    action_mean = data['train_a'][:,:-1].mean((0,1))
    action_scale = data['train_a'][:,:-1].std((0,1))+1e-8
    models = {}
    for name,bases in [('koopman_full512',[None]),
                       ('koopman_pca',[pca_basis[:,:r] for r in (16,32,64)])]:
        print(f'Fitting {name}...', flush=True)
        models[name] = fit_koopman(train,data['validation_z'],data['validation_a'],
            data['train_a'],mean,action_mean,action_scale,bases)
    z = data['test_z']
    predictions = {'gru': data['test_jepa']}
    model_reports = {}
    for name,model in models.items():
        scale,weights,basis = model['scale'],model['weights'],model['basis']
        rank = model['selection']['rank']
        predictions[name] = predict(z,data['test_a'],mean,scale,
                                   action_mean,action_scale,weights,basis)
        arrays = dict(latent_mean=mean,coordinate_scale=scale,action_mean=action_mean,
            action_scale=action_scale,weights=weights,A=weights[:rank].T,
            B=weights[rank:-1].T,c=weights[-1],display_basis=display_basis,
            uses_pca=basis is not None)
        if basis is not None:
            arrays['pca_basis'] = basis
        np.savez_compressed(OUT/f'{name}_model.npz',**arrays)
        model_reports[name] = dict(selection=model['selection'],candidates=model['candidates'],
            spectral_radius=float(np.abs(np.linalg.eigvals(weights[:rank])).max()))
        if basis is not None:
            reconstruction = ((z-mean)@basis)@basis.T+mean
            model_reports[name]['test_pca_reconstruction_mse'] = float(((reconstruction-z)**2).mean())
            model_reports[name]['explained_variance'] = float(variance[:rank].sum())
    errors = {}
    for name,values in predictions.items():
        assert np.isfinite(values).all() and np.array_equal(values[:,0],z[:,0]), name
        errors[name] = ((values-z)**2).mean(-1)
    np.savez_compressed(OUT/'test_predictions.npz',ground_truth=z,**predictions,
                        **{'mse_'+name:value for name,value in errors.items()})
    report = dict(checkpoint=str(checkpoint),
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        checkpoint_epoch=ckpt.get('epoch'),seed=SEED,dataset_cache=str(cache),
        split_counts={s:len(data[s+'_z']) for s in ['train','validation','test']},
        frames_per_trajectory=z.shape[1],models=model_reports,
        test_rollout_mse={name:float(value[:,1:].mean()) for name,value in errors.items()},
        display_pca_explained_variance=float(variance[:2].sum()),
        protocol='Frozen checkpoint GRU. Both Koopman models fit train only. Validation selects ridge independently and PCA rank (16/32/64). No validation refit. Identical test trajectories, initial latents and true actions. MSE excludes t=0.',
        model='q_next = A q + B standardized_action + c. Full model: standardized 512-D latent. PCA model: standardized train-PCA coordinates. Controlled affine approximations, no claim of Koopman invariance.')
    (OUT/'metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    np.savetxt(OUT/'test_mse_per_step.csv',
        np.column_stack([np.arange(z.shape[1])]+[v.mean(0) for v in errors.values()]),
        delimiter=',',header='time_step,'+','.join(name+'_mse' for name in errors),comments='')
    styles = {'gru':('GRU','#D55E00'),
              'koopman_full512':('Koopman (512-D)','#009E73'),
              'koopman_pca':('PCA + Koopman','#CC79A7')}
    for i in range(min(4,len(z))):
        fig,axs = plt.subplots(1,2,figsize=(13.33,7.5),constrained_layout=True)
        trajectories = [('Ground truth',z,'#0072B2')]+[
            (styles[name][0],values,styles[name][1]) for name,values in predictions.items()]
        for label,values,col in trajectories:
            points = (values[i]-mean)@display_basis
            axs[0].plot(points[:,0],points[:,1],color=col,label=label,lw=2)
            for k in range(0,len(points)-1,3):
                axs[0].annotate('',xy=points[k+1],xytext=points[k],
                    arrowprops=dict(arrowstyle='->',color=col,lw=1.5))
        axs[0].set(title='Shared train-fitted 2D PCA (display)',xlabel='PC1',ylabel='PC2')
        axs[0].set_aspect('equal',adjustable='box')
        for name,values in errors.items():
            label,col = styles[name]
            axs[1].plot(values[i],color=col,label=label)
        axs[1].set(title='Full 512-D prediction error',xlabel='Time step',ylabel='MSE',ylim=(0,None))
        for ax in axs:
            ax.legend(); ax.grid(alpha=.2)
        fig.suptitle(f'Held-out trajectory {i+1}: GRU vs Koopman vs PCA + Koopman',fontsize=17)
        fig.supxlabel(f'PCA + Koopman rank: {models["koopman_pca"]["selection"]["rank"]} | Arrows: forward time | Display coordinates are not physical position.',fontsize=10)
        for ext in ['png','pdf']:
            fig.savefig(OUT/f'trajectory_{i+1}_comparison.{ext}',dpi=200)
        plt.close(fig)
    fig,axs = plt.subplots(1,2,figsize=(13.33,7.5),constrained_layout=True)
    for name,values in errors.items():
        label,col = styles[name]
        avg = values.mean(0)
        se = values.std(0,ddof=1)/np.sqrt(len(values))
        axs[0].plot(avg,label=label,color=col,lw=2)
        axs[0].fill_between(np.arange(len(avg)),avg-1.96*se,avg+1.96*se,color=col,alpha=.15)
    axs[0].set(title='Held-out mean error (95% normal CI)',xlabel='Time step',ylabel='Full 512-D MSE',ylim=(0,None))
    axs[0].legend()
    axs[1].plot(np.arange(1,513),np.cumsum(variance),color='#0072B2')
    for r in (2,16,32,64):
        axs[1].axvline(r,color='gray',alpha=.3)
        axs[1].annotate(f'{r}: {variance[:r].sum():.1%}',(r,variance[:r].sum()),
            xytext=(8,-16 if r in (2,32) else 8),textcoords='offset points')
    axs[1].set(title='Train PCA cumulative explained variance',xlabel='Number of components',ylabel='Explained variance',xlim=(1,100),ylim=(0,1.03))
    for ax in axs: ax.grid(alpha=.2)
    fig.suptitle('Frozen e-11 encoder: three-method comparison',fontsize=18)
    for ext in ['png','pdf']:
        fig.savefig(OUT/f'evaluation_summary.{ext}',dpi=200)
    plt.close(fig)
    (OUT/'README.md').write_text(
        '# GRU / full-512 Koopman / PCA + Koopman comparison\n\n'
        'Frozen e-11 encoder and original GRU predictor. Shared simulator cache: '+str(cache)+'\n\n'
        'Default splits: 1200 train, 200 validation, 200 test trajectories; 33 frames each. '
        'Existing cache is reused. Separate seeds and whole trajectories; layouts may recur.\n\n'
        'Full Koopman fits all 512 standardized latent coordinates without PCA. '
        'PCA + Koopman selects rank 16/32/64. Both select ridge from 1e-6/1e-4/0.01/0.1/1 '
        'using validation autoregressive full-512 MSE. All fitting uses train only. '
        'Test MSE excludes the shared exact initial frame; all methods use true actions.\n\n'
        '*_model.npz stores A/B/c in standardized coordinates, means/scales, and optional pca_basis. '
        'test_predictions.npz stores ground_truth, gru, koopman_full512, koopman_pca and mse_* arrays. '
        'metrics.json and test_mse_per_step.csv report all three methods. '
        'Plots use shared train-fitted 2D PCA for display only.\n\n'
        'These are controlled affine approximations, not proof of Koopman invariance.\n\n'
        'Run from repository root: `.venv/Scripts/python.exe -m examples.ac_video_jepa.train_koopman`.\n',
        encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    main()
