"""Validate and present the 20-task single-encoding MPPI rerun."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'checkpoints/mppi_single_encode_20_20261014'
ORIGINAL = ROOT/'checkpoints/reduced_mpc_20_20261012'
OLD_TIMING = ROOT/'checkpoints/two_method_stage_20_20261014'
report = json.loads((OUT/'results.json').read_text())
previous = json.loads((ORIGINAL/'results.json').read_text())
data = pd.read_csv(OUT/'planning_per_timestep.csv')
assert report['status'] == 'complete'
assert report['settings']['methods'] == ['gru_mppi']
assert len(data) == 4000 and set(data.method) == {'gru_mppi'}
assert report['seeds'] == list(range(1000,1020))
assert report['tasks'] == previous['tasks']
assert report['checkpoint_sha256'] == previous['checkpoint_sha256']
assert report['mppi_config'] == previous['mppi_config']
assert not data.duplicated(['episode','timestep']).any()
assert set(data.episode) == set(range(20))
assert all(set(group.timestep) == set(range(200)) for _,group in data.groupby('episode'))
assert (data.seed == data.episode+1000).all()
assert (data.encoding_calls == 1).all()
assert (data.horizon == np.minimum(90,200-data.timestep)).all()
stages = ['encoding','projection','prediction','objective_setup','solver','optimizer_overhead']
assert np.allclose(data[[s+'_seconds' for s in stages]].sum(axis=1),data.planning_seconds,atol=1e-8)
assert np.isfinite(data[['planning_seconds']+[s+'_seconds' for s in stages]].to_numpy()).all()

episodes = []
for current in report['episodes']:
    ep = current['episode']
    old = next(p for p in previous['planning'] if p['method']=='gru_mppi' and p['episode']==ep)
    episodes.append(dict(episode=ep, seed=1000+ep,
        old_final_success=old['final_success'],new_final_success=current['final_success'],
        old_final_distance=old['final_distance'],new_final_distance=current['final_distance'],
        final_distance_change=current['final_distance']-old['final_distance']))
pd.DataFrame(episodes).to_csv(OUT/'episode_validation.csv',index=False)
validation = dict(records=4000,episodes=20,seeds=list(range(1000,1020)),
    task_layouts_match=True,checkpoint_match=True,mppi_config_match=True,encoding_calls_per_step=1,
    old_successes=sum(e['old_final_success'] for e in episodes),
    new_successes=sum(e['new_final_success'] for e in episodes),
    new_mean_final_distance=float(np.mean([e['new_final_distance'] for e in episodes])),
    max_abs_final_distance_change=float(max(abs(e['final_distance_change']) for e in episodes)))
(OUT/'validation.json').write_text(json.dumps(validation,indent=2))

old = pd.read_csv(OLD_TIMING/'summary.csv')
new = pd.read_csv(OUT/'summary.csv')
comparison = []
for scope in ['all_steps','horizon_90']:
    a = old[(old.method=='gru_mppi') & (old.scope==scope)].iloc[0]
    b = new[new.scope==scope].iloc[0]
    for field in ['encoding','prediction','objective_setup','optimizer_overhead','planning']:
        key = field+'_mean_ms'
        comparison.append(dict(scope=scope,component=field,old_mean_ms=a[key],new_mean_ms=b[key],
            reduction_percent=100*(a[key]-b[key])/a[key]))
pd.DataFrame(comparison).to_csv(OUT/'old_vs_single_encode.csv',index=False)

pca = old[(old.method=='pca_lti') & (old.scope=='horizon_90')].iloc[0]
mppi = new[new.scope=='horizon_90'].iloc[0]
components = [
    ('Encoding','encoding_mean_ms'),('PCA Projection','projection_mean_ms'),
    ('Dynamics Rollout','prediction_mean_ms'),('Quadratic Construction / Cost Evaluation','objective_setup_mean_ms'),
    ('SLSQP Solve','solver_mean_ms'),('Optimizer Updates & Other Overhead','optimizer_overhead_mean_ms'),
    ('Total Planning','planning_mean_ms')]
table = ['# Stage timing: 20 matched tasks, H = 90','',
    'Mean milliseconds per planning call. PCA-LTI uses the earlier measurement; GRU-MPPI uses the new single-encoding rerun. CUDA stage synchronization and first calls are included.','',
    '| Component | PCA-LTI (64-D) | GRU-MPPI (single encoding) |',
    '|---|---:|---:|']
for name,key in components:
    table.append(f'| {name} | {pca[key]:.2f} | {mppi[key]:.2f} |')
table += ['',
    'PCA-LTI overhead includes finite differences, line search and bookkeeping. MPPI overhead includes sampling, elite selection, distribution updates and bookkeeping. Goal encoding, model loading and environment execution are excluded.',
    '',f'Validation: {validation["new_successes"]}/20 successful tasks; seeds 1000–1019 and all task layouts match the original evaluation.']
(OUT/'comparison_table.md').write_text('\n'.join(table),encoding='utf-8')

plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'path'})
colors = ['#4C78A8','#9BBFDF','#55A688','#DDB567','#BF7563','#927EAC']
fig,ax = plt.subplots(figsize=(10,5))
bottom = np.zeros(2)
for (name,key),color in zip(components[:-1],colors):
    values = np.array([pca[key],mppi[key]])
    ax.bar(['PCA LTI (64-D)','GRU MPPI\nSingle encoding'],values,bottom=bottom,label=name,color=color,width=.5)
    bottom += values
for i,total in enumerate(bottom): ax.text(i,total+4,f'{total:.1f} ms',ha='center',fontweight='bold')
ax.set(ylabel='Mean time per planning call (ms)',title='20 matched tasks · H = 90',ylim=(0,bottom.max()*1.17))
ax.spines[['top','right']].set_visible(False)
ax.legend(bbox_to_anchor=(1.01,1),loc='upper left',frameon=False,fontsize=9)
fig.tight_layout()
for ext in ['png','svg','pdf']: fig.savefig(OUT/f'stage_timing_horizon_90.{ext}',dpi=220)
plt.close(fig)
print(json.dumps(validation,indent=2))
print(pd.DataFrame(comparison).to_string(index=False))
