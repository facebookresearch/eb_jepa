"""Create presentation figures and verify exclusive stage accounting."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'checkpoints/two_method_stage_20_20261014'
data = pd.read_csv(OUT/'planning_per_timestep.csv')
stages = ['encoding','projection','prediction','objective_setup','solver','optimizer_overhead']
assert len(data) == 8000
assert data.groupby('method').size().to_dict() == {'gru_mppi':4000,'pca_lti':4000}
assert not data.duplicated(['method','episode','timestep']).any()
assert np.allclose(data[[s+'_seconds' for s in stages]].sum(axis=1), data.planning_seconds, atol=1e-8)
report = json.loads((OUT/'results.json').read_text())
assert report['status'] == 'complete'
plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'path'})
colors = ['#4C78A8','#9BBFDF','#55A688','#DDB567','#BF7563','#927EAC']
for scope in ['all_steps','horizon_90']:
    subset = data if scope == 'all_steps' else data[data.horizon==90]
    values = subset.groupby('method')[[s+'_seconds' for s in stages]].mean().reindex(['pca_lti','gru_mppi'])*1000
    fig,ax = plt.subplots(figsize=(10,5))
    bottom = np.zeros(2)
    for stage,color in zip(stages,colors):
        v = values[stage+'_seconds'].to_numpy()
        ax.bar(['PCA LTI (64-D)','GRU MPPI'],v,bottom=bottom,label=stage.replace('_',' ').title(),color=color,width=.5)
        bottom += v
    for i,total in enumerate(bottom): ax.text(i,total+5,f'{total:.1f} ms',ha='center',fontweight='bold')
    ax.set(ylabel='Mean time per planning call (ms)',title=f'20 matched tasks · {"H = 90" if scope == "horizon_90" else "All 200 steps per task"}')
    ax.spines[['top','right']].set_visible(False)
    ax.set_ylim(0,bottom.max()*1.17)
    ax.legend(bbox_to_anchor=(1.01,1),loc='upper left',frameon=False)
    fig.tight_layout()
    for ext in ['png','svg','pdf']: fig.savefig(OUT/f'stage_timing_{scope}.{ext}',dpi=220)
    plt.close(fig)
print(values.round(3).to_string())
print('Verified 8,000 unique calls, complete status and stage sums.')
