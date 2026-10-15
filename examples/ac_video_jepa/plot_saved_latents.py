"""Plot cached JEPA embeddings without replaying evaluation."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT = Path(__file__).resolve().parents[2] / 'checkpoints/ac_video_jepa_gpu/latent_pca'
BLUE, ORANGE = '#0072B2', '#D55E00'


def arrows(ax, points, color, label):
    ax.plot(points[:, 0], points[:, 1], color=color, linewidth=2, label=label)
    # Each arrow connects consecutive time steps; zero-length steps are skipped.
    for start, end in zip(points[:-1], points[1:]):
        if np.linalg.norm(end - start) > 1e-8:
            ax.annotate('', xy=end, xytext=start,
                        arrowprops=dict(arrowstyle='->', color=color, lw=1.5,
                                        mutation_scale=12, shrinkA=0, shrinkB=0))


def pca_panel(ax, z, p, explained, title, zoom=True):
    arrows(ax, z, BLUE, 'Ground truth')
    arrows(ax, p, ORANGE, 'Predicted')
    ax.annotate('Start: t=0', z[0], xytext=(8, 12), textcoords='offset points', fontsize=9)
    ax.annotate(f'GT end: t={len(z)-1}', z[-1], xytext=(8, 14),
                textcoords='offset points', color=BLUE, fontsize=9)
    ax.annotate(f'Predicted end: t={len(p)-1}', p[-1], xytext=(8, -20),
                textcoords='offset points', color=ORANGE, fontsize=9)
    ax.set(title=title, xlabel=f'PC1 ({explained[0]:.1%} variance)',
           ylabel=f'PC2 ({explained[1]:.1%} variance)')
    ax.grid(alpha=.2)
    ax.legend(loc='best', fontsize=10)
    # Identical unit scale on both axes, with enough space for endpoint labels.
    lo, hi = np.minimum(z.min(0), p.min(0)), np.maximum(z.max(0), p.max(0))
    span = max((hi-lo).max(), .5) * 1.55
    mid = (hi+lo)/2
    ax.set_xlim(mid[0]-span/2, mid[0]+span/2)
    ax.set_ylim(mid[1]-span/2, mid[1]+span/2)
    ax.set_aspect('equal', adjustable='box')


def mse_panel(ax, mse, title):
    ax.plot(np.arange(len(mse)), mse, color=ORANGE, linewidth=2.5)
    ax.set(title=title, xlabel='Time step', ylabel='MSE across all 512 dimensions')
    ax.set_xlim(0, len(mse)-1)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=.2)
    ax.text(.04, .94, f'Mean MSE: {mse.mean():.4f}\nLower is better',
            transform=ax.transAxes, va='top', fontsize=10)


def render():
    with np.load(OUT / 'latents.npz') as data:
        true, pred = data['true'], data['predicted']
        z, p = data['true_pca'], data['predicted_pca']
    info = json.loads((OUT / 'metadata.json').read_text(encoding='utf-8'))
    explained = info['explained_variance_ratio']
    assert true.shape == pred.shape and true.shape[-1] == 512
    mse = ((true-pred)**2).mean(-1)
    assert np.isfinite(mse).all()
    plt.rcParams.update({'font.size': 11, 'axes.titlesize': 13})
    fig, axes = plt.subplots(len(z), 2, figsize=(13, 4*len(z)), constrained_layout=True)
    for i, row in enumerate(axes):
        pca_panel(row[0], z[i], p[i], explained, f'Trajectory {i+1}: 2D PCA')
        mse_panel(row[1], mse[i], f'Trajectory {i+1}: full 512-D prediction error')
        slide, sa = plt.subplots(1, 2, figsize=(13.33, 7.5), constrained_layout=True)
        pca_panel(sa[0], z[i], p[i], explained, '2D PCA trajectory')
        mse_panel(sa[1], mse[i], 'Full 512-D prediction error')
        slide.suptitle(f'JEPA latent prediction - Trajectory {i+1}', fontsize=20)
        slide.supxlabel(f'Blue: Ground truth | Orange: Predicted | Arrows: forward in time\n'
                        f'PCA retains {sum(explained):.1%} of sample variance; PC axes are not physical x/y.', fontsize=11)
        for ext in ('png', 'pdf'):
            slide.savefig(OUT / f'trajectory_{i+1}_pca_mse.{ext}', dpi=220)
        plt.close(slide)
    fig.suptitle('JEPA latents: 2D PCA trajectories and full 512-D MSE\n'
                 'Blue: Ground truth | Orange: Predicted | Arrows: forward in time', fontsize=17)
    for ext in ('png', 'pdf'):
        fig.savefig(OUT / f'trajectory_pca.{ext}', dpi=200)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13.33, 7.5), constrained_layout=True)
    for i in range(len(z)):
        arrows(axes[0], z[i], BLUE, 'Ground truth' if i==0 else '_nolegend_')
        arrows(axes[0], p[i], ORANGE, 'Predicted' if i==0 else '_nolegend_')
        axes[0].annotate(f'Trajectory {i+1}', z[i,0], xytext=(8,8), textcoords='offset points')
        axes[1].plot(np.arange(mse.shape[1]), mse[i], label=f'Trajectory {i+1}', linewidth=2)
    axes[0].set(title='All trajectories in shared PCA space', xlabel=f'PC1 ({explained[0]:.1%})', ylabel=f'PC2 ({explained[1]:.1%})')
    axes[0].set_aspect('equal', adjustable='box')
    axes[0].margins(.2)
    axes[1].set(title='Prediction error in the full 512-D space', xlabel='Time step', ylabel='MSE across all 512 dimensions', ylim=(0, None))
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend(fontsize=10)
    fig.suptitle('JEPA latent prediction: overview', fontsize=20)
    fig.supxlabel(f'Arrows show forward time. MSE measures Ground truth vs Predicted.\n'
                  f'Two PCA components retain {sum(explained):.1%} of sample variance.', fontsize=11)
    for ext in ('png', 'pdf'):
        fig.savefig(OUT / f'overview_pca.{ext}', dpi=220)
    plt.close(fig)
    np.savetxt(OUT / 'mse_per_step.csv', np.column_stack([np.arange(mse.shape[1]), mse.T]),
               delimiter=',', header='time_step,'+','.join(f'trajectory_{i+1}' for i in range(len(z))), comments='')
    print(f'Updated overview and trajectory plots; created {len(z)} widescreen slide figures in {OUT}')


if __name__ == '__main__':
    render()
