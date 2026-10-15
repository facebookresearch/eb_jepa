from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parent
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'], 'svg.fonttype': 'path'})
fig, ax = plt.subplots(figsize=(16, 9), dpi=180)
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
ax.set(xlim=(0,16), ylim=(0,9))
ax.axis('off')
fig.patch.set_facecolor('white')
ink = '#182D46'

def label(x,y,s,size=14,color=ink,weight='normal',ha='center'):
    ax.text(x,y,s,fontsize=size,color=color,fontweight=weight,ha=ha,va='center',linespacing=1.65)

def box(x,y,w,h,title,body,fill='#EFF5FC',edge='#C6D8EC',title_size=15,body_size=12):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.02,rounding_size=0.16',facecolor=fill,edgecolor=edge,linewidth=1.4))
    label(x+w/2,y+h-.38,title,title_size,weight='bold')
    label(x+w/2,y+(h-.55)/2,body,body_size)

def arrow(start,end,color='#6E8298',style='-',rad=0):
    ax.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=17,linewidth=1.7,color=color,linestyle=style,connectionstyle=f'arc3,rad={rad}'))

label(.65,8.42,'PCA + Linear MPC',29,weight='bold',ha='left')
label(.65,7.27,'01  Offline model fitting',17,color='#2C69A1',weight='bold',ha='left')

box(.65,4.91,3.05,1.9,'Frozen Encoder','Training data $(o_t, u_t, o_{t+1})$\n$z_t = E(o_t)$: 512-D latent')
box(4.08,4.91,3.1,1.9,'Fit PCA (train only)','Compare ranks: 16, 32, 64\nFit $P$, center $\\mu$, scale $D$\n$x_t = D^{-1}P^T(z_t-\\mu)$',body_size=12)
box(7.56,4.91,3.12,1.9,'Fit Ridge Regression','Inputs: $x_t$, $u_t$\nTarget: $x_{t+1}$')
box(11.06,4.91,4.22,1.9,'Selected Model: 64-D','$x_{t+1} = A x_t + B u_t + c$\n$A: 64\\times64$     $B: 64\\times2$',fill='#E8F5F1',edge='#91C8B6',title_size=16,body_size=16)
for a,b in [(3.7,4.08),(7.18,7.56),(10.68,11.06)]: arrow((a,5.86),(b,5.86))
box(4.08,3.63,6.6,.93,'15 rank / ridge candidates → Selected rank: 64','Ranks: 16 / 32 / 64; ridge: 1e-6 / 1e-4 / 0.01 / 0.1 / 1\nSelect by validation multi-step latent rollout MSE',fill='#F6F7FA',edge='#D8DFE8',title_size=13,body_size=10)
arrow((7.38,4.91),(7.38,4.56),style='--')
arrow((10.68,4.12),(12.1,4.91),style='--')

label(.65,3.13,'02  Linear MPC',17,color='#28775E',weight='bold',ha='left')
box(.65,1.03,3.05,1.63,'Encode & Project','Frozen Encoder + selected PCA\n512-D latent → 64-D state',fill='#E8F5F1',edge='#B4D9CB',body_size=12)
box(4.08,1.03,3.1,1.63,'Predict with A, B, c','Candidate actions $u_{t:t+H-1}$\nRoll out $x_{t+1:t+H}$',fill='#E8F5F1',edge='#B4D9CB',body_size=12)
box(7.56,1.03,3.12,1.63,'Optimize Actions','Track goal: $\\hat z=\\mu+PDx$\nAction penalty + norm bounds',fill='#E8F5F1',edge='#B4D9CB',body_size=12)
box(11.06,1.03,4.22,1.63,'Execute First Action','Observe environment response\nUpdate state and replan',fill='#E8F5F1',edge='#91C8B6')
for a,b in [(3.7,4.08),(7.18,7.56),(10.68,11.06)]: arrow((a,1.84),(b,1.84),color='#438A73')
ax.plot([14.8,15.65,15.65,5.63,5.63],[4.91,4.91,2.86,2.86,2.72],color='#438A73',linewidth=1.5,linestyle='--')
arrow((5.63,2.72),(5.63,2.66),color='#438A73')
ax.plot([13.17,13.17,2.18,2.18],[1.03,.57,.57,.88],color='#438A73',linewidth=1.7)
arrow((2.18,.88),(2.18,1.03),color='#438A73')
label(7.67,.30,'Observation feedback · Execute one step, then replan',12,color='#28775E')
for ext in ['svg','pdf','png']:
    fig.savefig(OUT/f'pca_linear_mpc_flowchart.{ext}',dpi=240,facecolor='white')
plt.close(fig)
print(OUT)

