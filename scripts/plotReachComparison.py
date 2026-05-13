"""
Compare Mamba vs LSTM reachability results from data/results/*.npz files.

Generates three plots:
  1. Full-state KL divergence over time (both models on one axes)
  2. Marginal CDF of final-state distance from centroid (full state)
  3. Pairplot of final-state distributions (True / Mamba / LSTM)

Usage:
  python scripts/plotReachComparison.py \
      --mamba data/results/3bp_mamba_orbit_2.1_retrograde_geo_to_moon_trainRatio_0.8_epoch_10_lr_0.01_train_timesteps_80.npz \
      --lstm  data/results/3bp_lstm_orbit_2.1_retrograde_geo_to_moon_trainRatio_0.8_epoch_10_lr_0.01_train_timesteps_80.npz
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns

parser = argparse.ArgumentParser(description='Compare Mamba and LSTM reachability results')
parser.add_argument('--mamba',   type=str, required=True, help='Path to mamba results .npz')
parser.add_argument('--lstm',    type=str, required=True, help='Path to lstm results .npz')
parser.add_argument('--pdf',     action='store_true',     help='Save as PDF instead of PNG')
parser.add_argument('--out-dir', type=str, default='plots', help='Output directory for plots')
args = parser.parse_args()

save_ext = 'pdf' if args.pdf else 'png'
os.makedirs(args.out_dir, exist_ok=True)

# ──────────────────────────────────────────────────────────
# Load
# ──────────────────────────────────────────────────────────
mamba_d = np.load(args.mamba, allow_pickle=True)
lstm_d  = np.load(args.lstm,  allow_pickle=True)

true_reach_m = mamba_d['true_reach']   # (T, N, D)
pred_reach_m = mamba_d['pred_reach']
final_true_m = mamba_d['final_true']   # (N, D)
final_pred_m = mamba_d['final_pred']
train_ts     = int(mamba_d['train_timesteps'])

pred_reach_l = lstm_d['pred_reach']
final_true_l = lstm_d['final_true']
final_pred_l = lstm_d['final_pred']

# ──────────────────────────────────────────────────────────
# Dimensionality
# ──────────────────────────────────────────────────────────
D     = final_true_m.shape[-1]
n_pos = D // 2

if D == 4:
    pos_lbl = ['X (km)', 'Y (km)']
    vel_lbl = ['Vx (km/s)', 'Vy (km/s)']
elif D == 6:
    pos_lbl = ['X (km)', 'Y (km)', 'Z (km)']
    vel_lbl = ['Vx (km/s)', 'Vy (km/s)', 'Vz (km/s)']
else:
    pos_lbl = [f'x{i}' for i in range(n_pos)]
    vel_lbl = [f'v{i}' for i in range(D - n_pos)]
state_labels = pos_lbl + vel_lbl

# ──────────────────────────────────────────────────────────
# Output prefix — derived from mamba filename
# ──────────────────────────────────────────────────────────
mamba_stem = os.path.splitext(os.path.basename(args.mamba))[0]
_pfx = os.path.join(args.out_dir, 'comparison_' + mamba_stem.replace('_mamba', ''))

# ──────────────────────────────────────────────────────────
# Style
# ──────────────────────────────────────────────────────────
sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams.update({
    'font.size':        16,
    'axes.titlesize':   18,
    'axes.labelsize':   16,
    'xtick.labelsize':  14,
    'ytick.labelsize':  14,
    'legend.fontsize':  14,
    'figure.titlesize': 18,
})
_palette = {'True': 'steelblue', 'Mamba': 'tomato', 'LSTM': 'seagreen'}

# ──────────────────────────────────────────────────────────
# 1. Full-state KL divergence over time
# ──────────────────────────────────────────────────────────
def _load_kl_full(d):
    for key in ('kl_4d', 'kl_6d'):
        if key in d.files:
            return d[key]
    raise KeyError(f"No full-state KL key found. Available keys: {d.files}")

kl_m = _load_kl_full(mamba_d)
kl_l = _load_kl_full(lstm_d)

n_frames = min(len(kl_m), len(kl_l))

if D == 4: # 4d == cr3bp
    # 2 hr observations
    obs_time = 2
    final_time = n_frames * obs_time 
    time_label = 'Time (hours)'
if D == 6: #6d == 2bp
    # 1 minute observations
    obs_time = 1
    final_time = n_frames * obs_time
    time_label = 'Time (minutes)'

t_axis   = np.arange(0, final_time, obs_time)

fig_kl, ax_kl = plt.subplots(figsize=(10, 5))
ax_kl.plot(t_axis, kl_m[:n_frames], color='tomato',   linewidth=1.5,
           label=f'Mamba (final = {kl_m[n_frames - 1]:.4f})')
ax_kl.plot(t_axis, kl_l[:n_frames], color='seagreen', linewidth=1.5,
           label=f'LSTM  (final = {kl_l[n_frames - 1]:.4f})')
ax_kl.axvline(x=train_ts, color='gray', linestyle='--', linewidth=1, label='Train/Test boundary')
ax_kl.set_xlabel(time_label)
ax_kl.set_ylabel(f'KL Divergence  D(true ‖ pred)  [{D}D]')
ax_kl.set_title(f'{D}D Full-State KL Divergence Over Time: Mamba vs LSTM')
ax_kl.legend()
plt.tight_layout()
plt.savefig(_pfx + f'_kl_full.{save_ext}')
plt.close(fig_kl)
print(f"Saved: {_pfx}_kl_full.{save_ext}")

# ──────────────────────────────────────────────────────────
# 2. Marginal CDF — final-state distance from true centroid
# ──────────────────────────────────────────────────────────
final_true = final_true_m  # reference distribution (same dataset split)

centroid = final_true.mean(axis=0)

dist_true  = np.linalg.norm(final_true   - centroid, axis=1)
dist_mamba = np.linalg.norm(final_pred_m - centroid, axis=1)
dist_lstm  = np.linalg.norm(final_pred_l - centroid, axis=1)

_df_cdf = pd.DataFrame({
    'Distance from centroid': np.concatenate([dist_true, dist_mamba, dist_lstm]),
    'Distribution': (['True']  * len(dist_true) +
                     ['Mamba'] * len(dist_mamba) +
                     ['LSTM']  * len(dist_lstm)),
})

fig_cdf, ax_cdf = plt.subplots(figsize=(8, 5))
sns.ecdfplot(data=_df_cdf, x='Distance from centroid', hue='Distribution',
             ax=ax_cdf, palette=_palette)
ax_cdf.set_ylabel('Cumulative Probability')
ax_cdf.set_title('Marginal CDF — Final State: Mamba vs LSTM vs True')
plt.tight_layout()
plt.savefig(_pfx + f'_marginal_cdf.{save_ext}')
plt.close(fig_cdf)
print(f"Saved: {_pfx}_marginal_cdf.{save_ext}")

# ──────────────────────────────────────────────────────────
# 3. Pairplot — True / Mamba / LSTM final-state distributions
# ──────────────────────────────────────────────────────────
_df_true  = pd.DataFrame(final_true,   columns=state_labels)
_df_true['Model'] = 'True'
_df_mamba = pd.DataFrame(final_pred_m, columns=state_labels)
_df_mamba['Model'] = 'Mamba'
_df_lstm  = pd.DataFrame(final_pred_l, columns=state_labels)
_df_lstm['Model'] = 'LSTM'
_df_pair  = pd.concat([_df_true, _df_mamba, _df_lstm], ignore_index=True)

g = sns.pairplot(
    _df_pair,
    hue='Model',
    plot_kws={'alpha': 0.25, 's': 6, 'rasterized': True},
    diag_kws={'rasterized': True},
    diag_kind='kde',
    palette={'True': 'steelblue', 'Mamba': 'tomato', 'LSTM': 'seagreen'},
)
g.figure.suptitle('Final State Pairplot: True vs Mamba vs LSTM', y=1.01)
_legend_handles = [
    mlines.Line2D([], [], marker='o', color='w', markerfacecolor='steelblue', markersize=10, label='True'),
    mlines.Line2D([], [], marker='o', color='w', markerfacecolor='tomato',    markersize=10, label='Mamba'),
    mlines.Line2D([], [], marker='o', color='w', markerfacecolor='seagreen',  markersize=10, label='LSTM'),
]
g.legend.remove()
g.figure.legend(handles=_legend_handles, title='Model', loc='upper right',
                frameon=True, fontsize=14, title_fontsize=14)
g.savefig(_pfx + f'_pairplot.{save_ext}', bbox_inches='tight')
plt.close(g.figure)
print(f"Saved: {_pfx}_pairplot.{save_ext}")

print("Done.")
