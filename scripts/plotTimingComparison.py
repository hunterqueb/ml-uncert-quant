"""
Compare Mamba vs LSTM wall-clock train/test times from data/results/*_times_*.csv files.

Prints summary statistics (mean, std, median, IQR, speedup, Mann-Whitney U test) and
generates two box plots, each with the individual runs overlaid:
  1. <stem>_boxplot         — train / test panels, Mamba vs LSTM on a shared axis
  2. <stem>_boxplot_spread  — 2x2 grid, each model/phase on its own linear axis, so
                              run-to-run variability is readable despite the ~10x gap

Usage:
  python scripts/plotTimingComparison.py \
      --mamba data/results/2bp_times_mamba_heo.csv \
      --lstm  data/results/2bp_times_lstm_heo.csv
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.stats import mannwhitneyu

parser = argparse.ArgumentParser(description='Compare Mamba and LSTM train/test times')
parser.add_argument('--mamba',   type=str, default='data/results/2bp_times_mamba_heo.csv',
                    help='Path to mamba timing .csv')
parser.add_argument('--lstm',    type=str, default='data/results/2bp_times_lstm_heo.csv',
                    help='Path to lstm timing .csv')
parser.add_argument('--png',     action='store_true',     help='Save as PNG instead of PDF')
parser.add_argument('--out-dir', type=str, default='plots', help='Output directory for plots')
parser.add_argument('--linear',  action='store_true',
                    help='Use a linear y-axis (default is log, since the models differ by ~10x)')
args = parser.parse_args()

save_ext = 'png' if args.png else 'pdf'
os.makedirs(args.out_dir, exist_ok=True)

# ──────────────────────────────────────────────────────────
# Load
# ──────────────────────────────────────────────────────────
df_m = pd.read_csv(args.mamba)
df_l = pd.read_csv(args.lstm)
df = pd.concat([df_m, df_l], ignore_index=True)

_palette = {'mamba': 'tomato', 'lstm': 'seagreen'}
_labels  = {'mamba': 'Mamba',  'lstm': 'LSTM'}
_devices = {'hunter-jetson': 'Jetson Nano 2GB'}   # hostname -> display name
models = ['mamba', 'lstm']

# Flag any run-config differences between the two files (other than model/timing)
_cfg_cols = [c for c in df.columns if c not in ('timestamp', 'model', 'train_s', 'test_s')]
_diffs = {c: {m: sorted(df.loc[df.model == m, c].astype(str).unique()) for m in models}
          for c in _cfg_cols
          if df.loc[df.model == 'mamba', c].astype(str).nunique() > 1
          or df.loc[df.model == 'lstm',  c].astype(str).nunique() > 1
          or set(df.loc[df.model == 'mamba', c].astype(str)) != set(df.loc[df.model == 'lstm', c].astype(str))}
if _diffs:
    print('Config differences between runs:')
    for c, v in _diffs.items():
        print(f'  {c:>16s}: ' + ', '.join(f'{_labels[m]}={v[m]}' for m in models))
    print()

# ──────────────────────────────────────────────────────────
# Summary statistics
# ──────────────────────────────────────────────────────────
phases = [('train_s', 'Train'), ('test_s', 'Test')]

def _summary(x):
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    return {'n': len(x), 'mean': x.mean(), 'std': x.std(ddof=1), 'median': med,
            'IQR': q3 - q1, 'min': x.min(), 'max': x.max(), 'CV %': 100 * x.std(ddof=1) / x.mean()}

rows = []
for col, name in phases:
    for m in models:
        rows.append({'phase': name, 'model': _labels[m],
                     **_summary(df.loc[df.model == m, col].to_numpy())})
stats = pd.DataFrame(rows).set_index(['phase', 'model'])
with pd.option_context('display.float_format', '{:,.2f}'.format):
    print('Wall-clock time [s]:')
    print(stats)
print()

speedups = {}
for col, name in phases:
    xm = df.loc[df.model == 'mamba', col].to_numpy()
    xl = df.loc[df.model == 'lstm',  col].to_numpy()
    speedups[col] = np.median(xl) / np.median(xm)
    p = mannwhitneyu(xm, xl, alternative='two-sided').pvalue
    print(f'{name:>5s}: LSTM/Mamba median ratio = {speedups[col]:.2f}x   '
          f'(mean ratio {xl.mean() / xm.mean():.2f}x, Mann-Whitney p = {p:.2e})')

# ──────────────────────────────────────────────────────────
# Box plot
# ──────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.size': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
rng = np.random.default_rng(0)

for ax, (col, name) in zip(axes, phases):
    data = [df.loc[df.model == m, col].to_numpy() for m in models]
    bp = ax.boxplot(data, positions=range(len(models)), widths=0.5, patch_artist=True,
                    showfliers=False, medianprops=dict(color='k', linewidth=1.5),
                    whiskerprops=dict(color='dimgray'), capprops=dict(color='dimgray'))
    for patch, m in zip(bp['boxes'], models):
        patch.set_facecolor(_palette[m])
        patch.set_alpha(0.35)
        patch.set_edgecolor(_palette[m])
    # Overlay individual runs (jittered) so outliers and n are visible
    for i, (x, m) in enumerate(zip(data, models)):
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(x)), x, s=18, color=_palette[m],
                   edgecolor='white', linewidth=0.5, zorder=3)
        ax.annotate(f'{np.median(x):,.0f} s', xy=(i + 0.3, np.median(x)),
                    va='center', ha='left', fontsize=9, color='dimgray')

    ax.set_xticks(range(len(models)), [_labels[m] for m in models])
    ax.set_xlim(-0.6, len(models) - 0.2)
    ax.set_title(f'{name} time  (LSTM {speedups[col]:.1f}× slower)')
    ax.set_ylabel('Wall-clock time [s]')
    if not args.linear:
        ax.set_yscale('log')
        ax.yaxis.set_major_locator(mticker.LogLocator(subs=(1, 2, 5)))
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:,.0f}'))
    ax.grid(axis='y', which='both', alpha=0.3)

_r = df.iloc[0]
fig.suptitle(f"{_r['orbit'].upper()} 2BP — {len(df_m)} Mamba / {len(df_l)} LSTM runs on {_devices.get(_r['host'], _r['host'])}",
             fontsize=12)
fig.tight_layout()

_stem = os.path.splitext(os.path.basename(args.mamba))[0].replace('_mamba', '')
out_path = os.path.join(args.out_dir, f'{_stem}_boxplot.{save_ext}')
fig.savefig(out_path, bbox_inches='tight')
print(f'\nSaved {out_path}')

# ──────────────────────────────────────────────────────────
# Spread plot — independent linear axes per model/phase
# ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(8, 6))
for r, (col, name) in enumerate(phases):
    for c, m in enumerate(models):
        ax = axes[r, c]
        x = df.loc[df.model == m, col].to_numpy()
        bp = ax.boxplot([x], positions=[0], widths=0.5, patch_artist=True, showfliers=False,
                        medianprops=dict(color='k', linewidth=1.5),
                        whiskerprops=dict(color='dimgray'), capprops=dict(color='dimgray'))
        bp['boxes'][0].set_facecolor(_palette[m])
        bp['boxes'][0].set_alpha(0.35)
        bp['boxes'][0].set_edgecolor(_palette[m])
        ax.scatter(rng.uniform(-0.12, 0.12, len(x)), x, s=18, color=_palette[m],
                   edgecolor='white', linewidth=0.5, zorder=3)
        st = stats.loc[(name, _labels[m])]
        ax.set_title(f"{_labels[m]} — {name}\nmedian {st['median']:,.0f} s",
                     fontsize=10)
        ax.set_xticks([])
        ax.set_xlim(-0.6, 0.6)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:,.0f}'))
        ax.grid(axis='y', alpha=0.3)
        if c == 0:
            ax.set_ylabel(f'{name} time [s]')
fig.suptitle('Run-to-run variability (independent y-axes)', fontsize=12)
fig.tight_layout()
out_path = os.path.join(args.out_dir, f'{_stem}_boxplot_spread.{save_ext}')
fig.savefig(out_path, bbox_inches='tight')
print(f'Saved {out_path}')
