"""
Compare Mamba vs LSTM reachability results from data/results/*.npz files.

Generates three plots:
  1. Full-state KL divergence over time (both models on one axes)
  2. Marginal CDF of final-state distance from centroid (full state)
  3. Pairplot of final-state distributions (True / Mamba / LSTM)

With --pdf-evolution, also generates two animations of the position PDF over time,
with True / Mamba / LSTM co-plotted in every frame:
  4. Single 3D axes — sigma ellipsoids (6D) or density surfaces (4D)
  5. Projection panels — KDE contours on each coordinate pair

Usage:
  python scripts/plotReachComparison.py \
      --mamba data/results/3bp_mamba_orbit_2.1_retrograde_geo_to_moon_trainRatio_0.8_epoch_10_lr_0.01_train_timesteps_80.npz \
      --lstm  data/results/3bp_lstm_orbit_2.1_retrograde_geo_to_moon_trainRatio_0.8_epoch_10_lr_0.01_train_timesteps_80.npz
"""
import argparse
import glob
import os

import numpy as np
import yaml
from scipy.integrate import solve_ivp
from qutils.orbital import (dim2NonDim6, nonDim2Dim4,
                            MU_EARTH_JGM2, RE_EARTH_JGM2, J2_JGM2)
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from scipy.stats import gaussian_kde

parser = argparse.ArgumentParser(description='Compare Mamba and LSTM reachability results')
parser.add_argument('--mamba',   type=str, required=True, help='Path to mamba results .npz')
parser.add_argument('--lstm',    type=str, required=True, help='Path to lstm results .npz')
parser.add_argument('--pdf',     action='store_true',     help='Save as PDF instead of PNG')
parser.add_argument('--out-dir', type=str, default='plots', help='Output directory for plots')
parser.add_argument('--pdf-evolution', action='store_true',
                    help='Also render the position-PDF time-evolution animations (slow)')
parser.add_argument('--frame-stride',  type=int, default=1,
                    help='Render every Nth frame of the evolution animations')
parser.add_argument('--fps',           type=int, default=20, help='Animation frame rate')
parser.add_argument('--sigma-levels',  type=int, default=3,
                    help='Number of nested contour levels in the projection figure; also '
                         'selects which sigma shell the 3D view draws (6D only)')
parser.add_argument('--camera', choices=('origin', 'track', 'global'), default='origin',
                    help="Evolution-animation view. 'origin': fixed camera centred on (0,0,0) "
                         "with equal spans, so motion is seen against the central body. "
                         "'track': constant-size window following the ensemble (largest PDF "
                         "detail). 'global': static limits spanning all data.")
parser.add_argument('--elev', type=float, default=30.0, help='3D camera elevation (degrees)')
parser.add_argument('--azim', type=float, default=-60.0, help='3D camera azimuth (degrees)')
parser.add_argument('--max-kde-samples', type=int, default=4000,
                    help='Cap on trajectories used per KDE fit (0 = use all). KDE cost is linear '
                         'in sample count while the estimate saturates well before this')
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

# ──────────────────────────────────────────────────────────
# 4/5. Position PDF time evolution — True / Mamba / LSTM together
#
# Two animations, both opt-in via --pdf-evolution (each takes minutes):
#   4. one 3D axes  — sigma ellipsoids (D=6) or density surfaces (D=4)
#   5. projection panels — KDE contours on each coordinate pair
#
# Animations are always written as mp4 (gif fallback) regardless of --pdf,
# since --pdf only selects the still-image format.
# ──────────────────────────────────────────────────────────
if args.pdf_evolution:

    n_frames_ev = min(true_reach_m.shape[0], pred_reach_m.shape[0], pred_reach_l.shape[0])
    frame_idx   = list(range(0, n_frames_ev, max(1, args.frame_stride)))

    # position-only views of the three reachability tubes: (T, N, n_pos)
    pos_true  = true_reach_m[:n_frames_ev, :, :n_pos]
    pos_mamba = pred_reach_m[:n_frames_ev, :, :n_pos]
    pos_lstm  = pred_reach_l[:n_frames_ev, :, :n_pos]
    _series   = [('True', pos_true), ('Mamba', pos_mamba), ('LSTM', pos_lstm)]

    def _nominal_traj(T):
        """Nominal (unperturbed) trajectory, (T, D), in the same frame/units as true_reach.

        2BP: config.yaml elements (periapsis alt = midpoint of lowerAlt/upperAlt), propagated
        with J2, which tracks the GMAT data to ~30 km over a run. GMAT's first saved row is
        epoch + 1 min, hence the grid starting at 60 s.
        CR3BP: the dataset stores the nominal IC (IC_GEO) under 'mu'; propagate it on the saved grid.
        """
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
        orbit = str(mamba_d['orbit'])
        dimensional = bool(mamba_d['dimensional'])
        if D == 4:
            ds = np.load(sorted(glob.glob(os.path.join(data_dir, 'cr3bp', orbit + '_*.npy')))[0])
            m = 7.348E22 / (5.974E24 + 7.348E22)

            def rhs(_, y):
                x, yy, vx, vy = y
                r1 = np.hypot(x + m, yy) ** 3
                r2 = np.hypot(x - 1 + m, yy) ** 3
                return [vx, vy,
                        2 * vy + x - (1 - m) * (x + m) / r1 - m * (x - 1 + m) / r2,
                        -2 * vx + yy - (1 - m) * yy / r1 - m * yy / r2]

            t = ds['t'][0].ravel()[:T]
            nom = solve_ivp(rhs, (t[0], t[-1]), ds['mu'], t_eval=t,
                            method='DOP853', rtol=1e-12, atol=1e-12).y.T
            return nonDim2Dim4(nom, 389703, 382981) if dimensional else nom

        cfg = glob.glob(os.path.join(data_dir, 'gmat', orbit,
                                     f"{int(mamba_d['prop_min'])}min-*", 'config.yaml'))[0]
        with open(cfg) as fh:
            c = yaml.safe_load(fh)
        # without eccentricity_spread the generator samples e ~ U(0, eccentricity)
        mu = MU_EARTH_JGM2
        e = c['eccentricity'] if 'eccentricity_spread' in c else 0.5 * c['eccentricity']
        i, raan, argp, nu = np.radians([c['inclination'], c['RAAN'],
                                        c['argPeriapsis'], c['trueAnomaly']])
        p = (RE_EARTH_JGM2 + 0.5 * (c['lowerAlt'] + c['upperAlt'])) * (1 + e)
        r_pf = p / (1 + e * np.cos(nu)) * np.array([np.cos(nu), np.sin(nu), 0.0])
        v_pf = np.sqrt(mu / p) * np.array([-np.sin(nu), e + np.cos(nu), 0.0])
        cO, sO, ci, si, cw, sw = (np.cos(raan), np.sin(raan), np.cos(i), np.sin(i),
                                  np.cos(argp), np.sin(argp))
        R = np.array([[cO * cw - sO * sw * ci, -cO * sw - sO * cw * ci,  sO * si],
                      [sO * cw + cO * sw * ci, -sO * sw + cO * cw * ci, -cO * si],
                      [sw * si,                 cw * si,                 ci]])

        def rhs(_, y):
            r = y[:3]
            rn = np.linalg.norm(r)
            z2 = (r[2] / rn) ** 2
            j2 = -1.5 * J2_JGM2 * mu * RE_EARTH_JGM2 ** 2 / rn ** 5 * r * np.array(
                [1 - 5 * z2, 1 - 5 * z2, 3 - 5 * z2])
            return np.r_[y[3:], -mu * r / rn ** 3 + j2]

        t = (np.arange(T) + 1) * 60.0
        nom = solve_ivp(rhs, (0.0, t[-1]), np.r_[R @ r_pf, R @ v_pf], t_eval=t,
                        method='DOP853', rtol=1e-11, atol=1e-11).y.T
        return nom if dimensional else dim2NonDim6(nom)

    # nominal path from the dataset's generating config, used as the motion trail in both figures
    nom_traj = _nominal_traj(n_frames_ev)[:, :n_pos]       # (T, n_pos)

    time_unit = 'hr' if D == 4 else 'min'

    def _axis_lims(arrays):
        """Shared min/max with 5% padding across every cloud and every frame."""
        combined = np.concatenate([a.reshape(-1, a.shape[-1]) for a in arrays], axis=0)
        mn, mx = combined.min(axis=0), combined.max(axis=0)
        pad = 0.05 * np.maximum(mx - mn, 1e-9)
        return mn - pad, mx + pad

    pos_lo, pos_hi = _axis_lims([pos_true, pos_mamba, pos_lstm])

    # 'origin': one fixed cube centred on (0,0,0), equal span on every axis, so the
    #   ensemble is seen moving against the central body. Scale is constant and the
    #   camera never moves; the per-frame cloud is small within it.
    # 'track': constant-size window following the ensemble — same fixed scale, but
    #   centred on the cloud, which is the only view where PDF shape is legible when
    #   the cloud is orders of magnitude smaller than the swept domain.
    # 'global': static limits spanning all data (not centred).
    _R = max(float(np.abs(np.concatenate([pos_lo, pos_hi])).max()), 1e-9)
    _origin_lo = np.full(n_pos, -_R)
    _origin_hi = np.full(n_pos,  _R)

    _ext = np.zeros(n_pos)
    for _fi in frame_idx:
        _allp = np.concatenate([s[_fi] for _, s in _series], axis=0)
        _ext = np.maximum(_ext, _allp.max(axis=0) - _allp.min(axis=0))
    _half = 0.65 * np.maximum(_ext, 1e-9)          # widest frame + 30% padding

    def _frame_lims(fi):
        if args.camera == 'origin':
            return _origin_lo, _origin_hi
        if args.camera == 'global':
            return pos_lo, pos_hi
        allp = np.concatenate([s[fi] for _, s in _series], axis=0)
        c = 0.5 * (allp.max(axis=0) + allp.min(axis=0))
        return c - _half, c + _half

    def _frame_label(fi):
        region = 'Train' if fi < train_ts else 'Test'
        return f'{region} Region — t = {fi * obs_time} {time_unit}'

    # sigma -> enclosed probability mass, same convention as the generator scripts
    _masses = [1.0 - np.exp(-0.5 * k ** 2) for k in range(1, args.sigma_levels + 1)]

    _legend_proxies = [
        mlines.Line2D([], [], color=_palette['True'],  linewidth=2, label='True'),
        mlines.Line2D([], [], color=_palette['Mamba'], linewidth=2, label='Mamba'),
        mlines.Line2D([], [], color=_palette['LSTM'],  linewidth=2, label='LSTM'),
    ]

    def _save_anim(anim, out_stem):
        try:
            anim.save(out_stem + '.mp4', writer=FFMpegWriter(fps=args.fps, bitrate=1800))
            print(f"Saved: {out_stem}.mp4")
        except Exception:
            anim.save(out_stem + '.gif', writer=PillowWriter(fps=args.fps))
            print(f"Saved: {out_stem}.gif  (ffmpeg unavailable)")

    # ──────────────────────────────────────────────────────
    # 4. Single 3D axes
    # ──────────────────────────────────────────────────────
    def _sigma_ellipsoid(pts, k, n_u=20, n_v=12):
        """Wireframe mesh of the k-sigma ellipsoid of a 3D point cloud."""
        mu   = pts.mean(axis=0)
        vals, vecs = np.linalg.eigh(np.cov(pts.T))
        radii = k * np.sqrt(np.clip(vals, 1e-30, None))
        u = np.linspace(0, 2 * np.pi, n_u)
        v = np.linspace(0, np.pi, n_v)
        sx = np.outer(np.cos(u), np.sin(v))
        sy = np.outer(np.sin(u), np.sin(v))
        sz = np.outer(np.ones_like(u), np.cos(v))
        sphere = np.vstack([sx.ravel(), sy.ravel(), sz.ravel()])
        ell = vecs @ (radii[:, None] * sphere) + mu[:, None]
        return [ell[i].reshape(sx.shape) for i in range(3)]

    n_grid = 60

    def _kde_sample(pts):
        """Evenly thin a cloud to --max-kde-samples. Deterministic, no RNG."""
        cap = args.max_kde_samples
        if cap and pts.shape[0] > cap:
            return pts[::int(np.ceil(pts.shape[0] / cap))]
        return pts

    def _tight_grid(pts_2d, pad=0.30):
        """Evaluation grid sized to this cloud, not to the window.

        The axes span the widest frame, but any single cloud may occupy a small
        part of it; gridding the whole window would leave a thin cloud barely one
        cell across and render as a polygon. Sizing the grid to the cloud keeps
        the effective resolution constant for free.
        """
        mn, mx = pts_2d.min(axis=0), pts_2d.max(axis=0)
        span = np.maximum(mx - mn, 1e-9)
        mn, mx = mn - pad * span, mx + pad * span
        GA, GB = np.meshgrid(np.linspace(mn[0], mx[0], n_grid),
                             np.linspace(mn[1], mx[1], n_grid), indexing='ij')
        return GA, GB, np.vstack([GA.ravel(), GB.ravel()])

    fig_3d = plt.figure(figsize=(11, 9))
    ax_3d  = fig_3d.add_subplot(111, projection='3d')

    def _style_3d(lo, hi):
        """Reapply after every cla(), or the view autoscales and jumps."""
        ax_3d.set_xlim(lo[0], hi[0])
        ax_3d.set_ylim(lo[1], hi[1])
        ax_3d.set_xlabel(pos_lbl[0])
        ax_3d.set_ylabel(pos_lbl[1])
        if n_pos >= 3:
            ax_3d.set_zlim(lo[2], hi[2])
            ax_3d.set_zlabel(pos_lbl[2])
            # equal spans deserve an equal box, or an origin-centred view still looks skewed
            if args.camera == 'origin':
                ax_3d.set_box_aspect((1, 1, 1))
        else:
            ax_3d.set_zlim(0, 1.05)
            ax_3d.set_zlabel('Relative density')
        ax_3d.view_init(elev=args.elev, azim=args.azim)      # pin the camera
        ax_3d.grid(alpha=0.2, linewidth=0.5)
        if args.camera == 'origin':
            ax_3d.scatter([0], [0], [0], s=55, color='k', marker='*', depthshade=False)

    def _update_3d(fi):
        lo, hi = _frame_lims(fi)
        ax_3d.cla()
        _style_3d(lo, hi)

        if n_pos >= 3:
            # One cage per model, at the outermost requested sigma level. Nested
            # shells read well as 2D contours (see the projection figure) but three
            # models x three shells is nine overlapping meshes and unreadable in 3D.
            for name, series in _series:
                pts = series[fi]
                ax_3d.scatter(pts[::5, 0], pts[::5, 1], pts[::5, 2],
                              s=3, alpha=0.12, color=_palette[name], rasterized=True)
                ex, ey, ez = _sigma_ellipsoid(pts, args.sigma_levels)
                ax_3d.plot_wireframe(ex, ey, ez, color=_palette[name],
                                     linewidth=0.7, alpha=0.55, rstride=2, cstride=2)
                mu = pts.mean(axis=0)
                ax_3d.scatter([mu[0]], [mu[1]], [mu[2]], s=45, color=_palette[name],
                              edgecolor='k', linewidth=0.6, depthshade=False)
            ax_3d.plot(nom_traj[:fi + 1, 0], nom_traj[:fi + 1, 1], nom_traj[:fi + 1, 2],
                       color='dimgray', linewidth=1.6, label='Nominal path')
        else:
            # z = p(x, y); normalize by the shared max so all three stay comparable
            surfaces = {}
            for name, series in _series:
                pts = _kde_sample(series[fi])
                GX, GY, grid_xy = _tight_grid(pts)
                try:
                    Z = gaussian_kde(pts.T)(grid_xy).reshape(GX.shape)
                except Exception:
                    continue
                surfaces[name] = (GX, GY, Z)
            peak = max((Z.max() for _, _, Z in surfaces.values()), default=0.0)
            if peak > 0:
                for name, (GX, GY, Z) in surfaces.items():
                    # hide each grid's near-zero floor, which would otherwise render
                    # as a solid coloured rectangle over that model's support region
                    Zn = np.where(Z / peak < 0.01, np.nan, Z / peak)
                    if name == 'True':
                        ax_3d.plot_surface(GX, GY, Zn, color=_palette[name],
                                           alpha=0.4, linewidth=0, antialiased=True,
                                           rstride=1, cstride=1)
                    else:
                        ax_3d.plot_wireframe(GX, GY, Zn, color=_palette[name],
                                             linewidth=0.5, alpha=0.7,
                                             rstride=4, cstride=4)
            ax_3d.plot(nom_traj[:fi + 1, 0], nom_traj[:fi + 1, 1], zs=0, zdir='z',
                       color='dimgray', linewidth=1.6, label='Nominal path')

        ax_3d.set_title(f'Position PDF Evolution — {_frame_label(fi)}')
        ax_3d.legend(handles=_legend_proxies +
                     [mlines.Line2D([], [], color='dimgray', linewidth=1.6, label='Nominal path')],
                     loc='upper left', fontsize=12)
        return []

    print(f"Rendering 3D PDF evolution ({len(frame_idx)} frames)...")
    anim_3d = FuncAnimation(fig_3d, _update_3d, frames=frame_idx,
                            interval=70, blit=False, repeat=False)
    _save_anim(anim_3d, _pfx + '_pdf_evolution_3d')
    plt.close(fig_3d)

    # ──────────────────────────────────────────────────────
    # 5. Projection contour panels
    # ──────────────────────────────────────────────────────
    _proj_pairs = [(0, 1), (0, 2), (1, 2)] if n_pos >= 3 else [(0, 1)]

    def _kde_contours(pts_2d):
        """Grid density plus contour levels at fixed enclosed-probability masses.

        Levels come from quantiles of the KDE evaluated at its own samples, so a
        contour means the same credible mass in every frame and for every model
        even as the absolute density magnitude changes by orders of magnitude.
        """
        pts_2d = _kde_sample(pts_2d)
        try:
            kde = gaussian_kde(pts_2d.T)
        except Exception:
            return None
        GA, GB, positions = _tight_grid(pts_2d)
        Z = kde(positions).reshape(GA.shape)
        d = kde(pts_2d.T)
        levels = np.quantile(d, [1.0 - m for m in _masses])
        levels = np.unique(np.sort(levels))          # contour needs strictly increasing
        if levels.size == 0 or not np.all(np.isfinite(levels)):
            return None
        return GA, GB, Z, levels

    fig_pr, axes_pr = plt.subplots(1, len(_proj_pairs),
                                   figsize=(6 * len(_proj_pairs), 6), squeeze=False)
    axes_pr = axes_pr.ravel()

    def _style_proj(lo, hi):
        for ax, (a, b) in zip(axes_pr, _proj_pairs):
            ax.set_xlim(lo[a], hi[a])
            ax.set_ylim(lo[b], hi[b])
            ax.set_xlabel(pos_lbl[a])
            ax.set_ylabel(pos_lbl[b])
            ax.grid(alpha=0.2, linewidth=0.5)
            if args.camera == 'origin':
                ax.set_aspect('equal')
                ax.plot(0, 0, marker='*', color='k', markersize=9)

    def _update_proj(fi):
        lo, hi = _frame_lims(fi)
        for ax in axes_pr:
            ax.cla()
        _style_proj(lo, hi)
        for ax, (a, b) in zip(axes_pr, _proj_pairs):
            for name, series in _series:
                res = _kde_contours(series[fi][:, [a, b]])
                if res is None:
                    continue
                GA, GB, Z, levels = res
                ax.contour(GA, GB, Z, levels=levels,
                           colors=_palette[name], linewidths=1.2, alpha=0.9)
            ax.plot(nom_traj[:fi + 1, a], nom_traj[:fi + 1, b],
                    color='dimgray', linewidth=1.4)
            ax.set_title(f'{pos_lbl[a].split(" ")[0]}–{pos_lbl[b].split(" ")[0]}')
        fig_pr.suptitle(f'Position PDF Projections — {_frame_label(fi)}')
        return []

    fig_pr.legend(handles=_legend_proxies, loc='lower center', ncol=3,
                  frameon=True, fontsize=13)
    fig_pr.subplots_adjust(bottom=0.20, top=0.86)
    print(f"Rendering PDF projection evolution ({len(frame_idx)} frames)...")
    anim_pr = FuncAnimation(fig_pr, _update_proj, frames=frame_idx,
                            interval=70, blit=False, repeat=False)
    _save_anim(anim_pr, _pfx + '_pdf_evolution_proj')
    plt.close(fig_pr)

print("Done.")
