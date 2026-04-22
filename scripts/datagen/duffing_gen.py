import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, Delaunay
from scipy.spatial.qhull import QhullError # import here for p36 compatibility
from matplotlib.collections import LineCollection

try:
    from numba import njit, prange
    _HAVE_NUMBA = True
except Exception:
    _HAVE_NUMBA = False


# -----------------------------
# Dynamics + numerical integrator
# -----------------------------
def duffing_oscillator_f(x, t,u, omega, zeta, alpha,beta,gamma, total_mass=2.0):
    """
    x = [x1, x2]
    x1dot = x2
    x2dot = -omega^2 x1 - beta*x1^3 - 2*zeta*omega x2 + u
    """
    x1, x2 = x
    dx1 = x2
    dx2 = -alpha * x1 - beta * (x1**3) - zeta * x2 + u / total_mass + gamma * np.cos(omega * t)
    
    return np.array([dx1, dx2], dtype=float)


def rk4_step(x, t, u, dt, omega, zeta, alpha, beta, gamma, total_mass=2.0):
    k1 = duffing_oscillator_f(x, t, u, omega, zeta, alpha, beta, gamma, total_mass=total_mass)
    k2 = duffing_oscillator_f(x + 0.5 * dt * k1, t + 0.5 * dt, u, omega, zeta, alpha, beta, gamma, total_mass=total_mass)
    k3 = duffing_oscillator_f(x + 0.5 * dt * k2, t + 0.5 * dt, u, omega, zeta, alpha, beta, gamma, total_mass=total_mass)
    k4 = duffing_oscillator_f(x + dt * k3,t + dt,u , omega,zeta,alpha,beta,gamma,total_mass=total_mass)
    return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


# -----------------------------
# Numba-accelerated kernels
# -----------------------------
if _HAVE_NUMBA:
    @njit
    def _duffing_oscillator_f_nb(x, t, u, omega, zeta, alpha, beta, gamma, total_mass):
        x1 = x[0]
        x2 = x[1]
        dx1 = x2
        dx2 = -alpha * x1 - beta * (x1**3) - zeta * x2 + u / total_mass + gamma * np.cos(omega * t)
        out = np.empty(2, dtype=np.float64)
        out[0] = dx1
        out[1] = dx2
        return out


    @njit
    def _rk4_step_nb(x, t, u, dt, omega, zeta, alpha, beta, gamma, total_mass):
        k1 = _duffing_oscillator_f_nb(x, t, u, omega, zeta, alpha, beta, gamma, total_mass)
        k2 = _duffing_oscillator_f_nb(x + 0.5 * dt * k1, t + 0.5 * dt, u, omega, zeta, alpha, beta, gamma, total_mass)
        k3 = _duffing_oscillator_f_nb(x + 0.5 * dt * k2, t + 0.5 * dt, u, omega, zeta, alpha, beta, gamma, total_mass)
        k4 = _duffing_oscillator_f_nb(x + dt * k3,t + dt,u , omega,zeta,alpha,beta,gamma,total_mass=total_mass)
        return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


    @njit(parallel=True)
    def _monte_carlo_reachable_set_numba(
        x0_posvel,
        steps,
        dt,
        omega,
        zeta,
        alpha,
        beta,
        gamma,
        total_mass,
        snapshot_indices,
        snapshots,
        X_final
    ):
        n_traj = x0_posvel.shape[0]
        n_snaps = snapshot_indices.shape[0]

        for i in prange(n_traj):
            x = np.empty(2, dtype=np.float64)
            x[0] = x0_posvel[i, 0]
            x[1] = x0_posvel[i, 1]

            for s in range(n_snaps):
                if snapshot_indices[s] == 0:
                    snapshots[s, i, 0] = x[0]
                    snapshots[s, i, 1] = x[1]

            for k in range(steps):
                x = _rk4_step_nb(x, k * dt, 0.0, dt, omega, zeta, alpha, beta, gamma, total_mass)
                k1 = k + 1
                for s in range(n_snaps):
                    if snapshot_indices[s] == k1:
                        snapshots[s, i, 0] = x[0]
                        snapshots[s, i, 1] = x[1]

            X_final[i, 0] = x[0]
            X_final[i, 1] = x[1]


# -----------------------------
# Convex hull (2D) - monotone chain
# -----------------------------
def _cross(o, a, b):
    # 2D cross product (OA x OB)
    return (a[0] - o[0])*(b[1] - o[1]) - (a[1] - o[1])*(b[0] - o[0])


def convex_hull_2d(points):
    """
    points: (N,2)
    Returns hull vertices in CCW order as (H,2). If N<3, returns unique points.
    Monotone chain. O(N log N).
    """
    pts = np.asarray(points, dtype=float)
    if pts.shape[0] == 0:
        return pts

    # sort lexicographically, unique
    pts = np.unique(pts, axis=0)
    if pts.shape[0] <= 2:
        return pts

    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]  # sort by x then y

    lower = []
    for p in pts:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(tuple(p))

    upper = []
    for p in pts[::-1]:
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(tuple(p))

    # concatenate, removing duplicate endpoints
    hull = lower[:-1] + upper[:-1]
    return np.array(hull, dtype=float)

# -----------------------------
# Alpha shape segments and area (for 2D point clouds, to get a tighter hull than convex hull)
# -----------------------------
def alpha_shape_segments_and_area(points, radius_quantile=0.65):
    n = points.shape[0]
    if n < 4:
        hull = ConvexHull(points)
        verts = hull.vertices
        cyc = np.column_stack([verts, np.roll(verts, -1)])
        return points[cyc], hull.volume

    try:
        tri = Delaunay(points)
    except QhullError:
        hull = ConvexHull(points)
        verts = hull.vertices
        cyc = np.column_stack([verts, np.roll(verts, -1)])
        return points[cyc], hull.volume

    simplices = tri.simplices
    p = points[simplices]
    a = np.linalg.norm(p[:, 1] - p[:, 0], axis=1)
    b = np.linalg.norm(p[:, 2] - p[:, 1], axis=1)
    c = np.linalg.norm(p[:, 0] - p[:, 2], axis=1)
    s = 0.5 * (a + b + c)
    area_sq = s * (s - a) * (s - b) * (s - c)
    area_sq = np.maximum(area_sq, 0.0)
    tri_area = np.sqrt(area_sq)

    valid = tri_area > 1e-12
    if not np.any(valid):
        hull = ConvexHull(points)
        verts = hull.vertices
        cyc = np.column_stack([verts, np.roll(verts, -1)])
        return points[cyc], hull.volume

    circum_r = np.full_like(tri_area, np.inf)
    circum_r[valid] = (a[valid] * b[valid] * c[valid]) / (4.0 * tri_area[valid])
    r_thresh = np.quantile(circum_r[valid], radius_quantile)
    keep = valid & (circum_r <= r_thresh)

    if not np.any(keep):
        hull = ConvexHull(points)
        verts = hull.vertices
        cyc = np.column_stack([verts, np.roll(verts, -1)])
        return points[cyc], hull.volume

    kept = simplices[keep]
    edges = np.concatenate(
        [kept[:, [0, 1]], kept[:, [1, 2]], kept[:, [2, 0]]],
        axis=0
    )
    edges = np.sort(edges, axis=1)
    uniq_edges, counts = np.unique(edges, axis=0, return_counts=True)
    boundary_edges = uniq_edges[counts == 1]

    return points[boundary_edges], tri_area[keep].sum()

def plot_alpha_shape(points, k, radius_quantile=0.95,ax=None):
    segs, area = alpha_shape_segments_and_area(points, radius_quantile)
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(points[:, 0], points[:, 1], s=5, alpha=0.03)
    lc = LineCollection(segs, linewidths=2,colors="C"+str(k%10))
    ax.add_collection(lc)
    plt.tight_layout()

# -----------------------------
# Monte Carlo reachability
# -----------------------------
def monte_carlo_reachable_set(
    x0_mean,
    x0_std,
    omega,
    zeta,
    alpha,
    beta,
    gamma,
    total_mass,
    dt,
    steps,
    n_traj,
    snapshot_indices=(0, 200, 400, 800),
    seed=0,
    mean_shift=None
):
    rng = np.random.default_rng(seed)

    x0_mean = np.asarray(x0_mean, dtype=float).reshape(2)
    if mean_shift is not None:
        x0_mean = x0_mean + np.asarray(mean_shift, dtype=float).reshape(2)
    rad = np.asarray(x0_std, dtype=float).reshape(2)

    snapshot_indices = tuple(int(i) for i in snapshot_indices if 0 <= i <= steps)
    snapshot_idx_arr = np.asarray(snapshot_indices, dtype=np.int64)

    # Precompute randomness in NumPy for Numba compatibility.
    x0_posvel = x0_mean + rng.uniform(-1.0, 1.0, size=(n_traj, 2)) * rad

    snapshots_arr = np.zeros((len(snapshot_indices), n_traj, 2), dtype=float)
    X_final = np.zeros((n_traj, 2), dtype=float)

    if _HAVE_NUMBA:
        _monte_carlo_reachable_set_numba(
            x0_posvel,
            steps,
            dt,
            omega,
            zeta,
            alpha,
            beta,
            gamma,
            total_mass,
            snapshot_idx_arr,
            snapshots_arr,
            X_final
        )
    else:
        for i in range(n_traj):
            # sample initial condition from a box
            x = x0_posvel[i].copy()

            for s, k in enumerate(snapshot_indices):
                if k == 0:
                    snapshots_arr[s, i] = x

            for k in range(steps):
                x = rk4_step(x, k * dt, 0.0, dt, omega, zeta, alpha=alpha, beta=beta, gamma=gamma, total_mass=total_mass)
                k1 = k + 1
                for s, snap_k in enumerate(snapshot_indices):
                    if snap_k == k1:
                        snapshots_arr[s, i] = x

            X_final[i] = x

    snapshots = {int(k): snapshots_arr[idx] for idx, k in enumerate(snapshot_indices)}
    return snapshots, X_final



def compute_hulls_for_snapshots(snapshots, downsample=None, seed=0):
    """
    snapshots: dict {k: (N,2) array}
    downsample: if not None, randomly pick this many points per snapshot for hull
    Returns dict {k: hull_vertices (H,2)}
    """
    rng = np.random.default_rng(seed)
    hulls = {}

    for k, X in snapshots.items():
        X = np.asarray(X, dtype=float)
        if X.shape[0] == 0:
            hulls[k] = X
            continue

        if downsample is not None and X.shape[0] > downsample:
            idx = rng.choice(X.shape[0], size=downsample, replace=False)
            X_use = X[idx]
        else:
            X_use = X

        hulls[k] = convex_hull_2d(X_use)

    return hulls    

if __name__ == "__main__":
    # System
    k = 0.5
    m = 2.0
    omega = np.sqrt(k / m)
    zeta = 0.05
    alpha = 0.0
    beta = 1.0
    gamma = 0.2
    total_mass = 2.0

    # args parse for delta v_radius
    import argparse
    parser = argparse.ArgumentParser(description="Duffing Oscillator Monte Carlo Reachability")
    parser.add_argument("--no-numba", action="store_true", help="Disable Numba acceleration")
    parser.add_argument("--steps", type=int, default=None, help="Number of snapshots to compute")
    parser.add_argument("--T", type=float, default=16.0, help="Total simulation time in seconds")
    parser.add_argument("--dt", type=float, default=0.02, help="Time step for integration")
    parser.add_argument("--n", type=int, default=20000, help="Number of Monte Carlo trajectories to simulate")
    parser.add_argument("--plot", action="store_true", help="Set this flag to plot the snapshots and hulls instead of saving data")
    parser.add_argument("--mu", type=np.array, default=[0.2, 0.0], help="Mean shift to apply to initial conditions")
    parser.add_argument("--sigma", type=float, default=0.2, help="Additional uniform noise radius to apply to initial conditions")
    if parser.parse_args().no_numba:
        _HAVE_NUMBA = False
    args = parser.parse_args()

    save_dir = "./data/test/"
    save_file = f"duffing_monte_carlo_trajectories_sigma_{args.sigma}_dt_{args.dt}_n_{args.n}.npz"

    # Time
    dt = args.dt
    steps = int(args.T / dt)

    # Monte Carlo
    n_traj = args.n  # increase for tighter hull

    x0_mean = args.mu  # center of initial box in (position, velocity)

    x0_std = [args.sigma, args.sigma]  # standard deviation of initial box in (position, velocity)
    
    if args.steps is not None:
        zoneNums = args.steps
    else:
        zoneNums = steps
    snapshot_indices = [0]

    for i in range(1, zoneNums + 1):
        snapshot_indices.append(int(i * steps / zoneNums))

    snapshots, X_final = monte_carlo_reachable_set(
        x0_mean=x0_mean,
        x0_std=x0_std,
        omega=omega,
        zeta=zeta,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        total_mass=total_mass,
        dt=dt,
        steps=steps,
        n_traj=n_traj,
        snapshot_indices=snapshot_indices,
        seed=1
    )

    if args.plot:

        # -----------------------------
        # Plotting
        # -----------------------------
        def plot_snapshots_and_final_hull(snapshots, X_final, dt, title="Monte Carlo + Convex Hull"):
            fig, ax = plt.subplots(figsize=(8, 6))

            # Plot snapshots as faint clouds
            snap_keys = sorted(snapshots.keys())
            for k in snap_keys:
                Xk = snapshots[k]
                ax.scatter(Xk[:, 0], Xk[:, 1], s=3, alpha=0.06)#, label=f"t={k*dt:.2f}s" if k != snap_keys[0] else None)
                # ax.scatter(Xk[:, 0], Xk[:, 1], s=3, alpha=0.6, label=f"t={k*dt:.2f}s" if k != snap_keys[0] else None)

            # Plot final points
            ax.scatter(X_final[:, 0], X_final[:, 1], s=6, alpha=0.18, label="final samples")

            # get alpha shape hull for final points and plot
            hull_final, area_final = alpha_shape_segments_and_area(X_final, radius_quantile=0.95)
            ax.add_collection(LineCollection(hull_final, linewidths=2, colors='k', label=f"final hull (area={area_final:.2f})"))

            ax.set_xlabel("x1 (position)")
            ax.set_ylabel("x2 (velocity)")
            ax.set_title(title)
            ax.grid(True)
            ax.legend(loc="best", frameon=True)
            plt.tight_layout()
            
        def plot_snapshot_hulls(hulls, dt, show_points=False, snapshots=None, title="Snapshot Hulls"):
            """
            hulls: dict {k: (H,2)}
            If show_points=True, also scatter the underlying snapshot points (provide snapshots dict).
            """
            fig, ax = plt.subplots(figsize=(8, 6))

            keys = sorted(hulls.keys())

            if show_points:
                if snapshots is None:
                    raise ValueError("snapshots must be provided if show_points=True")
                for k in keys:
                    Xk = snapshots[k]
                    ax.scatter(Xk[:, 0], Xk[:, 1], s=2, alpha=0.03)

            for k in keys:
                hull = hulls[k]
                if hull.shape[0] >= 3:
                    hull_closed = np.vstack([hull, hull[0]])
                    ax.plot(hull_closed[:, 0], hull_closed[:, 1], linewidth=2.0)
                elif hull.shape[0] == 2:
                    ax.plot(hull[:, 0], hull[:, 1], linewidth=2.0)
                elif hull.shape[0] == 1:
                    ax.scatter(hull[0, 0], hull[0, 1], s=40)

            ax.set_xlabel("x1 (position)")
            ax.set_ylabel("x2 (velocity)")
            ax.set_title(title)
            ax.grid(True)
            ax.legend(loc="best", frameon=True)
            plt.tight_layout()


        plot_snapshots_and_final_hull(
            snapshots,
            X_final,
            dt,
            title="Duffing Oscillator Reachability (Monte Carlo Trajectories)"
        )

        hulls = compute_hulls_for_snapshots(
            snapshots,
            downsample=8000,  # None for full set; use a number to speed up
            seed=123
        )

        plot_snapshot_hulls(
            hulls,
            dt,
            show_points=False,  # set True if you want clouds underneath
            snapshots=snapshots,
            title="Duffing Oscillator Reachability (Monte Carlo Convex Hulls with Nominal Trajectory)"
        )
        # solve nominal system for reference
        x_nom = np.array(x0_mean, dtype=float)
        traj_nom = np.zeros((steps + 1, 2), dtype=float)
        traj_nom[0] = x_nom
        for k in range(steps):
            if k == 2:
                x_nom[1] = x_nom[1] + 0.0  # nominal control is zero
            x_nom = rk4_step(x_nom, k * dt, 0.0, dt, omega, zeta, alpha=alpha, beta=beta, gamma=gamma, total_mass=total_mass)
            traj_nom[k + 1] = x_nom
        # plot nominal trajectory on top of last figure
        plt.plot(traj_nom[:, 0], traj_nom[:, 1], 'k--', linewidth=2.5, label="nominal trajectory")
        plt.legend(loc="best", frameon=True)

        fig, ax = plt.subplots(figsize=(8, 6))
        # plot snapshot alpha shapes
        for i, k in enumerate(snapshot_indices):
            plot_alpha_shape(snapshots[k],i, ax=ax, radius_quantile=0.95)
        ax.set_xlabel("x1 (position)")
        ax.set_ylabel("x2 (velocity)")
        ax.set_title("Duffing Oscillator Reachability (Monte Carlo Alpha Shapes)")
        ax.grid(True)
        plt.plot(traj_nom[:, 0], traj_nom[:, 1], 'k--', linewidth=2.5, label="nominal trajectory")

        
        # generate pdf for a single snapshot time (e.g. t=4s) by histogramming the points and fitting a KDE
        snapshot_time = args.T  # seconds
        snapshot_index = int(snapshot_time / dt)

        if snapshot_index in snapshots:
            from scipy.stats import gaussian_kde

            Xk = snapshots[snapshot_index]
            kde = gaussian_kde(Xk.T)

            # Create a grid for evaluation
            x_min, x_max = Xk[:, 0].min() - 0.5, Xk[:, 0].max() + 0.5
            y_min, y_max = Xk[:, 1].min() - 0.5, Xk[:, 1].max() + 0.5
            x_grid, y_grid = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
            grid_coords = np.vstack([x_grid.ravel(), y_grid.ravel()])
            pdf_values = kde(grid_coords).reshape(x_grid.shape)

            # Plot the PDF as a contour plot with improved contrast and less clutter.
            fig, ax = plt.subplots(figsize=(8, 6), dpi=140)
            levels = np.unique(np.quantile(pdf_values, np.linspace(0.05, 0.995, 20)))
            contourf = ax.contourf(
                x_grid, y_grid, pdf_values, levels=levels, cmap="cividis", extend="both"
            )
            ax.contour(
                x_grid,
                y_grid,
                pdf_values,
                levels=levels[::3],
                colors="k",
                linewidths=0.5,
                alpha=0.45,
            )
            cbar = fig.colorbar(contourf, ax=ax, pad=0.02)
            cbar.set_label("PDF value", fontsize=11)

            # Downsample displayed points so the KDE remains visible.
            sample_count = min(2000, len(Xk))
            sample_idx = np.random.choice(len(Xk), size=sample_count, replace=False)
            ax.scatter(
                Xk[sample_idx, 0],
                Xk[sample_idx, 1],
                s=8,
                c="white",
                edgecolors="black",
                linewidths=0.25,
                alpha=0.45,
                label="samples",
            )
            ax.set_xlabel("x1 (position)", fontsize=11)
            ax.set_ylabel("x2 (velocity)", fontsize=11)
            ax.set_title(f"State PDF at t={snapshot_time:.2f}s", fontsize=12)
            ax.grid(alpha=0.2, linestyle="--")
            ax.legend(frameon=True, loc="upper right")
            ax.set_aspect("equal", adjustable="box")
            fig.tight_layout()
            
            
        plt.show()

    else:
    
        # using snapshots, construct n_traj trajectories
        n_traj = X_final.shape[0]
        traj_list = []
        for i in prange(n_traj):
            traj = np.zeros((steps + 1, 2), dtype=float)
            traj[0] = snapshots[0][i]
            for k in range(steps):
                if k in snapshots:
                    traj[k + 1] = snapshots[k][i]
                else:
                    traj[k + 1] = traj[k]
            traj_list.append(traj)
        traj_list = np.array(traj_list) # n,steps,2

        if args.steps is not None:
            print(f"Saving Monte Carlo trajectories with snapshots at {args.steps} equispaced time points, mimicking discrete-time systems.")
        else:
            print(f"Saving Monte Carlo trajectories with snapshots at default propagation points.")

        # save trajectories to disk
        np.savez(save_dir + save_file, trajectories=traj_list, dt=dt,parameters = [alpha, beta, gamma, omega,zeta],mu=x0_mean,sigma=args.sigma,T=args.T)
