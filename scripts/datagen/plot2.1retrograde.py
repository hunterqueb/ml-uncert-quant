import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

from qutils.integrators import ode87
from qutils.orbital import nonDim2Dim4
parser = argparse.ArgumentParser(description='Generate data for the retrograde orbit in the Earth-Moon system.')
parser.add_argument('--plot', action='store_true', help='Whether to plot the results.')
parser.add_argument('--obs', type=float, default=2, help='The time delta (in hours) between observations.')
parser.add_argument('--n', type=int, default=10000, help='The number of systems to generate.')
parser.add_argument('--no-save', action='store_true', help='Whether to not save the results.')
parser.add_argument('--no-numba', action='store_true', help='Whether to not use numba.')
args = parser.parse_args()

save_dir = "./data/cr3bp/"
# ensure the save directory exists
os.makedirs(save_dir, exist_ok=True)
save_file = f"2.1_retrograde_geo_to_moon_obs-dt_{args.obs}_n_{args.n}_ND.npy"

save_path = os.path.join(save_dir, save_file)
# np.savez_compressed(save_path, trajectories=n_traj,mu = IC_GEO, std=std,dt=args.obs,t=t_arr)

dataset = np.load(save_path)['trajectories']
num_trajs = dataset.shape[0]
num_time_steps = dataset.shape[1]
print(dataset.shape)

problemDim = 4
m_1 = 5.974E24  # kg
m_2 = 7.348E22 # kg
mu = m_2/(m_1 + m_2)

DU = 389703
G = 6.67430e-11
TU = 382981
tEnd = 2.990440964/2



# dataset_dim = nonDim2Dim4(dataset, DU, TU)
plt.figure(figsize=(8, 6))
# plot the trajectories 
for i in range(len(dataset)):
    plt.plot(dataset[i][:, 0], dataset[i][:, 1])

plt.title('Dataset Retrograde Orbits in the Earth-Moon System')
plt.xlabel('x (non-dimensional)')
plt.ylabel('y (non-dimensional)')
plt.scatter(-mu, 0, c='k', marker='o', label='Earth')
plt.scatter(1-mu, 0, c='g', marker='o', label='Moon')
plt.grid()


def synodic_to_eci(ic, t, mu=mu):
    """Convert a 2D CR3BP synodic (rotating) state to Earth-centered inertial.

    ic : (..., 4) array  [x, y, xdot, ydot] in non-dimensional synodic coords
    t  : non-dimensional time (scalar or broadcastable array)
    Returns an array of the same shape in ECI non-dimensional coords.
    """
    theta = t  # ω = 1 in non-dim units
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    x_ec = ic[..., 0] + mu
    y_ec = ic[..., 1]
    xdot_rot = ic[..., 2]
    ydot_rot = ic[..., 3]

    x_eci  =  x_ec * cos_t - y_ec * sin_t
    y_eci  =  x_ec * sin_t + y_ec * cos_t
    vx_eci = (xdot_rot - y_ec) * cos_t - (ydot_rot + x_ec) * sin_t
    vy_eci = (xdot_rot - y_ec) * sin_t + (ydot_rot + x_ec) * cos_t

    return np.stack([x_eci, y_eci, vx_eci, vy_eci], axis=-1)

# convert the initial conditions of the dataset from 3BP to ECI frame
# dataset trajectories start at t=0 so theta=0; shift origin to Earth and apply Coriolis correction
ics_synodic = dataset[:, 0, :]          # (N, 4) initial conditions in synodic frame
ics_eci = synodic_to_eci(ics_synodic, t=1)

ics_eci = nonDim2Dim4(ics_eci, DU, TU)

plt.figure(figsize=(8, 6))
# plot the initial conditions xy phase space next to xdot ydot phase space

plt.subplot(1, 2, 1)
plt.title('Initial Conditions: XY Phase Space')
plt.scatter(ics_eci[:, 0], ics_eci[:, 1], c='r', marker='o', s=5)
plt.xlabel('x ECI (km)')
plt.ylabel('y ECI (km)')
plt.grid()

plt.subplot(1, 2, 2)
plt.title('Initial Conditions: Xdot Ydot Phase Space')
plt.scatter(ics_eci[:, 2], ics_eci[:, 3], c='r', marker='o', s=5)
plt.xlabel('vx ECI (km/s)')
plt.ylabel('vy ECI (km/s)')
plt.grid()

plt.suptitle('Dataset Initial Conditions for Retrograde Orbits in the Earth-Moon System')

print("Shape of dataset:", np.array(dataset).shape)


plt.show()
