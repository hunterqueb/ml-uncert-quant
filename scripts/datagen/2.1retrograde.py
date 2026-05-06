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

try:
    from numba import njit, prange
    _HAVE_NUMBA = True
except Exception:
    _HAVE_NUMBA = False

if args.no_numba:
    _HAVE_NUMBA = False

plotOn = args.plot


problemDim = 4
m_1 = 5.974E24  # kg
m_2 = 7.348E22 # kg
mu = m_2/(m_1 + m_2)

DU = 389703
G = 6.67430e-11
TU = 382981
tEnd = 2.990440964/2

IC = np.array((.965692323,0,0,-1.730853537))
# .099, 0 -- in geo


# mu = 4735.09237566722 in km


std = np.array((0.0001,0.0001,0.0001,0.0001))
print("IC in km and km/s",nonDim2Dim4(IC.reshape(1,problemDim),DU,TU))
print("std in km and km/s",nonDim2Dim4(std.reshape(1,problemDim),DU,TU))

tEnd = 26.51115905/(TU/86400)


print(mu * DU)
print(.099 * DU - mu * DU)
def system(t, Y,mu=mu):
    """Solve the CR3BP in nondimensional coordinates in 2d.
    
    The state vector is Y, with the first three components as the
    position of $m$, and the second three components its velocity.
    
    The solution is parameterized on $\\pi_2$, the mass ratio.
    """
    # Get the position and velocity from the solution vector
    x, y = Y[:2]
    xdot, ydot = Y[2:]

    # Define the derivative vector

    dydt1 = xdot
    dydt2 = ydot

    r1 = np.sqrt((x + mu)**2 + y**2)
    r2 = np.sqrt((x - 1 + mu)**2 + y**2)

    dydt3 = 2 * ydot + x - (1 - mu) * (x + mu) / r1**3 - mu * (x - 1 + mu) / r2**3
    dydt4 = -2 * xdot + y - (1 - mu) * y / r1**3 - mu * y / r2**3

    return np.array([dydt1, dydt2,dydt3,dydt4])


t0 = 0; tf = tEnd

delT = 0.001

nSamples = int(np.ceil((tf - t0) / delT))
t = np.linspace(t0, tf, nSamples)

# t , numericResult = ode1412(system,[t0,tf],IC,t)
t , numericResult = ode87(system,[t0,tf],IC,t,rtol=1e-15,atol=1e-15)

t = t / tEnd

# plot phase space of trajectory
plt.figure(figsize=(8, 6))
plt.plot(numericResult[:, 0], numericResult[:, 1], label='Trajectory')
plt.plot(-mu, 0, 'ko', label='Earth')
plt.plot((1-mu), 0, 'go', label='Moon')
plt.plot(IC[0], IC[1], 'rx', label='Initial Condition')
plt.title('Retrograde Orbit in the Earth-Moon System')
plt.xlabel('x (non-dimensional)')
plt.ylabel('y (non-dimensional)')
plt.legend()
plt.grid()

print("Finding GEO delta in km:",0.0005 * DU)
# find in trajectory where the orbit is in GEO around the earth
r_geo = 42164 / DU  # GEO radius in non-dimensional units
r_earth = np.sqrt((numericResult[:, 0] + mu)**2 + numericResult[:, 1]**2)
geo_indices = np.where(np.abs(r_earth - r_geo) < 0.0005)[0]
if len(geo_indices) > 0:
    plt.plot(numericResult[geo_indices, 0], numericResult[geo_indices, 1], 'bx', label='GEO')

# get first GEO_index - the second one (after the initial point)
# this one looks better
if len(geo_indices) > 0:
    first_geo_index = geo_indices[1]
    print("First GEO index:", first_geo_index)

IC_GEO = numericResult[first_geo_index, :]
print("IC_GEO in km and km/s",nonDim2Dim4(IC_GEO.reshape(1,problemDim),DU,TU))
# apply std to IC_GEO
IC_GEO_noisy = IC_GEO + np.random.normal(-1, 1, problemDim) * std
print("IC_GEO_noisy in km and km/s",nonDim2Dim4(IC_GEO_noisy.reshape(1,problemDim),DU,TU))

# found this value by trial and error
t0 = 0; tf = tEnd  * 0.722

# set delT to be approximately 4 hr in non-dimensional units
delT = args.obs * 3600 / TU

nSamples = int(np.ceil((tf - t0) / delT))
t = np.linspace(t0, tf, nSamples)

# t , numericResult = ode1412(system,[t0,tf],IC,t)
t , numericResult = ode87(system,[t0,tf],IC_GEO,t,rtol=1e-15,atol=1e-15)
t = t / tf

plt.figure(figsize=(8, 6))
plt.plot(numericResult[:, 0], numericResult[:, 1], label='Trajectory')
plt.plot(-mu, 0, 'ko', label='Earth')
plt.plot((1-mu), 0, 'go', label='Moon')
plt.plot(IC_GEO[0], IC_GEO[1], 'rx', label='Initial Condition')
plt.title('Retrograde Orbit in the Earth-Moon System (GEO Initial Condition)')
plt.xlabel('x (non-dimensional)')
plt.ylabel('y (non-dimensional)')
plt.legend()
plt.grid()



n_traj = []
t_arr = []

for i in range(args.n):
    delT = args.obs * 3600 / TU

    nSamples = int(np.ceil((tf - t0) / delT))
    t = np.linspace(t0, tf, nSamples)

    # Generate a new initial condition
    IC_noisy = IC_GEO + np.random.normal(-1, 1, problemDim) * std
    # Solve the system with the new initial condition
    t , numericResult = ode87(system,[t0,tf],IC_noisy,t,rtol=1e-15,atol=1e-15)
    n_traj.append(numericResult)
    t_arr.append(t)

n_traj = np.array(n_traj)
t_arr = np.array(t_arr)
n_traj_dim = nonDim2Dim4(n_traj, DU, TU)
plt.figure(figsize=(8, 6))
# plot the trajectories 
for i in range(len(n_traj)):
    plt.plot(n_traj[i][:, 0], n_traj[i][:, 1])

plt.title('Dataset Retrograde Orbits in the Earth-Moon System')
plt.xlabel('x (non-dimensional)')
plt.ylabel('y (non-dimensional)')
plt.scatter(-mu, 0, c='k', marker='o', label='Earth')
plt.scatter(1-mu, 0, c='g', marker='o', label='Moon')
plt.grid()

plt.figure(figsize=(8, 6))
# plot the initial conditions xy phase space next to xdot ydot phase space
plt.subplot(1, 2, 1)
for i in range(len(n_traj)):
    plt.scatter(n_traj_dim[i][0][0], n_traj_dim[i][0][1], c='r', marker='o', s=5)

plt.title('Dataset Initial Conditions for Retrograde Orbits in the Earth-Moon System')
plt.xlabel('x (km)')
plt.ylabel('y (km)')
# plt.scatter(-mu, 0, c='k', marker='o', label='Earth')
# plt.scatter(1-mu, 0, c='g', marker='o', label='Moon')
plt.grid()

plt.subplot(1, 2, 2)
for i in range(len(n_traj)):
    plt.scatter(n_traj_dim[i][0][2], n_traj_dim[i][0][3], c='r', marker='o', s=5)

plt.title('Dataset Initial Conditions for Retrograde Orbits in the Earth-Moon System')
plt.xlabel('xdot (km/s)')
plt.ylabel('ydot (km/s)')
plt.grid()

print("Shape of n_traj:", np.array(n_traj).shape)

save_dir = "./data/cr3bp/"
# ensure the save directory exists
os.makedirs(save_dir, exist_ok=True)
save_file = f"2.1_retrograde_geo_to_moon_obs-dt_{args.obs}_n_{args.n}_ND.npz"

save_path = os.path.join(save_dir, save_file)

if not args.no_save:
    np.savez_compressed(save_path, trajectories=n_traj,mu = IC_GEO, std=std,dt=args.obs,t=t_arr)


if plotOn is True:
    plt.show()
