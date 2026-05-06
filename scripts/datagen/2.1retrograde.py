import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from qutils.integrators import ode87
from qutils.orbital import nonDim2Dim4
parser = argparse.ArgumentParser(description='Generate data for the retrograde orbit in the Earth-Moon system.')
parser.add_argument('--plot', action='store_true', help='Whether to plot the results.')
parser.add_argument('--obs', type=float, default=2, help='The time delta (in hours) between observations.')
args = parser.parse_args()
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


std = np.array((0.001,0.001,1,1))
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

# set delT to be approximately 4 hr in non-dimensional units
delT = args.obs * 3600 / TU

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

if plotOn is True:
    plt.show()
