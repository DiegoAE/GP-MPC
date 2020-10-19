from sys import path

import casadi as ca
import matplotlib.pyplot as plt
import numpy as np

# Add gp_mpc pagkage to path
path.append(r"./../")

from gp_mpc import Model, GP


def plot_van_der_pol():
    """ Plot comparison of GP prediction with exact simulation
        on a 2000 step prediction horizon
    """
    Nt = 1000
    x0 = np.array([2., .201])

    cov = np.zeros((2,2))
    x = np.zeros((Nt,2))
    x_sim = np.zeros((Nt,2))
    x_sim2 = np.zeros((Nt,2))

    x[0] = x0
    x_sim[0] = x0
    x_sim2[0] = x0

    #gp.set_method('ME')         # Use Mean Equivalence as GP method
    for i in range(Nt-1):
        #x_t, cov = gp.predict(x[i], [], cov)
        #x[i + 1] = np.array(x_t).flatten()
        x_sim[i+1] = model.integrate(x0=x_sim[i], u=[], p=[])
        x_sim2[i+1] = x_sim2[i] + model.sampling_time() * np.array([
                x_sim2[i,1], -1.0 * x_sim2[i,0]])

    plt.figure()
    ax = plt.subplot(111)
    ax.plot(x_sim[:,0], x_sim[:,1], 'k-', linewidth=1.0, label='Exact')
    ax.plot(x_sim2[:,0], x_sim2[:,1], 'b-', linewidth=1.0, label='Euler')
    ax.set_ylabel('y')
    ax.set_xlabel('x')
    plt.legend(loc='best')
    plt.show()


def ode(x, u, z, p):
    """ Harmonic oscillator."""
    k = 1.0
    dxdt = [
            x[1],
            -k * x[0]
    ]
    return ca.vertcat(*dxdt)


if __name__ == "__main__":

    # Number of states.
    Nx = 2 

    # Number of inputs.
    Nu = 0

    # Covariance matrix of added noise.
    R_n = np.eye(Nx) * 1e-6
    dt = 0.01
    model = Model(Nx=Nx, Nu=Nu, ode=ode, dt=dt, R=R_n, clip_negative=True)
    plot_van_der_pol()
