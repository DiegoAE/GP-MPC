from sys import path

import casadi as ca
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms


# Add gp_mpc pagkage to path
path.append(r"./../")

from gp_mpc import Model, GP


def plot_harmonic_oscillator(time_steps, dt, gp, model):
    plt.figure()
    ax = plt.subplot(111)

    # The initial state is Gaussian distributed with the following moments:
    mu = np.array([2., .201])
    Sigma = np.eye(2) * 0.1
    mean_traj = np.zeros((time_steps, 2))

    # Setting prediction to use moment matching.
    gp.set_method('EM')
    approx_mu = np.copy(mu)
    approx_Sigma = np.copy(Sigma)
    approx_mean_traj = np.zeros((time_steps, 2))
    stdY = np.diag(gp._GP__stdY)  # needed to unnormalized predicted covariance.

    # Linear dynamics of harmonic oscillator:
    k = 2.0
    A = np.eye(2) + np.array([[0,1],[-k,0]]) * dt

    for t in range(time_steps):
        #confidence_ellipse(ax, mu, Sigma)
        #mu = A @ mu
        #Sigma = A @ Sigma @ A.T
        mu = model.integrate(x0=mu, u=[], p=[])
        mean_traj[t, :] = mu

    confidence_ellipse(ax, np.array(approx_mu),
            stdY @ np.array(approx_Sigma) @ stdY)
    for t in range(time_steps):

        # Note that the mean is unnormalized but the cov isn't.
        pred_mu, pred_Sigma = gp.predict(approx_mu, [], approx_Sigma)
        approx_mu = pred_mu
        approx_Sigma = pred_Sigma

        approx_mean_traj[t, :] = np.array(approx_mu).flatten()

    confidence_ellipse(ax, np.array(approx_mu),
            stdY @ np.array(approx_Sigma) @ stdY)

    ax.plot(mean_traj[:,0], mean_traj[:,1], 'k-', linewidth=1.0, label='GT')
    ax.plot(approx_mean_traj[:,0], approx_mean_traj[:,1], 'b-', linewidth=1.0,
            label='MM')
    ax.set_ylabel('x2')
    ax.set_xlabel('x1')
    plt.legend(loc='best')
    plt.show()


def ode(x, u, z, p):
    """ Harmonic oscillator."""
    k = 10
    dxdt = [
            x[1],
            -k * x[0]
    ]
    return ca.vertcat(*dxdt)


def confidence_ellipse(ax, mu, Sigma, n_std=2):
    lambda_, v = np.linalg.eig(Sigma)
    lambda_ = np.sqrt(lambda_)
    ellipse = Ellipse(mu, width=lambda_[0]*n_std*2, height=lambda_[1]*n_std*2,
            angle=np.rad2deg(np.arccos(v[0, 0])), facecolor='grey')
    return ax.add_artist(ellipse)


if __name__ == "__main__":

    """ System Parameters """
    dt = .01                    # Sampling time
    Nx = 2                      # Number of states
    Nu = 0                      # Number of inputs
    R_n = np.eye(Nx) * 1e-6     # Covariance matrix of added noise

    # Limits in the training data
    ulb = []    # No inputs are used
    uub = []    # No inputs are used
    xlb = [-4., -6.]
    xub = [4., 6.]

    N = 40          # Number of training data
    N_test = 100    # Number of test data

    """ Create simulation model and generate training/test data"""
    model          = Model(Nx=Nx, Nu=Nu, ode=ode, dt=dt, R=R_n, clip_negative=True)
    X, Y           = model.generate_training_data(N, uub, ulb, xub, xlb, noise=True)
    X_test, Y_test = model.generate_training_data(N_test, uub, ulb, xub, xlb, noise=True)

    """ Create GP model and optimize hyper-parameters"""
    gp = GP(X, Y, mean_func='zero', normalize=True, xlb=xlb, xub=xub, ulb=ulb,
            uub=uub, optimizer_opts=None)
    gp.validate(X_test, Y_test)
    plot_harmonic_oscillator(200, dt, gp, model)
