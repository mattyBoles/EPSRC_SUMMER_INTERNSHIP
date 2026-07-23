import torch
import numpy as np
import matplotlib.pyplot as plt
import json

from models import tanh_model
from lorenz import LorenzGenerator
from pathlib import Path

def analysis(MODEL_NAME: str,
             beta: float,
             dt: float = 0.01,
             QR_steps: int = 10,
             transient_steps: int = 5000,
             trajectory_steps: int = 10000,
             device: torch.device = 'cuda:0' if torch.cuda.is_available() else 'cpu') -> tuple[float, float, float, list[np.ndarray]]:
    '''
    This function analyses a models local geometry (via SVD) and long term seperation (via Lyapnuov spectra).
    We integrate over a number of transient sets.
    Every timestep we update Q with J.We find the physical J by doing std * J * 1/std.
    We find singular values and add them to a list.
    Every so often we renormalise Q and add thr diagonal of R to the list of lambdas to be averaged.

    Inputs:
        MODEL_NAME
    '''
    
    root_folder = Path(r".\output", MODEL_NAME)
    mean, std = torch.load(Path(root_folder,MODEL_NAME+"_stats.pt"))['mean'].to(device), torch.load(Path(root_folder,MODEL_NAME+"_stats.pt"))['std'].to(device)
    with open(Path(root_folder, MODEL_NAME+'_train.json'), 'r') as f:
            model_info = json.load(f)
            activation = model_info['ACTIVATION']
            hidden_size = model_info['HIDDEN_SIZE']

    model = tanh_model(hidden_size, activation=activation, beta = beta)
    model.load_state_dict(torch.load(Path(root_folder,MODEL_NAME+"_best_epoch.pth")))
    model = model.to(device)

    '''
    PLAN:
    1. Start at x0
    2. Run transient — iterate x = model(x) for n_transient steps, no accumulation
    3. Set Q = identity(3), log_growth = zeros(3)
    4. For each step:
    - Compute J = autograd_jacobian(model, xn)
    - Evolve Q = J @ Q
    - Step x = model(xn)
    - Every 10 steps:
        - QR decompose: Q, R = qr(Q)
        - Accumulate: log_growth += log(abs(diag(R)))
        - Count: n_renorm += 1
    5. Lyapunov = log_growth / (n_renorm * 10 * h)
    '''

    Q = np.eye(3)
    lambda_ = np.empty((3,0))
    x = torch.tensor([[1,1,0]], dtype=torch.float32)
    x = (x - mean)/std
    singular_values, l_vectors, r_vectors, zdot = [], [], [], []
    sigma = std.numpy()

    Q = np.eye(3)
    lambda_ = np.empty((3,0))
    x = torch.tensor([[np.random.uniform(-20, 20), np.random.uniform(-20, 20), np.random.uniform(0,50)]])
    x = (x - mean)/std
    for i in range(transient_steps):
        x = x.float()
        x = model(x)

    for i in range(trajectory_steps):
        J = torch.autograd.functional.jacobian(model, x)
        J = J.squeeze().detach().numpy()

        x = x.float()
        x_ = model(x)
        zdot.append(((x_[0,2] - x[0,2]) / dt).detach())
        Q = J @ Q
        J_physical = np.diag(sigma) @ J @ np.diag(1/sigma)
        U, S, Vt = np.linalg.svd(J_physical)

        singular_values.append(S)
        l_vectors.append(U)
        r_vectors.append(Vt)

        if (i+1) % QR_steps == 0:
                    
            Q, R = np.linalg.qr(Q)
            lambda_ = np.hstack([lambda_, np.array([[np.log(np.abs(R[0,0])+1e-10)/(dt*QR_steps)],
                        [np.log(np.abs(R[1,1])+1e-10)/(dt*QR_steps)],
                        [np.log(np.abs(R[2,2])+1e-10)/(dt*QR_steps)]])])

        x = x_#for zdot

    Lyapunov_spectrum = np.mean(lambda_, axis=1)

    singular_values = np.array(singular_values)
    zdot = np.array(zdot)

    return Lyapunov_spectrum[0], Lyapunov_spectrum[1],Lyapunov_spectrum[2], singular_values

if __name__ == '__main__':
    lyapunov_11, lyapunov_21, lyapunov_31, sv1 = analysis(MODEL_NAME="2026-07-22T14-46-52n_traj_500", hidden_size=8, activation='tanh', beta=1)
    # lyapunov_12, lyapunov_22, lyapunov_32, sv2 = analysis(MODEL_NAME="2026-07-20T09-57-15tanh64", hidden_size=64, activation='tanh', beta=1.5)

    c = LorenzGenerator()
    singular_values_real = c.find_lyapunov_spectrum(x = np.array([1,1,0]))['singular_values']

    sv_real = np.array(singular_values_real)


    fig, axes = plt.subplots(2,1)

    axes[0].plot(sv1[:1000, 0], label = 'sigma1-softplus, 0.5')
    axes[0].plot(sv1[:1000, 1], label = 'sigma2-softplus, 0.5')
    axes[0].plot(sv1[:1000, 2], label = 'sigma3-softplus, 0.5')

    # axes[0].plot(sv1[:1000, 0], label = 'sigma1-softplus, 1')
    # axes[1].plot(sv1[:1000, 1], label = 'sigma2-softplus, 1')
    # axes[2].plot(sv1[:1000, 2], label = 'sigma3-softplus, 1')
    # axes[2].axhline(y=0.8, color='r', linestyle='--', label='y=0.8')
    # axes[2].axhline(y=0.75, color='g', linestyle='--', label='y=0.75')
    # axes[2].axhline(y=0.85, color='b', linestyle='--', label='y=0.85')

    axes[1].plot(sv_real[:1000, 0], label = 'sigma1-real')
    axes[1].plot(sv_real[:1000, 1], label = 'sigma2-real')
    axes[1].plot(sv_real[:1000, 2], label = 'sigma3-real')


    axes[0].legend()
    axes[1].legend()
    # axes[2].legend()

    plt.show()