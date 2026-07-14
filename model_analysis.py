import torch
import numpy as np
import matplotlib.pyplot as plt

from models import tanh_model
from lorenz import LorenzGenerator
from pathlib import Path


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def analysis(root_folder, MODEL_NAME, hidden_size, activation):
    mean, std = torch.load(Path(root_folder,MODEL_NAME+"_stats.pt"))['mean'].to(device), torch.load(Path(root_folder,MODEL_NAME+"_stats.pt"))['std'].to(device)

    model = tanh_model(hidden_size, activation=activation)
    model.load_state_dict(torch.load(Path(root_folder,MODEL_NAME+"_model.pth")))
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

    n_transient = 5000

    n_steps = 10000

    t = 10
    h = 0.01

    Q = np.eye(3)
    lambda_ = np.empty((3,0))

    x = torch.tensor([[1,1,0]], dtype=torch.float32)

    x = (x - mean)/std

    singular_values, l_vectors, r_vectors, zdot = [], [], [], []


    sigma = std.numpy()

    # lambda1_list, lambda2_list, lambda3_list = [],[],[]
    # for i in range(20):
    Q = np.eye(3)
    lambda_ = np.empty((3,0))
    x = torch.tensor([[np.random.uniform(-20, 20), np.random.uniform(-20, 20), np.random.uniform(0,50)]])
    x = (x - mean)/std
    for i in range(n_transient):
        x = x.float()
        x = model(x)

    for i in range(n_steps):
        J = torch.autograd.functional.jacobian(model, x)
        J = J.squeeze().detach().numpy()

        x = x.float()
        x_ = model(x)
        zdot.append(((x_[0,2] - x[0,2]) / h).detach())
        Q = J @ Q
        J_physical = np.diag(sigma) @ J @ np.diag(1/sigma)
        U, S, Vt = np.linalg.svd(J_physical)

        singular_values.append(S)
        l_vectors.append(U)
        r_vectors.append(Vt)

        if (i+1) % t == 0:
                    
            Q, R = np.linalg.qr(Q)
            lambda_ = np.hstack([lambda_, np.array([[np.log(np.abs(R[0,0])+1e-10)/(h*t)],
                        [np.log(np.abs(R[1,1])+1e-10)/(h*t)],
                        [np.log(np.abs(R[2,2])+1e-10)/(h*t)]])])

        x = x_

    Lyapunov_spectrum = np.mean(lambda_, axis=1)

    fig, axes = plt.subplots(2,1)
    singular_values = np.array(singular_values)
    zdot = np.array(zdot)
    # axes[0].plot(zdot[150:1650], label='zdot')

    axes[0].plot(singular_values[:1000,0], label = 'sigma_1')
    axes[0].plot(singular_values[:1000,1], label = 'sigma_2')
    axes[0].plot(singular_values[:1000,2], label = 'sigma_3')
    axes[0].legend()
    axes[0].set_title('NN - Singular Values')
    axes[0].set_xlabel('timestep, h = 0.01')
    axes[0].set_ylabel('SV')

    c = LorenzGenerator()
    singular_values_real = c.find_lyapunov_spectrum(x = np.array([1,1,0]))['singular_values']
    # zdot_real = c.find_lyapunov_spectrum(x = np.array([1,1,0]))['zdot']
    # axes[1].plot(zdot_real[:1500], label='zdot')

    singular_values_real = np.array(singular_values_real)

    axes[1].plot(singular_values_real[:1000,0], label = 'sigma_1')
    axes[1].plot(singular_values_real[:1000,1], label = 'sigma_2')
    axes[1].plot(singular_values_real[:1000,2], label = 'sigma_3')
    axes[1].legend()
    axes[1].set_title('Lorenz - Discretised Propagator Singualr Values')
    axes[1].set_xlabel('timestep, h = 0.01')
    axes[1].set_ylabel('SV')
    plt.show()
    return Lyapunov_spectrum[0], Lyapunov_spectrum[1],Lyapunov_spectrum[2]
    # # lambda1_list.append(Lyapunov_spectrum[0])

# lyapunov_1, lyapunov_2, lyapunov_3 = analysis(root_folder=r'./output/2026-07-11T14-18-39_tanh_32', MODEL_NAME='2026-07-11T14-18-39_tanh_32', hidden_size = 32, activation='tanh')
# print(f"Lambda1: {lyapunov_1}\nLambda2: {lyapunov_2}\nLambda3: {lyapunov_3}\n")