import torch
import numpy as np
import matplotlib.pyplot as plt

from models import tanh_model
from lorenz import LorenzGenerator


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

mean, std = torch.load(r'.\output\2026-07-11T13-34-10_tanh_16\2026-07-11T13-34-10_tanh_16_stats.pt')['mean'].to(device), torch.load(r'.\output\2026-07-11T13-34-10_tanh_16\2026-07-11T13-34-10_tanh_16_stats.pt')['std'].to(device)

model = tanh_model(16, activation='tanh')
model.load_state_dict(torch.load(r'.\output\2026-07-11T13-34-10_tanh_16\2026-07-11T13-34-10_tanh_16_model.pth'))
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

lambda1_list, lambda2_list, lambda3_list = [],[],[]
for i in range(50):
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

        if i % t == 0:
                    
            Q, R = np.linalg.qr(Q)
            lambda_ = np.hstack([lambda_, np.array([[np.log(np.abs(R[0,0])+1e-10)/(h*t)],
                        [np.log(np.abs(R[1,1])+1e-10)/(h*t)],
                        [np.log(np.abs(R[2,2])+1e-10)/(h*t)]])])

        x = x_

    Lyapunov_spectrum = np.mean(lambda_, axis=1)
    lambda1_list.append(Lyapunov_spectrum[0])
    lambda2_list.append(Lyapunov_spectrum[1])
    lambda3_list.append(Lyapunov_spectrum[2])
    print(f"Lambda1: {Lyapunov_spectrum[0]}\nLambda2: {Lyapunov_spectrum[1]}\nLambda3: {Lyapunov_spectrum[2]}\n")
print('\n\n\n')
print(f"Lambda1: {np.mean(np.array(lambda1_list))}\nLambda2: {np.mean(np.array(lambda2_list))}\nLambda3: {np.mean(np.array(lambda3_list))}\n")
# fig, axes = plt.subplots(2,1)
# singular_values = np.array(singular_values)
# zdot = np.array(zdot)
# # axes[0].plot(zdot[150:1650], label='zdot')

# axes[0].plot(singular_values[:1000,0], label = 'sigma_1')
# axes[0].plot(singular_values[:1000,1], label = 'sigma_2')
# axes[0].plot(singular_values[:1000,2], label = 'sigma_3')

# c = LorenzGenerator()
# singular_values_real = c.find_lyapunov_spectrum(x = np.array([1,1,0]))['singular_values']
# #zdot_real = c.find_lyapunov_spectrum(x = np.array([1,1,0]))['zdot']
# #axes[1].plot(zdot_real[:1500], label='zdot')

# singular_values_real = np.array(singular_values_real)

# axes[1].plot(singular_values_real[:1000,0], label = 'sigma_1')
# axes[1].plot(singular_values_real[:1000,1], label = 'sigma_2')
# axes[1].plot(singular_values_real[:1000,2], label = 'sigma_3')
# plt.show()
