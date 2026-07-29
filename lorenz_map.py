import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from pathlib import Path
from lorenz import LorenzGenerator
from scipy.optimize import curve_fit
from models import tanh_model
import json


def exp_model(z, a, b, c):
    return a * np.exp(b * z) + c
x0 = np.array([2,4,9])

local_max = []

c = LorenzGenerator()

n_transient = 5000
n_steps = 1000
x = x0

traj = c.generate_trajectory(x0=x, n_steps = 15000, h=0.01)

z = traj[1000:,2]
peak_indices, _ = find_peaks(z, distance=20)  # distance prevents false peaks

z_maxima = z[peak_indices]

zn, zn1 = z_maxima[:-1], z_maxima[1:]

# sort by zn to find where the map peaks
order = np.argsort(zn)
zn_sorted, zn1_sorted = zn[order], zn1[order]
peak_idx = np.argmax(zn1_sorted)   # location of the map's peak along x

mask_lower = zn <= zn_sorted[peak_idx]   # rising branch (small z_n)
mask_upper = zn >  zn_sorted[peak_idx]   # falling branch (large z_n)

p0_low = [1, 0.3, 30]
popt_low, _ = curve_fit(exp_model, zn[mask_lower], zn1[mask_lower], p0=p0_low, maxfev=10000)

# falling branch: z_n+1 decreases as z_n increases past the peak, so b < 0
p0_high = [1, -0.3, 30]
popt_high, _ = curve_fit(exp_model, zn[mask_upper], zn1[mask_upper], p0=p0_high, maxfev=10000)

z_range_low  = np.linspace(zn[mask_lower].min(), zn[mask_lower].max(), 500)
z_range_high = np.linspace(zn[mask_upper].min(), zn[mask_upper].max(), 500)

coeffs_left  = np.polyfit(zn[mask_lower],  zn1[mask_lower],  6)
coeffs_right = np.polyfit(zn[mask_upper],  zn1[mask_upper],  6)


func_left  = np.poly1d(coeffs_left)
func_right = np.poly1d(coeffs_right)

# then just call them like normal functions
y_left  = func_left(z_range_low)
y_right = func_right(z_range_high)
fig, ax = plt.subplots(1,1)

ax.scatter(z_maxima[:-1], z_maxima[1:])

ax.plot(z_range_low,  y_left,  color='red',   linewidth=2)
ax.plot(z_range_high, y_right, color='green', linewidth=2)

print(np.argmax(z_maxima))

ax.set_xlabel('Z_n')
ax.set_ylabel('Z_n+1')
plt.show()

plt.close('all')

MODEL_NAME = '2026-07-20T12-42-24tanh1'

stats = torch.load(Path(r".\output", MODEL_NAME, MODEL_NAME + "_stats.pt"))
mean, std = stats['mean'], stats['std']
mean = mean.detach().numpy()
std = std.detach().numpy()

with open(Path(r'.\output', MODEL_NAME, MODEL_NAME+'_train.json'), 'r') as f:
            model_info = json.load(f)
            activation = model_info['ACTIVATION']
            hidden_size = model_info['HIDDEN_SIZE']

model = tanh_model(hidden_size, activation)
model.load_state_dict(torch.load(Path(r".\output", MODEL_NAME, MODEL_NAME + "_best_epoch.pth")))

W1 = model.linear1.weight.detach().numpy()
b1 = model.linear1.bias

W2 = model.linear2.weight.detach().numpy()
b2 = model.linear2.bias

x_model_traj = []

x_model = ((x0 - mean) / std)

for _ in range(15000):
    x_model = W1 @ x_model.reshape(-1,1) + b1.reshape(-1,1).detach().numpy()
    x_model = np.tanh(x_model)
    
    x_model =W2@(x_model)+ b2.reshape(-1,1).detach().numpy()
    
    x_model_traj.append(((x_model[:,0]*std)+mean))

z_model = np.asarray(x_model_traj)[1000:,2]
peak_indices, _ = find_peaks(z_model, distance=20)  # distance prevents false peaks

z_maxima = z_model[peak_indices]

zn, zn1 = z_maxima[:-1], z_maxima[1:]

fig, ax = plt.subplots()

ax.scatter(z_maxima[:-1], z_maxima[1:])

ax.plot(z_range_low,  y_left,  color='red',   linewidth=2)
ax.plot(z_range_high, y_right, color='green', linewidth=2)

z_range_low  = np.linspace(zn.min(), zn.max(), 500)
# ax.plot(z_range_low, z_range_low)

print(np.argmax(z_maxima))

ax.set_xlabel('Z_n')
ax.set_ylabel('Z_n+1')
ax.set_title('Return Map of Model, Compared with the True System')
plt.show()