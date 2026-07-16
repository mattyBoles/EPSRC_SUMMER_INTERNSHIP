import matplotlib.pyplot as plt
import numpy as np
import torch
from collections import deque
from pathlib import Path

from models import tanh_model

np.set_printoptions(suppress=True)


def animated_fig_model(MODEL_NAME: str,
                       hidden_size: int,
                       activation: str,
                       x: torch.Tensor,
                       device):
    

    stats = torch.load(Path(r".\output", MODEL_NAME, MODEL_NAME + "_stats.pt"))
    mean, std = stats['mean'].to(device), stats['std'].to(device)

    model = tanh_model(hidden_size, activation).to(device)
    model.load_state_dict(torch.load(Path(r".\output", MODEL_NAME, MODEL_NAME + "_model.pth")))

    W1 = model.linear1.weight
    b1 = model.linear1.bias

    x = ((x - mean) / std).float()

    x_ = []
    sv = []
    t = []
    vt2 = []

    fig = plt.figure()
    ax1 = fig.add_subplot(311, projection = '3d') # Lorenz Attrctor

    ax1.set_xlim([-20,20])
    ax1.set_ylim([-20,20])
    ax1.set_zlim([0,50])

    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")

    ax2 = fig.add_subplot(312) # SV
    ax3 = fig.add_subplot(313)

    line1, = ax1.plot([],[],[], lw=0.8)
    line2_1, = ax2.plot([],[], lw=0.8)
    line2_2, = ax2.plot([],[], lw=0.8)
    line2_3, = ax2.plot([],[], lw=0.8)
    line3_1, = ax3.plot([],[], lw=0.8)
    line3_2, = ax3.plot([],[], lw=0.8)
    line3_3, = ax3.plot([],[], lw=0.8)


    sch_sqrd = sech_squared()

    z_old = 0
    h_old = 0
    J_old = 0
    SV_old = 0

    plt.ion()
    count = 0
    while plt.fignum_exists(fig.number):
        count += 1
        if count % 5 == 0:
            x_jac = x.detach().clone().requires_grad_(True)
            J = torch.autograd.functional.jacobian(model, x_jac)
            J = J.squeeze().detach().numpy()

            z = (W1@x.T + b1.reshape(32,1))
            z_abs = np.linalg.norm(z.detach().numpy())

            #print(f"|z|: {z_abs}")
            #print("Increased") if (z_abs > z_old) else print("Decreased")

            h = (z.detach().numpy() > 0).astype(int)
            #print(f"Avg sech^2(x): {h.mean()}")
            #print("Increased") if (h.mean() > h_old) else print("Decreased")

            ##print(f"|J|: {np.linalg.norm(J)}")
            print(h)
            #print("Increased") if (np.linalg.norm(J) > J_old) else print("Decreased")

            U, SV, Vt= np.linalg.svd(J, compute_uv = True)
            sv.append(SV)
            vt2.append(Vt[2,:])
            #print(f"SV3: {SV[2]}")
            #print(Vt[2,:])
            #print("Increased\n\n") if (SV[2] > SV_old) else print("Decreased\n\n")

            z_old = z_abs
            h_old = h.mean()
            J_old = np.linalg.norm(J)
            SV_old = SV[2]
            
        with torch.no_grad():




            #print(f"|z|: {np.linalg.norm(z)}")

            x = model(x)
            

        x_.append(((x*std)+mean).detach().numpy())

        if len(x_) > 2500:
            x_ = x_[-2500:]
        if len(sv) > 1000:
            sv = sv[-1000:]
            t = t[-999:]
        
        if count % 5 == 0:
            t.append(count)
            x_arr = np.asarray(x_).squeeze()
            line1.set_data(x_arr[:,0], x_arr[:,1])
            line1.set_3d_properties(x_arr[:,2])


            sv_arr = np.asarray(sv)
            vt2_arr = np.asarray(vt2)
            line2_1.set_data(t, sv_arr[:,0])
            line2_2.set_data(t, sv_arr[:,1])
            line2_3.set_data(t, sv_arr[:,2])
            line3_1.set_data(t, vt2_arr[:,0])
            line3_2.set_data(t, vt2_arr[:,1])
            line3_3.set_data(t, vt2_arr[:,2])

            if count % 50 == 0:
                ax2.set_xlim([t[-1] - 750, t[-1] + 250])
                ax2.set_ylim([0.75, 1.15])


                ax3.set_xlim([t[-1] - 750, t[-1] + 250])
                ax3.relim()           # recompute limits from data
                ax3.autoscale_view()  # apply them
            plt.draw()
            plt.pause(0.05)
    
    plt.close('all')

    

class sech_squared(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return 1/ (torch.cosh(x) * torch.cosh(x))
    




if __name__ == "__main__":

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    animated_fig_model(MODEL_NAME="2026-07-11T19-51-00_relu_32",
                       hidden_size = 32,
                       activation = "relu",
                       x = torch.tensor([[1,1,25]]),
                       device = device
                       )