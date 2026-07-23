import matplotlib.pyplot as plt
import numpy as np
import torch
from collections import deque
from pathlib import Path

from models import tanh_model
from neuron_contraction_contribution import neuron_contraction_contribution
from lorenz import LorenzGenerator

np.set_printoptions(suppress=True)


def animated_fig_model(MODEL_NAME: str,
                       hidden_size: int,
                       activation: str,
                       x: torch.Tensor,
                       device):
    

    stats = torch.load(Path(r".\output", MODEL_NAME, MODEL_NAME + "_stats.pt"))
    mean, std = stats['mean'].to(device), stats['std'].to(device)
    mean = mean.detach().numpy()
    std = std.detach().numpy()

    model = tanh_model(hidden_size, activation).to(device)
    model.load_state_dict(torch.load(Path(r".\output", MODEL_NAME, MODEL_NAME + "_best_epoch.pth")))

    W1 = model.linear1.weight.detach().numpy()
    b1 = model.linear1.bias
    
    W2 = model.linear2.weight.detach().numpy()
    b2 = model.linear2.bias



    x_model = ((x - mean) / std).float()
    x_model = x_model.detach().numpy()

    x_model_list = []
    x_real_list = []
    sv_model = []
    sv_list = []
    t = []

    fig = plt.figure()
    ax1 = fig.add_subplot(311, projection = '3d') # Lorenz Attrctor

    ax1.set_xlim([-20,20])
    ax1.set_ylim([-20,20])
    ax1.set_zlim([0,50])

    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")

    ax2 = fig.add_subplot(312) # SV
    ax3 = fig.add_subplot(313) # SV

    line1, = ax1.plot([],[],[], lw=0.8)
    line2_1, = ax2.plot([],[], lw=0.8)
    line2_2, = ax2.plot([],[], lw=0.8)
    line2_3, = ax2.plot([],[], lw=0.8)

    line3_1, = ax3.plot([],[], lw=0.8)
    line3_2, = ax3.plot([],[], lw=0.8)
    line3_3, = ax3.plot([],[], lw=0.8)




    c = LorenzGenerator()

    x = x[0]
    h_list = []
    z_list = []
    delta_list = []
    v3_list = []
    

    h_old = None
    z_old = None
    plt.ion()
    neuron = 13
    count = 0
    while plt.fignum_exists(fig.number):
        count += 1

        # x_jac = x_model.detach().clone().requires_grad_(True)
        # J = torch.autograd.functional.jacobian(model, x_jac)
        # J = J.squeeze().detach().numpy()


        z = W1@x_model.reshape(3,1) + b1.reshape(-1,1).detach().numpy()
        h = d_dx_softplus(z,1.5)
        z_list.append(z[13])
        # if h_old is not None:
        #     h[13] = h_old
        #     z[13] = z_old
        J = (W2 @ np.diag(h.squeeze()) @ W1)

        
        h_list.append(h[13])




        U, SV, Vt= np.linalg.svd(J, compute_uv = True)
        sv_real = SV[2]
        

        sv_list.append(SV)
        
        try:
            if t[-1] == 600:
                h_old = 0.857
                z_old = -0.3965678
        except:
            print(' t = []')
        
        x_model = W1 @ x_model.reshape(-1,1) + b1.reshape(-1,1).detach().numpy()
        x_model = softplus(x_model, 1.5)
        
        x_model =W2@(x_model)+ b2.reshape(-1,1).detach().numpy()
        


        v3 = Vt[2, :]  # contracting direction at this point
        v3_list.append(v3)


            

        x_model_list.append(((x_model[:,0]*std)+mean))
        x_real_list.append(x)

        if len(x_model_list) > 2500:
            x_model_list = x_model_list[-2500:]
            x_real_list = x_real_list[-2500, :]

        # if len(sv_real) > 1000:
        #     sv_real = sv_real[-1000:]
        #     sv_model = sv_model[-1000:]
        #     t = t[-999:]
        
        t.append(count)
        if count == 1:
            continue
        x_model_arr = np.asarray(x_model_list).squeeze()
        line1.set_data(x_model_arr[:,0], x_model_arr[:,1])
        line1.set_3d_properties(x_model_arr[:,2])


        sv_arr = np.asarray(sv_list)
        h_arr = np.asarray(h_list)
        delta_arr = np.asarray(delta_list)
        z_arr = np.asarray(z_list)
        v3_arr = np.asarray(v3_list)
        # cj_arr1 = np.asarray(cj_list1)
        # cj_arr2 = np.asarray(cj_list2)
        # cj_arr3 = np.asarray(cj_list3)
        line2_1.set_data(t, sv_arr[:,0])
        line2_1.set_label('SV1')
        line2_2.set_data(t, sv_arr[:,1])
        line2_2.set_label('SV2')
        line2_3.set_data(t, sv_arr[:,2])
        line2_3.set_label('SV3')
        line3_1.set_data(t, v3_arr[:,0])
        line3_2.set_data(t, v3_arr[:,1])
        line3_3.set_data(t, v3_arr[:,2])
        # line3_1.set_label('h12 - contribution')
        # line3_2.set_data(t, cj_arr2)
        # line3_2.set_label('h13 - contribution')
        # line3_3.set_data(t, cj_arr3)
        # line3_3.set_label('h14 - contribution')
        # line3_2.set_data(t, h_arr[:,1])
        # line3_2.set_label('h10')
        # line3_3.set_data(t, h_arr[:,2])
        # line3_3.set_label('h11')



        if count % 50 == 0:
            ax2.set_xlim([t[-1] - 750, t[-1] + 250])
            ax2.relim()           # recompute limits from data
            ax2.autoscale_view()
            ax2.legend()
            ax3.set_xlim([t[-1] - 750, t[-1] + 250])
            ax3.relim()           # recompute limits from data
            ax3.autoscale_view()
            ax3.legend()
            # ax2.set_ylim([0.75, 1.15])
            # ax4.set_xlim([t[-1] - 750, t[-1] + 250])
            # ax4.set_ylim([0.75, 1.15])

        plt.pause(0.05)
    
    plt.close('all')

    

class sech_squared(torch.nn.Module):
    def __init__(self):
        super().__init__()#
    
    def forward(self, x):
        return 1/ (np.cosh(x) * np.cosh(x))
    

def softplus(x, beta):
    return (1/beta) * (np.log(1 + np.exp(beta*x)))

def d_dx_softplus(x, beta):
    return 1/(1+ np.exp(-1*beta*x))

if __name__ == "__main__":

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    animated_fig_model(MODEL_NAME="softplus_beta=0.75_model",
                       hidden_size = 16,
                       activation = "softplus",
                       x = torch.tensor([[1,1,0]]),
                       device = device
                       )