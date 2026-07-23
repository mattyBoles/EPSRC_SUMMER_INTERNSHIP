import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path

from models import tanh_model
from lorenz import LorenzGenerator

np.printoptions(suppress=True)

def animated_fig_plot(MODEL_NAME: str,
                      hidden_size: int,
                      activation: str,
                      x: np.ndarray) -> None:
    stats = torch.load(Path(r".\output", MODEL_NAME, MODEL_NAME + "_stats.pt"))
    mean, std = stats['mean'].detach().numpy(), stats['std'].detach().numpy()

    model = tanh_model(hidden_size, activation)
    model.load_state_dict(torch.load(Path(r".\output", MODEL_NAME, MODEL_NAME + "_best_epoch.pth")))

    W1 = model.linear1.weight.detach().numpy()
    b1 = model.linear1.bias.detach().numpy().reshape(-1,1)
    
    W2 = model.linear2.weight.detach().numpy()
    b2 = model.linear2.bias.detach().numpy().reshape(-1,1)

    x_model = ((x - mean) / std)
    x_model_l = []
    sv_true_l = []
    sv_fixed_l = []
    sch_true_l = [] #h[j] activation
    sch_fixed_l = []
    t = []


    fig = plt.figure()

    ax1= fig.add_subplot(311, projection='3d')
    ax1.set_xlim([-20, 20])
    ax1.set_ylim([-20, 20])
    ax1.set_zlim([0,50])
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    ax2 = fig.add_subplot(312) #SV3
    ax2.set_xlabel('Timestep, dt = 0.01')
    ax2.set_ylabel('SV3')

    ax3 = fig.add_subplot(313) #SV3
    ax3.set_xlabel('Timestep, dt = 0.01')
    ax3.set_ylabel('h[j] - neuron activation')


    line1, = ax1.plot([], [], [], lw=0.8)
    line2_1, = ax2.plot([], [], lw=0.8) #Original Sv3
    line2_2, = ax2.plot([], [], lw=0.8) # Fixed sv3
    line3_1, = ax3.plot([], [], lw=0.8) #Original h[j] activation
    line3_2, = ax3.plot([], [], lw=0.8) # Fixed h[j] activation

    c = LorenzGenerator()

    sch_fixed = None
    neuron_j = 13
    count = 0

    plt.ion()
    while plt.fignum_exists(fig.number):
        count +=1
        t.append(count)

        z = W1 @ x_model.reshape(-1,1) + b1

        a = np.tanh(z)#activated
        sch = sch2(z)#derivative of neurons
        sch_fixed = sch.copy()

        if len(t) > 0:
            if 640 < t[-1] < 669:
                sch_fixed[neuron_j] = sch_fixed_l[-1]
        
        x_model = (W2 @ a) + b2


        J_fixed = (W2 @ np.diag(sch_fixed.squeeze()) @ W1)
        sv_fixed = np.linalg.svd(J_fixed, compute_uv=False)
        sv_fixed_l.append(sv_fixed[2])
        sch_fixed_l.append(sch_fixed[neuron_j].item())
        
        J_real = (W2 @ np.diag(sch.squeeze()) @ W1)
        sv_true = np.linalg.svd(J_real, compute_uv=False)
        sv_true_l.append(sv_true[2])
        sch_true_l.append(sch[neuron_j].item())


        x_model_l.append(((x_model.squeeze() * std) + mean))

        if len(x_model_l) > 2500:
            x_model_l = x_model_l[-2500:]
        
        if len(sv_true_l) > 1000:
            sv_true_l = sv_true_l[-1000:]
            sv_fixed_l = sv_fixed_l[-1000:]
            sch_true_l = sch_true_l[-1000:]
            sch_fixed_l = sch_fixed_l[-1000:]
            t = t[-1000:]
        
        if count > 550:
        
            x_model_arr = np.asarray(x_model_l)
            line1.set_data(x_model_arr[:,0], x_model_arr[:,1])
            line1.set_3d_properties(x_model_arr[:,2])

            sv_true_arr = np.asarray(sv_true_l)
            sv_fixed_arr = np.asarray(sv_fixed_l)
            line2_1.set_data(t, sv_true_arr)
            line2_1.set_label('True Model SV3')
            line2_2.set_data(t, sv_fixed_arr)
            line2_2.set_label('Fixed Model SV3')

            sch_true_arr = np.asarray(sch_true_l)
            sch_fixed_arr = np.asarray(sch_fixed_l)
            line3_1.set_data(t, sch_true_arr)
            line3_1.set_label('True Model N_j Derivative')
            line3_2.set_data(t, sch_fixed_arr)
            line3_2.set_label('Fixed Model N_j Derivative')

            if count % 40 == 0:
                ax2.set_xlim([t[-1] - 750, t[-1] + 250])
                ax2.relim()           # recompute limits from data
                ax2.autoscale_view()
                ax2.legend()
                ax3.set_xlim([t[-1] - 750, t[-1] + 250])
                ax3.relim()           # recompute limits from data
                ax3.autoscale_view()
                ax3.legend()
            
            plt.pause(0.05)
    plt.close('all')

def sch2(x):
    return 1/ (np.cosh(x) * np.cosh(x))


def animated_true(x:np.ndarray):
    x_l = []
    sv_l = []
    t = []


    fig = plt.figure()

    ax1= fig.add_subplot(211, projection='3d')
    ax1.set_xlim([-20, 20])
    ax1.set_ylim([-20, 20])
    ax1.set_zlim([0,50])
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    ax2 = fig.add_subplot(212) #SV3
    ax2.set_xlabel('Timestep, dt = 0.01')
    ax2.set_ylabel('SV')

    line1, = ax1.plot([], [], [], lw=0.8)
    line2_1, = ax2.plot([], [], lw=0.8) #SV1
    line2_2, = ax2.plot([], [], lw=0.8) #SV2
    line2_3, = ax2.plot([], [], lw=0.8) #SV3

    c = LorenzGenerator()

    count = 0

    Q = np.eye(3)

    x_l.append(x)

    plt.ion()
    while plt.fignum_exists(fig.number):
        count +=1
        t.append(count)

        Q = np.eye(3)

        x, U = c.rk4_matrix_and_x(f=c.calc_derivatives, J=c.J, x=x, U=Q, h=0.01)

        SV = np.linalg.svd(U, compute_uv=False)

        sv_l.append(SV)

        print(U)

        x_l.append(x)

        if len(x_l) > 2500:
            x_l = x_l[-2500:]
        
        if len(sv_l) > 1000:
            sv_l = sv_l[-1000:]
            t = t[-1000:]

        x_arr = np.asarray(x_l)
        line1.set_data(x_arr[:,0], x_arr[:,1])
        line1.set_3d_properties(x_arr[:,2])

        sv_arr = np.asarray(sv_l)

        line2_1.set_data(t, sv_arr[:,0])
        line2_1.set_label('SV1')
        
        line2_2.set_data(t, sv_arr[:,1])
        line2_2.set_label('SV2')

        line2_3.set_data(t, sv_arr[:,2])
        line2_3.set_label('SV3')

        if count % 50 == 0:
            ax2.set_xlim([t[-1] - 750, t[-1] + 250])
            ax2.relim()           # recompute limits from data
            ax2.autoscale_view()
            ax2.legend()
        
        plt.pause(0.01)

    plt.close('all')





if __name__ == "__main__":

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    animated_fig_plot(MODEL_NAME="2026-07-11T13-34-10_tanh_16",
                       hidden_size = 16,
                       activation = "tanh",
                       x = np.array([[1,1,0]])
                       )
    
    # animated_true(x=np.array([5,5,8]))

