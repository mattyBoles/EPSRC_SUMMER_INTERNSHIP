import matplotlib.pyplot as plt
import numpy as np
import torch
from lorenz import LorenzGenerator

def plot_model(model,
               x0,
               n_steps,
               mean,
               std):
    
    model = model.to('cpu')

    generator = LorenzGenerator()
    model.eval()

    x_model = []
    x_model.append(x0)

    x = ((torch.tensor(x0) - mean) / std).float()
    
    with torch.inference_mode():
        for i in range(n_steps):
            x = model(x)
            x_model.append(np.array((x * std) + mean))

    x_true = generator.generate_trajectory(x0 = x0,
                                           n_steps = n_steps,
                                           h = 0.01)
    
    generator.plot(two_traj=False, traj1=x_true, traj2 = None, png_name = './output/REAL_LORENZ.png')
    generator.plot(two_traj=False, traj1=np.array(x_model), traj2 = None, png_name = './output/MODEL_TRAJ.png')



