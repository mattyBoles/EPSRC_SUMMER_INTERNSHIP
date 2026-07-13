import matplotlib.pyplot as plt
import numpy as np
import torch
from lorenz import LorenzGenerator

def plot_model(model,
               x0,
               n_steps,
               mean,
               std,
               output_dir,
               MODEL_NAME):
    
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
    
    generator.plot(png_name = f'{output_dir}/{MODEL_NAME}_REAL_LORENZ.png', traj1=x_true, traj2 = None, )
    generator.plot(png_name = f'{output_dir}/{MODEL_NAME}_MODEL_TRAJ.png', traj1=np.array(x_model), traj2 = None)



