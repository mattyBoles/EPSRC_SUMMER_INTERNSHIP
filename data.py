import pandas as pd
import torch
import numpy as np
from lorenz import LorenzGenerator
import random


RANDOM_SEED = 0

#REPRODUCABILITY
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.use_deterministic_algorithms(True)

class traj_Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 n_trajectories: int,
                 n_samples_per_traj: int,
                 n_transient: int,
                 h: int = 0.01):
        
        self.traj_generator = LorenzGenerator()

        self.n_trajectories = n_trajectories
        self.n_samples_per_traj = n_samples_per_traj
        self.n_transient = n_transient
        self.h = h

        self.samples, self.targets = self.generate_samples()

        print(f"Initialised Dataset:\n{self.n_trajectories} Trajectories \n{self.n_samples_per_traj} Samples per Trajectory\n{self.n_transient} Transient steps\nh = {self.h}")

    def generate_samples(self):
        samples = []
        targets = []

        for i in range(self.n_trajectories):

            x0 = np.array([np.random.uniform(-20, 20), np.random.uniform(-20, 20), np.random.uniform(0,50)])
            traj = self.traj_generator.generate_trajectory(x0 = x0,
                                                    n_steps = (self.n_samples_per_traj + self.n_transient),
                                                    h = self.h)
            traj = traj[self.n_transient+1:]

            samples.append(traj)

            last_target = self.traj_generator.rk4(self.traj_generator.calc_derivatives, x_ = traj[-1], h = self.h)

            targets.append(np.vstack([traj[1:], last_target]))
        
        samples = torch.flatten(torch.tensor(samples), start_dim=0, end_dim=1)
        targets = torch.flatten(torch.tensor(targets), start_dim=0, end_dim=1)
    
        return samples, targets 
        


    def __len__(self):
        return (self.n_samples_per_traj * self.n_trajectories)

    def __getitem__(self, idx):
        return self.samples[idx], self.targets[idx]
    
if __name__ == '__main__':
    dataset = traj_Dataset(n_trajectories=100, n_samples_per_traj=100, n_transient = 50)
    print('done')
