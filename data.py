import pandas as pd
import torch
import numpy as np
from lorenz import LorenzGenerator
import random
from typing import Optional

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
                 h: int = 0.01, 
                 mean: Optional[torch.Tensor] = None,
                 std: Optional[torch.Tensor] = None,):
        
        self.traj_generator = LorenzGenerator()

        self.n_trajectories = n_trajectories
        self.n_samples_per_traj = n_samples_per_traj
        self.n_transient = n_transient
        self.h = h

        self.samples, self.targets = self.generate_samples()

        if mean == None:
            self.mean =torch.mean(self.samples, axis=0)
            self.std =torch.std(self.samples, axis=0)
        else:
            self.mean = mean
            self.std = std
        
        self.samples = (self.samples - self.mean) / self.std
        self.targets = (self.targets - self.mean) / self.std

        print(f"Initialised Dataset:\n{self.n_trajectories} Trajectories \n{self.n_samples_per_traj} Samples per Trajectory\n{self.n_transient} Transient steps\nh = {self.h}")

    def generate_samples(self):
        samples = np.empty((0,3))
        targets = np.empty((0,3))
        for i in range(self.n_trajectories):

            x0 = np.array([np.random.uniform(-20, 20), np.random.uniform(-20, 20), np.random.uniform(0,50)])
            inner_samples, inner_targets = find_points_near_fixed_points(x0, n_samples=(self.n_samples_per_traj / 4))
            traj = self.traj_generator.generate_trajectory(x0 = x0,
                                                    n_steps = (self.n_samples_per_traj + self.n_transient),
                                                    h = self.h)
            traj = traj[self.n_transient+1:]

            samples = np.vstack([samples, traj, inner_samples])

            last_target = self.traj_generator.rk4(self.traj_generator.calc_derivatives, x_ = traj[-1], h = self.h)

            targets = np.vstack([targets,np.vstack([traj[1:], last_target]), inner_targets])
        
        samples = torch.tensor(samples)
        targets = torch.tensor(targets)

        return samples, targets 
        


    def __len__(self):
        return (self.n_samples_per_traj * self.n_trajectories)

    def __getitem__(self, idx):

        return self.samples[idx], self.targets[idx]
    
if __name__ == '__main__':
    dataset = traj_Dataset(n_trajectories=100, n_samples_per_traj=100, n_transient = 50)
    print('done')



def find_points_near_fixed_points(x0,
                                  n_samples = 100):
    g = LorenzGenerator()

    fixed_points = [np.array([ 8.485,  8.485, 27   ]), np.array([ -8.485,  -8.485, 27   ])]

    for i in range(5000):
        x = g.generate_trajectory(x0, n_steps=5000, h=0.01)[-1]

        sampled_targets = []
        sampled_points = []

        while len(sampled_points) < n_samples:
            xout = g.rk4(g.calc_derivatives, x, 0.01)
            for fixed_point in fixed_points:
                dist = np.linalg.norm((xout - fixed_point))
                if dist < 6:
                    sampled_points.append(x)
                    sampled_targets.append(xout)
            x = xout

    return np.array(sampled_points), np.array(sampled_targets)


