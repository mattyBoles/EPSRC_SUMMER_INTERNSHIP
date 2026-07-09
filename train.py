import numpy as np
import torch
import pandas as pd
import random
import os

from data import traj_Dataset
from models import tanh_model, avg_euclidean_error
from engine import train, test
from plot import plot_model


device = 'cuda:0' if torch.cuda.is_available() == True else 'cpu'
print(device)

RANDOM_SEED = 0

#REPRODUCABILITY
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.use_deterministic_algorithms(True)

os.makedirs('./output/', exist_ok=True)

n_trajectories = 200
n_samples_per_traj = 1000
n_transient = 5000
h = 0.01


train_set = traj_Dataset(n_trajectories=n_trajectories,
                         n_samples_per_traj=n_samples_per_traj,
                         n_transient=n_transient,
                         h=h,
                         mean = None,
                         std = None)

mean = train_set.mean
std = train_set.std

val_set = traj_Dataset(n_trajectories=int(n_trajectories/8),
                         n_samples_per_traj=n_samples_per_traj,
                         n_transient=n_transient,
                         h=h,
                         mean = mean,
                         std = std)

test_set = traj_Dataset(n_trajectories=int(n_trajectories/8),
                         n_samples_per_traj=n_samples_per_traj,
                         n_transient=n_transient,
                         h=h,
                         mean = mean,
                         std = std)


BATCH_SIZE = 32
lr = 0.001
NUM_EPOCHS = 200


train_loader = torch.utils.data.DataLoader(train_set, batch_size = 32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size = 32, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_set, batch_size = 32, shuffle=False)

model = tanh_model(8).to(device)

loss_fn = torch.nn.MSELoss()
optimiser = torch.optim.AdamW(model.parameters(), lr=lr)

acc_fn = avg_euclidean_error(mean = mean,
                             std = std)

train_results = train(model = model,
                      train_loader = train_loader,
                      val_loader = val_loader,
                      loss_fn = loss_fn,
                      optimiser = optimiser,
                      acc_fn = acc_fn,
                      NUM_EPOCHS = NUM_EPOCHS,
                      device = device)

trn_loss, trn_acc, trn_targets, trn_preds = test(model = model,
                                                 dataloader = train_loader,
                                                 loss_fn = loss_fn,
                                                 acc_fn = acc_fn,
                                                 device = device)
val_loss, val_acc, val_targets, val_preds = test(model = model,
                                                 dataloader = val_loader,
                                                 loss_fn = loss_fn,
                                                 acc_fn = acc_fn,
                                                 device = device)
test_loss, test_acc, test_targets, test_preds = test(model = model,
                                                 dataloader = test_loader,
                                                 loss_fn = loss_fn,
                                                 acc_fn = acc_fn,
                                                 device = device)

print('\n\n')
print('-----RESULTS-----')
print(f'| Train Loss : {trn_loss:.6f} | Train Average Euclidean Distance: {trn_acc} |\n')
print(f'| Val Loss : {val_loss:.6f} | Val Average Euclidean Distance: {val_acc} |\n')
print(f'| Test Loss : {test_loss:.6f} | Test Average Euclidean Distance: {test_acc} |\n')

torch.save(model.state_dict(), './output/model.pth')

plot_model(model = model,
           x0 = np.array([1,1,25]),
           n_steps = 10000,
           mean = mean,
           std = std)

