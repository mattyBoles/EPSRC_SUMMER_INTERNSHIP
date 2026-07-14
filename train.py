import numpy as np
import torch
import pandas as pd
import random
import os
import json
from datetime import datetime, timezone
from torch.optim.lr_scheduler import LinearLR,MultiStepLR

from data import traj_Dataset
from models import tanh_model, avg_euclidean_error
from engine import train, test
from plot import plot_model


config = {
    'NUM_EPOCHS': 2000,
    'hidden_size': 4,
    'n_traj': 80,
    'traj_length': 100,
    'activation': 'tanh'
}
def train_over_weekend(config):
    device = 'cuda:0' if torch.cuda.is_available() == True else 'cpu'
    print(device)

    RANDOM_SEED = 42

    #REPRODUCABILITY
    torch.manual_seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.use_deterministic_algorithms(True)

    MODEL_NAME = str(datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")) + '_' + f"{config['activation']}" + '_' + f"{str(config['hidden_size'])}"

    output_dir = f'./output/{MODEL_NAME}/'
    os.makedirs(output_dir, exist_ok=True)

    n_trajectories = config['n_traj']
    n_samples_per_traj = config['traj_length']
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
    NUM_EPOCHS = config['NUM_EPOCHS']


    train_loader = torch.utils.data.DataLoader(train_set, batch_size = 32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size = 32, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size = 32, shuffle=False)

    model = tanh_model(config['hidden_size'], config['activation']).to(device)

    loss_fn = torch.nn.MSELoss()
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0)

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

    torch.save(model.state_dict(), f'{output_dir}/{MODEL_NAME}_model.pth')
    torch.save(
        {"mean": mean, "std": std},
        f'{output_dir}/{MODEL_NAME}_stats.pt'
    )

    def to_py_float(x, dp: int = 4) -> float:
        '''
        Converts a torch.Tensor / np.generic / plain number to a native
        Python float, rounded to `dp` decimal places, for JSON serialization.
        '''
        if hasattr(x, "item"):  # torch.Tensor, np.generic
            x = x.item()
        return round(float(x), dp)

    json_output = {
            "MODEL_NAME":MODEL_NAME,
            "NUM_EPOCHS": config['NUM_EPOCHS'],
            "NUM_TRAJ": config['n_traj'],
            "TRAJ_LENGTH": config['traj_length'],
            "ACTIVATION": config['activation'],
            "HIDDEN_SIZE": config['hidden_size'],
            "TRAIN_LOSS": to_py_float(trn_loss),
            "TRAIN_AVERAGE_EUCLIDEAN_DISTANCE": to_py_float(trn_acc),
            "VAL_LOSS" : to_py_float(val_loss),
            "VAL_AVERAGE_EUCLIDEAN_DISTANCE": to_py_float(val_acc),
            "TEST_LOSS" : to_py_float(test_loss),
            "TEST_AVERAGE_EUCLIDEAN_DISTANCE": to_py_float(test_acc)
        }

    with open(output_dir + f"{MODEL_NAME}_train.json", "w") as f:
            json.dump(json_output, f, indent=2, default=str)


    plot_model(model = model,
            x0 = np.array([1,1,25]),
            n_steps = 10000,
            mean = mean,
            std = std,
            output_dir=output_dir,
            MODEL_NAME=MODEL_NAME)




if __name__ == '__main__':
     train_over_weekend(config = config)