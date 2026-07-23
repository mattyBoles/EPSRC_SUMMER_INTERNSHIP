import numpy as np
import torch
import pandas as pd
import random
import os
import json
from datetime import datetime, timezone

from data import traj_Dataset
from models import tanh_model, avg_euclidean_error
from engine import train, test
from plot import plot_model
from model_analysis import analysis


def train_model(config:dict) -> tuple[str, float, float, float]:
    '''
    Trains a model from scratch and does some quick lyapunov analysis on it. Also creates a best_model (based on lowest val loss) and last_model .pth to be loaded,
    stats.pt, which contain the mean and std of the train set for inferance, _MODEL_TRAJ, showing a typical trajectory,
    and a trin json, shwoing loss, Avg EUclidean distance, hyperparameters, and Lyapunov spectrum.

    Inputs:
        config (dict):
            'MODEL_NAME' (str): The name the model will be saved to.
            'NUM_EPOCHS (int): The number of epochs to run for.
            'hidden_size' (int): The width of the hidden layer.
            'n_traj' (int): The numebr of different trajectories to train on.
            'traj_length' (int): The number of consecutive points on each trajectory to train on.
            'activation' (str): The activation to train on, namely 'relu', 'tanh', 'arctan', or 'softplus'.
            'beta' (float): The beta parameter in softplus.
            'random_seed' (int): The random seed to use for data geenration.

    Returns:
        MODEL_NAME (str): The folder that the model was saved to.
        l1 (float): The first Lyapunov exponent.
        l2 (float): The second Lyapunov exponent.
        l3 (float): The third Lyapunov exponent.
    '''
    device = 'cuda:0' if torch.cuda.is_available() == True else 'cpu'
    print(device)

    RANDOM_SEED = config['random_seed']

    #REPRODUCABILITY
    torch.manual_seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.use_deterministic_algorithms(True)

    #MODEL_NAME = str(datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")) + '_6_width_model'
    MODEL_NAME = config['MODEL_NAME']

    output_dir = f'./output/{MODEL_NAME}/'
    os.makedirs(output_dir, exist_ok=True)

    n_trajectories = config['n_traj']
    n_samples_per_traj = config['traj_length']
    n_transient = 5000
    h = 0.01


    train_dataset = torch.load(r'.\dataset\train_dataset.pt')
    train_set = traj_Dataset(n_trajectories=n_trajectories,
                            n_samples_per_traj=n_samples_per_traj,
                            n_transient=n_transient,
                            h=h,
                            mean = None,
                            std = None,
                            RANDOM_SEED = RANDOM_SEED,
                            preloaded = train_dataset)
    

    mean = train_set.mean
    std = train_set.std


    val_dataset = torch.load(r'.\dataset\val_dataset.pt')
    val_set = traj_Dataset(n_trajectories=max(int(n_trajectories/8),4),
                            n_samples_per_traj=n_samples_per_traj,
                            n_transient=n_transient,
                            h=h,
                            mean = mean,
                            std = std,
                            RANDOM_SEED=RANDOM_SEED*10,
                            preloaded = val_dataset)
    
    test_dataset = torch.load(r'.\dataset\test_dataset.pt')
    test_set = traj_Dataset(n_trajectories=max(int(n_trajectories/8),4),
                            n_samples_per_traj=n_samples_per_traj,
                            n_transient=n_transient,
                            h=h,
                            mean = mean,
                            std = std,
                            RANDOM_SEED=RANDOM_SEED*100,
                            preloaded = test_dataset)


    BATCH_SIZE = 64
    lr = 0.001
    NUM_EPOCHS = config['NUM_EPOCHS']


    train_loader = torch.utils.data.DataLoader(train_set, batch_size = BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size = BATCH_SIZE, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size = BATCH_SIZE, shuffle=False)

    model = tanh_model(config['hidden_size'], config['activation'],RANDOM_SEED=RANDOM_SEED, beta = config['beta']).to(device)

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
                        std = std,
                        device = device)
    

    torch.save(model.state_dict(), f'{output_dir}/{MODEL_NAME}_last_epoch.pth')

    best_val_loss = train_results['val_loss'].index(min(train_results['val_loss']))
    model.load_state_dict(train_results['model_statedict'][best_val_loss])
    torch.save(model.state_dict(), f'{output_dir}/{MODEL_NAME}_best_epoch.pth')


    trn_loss, trn_acc, trn_MSE, trn_targets, trn_preds = test(model = model,
                                                    dataloader = train_loader,
                                                    loss_fn = loss_fn,
                                                    acc_fn = acc_fn,
                                                    std=std,
                                                    device = device)
    val_loss, val_acc, val_MSE, val_targets, val_preds = test(model = model,
                                                    dataloader = val_loader,
                                                    loss_fn = loss_fn,
                                                    acc_fn = acc_fn,
                                                    std=std,
                                                    device = device)
    test_loss, test_acc, test_MSE, test_targets, test_preds = test(model = model,
                                                    dataloader = test_loader,
                                                    loss_fn = loss_fn,
                                                    acc_fn = acc_fn,
                                                    std=std,
                                                    device = device)
    

    trn_MSE = [round(x.item(),4) for x in trn_MSE]
    val_MSE = [round(x.item(),4) for x in val_MSE]
    test_MSE = [round(x.item(),4) for x in test_MSE]

    print('\n\n')
    print('-----RESULTS-----')
    print(f'| Train MSE : {trn_MSE} | Train Average Euclidean Distance: {trn_acc} |\n')
    print(f'| Val MSE : {val_MSE} | Val Average Euclidean Distance: {val_acc} |\n')
    print(f'| Test MSE : {test_MSE} | Test Average Euclidean Distance: {test_acc} |\n')


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
    

    l1, l2, l3, _ = analysis(MODEL_NAME=MODEL_NAME, hidden_size=config['hidden_size'], activation = config['activation'], beta = config['beta'])

    json_output = {
            "MODEL_NAME":MODEL_NAME,
            "NUM_EPOCHS": config['NUM_EPOCHS'],
            "NUM_TRAJ": config['n_traj'],
            "TRAJ_LENGTH": config['traj_length'],
            "ACTIVATION": config['activation'],
            "HIDDEN_SIZE": config['hidden_size'],
            'BETA': config['beta'],
            "TRAIN_LOSS": to_py_float(trn_loss),
            "TRAIN_AVERAGE_EUCLIDEAN_DISTANCE": to_py_float(trn_acc),
            "VAL_LOSS" : to_py_float(val_loss),
            "VAL_AVERAGE_EUCLIDEAN_DISTANCE": to_py_float(val_acc),
            "TEST_LOSS" : to_py_float(test_loss),
            "TEST_AVERAGE_EUCLIDEAN_DISTANCE": to_py_float(test_acc),
            "Lyapunov1": l1,
            "Lyapunov2": l2,
            "Lyapunov3": l3,
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
    

    return MODEL_NAME, l1, l2, l3




if __name__ == '__main__':
    config = {
        "MODEL_NAME": 's',
        'NUM_EPOCHS': 200,
        'hidden_size': 16,
        'n_traj': 200,
        'traj_length': 1600,
        'activation': 'softplus',
        'beta': 1.5,
        'random_seed': 6
    }

    MODEL_NAME ,l1, l2, l3 = train_model(config=config)