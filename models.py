import torch
from torch import nn


class tanh_model(torch.nn.Module):
    def __init__(self, hidden_units, activation, RANDOM_SEED=None, beta = 1):
        super().__init__()
        if RANDOM_SEED is not None:
            torch.manual_seed(RANDOM_SEED)

        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = torch.relu
        elif activation == 'arctan':
            self.activation = torch.arctan
        elif activation == 'softplus':
            self.activation = lambda x: torch.nn.functional.softplus(x, beta=beta)
            
        self.linear1 = torch.nn.Linear(in_features=3, out_features=hidden_units)
        self.linear2 = torch.nn.Linear(in_features=hidden_units, out_features=3)


    def forward(self, x):

        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)

        return x
    


class avg_euclidean_error(torch.nn.Module):
    def __init__(self,
                 mean,
                 std):
        super().__init__()
        self.mean = mean
        self.std = std
    def forward(self, pred, target):
      pred = (pred * self.std.to(pred.device)) + self.mean.to(pred.device)
      target = (target * self.std.to(pred.device)) + self.mean.to(pred.device)
      error = torch.linalg.norm(pred - target, axis = 1)
      return torch.mean(error)
