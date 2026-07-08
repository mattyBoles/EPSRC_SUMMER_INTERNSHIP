import torch


class tanh_model(torch.nn.Module):
    def __init__(self, hidden_units):
        super().__init__()

        self.tanh = torch.tanh
        self.linear1 = torch.nn.Linear(in_features=3, out_features=hidden_units)
        self.linear2 = torch.nn.Linear(in_features=hidden_units, out_features=3)

    def forward(self, x):

        x = self.linear1(x)
        x = self.tanh(x)
        x = self.linear2(x)

        return x
    


class avg_euclidean_error(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, target):
        error = torch.linalg.norm(pred - target, axis = 1)
        return torch.mean(error)