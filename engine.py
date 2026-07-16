import torch
import numpy as np
from copy import deepcopy

def train_epoch(model,
                dataloader,
                loss_fn,
                optimiser,
                acc_fn,
                std,
                device):
    
    model = model.to(device)

    model.train()

    epoch_acc = 0
    epoch_loss = 0
    epoch_MSE = torch.tensor([0,0,0], dtype = torch.float64)

    n_batches = len(dataloader)

    for idx, (inputs, targets) in enumerate(dataloader):
        
      targets = targets.float().to(device)
      inputs = inputs.float().to(device)

      optimiser.zero_grad()

      preds = model(inputs)
    
      err = preds - targets
      MSE = torch.mean(err**2, dim=0)*std.squeeze()**2
      epoch_MSE += MSE
      loss = torch.mean(MSE)


    
      epoch_loss += loss.item() 
      
      acc = acc_fn(preds, targets)
      epoch_acc += acc.item()

      loss.backward()
      optimiser.step()

    epoch_loss /= n_batches
    epoch_acc /= n_batches
    epoch_MSE /= n_batches

    return epoch_loss, epoch_acc, epoch_MSE

def val_epoch(model,
                dataloader,
                loss_fn,
                acc_fn,
                std,
                device):
    
    model = model.to(device)

    model.eval()

    epoch_acc = 0
    epoch_loss = 0
    epoch_MSE = torch.tensor([0,0,0], dtype = torch.float64)

    n_batches = len(dataloader)
    with torch.inference_mode():
        for idx, (inputs, targets) in enumerate(dataloader):
            
            targets = targets.float().to(device)
            inputs = inputs.float().to(device)

            preds = model(inputs)
            
            err = preds - targets
            MSE = torch.mean(err**2, dim=0)*std.squeeze()**2

            epoch_MSE += MSE
            loss = torch.mean(MSE)
            epoch_loss += loss.item() 

            acc = acc_fn(preds, targets)
            epoch_acc += acc.item()


    epoch_loss /= n_batches
    epoch_acc /= n_batches
    epoch_MSE /= n_batches

    return epoch_loss, epoch_acc, epoch_MSE


def train(model,
          train_loader,
          val_loader,
          loss_fn,
          optimiser,
          acc_fn,
          NUM_EPOCHS,
          std,
          device):
    
    results = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'model_statedict': []
    }

    for epoch in range(1, NUM_EPOCHS+1):
        train_loss, train_acc, train_MSE = train_epoch(model = model,
                                            dataloader = train_loader,
                                            loss_fn = loss_fn,
                                            optimiser = optimiser,
                                            acc_fn = acc_fn,
                                            std = std,
                                            device = device)
        
        val_loss, val_acc, val_MSE = val_epoch(model = model,
                                      dataloader = val_loader,
                                      loss_fn = loss_fn,
                                      acc_fn = acc_fn,
                                      std = std,
                                      device = device)
        
        results['train_loss'].append(train_loss)
        results['train_acc'].append(train_acc)
        results['val_loss'].append(val_loss)
        results['val_acc'].append(val_acc)
        results['model_statedict'].append(deepcopy(model.state_dict()))

        train_MSE = [round(x.item(),4) for x in train_MSE]
        val_MSE = [round(x.item(),4) for x in val_MSE]

        if epoch % 10 == 0:
            print(f'| Epoch {epoch} |\n| Train MSE : {train_MSE} | Train Average Euclidean Distance: {train_acc} |\n| Val MSE : {val_MSE} | Val Average Euclidean Distance: {val_acc} |')
            print(model.linear2.weight)

    return results    
    

def test(model,
         dataloader,
         loss_fn,
         acc_fn,
         std,
         device):
    
    model = model.to(device)

    model.eval()

    total_acc = 0
    total_loss = 0
    total_MSE = torch.tensor([0,0,0], dtype = torch.float64)

    all_targets = []
    all_preds = []
    n_batches = len(dataloader)
    with torch.inference_mode():
        for idx, (inputs, targets) in enumerate(dataloader):
            
            targets = targets.float().to(device)
            inputs = inputs.float().to(device)

            preds = model(inputs)
                
            err = preds - targets
            MSE = torch.mean(err**2, dim=0)*std.squeeze()**2
            loss = torch.mean(MSE)
            total_loss += loss.item() 
            total_MSE += MSE

            acc = acc_fn(preds, targets)
            total_acc += acc.item()

            all_targets.append(targets)
            all_preds.append(preds)


    total_loss /= n_batches
    total_acc /= n_batches
    total_MSE /= n_batches

    return total_loss, total_acc, total_MSE, all_targets, all_preds