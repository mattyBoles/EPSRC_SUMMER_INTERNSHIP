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
    epoch_preds = []

    n_batches = len(dataloader)

    for idx, (inputs, targets) in enumerate(dataloader):
        
        targets = targets.float().to(device)
        inputs = inputs.float().to(device)

        def closure():
            optimiser.zero_grad()

            preds = model(inputs)
    
            err = preds - targets
            MSE = torch.mean(err**2, dim=0)*std.squeeze()**2
            loss = torch.mean(MSE)
            loss.backward()
            return loss

        loss = optimiser.step(closure)

    
        epoch_loss += loss.item() 
        epoch_preds.append(model(inputs))

    epoch_preds = torch.cat(epoch_preds, dim=0)
    acc = acc_fn(epoch_preds, targets)
    epoch_acc += acc.item()

    epoch_loss /= n_batches
    epoch_acc /= n_batches

    return epoch_loss, epoch_acc

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
    epoch_preds = []

    n_batches = len(dataloader)
    with torch.inference_mode():
        for idx, (inputs, targets) in enumerate(dataloader):
            
            targets = targets.float().to(device)
            inputs = inputs.float().to(device)

            def closure():

                preds = model(inputs)
        
                err = preds - targets
                MSE = torch.mean(err**2, dim=0)*std.squeeze()**2
                loss = torch.mean(MSE)
                return loss
           
            loss = closure()
           
               
            epoch_loss += loss.item() 
            epoch_preds.append(model(inputs))
           
        epoch_preds = torch.cat(epoch_preds, dim=0)
        acc = acc_fn(epoch_preds, targets)
        epoch_acc += acc.item()
           
        epoch_loss /= n_batches
        epoch_acc /= n_batches
           
        return epoch_loss, epoch_acc


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
        train_loss, train_acc = train_epoch(model = model,
                                            dataloader = train_loader,
                                            loss_fn = loss_fn,
                                            optimiser = optimiser,
                                            acc_fn = acc_fn,
                                            std = std,
                                            device = device)
        
        val_loss, val_acc = val_epoch(model = model,
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

        if epoch % 10 == 0:
            print(f'| Epoch {epoch} |\n| Train Loss : {train_loss} | Train Average Euclidean Distance: {train_acc} |\n| Val Loss : {val_loss} | Val Average Euclidean Distance: {val_acc} |')
            print(model.linear1.weight)

    return results    
    

def test(model,
         dataloader,
         loss_fn,
         acc_fn,
         std,
         device):
    
    model = model.to(device)

    model.eval()
    epoch_preds = []

    epoch_acc = 0.0
    epoch_loss = 0.0
    n_batches = len(dataloader)
    with torch.inference_mode():
        for idx, (inputs, targets) in enumerate(dataloader):
            
            targets = targets.float().to(device)
            inputs = inputs.float().to(device)

            def closure():
            
                preds = model(inputs)
                    
                err = preds - targets
                MSE = torch.mean(err**2, dim=0)*std.squeeze()**2
                loss = torch.mean(MSE)
                return loss
                       
            loss = closure()
                

            epoch_loss += loss.item() 
            epoch_preds.append(model(inputs))
           
        epoch_preds = torch.cat(epoch_preds, dim=0)
        acc = acc_fn(epoch_preds, targets)
        epoch_acc += acc.item()
           
        epoch_loss /= n_batches
        epoch_acc /= n_batches
           
        return epoch_loss, epoch_acc



