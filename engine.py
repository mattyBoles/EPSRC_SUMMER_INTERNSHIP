import torch
import numpy as np

def train_epoch(model,
                dataloader,
                loss_fn,
                optimiser,
                acc_fn,
                device):
    
    model = model.to(device)

    model.train()

    epoch_acc = 0
    epoch_loss = 0

    n_batches = len(dataloader)

    for idx, (inputs, targets) in enumerate(dataloader):
        
      targets = targets.float().to(device)
      inputs = inputs.float().to(device)

      optimiser.zero_grad()

      preds = model(inputs)
        
      loss = loss_fn(preds, targets)
      epoch_loss += loss.item() 
      
      acc = acc_fn(preds, targets)
      epoch_acc += acc.item()

      loss.backward()
      optimiser.step()

    epoch_loss /= n_batches
    epoch_acc /= n_batches

    return epoch_loss, epoch_acc

def val_epoch(model,
                dataloader,
                loss_fn,
                acc_fn,
                device):
    
    model = model.to(device)

    model.eval()

    epoch_acc = 0
    epoch_loss = 0

    n_batches = len(dataloader)
    with torch.inference_mode():
        for idx, (inputs, targets) in enumerate(dataloader):
            
            targets = targets.float().to(device)
            inputs = inputs.float().to(device)

            preds = model(inputs)
                
            loss = loss_fn(preds, targets)
            epoch_loss += loss.item() 

            acc = acc_fn(preds, targets)
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
          device):
    
    results = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    for epoch in range(1, NUM_EPOCHS+1):
        train_loss, train_acc = train_epoch(model = model,
                                            dataloader = train_loader,
                                            loss_fn = loss_fn,
                                            optimiser = optimiser,
                                            acc_fn = acc_fn,
                                            device = device)
        
        val_loss, val_acc = val_epoch(model = model,
                                      dataloader = val_loader,
                                      loss_fn = loss_fn,
                                      acc_fn = acc_fn,
                                      device = device)
        
        results['train_loss'].append(train_loss)
        results['train_acc'].append(train_acc)
        results['val_loss'].append(val_loss)
        results['val_acc'].append(val_acc)

        if epoch % 10 == 0:
            print(f'| Epoch {epoch} |\n| Train Loss : {train_loss:.6f} | Train Average Euclidean Distance: {train_acc} |\n| Val Loss : {val_loss:.6f} | Val Average Euclidean Distance: {val_acc} |')

    return results    
    

def test(model,
         dataloader,
         loss_fn,
         acc_fn,
         device):
    
    model = model.to(device)

    model.eval()

    total_acc = 0
    total_loss = 0

    all_targets = []
    all_preds = []
    n_batches = len(dataloader)
    with torch.inference_mode():
        for idx, (inputs, targets) in enumerate(dataloader):
            
            targets = targets.float().to(device)
            inputs = inputs.float().to(device)

            preds = model(inputs)
                
            loss = loss_fn(preds, targets)
            total_loss += loss.item() 

            acc = acc_fn(preds, targets)
            total_acc += acc.item()

            all_targets.append(targets)
            all_preds.append(preds)


    total_loss /= n_batches
    total_acc /= n_batches

    return total_loss, total_acc, all_targets, all_preds