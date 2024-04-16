import time
import numpy as np
import os
import torch
from torch import nn


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    """
    Evaluate the model on a given set.
    """
    model.eval()
    
    temp_loss = []
    y_pred = []
    y_true = []

    for data, target in loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = criterion(output, target)
        temp_loss.append(loss.item())
        y_pred.extend(output.argmax(dim=1).tolist())
        y_true.extend(target.tolist())
    
    temp_loss = np.mean(temp_loss)
    acc = (np.array(y_pred) == np.array(y_true)).mean()
    return temp_loss, acc

def save_model(model, optimizer, epoch, path, history):
    """
    Save model checkpoint.
    """
    # check if path exists 
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    

    checkpoint_dict = {
        'optimizer': optimizer.state_dict(),
        'model': model.state_dict(),
        'epoch': epoch,
        'history': history
    }
    torch.save(checkpoint_dict, path)

def load_model(path):
    """
    Load model checkpoint.
    """
    checkpoint = torch.load(path)
    return checkpoint['model'], checkpoint['optimizer'], checkpoint['epoch'], checkpoint['history']


def train(model, train_loader, test_loader, epochs, opt_func, 
          device, cfg, grad_clip=None, verbose=False, save_m=False):
    """
    Training cycle for the model.
    """
    torch.cuda.empty_cache()
    history = []

    # Define optimizer and loss function
    optimizer = opt_func

    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[35,57], gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    time_per_epoch = []
    epoch = 0

    best_loss = float('inf')
    best_model_weights = None
    best_epoch = 0

    for epoch in range(epochs):
        model.train()
        train_losses = []
        train_acc = []
        lrs = []
        start_time = time.time()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            # Forward pass
            output = model(data)
            loss = criterion(output, target)
            train_losses.append(loss.item())
            
            # Calculate train accuracy
            _, pred = output.max(1)
            correct = (pred == target).float().sum()
            train_acc.append((correct / data.size(0)).cpu())

            loss.backward()
            
            # Gradient clipping
            if grad_clip: 
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)
            
            optimizer.step()
            lrs.append(optimizer.param_groups[0]["lr"])
            if verbose:
                #print(f"\tBatch: {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f}")
                pass
        
        
        #scheduler.step()
        time_per_epoch.append(time.time() - start_time)
        # Evaluate model on test set
        test_loss, test_acc = evaluate(model, test_loader, device, criterion)
        history.append({'epoch': epoch, 'lrs': lrs, 'train_loss': np.mean(train_losses), 'train_acc': np.mean(train_acc),
                        'test_loss': test_loss, 'test_acc': test_acc, 'time_per_epoch': time_per_epoch})
        
        combined_loss = 1 * test_loss + 1 * np.mean(train_losses)

        if combined_loss < best_loss:
            print(f"New best model found! Loss: {combined_loss:.4f}")
            best_loss = combined_loss
            best_model_weights = model.state_dict()
            best_epoch = epoch

        if verbose:
            print(f"[Epoch: {epoch+1:02d}/{epochs:02d}] - {time_per_epoch[-1]:.2f}s | LR: {lrs[-1]:.6f} | Train Loss: {np.mean(train_losses):.4f} | Train Acc: {np.mean(train_acc)*100:.2f} | Test Loss: {test_loss:.4f} | Test Acc: {test_acc*100:.2f}")
    if save_m:
        save_model(model, optimizer, epoch, cfg.CHECKPOINTS_PATH(epoch, model.get_model_name()), history)
    if verbose:
        print(f"Total training time: {np.sum(time_per_epoch):.2f}s")

    return history, best_model_weights, best_epoch
