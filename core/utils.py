import os
import torch


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def linear_lr(epoch, n_epochs):
    return min(1, 2 * (1 - epoch / n_epochs))


def save_model(model, name, path='../tars'):
    if not os.path.exists(path):
        os.mkdir(path)
    torch.save(model.state_dict(), path+'/'+name+'.tar')
