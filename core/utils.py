import os

import torch
from torch import nn
from torch.nn import functional as F


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def linear_lr(epoch, n_epochs):
    return min(1, 2 * (1 - epoch / n_epochs))


def save_model(model, name, path='../tars'):
    if not os.path.exists(path):
        os.mkdir(path)
    torch.save(model.state_dict(), path+'/'+name+'.tar')


class SGVLB(nn.Module):
    def __init__(self, network, dataset_size):
        super().__init__()
        self.dataset_size = dataset_size
        self.network = network

    def forward(self, input, target, kl_weight=1.):
        kl = 0.
        for child in self.network.children():
            if hasattr(child, 'kl'):
                kl += child.kl()
        return (F.cross_entropy(input, target, size_average=True)
                * self.dataset_size) + kl_weight * kl
