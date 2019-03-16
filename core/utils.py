import os

import torch
from torch import nn
from torch.nn import functional as F


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def linear_lr(epoch, n_epochs, alpha):
    return min(1, 2 * (1 - alpha * epoch / n_epochs))


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
        return (F.cross_entropy(input, target, reduction='mean')
                + kl_weight * kl / self.dataset_size)
