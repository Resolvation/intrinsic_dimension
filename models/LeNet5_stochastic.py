import torch
from torch import nn
from torch.nn import functional as F

from models.layers import StochasticLinear, StochasticConv2d
from models.LeNet5 import LeNet5


class LeNet5_stochastic(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        self.n_classes = n_classes
        self.conv1 = StochasticConv2d(3, 6, 5)
        self.conv2 = StochasticConv2d(6, 16, 5)
        self.fc1 = StochasticLinear(16*5*5, 120)
        self.fc2 = StochasticLinear(120, 84)
        self.fc3 = StochasticLinear(84, n_classes)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), kernel_size=2)
        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=2)
        x = F.relu(self.fc1(x.view(x.shape[0], -1)))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def kl(self):
        res = 0.
        for child in self.children():
            res += child.kl()
        return res

    def load_weights(self, path):
        donor = LeNet5().cuda()
        donor.load_state_dict(torch.load(path))
        self.conv1.mu = donor.conv1.weight
        self.conv1.bias = donor.conv1.bias
        self.conv2.mu = donor.conv2.weight
        self.conv2.bias = donor.conv2.bias
        self.fc1.mu = donor.fc1.weight
        self.fc1.bias = donor.fc1.bias
        self.fc2.mu = donor.fc2.weight
        self.fc2.bias = donor.fc2.bias
        self.fc3.mu = donor.fc3.weight
        self.fc3.bias = donor.fc3.bias
        del donor
        return self

    def log_weights(self, writer, epoch):
        for i, child in enumerate(self.children()):
            writer.add_histogram(f'std/layer_{i + 1}',
                                 (child.log_sigma_sqr.view(-1) / 2).exp(),
                                 epoch)
