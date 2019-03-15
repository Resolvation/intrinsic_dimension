import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import Parameter

from models.layers import StochasticLinearOffset, StochasticConv2dOffset
from models.LeNet5 import LeNet5


class LeNet5_stochastic_id(nn.Module):
    def __init__(self, d=50, n_classes=10):
        super().__init__()
        self.n_classes = n_classes
        self.d = d
        self.mu = Parameter(torch.zeros(d))
        self.log_sigma_sqr = Parameter(torch.full([d], -6))
        self.conv1 = StochasticConv2dOffset(d, 3, 6, 5)
        self.conv2 = StochasticConv2dOffset(d, 6, 16, 5)
        self.fc1 = StochasticLinearOffset(d, 16*5*5, 120)
        self.fc2 = StochasticLinearOffset(d, 120, 84)
        self.fc3 = StochasticLinearOffset(d, 84, n_classes)

    def forward(self, x):
        if self.training:
            eps = torch.randn_like(self.log_sigma_sqr)
        else:
            eps = 0
        theta = self.mu + eps * (self.log_sigma_sqr / 2).exp()
        x = F.max_pool2d(F.relu(self.conv1(x, theta)), kernel_size=2)
        x = F.max_pool2d(F.relu(self.conv2(x, theta)), kernel_size=2)
        x = F.relu(self.fc1(x.view(x.shape[0], -1), theta))
        x = F.relu(self.fc2(x, theta))
        return self.fc3(x, theta)

    def kl(self):
        return ((self.log_sigma_sqr.exp() + self.mu * self.mu
                 - self.log_sigma_sqr).sum() - d) / 2

    def load_weights(self, path):
        donor = LeNet5()
        donor.load_state_dict(torch.load(path))
        self.conv1.A_0 = donor.conv1.weight
        self.conv1.b_0 = donor.conv1.bias
        self.conv2.A_0 = donor.conv2.weight
        self.conv2.b_0 = donor.conv2.bias
        self.fc1.A_0 = donor.fc1.weight
        self.fc1.b_0 = donor.fc1.bias
        self.fc2.A_0 = donor.fc2.weight
        self.fc2.b_0 = donor.fc2.bias
        self.fc3.A_0 = donor.fc3.weight
        self.fc3.b_0 = donor.fc3.bias
        del donor
        return self

    def log_weights(self, writer, epoch):
        writer.add_histogram(f'std/', (self.log_sigma_sqr.view(-1) / 2).exp(),
                             epoch)
