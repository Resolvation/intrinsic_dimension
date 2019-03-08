import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import init, Parameter
import torch.nn.functional as F


class Flatten(nn.Module):
    """Flattens input saving batch structure."""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class StochasticLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.mu = Parameter(torch.Tensor(out_features, in_features))
        self.sigma = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_normal_(self.mu)
        init.constant_(self.sigma, 1)
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, input):
        lrt_mean = F.linear(input, self.mu, self.bias)
        lrt_std = Variable.sqrt_(
            1e-16 + F.linear(input * input, self.sigma * self.sigma))
        if self.training:
            print('In stochastic mode.')
            eps = torch.randn_like(lrt_std)
        else:
            print('In determenistic mode.')
            eps = 0.
        kl = (self.sigma.exp() + self.mu * self.mu - self.sigma) / 2
        return lrt_mean + eps * lrt_std, kl

    def __repr__(self):
        return (self.__class__.__name__ + '('
                + 'in_features=' + str(self.in_features)
                + ', out_features=' + str(self.out_features)
                + ', bias=' + str(self.bias is not None) + ')')



class StochasticConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super().__init__()

    def forward(self, input):
        return
