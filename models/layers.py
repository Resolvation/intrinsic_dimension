import torch
from torch import nn
from torch.nn import init, Parameter
import torch.nn.functional as F
from torch.nn.modules.utils import _pair


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
        self.log_sigma_sqr = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, out_features))
        else:
            self.register_parameter('bias', None)
        self._reset_parameters()

    def _reset_parameters(self):
        init.kaiming_normal_(self.mu)
        init.constant_(self.log_sigma_sqr, -6)
        if self.bias is not None:
            init.zeros_(self.bias)

    def forward(self, input):
        h_mean = F.linear(input, self.mu, self.bias)
        h_std = (1e-16 + F.linear(input * input,
                                  self.log_sigma_sqr.exp())).sqrt()
        if self.training:
            eps = torch.randn_like(h_std)
        else:
            eps = 0.
        return h_mean + eps * h_std

    def kl(self):
        return ((self.log_sigma_sqr.exp() + self.mu * self.mu
                 - self.log_sigma_sqr).sum()
                - self.in_features * self.out_features) / 2

    def __repr__(self):
        return (self.__class__.__name__ + '('
                + 'in_features=' + str(self.in_features)
                + ', out_features=' + str(self.out_features)
                + ', bias=' + str(self.bias is not None) + ')')


class StochasticConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = 1
        self.mu = Parameter(torch.Tensor(
            out_channels, in_channels, *self.kernel_size
        ))
        self.log_sigma_sqr = Parameter(torch.Tensor(
            out_channels, in_channels, *self.kernel_size
        ))
        if bias:
            self.bias = Parameter(torch.Tensor(1, out_channels, 1, 1))
        else:
            self.register_parameter('bias', None)
        self._reset_parameters()

    def _reset_parameters(self):
        init.kaiming_normal_(self.mu)
        init.constant_(self.log_sigma_sqr, -6)
        if self.bias is not None:
            init.zeros_(self.bias)

    def forward(self, input):
        h_mean = F.conv2d(input, self.mu,
                          self.bias, self.stride,
                          self.padding, self.dilation)
        h_std = (1e-16 + F.conv2d(input * input, self.log_sigma_sqr.exp(),
                                  None, self.stride,
                                  self.padding, self.dilation))
        if self.training:
            eps = torch.randn_like(h_std)
        else:
            eps = 0.
        return h_mean + eps * h_std

    def kl(self):
        return ((self.log_sigma_sqr.exp() + self.mu * self.mu
                 - self.log_sigma_sqr).sum()
                - self.in_features * self.out_features) / 2

    def __repr__(self):
        s = ('{name}({in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        s += ', padding={padding}'
        s += ', dilation={dilation}'
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)
