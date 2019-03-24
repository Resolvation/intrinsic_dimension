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
        self.register_buffer('ext_weight', torch.zeros_like(self.mu))
        self.register_buffer('ext_bias', torch.zeros_like(self.bias))
        self._reset_parameters()

    def _reset_parameters(self):
        init.zeros_(self.mu)
        init.constant_(self.log_sigma_sqr, -12)
        if self.bias is not None:
            init.zeros_(self.bias)

    def forward(self, input):
        h_mean = F.linear(input, self.mu + self.ext_weight,
                          self.bias + self.ext_bias)
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
            out_channels, in_channels, *self.kernel_size))
        self.log_sigma_sqr = Parameter(torch.Tensor(
            out_channels, in_channels, *self.kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.register_buffer('ext_weight', torch.zeros_like(self.mu))
        self.register_buffer('ext_bias', torch.zeros_like(self.bias))
        self._reset_parameters()

    def _reset_parameters(self):
        init.zeros_(self.mu)
        init.constant_(self.log_sigma_sqr, -12)
        if self.bias is not None:
            init.zeros_(self.bias)

    def forward(self, input):
        h_mean = F.conv2d(input, self.mu + self.ext_weight,
                          self.bias + self.ext_bias, self.stride,
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
                - self.in_channels * self.out_channels
                * self.kernel_size[0] * self.kernel_size[1]) / 2

    def __repr__(self):
        s = ('{name}({in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        s += ', padding={padding}'
        s += ', dilation={dilation}'
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)


def dense_offset(d, shape):
    """
    Return dense offset matrix to use in Offset Layers.
    """
    return F.normalize(torch.randn(shape+[d]), dim=-1)


class StochasticLinearOffset(nn.Module):
    """
    Applies a variational linear transformation to the incoming data in given
    parameter subspace.
    """
    def __init__(self, d, in_features, out_features, bias=True):
        super().__init__()
        self.d = d
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("P_A",
                             dense_offset(d, [out_features, in_features]))
        self.register_buffer("A_0", torch.Tensor(out_features, in_features))
        self.weight = lambda theta: torch.matmul(self.P_A, theta) + self.A_0
        if bias:
            self.register_buffer("P_b", dense_offset(d, [out_features]))
            self.register_buffer("b_0", torch.Tensor(out_features))
            self.bias = lambda theta: torch.matmul(self.P_b, theta) + self.b_0
        else:
            self.register_buffer("P_b", None)
            self.register_buffer("b_0", None)
            self.bias = lambda theta: None
        self._reset_parameters()

    def _reset_parameters(self):
        init.kaiming_normal_(self.A_0)
        if self.bias is not None:
            init.zeros_(self.b_0)

    def forward(self, input, theta):
        return F.linear(input, self.weight(theta), self.bias(theta))


class StochasticConv2dOffset(nn.Module):
    def __init__(self, d, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, bias=True):
        super().__init__()
        self.d = d
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = 1
        self.bias = bias
        self.register_buffer("P_A", dense_offset(d, [out_channels, in_channels,
                                                     *self.kernel_size]))
        self.register_buffer("A_0", torch.Tensor(
            out_channels, in_channels, *self.kernel_size))
        self.weight = lambda theta: torch.matmul(self.P_A, theta) + self.A_0
        if bias:
            self.register_buffer("P_b", dense_offset(d, [out_channels]))
            self.register_buffer("b_0", torch.Tensor(out_channels))
            self.bias = lambda theta: torch.matmul(self.P_b, theta) + self.b_0
        else:
            self.register_buffer("P_b", None)
            self.register_buffer("b_0", None)
            self.bias = lambda theta: None

    def _reset_parameters(self):
        init.kaiming_normal_(self.A_0)
        if self.bias is not None:
            init.zeros_(self.b_0)

    def forward(self, input, theta):
        return F.conv2d(input, self.weight(theta), self.bias(theta),
                        stride=self.stride, padding=self.padding,
                        dilation=self.dilation, groups=self.groups)
