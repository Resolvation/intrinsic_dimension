import torch
import torch.nn as nn
import torch.nn.functional as F


def dense_offset(d, shape):
    ww = torch.randn(shape[:-1]+[d]+[shape[-1]])
    return F.normalize(ww, dim=-2)


class LinearOffset(nn.Module):

    def __init__(self, d, in_features, out_features, bias=True):
        super().__init__()
        self.d = d
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.register_buffer('P_A', dense_offset(d, [out_features, in_features]))
        self.register_buffer('theta0_A', torch.empty([out_features, in_features]))
        nn.init.kaiming_normal_(self.theta0_A, nonlinearity='relu')
        if bias:
            self.register_buffer('P_b', dense_offset(d, [out_features]))
            self.register_buffer('theta0_b', torch.randn([out_features]))
        else:
            self.register_buffer('P_b', None)
            self.register_buffer('theta0_b', None)

    def forward(self, input, theta_d):
        def A():
            return torch.matmul(theta_d, self.P_A) + self.theta0_A

        def b():
            return torch.matmul(theta_d, self.P_b) + self.theta0_b

        if self.bias:
            return F.linear(input, A(), b())
        else:
            return F.linear(input, A())


class BayesLinear(nn.Module):
    '''Applies a variational linear transformation to the incoming data.'''

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        setattr(self, 'train', True)
        self.in_features = in_features
        self.out_features = out_features
        self.mu = nn.Parameter(torch.randn(out_features, in_features))
        self.log_sigma_sqr = nn.Parameter(torch.full((out_features, in_features), -12))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        h_mean = F.linear(x, self.mu, self.bias)
        h_std = (1e-16 + F.linear(x ** 2, self.log_sigma_sqr.exp())).sqrt()
        if self.train:
            eps = torch.randn_like(h_std)
        else:
            eps = 0.
        return h_mean + eps * h_std

    def KL(self):
        got = ((self.mu**2 + self.log_sigma_sqr.exp() - self.log_sigma_sqr).sum()\
               - self.in_features*self.out_features) / 2
        return got


class NetWrapper(nn.Module):

    def __init__(self):
        super().__init__()

    def set_flag(self, flag_name, value):
        setattr(self, flag_name, value)
        for child in self.children():
            if hasattr(child, flag_name):
                setattr(child, flag_name, value)
