import torch
import torch.nn as nn
import torch.nn.functional as F


def dense_offset(d, shape):
    ww = torch.randn(shape[:-1]+[d]+[shape[-1]])
    return F.normalize(ww, dim=-2)


class LinearOffsetLayer(nn.Module):
    def __init__(self, d, in_features, out_features, bias=True):
        super().__init__()
        self.d = d
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.register_buffer('theta0_A', torch.empty([out_features, in_features]))
        nn.init.kaiming_normal_(self.theta0_A, nonlinearity='relu')
        self.register_buffer('P_A', dense_offset(d, [out_features, in_features]))
        if bias:
            self.register_buffer('theta0_b', torch.randn([out_features]))
            self.register_buffer('P_b', dense_offset(d, [out_features]))
        else:
            self.register_buffer('theta0_b', None)
            self.register_buffer('P_b', None)

    def forward(self, input, theta_d):
        def A():
            return torch.matmul(theta_d, self.P_A) + self.theta0_A

        def b():
            return torch.matmul(theta_d, self.P_b) + self.theta0_b

        if self.bias:
            return F.linear(input, A(), b())
        else:
            return F.linear(input, A())
