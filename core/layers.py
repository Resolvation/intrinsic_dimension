import torch
import torch.nn as nn
import torch.nn.functional as F


def dense_offset(d, shape):
    """
    Return dense offset matrix to use in Offset Layers.
    """
    return F.normalize(torch.randn(shape[:-1]+[d]+[shape[-1]]), dim=-2)


class LinearOffset(nn.Module):
    """
    Apply linear transformation to the incoming data in given parameter subspace.
    """
    def __init__(self, d, in_features, out_features, bias=True):
        super().__init__()
        self.d = d
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.register_parameter("theta", nn.Parameter(torch.zeros(d)))
        self.register_buffer("P_A",
                             dense_offset(d, [out_features, in_features]))
        self.register_buffer("A_0", torch.empty([out_features, in_features]))
        nn.init.kaiming_normal_(self.A_0, nonlinearity="relu")
        if bias:
            self.register_buffer("P_b", dense_offset(d, [out_features]))
            self.register_buffer("b_0", torch.zeros([out_features]))
        else:
            self.register_buffer("P_b", None)
            self.register_buffer("b_0", None)

    def forward(self, input, theta=None):
        if theta is None:
            A = lambda: torch.matmul(self.theta, self.P_A) + self.A_0
            b = lambda: torch.matmul(self.theta, self.P_b) + self.b_0
        else:
            A = lambda: torch.matmul(theta, self.P_A) + self.A_0
            b = lambda: torch.matmul(theta, self.P_b) + self.b_0
        if self.bias:
            return F.linear(input, A(), b())
        else:
            return F.linear(input, A())


class StochasticModule(nn.Module):
    """
    torch.nn.Module with stochastic attribute.
    """
    def __init__(self):
        super().__init__()
        setattr(self, "stochastic", True)

    def setattr(self, name, value):
        setattr(self, name, value)
        for child in self.children():
            setattr(child, name, value)

    def KL(self):
        return sum([child.KL() for child in self.children()])


class BayesLinear(StochasticModule):
    """
    Applies a variational linear transformation to the incoming data.
    """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.mu = nn.Parameter(torch.randn(out_features, in_features))
        self.log_sigma_sqr = nn.Parameter(
            torch.full((out_features, in_features), -12))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        h_mean = F.linear(x, self.mu, self.bias)
        h_std = (1e-16 + F.linear(x ** 2, self.log_sigma_sqr.exp())).sqrt()
        if self.stochastic:
            eps = torch.randn_like(h_std)
        else:
            eps = 0.
        return h_mean + eps * h_std

    def KL(self):
        return ((self.mu ** 2 + self.log_sigma_sqr.exp()
                 - self.log_sigma_sqr).sum()
                - self.in_features*self.out_features) / 2


class BayesLinearOffset(StochasticModule):
    """
    Applies a variational linear transformation to the incoming data in given
    parameter subspace.
    """
    def __init__(self, d, in_features, out_features, bias=True):
        super().__init__()
        self.d = d
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.register_parameter("mu", nn.Parameter(torch.zeros(d)))
        self.register_parameter(
            "log_sigma_sqr", nn.Parameter(torch.full(tuple([d]), -12)))
        self.register_buffer(
            "P_A", dense_offset(d, [out_features, in_features]))
        self.register_buffer("A_0", torch.empty([out_features, in_features]))
        nn.init.kaiming_normal_(self.A_0, nonlinearity="relu")
        if bias:
            self.register_buffer("P_b", dense_offset(d, [out_features]))
            self.register_buffer("b_0", torch.randn([out_features]))
        else:
            self.register_buffer("P_b", None)
            self.register_buffer("b_0", None)

    def forward(self, input, theta=None):
        if theta == None:
            if self.stochastic:
                eps = torch.randn_like(self.log_sigma_sqr)
            else:
                eps = 0
            theta = self.mu + eps * (1e-16 + self.log_sigma_sqr.exp()).sqrt()
            A = lambda: torch.matmul(theta, self.P_A) + self.A_0
            b = lambda: torch.matmul(theta, self.P_b) + self.b_0
        else:
            A = lambda: torch.matmul(theta, self.P_A) + self.A_0
            b = lambda: torch.matmul(theta, self.P_b) + self.b_0
        if self.bias:
            return F.linear(input, A(), b())
        else:
            return F.linear(input, A())

    def KL(self):
        return ((self.mu ** 2 + self.log_sigma_sqr.exp()
                 - self.log_sigma_sqr).sum() - self.d) / 2
