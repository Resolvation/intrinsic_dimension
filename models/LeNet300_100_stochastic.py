from torch import nn
from torch.nn import functional as F

from models.layers import StochasticLinear


class LeNet300_100(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        self.n_classes = 10
        self.fc1 = StochasticLinear(28*28, 300)
        self.fc2 = StochasticLinear(300, 100)
        self.fc3 = StochasticLinear(100, 10)

    def forward(self, input):
        h = F.relu(self.fc1(input.view(input.shape[0], -1)))
        h = F.relu(self.fc2(h))
        return self.fc3(h)

    def kl(self):
        res = 0.
        for child in self.children():
            res += child.kl()
        return res
