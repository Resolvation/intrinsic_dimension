from torch import nn
from torch.nn import functional as F

from models.layers import StochasticLinear


class LeNet300_100_stochastic(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        self.n_classes = 10
        self.fc1 = StochasticLinear(28*28, 300)
        self.fc2 = StochasticLinear(300, 100)
        self.fc3 = StochasticLinear(100, n_classes)

    def forward(self, input):
        h = F.relu(self.fc1(input.view(input.shape[0], -1)))
        h = F.relu(self.fc2(h))
        return self.fc3(h)

    def kl(self):
        res = 0.
        for child in self.children():
            res += child.kl()
        return res

    def log_weights(self, writer, epoch):
        for i, child in enumerate(self.children()):
            writer.add_histogram(f'std/layer_{i + 1}',
                                 (child.log_sigma_sqr.view(-1) / 2).exp(),
                                 epoch)
