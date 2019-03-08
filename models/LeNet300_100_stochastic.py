from torch import nn
from torch.nn import functional as F
from torch.nn import init

from models.layers import StochasticLinear


class LeNet300_100(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = StochasticLinear(28*28, 300)
        self.fc2 = StochasticLinear(300, 100)
        self.fc3 = StochasticLinear(100, 10)

    def _reset_parameters(self):
        init.kaiming_normal_(self.fc1.mu)
        init.constant_(self.fc1.bias, 0)
        init.kaiming_normal_(self.fc2.mu)
        init.constant_(self.fc2.bias, 0)
        init.kaiming_normal_(self.fc3.mu)
        init.constant_(self.fc3.bias, 0)

    def forward(self, input):
        h = F.relu(self.fc1(input.view(input.shape[0], -1)))
        h = F.relu(self.fc2(h))
        return self.fc3(h)

    def kl(self):
        return self.fc1.kl() + self.fc2.kl() + self.fc3.kl()
