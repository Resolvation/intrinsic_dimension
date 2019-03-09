from torch import nn
from torch.nn import functional as F
from models.layers import StochasticLinear, StochasticConv2d


class LeNet5(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        self.n_classes = n_classes
        self.conv1 = StochasticConv2d(3, 6, 5)
        self.conv2 = StochasticConv2d(6, 16, 5)
        self.fc1 = StochasticLinear(16*5*5, 120)
        self.fc2 = StochasticLinear(120, 84)
        self.fc3 = StochasticLinear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), kernel_size=2)
        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=2)
        x = F.relu(self.fc1(x.view(x.shape[0], -1)))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def kl(self):
        res = 0.
        for child in self.children():
            res += child.kl()
        return res
