from torch import nn
from models.layers import Flatten


class LeNet300_100(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        self.n_classes = n_classes
        self.classificator = nn.Sequential(
            Flatten(),
            nn.Linear(28*28, 300),
            nn.ReLU(),
            nn.Linear(300, 100),
            nn.ReLU(),
            nn.Linear(100, n_classes)
        )

    def forward(self, x):
        return self.classificator.forward(x)
