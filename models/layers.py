from torch import nn


class Flatten(nn.Module):
    """Flattens input saving batch structure."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)
