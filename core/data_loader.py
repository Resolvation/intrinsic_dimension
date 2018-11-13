from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def mnist_data(path="~/Datasets/", batch_size=100):
    """
    Return MNIST data_loaders with default parameters.
    """
    train_loader = DataLoader(
        datasets.MNIST(path, train=True, download=True,
                       transform=transforms.ToTensor()),
        batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(
        datasets.MNIST(path, train=False, download=True,
                       transform=transforms.ToTensor()),
        batch_size=batch_size, shuffle=True, num_workers=4)
    return train_loader, test_loader
