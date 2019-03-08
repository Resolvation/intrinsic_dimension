import os

from torch.utils import data
from torchvision import datasets, transforms


def mnist_loaders(path='~/Datasets/MNIST', batch_size=128, num_workers=4):
    """https://github.com/pytorch/examples/blob/master/mnist/main.py"""
    path = os.path.expanduser(path)
    if not os.path.isdir(path):
        os.mkdir(path)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1307,), std=(0.3081,)),
    ])

    train_loader = data.DataLoader(
        datasets.MNIST(path, train=True, download=True,
                       transform=transform),
        batch_size=batch_size, num_workers=num_workers, shuffle=True
    )
    test_loader = data.DataLoader(
        datasets.MNIST(path, train=False, download=True,
                       transform=transform),
        batch_size=batch_size, num_workers=num_workers, shuffle=False
    )

    return train_loader, test_loader


def cifar10_loaders(path='~/Datasets/CIFAR10', batch_size=512, num_workers=4):
    """https://github.com/kuangliu/pytorch-cifar/blob/master/main.py"""
    path = os.path.expanduser(path)
    if not os.path.isdir(path):
        os.mkdir(path)

    normalize = transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                                     std=(0.2023, 0.1994, 0.2010))

    resize = 32

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.Resize((resize, resize)),
        transforms.ToTensor(),
        normalize
    ])
    transform_test = transforms.Compose([
        transforms.Resize((resize, resize)),
        transforms.ToTensor(),
        normalize
    ])

    train_loader = data.DataLoader(
        datasets.CIFAR10(path, train=True, download=True,
                         transform=transform_train),
        batch_size=batch_size, num_workers=num_workers, shuffle=True
    )
    test_loader = data.DataLoader(
        datasets.CIFAR10(path, train=False, download=True,
                         transform=transform_test),
        batch_size=batch_size, num_workers=num_workers, shuffle=False
    )

    return train_loader, test_loader


def imagenet_loaders(path='~/Datasets/ImageNet', batch_size=128, num_workers=4):
    """https://github.com/pytorch/examples/blob/master/imagenet/main.py"""
    path = os.path.expanduser(path)
    trainpath = os.path.join(path, 'train')
    if not os.path.isdir(trainpath):
        os.mkdir(trainpath)
    testpath = os.path.join(path, 'test')
    if not os.path.isdir(testpath):
        os.mkdir(testpath)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])

    train_loader = data.DataLoader(
        datasets.ImageFolder(trainpath, transform=transform_train),
        batch_size=batch_size, num_workers=num_workers, shuffle=True,
        pin_memory=True
    )
    test_loader = data.DataLoader(
        datasets.ImageFolder(testpath, transform=transform_test),
        batch_size=batch_size, num_workers=num_workers, shuffle=False,
        pin_memory=True
    )

    return train_loader, test_loader
