import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, transforms


def mnist_data(path='~/Datasets/', batch_size=100):
    '''Returns dataloaders with default parameters.'''
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(path, train=True, download=True,
                       transform=transforms.ToTensor()),
        batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(path, train=False, download=True,
                       transform=transforms.ToTensor()),
        batch_size=batch_size, shuffle=True, num_workers=4)
    return train_loader, test_loader


def train(model, device, train_loader, criterion, optimizer, n_epochs=200):
    steps, losses = [0], []
    print('Train:')
    start_time = time.time()
    for epoch in range(n_epochs):
        batch_loss = 0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            batch_loss += loss.item()
            if (batch_idx + 1) % n_epochs == 0:
                steps.append(steps[-1]+labels.shape[0])
                losses.append(loss.item())
        if (epoch + 1) % 5 == 0:
            print('Epoch: %d\tLoss: %.4f' % (epoch+1, batch_loss/len(train_loader)))
    hours, remainder = divmod(time.time() - start_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print('Time: %02d:%02d:%02d' % (hours, minutes, seconds))
    return steps[1:], losses


def test(model, device, test_loader, criterion):
    correct, test_loss = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            test_loss += criterion(outputs, labels)
            pred = outputs.max(1, keepdim=True)[1]
            correct += pred.eq(labels.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    accuracy = 100 * correct / len(test_loader.dataset)
    print('Test:\nAverage loss: %.4f\tAccuracy: %.2f' % (test_loss, accuracy))


def evaluate(model, device, test_loader, criterion, num_ens=1):
    """Calculate ensemble accuracy and loss."""
    correct, test_loss = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            for _ in range(num_ens):
                outputs = model(images)
                test_loss += criterion(outputs, labels)
                pred = outputs.max(1, keepdim=True)[1]
                correct += pred.eq(labels.view_as(pred)).sum().item()
    test_loss /= num_ens*len(test_loader.dataset)
    accuracy = 100*correct/num_ens/len(test_loader.dataset)
    print('Test:\nAverage loss: %.4f\tAccuracy: %.2f' % (test_loss, accuracy))


def save(model, path='last_checkpoint.tar'):
    torch.save(model.state_dict(), path)
