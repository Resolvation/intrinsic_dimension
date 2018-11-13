import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def evaluate(device, net, data_loader, criterion, **kwargs):
    """
    Test net of device.
    """
    if len(kwargs) == 0:
        evaluate_determinated_net(device, net, data_loader, criterion)
    else:
        evaluate_stochastic_net(device, net, data_loader, criterion, **kwargs)


def evaluate_determinated_net(device, net, data_loader, criterion):
    """
    Test net on device and return accuracy and loss.
    """
    correct, loss = 0, 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss += criterion(outputs, labels)
            pred = outputs.max(1, keepdim=True)[1]
            correct += pred.eq(labels.view_as(pred)).sum().item()
    loss /= len(data_loader.dataset)
    accuracy = correct / len(data_loader.dataset)
    print("Average loss: {:>010.4f}    Accuracy: {:>6.2f}".format(loss, 100*accuracy))


def evaluate_stochastic_net(device, net, data_loader, criterion, num_nets=10):
    net.set_flag("train", False)
    print("Mean")
    evaluate_determinated_net(device, net, data_loader, criterion)
    net.set_flag("train", True)
    print("Ensemble")
    total_correct = 0
    correct, test_loss = torch.zeros(10).to(device), torch.zeros(10).to(device)
    pred = torch.empty(num_nets, data_loader.batch_size, dtype=torch.long).to(device)
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            for i in range(num_nets):
                outputs = net(images)
                test_loss[i] += criterion(outputs, labels)
                pred[i] = outputs.max(1, keepdim=True)[1].view_as(pred[i])
                correct[i] += pred[i].eq(labels.view(pred[i].shape)).sum().item()
            total_pred = torch.Tensor([np.argmax(np.bincount(image)) for image in pred.transpose(0, 1)]).to(torch.long).to(device)
            total_correct += total_pred.eq(labels.view(pred[0].shape)).sum().item()
    for i in range(num_nets):
        print("Net id: {:02}    Average loss: {:>010.4f}    Accuracy: {:>6.2f}".format(i+1, test_loss[i]/len(data_loader.dataset), 100*correct[i]/len(data_loader.dataset)))
    print("Average accuracy: {:>6.2f}".format(100*torch.mean(correct)/len(data_loader.dataset)))
    print("Ensamble:\tAccuracy: {:>6.2f}".format(100*total_correct/len(data_loader.dataset)))
