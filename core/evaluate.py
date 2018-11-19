import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.layers import StochasticModule


def evaluate(logger, device, model, data_loader, criterion, **kwargs):
    """
    Evaluate model of device.
    """
    if not isinstance(model, StochasticModule):
        evaluate_determinated_model(logger, device, model,
                                    data_loader, criterion, **kwargs)
    else:
        evaluate_stochastic_model(logger, device, model,
                                  data_loader, criterion, **kwargs)


def evaluate_determinated_model(logger, device, model, data_loader, criterion):
    """
    Test model on device and return accuracy and loss.
    """
    correct, loss = 0, 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss += criterion(outputs, labels)
            pred = outputs.max(1, keepdim=True)[1]
            correct += pred.eq(labels.view_as(pred)).sum().item()
    loss /= len(data_loader.dataset)
    accuracy = 100 * correct / len(data_loader.dataset)
    print("Average loss: {:>010.4f}    Accuracy: {:>6.2f}".format(
        loss, accuracy))
    logger["writer"].add_scalar(logger["tag"]+"accuracy", accuracy,
                                logger["global_step"])
    logger["writer"].add_scalar(logger["tag"]+"loss", loss,
                                logger["global_step"])


def evaluate_stochastic_model(logger, device, model,
                              data_loader, criterion, num_models=10):
    model.setattr("stochastic", False)
    print("Mean")
    logger["tag"] = logger["tag"] + "mean/"
    evaluate_determinated_model(logger, device, model, data_loader, criterion)
    model.setattr("stochastic", True)
    logger["tag"] = logger["tag"][:-5] + "stochastic/"
    print("Ensemble")
    total_correct = 0
    correct = torch.zeros(num_models).to(device)
    test_loss = torch.zeros(num_models).to(device)
    pred = torch.empty(num_models, data_loader.batch_size,
                       dtype=torch.long).to(device)
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            for i in range(num_models):
                outputs = model(images)
                test_loss[i] += criterion(outputs, labels)
                pred[i] = outputs.max(1, keepdim=True)[1].view_as(pred[i])
                correct[i] += pred[i].eq(labels.view_as(pred[i])).sum().item()
            total_pred = torch.Tensor([np.argmax(np.bincount(image)) \
                for image in pred.transpose(0, 1)]).to(torch.long).to(device)
            total_correct += total_pred.eq(labels.view_as(pred[0])).sum().item()
    avg_acc = 100 * torch.mean(correct) / len(data_loader.dataset)
    ensemble_acc = 100 * total_correct / len(data_loader.dataset)
    fmt = "model id: {:02}    Average loss: {:>010.4f}    Accuracy: {:>6.2f}"
    test_loss /= len(data_loader)
    correct /= len(data_loader.dataset) / 100
    losses, acc = dict(), dict()
    for i in range(num_models):
        print(fmt.format(i+1, test_loss[i], correct[i]))
        losses["Net "+str(i+1)] = test_loss[i]
        acc["Net "+str(i+1)] = correct[i]
    logger["writer"].add_scalars(logger["tag"]+"nets/average loss", losses,
                                 logger["global_step"])
    logger["writer"].add_scalars(logger["tag"]+"nets/accuracy", acc,
                                 logger["global_step"])
    print("Average accuracy: {:>6.2f}".format(avg_acc))
    print("Ensemble accuracy: {:>6.2f}".format(ensemble_acc))
    logger["writer"].add_scalar(logger["tag"]+"nets/average accuracy", avg_acc,
                                logger["global_step"])
    logger["writer"].add_scalar(logger["tag"]+"ensemble/accuracy", ensemble_acc,
                                logger["global_step"])
