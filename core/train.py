from datetime import datetime

import torch
from torch.nn import functional as F


def train(model, device, train_loader, criterion, optimizer, epoch,
          verbose=False):
    total_loss = 0.
    correct = 0
    model.train()
    start_time = datetime.now()

    for features, labels in train_loader:
        features, labels = features.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(features)

        loss = criterion(outputs, labels)
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

        pred = outputs.argmax(1, keepdim=True)
        correct += pred.eq(labels.view_as(pred)).sum().item()

    avg_loss = total_loss / len(train_loader)
    accuracy = 100 * correct / len(train_loader.dataset)
    time = str(datetime.now() - start_time).split('.')[0]

    if verbose:
        print(f'-->Epoch: {epoch:03d}    Loss: {avg_loss:.8f}    '
              f'Accuracy: {accuracy:.02f}    Time: {time}')

    return avg_loss, accuracy


def logmeanexp(x, dim=None, keepdim=False):
    """Stable computation of log(mean(exp(x))"""
    if dim is None:
        x, dim = x.view(-1), 0
    xm, _ = torch.max(x, dim, keepdim=True)
    x = xm + torch.log(torch.mean(torch.exp(x - xm), dim, keepdim=True))
    return x if keepdim else x.squeeze(dim)


def test_classifier(model, device, test_loader,
                    criterion, verbose=False, n_ens=1):
    total_loss = 0.
    correct = 0
    if n_ens == 1:
        model.eval()
    else:
        model.train()

    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = torch.empty(features.shape[0], model.n_classes,
                                  n_ens, device=device)

            for i in range(n_ens):
                outputs[:, :, i] = F.log_softmax(model(features), dim=1)

            outputs = logmeanexp(outputs, dim=2)

            loss = criterion(outputs, labels)
            total_loss += loss.item()

            pred = outputs.argmax(1, keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()

    avg_loss = total_loss / len(test_loader)
    accuracy = 100 * correct / len(test_loader.dataset)

    if verbose:
        print(f'  Average loss: {avg_loss:.6f}')
        print(f'  Accuracy: {accuracy:.2f}')

    return avg_loss, accuracy


def eval_determenistic(writer, model, device, train_loader, test_loader,
                       criterion, epoch, verbose):
    if verbose:
        print('Trainset:')
    avg_loss_trainset, accuracy_trainset = test_classifier(
        model, device, train_loader, criterion, verbose)

    if verbose:
        print('Test set:')
    avg_loss_testset, accuracy_testset = test_classifier(
        model, device, test_loader, criterion, verbose)

    writer.add_scalars('testing/loss', {
        'trainset': avg_loss_trainset,
        'testset': avg_loss_testset
    }, epoch)
    writer.add_scalars('testing/accuracy', {
        'trainset': accuracy_trainset,
        'testset': accuracy_testset
    }, epoch)


def eval_stochastic(writer, model, device, train_loader, test_loader,
                    trainset_criterion, testset_criterion, epoch,
                    verbose=True, n_ens=10):
    if verbose:
        print('Trainset:')
        print(' Mean:')
    avg_loss_trainset_mean, accuracy_trainset_mean = test_classifier(
        model, device, train_loader, trainset_criterion, verbose)
    if verbose:
        print(' Ensemble:')
    avg_loss_trainset_ens, accuracy_trainset_ens = test_classifier(
        model, device, train_loader, trainset_criterion, verbose, n_ens)

    if verbose:
        print('Testset:')
        print(' Mean:')
    avg_loss_testset_mean, accuracy_testset_mean = test_classifier(
        model, device, test_loader, testset_criterion, verbose)
    if verbose:
        print(' Ensemble:')
    avg_loss_testset_ens, accuracy_testset_ens = test_classifier(
        model, device, test_loader, testset_criterion, verbose, n_ens)

    writer.add_scalars('testing/loss', {
        'trainset_mean': avg_loss_trainset_mean,
        'trainset_ens': avg_loss_trainset_ens,
        'test_mean': avg_loss_testset_mean,
        'testset_ens': avg_loss_testset_ens
    }, epoch)
    writer.add_scalars('testing/accuracy', {
        'trainset_mean': accuracy_trainset_mean,
        'trainset_ens': accuracy_trainset_ens,
        'testset_mean': accuracy_testset_mean,
        'testset_ens': accuracy_testset_ens
    }, epoch)
