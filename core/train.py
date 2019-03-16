from datetime import datetime

import torch
from torch.nn import functional as F

from .metrics import CELoss


def train(model, device, train_loader, criterion, optimizer, epoch,
          verbose=False):
    total_loss = 0.
    nllloss = 0.
    correct = 0
    total_logits = []
    total_labels = []
    model.train()
    start_time = datetime.now()

    for features, labels in train_loader:
        features, labels = features.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(features)

        loss = criterion(outputs, labels)
        total_loss += loss.item()

        nllloss += F.cross_entropy(outputs, labels)
        total_logits.append(outputs)
        total_labels.append(labels)

        loss.backward()
        optimizer.step()

        pred = outputs.argmax(1, keepdim=True)
        correct += pred.eq(labels.view_as(pred)).sum().item()

    avg_loss = total_loss / len(train_loader)
    nllloss /= len(train_loader)
    accuracy = 100 * correct / len(train_loader.dataset)
    time = str(datetime.now() - start_time).split('.')[0]

    eceloss, mceloss = CELoss(15)(torch.cat(total_logits),
                                  torch.cat(total_labels))

    if verbose:
        print(f'-->Epoch: {epoch:03d}    Loss: {avg_loss:8.06f}    '
              f'NLLloss: {nllloss:8.06f}    '
              f'ECELoss: {eceloss*100:5.02f}%    '
              f'MCELoss: {mceloss*100:5.02f}%    '
              f'Accuracy: {accuracy:5.02f}%    Time: {time}')

    return avg_loss, nllloss, eceloss, mceloss, accuracy


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
    nllloss = 0.
    total_logits = []
    total_labels = []
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

            nllloss += F.cross_entropy(outputs, labels)
            total_logits.append(outputs)
            total_labels.append(labels)

            pred = outputs.argmax(1, keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()

    avg_loss = total_loss / len(test_loader)
    nllloss /= len(test_loader)
    accuracy = 100 * correct / len(test_loader.dataset)

    eceloss, mceloss = CELoss(15)(torch.cat(total_logits),
                                  torch.cat(total_labels))

    if verbose:
        print(f'  Loss: {avg_loss:.06f}')
        print(f'  NLLloss: {nllloss:.06f}')
        print(f'  ECELoss: {eceloss*100:.02f}%')
        print(f'  MCELoss: {mceloss*100:.02f}%')
        print(f'  Accuracy: {accuracy:.02f}%')

    return avg_loss, nllloss, eceloss, mceloss, accuracy


def eval_determenistic(writer, model, device, train_loader, test_loader,
                       criterion, epoch, verbose):
    if verbose:
        print('Trainset:')
    res_trainset = test_classifier(
        model, device, train_loader, criterion, verbose)

    if verbose:
        print('Test set:')
    res_testset = test_classifier(
        model, device, test_loader, criterion, verbose)

    writer.add_scalars('testing/loss', {
        'trainset': res_trainset[0],
        'testset': res_testset[0]
    }, epoch)
    writer.add_scalars('testing/nllloss', {
        'trainset': res_trainset[1],
        'testset': res_testset[1]
    }, epoch)
    writer.add_scalars('testing/eceloss', {
        'trainset': res_trainset[2],
        'testset': res_testset[2]
    }, epoch)
    writer.add_scalars('testing/mceloss', {
        'trainset': res_trainset[3],
        'testset': res_testset[3]
    }, epoch)
    writer.add_scalars('testing/accuracy', {
        'trainset': res_trainset[4],
        'testset': res_testset[4]
    }, epoch)


def eval_stochastic(writer, model, device, train_loader, test_loader,
                    trainset_criterion, testset_criterion, epoch,
                    verbose=True, n_ens=10):
    if verbose:
        print('Trainset:')
        print(' Mean:')
    res_trainset_mean = test_classifier(
        model, device, train_loader, trainset_criterion, verbose)
    if verbose:
        print(' Ensemble:')
    res_trainset_ens = test_classifier(
        model, device, train_loader, trainset_criterion, verbose, n_ens)

    if verbose:
        print('Testset:')
        print(' Mean:')
    res_testset_mean = test_classifier(
        model, device, test_loader, testset_criterion, verbose)
    if verbose:
        print(' Ensemble:')
    res_testset_ens = test_classifier(
        model, device, test_loader, testset_criterion, verbose, n_ens)

    writer.add_scalars('testing/loss', {
        'trainset_mean': res_trainset_mean[0],
        'trainset_ens': res_trainset_ens[0],
        'test_mean': res_testset_mean[0],
        'testset_ens': res_testset_ens[0]
    }, epoch)
    writer.add_scalars('testing/nllloss', {
        'trainset_mean': res_trainset_mean[1],
        'trainset_ens': res_trainset_ens[1],
        'test_mean': res_testset_mean[1],
        'testset_ens': res_testset_ens[1]
    }, epoch)
    writer.add_scalars('testing/eceloss', {
        'trainset_mean': res_trainset_mean[2],
        'trainset_ens': res_trainset_ens[2],
        'test_mean': res_testset_mean[2],
        'testset_ens': res_testset_ens[2]
    }, epoch)
    writer.add_scalars('testing/mceloss', {
        'trainset_mean': res_trainset_mean[3],
        'trainset_ens': res_trainset_ens[3],
        'testset_mean': res_testset_mean[3],
        'testset_ens': res_testset_ens[3]
    }, epoch)
    writer.add_scalars('testing/accuracy', {
        'trainset_mean': res_trainset_mean[4],
        'trainset_ens': res_trainset_ens[4],
        'testset_mean': res_testset_mean[4],
        'testset_ens': res_testset_ens[4]
    }, epoch)
