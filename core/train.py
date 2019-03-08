from datetime import datetime

import torch


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


def test_classifier(model, device, test_loader, criterion, verbose=False):
    total_loss = 0.
    correct = 0
    model.eval()

    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)

            outputs = model(features)

            loss = criterion(outputs, labels)
            total_loss += loss.item()

            pred = outputs.argmax(1, keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()

    avg_loss = total_loss / len(test_loader)
    accuracy = 100 * correct / len(test_loader.dataset)

    if verbose:
        print(f' Average loss: {avg_loss:.6f}')
        print(f' Accuracy: {accuracy:.2f}')

    return avg_loss, accuracy
