def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def linear_lr(epoch, n_epochs, alpha):
    return min(1, 2 * (1 - alpha * epoch / n_epochs))
