import torch
from torch import nn
from torch.nn import functional as F


class SGVLB(nn.Module):
    def __init__(self, network, dataset_size):
        super().__init__()
        self.dataset_size = dataset_size
        self.network = network

    def forward(self, input, target, kl_weight=1.):
        return (F.cross_entropy(input, target, reduction='mean')
                + kl_weight * self.network.kl() / self.dataset_size)


class CELoss(nn.Module):
    """
    Got it from https://github.com/gpleiss/temperature_scaling/blob/master/temperature_scaling.py

    Calculates the Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).
    The input to this loss is the logits of a model, NOT the softmax scores.
    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    We then return a weighted average of the gaps, based on the number
    of samples in each bin
    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=20):
        """
        n_bins (int): number of confidence interval bins
        """
        super().__init__()
        self.n_bins = n_bins

    def forward(self, logits, labels):
        size = len(labels)
        bin_size = size // self.n_bins
        bin_lower = torch.linspace(0, size - bin_size, self.n_bins).int()
        bin_upper = bin_lower + bin_size

        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        s_confidences, s_indexes = torch.sort(confidences)
        s_accuracies = accuracies[s_indexes]

        ece, mce = 0, 0

        for low, up in zip(bin_lower, bin_upper):
            c = torch.sum(s_confidences[low: up]).item()
            a = torch.sum(s_accuracies[low: up]).item()
            ce = abs(c - a) / bin_size
            ece += ce
            mce = max(mce, ce)

        ece /= self.n_bins

        return ece, mce
