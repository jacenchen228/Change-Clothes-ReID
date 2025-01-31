from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .cross_entropy_loss import CrossEntropyLoss, CrossEntropyLabelSmooth
from .hard_mine_triplet_loss import TripletLoss
from .deep_info_max_loss import DeepInfoMaxLoss
from .div_loss import AdvDivLoss
from .circle_loss import CircleLoss
from .fast_losses import FastCrossEntropyLoss, FastTripletLoss


def DeepSupervision(criterion, xs, y):
    """DeepSupervision

    Applies criterion to each element in a list.

    Args:
        criterion: loss function
        xs: tuple of inputs
        y: ground truth
    """
    loss = 0.
    for idx, x in enumerate(xs):
        loss += criterion(x, y)
    loss /= len(xs)
    return loss


def DeepSupervisionDIM(criterion, xs, ys):
    """DeepSupervision

    Applies criterion to each element in a list.

    Args:
        criterion: loss function
        xs: tuple of input1
        ys: tuple of input2
    """
    loss = 0.
    for x, y in zip(xs, ys):
        loss += criterion(x, y)
    loss /= len(xs)
    return loss
