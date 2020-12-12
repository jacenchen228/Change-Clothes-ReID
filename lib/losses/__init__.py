from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .cross_entropy_loss import CrossEntropyLoss
from .hard_mine_triplet_loss import TripletLoss
from .deep_info_max_loss import DeepInfoMaxLoss
from .div_loss import AdvDivLoss
from .cloth_sensitive_triplet_loss import TripletLoss_Cloth_Sen
from .cloth_insensitive_triplet_loss import TripletLoss_Cloth_Insen


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
