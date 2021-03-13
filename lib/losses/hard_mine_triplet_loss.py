from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn
from torch.nn import functional as F


class TripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.
    
    Reference:
        Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.


    Imported from `<https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py>`_.
    
    Args:
        margin (float, optional): margin for triplet. Default is 0.3.
    """
    
    def __init__(self, margin=0.3, normalize_feature=True):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.normalize_feature = normalize_feature

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): feature matrix with shape (batch_size, feat_dim).
            targets (torch.LongTensor): ground truth labels with shape (num_classes).
        """
        if self.normalize_feature:
            inputs = F.normalize(inputs, p=2, dim=1)

        if(inputs.ndim != 2):
            inputs = inputs.view(inputs.size(0), -1)

        n = inputs.size(0)
        
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        
        # # For each anchor, find the hardest positive and negative
        # mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        # dist_ap, dist_an = [], []
        # for i in range(n):
        #     dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
        #     dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        # dist_ap = torch.cat(dist_ap)
        # dist_an = torch.cat(dist_an)

        # Weightws mining for triplet loss
        is_pos = targets.view(n, 1).expand(n, n).eq(targets.view(n, 1).expand(n, n).t()).float()
        is_neg = targets.view(n, 1).expand(n, n).ne(targets.view(n, 1).expand(n, n).t()).float()

        dist_ap, dist_an = weighted_example_mining(dist, is_pos, is_neg)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        return self.ranking_loss(dist_an, dist_ap, y)

def weighted_example_mining(dist_mat, is_pos, is_neg):
    """For each anchor, find the weighted positive and negative sample.
    Args:
      dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
      is_pos:
      is_neg:
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
    """
    assert len(dist_mat.size()) == 2

    is_pos = is_pos
    is_neg = is_neg
    dist_ap = dist_mat * is_pos
    dist_an = dist_mat * is_neg

    weights_ap = softmax_weights(dist_ap, is_pos)
    weights_an = softmax_weights(-dist_an, is_neg)

    dist_ap = torch.sum(dist_ap * weights_ap, dim=1)
    dist_an = torch.sum(dist_an * weights_an, dim=1)

    return dist_ap, dist_an


def softmax_weights(dist, mask):
    max_v = torch.max(dist * mask, dim=1, keepdim=True)[0]
    diff = dist - max_v
    Z = torch.sum(torch.exp(diff) * mask, dim=1, keepdim=True) + 1e-6  # avoid division by zero
    W = torch.exp(diff) * mask / Z
    return W
