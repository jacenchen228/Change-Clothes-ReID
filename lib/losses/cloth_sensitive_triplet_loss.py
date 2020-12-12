from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn


class TripletLoss_Cloth_Sen(nn.Module):
    """Triplet loss with hard positive/negative mining.

    Reference:
        Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.

    Imported from `<https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py>`_.

    Args:
        margin (float, optional): margin for triplet. Default is 0.3.
    """

    def __init__(self, num_class, margin=0.3):
        super(TripletLoss_Cloth_Sen, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.num_class = num_class

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): feature matrix with shape (batch_size, feat_dim).
            targets (torch.LongTensor): ground truth labels with shape (num_classes).
        """
        n = inputs.size(0)

        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        # For each anchor, find the hardest positive and pseudo negative (in different clothes)
        mask_positive = targets.expand(n, n).eq(targets.expand(n, n).t())
        mask_pseudo_negative = targets.expand(n, n).eq(targets.expand(n, n).t() + self.num_class) + \
                               targets.expand(n, n).eq(targets.expand(n, n).t() - self.num_class)
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask_positive[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask_pseudo_negative[i]].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        return self.ranking_loss(dist_an, dist_ap, y)
