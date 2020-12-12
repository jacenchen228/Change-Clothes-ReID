from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn


class OFPenalty(nn.Module):

    def __init__(self):
        super().__init__()

        self.beta = 1e-6

    def dominant_eigenvalue(self, A):

        B, N, _ = A.size()
        x = torch.randn(B, N, 1, device='cuda')

        for _ in range(1):
            x = torch.bmm(A, x)
        # x: 'B x N x 1'
        numerator = torch.bmm(
            torch.bmm(A, x).view(B, 1, N),
            x
        ).squeeze()
        denominator = (torch.norm(x.view(B, N), p=2, dim=1) ** 2).squeeze()

        return numerator / denominator

    def get_singular_values(self, A):

        AAT = torch.bmm(A, A.permute(0, 2, 1))
        B, N, _ = AAT.size()
        largest = self.dominant_eigenvalue(AAT)
        I = torch.eye(N, device='cuda').expand(B, N, N)  # noqa
        I = I * largest.view(B, 1, 1).repeat(1, N, N)  # noqa
        tmp = self.dominant_eigenvalue(AAT - I)
        return tmp + largest, largest

    def apply_penalty(self, x):
        batches, channels, height, width = x.size()
        W = x.view(batches, channels, -1)
        smallest, largest = self.get_singular_values(W)
        singular_penalty = (largest - smallest) * self.beta

        return singular_penalty.sum() / (x.size(0) / 32.)  # Quirk: normalize to 32-batch case

    def forward(self, features):

        singular_penalty = self.apply_penalty(features)

        return singular_penalty
