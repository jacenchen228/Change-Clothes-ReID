import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepInfoMaxLoss(nn.Module):
    def __init__(self, margin=0.8):
        super(DeepInfoMaxLoss, self).__init__()
        self.margin = margin

    def forward(self, ej, em):
        # see appendix 1A of https://arxiv.org/pdf/1808.06670.pdf
        Ej = -F.softplus(-ej)
        Em = F.softplus(em)
        loss = Em - Ej
        margin = torch.full(loss.shape, self.margin).cuda()
        dim_loss = torch.max(torch.cat([loss, margin], dim=1), dim=1)[0]
        dim_loss = torch.mean(dim_loss)

        return dim_loss

