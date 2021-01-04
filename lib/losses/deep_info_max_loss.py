import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepInfoMaxLoss(nn.Module):
    def __init__(self, margin=0.8):
        super(DeepInfoMaxLoss, self).__init__()
        self.margin = margin

    def forward(self, ej, em):
        if isinstance(ej, (tuple, list)) and isinstance(em, (tuple, list)):
            dim_loss = None
            for ej_i, em_i in zip(ej, em):
                Ej_i = -F.softplus(-ej_i)
                Em_i = F.softplus(em_i)
                loss_i = Em_i - Ej_i
                margin_i = torch.full(loss_i.shape, self.margin).cuda()
                dim_loss_i = torch.max(torch.cat([loss_i, margin_i], dim=1), dim=1)[0]
                if dim_loss is None:
                    dim_loss = torch.mean(dim_loss_i)
                else:
                    dim_loss = dim_loss + torch.mean(dim_loss_i)

            return dim_loss / len(ej)

        if isinstance(ej, (tuple, list)) or isinstance(em, (tuple, list)):
            raise ValueError('Ej and Em can not be one tuple(or list) and one digit.')

        # see appendix 1A of https://arxiv.org/pdf/1808.06670.pdf
        Ej = -F.softplus(-ej)
        Em = F.softplus(em)
        loss = Em - Ej
        margin = torch.full(loss.shape, self.margin).cuda()
        dim_loss = torch.max(torch.cat([loss, margin], dim=1), dim=1)[0]
        dim_loss = torch.mean(dim_loss)

        return dim_loss

