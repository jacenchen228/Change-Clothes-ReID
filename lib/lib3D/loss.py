import copy
from kornia.geometry.transform.pyramid import PyrDown
from functools import reduce

import torch
from torch import nn

from .render_pytorch import Renderer


def keypoint_loss(pred_keypoints_2d, gt_keypoints_2d, criterion_keypoints, gt_weight=1):
    """ Compute 2D reprojection loss on the keypoints.
    The loss is weighted by the confidence.
    The available keypoints are different for each dataset.
    """
    conf = gt_keypoints_2d[:, :, -1].unsqueeze(-1).clone()
    # print('mean of confidence is {}'.format(torch.mean(conf)))
    conf *= gt_weight
    loss = criterion_keypoints(pred_keypoints_2d, gt_keypoints_2d[:, :, :-1])
    # loss = (conf * loss_unweighted).mean()
    loss = loss.mean()

    return loss


def batch_encoder_disc_l2_loss(disc_value):
    k = disc_value.shape[0]
    return torch.sum((disc_value - 1.0) ** 2) * 1.0 / k


def batch_adv_disc_l2_loss(real_disc_value, fake_disc_value):
    ka = real_disc_value.shape[0]
    kb = fake_disc_value.shape[0]
    lb, la = torch.sum(fake_disc_value ** 2) / kb, torch.sum((real_disc_value - 1) ** 2) / ka
    return la + lb


class SilhouetteLoss(nn.Module):
    def  __init__(self, img_size, faces, use_gpu, focal_length, batch_size, mask_weight=0.7, silhouette_base=20000):
        super(SilhouetteLoss, self).__init__()
        self.img_size = img_size
        self.focal_length = focal_length
        self.renderer = Renderer(img_size=img_size, faces=faces, use_gpu=use_gpu,
                        focal_length=focal_length, batch_size=batch_size)
        # self.mse = nn.MSELoss()
        self.bce = nn.BCELoss(size_average=True)
        self.mask_weight = mask_weight
        self.silhouette_base = silhouette_base
        self.prydown = PyrDown()

    def forward(self, pred_vertices, pred_cameras, gt_masks):
        camera_translations = torch.stack([
            pred_cameras[:, 1], pred_cameras[:, 2],
            2 * self.focal_length / (self.img_size * pred_cameras[:, 0] + 1e-9)], dim=-1).unsqueeze(1)

        pred_vertices1 = pred_vertices + camera_translations

        proj_silhouettes = self.renderer.render_silhouette(pred_vertices1)

        proj_silhouettes[proj_silhouettes>0] = 1

        # print('The unique value of pred is {}, gt is {}'.
        #       format(torch.unique(proj_silhouettes), torch.unique(gt_masks)))

        gt_masks = gt_masks.squeeze(1)

        silhouette_loss = self.mask_weight * gt_masks * (1 - proj_silhouettes) + (1 - self.mask_weight) * (1 - gt_masks) * proj_silhouettes
        silhouette_loss = silhouette_loss.mean(0).sum() / self.silhouette_base

        # silhouette_loss = torch.sum(torch.mean((proj_silhouettes - gt_masks) ** 2, dim=0)) / self.silhouette_base

        # silhouette_loss = self.bce(proj_silhouettes, gt_masks)

        return silhouette_loss


    def pyramid_gaussian_loss(self, input_objective, n_levels=3):
        ''' pyramid gaussian transformation for losses
         Reference:
            https://github.com/polmorenoc/opendr/blob/master/opendr/filters.py
        '''

        norm2 = lambda x: x / torch.prod(x.shape)

        cur_obj = input_objective

        input_objective = norm2(input_objective)
        output_objectives = [input_objective]

        for ik in range(n_levels):
            cur_obj = self.prydown(cur_obj)
            output_objectives.append(norm2(cur_obj))

        andit = lambda a: reduce(lambda x, y: torch.cat([x.reshape(-1), y.reshape(-1)]), a)
        output_objectives = andit(output_objectives)

        return output_objectives


class VerticeLoss(nn.Module):
    def __init__(self, val_base=1.0):
        super(VerticeLoss, self).__init__()
        self.val_base = val_base

    def forward(self, pred_vertices, gt_vertices):

        vertice_loss = torch.abs(self.val_base * pred_vertices - self.val_base * gt_vertices).mean()

        return vertice_loss

