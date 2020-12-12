import copy
import math

import torch
from torch import nn

from .resnet_strong_base import resnet50, resnet101, resnext101_32x8d
from .resnet_ibn_a import resnet50_ibn_a, resnet101_ibn_a
from lib.lib3D import HMR_HEAD, SMPL, Discriminator, batch_rodrigues

SMPL_MODEL_DIR = '/home/jiaxing/SPIN-master/data/smpl'

class MGN3D(nn.Module):
    def __init__(self, num_classes=1000, stripes=[3], num_layers=50, batch_size=32, **kwargs):
        super(MGN3D, self).__init__()
        self.stripes = stripes
        self.batch_size = batch_size
        if num_layers == 50:
            resnet = resnet50(pretrained=True, last_stride=1)
        elif num_layers == 101:
            resnet = resnet101(pretrained=True, last_stride=1)
        elif num_layers == '101_32x8d':
            resnet = resnext101_32x8d(pretrained=True, last_stride=1)
        elif num_layers == '50_ibn':
            resnet = resnet50_ibn_a(pretrained=True, last_stride=1)
        elif num_layers == '101_ibn':
            resnet = resnet101_ibn_a(pretrained=True, last_stride=1)
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3[0],
        )

        res_conv4 = nn.Sequential(*resnet.layer3[1:])
        self.gap = nn.AdaptiveAvgPool2d(1)

        reduction = nn.Sequential(nn.Conv2d(2048, 256, 1, bias=False), nn.BatchNorm2d(256))  # , nn.ReLU())
        self._init_reduction(reduction)
        fc_layer = nn.Sequential(nn.Dropout(), nn.Linear(256, num_classes))
        self._init_fc(fc_layer)

        branches = []
        for stripe_id, stripe in enumerate(stripes):
            embedding_layers = nn.ModuleList([copy.deepcopy(reduction) for _ in range(stripe+1)])
            fc_layers = nn.ModuleList([copy.deepcopy(fc_layer) for _ in range(stripe+1)])
            branches.append(
                nn.ModuleList([
                    nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(resnet.layer4)),
                    embedding_layers, fc_layers])
            )
        self.branches = nn.ModuleList(branches)
        self.estimator3D = HMR_HEAD(smpl_mean_params='/home/jiaxing/SPIN-master/data/smpl_mean_params.npz',
                                    res_conv4=res_conv4, res_conv5=resnet.layer4)
        # Load SMPL model
        self.smpl = SMPL(SMPL_MODEL_DIR,
                    batch_size=self.batch_size,
                    create_transl=False)

        # Discriminator architecture
        self.discriminator = Discriminator()

    @staticmethod
    def _init_reduction(reduction):
        # conv
        # nn.init.kaiming_normal_(reduction[0].weight, mode='fan_in')
        nn.init.normal_(reduction[0].weight, std=math.sqrt(2. / 256))
        # bn
        nn.init.constant_(reduction[1].weight, 1.)
        nn.init.constant_(reduction[1].bias, 0.)

    @staticmethod
    def _init_fc(fc):
        # nn.init.kaiming_normal_(fc.weight, mode='fan_out')
        nn.init.normal_(fc[1].weight, std=0.001)
        nn.init.constant_(fc[1].bias, 0.)

    def forward(self, x, real_params, return_featuremaps=False, return_params_3D=False):
        '''
        ('input.shape:', (64, 3, 384, 128))
        '''
        xf_base = self.backbone(x)

        # 3D Sub-network
        xf_shape, pred_rotmats, pred_betas, pred_cam, pred_displace = self.estimator3D(xf_base)
        pred_params = {'rotmat':pred_rotmats,
                       'beta':pred_betas,
                       'cam':pred_cam}

        pred_displace = pred_displace.view(pred_displace.shape[0], 6890, 3)

        pred_outputs1 = self.smpl(betas=pred_betas, body_pose=pred_rotmats[:,1:],
                                 global_orient=pred_rotmats[:,0].unsqueeze(1), pose2rot=False, v_personal=pred_displace)
        pred_outputs2 = self.smpl(betas=pred_betas, body_pose=pred_rotmats[:,1:],
                                 global_orient=pred_rotmats[:,0].unsqueeze(1), pose2rot=False)

        # ReID Feature Learning
        logits, tri_logits = [], []
        for idx, stripe in enumerate(self.stripes):
            branch = self.branches[idx]
            backbone, reduces, fcs = branch
            net = backbone(xf_base)
            # global
            global_feat = self.gap(net)
            global_feat_reduce = reduces[0](global_feat).squeeze(dim=3).squeeze(dim=2)
            global_feat_logit = fcs[0](global_feat_reduce)
            logits.append(global_feat_logit)
            tri_logits.append(global_feat_reduce)
            # local
            local_tri_logits = []
            for i in range(stripe):
                # stride = 24 // stripe
                stride = 14 // stripe
                local_feat = net[:, :, i*stride: (i+1)*stride, :]
                local_feat = self.gap(local_feat)
                local_feat_reduce = reduces[i+1](local_feat).squeeze(dim=3).squeeze(dim=2)
                local_feat_logit = fcs[i+1](local_feat_reduce)
                logits.append(local_feat_logit)
                local_tri_logits.append(local_feat_reduce)
            tri_logits.append(torch.cat(local_tri_logits, dim=1))

        if not self.training:
            return [torch.cat(tri_logits, dim=1)]

        # discriminator output
        encoder_disc_value = self.discriminator(pred_betas, pred_rotmats.view(-1, 24, 9))
        gen_disc_value = self.discriminator(pred_betas.detach(), pred_rotmats.detach().view(-1, 24, 9))

        real_poses, real_shapes = real_params[:, :72], real_params[:, 72:]
        real_rotmats = batch_rodrigues(real_poses.contiguous().view(-1, 3)).view(-1, 24, 9)
        real_disc_value = self.discriminator(real_shapes, real_rotmats)

        return logits, tri_logits, pred_params, pred_outputs1, pred_outputs2, encoder_disc_value, gen_disc_value, real_disc_value


