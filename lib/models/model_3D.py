'''
    file:   model.py

    date:   2018_05_03
    author: zhangxiong(1025679612@qq.com)
'''

import numpy as np
import sys

import torch
import torch.nn as nn
from torch.nn import functional as F

from lib.hmr import LinearModel, SMPL, resnet50
from lib.hmr import load_mean_theta, batch_orth_proj
from lib.hmr.config import *

class ThetaRegressor(LinearModel):
    def __init__(self, fc_layers, use_dropout, drop_prob, use_ac_func, iterations, mean_theta_path, batch_size=64):
        super(ThetaRegressor, self).__init__(fc_layers, use_dropout, drop_prob, use_ac_func)
        self.iterations = iterations
        batch_size = batch_size
        mean_theta = np.tile(load_mean_theta(mean_theta_path), batch_size).reshape((batch_size, -1))
        self.register_buffer('mean_theta', torch.from_numpy(mean_theta).float())
    '''
        param:
            inputs: is the output of encoder, which has 2048 features
        
        return:
            a list contains [ [theta1, theta1, ..., theta1], [theta2, theta2, ..., theta2], ... , ], shape is iterations X N X 85(or other theta count)
    '''
    def forward(self, inputs):
        thetas = []
        shape = inputs.shape
        theta = self.mean_theta[:shape[0], :]
        for _ in range(self.iterations):
            # print(inputs.shape, theta.shape)
            total_inputs = torch.cat([inputs, theta], 1)
            theta = theta + self.fc_blocks(total_inputs)
            thetas.append(theta)
        return thetas


class ResNet_3D(nn.Module):
    def __init__(self, num_classes, batch_size, **kwargs):
        super(ResNet_3D, self).__init__()
        self._read_configs()
        self.batch_size = batch_size
        self.feature_dim = 2048

        print('start creating sub modules...')
        self._create_sub_modules()

        # # bnneck layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.bnneck = nn.BatchNorm1d(self.feature_dim)

        # classifiers
        self.classifier = nn.Linear(self.feature_dim, num_classes, bias=False)

        self._init_params()

    def _read_configs(self):
        def _check_config():
            enable_inter_supervions = ENABLE_INTER_SUPERVISION
            assert not enable_inter_supervions

        _check_config()

        self.beta_count = BETA_COUNT
        self.smpl_model = SMPL_MODEL
        self.smpl_mean_theta_path = SMPL_MEAN_THETA_PATH
        self.total_theta_count = TOTAL_THETA_COUNT
        self.joint_count = JOINT_COUNT

    def _create_sub_modules(self):
        '''
            ddd smpl model, SMPL can create a mesh from beta & theta
        '''
        self.smpl = SMPL(self.smpl_model, obj_saveable=True, batch_size=self.batch_size, joint_type='lsp')

        '''
            only resnet50 and hourglass is allowd currently, maybe other encoder will be allowd later.
        '''
        print('creating resnet50')
        self.encoder = Resnet.load_Res50Model()

        '''
            regressor can predict betas(include beta and theta which needed by SMPL) from coder extracted from encoder in a iteratirve way
        '''
        fc_layers = [self.feature_dim + self.total_theta_count, 1024, 1024, 85]
        use_dropout = [True, True, False]
        drop_prob = [0.5, 0.5, 0.5]
        use_ac_func = [True, True, False] #unactive the last layer
        iterations = 3
        self.regressor = ThetaRegressor(fc_layers, use_dropout, drop_prob, use_ac_func, iterations,
                                        mean_theta_path=self.smpl_mean_theta_path, batch_size=self.batch_size)
        self.iterations = iterations

        print('finished create the encoder modules...')

    def forward(self, inputs, return_featuremaps=True):
        feature = self.encoder(inputs)

        v1= self.avgpool(feature)
        v1 = v1.view(v1.size(0), -1)

        v1_new = self.bnneck(v1)

        if not self.training:

            # test_feat0 = torch.cat([F.normalize(v1, p=2, dim=1),
            #                         F.normalize(v1_parts, p=2, dim=1).view(v1_parts.size(0), -1)], dim=1)
            # return [test_feat0, test_feat1, test_feat2]

            test_feat0 = F.normalize(v1, p=2, dim=1)
            test_feat1 = F.normalize(v1_new, p=2, dim=1)
            return [test_feat0, test_feat1]

        y1 = self.classifier(v1_new)

        # Iterative 3D Regression
        thetas = self.regressor(v1)
        detail_info = []
        for theta in thetas:
            detail_info.append(self._calc_detail_info(theta))

        # total_predict_thetas = self._accumulate_thetas(detail_info)
        (predict_theta, _, predict_j2d, _, _) = detail_info[-1]

        return [y1], [v1], predict_theta, predict_j2d

    '''
        purpose:
            calc verts, joint2d, joint3d, Rotation matrix

        inputs:
            theta: N X (3 + 72 + 10)

        return:
            thetas, verts, j2d, j3d, Rs
    '''

    def _calc_detail_info(self, theta):
        cam = theta[:, 0:3].contiguous()
        pose = theta[:, 3:75].contiguous()
        shape = theta[:, 75:].contiguous()
        verts, j3d, Rs = self.smpl(beta = shape, theta = pose, get_skin = True)
        j2d = batch_orth_proj(j3d, cam)

        return (theta, verts, j2d, j3d, Rs)

    def _accumulate_thetas(self, generator_outputs):
        thetas = []
        for (theta, verts, j2d, j3d, Rs) in generator_outputs:
            thetas.append(theta)
        return torch.cat(thetas, 0)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

def resnet50_3D(num_classes=150, loss='softmax', **kwargs):
    model = ResNet_3D(
        num_classes=num_classes,
        loss=loss,
        layers_rgb=[3, 4, 6, 3],
        last_stride=1,
        fc_dims=None,
        dropout_p=None,
        **kwargs
    )
    return model
