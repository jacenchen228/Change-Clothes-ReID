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


class Model_3D_2stream(nn.Module):
    def __init__(self, num_classes, batch_size, **kwargs):
        super(Model_3D_2stream, self).__init__()
        self._read_configs()
        self.batch_size = batch_size
        self.feature_dim = 2048

        # fc layers
        self.feature_dim_3d = 2048
        self.fc = None

        # self.feature_dim_3d = 256
        # self.fc = self._construct_fc_layer([self.feature_dim_3d], self.feature_dim)

        # # bnneck layers
        self.feature_dim = 2048
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.bnneck_fused = nn.BatchNorm1d(self.feature_dim + self.feature_dim_3d)
        self.bnneck_rgb = nn.BatchNorm1d(self.feature_dim)
        self.bnneck_3d = nn.BatchNorm1d(self.feature_dim_3d)

        # classifiers
        # self.classifier_fused = nn.Linear(self.feature_dim + self.feature_dim_3d, num_classes, bias=False)
        self.classifier_rgb = nn.Linear(self.feature_dim, num_classes, bias=False)
        self.classifier_3d = nn.Linear(self.feature_dim_3d, num_classes, bias=False)

        print('start creating sub modules...')
        self._create_sub_modules()

        self._init_params()

        '''
            only resnet50 and hourglass is allowd currently, maybe other encoder will be allowd later.
        '''
        print('creating resnet50')
        # self.encoder = Resnet.load_Res50Model(input_channel=20)
        # self.encoder = Resnet.load_Res50Model(input_channel=1)

        # self.encoder_reid = Resnet.load_Res50Model(input_channel=3)

        self.encoder = resnet50(input_channel=1)
        self.encoder_reid = resnet50()

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
        # '''
        #     ddd smpl model, SMPL can create a mesh from beta & theta
        # '''
        # self.smpl = SMPL(self.smpl_model, obj_saveable=True, batch_size=self.batch_size, joint_type='lsp')
        #
        # '''
        #     regressor can predict betas(include beta and theta which needed by SMPL) from coder extracted from encoder in a iteratirve way
        # '''
        # fc_layers = [self.feature_dim_3d + self.total_theta_count, 1024, 1024, 85]
        # use_dropout = [True, True, False]
        # drop_prob = [0.5, 0.5, 0.5]
        # use_ac_func = [True, True, False]  # unactive the last layer
        # iterations = 3
        # self.regressor = ThetaRegressor(fc_layers, use_dropout, drop_prob, use_ac_func, iterations,
        #                                 mean_theta_path=self.smpl_mean_theta_path, batch_size=self.batch_size)
        # self.iterations = iterations

        print('finished create the encoder modules...')

    def _construct_fc_layer(self, fc_dims, input_dim, dropout_p=None):
        """Constructs fully connected layer

        Args:
            fc_dims (list or tuple): dimensions of fc layers, if None, no fc layers are constructed
            input_dim (int): input dimension
            dropout_p (float): dropout probability, if None, dropout is unused
        """
        if fc_dims is None:
            self.feature_dim = input_dim
            return None

        assert isinstance(fc_dims, (list, tuple)), 'fc_dims must be either list or tuple, but got {}'.format(
            type(fc_dims))

        layers = []
        for dim in fc_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout_p is not None:
                layers.append(nn.Dropout(p=dropout_p))
            input_dim = dim

        self.feature_dim = fc_dims[-1]

        return nn.Sequential(*layers)

    def forward(self, inputs_seg, inputs_rgb=None, return_featuremaps=False, return_params_3D=False):
        feature_3d = self.encoder(inputs_seg)
        v_3d = self.avgpool(feature_3d)
        v_3d = v_3d.view(v_3d.size(0), -1)

        if self.fc is not None:
            v_3d = self.fc(v_3d)

        v_3d_new = self.bnneck_3d(v_3d)

        # # Iterative 3D Regression
        # thetas = self.regressor(v_3d)
        # detail_info = []
        # for theta in thetas:
        #     detail_info.append(self._calc_detail_info(theta))
        #
        # # total_predict_thetas = self._accumulate_thetas(detail_info)
        # (predict_theta, _, predict_j2d, _, _) = detail_info[-1]
        # # (_, _, predict_j2d, _, _) = detail_info[-1]
        #
        # if return_params_3D:
        #     return predict_theta

        if inputs_rgb is None:
            if not self.training:
                test_feat0 = F.normalize(v_3d, p=2, dim=1)
                test_feat1 = F.normalize(v_3d_new, p=2, dim=1)

                return [test_feat0, test_feat1]

            # return predict_theta, predict_j2d
            return 0

        else:
            feature_reid = self.encoder_reid(inputs_rgb)

            if return_featuremaps:
                return feature_reid

            v_reid = self.avgpool(feature_reid)
            v_reid = v_reid.view(v_reid.size(0), -1)

            v_reid_new = self.bnneck_rgb(v_reid)

            # v_fused = torch.cat([v_3d, v_reid], dim=1)
            # v1_fused = self.avgpool(feature + feature_reid)
            # v1_fused = self.avgpool(feature * feature_reid)
            # v1_fused = v1_fused.view(v1_fused.size(0), -1)
            # v_fused_new = self.bnneck_fused(v_fused)

            if not self.training:
                # test_feat0 = torch.cat([F.normalize(v1, p=2, dim=1),
                #                         F.normalize(v1_parts, p=2, dim=1).view(v1_parts.size(0), -1)], dim=1)
                # return [test_feat0, test_feat1, test_feat2]

                test_feat0 = F.normalize(v_3d, p=2, dim=1)
                test_feat1 = F.normalize(v_3d_new, p=2, dim=1)

                test_feat2 = F.normalize(v_reid, p=2, dim=1)
                test_feat3 = F.normalize(v_reid_new, p=2, dim=1)

                # test_feat4 = F.normalize(v_fused, p=2, dim=1)
                # test_feat5 = F.normalize(v_fused_new, p=2, dim=1)

                test_feat4 = torch.cat([test_feat0, test_feat2], dim=1)
                test_feat5 = torch.cat([test_feat1, test_feat3], dim=1)

                return [test_feat0, test_feat1, test_feat2, test_feat3, test_feat4, test_feat5]
                # return [test_feat2, test_feat3]
                # return [test_feat0, test_feat1, test_feat4, test_feat5, test_feat6, test_feat7]
                # return [test_feat0, test_feat1, test_feat2, test_feat3], predict_theta

            # y_fused = self.classifier_fused(v_fused_new)
            y_reid = self.classifier_rgb(v_reid_new)
            y_3d = self.classifier_3d(v_3d_new)

            # return [y_fused, y_reid, y_3d], [v_fused, v_reid, v_3d], predict_theta, predict_j2d
            # return [y_fused, y_reid, y_3d], [v_fused, v_reid, v_3d]
            # return [y_reid, y_3d], [v_reid, v_3d], predict_theta, predict_j2d
            return [y_reid, y_3d], [v_reid, v_3d]
            # return [y_reid], [v_reid]

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
        verts, j3d, Rs = self.smpl(beta=shape, theta=pose, get_skin=True)
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


def model_3D_2stream(num_classes=150, loss='softmax', **kwargs):
    model = Model_3D_2stream(
        num_classes=num_classes,
        loss=loss,
        layers_rgb=[3, 4, 6, 3],
        last_stride=1,
        fc_dims=None,
        dropout_p=None,
        **kwargs
    )
    return model
