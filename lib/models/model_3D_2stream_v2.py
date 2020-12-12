'''
    file:   model.py

    date:   2018_05_03
    author: zhangxiong(1025679612@qq.com)
'''

import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F

from lib.lib3D import resnet50
from lib.lib3D import hmr_new, SMPL

SMPL_MODEL_DIR = '/home/jiaxing/SPIN-master/data/smpl'

class Model_3D_2stream(nn.Module):
    def __init__(self, num_classes, batch_size, **kwargs):
        super(Model_3D_2stream, self).__init__()
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

        # human body displacement predictor
        # from feature dim of 3d encoder network 2048 -> displacements of 6890 vertices
        # self.displacement_predictor = nn.Linear(2048, 6890*3, bias=False)

        self._init_params()

        # Load SMPL model
        self.smpl = SMPL(SMPL_MODEL_DIR,
                    batch_size=self.batch_size,
                    create_transl=False)

        # Create HMR model and ReID backbone
        self.estimator_3D = hmr_new(smpl_mean_params='/home/jiaxing/SPIN-master/data/smpl_mean_params.npz', pretrained=False)
        # self.estimator_3D = hmr_new(smpl_mean_params='/home/jiaxing/SPIN-master/data/smpl_mean_params.npz', pretrained=False)
        self.encoder_reid = resnet50()

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

    def forward(self, inputs1, inputs2=None, return_featuremaps=False, return_params_3D=False):
        v_3d, v_displacement, pred_rotmat, pred_betas, pred_cam, pred_thetas = self.estimator_3D(inputs1)
        pred_params = {'rotmat':pred_rotmat,
                       'beta':pred_betas,
                       'cam':pred_cam}
        pred_outputs = self.smpl(betas=pred_betas, body_pose=pred_rotmat[:,1:], global_orient=pred_rotmat[:,0].unsqueeze(1), pose2rot=False)

        # # Apply personalized displacement to original SMPL model
        # pred_displacement = self.displacement_predictor(v_displacement)
        # pred_displacement = pred_displacement.view(pred_displacement.shape[0], 6890, 3)
        # pred_outputs += pred_displacement

        if self.fc is not None:
            v_3d = self.fc(v_3d)

        v_3d_new = self.bnneck_3d(v_3d)

        if return_params_3D:
            return pred_rotmat, pred_betas, pred_cam, pred_outputs

        if inputs2 is None:
            if not self.training:
                test_feat0 = F.normalize(v_3d, p=2, dim=1)
                test_feat1 = F.normalize(v_3d_new, p=2, dim=1)

                return [test_feat0, test_feat1]

            return pred_params, pred_outputs

        else:
            feature_reid = self.encoder_reid(inputs2)

            if return_featuremaps:
                return feature_reid

            v_reid = self.avgpool(feature_reid)
            v_reid = v_reid.view(v_reid.size(0), -1)

            v_reid_new = self.bnneck_rgb(v_reid)

            # v_fused = torch.cat([v_3d, v_reid], dim=1)
            # v_fused_new = self.bnneck_fused(v_fused)

            if not self.training:
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
                # return [test_feat0, test_feat1, test_feat2, test_feat3]

            # y_fused = self.classifier_fused(v_fused_new)
            y_reid = self.classifier_rgb(v_reid_new)
            y_3d = self.classifier_3d(v_3d_new)

            # return [y_fused, y_reid, y_3d], [v_fused, v_reid, v_3d], pred_params, pred_outputs, [v_reid, v_3d, v_reid_new, v_3d_new]
            return [y_reid, y_3d], [v_reid, v_3d], pred_params, pred_outputs
            # return [y_fused, y_reid, y_3d], [v_fused, v_reid, v_3d]
            # return [y_reid, y_3d], [v_reid, v_3d], pred_params, pred_outputs
            # return [y_reid, y_3d], [v_reid, v_3d]

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


def model_3D_2stream_v2(num_classes=150, loss='softmax', **kwargs):
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
