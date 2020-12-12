"""
Code source: https://github.com/pytorch/vision
"""
from __future__ import absolute_import
from __future__ import division

__all__ = ['pcbp4_2stream', 'pcbp6_2stream']

import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.model_zoo as model_zoo

from .pcb_backbone import pcb_p6, pcb_p4


class MyModel(nn.Module):

    def __init__(self, num_classes, encoder1, encoder2, part_num):
        super(MyModel, self).__init__()

        self.feature_dim_base = 512
        self.expansion = 4
        self.feature_dim = self.feature_dim_base * self.expansion
        self.reduced_dim = 256
        self.part_num = part_num

        # bnneck layers
        self.bnneck1 = nn.ModuleList([nn.BatchNorm1d(self.reduced_dim) for _ in range(self.part_num)])
        self.bnneck2 = nn.ModuleList([nn.BatchNorm1d(self.reduced_dim) for _ in range(self.part_num)])

        # classifiers
        self.classifier1 = nn.ModuleList([nn.Linear(self.reduced_dim, num_classes) for _ in range(self.part_num)])
        self.classifier2 = nn.ModuleList([nn.Linear(self.reduced_dim, 2*num_classes) for _ in range(self.part_num)])

        self._init_params()

        self.encoder1 = encoder1
        self.encoder2 = encoder2

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

    def forward(self, x, return_featuremaps=False):
        v_h1, v_g1 = self.encoder1(x)
        v_h2, v_g2 = self.encoder2(x)

        v_h1s_new = []
        v_h2s_new = []
        for i in range(self.part_num):
            v_h1_i = v_h1[:, :, i, :]
            v_h1_i = v_h1_i.view(v_h1_i.size(0), -1)
            v_h1s_new.append(self.bnneck1[i](v_h1_i))

            v_h2_i = v_h2[:, :, i, :]
            v_h2_i = v_h2_i.view(v_h2_i.size(0), -1)
            v_h2s_new.append(self.bnneck2[i](v_h2_i))

        v_h1_new = torch.cat(v_h1s_new, dim=1)
        v_h2_new = torch.cat(v_h2s_new, dim=1)
        if not self.training:
            test_feat0 = F.normalize(v_g1, p=2, dim=1)
            test_feat1 = F.normalize(v_g2, p=2, dim=1)

            test_feat2 = F.normalize(v_h1_new, p=2, dim=1)
            test_feat3 = F.normalize(v_h2_new, p=2, dim=1)

            test_feat4 = torch.cat([test_feat0, test_feat2], dim=1)
            test_feat5 = torch.cat([test_feat1, test_feat3], dim=1)

            return [test_feat0, test_feat1, test_feat2, test_feat3, test_feat4, test_feat5]

        y1s , y2s = [], []
        for i in range(self.part_num):
            y1 = self.classifier1[i](v_h1s_new[i])
            y2 = self.classifier2[i](v_h2s_new[i])

            y1s.append(y1)
            y2s.append(y2)

        return [y1s, v_g1], [y2s, v_g2], [v_g1, v_g2, v_h1_new, v_h2_new]


def init_pretrained_weights(model, model_url):
    """Initializes model with pretrained weights.

    Layers that don't match with pretrained layers in name or size are kept unchanged.
    """
    pretrain_dict = model_zoo.load_url(model_url)
    model_dict = model.state_dict()
    pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)

"""ResNet"""


def pcbp4_2stream(num_classes=150, **kwargs):
    encoder1 = pcb_p4(num_classes, loss='softmax', pretrained=True, **kwargs)
    encoder2 = pcb_p4(num_classes, loss='softmax', pretrained=True, **kwargs)

    model = MyModel(
        num_classes=num_classes,
        encoder1=encoder1,
        encoder2=encoder2,
        part_num=4,
        fc_dims=None,
        dropout_p=None,
        **kwargs
    )

    return model


def pcbp6_2stream(num_classes=150, **kwargs):
    encoder1 = pcb_p6(num_classes, loss='softmax', pretrained=True, **kwargs)
    encoder2 = pcb_p6(num_classes, loss='softmax', pretrained=True, **kwargs)

    model = MyModel(
        num_classes=num_classes,
        encoder1=encoder1,
        encoder2=encoder2,
        part_num=6,
        fc_dims=None,
        dropout_p=None,
        **kwargs
    )

    return model


