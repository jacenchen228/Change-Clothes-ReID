"""
Code source: https://github.com/pytorch/vision
"""
from __future__ import absolute_import
from __future__ import division

__all__ = ['dim_part_resnet50', 'dim_img_part_resnet50', 'dim_img_part_resnet34', 'dim_img_part_resnet18']

import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
import torchvision
import torch.utils.model_zoo as model_zoo

# from torchreid.utils.roi.roi_layers import ROIAlign, ROIPool
from torchreid.utils.gcn_layer import GraphConvolution, SimlarityLayer
from torchreid.utils.gat.models import GAT
from torchreid.utils.DIM.model_ori import GlobalDiscriminator, LocalDiscriminator, PartDiscriminator
from torchreid.utils import extractColorFeats


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
}

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """Residual network.

    Reference:
        - He et al. Deep Residual Learning for Image Recognition. CVPR 2016.
        - Xie et al. Aggregated Residual Transformations for Deep Neural Networks. CVPR 2017.

    Public keys:
        - ``resnet18``: ResNet18.
        - ``resnet34``: ResNet34.
        - ``resnet50``: ResNet50.
        - ``resnet101``: ResNet101.
        - ``resnet152``: ResNet152.
        - ``resnext50_32x4d``: ResNeXt50.
        - ``resnext101_32x8d``: ResNeXt101.
        - ``resnet50_fc512``: ResNet50 + FC.
    """

    def __init__(self, num_classes, loss, block, layers, block_aux, layers_aux, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, last_stride=2, fc_dims=None, dropout_p=None, pretrained=True, **kwargs):
        super(ResNet, self).__init__()
        self.cnt = 0

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.loss = loss
        self.feature_dim = 512 * block.expansion
        self.feature_dim_aux = 512 * block_aux.expansion
        self.inplanes = 64
        self.dilation = 1
        self.part_num = 3
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=last_stride,
                                       dilate=replace_stride_with_dilation[2])
        self.inplanes = 256 * block.expansion
        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = self._construct_fc_layer(fc_dims, 512 * block.expansion, dropout_p)
        self.classifier = nn.Linear(self.feature_dim, num_classes)

        # backbone network for auxiliary modality
        self.inplanes = 64  # 因为前面self._make_layer会改变self.inplanes的值
        self.conv1_aux = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1_aux = nn.BatchNorm2d(64)
        self.layer1_aux = self._make_layer(block_aux, 64, layers_aux[0])
        self.layer2_aux = self._make_layer(block_aux, 128, layers_aux[1], stride=2)
        self.layer3_aux = self._make_layer(block_aux, 256, layers_aux[2], stride=2)
        self.layer4_aux = self._make_layer(block_aux, 512, layers_aux[3], stride=last_stride)
        self.classifier_aux = nn.Linear(self.feature_dim_aux, num_classes)

        # fusion of feat and feat_aux
        self.classifier_fuse = nn.Linear(self.feature_dim_aux, num_classes)

        # network for part modeling
        self.part_avgpool = nn.AdaptiveAvgPool2d((self.part_num, 1))
        self.feature_dim_gcn1 = 512 * block.expansion
        self.feature_dim_gcn2 = 512 * block_aux.expansion
        self.gcn1 = GraphConvolution(self.feature_dim_gcn1, self.feature_dim_gcn1)
        # self.gcn2 = GraphConvolution(self.feature_dim_gcn2, self.feature_dim_gcn2)


        # Mutual Learning Module
        self.part_height = 16 // self.part_num
        self.residue = 16 % self.part_num
        discriminator_list = list()
        for i in range(self.part_num):
            height = self.part_height

            flag = self.part_num - 1 - i
            if flag < self.residue:
                height += 1

            discriminator_list.append(GlobalDiscriminator(height, 8, in_feature_dim=512 * block.expansion,
                                 feature_dim=256))
        self.discriminators = nn.ModuleList(discriminator_list)

        self._init_params()

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

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

        assert isinstance(fc_dims, (list, tuple)), 'fc_dims must be either list or tuple, but got {}'.format(type(fc_dims))

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

    def featuremaps(self, x1, x2):
        x1 = self.conv1(x1)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        x1 = self.maxpool(x1)
        x1 = self.layer1(x1)
        x1 = self.layer2(x1)
        x1 = self.layer3(x1)
        x1 = self.layer4(x1)

        x2 = self.conv1_aux(x2)
        x2 = self.bn1_aux(x2)
        x2 = self.relu(x2)
        x2 = self.maxpool(x2)
        x2 = self.layer1_aux(x2)
        x2 = self.layer2_aux(x2)
        x2 = self.layer3_aux(x2)
        x2 = self.layer4_aux(x2)

        x_fuse = x1 * x2

        return x1, x2, x_fuse

    def cos_sim(self, feats):
        feats_normed = F.normalize(feats, p=2, dim=2)
        adj_mat = torch.matmul(feats_normed, feats_normed.transpose(1, 2))

        return adj_mat

    def forward(self, x1, x2, return_featuremaps=False):
        f1, f2, f_fuse = self.featuremaps(x1, x2)

        if return_featuremaps:
            return f1

        v1 = self.part_avgpool(f1)
        v2 = self.part_avgpool(f2)
        v_fuse = self.part_avgpool(f_fuse)

        # GCN feature embedding
        v1 = v1.transpose(1, 2)
        v1 = v1.view(v1.size(0), v1.size(1), -1)
        v2 = v2.transpose(1, 2)
        v2 = v2.view(v2.size(0), v2.size(1), -1)
        v_fuse = v_fuse.transpose(1, 2)
        v_fuse = v_fuse.view(v_fuse.size(0), v_fuse.size(1), -1)

        adj1 = self.cos_sim(v1)
        # adj2 = self.cos_sim(v2)

        v1 = self.gcn1(v1, adj1)
        # v2 = self.gcn2(v2, adj2)

        v1 = torch.max(v1, dim=1)[0]
        v2 = torch.max(v2, dim=1)[0]
        v_fuse = torch.max(v_fuse, dim=1)[0]

        if self.fc is not None:
            v_fuse = self.fc(v_fuse)

        if not self.training:
            test_feat1 = v1
            test_feat2 = v2
            test_feat3 = v_fuse
            test_feat4 = torch.cat([test_feat1, test_feat2], dim=1)
            test_feat5 = torch.cat([test_feat1, test_feat3], dim=1)
            test_feat6 = torch.cat([test_feat2, test_feat3], dim=1)
            test_feat7 = torch.cat([test_feat1, test_feat2, test_feat3], dim=1)

            return [test_feat1, test_feat2, test_feat3, test_feat4, test_feat5, test_feat6, test_feat7]

        y1 = self.classifier(v1)
        y2 = self.classifier_aux(v2)
        y_fuse = self.classifier_fuse(v_fuse)

        # Mutual learning for part feat
        ej_part, em_part = [], []
        tmp_height = 0
        for i in range(self.part_num):
            height = self.part_height
            flag = self.part_num - 1 - i
            if flag < self.residue:
                height += 1

            f1_i = f1[:, :, tmp_height:tmp_height+height, :]
            f2_i = f2[:, :, tmp_height:tmp_height+height, :]
            prime_i = torch.cat((f2_i[1:], f2_i[0].unsqueeze(0)), dim=0)

            ej_part_i = self.discriminators[i](f1_i, f2_i)
            em_part_i = self.discriminators[i](f1_i, prime_i)

            ej_part.append(ej_part_i)
            em_part.append(em_part_i)

            tmp_height += height

        # ej_part, em_part = [], []
        # prime_part = torch.cat((v_aux[1:], v_aux[0].unsqueeze(0)), dim=0)
        # for i in range(self.part_num):
        #     fpart_i = fpart[:, i, :]
        #     # fpart_i = fpart_ori[:, i, :]
        #
        #     # ej_part_i = self.discriminators_part[i](fpart_i, fpart_aux)
        #     ej_part_i = self.discriminators_part[i](fpart_i, v_aux)
        #     em_part_i = self.discriminators_part[i](fpart_i, prime_part)
        #
        #     ej_part.append(ej_part_i)
        #     em_part.append(em_part_i)

        if self.loss == 'softmax':
            return y1, y2, y_fuse, ej_part, em_part
        elif self.loss == 'triplet':
            return y1, v1
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))

# def init_pretrained_weights(model, model_url):
#     """Initializes model with pretrained weights.
#
#     Layers that don't match with pretrained layers in name or size are kept unchanged.
#     """
#     pretrain_dict = model_zoo.load_url(model_url)
#     model_dict = model.state_dict()
#     pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
#     model_dict.update(pretrain_dict)
#     model.load_state_dict(model_dict)

def init_pretrained_weights(model, model_url):
    """Initializes model with pretrained weights.

    Layers that don't match with pretrained layers in name or size are kept unchanged.
    """
    pretrain_dict = model_zoo.load_url(model_url)
    model_dict = model.state_dict()
    pretrain_dict_ = dict()
    for k, v in pretrain_dict.items():
        # print(k, ':', v.size())
        if k in model_dict and model_dict[k].size() == v.size():
            pretrain_dict_[k] = v
            # print(k)
        elif k == 'conv1.weight':
            print('con1.weight initialization done!')
            pretrain_dict_[k] = torch.mean(v, dim=1, keepdim=True)

    model_dict.update(pretrain_dict_)
    model.load_state_dict(model_dict)

def init_pretrained_weights_hybrid(model, model_url1, model_url2):
    """
    Initialize model with pretrained weights.
    Layers that don't match with pretrained layers in name or size are kept unchanged.
    """
    '''
    model_url1: 对应img模块的网络架构
    model_url2: 对应sketch模块的网络架构
    '''
    pretrain_dict1 = model_zoo.load_url(model_url1)
    pretrain_dict2 = model_zoo.load_url(model_url2)
    model_dict = model.state_dict()

    pretrain_dict1_match = {k: v for k, v in pretrain_dict1.items() if k in model_dict and model_dict[k].size() == v.size()}
    model_dict.update(pretrain_dict1_match)

    # 利用预训练模型初始化sketch feature的网络
    pretrain_dict2_match = {}
    for k, v in pretrain_dict2.items():
        '''
        k的格式如：
        layer4.2.conv1.weight    layer4.2.conv1.weight
        layer4.2.bn1.running_mean    layer4.2.bn1.running_var
        layer4.2.bn1.weight    layer4.2.bn1.bias
        '''
        nameList = k.split('.')
        layer_name = ''
        for idx, part in enumerate(nameList):
            if idx == 0:
                layer_name += (part + '_aux')
            else:
                layer_name += ('.' + part)

        if layer_name in model_dict and model_dict[layer_name].size() == v.size():
            pretrain_dict2_match[layer_name] = v
    model_dict.update(pretrain_dict2_match)

    model.load_state_dict(model_dict)
    print("Initialized model with pretrained weights from {}".format(model_url1))
    print("Initialized model with pretrained weights from {}".format(model_url2))

"""ResNet"""
def dim_part_resnet50(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = ResNet(
        num_classes=num_classes,
        loss=loss,
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        block_aux=BasicBlock,
        layers_aux=[2, 2, 2, 2],
        last_stride=1,
        fc_dims=None,
        dropout_p=None,
        **kwargs
    )
    if pretrained:
        init_pretrained_weights_hybrid(model, model_urls['resnet50'], model_urls['resnet18'])
    return model

# def dim_img_global_resnet50(num_classes, loss='softmax', pretrained=True, **kwargs):
#     model = ResNet(
#         num_classes=num_classes,
#         loss=loss,
#         block=Bottleneck,
#         layers=[3, 4, 6, 3],
#         block_aux=BasicBlock,
#         layers_aux=[2, 2, 2, 2],
#         last_stride=1,
#         fc_dims=None,
#         dropout_p=None,
#         contour_based=False,
#         **kwargs
#     )
#     if pretrained:
#         init_pretrained_weights_hybrid(model, model_urls['resnet50'], model_urls['resnet18'])
#     return model

def dim_img_part_resnet50(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = ResNet(
        num_classes=num_classes,
        loss=loss,
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        block_aux=Bottleneck,
        layers_aux=[3, 4, 6, 3],
        last_stride=1,
        fc_dims=None,
        dropout_p=None,
        **kwargs
    )
    if pretrained:
        init_pretrained_weights_hybrid(model, model_urls['resnet50'], model_urls['resnet50'])
    return model

# def dim_img_global_resnet34(num_classes, loss='softmax', pretrained=True, **kwargs):
#     model = ResNet(
#         num_classes=num_classes,
#         loss=loss,
#         block=BasicBlock,
#         layers=[3, 4, 6, 3],
#         last_stride=1,
#         fc_dims=None,
#         dropout_p=None,
#         contour_based=False,
#         **kwargs
#     )
#     if pretrained:
#         init_pretrained_weights_hybrid(model, model_urls['resnet34'], model_urls['resnet18'])
#     return model

def dim_img_part_resnet34(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = ResNet(
        num_classes=num_classes,
        loss=loss,
        block=BasicBlock,
        layers=[3, 4, 6, 3],
        block_aux=BasicBlock,
        layers_aux=[3, 4, 6, 3],
        last_stride=1,
        fc_dims=None,
        dropout_p=None,
        **kwargs
    )
    if pretrained:
        init_pretrained_weights_hybrid(model, model_urls['resnet34'], model_urls['resnet34'])
    return model

# def dim_img_global_resnet18(num_classes, loss='softmax', pretrained=True, **kwargs):
#     model = ResNet(
#         num_classes=num_classes,
#         loss=loss,
#         block=BasicBlock,
#         layers=[2, 2, 2, 2],
#         last_stride=1,
#         fc_dims=None,
#         dropout_p=None,
#         contour_based=False,
#         pretrained=pretrained,
#         **kwargs
#     )
#     if pretrained:
#         init_pretrained_weights_hybrid(model, model_urls['resnet18'], model_urls['resnet18'])
#     return model

def dim_img_part_resnet18(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = ResNet(
        num_classes=num_classes,
        loss=loss,
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        block_aux=BasicBlock,
        layers_aux=[2, 2, 2, 2],
        last_stride=1,
        fc_dims=None,
        dropout_p=None,
        **kwargs
    )
    if pretrained:
        init_pretrained_weights_hybrid(model, model_urls['resnet18'], model_urls['resnet18'])
    return model


