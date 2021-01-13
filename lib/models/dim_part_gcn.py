"""
Code source: https://github.com/pytorch/vision
"""
from __future__ import absolute_import
from __future__ import division

__all__ = ['dim_part_gcn50', 'dim_part_gcn34']

import random

import torch
from torch import nn
from torch.nn import functional as F
import torchvision
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable

from lib.utils.DIM.model import Discriminator
from lib.utils.gcn_layer_ori import GraphConvolution
# from lib.utils.gcn_layer import GraphConvolution
from lib.utils import GeneralizedMeanPoolingP, GeneralizedMeanPooling
from lib.utils import CircleSoftmax

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


class DimReduceLayer(nn.Module):

    def __init__(self, in_channels, out_channels, nonlinear):
        super(DimReduceLayer, self).__init__()
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))

        if nonlinear == 'relu':
            layers.append(nn.ReLU(inplace=True))
        elif nonlinear == 'leakyrelu':
            layers.append(nn.LeakyReLU(0.1))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class MyModel(nn.Module):

    def __init__(self, num_classes, loss, block_rgb, layers_rgb, block_contour, layers_contour, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, last_stride=2, fc_dims=None, dropout_p=None, part_num=3, part_weight=1.0, **kwargs):
        super(MyModel, self).__init__()
        self.cnt = 0

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.loss = loss
        self.feature_dim_base = 512
        self.feature_dim = self.feature_dim_base * block_rgb.expansion
        self.inplanes = 64
        self.dilation = 1
        self.part_num = part_num
        self.part_weight = part_weight
        self.layer_num = 4  # for resnet, layer_num = 4
        self.reduced_dim = 256
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
        self.layer1 = self._make_layer(block_rgb, 64, layers_rgb[0])
        self.layer2 = self._make_layer(block_rgb, 128, layers_rgb[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block_rgb, 256, layers_rgb[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block_rgb, self.feature_dim_base, layers_rgb[3], stride=last_stride,
                                       dilate=replace_stride_with_dilation[2])
        self.inplanes = 256 * block_rgb.expansion
        # self.layer4_part = self._make_layer(block_rgb, self.feature_dim_base, layers_rgb[3], stride=last_stride,
        #                                dilate=replace_stride_with_dilation[2])
        self.conv5 = DimReduceLayer(self.feature_dim_base*block_rgb.expansion, self.reduced_dim, nonlinear='relu')

        # FC layers definition
        if fc_dims is None:
            self.fc = None
        else:
            self.fc = self._construct_fc_layer(fc_dims, 512 * block_rgb.expansion, dropout_p)

        # Backbone network for contour feature extraction
        self.inplanes = 64
        self.conv1_contour = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1_contour = nn.BatchNorm2d(64)
        self.layer1_contour = self._make_layer(block_contour, 64, layers_contour[0])
        self.layer2_contour = self._make_layer(block_contour, 128, layers_contour[1], stride=2)
        self.layer3_contour = self._make_layer(block_contour, 256, layers_contour[2], stride=2)
        self.layer4_contour = self._make_layer(block_contour, self.feature_dim_base, layers_contour[3], stride=last_stride)
        self.conv5_contour = DimReduceLayer(self.feature_dim_base*block_contour.expansion, self.reduced_dim, nonlinear='relu')

        # Network for contour graph modeling
        # self.parts_avgpool_contour = nn.AdaptiveAvgPool2d((self.part_num, 1))
        self.feature_dim_gnn = self.feature_dim_base * block_contour.expansion

        # Pooling layers
        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.parts_avgpool = nn.AdaptiveAvgPool2d((self.part_num, 1))
        self.parts_avgpool_contour = nn.AdaptiveAvgPool2d((self.part_num, 3))
        # # Gem pooling layers
        # Version with learnable parameters
        # self.global_gempool = GeneralizedMeanPoolingP(output_size=(1, 1))
        # self.global_gempool_contour = GeneralizedMeanPoolingP(output_size=(1, 1))
        # self.parts_gempool = GeneralizedMeanPoolingP(output_size=(self.part_num, 1))
        # self.parts_gempool_contour = GeneralizedMeanPoolingP(output_size=(self.part_num, 1))
        # Version without learnable parameters
        # self.global_gempool = GeneralizedMeanPooling(norm=3, output_size=(1, 1))
        # self.global_gempool_contour = GeneralizedMeanPooling(norm=3, output_size=(1, 1))
        # self.parts_gempool = GeneralizedMeanPooling(norm=3, output_size=(self.part_num, 1))
        # self.parts_gempool_contour = GeneralizedMeanPooling(norm=3, output_size=(self.part_num, 1))

        # Bnneck layers
        self.bnneck_rgb = nn.BatchNorm1d(self.feature_dim_base*block_rgb.expansion)
        self.bnneck_rgb.bias.requires_grad_(False)  # no shift
        self.bnneck_rgb_part = nn.ModuleList([nn.BatchNorm1d(self.reduced_dim) for _ in range(self.part_num)])
        for module in self.bnneck_rgb_part:
            module.bias.requires_grad_(False)   # no shift
        self.bnneck_contour = nn.BatchNorm1d(self.feature_dim_base*block_contour.expansion)
        self.bnneck_contour.bias.requires_grad_(False)  # no shift
        self.bnneck_contour_part = nn.ModuleList([nn.BatchNorm1d(self.reduced_dim) for _ in range(self.part_num)])
        for module in self.bnneck_contour_part:
            module.bias.requires_grad_(False)   # no shift

        # Classifiers
        self.classifier = nn.Linear(self.feature_dim_base*block_rgb.expansion, num_classes, bias=False)
        self.classifier_contour = nn.Linear(self.feature_dim_base*block_contour.expansion, num_classes, bias=False)
        # self.classifiers_part = nn.ModuleList([nn.Linear(self.reduced_dim, num_classes) for _ in range(self.part_num)])
        # self.classifiers_contour_part = nn.ModuleList([nn.Linear(self.reduced_dim, num_classes) for _ in range(self.part_num)])
        # cricle softmax classifiers
        # self.classifier = CircleSoftmax(self.feature_dim_base*block_rgb.expansion, num_classes, scale=64, margin=0.35)
        # self.classifier_contour = CircleSoftmax(self.feature_dim_base*block_contour.expansion, num_classes, scale=64, margin=0.35)

        # Specify output dim for each layer
        layer_base_dims = [64, 128, 256, 512]
        part_reduced_dims = [64, 128, 256, 512]
        global_discriminator_layers = [[256, 64, 16], [512, 128, 32], [1024, 256, 64, 16], [2048, 512, 128, 32]]
        part_discriminator_layers = [[64, 16], [128, 32], [256, 64, 16], [512, 128, 32]]

        # Part reduced layers
        self.appearance_reduced_layers = nn.ModuleList([DimReduceLayer(layer_base_dims[idx]*block_rgb.expansion,
                                        part_reduced_dims[idx], nonlinear='relu') for idx in range(self.layer_num)])
        self.contour_reduced_layers = nn.ModuleList([DimReduceLayer(layer_base_dims[idx]*block_contour.expansion,
                                        part_reduced_dims[idx], nonlinear='relu') for idx in range(self.layer_num)])

        # Contour Hierarchical Graph modeling module
        self.contour_gnns = []
        self.contour_gnn_bns = []
        for idx in range(self.layer_num):
            base_dim = layer_base_dims[idx]
            gnns = nn.ModuleList([GraphConvolution(base_dim*block_contour.expansion, base_dim*block_contour.expansion, bias=True)
                                       for _ in range(self.part_num + 1)])
            bns_gnn = nn.ModuleList([nn.BatchNorm1d(base_dim*block_contour.expansion) for _ in range(self.part_num + 1)])

            self.contour_gnns.append(gnns)
            self.contour_gnn_bns.append(bns_gnn)
        self.contour_gnns = nn.ModuleList(self.contour_gnns)
        self.contour_gnn_bns = nn.ModuleList(self.contour_gnn_bns)

        # Mutual information learning module
        # self.global_discriminators = []
        self.part_discriminators = []
        for idx in range(self.layer_num):
            global_dims = global_discriminator_layers[idx]
            part_dims = part_discriminator_layers[idx]
            base_dim = layer_base_dims[idx]
            part_reduced_dim = part_reduced_dims[idx]

            # global_discriminator = Discriminator(in_dim1=base_dim*block_rgb.expansion, in_dim2=base_dim*block_contour.expansion,
            #                                       layers_dim=global_dims)
            part_discriminators = nn.ModuleList([Discriminator(part_reduced_dim, in_dim2=part_reduced_dim, layers_dim=part_dims)
                                                 for _ in range(self.part_num)])

            # self.global_discriminators.append(global_discriminator)
            self.part_discriminators.append(part_discriminators)
        # self.global_discriminators = nn.ModuleList(self.global_discriminators)
        self.part_discriminators = nn.ModuleList(self.part_discriminators)

        # for name, module in self.named_modules():
        #     print(name, module)

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

    def normalize(self, adj_mat):
        dim = adj_mat.shape[2]
        normalize_base = torch.sum(adj_mat, dim=2)
        normalize_base = torch.pow(normalize_base, -1)
        normalize_base = normalize_base.unsqueeze(2).repeat(1, 1, dim)

        adj_mat_nor = adj_mat * normalize_base

        return adj_mat_nor

    def cos_sim(self, feats):
        feats_normed = F.normalize(feats, p=2, dim=2)
        adj_mat = torch.matmul(feats_normed, feats_normed.transpose(1, 2))

        return adj_mat

    def hierarchical_graph_modeling(self, v, layer_idx):
        # fine-grained scale graph learning
        v = v.permute(0, 2, 3, 1)
        v_local = torch.zeros(v.shape[0], v.shape[1], v.shape[3]).cuda()
        for idx in range(v.shape[1]):
            part_i = v[:, idx, ...]
            adj_mat_i = self.cos_sim(part_i)
            adj_mat_i = self.normalize(adj_mat_i)
            part_i_new = self.contour_gnns[layer_idx][idx](part_i, adj_mat_i)
            part_i_new = self.contour_gnn_bns[layer_idx][idx](part_i_new.transpose(1, 2))
            part_i_new = part_i_new.transpose(1, 2)
            part_i_new = self.relu(part_i_new)

            v_local[:, idx, :] = torch.max(part_i_new, dim=1)[0]

        # coarse-grained scale graph learning
        adj_mat = self.cos_sim(v_local)
        adj_mat = self.normalize(adj_mat)
        v_global = self.contour_gnns[layer_idx][-1](v_local, adj_mat)
        v_global = self.contour_gnn_bns[layer_idx][-1](v_global.transpose(1, 2))
        v_global = v_global.transpose(1, 2)
        v_global = self.relu(v_global)
        v_global = torch.max(v_global, dim=1)[0]

        return v_global, v_local

    def _get_features(self, x1, x2):
        # To store intermedia features
        # appearance_global_features = []
        appearance_part_features = []
        # contour_global_features = []
        contour_part_features = []

        # Feature extraction for rgb images
        x1 = self.conv1(x1)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        x1 = self.maxpool(x1)

        # Layer 1 -> Layer 4
        x1 = self.layer1(x1)
        # appearance_global_features.append(self.global_avgpool(x1).view(x1.size(0), -1))
        appearance_part_features.append(self.appearance_reduced_layers[0](
            self.parts_avgpool(x1)).view(x1.size(0), -1, self.part_num))

        x1 = self.layer2(x1)
        # appearance_global_features.append(self.global_avgpool(x1).view(x1.size(0), -1))
        appearance_part_features.append(self.appearance_reduced_layers[1](
            self.parts_avgpool(x1)).view(x1.size(0), -1, self.part_num))

        x1 = self.layer3(x1)
        # appearance_global_features.append(self.global_avgpool(x1).view(x1.size(0), -1))
        appearance_part_features.append(self.appearance_reduced_layers[2](
            self.parts_avgpool(x1)).view(x1.size(0), -1, self.part_num))

        x1 = self.layer4(x1)
        # appearance_global_features.append(self.global_avgpool(x1).view(x1.size(0), -1))
        appearance_part_features.append(self.appearance_reduced_layers[3](
            self.parts_avgpool(x1)).view(x1.size(0), -1, self.part_num))

        # Feature extraction for contour images
        x2 = self.conv1_contour(x2)
        x2 = self.bn1_contour(x2)
        x2 = self.relu(x2)
        x2 = self.maxpool(x2)

        # Layer1 -> Layer4
        x2 = self.layer1_contour(x2)
        contour_global, contour_part = self.hierarchical_graph_modeling(self.parts_avgpool_contour(x2), layer_idx=0)
        # contour_global_features.append(contour_global)
        contour_part = contour_part.transpose(1, 2).unsqueeze(3)
        contour_part = self.contour_reduced_layers[0](contour_part)
        contour_part_features.append(contour_part.view(x2.size(0), -1, self.part_num))

        x2 = self.layer2_contour(x2)
        contour_global, contour_part = self.hierarchical_graph_modeling(self.parts_avgpool_contour(x2), layer_idx=1)
        # contour_global_features.append(contour_global)
        contour_part = contour_part.transpose(1, 2).unsqueeze(3)
        contour_part = self.contour_reduced_layers[1](contour_part)
        contour_part_features.append(contour_part.view(x2.size(0), -1, self.part_num))

        x2 = self.layer3_contour(x2)
        contour_global, contour_part = self.hierarchical_graph_modeling(self.parts_avgpool_contour(x2), layer_idx=2)
        # contour_global_features.append(contour_global)
        contour_part = contour_part.transpose(1, 2).unsqueeze(3)
        contour_part = self.contour_reduced_layers[2](contour_part)
        contour_part_features.append(contour_part.view(x2.size(0), -1, self.part_num))

        x2 = self.layer4_contour(x2)
        contour_global, contour_part = self.hierarchical_graph_modeling(self.parts_avgpool_contour(x2), layer_idx=3)
        # contour_global_features.append(contour_global)
        contour_part = contour_part.transpose(1, 2).unsqueeze(3)
        contour_part = self.contour_reduced_layers[3](contour_part)
        contour_part_features.append(contour_part.view(x2.size(0), -1, self.part_num))

        return x1, x2, appearance_part_features, contour_part_features

    def deep_info_max(self, part_features1, part_features2):
        # To store mutual information items
        ejs_part = []
        ems_part = []

        for layer_idx, (v1_parts, v2_parts) in enumerate(zip(part_features1, part_features2)):
            # # Calculate global mutual information
            # random_idxs = list(range(v2.size(0)))
            # random.shuffle(random_idxs)
            # v2_shuffle = v2[random_idxs]

            # ej = self.global_discriminators[layer_idx](v1, v2)
            # em = self.global_discriminators[layer_idx](v1, v2_shuffle)

            # Calculate local mutual information
            ej_part = []
            em_part = []
            for idx in range(self.part_num):
                v1_part_i = v1_parts[:, :, idx]
                v1_part_i = v1_part_i.view(v1_part_i.size(0), -1)
                v2_part_i = v2_parts[:, :, idx]
                random_idxs_i = list(range(v2_part_i.size(0)))
                random.shuffle(random_idxs_i)
                v2_part_i_shuffle = v2_part_i[random_idxs_i]

                ej_part_i = self.part_discriminators[layer_idx][idx](v1_part_i, v2_part_i)
                em_part_i = self.part_discriminators[layer_idx][idx](v1_part_i, v2_part_i_shuffle)

                ej_part.append(ej_part_i)
                em_part.append(em_part_i)

            ejs_part.append(ej_part)
            ems_part.append(em_part)

        return ejs_part, ems_part

    def forward(self, x1, x2, return_featuremaps=False, targets=None):
        f1, f2, appearance_part_features, contour_part_features\
            = self._get_features(x1, x2)

        if return_featuremaps:
            return f1

        # generate rgb features (global + part)
        v1 = self.global_avgpool(f1)
        # v1 = self.global_gempool(f1)
        v1 = v1.view(v1.size(0), -1)
        v1_parts = self.parts_avgpool(f1)
        # v1_parts = self.parts_gempool(f1)
        v1_parts = self.conv5(v1_parts)
        v1_parts = v1_parts.view(v1_parts.size(0), v1_parts.size(1), -1)

        # generate contour features (global + part)
        v2 = self.global_avgpool(f2)
        # v2 = self.global_gempool_contour(f2)
        v2 = v2.view(v2.size(0), -1)
        v2_parts = self.parts_avgpool(f2)
        # v2_parts = self.parts_gempool_contour(f2)
        v2_parts = self.conv5_contour(v2_parts)
        v2_parts = v2_parts.view(v2_parts.size(0), v2_parts.size(1), -1)

        # bnneck operation
        v1_new = self.bnneck_rgb(v1)
        v1_parts_new = torch.zeros_like(v1_parts).squeeze(-1)
        for idx in range(self.part_num):
            v1_parts_new[:, :, idx] = self.bnneck_rgb_part[idx](v1_parts[:, :, idx].view(v1_parts.size(0), -1))
        v2_new = self.bnneck_contour(v2)
        v2_parts_new = torch.zeros_like(v2_parts).squeeze(-1)
        for idx in range(self.part_num):
            v2_parts_new[:, :, idx] = self.bnneck_contour_part[idx](v2_parts[:, :, idx].view(v2_parts.size(0), -1))

        # when evaluation
        if not self.training:
            global_feat = F.normalize(v1_new, p=2, dim=1)
            part_feat = F.normalize(v1_parts_new, p=2, dim=1).view(v1_parts_new.size(0), -1)
            concate_feat = torch.cat([global_feat, self.part_weight*part_feat], dim=1)
            concate_feat1 = F.normalize(torch.cat([v1_new, self.part_weight*
                                                  v1_parts_new.view(v1_parts_new.size(0), -1)], dim=1), p=2, dim=1)

            # global_contour_feat = F.normalize(v2_new, p=2, dim=1)
            # part_contour_feat = F.normalize(v2_parts_new.view(v2_parts_new.size(0), -1), p=2, dim=1)
            # concate_contour_feat = F.normalize(torch.cat([global_contour_feat, part_contour_feat], dim=1), p=2, dim=1)

            concate_contour_feat = F.normalize(torch.cat([v2_new, self.part_weight*
                                                  v2_parts_new.view(v2_parts_new.size(0), -1)], dim=1), p=2, dim=1)

            return [global_feat, part_feat, concate_feat, concate_feat1, concate_contour_feat]

        # predict probability
        y1 = self.classifier(v1_new)
        # y1 = self.classifier(v1_new, targets)   # specify for circle softmax classifier
        # y1_parts = []
        # for idx in range(self.part_num):
        #     v1_part_i = v1_parts_new[:, :, idx]
        #     v1_part_i = v1_part_i.view(v1_part_i.size(0), -1)
        #     y1_part_i = self.classifiers_part[idx](v1_part_i)
        #     y1_parts.append(y1_part_i)
        y2 = self.classifier_contour(v2_new)
        # y2 = self.classifier(v2_new, targets)   # specify for circle softmax classifier
        # y2_parts = []
        # for idx in range(self.part_num):
        #     v2_part_i = v2_parts_new[:, :, idx]
        #     v2_part_i = v2_part_i.view(v2_part_i.size(0), -1)
        #     y2_part_i = self.classifiers_contour_part[idx](v2_part_i)
        #     y2_parts.append(y2_part_i)

        # calcuate mutual information
        ej_parts, em_parts = self.deep_info_max(appearance_part_features, contour_part_features)

        # return [y1, y1_parts, y2, y2_parts], [v1, v2, v1_parts, v2_parts], \
        #        ej, em, ej_part, em_part
        return [y1, y2], [v1, v2, v1_parts_new, v2_parts_new], ej_parts, em_parts


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

    pretrain_dict1_match = {k: v for k, v in pretrain_dict1.items() if
                            k in model_dict and model_dict[k].size() == v.size()}
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
        name_list = k.split('.')
        layer_name = ''
        for idx, part in enumerate(name_list):
            if idx == 0:
                layer_name += (part + '_contour')
            else:
                layer_name += ('.' + part)

        if layer_name in model_dict and model_dict[layer_name].size() == v.size():
            print('Layer {} initialization done!'.format(layer_name))
            pretrain_dict2_match[layer_name] = v

        # # Initialize img part layer4
        # if 'layer4' in name_list:
        #     print('Layer4 Part Initialization Done!')
        #     layer_name1 = ''
        #     for idx, part in enumerate(name_list):
        #         if idx == 0:
        #             layer_name1 += (part + '_part')
        #         else:
        #             layer_name1 += ('.' + part)
        #     if layer_name1 in model_dict and model_dict[layer_name1].size() == v.size():
        #         pretrain_dict2_match[layer_name1] = v

    model_dict.update(pretrain_dict2_match)

    model.load_state_dict(model_dict)
    print("Initialized model with pretrained weights from {}".format(model_url1))
    print("Initialized model with pretrained weights from {}".format(model_url2))

def dim_part_gcn50(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = MyModel(
        num_classes=num_classes,
        loss=loss,
        block_rgb=Bottleneck,
        layers_rgb=[3, 4, 6, 3],
        block_contour=Bottleneck,
        layers_contour=[3, 4, 6, 3],
        last_stride=1,
        fc_dims=None,
        dropout_p=None,
        **kwargs
    )

    if pretrained:
        init_pretrained_weights_hybrid(model, model_urls['resnet50'], model_urls['resnet50'])

    return model


def dim_part_gcn34(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = MyModel(
        num_classes=num_classes,
        loss=loss,
        block_rgb=Bottleneck,
        layers_rgb=[3, 4, 6, 3],
        block_contour=BasicBlock,
        layers_contour=[3, 4, 6, 3],
        last_stride=1,
        fc_dims=None,
        dropout_p=None,
        **kwargs
    )

    if pretrained:
        init_pretrained_weights_hybrid(model, model_urls['resnet50'], model_urls['resnet34'])

    return model


