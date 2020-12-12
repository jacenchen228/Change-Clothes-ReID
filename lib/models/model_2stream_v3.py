import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.utils.model_zoo as model_zoo

from .resnet_base import resnet50_base, resnet18_base
from lib.lib3D import hmr_new, SMPL, batch_rodrigues, Discriminator

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
}

SMPL_MODEL_DIR = '/home/jiaxing/SPIN-master/data/smpl'


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    """ Redefinition of Bottleneck residual block
        Adapted from the official PyTorch implementation
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


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


class Model_2stream(nn.Module):
    def __init__(self, num_classes, batch_size, reduce_dim=768, norm_layer=None, **kwargs):
        super(Model_2stream, self).__init__()
        self.batch_size = batch_size
        self.dilation = 1
        self.groups = 1
        self.base_width = 64

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        block = BasicBlock
        self.inplanes = 256
        self.shape_extractor = self._make_layer(block, 512, 2, stride=2)

        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.global_maxpool = nn.AdaptiveMaxPool2d(1)

        self.embedding_layer = nn.Conv2d(4096, reduce_dim, kernel_size=1, stride=1, bias=False)
        nn.init.kaiming_normal_(self.embedding_layer.weight, mode='fan_out')

        reduce_dim3d = 384
        self.embedding_layer3d = nn.Conv2d(1024, reduce_dim3d, kernel_size=1, stride=1, bias=False)
        # self.embedding_layer3d = nn.Conv2d(4096, reduce_dim3d, kernel_size=1, stride=1, bias=False)
        nn.init.kaiming_normal_(self.embedding_layer3d.weight, mode='fan_out')

        # reduce_dim_concate = 768
        # self.embedding_layer_concate = nn.Conv2d(reduce_dim+reduce_dim3d, reduce_dim_concate, kernel_size=1, stride=1, bias=False)
        # nn.init.kaiming_normal_(self.embedding_layer_concate.weight, mode='fan_out')

        # self.bn = nn.Sequential(nn.BatchNorm2d(reduce_dim+reduce_dim3d))
        # self._init_bn(self.bn)
        # self.classifier = nn.Sequential(nn.Dropout(), nn.Linear(reduce_dim+reduce_dim3d, num_classes))
        # self._init_fc(self.classifier)
        # self.bn = nn.Sequential(nn.BatchNorm2d(reduce_dim_concate))
        # self._init_bn(self.bn)
        # self.classifier = nn.Sequential(nn.Dropout(), nn.Linear(reduce_dim_concate, num_classes))
        # self._init_fc(self.classifier)
        self.bn_reid = nn.Sequential(nn.BatchNorm2d(reduce_dim))
        self._init_bn(self.bn_reid)
        self.classifier_reid = nn.Sequential(nn.Dropout(), nn.Linear(reduce_dim, num_classes))
        self._init_fc(self.classifier_reid)
        self.bn3d = nn.Sequential(nn.BatchNorm2d(reduce_dim3d))
        self._init_bn(self.bn3d)
        self.classifier3d = nn.Sequential(nn.Dropout(), nn.Linear(reduce_dim3d, num_classes))
        self._init_fc(self.classifier3d)

        # Load SMPL model
        self.smpl = SMPL(SMPL_MODEL_DIR,
                    batch_size=self.batch_size,
                    create_transl=False)

        # Create HMR model and ReID backbone
        self.reid_encoder = resnet50_base(num_classes=num_classes, pretrained=True)
        self.estimator3d = resnet18_base(num_classes=num_classes, pretrained=True)

    @staticmethod
    def _init_bn(bn):
        nn.init.constant_(bn[0].weight, 1.)
        nn.init.constant_(bn[0].bias, 0.)

    @staticmethod
    def _init_fc(fc):
        # nn.init.kaiming_normal_(fc.weight, mode='fan_out')
        nn.init.normal_(fc[1].weight, std=0.001)
        nn.init.constant_(fc[1].bias, 0.)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, inputs1, inputs2, real_params=None, return_params3D=False):
        # global reid feature extraction
        f = self.reid_encoder(inputs1)
        v1 = self.global_avgpool(f)
        v2 = self.global_maxpool(f)
        v = torch.cat([v1, v2], dim=1)
        v = self.embedding_layer(v)

        # 3D shape feature extraction
        f3d = self.estimator3d(inputs2)
        f3d = self.shape_extractor(f3d)
        v3d1 = self.global_avgpool(f3d)
        v3d2 = self.global_maxpool(f3d)
        v3d = torch.cat([v3d1, v3d2], dim=1)
        v3d = self.embedding_layer3d(v3d)

        # vconcate = torch.cat([v, v3d], dim=1)
        # vconcate = self.embedding_layer_concate(vconcate)
        # vconcate_bnneck = self.bn(vconcate).squeeze(dim=3).squeeze(dim=2)
        # vconcate = vconcate.squeeze(dim=3).squeeze(dim=2)

        v_bnneck = self.bn_reid(v).squeeze(dim=3).squeeze(dim=2)
        v = v.squeeze(dim=3).squeeze(dim=2)

        v3d_bnneck = self.bn3d(v3d).squeeze(dim=3).squeeze(dim=2)
        v3d = v3d.squeeze(dim=3).squeeze(dim=2)

        if not self.training:
            v_concate1 = torch.cat([F.normalize(v_bnneck, p=2, dim=1), F.normalize(v3d_bnneck, p=2, dim=1)], dim=1)
            v_concate2 = torch.cat([F.normalize(v, p=2, dim=1), F.normalize(v3d, p=2, dim=1)], dim=1)

            return [v_concate1, v_concate2, v_bnneck, v, v3d_bnneck, v3d]

            # return [vconcate_bnneck, vconcate, v_bnneck, v, v3d_bnneck, v3d]

        # y = self.classifier(vconcate_bnneck)
        y_reid = self.classifier_reid(v_bnneck)
        y3d = self.classifier3d(v3d_bnneck)

        # return [y, y_reid, y3d], [vconcate, v, v3d]
        return [y_reid, y3d], [v, v3d]


def init_shape_extractor(model, model_url):
    """Initializes model with pretrained weights.

    Layers that don't match with pretrained layers in name or size are kept unchanged.
    """
    pretrain_dict = model_zoo.load_url(model_url)
    model_dict = model.state_dict()
    pretrain_dict_match = dict()

    for k, v in pretrain_dict.items():
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
                layer_name += 'shape_extractor'
            else:
                layer_name += ('.' + part)

        if layer_name in model_dict and model_dict[layer_name].size() == v.size():
            print('shape extractor:', layer_name)
            pretrain_dict_match[layer_name] = v

    model_dict.update(pretrain_dict_match)
    model.load_state_dict(model_dict)


def model_2stream_v3(num_classes=150, loss='softmax', **kwargs):
    model = Model_2stream(
        num_classes=num_classes,
        loss=loss,
        layers=[3, 4, 6, 3],
        last_stride=1,
        fc_dims=None,
        dropout_p=None,
        **kwargs
    )
    init_shape_extractor(model, model_urls['resnet18'])

    return model
