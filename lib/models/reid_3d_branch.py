import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.nn import functional as F

from lib.lib3D import hmr_new, SMPL, Discriminator, batch_rodrigues, allInOne

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


class My3DBranch(nn.Module):
    def __init__(self, batch_size=16, num_classes=150, **kwargs):
        super(My3DBranch, self).__init__()

        self.batch_size = batch_size

        # reid sub-networks

        # when backbone = resnet50
        block = Bottleneck

        # # when backbone = resnet18
        # block = BasicBlock

        self.feature_dim3d = 512 * block.expansion
        self.inplanes = 256 * block.expansion
        self.shape_extractor = self._make_layer(block, 512, 3, stride=2)

        # ReID global feature
        self.inplanes = 128 * block.expansion
        # self.layer2_reid = self._make_layer(block, 128, 4, stride=2)
        self.layer3_reid = self._make_layer(block, 256, 6, stride=2)
        self.layer4_reid = self._make_layer(block, 512, 3, stride=2)

        self.classifier = nn.Linear(self.feature_dim3d, num_classes, bias=False)
        self.bnneck = nn.BatchNorm1d(self.feature_dim3d)
        self.classifier3d = nn.Linear(self.feature_dim3d, num_classes, bias=False)
        self.bnneck3d = nn.BatchNorm1d(self.feature_dim3d)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self._init_params()

        # Load SMPL model
        self.smpl = SMPL(SMPL_MODEL_DIR,
                    batch_size=self.batch_size,
                    create_transl=False)

        # Create HMR model and ReID backbone
        # self.estimator3D = hmr_new(smpl_mean_params='/home/jiaxing/SPIN-master/data/smpl_mean_params.npz', pretrained=False)
        self.estimator3D = allInOne(smpl_mean_params='/home/jiaxing/SPIN-master/data/smpl_mean_params.npz', pretrained=False)
        self.discriminator = Discriminator()

    def forward(self, inputs, real_params=None, return_featuremaps=False, return_params_3D=False):
        xf, xf_shape, pred_rotmats, pred_betas, pred_cam, pred_displace = self.estimator3D(inputs)
        pred_params = {'rotmat':pred_rotmats,
                       'beta':pred_betas,
                       'cam':pred_cam}

        pred_displace = pred_displace.view(pred_displace.shape[0], 6890, 3)

        pred_outputs1 = self.smpl(betas=pred_betas, body_pose=pred_rotmats[:,1:],
                                 global_orient=pred_rotmats[:,0].unsqueeze(1), pose2rot=False, v_personal=pred_displace)
        pred_outputs2 = self.smpl(betas=pred_betas, body_pose=pred_rotmats[:,1:],
                                 global_orient=pred_rotmats[:,0].unsqueeze(1), pose2rot=False)

        # xf = self.layer2_reid(xf)
        xf = self.layer3_reid(xf)
        xf = self.layer4_reid(xf)
        xf = self.avgpool(xf)
        v = xf.view(xf.shape[0], -1)
        v_bn = self.bnneck(v)

        xf3d = self.shape_extractor(xf_shape)
        xf3d = self.avgpool(xf3d)
        v3d = xf3d.view(xf3d.shape[0], -1)
        v3d_bn = self.bnneck3d(v3d)

        pred_params, pred_outputs1, pred_outputs2, encoder_disc_value, gen_disc_value, real_disc_value

        if not self.training:
            v_concate1 = torch.cat([v, v3d], dim=1)
            v_concate2 = torch.cat([v_bn, v3d_bn], dim=1)

            return [v, v_bn, v3d, v3d_bn, v_concate1, v_concate2]

        # discriminator output
        encoder_disc_value = self.discriminator(pred_betas, pred_rotmats.view(-1, 24, 9))
        gen_disc_value = self.discriminator(pred_betas.detach(), pred_rotmats.detach().view(-1, 24, 9))

        real_poses, real_shapes = real_params[:, :72], real_params[:, 72:]
        real_rotmats = batch_rodrigues(real_poses.contiguous().view(-1, 3)).view(-1, 24, 9)
        real_disc_value = self.discriminator(real_shapes, real_rotmats)

        y = self.classifier(v_bn)
        y3d = self.classifier3d(v3d_bn)

        return [y, y3d], [v, v3d], pred_params, pred_outputs1, pred_outputs2, encoder_disc_value, gen_disc_value, real_disc_value
        # return [v3d], pred_params, pred_outputs1, pred_outputs2, encoder_disc_value, gen_disc_value, real_disc_value

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

        layer_name1 = ''
        for idx, part in enumerate(nameList):
            if idx == 0:
                layer_name1 += 'layer2_reid'
            else:
                layer_name1 += ('.' + part)

        if layer_name1 in model_dict and model_dict[layer_name1].size() == v.size():
            print('shape extractor:', layer_name1)
            pretrain_dict_match[layer_name1] = v

        layer_name2 = ''
        for idx, part in enumerate(nameList):
            if idx == 0:
                layer_name2 += 'layer3_reid'
            else:
                layer_name2 += ('.' + part)

        if layer_name2 in model_dict and model_dict[layer_name2].size() == v.size():
            print('shape extractor:', layer_name2)
            pretrain_dict_match[layer_name2] = v

        layer_name3 = ''
        for idx, part in enumerate(nameList):
            if idx == 0:
                layer_name3 += 'layer4_reid'
            else:
                layer_name3 += ('.' + part)

        if layer_name3 in model_dict and model_dict[layer_name3].size() == v.size():
            print('shape extractor:', layer_name3)
            pretrain_dict_match[layer_name3] = v

    model_dict.update(pretrain_dict_match)
    model.load_state_dict(model_dict)


def reid_3d_branch(**kwargs):
    model = My3DBranch(**kwargs)

    init_shape_extractor(model, model_urls['resnet50'])

    return model
