import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import math

from .utils import batch_rodrigues

__all__ = ['hmr_new']


def rot6d_to_rotmat(x):
    """Convert 6D rotation representation to 3x3 rotation matrix.
    Based on Zhou et al., "On the Continuity of Rotation Representations in Neural Networks", CVPR 2019
    Input:
        (B,6) Batch of 6-D rotation representations
    Output:
        (B,3,3) Batch of corresponding rotation matrices
    """
    x = x.view(-1, 3, 2)
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2)
    return torch.stack((b1, b2, b3), dim=-1)


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


class HMR(nn.Module):
    """ SMPL Iterative Regressor with ResNet50 backbone
    """

    def __init__(self, block, layers, smpl_mean_params):
        self.inplanes = 64
        super(HMR, self).__init__()
        npose = 24 * 6
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # separately extract shape- and pose-related code
        self.fc_shape_base = nn.Linear(512*block.expansion, 512*block.expansion)
        self.bn_shape_base = nn.BatchNorm1d(512*block.expansion)
        self.fc_shape = nn.Linear(512*block.expansion + 10, 1024)
        self.bn_shape = nn.BatchNorm1d(1024)
        self.decshape = nn.Linear(1024, 10)
        self.dedisplace = nn.Linear(1024, 6890 * 3, bias=False)

        self.fc_pose_base = nn.Linear(512*block.expansion, 512*block.expansion)
        self.bn_pose_base = nn.BatchNorm1d(512*block.expansion)
        self.fc_pose = nn.Linear(512 * block.expansion + npose + 3, 1024)
        self.bn_pose = nn.BatchNorm1d(1024)
        self.decpose = nn.Linear(1024, npose)
        self.deccam = nn.Linear(1024, 3)

        nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
        nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
        nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)
        nn.init.xavier_uniform_(self.dedisplace.weight, gain=0.0001)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        mean_params = np.load(smpl_mean_params)
        init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)
        # init_pose = torch.zeros(npose).float().unsqueeze(0)
        init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)
        init_cam = torch.from_numpy(mean_params['cam']).unsqueeze(0)
        init_displace = torch.zeros(6890 * 3).float().unsqueeze(0)
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)
        self.register_buffer('init_cam', init_cam)
        self.register_buffer('init_displace', init_displace)

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

    def forward(self, x, init_pose=None, init_shape=None, init_cam=None, init_displace=None, n_iter=3):

        batch_size = x.shape[0]

        if init_pose is None:
            init_pose = self.init_pose.expand(batch_size, -1)
        if init_shape is None:
            init_shape = self.init_shape.expand(batch_size, -1)
        if init_cam is None:
            init_cam = self.init_cam.expand(batch_size, -1)
        if init_displace is None:
            init_displace = self.init_displace.expand(batch_size, -1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        xf = self.avgpool(x4)
        xf = xf.view(xf.size(0), -1)

        xf_shape_base = self.fc_shape_base(xf)
        xf_shape_base = self.bn_shape_base(xf_shape_base)

        xf_pose_base = self.fc_pose_base(xf)
        xf_pose_base = self.bn_pose_base(xf_pose_base)

        pred_pose = init_pose
        pred_shape = init_shape
        pred_cam = init_cam
        pred_displace = init_displace
        for i in range(n_iter):
            xf_shape = self.fc_shape(torch.cat([xf_shape_base, pred_shape], 1))
            xf_shape = self.bn_shape(xf_shape)

            xf_pose = self.fc_pose(torch.cat([xf_pose_base, pred_pose, pred_cam], 1))
            xf_pose = self.bn_pose(xf_pose)

            pred_shape = self.decshape(xf_shape) + pred_shape
            pred_displace = self.dedisplace(xf_shape) + pred_displace
            pred_pose = self.decpose(xf_pose) + pred_pose
            pred_cam = self.deccam(xf_pose) + pred_cam

        pred_rotmat = rot6d_to_rotmat(pred_pose).view(batch_size, 24, 3, 3)

        return xf_shape_base, pred_rotmat, pred_shape, pred_cam, pred_displace


def hmr_new(smpl_mean_params, pretrained=True, **kwargs):
    """ Constructs an HMR model with ResNet50 backbone.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = HMR(Bottleneck, [3, 4, 6, 3], smpl_mean_params, **kwargs)
    # if pretrained:
    #     resnet_imagenet = resnet.resnet50(pretrained=True)
    #     model.load_state_dict(resnet_imagenet.state_dict(), strict=False)

    if pretrained:
        checkpoint = torch.load('/home/jiaxing/SPIN-master/data/model_checkpoint.pt')
        model.load_state_dict(checkpoint['model'], strict=False)

    return model

