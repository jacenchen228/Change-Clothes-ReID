import torch
import torch.nn as nn
from torchreid.models import resnet18, resnet34

class PartGlobalDiscriminator(nn.Module):
    def __init__(self, feature_dim):
        super(PartGlobalDiscriminator, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        self.linear1 = nn.Linear(feature_dim, 256)
        self.linear1_ = nn.Linear(feature_dim, 256)

        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, 256)
        self.linear4 = nn.Linear(256, 1)

    def forward(self, y, y_aux):
        h_y = self.linear1(y)
        h_aux = self.linear1_(y_aux)

        h = torch.cat((h_y, h_aux), dim=1)
        h = self.relu(self.linear2(h))
        h = self.relu(self.linear3(h))

        return self.linear4(h)

# class LocalDiscriminator(nn.Module):
#     def __init__(self, height, width, feature_dim, num_classes, pretrained=True, flag=True):
#         super().__init__()
#         self.height = height
#         self.width = width
#         self.feature_dim = feature_dim
#
#         # # 原来的embedding layers
#         # if flag:
#         #     self.c0 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
#         # else:
#         #     self.c0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)
#         # self.c1 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
#         # self.c2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
#         # self.c3 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
#         # # self.c4 = nn.Conv2d(512, 2048, kernel_size=3, stride=1, padding=1)
#         # self.bn0 = nn.BatchNorm2d(64)
#         # self.bn1 = nn.BatchNorm2d(128)
#         # self.bn2 = nn.BatchNorm2d(256)
#         # self.bn3 = nn.BatchNorm2d(512)
#         # # self.bn4 = nn.BatchNorm2d(2048)
#
#         self.c0_ = nn.Conv2d(self.feature_dim, 512, kernel_size=3, stride=1, padding=1)
#
#         self.relu = nn.ReLU(inplace=True)
#
#         self.conv0 = nn.Conv2d(1024, 512, kernel_size=1)
#         self.conv1 = nn.Conv2d(512, 512, kernel_size=1)
#         self.conv2 = nn.Conv2d(512, 1, kernel_size=1)
#
#     def _transform(self, M):
#         h_M = self.c0(M)
#         h_M = self.bn0(h_M)
#         h_M = self.relu(h_M)
#
#         h_M = self.c1(h_M)
#         h_M = self.bn1(h_M)
#         h_M = self.relu(h_M)
#
#         h_M = self.c2(h_M)
#         h_M = self.bn2(h_M)
#         h_M = self.relu(h_M)
#
#         h_M = self.c3(h_M)
#         h_M = self.bn3(h_M)
#         h_M = self.relu(h_M)
#
#         # h_M = self.c4(h_M)
#         # h_M = self.bn4(h_M)
#         # h_M = self.relu(h_M)
#
#         return h_M
#
#     def forward(self, y, M):
#         h_y = self.c0_(y)
#
#         h_M = self._transform(M)
#
#         h = torch.cat((h_y, h_M), dim=1)
#
#         h = self.relu(self.conv0(h))
#         h = self.relu(self.conv1(h))
#         return self.conv2(h)

