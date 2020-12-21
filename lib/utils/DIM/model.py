import torch
import torch.nn as nn

class GlobalEncoder(nn.Module):
    def __init__(self, height, width, in_feature_dim, feature_dim):
        super(GlobalEncoder, self).__init__()
        self.height = height
        self.width = width
        self.in_feature_dim = in_feature_dim
        self.feature_dim = feature_dim

        self.conv = nn.Conv2d(self.in_feature_dim, self.feature_dim, kernel_size=3, stride=1, padding=1)
        self.conv_ = nn.Conv2d(self.in_feature_dim, self.feature_dim, kernel_size=3, stride=1, padding=1)
        self.linear = nn.Linear(self.feature_dim * height * width, 512)
        self.linear_ = nn.Linear(self.feature_dim * height * width, 512)

    def forward(self, y, y_aux):
        h_y1 = self.conv(y)
        h_y1 = self.linear(h_y1.view(h_y1.shape[0], -1))

        h_y2 = self.conv_(y_aux)
        h_y2 = self.linear_(h_y2.view(h_y2.shape[0], -1))

        return h_y1, h_y2

class GlobalDiscriminator(nn.Module):
    def __init__(self):
        super(GlobalDiscriminator, self).__init__()
        # self.height = height
        # self.width = width
        # self.in_feature_dim = in_feature_dim
        # self.feature_dim = feature_dim

        self.relu = nn.ReLU(inplace=True)
        self.linear1 = nn.Linear(1024, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, 1)

    def forward(self, h_y1, h_y2):
        h = torch.cat((h_y1, h_y2), dim=1)
        h = self.relu(self.linear1(h))
        h = self.relu(self.linear2(h))

        return self.linear3(h)

class LocalDiscriminator(nn.Module):
    def __init__(self, height, width, feature_dim, num_classes, pretrained=True, flag=True):
        super().__init__()
        self.height = height
        self.width = width
        self.feature_dim = feature_dim

        # # 原来的embedding layers
        # if flag:
        #     self.c0 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        # else:
        #     self.c0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)
        # self.c1 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        # self.c2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        # self.c3 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        # # self.c4 = nn.Conv2d(512, 2048, kernel_size=3, stride=1, padding=1)
        # self.bn0 = nn.BatchNorm2d(64)
        # self.bn1 = nn.BatchNorm2d(128)
        # self.bn2 = nn.BatchNorm2d(256)
        # self.bn3 = nn.BatchNorm2d(512)
        # # self.bn4 = nn.BatchNorm2d(2048)

        self.c0_ = nn.Conv2d(self.feature_dim, 512, kernel_size=3, stride=1, padding=1)

        self.relu = nn.ReLU(inplace=True)

        self.conv0 = nn.Conv2d(1024, 512, kernel_size=1)
        self.conv1 = nn.Conv2d(512, 512, kernel_size=1)
        self.conv2 = nn.Conv2d(512, 1, kernel_size=1)

    def _transform(self, M):
        h_M = self.c0(M)
        h_M = self.bn0(h_M)
        h_M = self.relu(h_M)

        h_M = self.c1(h_M)
        h_M = self.bn1(h_M)
        h_M = self.relu(h_M)

        h_M = self.c2(h_M)
        h_M = self.bn2(h_M)
        h_M = self.relu(h_M)

        h_M = self.c3(h_M)
        h_M = self.bn3(h_M)
        h_M = self.relu(h_M)

        # h_M = self.c4(h_M)
        # h_M = self.bn4(h_M)
        # h_M = self.relu(h_M)

        return h_M

    def forward(self, y, M):
        h_y = self.c0_(y)

        h_M = self._transform(M)

        h = torch.cat((h_y, h_M), dim=1)

        h = self.relu(self.conv0(h))
        h = self.relu(self.conv1(h))
        return self.conv2(h)

# class PartEncoder(nn.Module):
#     def __init__(self, height, width, in_feature_dim):
#         super(PartEncoder, self).__init__()
#         self.height = height
#         self.width = width
#         self.in_feature_dim = in_feature_dim
#
#         self.linear = nn.Linear(self.in_feature_dim * height * width, 512)
#         self.linear_ = nn.Linear(self.in_feature_dim * height * width, 512)
#
#     def forward(self, y, y_aux):
#         h_y1 = self.linear(y)
#         h_y2 = self.linear_(y_aux)
#
#         return h_y1, h_y2

class PartEncoder(nn.Module):
    def __init__(self, height, width, in_feature_dim):
        super(PartEncoder, self).__init__()
        self.height = height
        self.width = width
        self.in_feature_dim = in_feature_dim

        self.linear = nn.Linear(self.in_feature_dim * height * width, 512)
        # self.linear_ = nn.Linear(self.in_feature_dim * height * width, 512)

    def forward(self, y):
        h_y1 = self.linear(y)
        # h_y2 = self.linear_(y_aux)

        return h_y1

class PartDiscriminator(nn.Module):
    def __init__(self):
        super(PartDiscriminator, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.linear1 = nn.Linear(1024, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, 1)

    def forward(self, h_y1, h_y2):
        h = torch.cat((h_y1, h_y2), dim=1)
        h = self.relu(self.linear1(h))
        h = self.relu(self.linear2(h))

        return self.linear3(h)

