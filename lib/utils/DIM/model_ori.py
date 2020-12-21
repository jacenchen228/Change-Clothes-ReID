import torch
import torch.nn as nn
from torch.autograd import Variable


class GlobalDiscriminator(nn.Module):
    def __init__(self, in_feature_dim, feature_dim):
        super(GlobalDiscriminator, self).__init__()
        # self.height = height
        # self.width = width
        self.in_feature_dim = in_feature_dim
        self.feature_dim = feature_dim

        self.conv0 = nn.Conv2d(self.in_feature_dim, self.feature_dim, kernel_size=3, stride=1, padding=1)
        self.conv1 = nn.Conv2d(self.feature_dim, 128, kernel_size=3, stride=1, padding=1)
        self.linear1 = nn.Linear(16 * 8 * 128 + self.in_feature_dim, 1024)
        self.linear2 = nn.Linear(1024, 256)
        self.linear3 = nn.Linear(256, 1)

        # self.conv0_ = nn.Conv2d(self.in_feature_dim, self.feature_dim, kernel_size=3, stride=1, padding=1)
        # self.linear0 = nn.Linear(self.feature_dim * height * width, 512)
        # self.linear0_ = nn.Linear(self.feature_dim * height * width, 512)

        self.relu = nn.ReLU(inplace=True)

        # self.linear_shortcut = nn.Linear(in_feature_dim, 256)

        self._init_params()

    # def forward(self, y, M0, M1):
    #     h_y0 = self.conv0(y)
    #     h_y1 = self.conv0_(y)
    #     # h_y = self.global_avgpool(h_y)
    #     # h_y = h_y.view(h_y.shape[0], -1)
    #     h_y0 = self.l0(h_y0.view(h_y0.shape[0], -1))
    #     h_y1 = self.l0_(h_y1.view(h_y1.shape[0], -1))
    #     # h_y1 = self.l0_(h_y0.view(h_y0.shape[0], -1))
    #
    #     h_M0= self._transform(M0)
    #     h_M0 = self.l1(h_M0.view(h_M0.shape[0], -1))
    #     #h_M = self.global_avgpool(h_M)
    #     #h_M = h_M.view(h_M.shape[0], -1)
    #
    #     h_M1 = self.l1_(M1.view(M1.shape[0], -1))
    #
    #     h0 = torch.cat((h_y0, h_M0), dim=1)
    #     h0 = self.relu(self.l2(h0))
    #     h0 = self.relu(self.l3(h0))
    #
    #     h1 = torch.cat((h_y1, h_M1), dim=1)
    #     # h1 = torch.cat((h_y0, h_M1), dim=1)
    #     h1 = self.relu(self.l2_(h1))
    #     h1 = self.relu(self.l3_(h1))
    #
    #     return self.l4(h0), self.l4_(h1)

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

    def forward(self, y1, y2, flag=True):
        y2 = self.conv0(y2)
        y2 = self.relu(y2)
        y2 = self.conv1(y2)
        y2 = self.relu(y2)
        y2 = y2.view(y2.shape[0], -1)

        # h_y1 = self.conv0(y)
        # h_y1 = self.linear0(h_y1.view(h_y1.shape[0], -1))
        #
        # h_y2 = self.conv0_(y_aux)
        # h_y2 = self.linear0_(h_y2.view(h_y2.shape[0], -1))

        # h = torch.cat((h_y1, h_y2), dim=1)
        h = torch.cat((y1, y2), dim=1)

        # h_shortcut = self.linear_shortcut(h)

        h = self.relu(self.linear1(h))
        h = self.relu(self.linear2(h))

        # h = h + h_shortcut

        # h = self.linear3(h)

        # h = Variable(h, requires_grad=True)
        # h.register_hook(lambda grad: print(torch.mean(grad, dim=0)))

        return self.linear3(h)

class PartDiscriminator(nn.Module):
    def __init__(self, in_feature_dim, feature_dim, vec_feature_dim, part_height):
        super(PartDiscriminator, self).__init__()
        # self.height = height
        # self.width = width
        self.in_feature_dim = in_feature_dim
        self.feature_dim = feature_dim

        self.conv0 = nn.Conv2d(self.in_feature_dim, self.feature_dim, kernel_size=3, stride=1, padding=1)
        self.conv1 = nn.Conv2d(self.feature_dim, 128, kernel_size=3, stride=1, padding=1)
        self.linear1 = nn.Linear(part_height * 8 * 128 + vec_feature_dim, 1024)
        self.linear2 = nn.Linear(1024, 256)
        self.linear3 = nn.Linear(256, 1)

        # self.conv0_ = nn.Conv2d(self.in_feature_dim, self.feature_dim, kernel_size=3, stride=1, padding=1)
        # self.linear0 = nn.Linear(self.feature_dim * height * width, 512)
        # self.linear0_ = nn.Linear(self.feature_dim * height * width, 512)

        self.relu = nn.ReLU(inplace=True)

        # self.linear_shortcut = nn.Linear(in_feature_dim, 256)

        self._init_params()

    # def forward(self, y, M0, M1):
    #     h_y0 = self.conv0(y)
    #     h_y1 = self.conv0_(y)
    #     # h_y = self.global_avgpool(h_y)
    #     # h_y = h_y.view(h_y.shape[0], -1)
    #     h_y0 = self.l0(h_y0.view(h_y0.shape[0], -1))
    #     h_y1 = self.l0_(h_y1.view(h_y1.shape[0], -1))
    #     # h_y1 = self.l0_(h_y0.view(h_y0.shape[0], -1))
    #
    #     h_M0= self._transform(M0)
    #     h_M0 = self.l1(h_M0.view(h_M0.shape[0], -1))
    #     #h_M = self.global_avgpool(h_M)
    #     #h_M = h_M.view(h_M.shape[0], -1)
    #
    #     h_M1 = self.l1_(M1.view(M1.shape[0], -1))
    #
    #     h0 = torch.cat((h_y0, h_M0), dim=1)
    #     h0 = self.relu(self.l2(h0))
    #     h0 = self.relu(self.l3(h0))
    #
    #     h1 = torch.cat((h_y1, h_M1), dim=1)
    #     # h1 = torch.cat((h_y0, h_M1), dim=1)
    #     h1 = self.relu(self.l2_(h1))
    #     h1 = self.relu(self.l3_(h1))
    #
    #     return self.l4(h0), self.l4_(h1)

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

    def forward(self, y1, y2, flag=True):
        y2 = self.conv0(y2)
        y2 = self.relu(y2)
        y2 = self.conv1(y2)
        y2 = self.relu(y2)
        y2 = y2.view(y2.shape[0], -1)

        # h_y1 = self.conv0(y)
        # h_y1 = self.linear0(h_y1.view(h_y1.shape[0], -1))
        #
        # h_y2 = self.conv0_(y_aux)
        # h_y2 = self.linear0_(h_y2.view(h_y2.shape[0], -1))

        # h = torch.cat((h_y1, h_y2), dim=1)
        h = torch.cat((y1, y2), dim=1)

        # h_shortcut = self.linear_shortcut(h)

        h = self.relu(self.linear1(h))
        h = self.relu(self.linear2(h))

        # h = h + h_shortcut

        # h = self.linear3(h)

        # h = Variable(h, requires_grad=True)
        # h.register_hook(lambda grad: print(torch.mean(grad, dim=0)))

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