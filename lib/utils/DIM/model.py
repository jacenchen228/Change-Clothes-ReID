import torch
import torch.nn as nn


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


class PartDiscriminator(nn.Module):
    def __init__(self, feature_dim):
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

