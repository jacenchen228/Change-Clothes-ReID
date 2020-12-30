import torch
import torch.nn as nn


def build_linear_block(in_dim, out_dim):
    return nn.Sequential(
      nn.Linear(in_dim, out_dim),
      nn.BatchNorm1d(out_dim),
      nn.ReLU(inplace=True)
    )


class Discriminator(nn.Module):
    def __init__(self, in_dim1, in_dim2, layers_dim):
        super(Discriminator, self).__init__()

        self.linear1 = build_linear_block(in_dim1, layers_dim[0]//2)
        self.linear2 = build_linear_block(in_dim2, layers_dim[0]//2)

        self.linears = nn.Sequential()
        for idx in range(len(layers_dim)):
            if idx == 0:
                self.linears.add_module('linear'+str(idx), build_linear_block(layers_dim[0], layers_dim[0]))
            else:
                self.linears.add_module('linear'+str(idx), build_linear_block(layers_dim[idx-1], layers_dim[idx]))

        self.linear_final = nn.Linear(layers_dim[-1], 1)

        self._init_params()

    def forward(self, x1, x2):
        y1 = self.linear1(x1)
        y2 = self.linear2(x2)

        y = torch.cat([y1, y2], dim=1)
        y = self.linears(y)
        y = self.linear_final(y)

        return y

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
