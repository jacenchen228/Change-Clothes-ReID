import torch
import torch.nn as nn


def build_linear_block(in_dim, out_dim):
    return nn.Sequential([
      nn.Linear(in_dim, out_dim),
      nn.BatchNorm1d(out_dim),
      nn.ReLU(inplace=True)
    ])


class Discriminator(nn.Module):
    def __init__(self, in_dim1, in_dim2, layers_dim):
        super(Discriminator, self).__init__()

        self.linear1 = build_linear_block(in_dim1, layers_dim[0]//2)
        self.linear2 = build_linear_block(in_dim2, layers_dim[0]//2)

        linear_layers = []
        for idx in range(len(layers_dim)):
            if idx == 0:
                linear_layers.append(build_linear_block(layers_dim[0], layers_dim[0]))
            else:
                linear_layers.append(build_linear_block(layers_dim[idx-1], layers_dim[idx]))
        self.linears = nn.Sequential(linear_layers)

        self.linear_final = nn.Linear(layers_dim[-1], 1)

    def forward(self, x1, x2):
        y1 = self.linear1(x1)
        y2 = self.linear2(x2)

        y = torch.cat([y1, y2], dim=1)
        y = self.linears(y)
        y = self.linear_final(y)

        return y


