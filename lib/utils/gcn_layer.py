import math

import torch
from torch import nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.relu = nn.ReLU(inplace=True)
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # stdv = 1. / math.sqrt(self.weight.size(1))
        # self.weight.data.uniform_(-stdv, stdv)
        # if self.bias is not None:
        #     self.bias.data.uniform_(-stdv, stdv)
        # nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
        nn.init.normal_(self.weight, 0, 0.05)
        # nn.init.uniform_(self.weight, 0, 0.01)
        # nn.init.constant_(self.bias, 0)

        # print(torch.max(self.weight), torch.min(self.weight), torch.mean(self.weight))

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        # output = self.delta * torch.matmul(adj, support) + (1-self.delta) * support
        output = torch.matmul(adj, support)

        if self.bias is not None:
            # return self.relu(output + self.bias)
            return output + self.bias
        else:
            # return self.relu(output)
            return output

    # def forward(self, input):
    #     output = torch.matmul(input, self.weight)
    #
    #     # if self.cnt < 3:
    #     #     print(self.weight)
    #     #     self.cnt += 1
    #
    #     if self.bias is not None:
    #         return output + self.bias
    #     else:
    #         return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class SimlarityLayer(Module):
    """
    Render similarity matrix of GCN as a learnable layer
    """

    def __init__(self, node_num):
        super(SimlarityLayer, self).__init__()
        self.node_num = node_num
        self.cnt = 0
        self.delta = 0.75
        self.weight = Parameter(torch.FloatTensor(node_num, node_num))
        self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # stdv = 1. / math.sqrt(self.weight.size(1))
        # self.weight.data.uniform_(-stdv, stdv)
        # if self.bias is not None:
        #     self.bias.data.uniform_(-stdv, stdv)
        # nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.weight, 1)

    def forward(self, input, valid_mask):
        if self.cnt < 3:
            print(self.weight)
            self.cnt += 1
        adj_mat = F.softmax(self.weight.expand(valid_mask.size(0), self.weight.size(0), self.weight.size(1)), dim=2)
        adj_mat = adj_mat * valid_mask
        adj_mat = \
            adj_mat + torch.eye(self.weight.size(0)).expand(valid_mask.size(0), self.weight.size(0), self.weight.size(1)).cuda()
        adj_mat_nor = self.diag_normalization(adj_mat)
        output = torch.matmul(adj_mat_nor, input)
        # output = self.delta*output + (1-self.delta)*input     # 对经过GCN之后的feature和未经GCN的feature进行加权
        # output = torch.matmul(self.weight, input)
        return output

    def diag_normalization(self, adj_mat):
        # 根据论文里面的公式先提前计算出A^(~)
        batch_size, node_num = adj_mat.size(0), adj_mat.size(1)
        D_mat = torch.zeros(batch_size, node_num, node_num)
        # adjMat = adjMat * valid_mask

        row_sum = adj_mat.sum(dim=2)
        D_mat[:, range(node_num), range(node_num)] = row_sum
        # for batch_id in range(batch_size):
        #     D[batch_id] = torch.diag(row_sum[batch_id])
        D_mat = torch.pow(D_mat, -0.5)

        inf_mask = torch.isinf(D_mat)
        D_mat[inf_mask] = 0
        D_mat = D_mat.float().cuda()

        adj_mat_nor = torch.matmul(torch.matmul(D_mat, adj_mat), D_mat)

        return adj_mat_nor
