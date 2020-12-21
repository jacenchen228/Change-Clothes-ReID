import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import GraphAttentionLayer, SpGraphAttentionLayer


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nout, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nout, dropout=dropout, alpha=alpha, concat=False)

    # def forward(self, x, adj):
    #     x = F.dropout(x, self.dropout, training=self.training)
    #     x = torch.cat([att(x, adj) for att in self.attentions], dim=2)
    #     x = F.dropout(x, self.dropout, training=self.training)
    #     x = F.elu(self.out_att(x, adj))
    #
    #     # return F.log_softmax(x, dim=2)
    #     return x

    def forward(self, x, adj):
        # 此处输出的是softmax之后的相似性矩阵供给GCN，而不是像上面一样的feature
        x = F.dropout(x, self.dropout, training=self.training)
        # 在这种情况下head_num必须设置为1
        adj_mat = torch.cat([att(x, adj) for att in self.attentions], dim=2)

        return adj_mat

class SpGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Sparse version of GAT."""
        super(SpGAT, self).__init__()
        self.dropout = dropout

        self.attentions = [SpGraphAttentionLayer(nfeat, 
                                                 nhid, 
                                                 dropout=dropout, 
                                                 alpha=alpha, 
                                                 concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = SpGraphAttentionLayer(nhid * nheads, 
                                             nclass, 
                                             dropout=dropout, 
                                             alpha=alpha, 
                                             concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)

