
'''
    file:   LinearModel.py

    date:   2018_04_29
    author: zhangxiong(1025679612@qq.com)
'''

import torch.nn as nn
import sys


class LinearModel(nn.Module):
    '''
        input param:
            fc_layers: a list of neuron count, such as [2133, 1024, 1024, 85]
            use_dropout: a list of bool define use dropout or not for each layer, such as [True, True, False]
            drop_prob: a list of float defined the drop prob, such as [0.5, 0.5, 0]
            use_ac_func: a list of bool define use active function or not, such as [True, True, False]
    '''
    def __init__(self, fc_layers, use_dropout, drop_prob, use_ac_func):
        super(LinearModel, self).__init__()
        self.fc_layers     = fc_layers
        self.use_dropout   = use_dropout
        self.drop_prob     = drop_prob
        self.use_ac_func   = use_ac_func

        if not self._check():
            msg = 'wrong LinearModel parameters!'
            print(msg)
            sys.exit(msg)

        self.create_layers()

    def _check(self):
        while True:
            if not isinstance(self.fc_layers, list):
                print('fc_layers require list, get {}'.format(type(self.fc_layers)))
                break

            if not isinstance(self.use_dropout, list):
                print('use_dropout require list, get {}'.format(type(self.use_dropout)))
                break

            if not isinstance(self.drop_prob, list):
                print('drop_prob require list, get {}'.format(type(self.drop_prob)))
                break

            if not isinstance(self.use_ac_func, list):
                print('use_ac_func require list, get {}'.format(type(self.use_ac_func)))
                break

            l_fc_layer = len(self.fc_layers)
            l_use_drop = len(self.use_dropout)
            l_drop_porb = len(self.drop_prob)
            l_use_ac_func = len(self.use_ac_func)

            return l_fc_layer >= 2 and l_use_drop < l_fc_layer and l_drop_porb < l_fc_layer and l_use_ac_func < l_fc_layer and l_drop_porb == l_use_drop

        return False

    def create_layers(self):
        l_fc_layer = len(self.fc_layers)
        l_use_drop = len(self.use_dropout)
        l_drop_porb = len(self.drop_prob)
        l_use_ac_func = len(self.use_ac_func)

        self.fc_blocks = nn.Sequential()

        for _ in range(l_fc_layer - 1):
            self.fc_blocks.add_module(
                name = 'regressor_fc_{}'.format(_),
                module = nn.Linear(in_features = self.fc_layers[_], out_features = self.fc_layers[_ + 1])
            )

            if _ < l_use_ac_func and self.use_ac_func[_]:
                self.fc_blocks.add_module(
                    name = 'regressor_af_{}'.format(_),
                    module = nn.ReLU()
                )

            if _ < l_use_drop and self.use_dropout[_]:
                self.fc_blocks.add_module(
                    name = 'regressor_fc_dropout_{}'.format(_),
                    module = nn.Dropout(p = self.drop_prob[_])
                )

    def forward(self, inputs):
        msg = 'the base class [LinearModel] is not callable!'
        sys.exit(msg)
