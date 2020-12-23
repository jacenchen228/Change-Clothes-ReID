import sys
import os.path as osp
from apex import amp

import torch
from torch import nn

from lib.utils import save_checkpoint, visactmap
from .trainer import Trainer
from .evaluator import Evaluator
from .evaluatorGeneral import EvaluatorGeneral


ID2FEAT_NAME = {
    0: 'ori'
}

class Engine(object):
    def __init__(self, trainloader, queryloader, galleryloader, model, optimizer, lr_scheduler,
                 flag_general=False, **kwargs):
        self.trainloader = trainloader
        self.queryloader = queryloader
        self.galleryloader = galleryloader
        self.model = model
        self.optimizer = optimizer
        self.visualize_freq = 10
        self.model = model

        model = nn.DataParallel(model)
        model = model.cuda()

        self.trainer = Trainer(trainloader, model, optimizer, lr_scheduler, **kwargs)
        if flag_general:
            self.evaluator = EvaluatorGeneral(queryloader, galleryloader, model, **kwargs)
        else:
            self.evaluator = Evaluator(queryloader, galleryloader, model, **kwargs)

    def run(self, max_epoch, test_only, eval_freq, if_visactmap, save_dir, **kwargs):

        if test_only:
            self.evaluator.evaluate()

            return

        if if_visactmap:
            visactmap(self.queryloader, self.model, save_dir=save_dir, **kwargs)

            return

        # variables to record best performance
        max_rank1_global = 0
        max_epoch_global = 0
        max_feat_global = 0
        max_rank1s, max_epochs = [0]*len(ID2FEAT_NAME), [0]*len(ID2FEAT_NAME)
        for epoch in range(0, max_epoch):
            self.trainer.train(epoch, **kwargs)

            if eval_freq > 0 and (epoch + 1) % eval_freq == 0 and (
                    epoch + 1) != max_epoch:

                rank1s = self.evaluator.evaluate()

                # update global best performance
                save_flag = True
                for i, rank1 in enumerate(rank1s):
                    if rank1 > max_rank1s[i]:
                        max_rank1s[i] = rank1
                        max_epochs[i] = epoch + 1
                    if rank1 > max_rank1_global:
                        max_rank1_global = rank1
                        max_epoch_global = epoch + 1
                        max_feat_global = i
                        if save_flag:
                            self._save_checkpoint(epoch, rank1, save_dir)
                            save_flag = False
                for i, (max_rank1_i, max_epoch_i) in enumerate(zip(max_rank1s, max_epochs)):
                    print('Maximum Rank-1 with Feature-{} is {} in the Epoch {}'.format(ID2FEAT_NAME[i], max_rank1_i, max_epoch_i))
                print('Maximum Rank-1 globally is {} in the Epoch {} with Feature-{}'.format(
                    max_rank1_global, max_epoch_global, ID2FEAT_NAME[max_feat_global]))
                print(save_dir)

        if max_epoch > 0:
            print('=> Final test')
            self.evaluator.evaluate()

            save_flag = True
            for i, rank1 in enumerate(rank1s):
                if rank1 > max_rank1s[i]:
                    max_rank1s[i] = rank1
                    max_epochs[i] = epoch + 1
                if rank1 > max_rank1_global:
                    max_rank1_global = rank1
                    max_epoch_global = epoch + 1
                    max_feat_global = i
                    if save_flag:
                        self._save_checkpoint(epoch, rank1, save_dir)
                        save_flag = False
            for i, (max_rank1_i, max_epoch_i) in enumerate(zip(max_rank1s, max_epochs)):
                print('Maximum Rank-1 with Feature-{} is {} in the Epoch {}'.format(ID2FEAT_NAME[i], max_rank1_i, max_epoch_i))
            print('Maximum Rank-1 globally is {} in the Epoch {} with Feature-{}'.format(
                max_rank1_global, max_epoch_global, ID2FEAT_NAME[max_feat_global]))
            print(save_dir)

    def _save_checkpoint(self, epoch, rank1, save_dir, is_best=False):
        save_checkpoint({
            'state_dict': self.model.state_dict(),
            'epoch': epoch + 1,
            'rank1': rank1,
            'optimizer': self.optimizer.state_dict(),
        }, save_dir, is_best=is_best)


