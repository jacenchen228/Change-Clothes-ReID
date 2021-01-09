import sys
import os.path as osp
from apex import amp

import torch
from torch import nn

from lib.utils import save_checkpoint, visactmap
from .trainer import Trainer
from .evaluator import Evaluator, ID2FEAT_NAME
from .evaluatorGeneral import EvaluatorGeneral


class Engine(object):
    def __init__(self, trainloader, queryloader, galleryloader, model, optimizer, lr_scheduler,
                 flag_general=False, concern_indicator='rank', start_save_epoch=1, **kwargs):
        self.trainloader = trainloader
        self.queryloader = queryloader
        self.galleryloader = galleryloader
        self.model = model
        self.optimizer = optimizer
        self.visualize_freq = 10
        self.model = model
        self.concern_indicator = concern_indicator
        self.start_save_epoch = start_save_epoch

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
        max_indicator_global = 0
        max_epoch_global = 0
        max_feat_global = 0
        max_indicators, max_epochs = [0]*len(ID2FEAT_NAME), [0]*len(ID2FEAT_NAME)
        for epoch in range(0, max_epoch):
            self.trainer.train(epoch, **kwargs)

            if eval_freq > 0 and (epoch + 1) % eval_freq == 0 and (
                    epoch + 1) != max_epoch:

                rank1s, mAPs = self.evaluator.evaluate()

                if self.concern_indicator == 'rank':
                    indicators = rank1s
                else:
                    indicators = mAPs

                # update global best performance
                save_flag = True
                for i, indicator in enumerate(indicators):
                    if indicator > max_indicators[i]:
                        max_indicators[i] = indicator
                        max_epochs[i] = epoch + 1
                    if indicator > max_indicator_global:
                        max_indicator_global = indicator
                        max_epoch_global = epoch + 1
                        max_feat_global = i
                        if save_flag and epoch+1 >= self.start_save_epoch:
                            self._save_checkpoint(epoch, save_dir)
                            save_flag = False
                for i, (max_indicator_i, max_epoch_i) in enumerate(zip(max_indicators, max_epochs)):
                    print('Maximum {} with Feature-{} is {} in the Epoch {}'.format(self.concern_indicator, ID2FEAT_NAME[i], max_indicator_i, max_epoch_i))
                print('Maximum {} globally is {} in the Epoch {} with Feature-{}'.format(
                    self.concern_indicator, max_indicator_global, max_epoch_global, ID2FEAT_NAME[max_feat_global]))
                print(save_dir)

        if max_epoch > 0:
            print('=> Final test')
            rank1s, mAPs = self.evaluator.evaluate()

            if self.concern_indicator == 'rank':
                indicators = rank1s
            else:
                indicators = mAPs

            save_flag = True
            for i, indicator in enumerate(indicators):
                if indicator > max_indicators[i]:
                    max_indicators[i] = indicator
                    max_epochs[i] = max_epoch + 1
                if indicator > max_indicator_global:
                    max_indicator_global = indicator
                    max_epoch_global = max_epoch + 1
                    max_feat_global = i
                    if save_flag:
                        self._save_checkpoint(max_epoch, save_dir)
                        save_flag = False
            for i, (max_indicator_i, max_epoch_i) in enumerate(zip(max_indicators, max_epochs)):
                print('Maximum {} with Feature-{} is {} in the Epoch {}'.format(self.concern_indicator, ID2FEAT_NAME[i], max_indicator_i, max_epoch_i))
            print('Maximum {} globally is {} in the Epoch {} with Feature-{}'.format(
                self.concern_indicator, max_indicator_global, max_epoch_global, ID2FEAT_NAME[max_feat_global]))
            print(save_dir)

    def _save_checkpoint(self, epoch, save_dir, is_best=False):
        save_checkpoint({
            'state_dict': self.model.state_dict(),
            'epoch': epoch + 1,
            'optimizer': self.optimizer.state_dict(),
        }, save_dir, is_best=is_best)


