import sys
import os.path as osp
from apex import amp

import torch
from torch import nn

from lib.utils import save_checkpoint, visactmap, save_3d_param, \
    visualize_smpl, smpl_statistics, visualize_3d_silhouette, visualize_3d_rgb
from lib.sync_batchnorm import convert_model
from .trainer import Trainer
from .trainer3D import Trainer3D
from .trainerOnly3D import TrainerOnly3D
from .evaluator import Evaluator
from .evaluator3D import Evaluator3D, ID2FEAT_NAME
from .evaluatorGeneral import EvaluatorGeneral


class Engine(object):
    def __init__(self, trainloader, queryloader, galleryloader, model, optimizer, lr_scheduler, flag_3d=False,
                 flag_general=False, trainloader_entire=None, **kwargs):
        self.trainloader = trainloader
        self.queryloader = queryloader
        self.galleryloader = galleryloader
        self.model = model
        self.optimizer = optimizer
        self.visualize_freq = 10

        # specify for the ensembled model with global branch feature and 3D shape feature
        self.trainloader_entire = trainloader_entire
        self.model = model

        # specify optimizer and scheduler
        # optimizer3d1 = torch.optim.Adam(
        #     [{'params': self.model.estimator3D.parameters()}],
        #     lr=0.00001,
        #     weight_decay=0.0001
        # )
        # optimizer3d2 = torch.optim.Adam(
        #     [{'params': self.model.discriminator.parameters(), },
        #      {'params': self.model.shape_extractor.parameters()},
        #      {'params': self.model.embedding_layer3d.parameters()},
        #      {'params': self.model.bn3d.parameters()},
        #      {'params': self.model.classifier3d.parameters()}],
        #     lr=0.0001,weight_decay=0.0001)

        '''
        正常请况下3d分支的优化器
        '''
        optimizer3d_enc = torch.optim.Adam(
            [{'params': self.model.estimator3d.parameters(), 'lr': 0.00001}],
             # {'params': self.model.shape_extractor.parameters(), 'lr': shape_lr}],
             # {'params': self.model.embedding_layer3d.parameters(), 'lr': shape_lr},
             # {'params': self.model.bn3d.parameters(), 'lr': shape_lr},
             # {'params': self.model.classifier3d.parameters(), 'lr': shape_lr}],
            lr=0.0001, weight_decay=0.0001)

        '''
        判别器网络的优化器，当跑无3d重建的版本时需注释
        '''
        # optimizer3d_disc = torch.optim.Adam(
        #     [{'params': self.model.discriminator.parameters(), 'lr': 0.0001}],
        #     lr=0.0001, weight_decay=0.0001
        # )

        '''
        ReID using sgd
        '''
        optimizer_reid = torch.optim.SGD(
             [{'params': self.model.reid_encoder.parameters()},
              {'params': self.model.conv5.parameters()},
             {'params': self.model.shape_extractor.parameters()},
             {'params': self.model.embedding_layer3d.parameters()},
             # {'params': self.model.embedding_layer.parameters()},
             # {'params': self.model.embedding_layer_concate.parameters()},
             # {'params': self.model.bn.parameters()},
             # {'params': self.model.classifier.parameters()},
             # {'params': self.model.bn_reid.parameters()},
             {'params': self.model.classifier_reid.parameters()},
             {'params': self.model.bn3d.parameters()},
             {'params': self.model.classifier3d.parameters()}],
            lr=0.05, weight_decay=0.0005, momentum=0.9)

        '''
        ReID using amsgrad
        '''
        # optimizer_reid = torch.optim.Adam(
        #      [{'params': self.model.reid_encoder.parameters()},
        #       {'params': self.model.conv5.parameters()},
        #      {'params': self.model.shape_extractor.parameters()},
        #      {'params': self.model.embedding_layer3d.parameters()},
        #      # {'params': self.model.embedding_layer.parameters()},
        #      # {'params': self.model.embedding_layer_concate.parameters()},
        #      # {'params': self.model.bn.parameters()},
        #      # {'params': self.model.classifier.parameters()},
        #      {'params': self.model.bn_reid.parameters()},
        #      {'params': self.model.classifier_reid.parameters()},
        #      {'params': self.model.bn3d.parameters()},
        #      {'params': self.model.classifier3d.parameters()}],
        #     lr=0.0008, weight_decay=0.0005, betas=(0.9, 0.99), amsgrad=True)

        # optimizer_reid = torch.optim.SGD(self.model.parameters(),
        #                                  lr=0.05, weight_decay=0.0005, momentum=0.9)

        # model, optimizers = amp.initialize(model, [optimizer3d1, optimizer3d2, optimizer_reid],
        #                                   opt_level="O 1",
        #                                   keep_batchnorm_fp32=None,
        #                                   loss_scale=None)
        optimizers = [optimizer3d_enc, optimizer_reid]
        model = nn.DataParallel(model)
        model = model.cuda()

        if flag_3d:
            self.trainer = Trainer3D(trainloader, model, optimizers, lr_scheduler,
                                     trainloader_entire=trainloader_entire, **kwargs)
            # self.trainer = TrainerOnly3D(trainloader, model, optimizer, lr_scheduler, **kwargs)
            self.evaluator = Evaluator3D(queryloader, galleryloader, model, **kwargs)
            self.init_vertices_pool = {}
        else:
            self.trainer = Trainer(trainloader, model, optimizers, lr_scheduler, **kwargs)
            # self.evaluator = Evaluator(queryloader, galleryloader, model, **kwargs)
            self.evaluator = Evaluator3D(queryloader, galleryloader, model, **kwargs)
        if flag_general:
            self.evaluator = EvaluatorGeneral(queryloader, galleryloader, model, **kwargs)
        self.save_3d_flag = True

    def run(self, max_epoch, test_only, eval_freq, if_visactmap, save_dir, if_save_param, **kwargs):
        # save_dir = osp.join(save_prefix, save_dir)

        if test_only:
            self.evaluator.evaluate()
            # visualize_smpl(self.model, self.queryloader, osp.join(save_dir, 'init_'))

            # visualize_3d_silhouette(self.model, self.queryloader, osp.join(save_dir, 'init_query1'), **kwargs)

            # visualize_3d_rgb(self.model, self.trainloader, osp.join(save_dir, 'init_query'), **kwargs)

            return

        if if_visactmap:
            # user specific for which dataloader to visualize
            visactmap(self.queryloader, self.model, save_dir=save_dir, **kwargs)
            return

        if if_save_param:
            # user specific for which dataloader to visualize
            save_3d_param(self.trainloader, self.model, save_dir=save_dir, **kwargs)
            return

        # variables to record best performance
        max_rank1_global = 0
        max_epoch_global = 0
        max_feat_global = 0
        max_rank1s, max_epochs = [0]*len(ID2FEAT_NAME), [0]*len(ID2FEAT_NAME)
        for epoch in range(0, max_epoch):
            # if epoch == 0:
            #     visualize_3d_rgb(self.model, self.queryloader, osp.join(save_dir, 'init_query'), **kwargs)

            self.trainer.train(epoch, self.queryloader, **kwargs)

            # if (epoch + 1) % self.visualize_freq == 0:
            #     visualize_smpl(self.model, self.queryloader, osp.join(save_dir, 'epoch_'+str(epoch+1)+'_'))

            # save the model pretrained by 3D related loss
            # before training with ReID loss
            # if epoch >= self.trainer.pretrain_3d_epoch and self.save_3d_flag:
            #
            #     self._save_checkpoint(epoch, 0, save_dir)
            #     self.save_3d_flag = False

            if eval_freq > 0 and (epoch + 1) % eval_freq == 0 and (
                    epoch + 1) != max_epoch and epoch >= self.trainer.pretrain_3d_epoch:

                # sys.exit()

                rank1s = self.evaluator.evaluate()

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


