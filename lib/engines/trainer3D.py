import time
import datetime
import sys
import os.path as osp
# from apex import amp
from apex import amp

import torch
from torch import nn
from torch.nn import functional as F

from lib.utils import AverageMeter, open_all_layers, open_specified_layers, OFPenalty, visualize_3d_rgb
from lib.losses import *
# from lib.hmr import shape_l2_loss, pose_l2_loss, kp_2d_l1_loss
from lib.lib3D import keypoint_loss, proj_2djoint, SilhouetteLoss, IMG_RES, FOCAL_LENGTH, \
    batch_encoder_disc_l2_loss, batch_adv_disc_l2_loss, VerticeLoss, RealMotionWarpper
from lib.optim import CosineAnnealingWarmUp


class Trainer3D(object):
    def __init__(self, trainloader, model, optimizer, scheduler,
                 max_epoch, num_train_pids, use_gpu=True, fixbase_epoch=0, open_layers=None,
                 print_freq=10, margin=0.3, label_smooth=True, batch_size=64, pretrain_3d_epoch=0,
                 save_dir=None, joint_loss_weight=0.0001, trainloader_entire=None, **kwargs):
        self.trainloader = trainloader
        # self.model = model
        # self.optimizer = optimizer
        # self.scheduler = scheduler

        real_motion_dir = '/home/jiaxing/CMU'
        self.realmotionloader = torch.utils.data.DataLoader(
            RealMotionWarpper(real_motion_dir),
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=False,
            drop_last=True,
        )

        # specify optimizer and scheduler
        # self.optimizer_encoder = torch.optim.Adam(
        #     [{'params': self.model.estimator3D.parameters()}],
        #     lr=0.00001,
        #     weight_decay=0.0001
        # )
        # self.optimizer_disc = torch.optim.Adam(
        #     [{'params': self.model.discriminator.parameters()}],
        #     lr=0.0001,
        #     weight_decay=0.0001
        # )

        # self.optimizer_reid = torch.optim.SGD(
        #     [{'params': self.model.module.backbone.parameters()},
        #      {'params': self.model.module.branches.parameters()}],
        #     lr=0.05, weight_decay=0.0005, momentum=0.9)

        # self.optimizer_reid = torch.optim.Adam(
        #     [{'params': self.model.backbone.parameters()},
        #      {'params': self.model.branches.parameters()}],
        #     lr=0.0001, weight_decay=0.0001)

        self.optimizer3d_enc, self.optimizer3d_disc, self.optimizer_reid = optimizer

        self.model = model
        # self.model = nn.DataParallel(self.model).cuda()

        self.scheduler3d_enc = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer3d_enc,
            milestones=[20, 40],
            gamma=0.9
        )

        self.scheduler3d_disc = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer3d_disc,
            milestones=[20, 40],
            gamma=0.9
        )

        self.scheduler_reid = CosineAnnealingWarmUp(self.optimizer_reid, T_0=5, T_end=max_epoch, warmup_factor=0, last_epoch=-1)

        self.use_gpu = use_gpu
        self.train_len = len(self.trainloader)
        self.max_epoch = max_epoch
        self.fixbase_epoch = fixbase_epoch
        self.open_layers = open_layers
        self.print_freq = print_freq
        self.save_dir = save_dir
        self.batch_size = batch_size

        # specify losses
        self.criterion_t = TripletLoss(margin=margin)
        # self.criterion_x = CrossEntropyLoss(
        #     num_classes=num_train_pids,
        #     use_gpu=self.use_gpu,
        #     label_smooth=label_smooth)
        self.criterion_x = nn.CrossEntropyLoss().cuda()
        self.criterion_keypoints = torch.nn.MSELoss(reduction='none')
        self.criterion_silhouette = SilhouetteLoss(IMG_RES, self.model.module.smpl.faces, use_gpu,
                                                    FOCAL_LENGTH, batch_size, silhouette_base=20000)
        self.criterion_vertice = VerticeLoss(val_base=1.0)
        self.criterion_encoder_adv = batch_encoder_disc_l2_loss
        self.criterion_discri_adv = batch_adv_disc_l2_loss

        # epoch num of pretraining 3d branch
        self.pretrain_3d_epoch = pretrain_3d_epoch

        # weight of joint loss projected from 3D to 2D
        self.joint_loss_weight = joint_loss_weight
        self.silhouette_loss_weight = 1.0
        self.vertice_loss_weight = 1800.0

    def train(self, epoch, queryloader=None, **kwargs):
        losses_enc = AverageMeter()
        losses_disc = AverageMeter()
        losses_trip = AverageMeter()
        losses_cent = AverageMeter()
        losses_joint = AverageMeter()
        losses_silhouette = AverageMeter()
        losses_vertice = AverageMeter()
        losses_enc_adv = AverageMeter()
        losses_disc_adv = AverageMeter()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        self.model.train()
        if (epoch + 1) <= self.fixbase_epoch and self.open_layers is not None:
            print('* Only train {} (epoch: {}/{})'.format(self.open_layers, epoch + 1, self.fixbase_epoch))
            open_specified_layers(self.model, self.open_layers)
        else:
            open_all_layers(self.model)

        if epoch % 5 == 0 and epoch != 0:
            visualize_3d_rgb(self.model, self.trainloader, osp.join(self.save_dir, 'pretrain3d_epoch_' + str(epoch)), **kwargs)
            self.model.train()

        self.optimizer_reid.step()
        end = time.time()
        realmotionloader = iter(self.realmotionloader)
        for batch_idx, data in enumerate(self.trainloader):
            data_time.update(time.time() - end)

            self.scheduler_reid.step(epoch + float(batch_idx) / len(self.trainloader))

            try:
                real_motion_params = next(realmotionloader)
            except StopIteration:
                realmotionloader = iter(self.realmotionloader)
                real_motion_params = next(realmotionloader)

            imgs, pids, joints, inputs3d, img_paths, masks = self._parse_data(data)

            if self.use_gpu:
                imgs = imgs.cuda()
                pids = pids.cuda()
                joints = joints.cuda()
                inputs3d = inputs3d.cuda()
                masks = masks.cuda()
                real_motion_params = real_motion_params.cuda()

            # self.optimizer.zero_grad()

            # sys.exit()

            self.optimizer_reid.zero_grad()
            self.optimizer3d_enc.zero_grad()
            cent_items, trip_items, pred_params, pred_outputs1, pred_outputs2, encoder_disc_value, gen_disc_value, real_disc_value\
                = self.model(imgs, inputs3d, real_motion_params)
            # cent_items, trip_items, pred_params, encoder_disc_value, gen_disc_value, real_disc_value, pred_outputs1 \
            #     = self.model(imgs, inputs3d, real_motion_params)
            # trip_items, pred_params, pred_outputs1, pred_outputs2, encoder_disc_value, gen_disc_value, real_disc_value \
            #     = self.model(inputs_3d, real_motion_params)

            # 3d->2d joints loss
            joints2d_pred = proj_2djoint(pred_params['cam'], pred_outputs1.joints)
            loss_joint = keypoint_loss(joints2d_pred, joints, self.criterion_keypoints)

            # 3d->2d silhouette loss
            loss_silhouette = self.criterion_silhouette(pred_outputs1.vertices, pred_params['cam'], masks)

            # vertice-wise loss to constrain the scale of free-form displacement D
            loss_vertice = self.criterion_vertice(pred_outputs1.vertices, pred_outputs2.vertices)

            # reid loss
            losses_cent_list = list()
            for item in cent_items:
                loss_cent = self._compute_loss(self.criterion_x, item, pids)
                losses_cent_list.append(loss_cent.unsqueeze(0))
            losses_trip_list = list()
            for item in trip_items:
                loss_trip = self._compute_loss(self.criterion_t, item, pids)
                losses_trip_list.append(loss_trip.unsqueeze(0))

            loss_cent_sum = torch.sum(torch.cat(losses_cent_list))
            loss_trip_sum = torch.sum(torch.cat(losses_trip_list))

            # # orthogonal loss
            # combine_feat0 = torch.cat([orthogonal_items[0].unsqueeze(2).unsqueeze(3), orthogonal_items[1].unsqueeze(2).unsqueeze(3)], dim=2)
            # combine_feat1 = torch.cat([orthogonal_items[2].unsqueeze(2).unsqueeze(3), orthogonal_items[3].unsqueeze(2).unsqueeze(3)], dim=2)
            # loss_of = self.of_penalty(combine_feat0) + self.of_penalty(combine_feat1)

            # adversarial loss
            loss_enc_adv = self.criterion_encoder_adv(encoder_disc_value)
            loss_disc_adv = self.criterion_discri_adv(real_disc_value, gen_disc_value)

            loss_enc = self.joint_loss_weight * loss_joint + self.silhouette_loss_weight * loss_silhouette \
                       + self.vertice_loss_weight * loss_vertice + loss_enc_adv + loss_cent_sum + loss_trip_sum
            # loss_enc = self.silhouette_loss_weight * loss_silhouette + loss_enc_adv + loss_cent_sum + loss_trip_sum
            loss_disc = loss_disc_adv

            losses_cent.update(loss_cent_sum.item()/len(cent_items), pids.size(0))
            losses_trip.update(loss_trip_sum.item()/len(trip_items), pids.size(0))

            # # add apex setting
            # with amp.scale_loss(loss_enc, [self.optimizer_encoder, self.optimizer_reid]) as scaled_loss:
            #     scaled_loss.backward()
            # with amp.scale_loss(loss_disc, self.optimizer_disc) as scaled_loss:
            #     scaled_loss.backward()

            loss_enc.backward()
            self.optimizer_reid.step()
            self.optimizer3d_enc.step()

            self.optimizer3d_disc.zero_grad()
            loss_disc.backward()
            self.optimizer3d_disc.step()

            batch_time.update(time.time() - end)

            losses_enc.update(loss_enc.item(), pids.size(0))
            losses_disc.update(loss_disc.item(), pids.size(0))
            losses_joint.update(loss_joint.item(), pids.size(0))
            losses_vertice.update(loss_vertice.item(), pids.size(0))
            losses_silhouette.update(loss_silhouette.item(), pids.size(0))
            losses_enc_adv.update(loss_enc_adv.item(), self.batch_size)
            losses_disc_adv.update(loss_disc_adv.item(), self.batch_size)

            if (batch_idx) % self.print_freq == 0:
                # estimate remaining time
                num_batches = self.train_len
                eta_seconds = batch_time.avg * (
                            num_batches - (batch_idx + 1) + (self.max_epoch - (epoch + 1)) * num_batches)
                eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
                print('Epoch: [{0}/{1}][{2}/{3}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss_Enc {loss_enc.val:.4f} ({loss_enc.avg:.4f})\t'
                      'Loss_Disc {loss_disc.val:.4f} ({loss_disc.avg:.4f})\t'
                      'C {loss_cent.val:.4f} ({loss_cent.avg:.4f})\t'
                      'T {loss_trip.val:.4f} ({loss_trip.avg:.4f})\t'
                      'J {loss_joint.val:.4f} ({loss_joint.avg:.4f})\t'
                      'Silhouette {loss_sil.val:.4f} ({loss_sil.avg:.4f})\t'
                      'Vertice {loss_vertice.val:.4f} ({loss_vertice.avg:.4f})\t'
                      'Enc Adv {loss_enc_adv.val:.4f} ({loss_enc_adv.avg:.4f})\t'
                      'Disc Adv {loss_disc_adv.val:.4f} ({loss_disc_adv.avg:.4f})\t'
                      'Lr1 {lr1:.6f}\t'
                      'Lr2 {lr2:.6f}\t'
                      'Lr3 {lr3:.6f}\t'
                      'Eta {eta}'.format(
                      epoch + 1, self.max_epoch, batch_idx + 1, self.train_len,
                      batch_time=batch_time,
                      data_time=data_time,
                      loss_enc=losses_enc,
                      loss_disc=losses_disc,
                      loss_cent=losses_cent,
                      loss_trip=losses_trip,
                      loss_joint=losses_joint,
                      loss_sil=losses_silhouette,
                      loss_vertice=losses_vertice,
                      loss_enc_adv=losses_enc_adv,
                      loss_disc_adv=losses_disc_adv,
                    # lr=self.optimizer.param_groups[0]['lr'],
                    lr1=self.optimizer3d_enc.param_groups[0]['lr'],
                    lr2=self.optimizer3d_disc.param_groups[0]['lr'],
                    lr3=self.optimizer_reid.param_groups[0]['lr'],
                    eta=eta_str
                )
                )

            end = time.time()

        # if self.scheduler is not None:
        #     self.scheduler.step()

        self.scheduler3d_enc.step()
        self.scheduler3d_disc.step()

    def _parse_data(self, data):
        imgs = data[0]
        pids = data[1]
        img_paths = data[3]

        joints = data[4]

        inputs3d = data[6]
        masks = data[7]

        return imgs, pids, joints, inputs3d, img_paths, masks

    def _compute_loss(self, criterion, outputs, targets):
        if isinstance(outputs, (tuple, list)):
            if isinstance(targets, (tuple, list)):
                loss = DeepSupervisionDIM(criterion, outputs, targets)
            else:
                loss = DeepSupervision(criterion, outputs, targets)
        else:
            loss = criterion(outputs, targets)
        return loss


