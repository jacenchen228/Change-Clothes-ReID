import time
import datetime
import sys
import os.path as osp
# from apex import amp

import torch

from lib.utils import AverageMeter, open_all_layers, open_specified_layers, visualize_3d_rgb
from lib.losses import *
# from lib.hmr import shape_l2_loss, pose_l2_loss, kp_2d_l1_loss
from lib.lib3D import keypoint_loss, proj_2djoint, SilhouetteLoss, IMG_RES, FOCAL_LENGTH, VerticeLoss, \
    batch_encoder_disc_l2_loss, batch_adv_disc_l2_loss, RealMotionWarpper


class TrainerOnly3D(object):
    def __init__(self, trainloader, model, optimizer, scheduler,
                 max_epoch, num_train_pids, use_gpu=True, fixbase_epoch=0, open_layers=None,
                 print_freq=10, margin=0.3, label_smooth=True, batch_size=64, pretrain_3d_epoch=0, save_dir=None, joint_loss_weight=0.0001, **kwargs):
        self.trainloader = trainloader
        self.model = model

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
        self.optimizer_encoder = torch.optim.Adam(
            self.model.module.estimator3D.parameters(),
            lr=0.00001,
            weight_decay=0.0001
        )
        self.optimizer_disc = torch.optim.Adam(
            self.model.module.discriminator.parameters(),
            lr=0.0001,
            weight_decay=0.0001
        )

        self.scheduler_encoder = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer_encoder,
            milestones=[20, 40],
            gamma=0.9
        )

        self.scheduler_disc = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer_disc,
            milestones=[20, 40],
            gamma=0.9
        )

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
        self.vertice_loss_weight = 900.0

        self.pretrain_3d_epoch_joint = 0

    def train(self, epoch, queryloader=None, **kwargs):
        losses = AverageMeter()
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

        end = time.time()
        realmotionloader = iter(self.realmotionloader)
        for batch_idx, data in enumerate(self.trainloader):
            data_time.update(time.time() - end)

            try:
                real_motion_params = next(realmotionloader)
            except StopIteration:
                realmotionloader = iter(self.realmotionloader)
                real_motion_params = next(realmotionloader)

            pids, joints, inputs_3d, img_paths, masks = self._parse_data(data)

            if self.use_gpu:
                pids = pids.cuda()
                joints = joints.cuda()
                inputs_3d = inputs_3d.cuda()
                masks = masks.cuda()
                real_motion_params = real_motion_params.cuda()
                # vertices = vertices.cuda()

            self.optimizer_encoder.zero_grad()
            self.optimizer_disc.zero_grad()

            feat_3d, pred_params, pred_outputs1, pred_outputs2, encoder_disc_value, gen_disc_value, real_disc_value \
                = self.model(inputs_3d, real_motion_params)
            # feat_3d, pred_params, pred_outputs1, encoder_disc_value, gen_disc_value, real_disc_value \
            #     = self.model(inputs_3d, real_motion_params)

            # 3d joint losses V2
            joints2d_pred = proj_2djoint(pred_params['cam'], pred_outputs1.joints)
            loss_joint = keypoint_loss(joints2d_pred, joints, self.criterion_keypoints)

            # adversarial loss
            loss_enc_adv = self.criterion_encoder_adv(encoder_disc_value)
            loss_disc_adv = self.criterion_discri_adv(real_disc_value, gen_disc_value)

            # vertice-wise loss to constrain the scale of free-form displacement D
            loss_vertice = self.criterion_vertice(pred_outputs1.vertices, pred_outputs2.vertices)

            if epoch >= self.pretrain_3d_epoch_joint:
                # 3d silhouette loss1
                loss_silhouette = self.criterion_silhouette(pred_outputs1.vertices, pred_params['cam'], masks)

                loss = self.joint_loss_weight * loss_joint + self.silhouette_loss_weight * loss_silhouette + self.vertice_loss_weight * loss_vertice + loss_enc_adv
                # loss = self.joint_loss_weight * loss_joint + self.silhouette_loss_weight * loss_silhouette + loss_enc_adv
            else:
                loss = loss_joint + loss_enc_adv

            loss_disc = loss_disc_adv

            # # add apex setting
            # with amp.scale_loss(loss, self.optimizer) as scaled_loss:
            #     scaled_loss.backward()
            loss.backward()
            self.optimizer_encoder.step()

            loss_disc.backward()
            self.optimizer_disc.step()

            batch_time.update(time.time() - end)

            losses.update(loss.item(), self.batch_size)
            losses_joint.update(loss_joint.item(), self.batch_size)
            losses_enc_adv.update(loss_enc_adv.item(), self.batch_size)
            losses_disc_adv.update(loss_disc_adv.item(), self.batch_size)
            losses_vertice.update(loss_vertice.item(), self.batch_size)

            if epoch >= self.pretrain_3d_epoch_joint:
                losses_silhouette.update(loss_silhouette.item(), self.batch_size)

            if (batch_idx + 1) % self.print_freq == 0:
                # estimate remaining time
                num_batches = self.train_len
                eta_seconds = batch_time.avg * (
                            num_batches - (batch_idx + 1) + (self.max_epoch - (epoch + 1)) * num_batches)
                eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))

                print('Epoch: [{0}/{1}][{2}/{3}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'J {loss_joint.val:.4f} ({loss_joint.avg:.4f})\t'
                      'Silhouette {loss_sil.val:.4f} ({loss_sil.avg:.4f})\t'
                      'Vertice {loss_vertice.val:.4f} ({loss_vertice.avg:.4f})\t'
                      'Enc Adv {loss_enc_adv.val:.4f} ({loss_enc_adv.avg:.4f})\t'
                      'Disc Adv {loss_disc_adv.val:.4f} ({loss_disc_adv.avg:.4f})\t'
                      'Lr1 {lr1:.6f}\t'
                      'Lr2 {lr2:.6f}\t'
                      'Eta {eta}'.format(
                      epoch + 1, self.max_epoch, batch_idx + 1, self.train_len,
                      batch_time=batch_time,
                      data_time=data_time,
                      loss=losses,
                      loss_joint=losses_joint,
                      loss_sil=losses_silhouette,
                      loss_vertice=losses_vertice,
                      loss_enc_adv=losses_enc_adv,
                      loss_disc_adv=losses_disc_adv,
                      lr1=self.optimizer_encoder.param_groups[0]['lr'],
                      lr2=self.optimizer_disc.param_groups[0]['lr'],
                      eta=eta_str
                )
                )

            end = time.time()

        self.scheduler_encoder.step()
        self.scheduler_disc.step()

    def _parse_data(self, data):
        pids = data[1]
        img_paths = data[3]

        joints = data[4]

        inputs_3d = data[6]
        masks = data[7]

        # vertices = data[8]

        return pids, joints, inputs_3d, img_paths, masks

    def _compute_loss(self, criterion, outputs, targets):
        if isinstance(outputs, (tuple, list)):
            if isinstance(targets, (tuple, list)):
                loss = DeepSupervisionDIM(criterion, outputs, targets)
            else:
                loss = DeepSupervision(criterion, outputs, targets)
        else:
            loss = criterion(outputs, targets)
        return loss


