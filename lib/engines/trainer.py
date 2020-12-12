import time
import datetime
from apex import amp

import torch
from torch import nn
import torch.nn.functional as F

from lib.utils import AverageMeter, open_all_layers, open_specified_layers, OFPenalty
from lib.losses import *
from lib.optim import CosineAnnealingWarmUp


class Trainer(object):
    def __init__(self, trainloader, model, optimizer, scheduler,
                 max_epoch, num_train_pids, use_gpu=True, fixbase_epoch=0, open_layers=None,
                 print_freq=10, margin=0.3, label_smooth=True, save_dir=None, **kwargs):
        self.trainloader = trainloader
        self.model = model
        # self.optimizer = optimizer
        # self.scheduler = scheduler

        self.optimizer3d_enc, self.optimizer_reid = optimizer
        # self.optimizer_reid = optimizer[0]
        self.scheduler = CosineAnnealingWarmUp(self.optimizer_reid, T_0=5,
                                               T_end=80, warmup_factor=0,
                                               last_epoch=-1)

        self.scheduler3d_enc = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer3d_enc,
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

        self.criterion_x = nn.CrossEntropyLoss().cuda()
        # self.criterion_x = CrossEntropyLoss(
        #     num_classes=num_train_pids,
        #     use_gpu=self.use_gpu,
        #     label_smooth=label_smooth
        # )
        self.criterion_t = TripletLoss(margin=margin)

        # self.criterion_kl = torch.nn.KLDivLoss(reduction='batchmean')

        # self.criterion_t_sen = TripletLoss_Cloth_Sen(margin=margin, num_class=num_train_pids)
        # self.criterion_x_sen = CrossEntropyLoss(
        #     num_classes=num_train_pids*2,
        #     use_gpu=self.use_gpu,
        #     label_smooth=label_smooth
        # )

        # self.criterion_t_insen = TripletLoss_Cloth_Insen(margin=margin, num_class=num_train_pids)
        # self.criterion_x_insen = CrossEntropyLoss(
        #     num_classes=num_train_pids,
        #     use_gpu=self.use_gpu,
        #     label_smooth=label_smooth
        # )

        # orthogonal constraint
        # self.of_penalty = OFPenalty()

        self.pretrain_3d_epoch = 0  # just for prevent bugs, not for specific objects

    def train(self, epoch, queryloader=None, **kwargs):
        losses = AverageMeter()
        losses_trip1 = AverageMeter()
        losses_cent1 = AverageMeter()
        losses_dml = AverageMeter()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        self.model.train()
        if (epoch + 1) <= self.fixbase_epoch and self.open_layers is not None:
            print('* Only train {} (epoch: {}/{})'.format(self.open_layers, epoch + 1, self.fixbase_epoch))
            open_specified_layers(self.model, self.open_layers)
        else:
            open_all_layers(self.model)

        self.optimizer_reid.step()
        end = time.time()
        for batch_idx, data in enumerate(self.trainloader):
            data_time.update(time.time() - end)

            self.scheduler.step(epoch + float(batch_idx) / len(self.trainloader))

            imgs, pids, inputs3d = self._parse_data(data)
            if self.use_gpu:
                imgs = imgs.cuda()
                pids = pids.cuda()
                inputs3d = inputs3d.cuda()

            self.optimizer_reid.zero_grad()
            self.optimizer3d_enc.zero_grad()
            cent_items, trip_items = self.model(imgs, inputs3d)
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

            # # deep mutual learning loss
            # outputs1, outputs2 = cent_items
            # loss_dml1 = self.criterion_kl(F.log_softmax(outputs1, dim=1), F.softmax(outputs2, dim=1))
            # loss_dml2 = self.criterion_kl(F.log_softmax(outputs2, dim=1), F.softmax(outputs1, dim=1))
            # loss_dml = loss_dml1 + loss_dml2

            # cloth_insen_items, cloth_sen_items, orthogonal_items = self.model(imgs)
            #
            # loss_cent_insen = self._compute_loss(self.criterion_x_insen, cloth_insen_items[0], pids)
            # loss_trip_insen = self._compute_loss(self.criterion_t_insen, cloth_insen_items[1], pids_relabel)
            # loss_cent_sen = self._compute_loss(self.criterion_x_sen, cloth_sen_items[0], pids_relabel)
            # loss_trip_sen = self._compute_loss(self.criterion_t_sen, cloth_sen_items[1], pids_relabel)

            # # orthogonal loss
            # combine_feat0 = torch.cat([orthogonal_items[0].unsqueeze(2).unsqueeze(3), orthogonal_items[1].unsqueeze(2).unsqueeze(3)], dim=2)
            # combine_feat1 = torch.cat([orthogonal_items[2].unsqueeze(2).unsqueeze(3), orthogonal_items[3].unsqueeze(2).unsqueeze(3)], dim=2)
            # loss_of = self.of_penalty(combine_feat0) + self.of_penalty(combine_feat1)

            # loss = loss_cent_insen + loss_trip_insen + loss_cent_sen + loss_trip_sen + loss_of * 5
            # loss = loss_cent_insen + loss_trip_insen + loss_cent_sen + loss_trip_sen
            loss = loss_cent_sum + loss_trip_sum

            # add apex setting
            # with amp.scale_loss(loss, self.optimizer) as scaled_loss:
            #     scaled_loss.backward()
            loss.backward()
            self.optimizer_reid.step()
            self.optimizer3d_enc.step()

            batch_time.update(time.time() - end)

            losses.update(loss.item(), pids.size(0))
            losses_cent1.update(loss_cent_sum.item()/len(cent_items), pids.size(0))
            losses_trip1.update(loss_trip_sum.item()/len(trip_items), pids.size(0))
            # losses_dml.update(loss_dml.item(), pids.size(0))

            # losses_cent1.update(loss_cent_insen.item(), pids.size(0))
            # losses_trip1.update(loss_trip_insen.item(), pids.size(0))
            # losses_cent2.update(loss_cent_sen.item(), pids.size(0))
            # losses_trip2.update(loss_trip_sen.item(), pids.size(0))

            # losses_of.update(loss_of.item(), pids.size(0))

            if (batch_idx) % self.print_freq == 0:
                # estimate remaining time
                num_batches = self.train_len
                eta_seconds = batch_time.avg * (
                            num_batches - (batch_idx + 1) + (self.max_epoch - (epoch + 1)) * num_batches)
                eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
                print('Epoch: [{0}/{1}][{2}/{3}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'C {loss_cent1.val:.4f} ({loss_cent1.avg:.4f})\t'
                      'T {loss_trip1.val:.4f} ({loss_trip1.avg:.4f})\t'
                      'DML {loss_dml.val:.4f} ({loss_dml.avg:.4f})\t'
                      'Lr1 {lr:.6f}\t'
                      'Eta {eta}'.format(
                    epoch + 1, self.max_epoch, batch_idx + 1, self.train_len,
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    loss_cent1=losses_cent1,
                    loss_trip1=losses_trip1,
                    loss_dml=losses_dml,
                    lr=self.optimizer_reid.param_groups[0]['lr'],
                    eta=eta_str
                )
                )

            end = time.time()

        # if self.scheduler is not None:
        #     self.scheduler.step()
        self.scheduler3d_enc.step()

    def _parse_data(self, data):
        imgs = data[0]
        pids = data[1]
        inputs3d = data[6]
        # pids_relabel = data[8]

        return imgs, pids, inputs3d

    def _compute_loss(self, criterion, outputs, targets):
        if isinstance(outputs, (tuple, list)):
            if isinstance(targets, (tuple, list)):
                loss = DeepSupervisionDIM(criterion, outputs, targets)
            else:
                loss = DeepSupervision(criterion, outputs, targets)
        else:
            loss = criterion(outputs, targets)
        return loss
