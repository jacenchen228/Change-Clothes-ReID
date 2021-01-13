import time
import datetime
from apex import amp

import torch
from torch import nn
import torch.nn.functional as F

from lib.utils import AverageMeter, open_all_layers, open_specified_layers, OFPenalty
from lib.losses import *


class Trainer(object):
    def __init__(self, trainloader, model, optimizer, scheduler,
                 max_epoch, num_train_pids, use_gpu=True, fixbase_epoch=0, open_layers=None,
                 print_freq=10, margin=0.3, label_smooth=True, save_dir=None, **kwargs):
        self.trainloader = trainloader
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.use_gpu = use_gpu
        self.train_len = len(self.trainloader)
        self.max_epoch = max_epoch
        self.fixbase_epoch = fixbase_epoch
        self.open_layers = open_layers
        self.print_freq = print_freq
        self.save_dir = save_dir

        # self.criterion_x = nn.CrossEntropyLoss().cuda()
        self.criterion_x = CrossEntropyLabelSmooth(num_train_pids)
        # self.criterion_x = CircleLoss(margin=0.25, gamma=128)
        self.criterion_t = TripletLoss(margin=margin)
        self.criterion_dim = DeepInfoMaxLoss(margin=0.8)

    def train(self, epoch, **kwargs):
        losses = AverageMeter()
        losses_trip = AverageMeter()
        losses_cent = AverageMeter()
        losses_dim = AverageMeter()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        self.model.train()
        if (epoch + 1) <= self.fixbase_epoch and self.open_layers is not None:
            print('* Only train {} (epoch: {}/{})'.format(self.open_layers, epoch + 1, self.fixbase_epoch))
            open_specified_layers(self.model, self.open_layers)
        else:
            open_all_layers(self.model)

        end = time.time()
        for batch_idx, data in enumerate(self.trainloader):
            data_time.update(time.time() - end)

            # self.scheduler.step(epoch + float(batch_idx) / len(self.trainloader))

            imgs, contours, pids = self._parse_data(data)
            if self.use_gpu:
                imgs = imgs.cuda()
                contours = contours.cuda()
                pids = pids.cuda()

            self.optimizer.zero_grad()

            # loss for our model
            cent_items, trip_items, ejs, ems, ejs_part, ems_part = self.model(imgs, contours, targets=pids)

            # cent_items, trip_items, ejs, ems = self.model(imgs, contours, targets=pids)

            # loss for baseline model (without mutual infoamtion related loss)
            # cent_items, trip_items = self.model(imgs, contours)

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

            loss = loss_cent_sum + loss_trip_sum

            # calculate dim loss
            loss_dim1 = self._compute_loss(self.criterion_dim, ejs, ems)
            loss_dim2 = self._compute_loss(self.criterion_dim, ejs_part, ems_part)
            loss_dim = loss_dim1 + loss_dim2
            # loss_dim = loss_dim1
            loss += 0.1 * loss_dim

            # add apex setting
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
            # loss.backward()

            self.optimizer.step()

            batch_time.update(time.time() - end)

            losses.update(loss.item(), pids.size(0))
            losses_cent.update(loss_cent_sum.item()/len(cent_items), pids.size(0))
            losses_trip.update(loss_trip_sum.item()/len(trip_items), pids.size(0))
            losses_dim.update(loss_dim.item(), pids.size(0))

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
                      'DIM {loss_dim.val:.4f} ({loss_dim.avg:.4f})\t'
                      'Lr1 {lr:.6f}\t'
                      'Eta {eta}'.format(
                      epoch + 1, self.max_epoch, batch_idx + 1, self.train_len,
                      batch_time=batch_time,
                      data_time=data_time,
                      loss=losses,
                      loss_cent1=losses_cent,
                      loss_trip1=losses_trip,
                      loss_dim=losses_dim,
                      lr=self.optimizer.param_groups[0]['lr'],
                      eta=eta_str
                )
                )

            end = time.time()

        self.scheduler.step()

    def _parse_data(self, data):
        imgs = data[0]
        contours = data[1]
        pids = data[2]

        return imgs, contours, pids

    def _compute_loss(self, criterion, outputs, targets):
        if isinstance(outputs, (tuple, list)):
            if isinstance(targets, (tuple, list)):
                loss = DeepSupervisionDIM(criterion, outputs, targets)
            else:
                loss = DeepSupervision(criterion, outputs, targets)
        else:
            loss = criterion(outputs, targets)
        return loss
