from __future__ import absolute_import
from __future__ import print_function

import argparse


def init_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--app', type=str, default='image', choices=['image', 'video'],
                        help='application')
    parser.add_argument('--loss', type=str, default='softmax',
                        choices=['softmax', 'softmax_compare', 'triplet', 'triplet_compare', 'triplet_baseline',
                                 'triplet_feat_split', 'triplet_dml'],
                        help='methodology')
    parser.add_argument('--flag-general', action='store_true', help='if use general protocal')

    # ************************************************************
    # Datasets
    # ************************************************************
    parser.add_argument('--root', type=str, default='reid-data', required=True,
                        help='root path to data directory')
    parser.add_argument('-s', '--sources', type=str, required=True,
                        help='source datasets (delimited by space)')
    parser.add_argument('-t', '--targets', type=str, required=False, nargs='+',
                        help='target datasets (delimited by space)')
    parser.add_argument('-j', '--workers', type=int, default=4,
                        help='number of data loading workers (tips: 4 or 8 times number of gpus)')
    parser.add_argument('--split-id', type=int, default=0,
                        help='split index (note: 0-based)')
    parser.add_argument('--height', type=int, default=256,
                        help='height of an image')
    parser.add_argument('--width', type=int, default=128,
                        help='width of an image')
    parser.add_argument('--train-sampler', type=str, default='RandomSampler',
                        help='sampler for trainloader')
    parser.add_argument('--combineall', action='store_true',
                        help='combine all data in a dataset (train+query+gallery) for training')
    parser.add_argument('--transforms', type=str, nargs='+',
                        help='transformations applied to model training')

    # ************************************************************
    # Video datasets
    # ************************************************************
    parser.add_argument('--seq-len', type=int, default=15,
                        help='number of images to sample in a tracklet')
    parser.add_argument('--sample-method', type=str, default='evenly',
                        help='how to sample images from a tracklet')
    parser.add_argument('--pooling-method', type=str, default='avg', choices=['avg', 'max'],
                        help='how to pool features over a tracklet (for video reid)')

    # ************************************************************
    # Dataset-specific setting
    # ************************************************************
    parser.add_argument('--cuhk03-labeled', action='store_true',
                        help='use labeled images, if false, use detected images')
    parser.add_argument('--cuhk03-classic-split', action='store_true',
                        help='use classic split by Li et al. CVPR\'14')
    parser.add_argument('--use-metric-cuhk03', action='store_true',
                        help='use cuhk03\'s metric for evaluation')

    parser.add_argument('--market1501-500k', action='store_true',
                        help='add 500k distractors to the gallery set for market1501')

    # ************************************************************
    # Optimization options
    # ************************************************************
    parser.add_argument('--optim', type=str, default='adam',
                        help='optimization algorithm (see optimizers.py)')
    parser.add_argument('--lr', type=float, default=0.0003,
                        help='initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=5e-04,
                        help='weight decay')
    # sgd
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum factor for sgd and rmsprop')
    parser.add_argument('--sgd-dampening', type=float, default=0,
                        help='sgd\'s dampening for momentum')
    parser.add_argument('--sgd-nesterov', action='store_true',
                        help='whether to enable sgd\'s Nesterov momentum')
    # rmsprop
    parser.add_argument('--rmsprop-alpha', type=float, default=0.99,
                        help='rmsprop\'s smoothing constant')
    # adam/amsgrad
    parser.add_argument('--adam-beta1', type=float, default=0.9,
                        help='exponential decay rate for adam\'s first moment')
    parser.add_argument('--adam-beta2', type=float, default=0.999,
                        help='exponential decay rate for adam\'s second moment')

    # ************************************************************
    # Training hyperparameters
    # ************************************************************
    parser.add_argument('--max-epoch', type=int, default=60,
                        help='maximum epochs to run')
    parser.add_argument('--start-epoch', type=int, default=0,
                        help='manual epoch number (useful when restart)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='batch size')

    parser.add_argument('--fixbase-epoch', type=int, default=0,
                        help='number of epochs to fix base layers')
    parser.add_argument('--open-layers', type=str, nargs='+', default=None,
                        help='open specified layers for training while keeping others frozen')
    parser.add_argument('--fixed-layers', type=str, nargs='+', default=None,
                        help='fix specified layers for training while keeping others training')
    parser.add_argument('--staged-lr', action='store_true',
                        help='set different lr to different layers')
    parser.add_argument('--new-layers', type=str, nargs='+', default=['classifier'],
                        help='newly added layers with default lr')
    parser.add_argument('--base-lr-mult', type=float, default=1,
                        help='learning rate multiplier for base layers')
    parser.add_argument('--specific-layers', type=str, nargs='+', default=None,
                        help='specific layers with specific lr')
    parser.add_argument('--specific-lr', type=float, default=0.002,
                        help='specific lr for specific layers')

    # ************************************************************
    # Learning rate scheduler options
    # ************************************************************
    parser.add_argument('--lr-scheduler', type=str, default='multi_step',
                        help='learning rate scheduler (see lr_schedulers.py)')
    parser.add_argument('--stepsize', type=int, default=[20, 40], nargs='+',
                        help='stepsize to decay learning rate')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='learning rate decay')

    # ************************************************************
    # Cross entropy loss
    # ************************************************************
    parser.add_argument('--label-smooth', action='store_true',
                        help='use label smoothing regularizer in cross entropy loss')

    # ************************************************************
    # Hard triplet loss
    # ************************************************************
    parser.add_argument('--margin', type=float, default=0.3,
                        help='margin for triplet loss')
    parser.add_argument('--num-instances', type=int, default=4,
                        help='number of instances per identity')
    parser.add_argument('--weight-t', type=float, default=1,
                        help='weight to balance hard triplet loss')
    parser.add_argument('--weight-x', type=float, default=0,
                        help='weight to balance cross entropy loss (default is 0)')

    # ************************************************************
    # DIMloss
    # ************************************************************
    parser.add_argument('--joint-loss-weight', type=float, default=1,
                        help='weight of proj joint loss')
    parser.add_argument('--dim-margin', type=float, default=0.8,
                        help='margin for dim loss')

    # ************************************************************
    # Architecture
    # ************************************************************
    parser.add_argument('-a', '--arch', type=str, default='resnet50',
                        help='model architecture')
    parser.add_argument('--no-pretrained', action='store_true',
                        help='do not load pretrained weights')
    parser.add_argument('--part-num-rgb', type=int, default=3,
                        help='part number of RGB feature')
    parser.add_argument('--part-num-contour', type=int, default=3,
                        help='part number of contour feature')

    # ************************************************************
    # Test settings
    # ************************************************************
    parser.add_argument('--load-weights', type=str, default='',
                        help='load pretrained weights but ignore layers that do not match in size')
    parser.add_argument('--evaluate', action='store_true',
                        help='evaluate only')
    parser.add_argument('--eval-freq', type=int, default=-1,
                        help='evaluation frequency (set to -1 to test only in the end)')
    parser.add_argument('--start-eval', type=int, default=0,
                        help='start to evaluate after a specific epoch')
    parser.add_argument('--dist-metric', type=str, default='euclidean',
                        help='distance metric')
    parser.add_argument('--normalize-feature', action='store_true',
                        help='normalize feature vectors before calculating distance')
    parser.add_argument('--ranks', type=int, default=[1, 5, 10, 20], nargs='+',
                        help='cmc ranks')
    parser.add_argument('--rerank', action='store_true',
                        help='use person re-ranking (by Zhong et al. CVPR2017)')

    parser.add_argument('--visrank', action='store_true',
                        help='visualize ranked results, only available in evaluation mode')
    parser.add_argument('--visrank-topk', type=int, default=10,
                        help='visualize topk ranks')
    parser.add_argument('--visactmap', action='store_true',
                        help='visualize CNN activation maps')

    # ************************************************************
    # Miscs
    # ************************************************************
    parser.add_argument('--print-freq', type=int, default=20,
                        help='print frequency')
    parser.add_argument('--seed', type=int, default=1,
                        help='manual seed')
    parser.add_argument('--resume', type=str, default='', metavar='PATH',
                        help='resume from a checkpoint')
    parser.add_argument('--save-dir', type=str, default='log',
                        help='path to save log and model weights')
    parser.add_argument('--use-cpu', action='store_true',
                        help='use cpu')
    parser.add_argument('--gpu-devices', type=str, default='0',
                        help='gpu device ids for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--use-avai-gpus', action='store_true',
                        help='use available gpus instead of specified devices')

    return parser


def imagedata_kwargs(parsed_args):
    return {
        'root': parsed_args.root,
        'source': parsed_args.sources,
        'target': parsed_args.targets,
        'height': parsed_args.height,
        'width': parsed_args.width,
        'transforms': parsed_args.transforms,
        'use_cpu': parsed_args.use_cpu,
        'split_id': parsed_args.split_id,
        'batch_size': parsed_args.batch_size,
        'workers': parsed_args.workers,
        'num_instances': parsed_args.num_instances,
        'train_sampler': parsed_args.train_sampler,
        # image
        'cuhk03_labeled': parsed_args.cuhk03_labeled,
        'cuhk03_classic_split': parsed_args.cuhk03_classic_split,
        'market1501_500k': parsed_args.market1501_500k,
    }


def optimizer_kwargs(parsed_args):
    return {
        'optim': parsed_args.optim,
        'lr': parsed_args.lr,
        'weight_decay': parsed_args.weight_decay,
        'momentum': parsed_args.momentum,
        'sgd_dampening': parsed_args.sgd_dampening,
        'sgd_nesterov': parsed_args.sgd_nesterov,
        'rmsprop_alpha': parsed_args.rmsprop_alpha,
        'adam_beta1': parsed_args.adam_beta1,
        'adam_beta2': parsed_args.adam_beta2,
        'staged_lr': parsed_args.staged_lr,
        'new_layers': parsed_args.new_layers,
        'base_lr_mult': parsed_args.base_lr_mult,
        'specific_layers': parsed_args.specific_layers,
        'specific_lr': parsed_args.specific_lr
    }


def lr_scheduler_kwargs(parsed_args):
    return {
        'lr_scheduler': parsed_args.lr_scheduler,
        'stepsize': parsed_args.stepsize,
        'gamma': parsed_args.gamma,
        'max_epoch': parsed_args.max_epoch
    }


def engine_kwargs(parsed_args):
    return {
        'save_dir': parsed_args.save_dir,
        'max_epoch': parsed_args.max_epoch,
        'start_epoch': parsed_args.start_epoch,
        'fixbase_epoch': parsed_args.fixbase_epoch,
        'open_layers': parsed_args.open_layers,
        'fixed_layers': parsed_args.fixed_layers,
        'start_eval': parsed_args.start_eval,
        'eval_freq': parsed_args.eval_freq,
        'test_only': parsed_args.evaluate,
        'print_freq': parsed_args.print_freq,
        'dist_metric': parsed_args.dist_metric,
        'normalize_feature': parsed_args.normalize_feature,
        'visrank': parsed_args.visrank,
        'visrank_topk': parsed_args.visrank_topk,
        'use_metric_cuhk03': parsed_args.use_metric_cuhk03,
        'ranks': parsed_args.ranks,
        'rerank': parsed_args.rerank,
        'if_visactmap': parsed_args.visactmap,
        'margin': parsed_args.margin,
        'label_smooth': parsed_args.label_smooth,
        'height': parsed_args.height,   # visualize ranklist
        'width': parsed_args.width,
        'flag_general': parsed_args.flag_general,
        'batch_size': parsed_args.batch_size,
        'joint_loss_weight': parsed_args.joint_loss_weight
    }
