import sys
import os
import os.path as osp
import warnings
import time
from apex import amp

import torch
import torch.nn as nn

from opts import (
    init_parser, imagedata_kwargs,optimizer_kwargs,
    lr_scheduler_kwargs, engine_kwargs
)

from lib.dataset import init_image_dataset
from lib.models import build_model
from lib.optim import build_optimizer, build_lr_scheduler
from lib.engines import Engine
from lib.utils import DataWarpper, Logger
from lib.utils import (build_transforms, build_train_sampler, load_pretrained_weights,
                       resume_from_checkpoint, check_isfile, collect_env_info, set_random_seed,
                       compute_model_complexity)

parser = init_parser()
args = parser.parse_args()

def main():
    global args

    set_random_seed(args.seed)
    if not args.use_avai_gpus:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    use_gpu = torch.cuda.is_available() and not args.use_cpu
    log_name = 'test.log' if args.evaluate else 'train.log'
    log_name += time.strftime('-%Y-%m-%d-%H-%M-%S')
    sys.stdout = Logger(osp.join(args.save_dir, log_name))
    print('** Arguments **')
    arg_keys = list(args.__dict__.keys())
    arg_keys.sort()
    for key in arg_keys:
        print('{}: {}'.format(key, args.__dict__[key]))
    print('\n')
    print('Collecting env info ...')
    print('** System info **\n{}\n'.format(collect_env_info()))
    if use_gpu:
        torch.backends.cudnn.benchmark = True
        # torch.backends.cudnn.benchmark = False
        # torch.backends.cudnn.deterministic = True
    else:
        warnings.warn('Currently using CPU, however, GPU is highly recommended')

    # load data related args
    data_args = imagedata_kwargs(args)

    # initialize dataset
    dataset = init_image_dataset(name=data_args['source'], **data_args)

    # build data transformer
    transforms_tr, transforms_te = build_transforms(**data_args)

    # load train data
    trainset = dataset.train
    train_sampler = build_train_sampler(
        trainset, data_args['train_sampler'],
        batch_size=data_args['batch_size'],
        num_instances=data_args['num_instances'],
        num_train_pids=dataset.num_train_pids
    )
    trainloader = torch.utils.data.DataLoader(
        DataWarpper(data=trainset, transforms=transforms_tr),
        sampler=train_sampler,
        batch_size=data_args['batch_size'],
        shuffle=False,
        num_workers=data_args['workers'],
        pin_memory=False,
        drop_last=True,
    )

    # load test data
    queryset = dataset.query
    queryloader = torch.utils.data.DataLoader(
        DataWarpper(data=queryset, transforms=transforms_te),
        batch_size=data_args['batch_size'],
        shuffle=False,
        num_workers=data_args['workers'],
        pin_memory=False,
        drop_last=False
    )

    galleryset = dataset.gallery
    galleryloader = torch.utils.data.DataLoader(
        DataWarpper(data=galleryset, transforms=transforms_te),
        batch_size=data_args['batch_size'],
        shuffle=False,
        num_workers=data_args['workers'],
        pin_memory=False,
        drop_last=False
    )

    print('Building model: {}'.format(args.arch))
    model = build_model(
        name=args.arch,
        num_classes=dataset.num_train_pids,
        pretrained=(not args.no_pretrained),
        use_gpu=use_gpu,
        batch_size=args.batch_size,
        part_num=args.part_num,
        part_weight=args.part_weight
    )
    model = model.cuda()

    # num_params, flops = compute_model_complexity(model, (1, 3, args.height, args.width))
    # print('Model complexity: params={:,} flops={:,}'.format(num_params, flops))

    if args.load_weights and check_isfile(args.load_weights):
        load_pretrained_weights(model, args.load_weights)

    optimizer = build_optimizer(model, **optimizer_kwargs(args))

    scheduler = build_lr_scheduler(optimizer, **lr_scheduler_kwargs(args))

    model, optimizer = amp.initialize(model, optimizer,
                                      opt_level="O1",
                                      keep_batchnorm_fp32=None,
                                      loss_scale=None)

    if use_gpu:
        model = nn.DataParallel(model)

    if args.resume and check_isfile(args.resume):
        args.start_epoch = resume_from_checkpoint(args.resume, model, optimizer=optimizer)

    print('Building {}-engine for {}-reid'.format(args.loss, args.app))
    engine = Engine(trainloader, queryloader, galleryloader, model, optimizer, scheduler,
                    query=queryset, gallery=galleryset, use_gpu=use_gpu, num_train_pids=dataset.num_train_pids, **engine_kwargs(args))
    engine.run(**engine_kwargs(args), use_gpu=use_gpu)


if __name__ == '__main__':
    torch.set_num_threads(6)  # set the number of multi processor
    torch.multiprocessing.set_sharing_strategy('file_system')
    main()
