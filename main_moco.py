#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import builtins
import pathlib
import math
import string
import os
import random
import shutil
import time
import warnings

import torch
import torchvision
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from imagenet import ImageNet, SubsetSampler # Kakao brain stuff.
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import SubsetRandomSampler, Sampler, Subset, ConcatDataset
from RandAugment import RandAugment
import slm_utils.get_faa_transforms
import moco.loader
import moco.builder
import numpy as np
# SLM: add this in
# import data_loader

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


# ONLINE LOGGING
import wandb


parser = argparse.ArgumentParser(description='PyTorch SSL Training')

#########
# WANDB #
#########
parser.add_argument('--notes', type=str, default='', help='wandb notes')
default_id = ''.join([random.choice(string.ascii_letters + string.digits) for n in range(32)])
parser.add_argument('--name', type=str, default=default_id, help='wandb id/name')
parser.add_argument('--id', type=str, default=default_id, help='wandb id/name')
parser.add_argument('--wandbproj', type=str, default='autoself', help='wandb project name')

parser.add_argument('--dataid', help='id of dataset', default="cifar10", choices=('cifar10', 'imagenet', 'svhn', 'logos'))
parser.add_argument('--checkpoint-interval', default=100, type=int,
                    help='how often to checkpoint')
parser.add_argument('--nosave_latest', action='store_true', help='include flag to not save the latest epoch during trainig (risky with big datasets since youll only have the checkpoints)')
parser.add_argument('--image-log-interval', default=10, type=int,
                    help='how often to log example images')
parser.add_argument('--upload_checkpoints', action='store_true', help='Upload checkpoints to wandb')


################
# ORIGNAL ARGS #
################
parser.add_argument('--data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')

parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--checkpoint_fp', default='/userdata/smetzger/all_deepul_files/ckpts', type=str,
                     help='where to store checkpointed models')
parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

# moco specific configs:
parser.add_argument('--moco-dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--moco-k', default=65536, type=int,
                    help='queue size; number of negative keys (default: 65536)')
parser.add_argument('--moco-m', default=0.999, type=float,
                    help='moco momentum of updating key encoder (default: 0.999)')
parser.add_argument('--moco-t', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')


parser.add_argument('--kfold', default=None, type=int,
    help="which fold to use")


# options for moco v2
parser.add_argument('--mlp', action='store_true',
                    help='use mlp head')
parser.add_argument('--aug-plus', action='store_true',
                    help='use moco v2 data augmentation')
parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule')
# Fast AutoAugment Args.
parser.add_argument('--faa_aug', action='store_true',
                    help='use FastAutoAugment CIFAR10 augmentations')
parser.add_argument('--randomcrop', action='store_true',
                    help='use the random crop instead of randomresized crop, for FAA augmentations')
parser.add_argument('--gauss', action='store_true',
                    help='blur with FAA augs')

parser.add_argument('--rotnet', action='store_true', help='set true to add a rot net head')
parser.add_argument('--nomoco', action='store_true', help='set true to **not** have the moco head (moco head by default)')

# RandAug
parser.add_argument('--rand_aug', action='store_true', help='use RandAugment (set m and n appropriately)')
parser.add_argument('--rand_aug_m', default=9, type=int, help='RandAugment M (magnitude of augments)')
parser.add_argument('--rand_aug_n', default=2, type=int, help='RandAugment N (number of augs)')
parser.add_argument('--rand_aug_orig', action='store_true', help='use RandAugment orginal transforms')
parser.add_argument('--rand_aug_linear_m', action='store_true', help='use RandAugment and scale m linearly')
parser.add_argument('--rand_aug_m_min', default=4, type=int, help='RandAugment M when linearly scaling')
parser.add_argument('--rand_aug_m_max', default=11, type=int, help='RandAugment M when linearly scaling')
parser.add_argument('--rand_aug_top_k', default=0, type=int, help='RandAugment only use the top k augments')

parser.add_argument('--rand_resize_only', action='store_true', help='Use only random resized crop')
parser.add_argument('--custom_aug_name', default=None, type=str,
    help='name of custom augmentation')
parser.add_argument('--single_aug_idx', default=None, type=int, help='Which of the single augmentations to use')

parser.add_argument('--reduced_imgnet', action='store_true', help='Use a random set of 50k imagenet examples')

parser.add_argument('--sigma', type=float, default=2.0, help ='sigma for gblur')
parser.add_argument('--rrc_param', type=float, default=.2, help='rrc lower bound')

ngpus_per_node = torch.cuda.device_count()



def main():
    args = parser.parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()

    # set the checkpoint id
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size

        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    CHECKPOINT_ID = "{}_{}epochs_{}bsz_{:0.4f}lr" \
        .format(args.id[:5], args.epochs, args.batch_size, args.lr)
    if args.mlp:
        CHECKPOINT_ID += "_mlp"
    if args.aug_plus:
        CHECKPOINT_ID += "_augplus"
    if args.cos:
        CHECKPOINT_ID += "_cos"
    if args.faa_aug:
        CHECKPOINT_ID += "_faa"
    if args.randomcrop:
        CHECKPOINT_ID += "_randcrop"
    if args.rotnet:
        CHECKPOINT_ID += "_rotnet"
    if args.rand_aug:
        CHECKPOINT_ID += "_randaug"
    if not(args.kfold == None):
        CHECKPOINT_ID += "_fold_%d" %(args.kfold)
    if not(args.custom_aug_name == None):
        CHECKPOINT_ID += "_custom_aug_" + args.custom_aug_name

    CHECKPOINT_ID += args.dataid

    args.gpu = gpu

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    heads = {}
    if not args.nomoco:
        heads["moco"] = {
            "num_classes": args.moco_dim
        }
    if args.rotnet:
        heads["rotnet"] = {
            "num_classes": 4
        }
    model = moco.builder.MoCo(
        models.__dict__[args.arch],
        K=args.moco_k, m=args.moco_m, T=args.moco_t, mlp=args.mlp, dataid=args.dataid,
        multitask_heads=heads
    )
    print(model)


    # setup file structure for saving
    pathlib.Path(args.checkpoint_fp).mkdir(parents=True, exist_ok=True)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        # raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.id=checkpoint['id']
            args.name=checkpoint['name']
            CHECKPOINT_ID = checkpoint['name']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code

    # Set up crops and normalization depending on the dataset.

    # Cifar 10 crops and normalization.
    if args.dataid == "cifar10" or args.dataid =="svhn":
        _CIFAR_MEAN, _CIFAR_STD = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        normalize = transforms.Normalize(mean=_CIFAR_MEAN, std=_CIFAR_STD)
        if not args.randomcrop:
            random_resized_crop = transforms.RandomResizedCrop(28, scale=(args.rrc_param, 1.))
        else:
            # Use the crop they were using in Fast AutoAugment.
            random_resized_crop = transforms.RandomCrop(32, padding=4)

    # Use the imagenet parameters.
    elif args.dataid == "imagenet" or args.dataid == "logos":
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        random_resized_crop = transforms.RandomResizedCrop(224, scale=(0.2, 1.))



    if args.aug_plus and (args.faa_aug or
                          args.rand_aug or args.rand_aug_orig or not(args.custom_aug_name == None)):
        raise Exception("Cannot have multiple augs on command line")

    if args.aug_plus:
        # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
        augmentation = [
            random_resized_crop,
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([moco.loader.GaussianBlur([args.sigma/20, args.sigma])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]
    elif args.faa_aug:
        augmentation, _ = slm_utils.get_faa_transforms.get_faa_transforms_cifar_10(args.randomcrop, args.gauss)
        transformations = moco.loader.TwoCropsTransform(augmentation)
    elif args.rand_aug_orig:
        print("Using random aug original")
        augmentation = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            RandAugment(args.rand_aug_n, args.rand_aug_m),
            transforms.ToTensor(),
            normalize
        ]
    elif args.rand_aug:
        randaug_n = args.rand_aug_n
        if args.rand_aug_linear_m:
            print("Using random aug with linear m")
            randaug_m = args.rand_aug_m_min
        else:
            randaug_m = args.rand_aug_m
            print("Using random aug")
        if args.rand_aug_top_k > 0:
            randaug = TopRandAugment(randaug_n, randaug_m, args.rand_aug_top_k)
        else:
            randaug = RandAugment(randaug_n, randaug_m)
        augmentation = [
            random_resized_crop,
            transforms.RandomHorizontalFlip(),
            randaug,
            transforms.ToTensor(),
            normalize
        ]
    elif args.rand_resize_only and args.custom_aug_name == None:
        print("Using random resize only")
        augmentation = [
            random_resized_crop,
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]
    elif not args.custom_aug_name == None:
        augmentation, _ = slm_utils.get_faa_transforms.load_custom_transforms(name=args.custom_aug_name, randomcrop=args.randomcrop,
            aug_idx=args.single_aug_idx, dataid=args.dataid)

        print('using custom augs', augmentation)

        transformations = moco.loader.TwoCropsTransform(augmentation)
    else:
        # MoCo v1's aug: the same as InstDisc https://arxiv.org/abs/1805.01978
        print('using v1 augs')
        augmentation = [
            random_resized_crop,
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]


    if not args.faa_aug and args.custom_aug_name == None:
        print('using augmentation', augmentation)
        transformations = moco.loader.TwoCropsTransform(transforms.Compose(augmentation))


    print('xforms', transformations)

    if args.dataid == "imagenet" and not args.reduced_imgnet:
        train_dataset = datasets.ImageFolder(
            args.data,
            transformations)

    elif args.dataid == "logos" and not args.reduced_imgnet:
        train_dataset = data_loader.GetLoader(data_root=args.data,
        data_list='train_images_root.txt',
        transform=transformations)


    elif (args.dataid == "imagenet" or args.dataid == 'logos') and args.reduced_imgnet:
        # idx120 = [16, 23, 52, 57, 76, 93, 95, 96, 99, 121, 122, 128, 148, 172, 181, 189, 202, 210, 232, 238, 257, 258, 259, 277, 283, 289, 295, 304, 307, 318, 322, 331, 337, 338, 345, 350, 361, 375, 376, 381, 388, 399, 401, 408, 424, 431, 432, 440, 447, 462, 464, 472, 483, 497, 506, 512, 530, 541, 553, 554, 557, 564, 570, 584, 612, 614, 619, 626, 631, 632, 650, 657, 658, 660, 674, 675, 680, 682, 691, 695, 699, 711, 734, 736, 741, 754, 757, 764, 769, 770, 780, 781, 787, 797, 799, 811, 822, 829, 830, 835, 837, 842, 843, 845, 873, 883, 897, 900, 902, 905, 913, 920, 925, 937, 938, 940, 941, 944, 949, 959]

        if args.dataid == "imagenet":
            total_trainset = ImageNet(root=args.data, transform=transformations) # TODO for LINCLS, make this train and test xforms.

        else:
            total_trainset = data_loader.GetLoader(data_root=args.data,
                    data_list='train_images_root.txt',
                    transform=transformations)

        train_idx = np.arange(len(total_trainset))

        np.random.seed(1337) #fingers crossed.
        np.random.shuffle(train_idx)
        train_idx = train_idx[:50000]

        kfold = args.kfold

        print('KFOLD BEING USED', kfold)
        subset = np.arange(kfold*10000, (kfold+1)*10000)
        print('start', 'end', kfold*10000, (kfold+1)*10000)
        valid_idx = train_idx[subset]
        train_idx = np.delete(train_idx, subset)

        print('first val_idx', valid_idx[:10])

        train_dataset = total_trainset

        train_dataset = Subset(train_dataset, train_idx)
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetSampler(valid_idx)

        print(len(train_dataset))
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        print(len(train_dataset))


        print('first 10 train', train_idx[:10])
        print('first 10 valid', valid_idx[:10])
        print('len train', len(train_idx))
        print('len valid', len(valid_idx))

        for i in valid_idx:
            if i in train_idx:
                raise Exception("Valid idx in train idx: this is unexpected")
        print('train_sampler', train_sampler)



    elif args.dataid == "cifar10":
        train_dataset = torchvision.datasets.CIFAR10(args.data,
                                                     transform=transformations,
                                                     download=True)

    elif args.dataid == "svhn":
        train_dataset = torchvision.datasets.SVHN(args.data,
            transform=transformations,
            download=True)
    else:
        raise NotImplementedError("Support for the following dataset is not yet implemented: {}".format(args.dataid))

    if not args.kfold == None and not args.reduced_imgnet:
        torch.manual_seed(1337)
        print('before: K FOLD', args.kfold, len(train_dataset))
        lengths = [len(train_dataset)//5]*5
        print(lengths)
        lengths[-1] = int(lengths[-1] + (len(train_dataset)-np.sum(lengths)))
        print(lengths)
        folds = torch.utils.data.random_split(train_dataset, lengths )
        print(len(folds))
        folds.pop(args.kfold)
        print(len(folds))
        train_dataset = torch.utils.data.ConcatDataset(folds)
        print(len(train_dataset))

    else:
        print("NO KFOLD ARG", args.kfold, ' or ', args.reduced_imgnet)

    if args.distributed and not args.reduced_imgnet:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    elif not args.reduced_imgnet:
        train_sampler = None
    print('train sampler', train_sampler)

    torch.manual_seed(1337)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)


    print(len(train_loader))

    # CR: only the master will report to wandb for now
    if not args.multiprocessing_distributed or args.rank % ngpus_per_node == 0:
        wandb.init(project=args.wandbproj,
               name=CHECKPOINT_ID, id=args.id, resume=args.resume,
               config=args.__dict__, notes=args.notes)
        print(model)

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        if args.rand_aug_linear_m:
            mval = args.rand_aug_m_min + math.floor(float(epoch) / float(args.epochs) * (args.rand_aug_m_max - args.rand_aug_m_min + 1))
            print("Rand aug m: {}".format(mval))
            randaug.m = mval

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args, CHECKPOINT_ID)

        # save current epoch
        if not args.nosave_latest and (not args.multiprocessing_distributed or args.rank % ngpus_per_node == 0):
            print("saving latest epoch")
            cp_filename = "{}_latest.tar".format(CHECKPOINT_ID[:5])
            cp_fullpath = os.path.join(args.checkpoint_fp, cp_filename)
            torch.save({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'id': args.id,
                'name': CHECKPOINT_ID,
            }, cp_fullpath)
            print("saved latest epoch")


        if (epoch % args.checkpoint_interval == 0 or epoch == args.epochs-1) \
           and (not args.multiprocessing_distributed or
                (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0)):
            cp_filename = "{}_{:04d}.tar".format(CHECKPOINT_ID, epoch)
            cp_fullpath = os.path.join(args.checkpoint_fp, cp_filename)
            torch.save({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'id': args.id,
                'name': CHECKPOINT_ID,
            }, cp_fullpath)
            if args.upload_checkpoints:
                print("Uploading wandb checkpoint")
                wandb.save(cp_fullpath)
            if epoch == args.epochs - 1:
                print("Saving final results to wandb")
                wandb.save(cp_fullpath)

    print("Done - wrapping up")


def train(train_loader, model, criterion, optimizer, epoch, args, CHECKPOINT_ID):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    rot_losses = AverageMeter('Rot Loss', ':.4e')
    moco_losses = AverageMeter('MOCO Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        args.multiprocessing_distributed and args.rank % ngpus_per_node == 0,
        len(train_loader),
        [batch_time, data_time, losses, rot_losses, top1, top5],
        prefix="{} Epoch: [{}]".format(CHECKPOINT_ID[:5],epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, _) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images[0] = images[0].cuda(args.gpu, non_blocking=True)
            images[1] = images[1].cuda(args.gpu, non_blocking=True)

        if i == 0 and epoch % args.image_log_interval == 0 and args.multiprocessing_distributed and args.rank % ngpus_per_node == 0:
            eximg0 = wandb.Image(images[0][0].permute(1,2,0).cpu().numpy())
            eximg1 = wandb.Image(images[1][0].permute(1,2,0).cpu().numpy())
            wandb.log({"example comparison image": [eximg0, eximg1]})

        # compute output
        if args.rotnet:
            use_images = images[0]
            nimages = use_images.shape[0]
            n_rot_images = 4*use_images.shape[0]

            # rotate images all 4 ways at once
            rotated_images = torch.zeros([n_rot_images, use_images.shape[1], use_images.shape[2], use_images.shape[3]]).cuda()
            rot_classes = torch.zeros([n_rot_images]).long().cuda()

            rotated_images[:nimages] = use_images
            # rotate 90
            rotated_images[nimages:2*nimages] = use_images.flip(3).transpose(2,3)
            rot_classes[nimages:2*nimages] = 1
            # rotate 180
            rotated_images[2*nimages:3*nimages] = use_images.flip(3).flip(2)
            rot_classes[2*nimages:3*nimages] = 2
            # rotate 270
            rotated_images[3*nimages:4*nimages] = use_images.transpose(2,3).flip(3)
            rot_classes[3*nimages:4*nimages] = 3

            # if i == 0:
            #     eximg0 = wandb.Image(use_images[0].permute(1,2,0).cpu().numpy())
            #     eximg1 = wandb.Image(rotated_images[0].permute(1,2,0).cpu().numpy())
            #     wandb.log({"example rotated image": [eximg0, eximg1]})
            target = rot_classes
            output = model(head="rotnet", im_q=rotated_images)
            rot_loss = criterion(output, target)
            rot_losses.update(rot_loss.item(), images[0].size(0))

        if not args.nomoco:
            output, target = model(head="moco", im_q=images[0], im_k=images[1])
            moco_loss = criterion(output, target)
            moco_losses.update(moco_loss.item(), images[0].size(0))

        # acc1/acc5 are (K+1)-way contrast classifier accuracy
        # measure accuracy and record loss
        if args.rotnet:
            topk=(1,)
        else:
            topk=(1,5)
        accs = accuracy(output, target, topk=topk)
        top1.update(accs[0][0], output.size(0))
        if args.rotnet:
            top5.update(0, output.size(0))
        else:
            top5.update(accs[1][0], output.size(0))

        optimizer.zero_grad()
        if not args.nomoco and args.rotnet:
            rot_loss.backward(retain_graph=True)
            moco_loss.backward()
            loss = rot_loss.item() + moco_loss.item()
        elif not args.nomoco:
            moco_loss.backward()
            loss = moco_loss.item()
        elif args.rotnet:
            rot_loss.backward()
            loss = rot_loss.item()
        losses.update(loss)
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, main_node, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.main_node = main_node

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
        if self.main_node:
            wandb.log({meter.name: meter.avg for meter in self.meters})

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class TopRandAugment(RandAugment):
    def __init__(self, n, m, topk):
        super().__init__(n, m)
        # l = [
        #     (AutoContrast, 0, 1),
        #     (Equalize, 0, 1),
        #     (Invert, 0, 1),
        #     (Rotate, 0, 30),
        #     (Posterize, 0, 4),
        #     (Solarize, 0, 256),
        #     (SolarizeAdd, 0, 110),
        #     (Color, 0.1, 1.9),
        #     (Contrast, 0.1, 1.9),
        #     (Brightness, 0.1, 1.9),
        #     (Sharpness, 0.1, 1.9),
        #     (ShearX, 0., 0.3),
        #     (ShearY, 0., 0.3),
        #     (CutoutAbs, 0, 40),
        #     (TranslateXabs, 0., 100),
        #     (TranslateYabs, 0., 100),
        # ]

        # optimal orderings from rotnet results
        cifar_rotnet_ordering = [
            8, # contrast
            15, # trans y
            14, # trans x
            7, # color
            12, # shear y
            11, # shear x
            9, # brightness
            4, # posterize
            13, # cutout
            3, # rotate
            10, # sharpness
            5, # solarize
            2, # invert
            0, # autocontrast
            1, # equalize
        ]
        print("Using top {} cifar rotnet orderings in randaug".format(topk))
        self.augment_list = [self.augment_list[i] for i in cifar_rotnet_ordering][:topk]
        print(self.augment_list)

if __name__ == '__main__':
    main()
