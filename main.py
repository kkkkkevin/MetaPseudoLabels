import argparse
import logging
import math
import os
import random
import wandb

import numpy as np
import torch

from torch import nn
from torch.cuda import amp, manual_seed_all
from torch import optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torch.distributed import barrier, init_process_group, get_world_size

from engine import _train_loop, evaluate  # , train_loop
from datasets.data import DATASET_GETTERS
from datasets.lemon_dataset import get_lemon_datasets
from models.models import build_wideresnet, ModelEMA
from utils.utils import create_loss_fn, module_load_state_dict

logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name',
                        type=str,
                        required=True,
                        help='experiment name')
    parser.add_argument('--data-path',
                        default='./data',
                        type=str,
                        help='data path')
    parser.add_argument('--train-label-file-path',
                        default='/workspaces/data/'
                                'ObjectClassify/lemon/train_images.csv',
                        type=str,
                        help='data path')
    parser.add_argument('--save-path',
                        default='./checkpoint',
                        type=str,
                        help='save path')
    parser.add_argument('--dataset',
                        default='cifar10',
                        type=str,
                        # choices=['cifar10', 'cifar100'],
                        help='dataset name')
    parser.add_argument('--num-labeled',
                        type=int,
                        default=4000,
                        help='number of labeled data')
    parser.add_argument("--expand-labels",
                        action="store_true",
                        help="expand labels to fit eval steps")
    parser.add_argument('--total-steps',
                        default=300000,
                        type=int,
                        help='number of total steps to run')
    parser.add_argument('--eval-step',
                        default=1000,
                        type=int,
                        help='number of eval steps to run')
    parser.add_argument('--start-step',
                        default=0,
                        type=int,
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--workers',
                        default=8,
                        type=int,
                        help='number of workers')
    parser.add_argument('--num-classes',
                        # default=10,
                        default=4,
                        type=int,
                        help='number of classes')
    parser.add_argument('--dense-dropout',
                        default=0,
                        type=float,
                        help='dropout on last dense layer')
    parser.add_argument('--resize',
                        default=32,
                        type=int,
                        help='resize image')
    parser.add_argument('--batch-size',
                        default=64,
                        type=int,
                        help='train batch size')
    parser.add_argument('--lr',
                        default=0.01,
                        type=float,
                        help='learning late')
    parser.add_argument('--momentum',
                        default=0.9,
                        type=float,
                        help='SGD Momentum')
    parser.add_argument('--nesterov',
                        action='store_true',
                        help='use nesterov')
    parser.add_argument('--weight-decay',
                        default=0,
                        type=float)
    parser.add_argument('--ema',
                        default=0,
                        type=float,
                        help='EMA decay rate')
    parser.add_argument('--warmup-steps',
                        default=0,
                        type=int,
                        help='warmup steps')
    parser.add_argument('--grad-clip',
                        default=0.,
                        type=float,
                        help='gradient norm clipping')
    parser.add_argument('--resume',
                        default='',
                        type=str,
                        help='path to checkpoint')
    parser.add_argument('--evaluate',
                        action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--top-range',
                        nargs="+",
                        type=int,
                        help="use it like this. --top-range 1 5")
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training')
    parser.add_argument('--label-smoothing',
                        default=0,
                        type=float,
                        help='label smoothing alpha')
    parser.add_argument('--mu',
                        default=7,
                        type=int,
                        help='coefficient of unlabeled batch size')
    parser.add_argument('--threshold',
                        default=0.95,
                        type=float,
                        help='pseudo label threshold')
    parser.add_argument('--temperature',
                        default=1,
                        type=float,
                        help='pseudo label temperature')
    parser.add_argument('--lambda-u',
                        default=1,
                        type=float,
                        help='coefficient of unlabeled loss')
    parser.add_argument('--uda-steps',
                        default=1,
                        type=float,
                        help='warmup steps of lambda-u')
    parser.add_argument("--randaug",
                        nargs="+",
                        type=int,
                        help="use it like this. --randaug 2 10")
    parser.add_argument("--amp",
                        action="store_true",
                        help="use 16-bit (mixed) precision")
    parser.add_argument('--world-size',
                        default=-1,
                        type=int,
                        help='number of nodes for distributed training')
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="For distributed training: local_rank")
    return parser.parse_args()


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    manual_seed_all(args.seed)


def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=0.5,
                                    last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0,
                   0.5 * (1.0 + math.cos(
                       math.pi *
                       float(num_cycles) * 2.0 * progress))
                   )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def main(args):

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARNING)

    logger.warning(
        f"Process rank: {args.local_rank}, "
        f"device: {args.device}, "
        f"distributed training: {bool(args.local_rank != -1)}, "
        f"16-bits training: {args.amp}")

    logger.info(dict(args._get_kwargs()))

    if args.local_rank in [-1, 0]:
        args.writer = SummaryWriter(f"results/{args.name}")

    if args.seed is not None:
        set_seed(args)

    if args.local_rank not in [-1, 0]:
        barrier()

    # labeled_dataset, unlabeled_dataset, test_dataset = DATASET_GETTERS[args.dataset](
    #    args)
    labeled_dataset, unlabeled_dataset, test_dataset = get_lemon_datasets(
        args)
    if args.local_rank == 0:
        barrier()

    train_sampler = RandomSampler if args.local_rank == -1 else DistributedSampler
    labeled_loader = DataLoader(
        labeled_dataset,
        sampler=train_sampler(labeled_dataset),
        batch_size=args.batch_size,
        num_workers=args.workers,
        drop_last=True)

    unlabeled_loader = DataLoader(
        unlabeled_dataset,
        sampler=train_sampler(unlabeled_dataset),
        batch_size=args.batch_size * args.mu,
        num_workers=args.workers,
        drop_last=True)

    test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=args.batch_size,
        num_workers=args.workers)

    if args.local_rank not in [-1, 0]:
        barrier()

    teacher_model = build_wideresnet(args, depth=28, widen_factor=2)
    student_model = build_wideresnet(args, depth=28, widen_factor=2)

    if args.local_rank == 0:
        barrier()

    teacher_model.to(args.device)
    student_model.to(args.device)
    avg_student_model = None
    if args.ema > 0:
        avg_student_model = ModelEMA(student_model, args.ema)

    criterion = create_loss_fn(args)

    no_decay = ['bn', 'bias']
    teacher_parameters = [
        {
            'params': [p for n, p in teacher_model.named_parameters() if not any(
                nd in n for nd in no_decay)],
            'weight_decay': args.weight_decay
        },
        {
            'params': [p for n, p in teacher_model.named_parameters() if any(
                nd in n for nd in no_decay)],
            'weight_decay': 0.0}
    ]
    student_parameters = [
        {
            'params': [p for n, p in student_model.named_parameters() if not any(
                nd in n for nd in no_decay)],
            'weight_decay': args.weight_decay},
        {
            'params': [p for n, p in student_model.named_parameters() if any(
                nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]

    t_optimizer = optim.SGD(teacher_parameters,
                            lr=args.lr,
                            momentum=args.momentum,
                            nesterov=args.nesterov)
    s_optimizer = optim.SGD(student_parameters,
                            lr=args.lr,
                            momentum=args.momentum,
                            nesterov=args.nesterov)

    t_scheduler = get_cosine_schedule_with_warmup(t_optimizer,
                                                  args.warmup_steps,
                                                  args.total_steps)
    s_scheduler = get_cosine_schedule_with_warmup(s_optimizer,
                                                  0,
                                                  args.total_steps)

    t_scaler = amp.GradScaler(enabled=args.amp)
    s_scaler = amp.GradScaler(enabled=args.amp)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info(f"=> loading checkpoint '{args.resume}'")
            loc = f'cuda:{args.gpu}'
            checkpoint = torch.load(args.resume, map_location=loc)
            args.start_step = checkpoint['step']

            args.best_top1 = checkpoint['best_top1']
            args.best_top5 = checkpoint['best_top5']

            t_optimizer.load_state_dict(checkpoint['teacher_optimizer'])
            s_optimizer.load_state_dict(checkpoint['student_optimizer'])

            t_scheduler.load_state_dict(checkpoint['teacher_scheduler'])
            s_scheduler.load_state_dict(checkpoint['student_scheduler'])

            t_scaler.load_state_dict(checkpoint['teacher_scaler'])
            s_scaler.load_state_dict(checkpoint['student_scaler'])

            try:
                teacher_model.load_state_dict(checkpoint['teacher_state_dict'])
            except BaseException:
                module_load_state_dict(
                    teacher_model, checkpoint['teacher_state_dict'])
            try:
                student_model.load_state_dict(checkpoint['student_state_dict'])
            except BaseException:
                module_load_state_dict(
                    student_model, checkpoint['student_state_dict'])
            if avg_student_model is not None:
                try:
                    avg_student_model.load_state_dict(
                        checkpoint['avg_state_dict'])
                except BaseException:
                    module_load_state_dict(
                        avg_student_model, checkpoint['avg_state_dict'])
            logger.info(
                f"=> loaded checkpoint '{args.resume}' (step {checkpoint['step']})")
        else:
            logger.info(f"=> no checkpoint found at '{args.resume}'")

    if args.local_rank != -1:
        teacher_model = nn.parallel.DistributedDataParallel(
            teacher_model, device_ids=[args.local_rank],
            output_device=args.local_rank, find_unused_parameters=True)
        student_model = nn.parallel.DistributedDataParallel(
            student_model, device_ids=[args.local_rank],
            output_device=args.local_rank, find_unused_parameters=True)

    if args.evaluate:
        if avg_student_model is not None:
            student_model = avg_student_model
        evaluate(args, test_loader, student_model, criterion)
        return

    logger.info("***** Running Training *****")
    logger.info(f"   Task = {args.dataset}@{args.num_labeled}")
    logger.info(f"   Total steps = {args.total_steps}")

    teacher_model.zero_grad()
    student_model.zero_grad()
    _train_loop(
        args,
        labeled_loader,
        unlabeled_loader,
        test_loader,
        teacher_model,
        student_model,
        avg_student_model,
        criterion,
        t_optimizer,
        s_optimizer,
        t_scheduler,
        s_scheduler,
        t_scaler,
        s_scaler)
    return


if __name__ == '__main__':
    args = get_args()
    args.best_top1 = 0.
    args.best_top5 = 0.

    if args.local_rank != -1:
        args.gpu = args.local_rank
        init_process_group(backend='nccl')
        args.world_size = get_world_size()
    else:
        args.gpu = 0
        args.world_size = 1

    args.device = torch.device('cuda', args.gpu)

    main(args)
