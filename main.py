import argparse
import logging
import time
import math
import os
import random
from tqdm import tqdm
import numpy as np

import torch
from torch import nn, optim, cuda
from torch.cuda import amp, manual_seed_all
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torch.distributed import barrier, init_process_group, get_world_size

from engine import train, evaluate, train_finetune
from datasets.data import DATASET_GETTERS
from datasets.lemon_dataset import get_lemon_datasets
from models.models import build_wideresnet, ModelEMA
from utils.loss import create_loss_fn
from utils.utils import AverageMeter, model_load_state_dict, save_checkpoint

logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser()

    # Run mode option --------------------------------------------------------
    parser.add_argument('--evaluate',
                        action='store_true',
                        help='only evaluate model on validation set')
    parser.add_argument('--finetune',
                        action='store_true',
                        help='only finetune model on labeled dataset')

    # Savedir setting --------------------------------------------------------
    parser.add_argument('--name',
                        type=str,
                        required=True,
                        help='experiment name')
    parser.add_argument('--data-path',
                        default='./data',
                        type=str,
                        help='data path')
    parser.add_argument('--save-path',
                        default='./checkpoint',
                        type=str,
                        help='save path')

    # Dataset Setting --------------------------------------------------------
    parser.add_argument('--dataset',
                        default='cifar10',
                        type=str,
                        help='dataset name')
    parser.add_argument('--num-labeled',
                        type=int,
                        default=4000,
                        help='number of labeled data')
    parser.add_argument('--train-label-file-path',
                        default='/workspaces/data/'
                        'ObjectClassify/lemon/train_images.csv',
                        type=str,
                        help='data path')
    parser.add_argument('--num-classes',
                        # default=10,
                        default=4,
                        type=int,
                        help='number of classes')

    # Dataset params ---------------------------------------------------------
    parser.add_argument('--expand-labels',
                        action='store_true',
                        help='expand labels to fit eval steps')
    parser.add_argument('--top-range',
                        nargs='+',
                        default=[1, 4],
                        type=int,
                        help='use it like this. --top-range 1 5')

    # Augument params --------------------------------------------------------
    parser.add_argument('--img-size',
                        nargs='+',
                        default=[640, 640],
                        type=int,
                        help='img origin size [height, width]')
    parser.add_argument('--randaug',
                        nargs='+',
                        default=[2, 10],
                        type=int,
                        help='use it like this. --randaug 2 10')
    parser.add_argument('--resize',
                        default=32,
                        type=int,
                        help='resize image')
    parser.add_argument('--brightness',
                        nargs='+',
                        default=[1.8, 0.1],
                        type=float,
                        help='brightnes params [max_v, bias]')

    # Train setting ----------------------------------------------------------
    parser.add_argument('--resume',
                        default='',
                        type=str,
                        help='path to checkpoint')
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
    parser.add_argument('--batch-size',
                        default=64,
                        type=int,
                        help='train batch size')
    parser.add_argument('--mu',
                        default=7,
                        type=int,
                        help='coefficient of unlabeled batch size')
    parser.add_argument('--world-size',
                        default=-1,
                        type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--local_rank',
                        type=int,
                        default=-1,
                        help='For distributed training: local_rank')
    # Train params -----------------------------------------------------------
    parser.add_argument('--seed',
                        default=42,
                        type=int,
                        help='seed for initializing training')
    parser.add_argument('--amp',
                        action='store_true',
                        help='use 16-bit (mixed) precision')

    # Optim params -----------------------------------------------------------
    parser.add_argument('--weight-decay',
                        default=0,
                        type=float,
                        help='train weight decay')
    parser.add_argument('--lr',
                        default=0.01,
                        type=float,
                        help='train learning late')
    parser.add_argument('--momentum',
                        default=0.9,
                        type=float,
                        help='SGD Momentum')
    parser.add_argument('--nesterov',
                        action='store_true',
                        help='use nesterov')
    parser.add_argument('--warmup-steps',
                        default=0,
                        type=int,
                        help='warmup steps')

    # Model params -----------------------------------------------------------
    parser.add_argument('--dense-dropout',
                        default=0,
                        type=float,
                        help='dropout on last dense layer')
    parser.add_argument('--ema',
                        default=0,
                        type=float,
                        help='EMA decay rate')

    # Finetune params --------------------------------------------------------
    parser.add_argument('--finetune-epochs',
                        default=10,
                        type=int,
                        help='finetune epochs')
    parser.add_argument('--finetune-batch-size',
                        default=512,
                        type=int,
                        help='finetune batch size')
    parser.add_argument('--finetune-lr',
                        default=1e-5,
                        type=float,
                        help='finetune learning late')
    parser.add_argument('--finetune-weight-decay',
                        default=0,
                        type=float,
                        help='finetune weight decay')
    parser.add_argument('--finetune-momentum',
                        default=0,
                        type=float,
                        help='finetune SGD Momentum')

    # Loss params ------------------------------------------------------------
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
    parser.add_argument('--student-wait-steps',
                        default=0,
                        type=int,
                        help='warmup steps')
    parser.add_argument('--grad-clip',
                        default=0.,
                        type=float,
                        help='gradient norm clipping')
    parser.add_argument('--label-smoothing',
                        default=0,
                        type=float,
                        help='label smoothing alpha')

    return parser.parse_args()


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    manual_seed_all(args.seed)


def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_wait_steps=0,
                                    num_cycles=0.5,
                                    last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_wait_steps:
            return 0.0

        if current_step < num_warmup_steps + num_wait_steps:
            return float(current_step) / \
                float(max(1, num_warmup_steps + num_wait_steps))

        t1 = float(current_step - num_warmup_steps - num_wait_steps)
        t2 = float(
            max(1, num_training_steps - num_warmup_steps - num_wait_steps))
        progress = t1 / t2
        val = 0.5 * (1.0 + math.cos(
            math.pi * float(num_cycles) * 2.0 * progress))
        return max(0.0, val)

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_lr(optimizer):
    return optimizer.param_groups[0]['lr']


def train_loop(
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
        s_scaler):
    logger.info("***** Running Training *****")
    logger.info(f"   Task = {args.dataset}@{args.num_labeled}")
    logger.info(f"   Total steps = {args.total_steps}")

    labeled_epoch = 0
    unlabeled_epoch = 0
    date_time = AverageMeter()

    if args.world_size > 1:
        labeled_loader.sampler.set_epoch(labeled_epoch)
        unlabeled_loader.sampler.set_epoch(unlabeled_epoch)

    moving_dot_product = torch.empty(1).to(args.device)
    limit = 3.0**(0.5)  # 3 = 6 / (f_in + f_out)
    nn.init.uniform_(moving_dot_product, -limit, limit)

    labeled_iter = iter(labeled_loader)
    unlabeled_iter = iter(unlabeled_loader)

    for eval_cnt in range(args.start_step,
                          (args.total_steps // args.eval_step)):
        batch_time = AverageMeter()
        s_losses = AverageMeter()
        t_losses = AverageMeter()
        t_losses_l = AverageMeter()
        t_losses_u = AverageMeter()
        t_losses_mpl = AverageMeter()
        mean_mask = AverageMeter()

        pbar = tqdm(range(args.eval_step),
                    disable=args.local_rank not in [-1, 0])
        # Train
        teacher_model.train()
        student_model.train()
        for step in range(args.eval_step):
            end = time.time()
            # Data Loader
            try:
                images_l, targets = labeled_iter.next()
            except BaseException:
                if args.world_size > 1:
                    labeled_epoch += 1
                    labeled_loader.sampler.set_epoch(labeled_epoch)
                labeled_iter = iter(labeled_loader)
                images_l, targets = labeled_iter.next()

            try:
                (images_ori, images_aug), _ = unlabeled_iter.next()
            except BaseException:
                if args.world_size > 1:
                    unlabeled_epoch += 1
                    unlabeled_loader.sampler.set_epoch(unlabeled_epoch)
                unlabeled_iter = iter(unlabeled_loader)
                (images_ori, images_aug), _ = unlabeled_iter.next()

            s_loss, t_loss, t_loss_l, t_loss_u, t_loss_mpl, mask, t_optimizer, s_optimizer = train(
                args, step,
                images_l, images_ori, images_aug,
                targets,
                teacher_model, student_model,
                avg_student_model,
                criterion,
                t_scaler, s_scaler,
                t_optimizer, s_optimizer,
                t_scheduler, s_scheduler,
                moving_dot_product)

            s_losses.update(s_loss.item())
            t_losses.update(t_loss.item())
            t_losses_l.update(t_loss_l.item())
            t_losses_u.update(t_loss_u.item())
            t_losses_mpl.update(t_loss_mpl.item())
            mean_mask.update(mask.mean().item())
            batch_time.update(time.time() - end)
            date_time.update(time.time() - end)

            pbar.set_description(
                "Train Iter: "
                f"{eval_cnt*args.eval_step+step+1:3}/{args.total_steps:3}.")
            pbar.set_postfix(
                LR=f"{get_lr(s_optimizer):.4f}",
                Date=f"{date_time.sum:.2f}s",
                Batch=f"{batch_time.avg:.2f}s",
                S_Loss=f"{s_losses.avg:.4f}",
                T_Loss=f"{t_losses.avg:.4f}",
                Mask=f"{mean_mask.avg:.4f}")
            pbar.update()
        pbar.close()

        if args.local_rank in [-1, 0]:
            args.writer.add_scalar(
                "lr",
                get_lr(s_optimizer),
                (eval_cnt + 1) * args.eval_step)

        args.num_eval = eval_cnt
        # if ((eval_cnt + 1) * args.eval_step) % args.eval_step == 0:
        if args.local_rank in [-1, 0]:
            args.writer.add_scalar(
                "train/1.student_loss",
                s_losses.avg,
                args.num_eval)
            args.writer.add_scalar(
                "train/2.teacher_loss",
                t_losses.avg,
                args.num_eval)
            args.writer.add_scalar(
                "train/3.teacher_loss_labeled",
                t_losses_l.avg,
                args.num_eval)
            args.writer.add_scalar(
                "train/4.teacher_loss_unlabeled",
                t_losses_u.avg,
                args.num_eval)
            args.writer.add_scalar(
                "train/5.teacher_loss_mpl",
                t_losses_mpl.avg,
                args.num_eval)
            args.writer.add_scalar(
                "train/6.mask",
                mean_mask.avg,
                args.num_eval)
            # Test
            test_model = avg_student_model if avg_student_model is not None else student_model
            test_loss, top1, top5 = evaluate(
                args, test_loader, test_model, criterion, args.top_range)

            args.writer.add_scalar("test/loss", test_loss, args.num_eval)
            args.writer.add_scalar("test/acc@1", top1, args.num_eval)
            args.writer.add_scalar(
                f"test/acc@{args.top_range}", top5, args.num_eval)

            is_best = top1 > args.best_top1
            if is_best:
                args.best_top1 = top1
                args.best_top5 = top5

            logger.info(f"top-1 acc: {top1:.2f}")
            logger.info(f"Best top-1 acc: {args.best_top1:.2f}")

            save_checkpoint(args, {
                'step': (eval_cnt + 1) * args.eval_step,
                'teacher_state_dict': teacher_model.state_dict(),
                'student_state_dict': student_model.state_dict(),
                'avg_state_dict': avg_student_model.state_dict() if avg_student_model is not None else None,
                'best_top1': args.best_top1,
                'best_top5': args.best_top5,
                'teacher_optimizer': t_optimizer.state_dict(),
                'student_optimizer': s_optimizer.state_dict(),
                'teacher_scheduler': t_scheduler.state_dict(),
                'student_scheduler': s_scheduler.state_dict(),
                'teacher_scaler': t_scaler.state_dict(),
                'student_scaler': s_scaler.state_dict(),
            }, is_best)

    # finetune
    del t_scaler, t_scheduler, t_optimizer, teacher_model, unlabeled_loader
    del s_scaler, s_scheduler, s_optimizer
    ckpt_name = f'{args.save_path}/{args.name}_best.pth.tar'
    loc = f'cuda:{args.gpu}'
    checkpoint = torch.load(ckpt_name, map_location=loc)
    logger.info(f"=> loading checkpoint '{ckpt_name}'")
    if checkpoint['avg_state_dict'] is not None:
        model_load_state_dict(student_model, checkpoint['avg_state_dict'])
    else:
        model_load_state_dict(student_model, checkpoint['student_state_dict'])

    finetune(args, labeled_loader, test_loader, student_model, criterion)

    cuda.empty_cache()


def finetune(args, train_loader, test_loader, model, criterion) -> None:
    train_sampler = RandomSampler if args.local_rank == -1 else DistributedSampler
    labeled_loader = DataLoader(
        train_loader.dataset,
        sampler=train_sampler(train_loader.dataset),
        batch_size=args.finetune_batch_size,
        num_workers=args.workers,
        pin_memory=True)
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.finetune_lr,
        momentum=args.finetune_momentum,
        weight_decay=args.finetune_weight_decay)
    scaler = amp.GradScaler(enabled=args.amp)

    logger.info("***** Running Finetuning *****")
    logger.info(
        f"   Finetuning steps = {len(labeled_loader)*args.finetune_epochs}")

    for epoch in range(args.finetune_epochs):
        if args.world_size > 1:
            labeled_loader.sampler.set_epoch(epoch + 624)

        step, losses = train_finetune(
            args,
            epoch,
            labeled_loader,
            model,
            criterion,
            scaler,
            optimizer)

        if args.local_rank in [-1, 0]:
            args.writer.add_scalar("finetune/train_loss", losses.avg, epoch)

            # Test
            test_loss, top1, top5 = evaluate(
                args, test_loader, model, criterion)
            # Add log
            args.writer.add_scalar("finetune/test_loss", test_loss, epoch)
            args.writer.add_scalar("finetune/acc@1", top1, epoch)
            args.writer.add_scalar("finetune/acc@5", top5, epoch)

            # update best
            is_best = top1 > args.best_top1
            if is_best:
                args.best_top1 = top1
                args.best_top5 = top5

            logger.info(f"top-1 acc: {top1:.2f}")
            logger.info(f"Best top-1 acc: {args.best_top1:.2f}")

            save_checkpoint(
                args,
                {
                    'step': step + 1,
                    'best_top1': args.best_top1,
                    'best_top5': args.best_top5,
                    'student_state_dict': model.state_dict(),
                    'avg_state_dict': None,
                    'student_optimizer': optimizer.state_dict(),
                },
                is_best,
                finetune=True)


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

    no_decay = ['bn']
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

    t_optimizer = optim.SGD(
        teacher_parameters,
        lr=args.lr,
        momentum=args.momentum,
        nesterov=args.nesterov)
    s_optimizer = optim.SGD(
        student_parameters,
        lr=args.lr,
        momentum=args.momentum,
        nesterov=args.nesterov)

    t_scheduler = get_cosine_schedule_with_warmup(
        t_optimizer,
        args.warmup_steps,
        args.total_steps)
    s_scheduler = get_cosine_schedule_with_warmup(
        s_optimizer,
        args.warmup_steps,
        args.total_steps,
        args.student_wait_steps)

    t_scaler = amp.GradScaler(enabled=args.amp)
    s_scaler = amp.GradScaler(enabled=args.amp)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info(f"=> loading checkpoint '{args.resume}'")
            loc = f'cuda:{args.gpu}'
            checkpoint = torch.load(args.resume, map_location=loc)
            args.best_top1 = checkpoint['best_top1'].to(torch.device('cpu'))
            args.best_top5 = checkpoint['best_top5'].to(torch.device('cpu'))

            if not (args.evaluate or args.finetune):
                args.start_step = checkpoint['step']
                t_optimizer.load_state_dict(checkpoint['teacher_optimizer'])
                s_optimizer.load_state_dict(checkpoint['student_optimizer'])
                t_scheduler.load_state_dict(checkpoint['teacher_scheduler'])
                s_scheduler.load_state_dict(checkpoint['student_scheduler'])
                t_scaler.load_state_dict(checkpoint['teacher_scaler'])
                s_scaler.load_state_dict(checkpoint['student_scaler'])
                model_load_state_dict(
                    teacher_model, checkpoint['teacher_state_dict'])
                if avg_student_model is not None:
                    model_load_state_dict(
                        avg_student_model, checkpoint['avg_state_dict'])

            else:
                if checkpoint['avg_state_dict'] is not None:
                    model_load_state_dict(
                        student_model, checkpoint['avg_state_dict'])
                else:
                    model_load_state_dict(
                        student_model, checkpoint['student_state_dict'])

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

    if args.finetune:
        del t_scaler, t_scheduler, t_optimizer, teacher_model, unlabeled_loader, labeled_loader
        del s_scaler, s_scheduler, s_optimizer
        finetune(args, labeled_dataset, test_loader, student_model, criterion)
        return

    if args.evaluate:
        del t_scaler, t_scheduler, t_optimizer, teacher_model, unlabeled_loader, labeled_loader
        del s_scaler, s_scheduler, s_optimizer
        evaluate(args, test_loader, student_model, criterion)
        return

    teacher_model.zero_grad()
    student_model.zero_grad()
    train_loop(
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
