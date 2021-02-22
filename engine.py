import logging
import time
from tqdm import tqdm

import torch
from torch import nn
from torch.cuda import amp
from torch.nn import functional as F

from utils.utils import AverageMeter, reduce_tensor, accuracy

logger = logging.getLogger(__name__)


def train(
        args, step,
        images_l, images_uw, images_us,
        targets,
        teacher_model, student_model,
        avg_student_model,
        criterion,
        t_scaler, s_scaler,
        t_optimizer, s_optimizer,
        t_scheduler, s_scheduler,
        moving_dot_product):

    images_l = images_l.to(args.device)
    images_uw = images_uw.to(args.device)
    images_us = images_us.to(args.device)
    targets = targets.to(args.device)

    # Updating the Student
    with amp.autocast(enabled=args.amp):
        batch_size = images_l.shape[0]

        # Forward
        t_images = torch.cat((images_l, images_uw, images_us))
        t_logits = teacher_model(t_images)

        t_logits_l = t_logits[:batch_size]
        t_logits_uw, t_logits_us = t_logits[batch_size:].chunk(2)
        del t_logits

        t_loss_l = criterion(t_logits_l, targets)

        soft_pseudo_label = torch.softmax(
            t_logits_uw.detach() / args.temperature, dim=-1)
        max_probs, hard_pseudo_label = torch.max(soft_pseudo_label, dim=-1)
        mask = max_probs.ge(args.threshold).float()
        t_loss_u = torch.mean(
            -(soft_pseudo_label * torch.log_softmax(
                t_logits_us, dim=-1)).sum(dim=-1) * mask)
        weight_u = args.lambda_u * min(1., (step + 1) / args.uda_steps)
        t_loss_uda = t_loss_l + weight_u * t_loss_u

        # Forward
        s_images = torch.cat((images_l, images_us))
        s_logits = student_model(s_images)

        s_logits_l = s_logits[:batch_size]
        s_logits_us = s_logits[batch_size:]
        del s_logits

        s_loss_l_old = F.cross_entropy(s_logits_l.detach(), targets)
        s_loss = criterion(s_logits_us, hard_pseudo_label)

    # Backward
    s_scaler.scale(s_loss).backward()

    # Optimize
    if args.grad_clip > 0:
        s_scaler.unscale_(s_optimizer)
        nn.utils.clip_grad_norm_(
            student_model.parameters(),
            args.grad_clip)
    s_scaler.step(s_optimizer)
    s_scaler.update()

    # Scheduler
    s_scheduler.step()

    if args.ema > 0:
        avg_student_model.update_parameters(student_model)

    # Updating the Teacher
    with amp.autocast(enabled=args.amp):
        # Forward
        with torch.no_grad():
            s_logits_l = student_model(images_l)

        s_loss_l_new = F.cross_entropy(s_logits_l.detach(), targets)

        dot_product = s_loss_l_new - s_loss_l_old
        moving_dot_product = moving_dot_product * 0.99 + dot_product * 0.01
        dot_product = dot_product - moving_dot_product
        _, hard_pseudo_label = torch.max(t_logits_us.detach(), dim=-1)
        t_loss_mpl = dot_product * F.cross_entropy(
            t_logits_us,
            hard_pseudo_label)
        t_loss = t_loss_uda + t_loss_mpl

    # Backward
    t_scaler.scale(t_loss).backward()

    # Optimize
    if args.grad_clip > 0:
        t_scaler.unscale_(t_optimizer)
        nn.utils.clip_grad_norm_(
            teacher_model.parameters(),
            args.grad_clip)
    t_scaler.step(t_optimizer)
    t_scaler.update()

    # Scheduler
    t_scheduler.step()

    teacher_model.zero_grad()
    student_model.zero_grad()

    if args.world_size > 1:
        s_loss = reduce_tensor(s_loss.detach(), args.world_size)
        t_loss = reduce_tensor(t_loss.detach(), args.world_size)
        t_loss_l = reduce_tensor(t_loss_l.detach(), args.world_size)
        t_loss_u = reduce_tensor(t_loss_u.detach(), args.world_size)
        t_loss_mpl = reduce_tensor(t_loss_mpl.detach(), args.world_size)
        mask = reduce_tensor(mask, args.world_size)

    return s_loss, t_loss, t_loss_l, t_loss_u, t_loss_mpl, mask, t_optimizer, s_optimizer


def evaluate(args, test_loader, model, criterion, top_range=(1, 5)):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top_high = AverageMeter()
    top_low = AverageMeter()
    model.eval()

    test_iter = tqdm(test_loader, disable=args.local_rank not in [-1, 0])
    with torch.no_grad():
        end = time.time()

        for step, (images, targets) in enumerate(test_iter):
            data_time.update(time.time() - end)
            batch_size = targets.shape[0]
            images = images.to(args.device)
            targets = targets.to(args.device)

            with amp.autocast(enabled=args.amp):
                outputs = model(images)
                loss = criterion(outputs, targets)

            acc_high, acc_low = accuracy(outputs, targets, top_range)
            losses.update(loss.item(), batch_size)
            top_high.update(acc_high[0], batch_size)
            top_low.update(acc_low[0], batch_size)

            batch_time.update(time.time() - end)
            end = time.time()

            test_iter.set_description(
                f"Test Iter: {step+1:3}/{len(test_loader):3}. "
                f"Data: {data_time.avg:.2f}s. "
                f"Batch: {batch_time.avg:.2f}s. "
                f"Loss: {losses.avg:.4f}. "
                f"top{top_range[0]}: {top_high.avg:.2f}. "
                f"top{top_range[1]}: {top_low.avg:.2f}. ")

        test_iter.close()

        return losses.avg, top_high.avg, top_low.avg


def train_finetune(
        args,
        epoch,
        labeled_loader,
        model,
        criterion,
        scaler,
        optimizer):
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    labeled_iter = tqdm(
        labeled_loader, disable=args.local_rank not in [-1, 0])

    step = 0
    end = time.time()
    for step, (images, targets) in enumerate(labeled_iter):
        data_time.update(time.time() - end)
        batch_size = targets.shape[0]
        images = images.to(args.device)
        targets = targets.to(args.device)
        with amp.autocast(enabled=args.amp):
            model.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if args.world_size > 1:
            loss = reduce_tensor(loss.detach(), args.world_size)
        losses.update(loss.item(), batch_size)
        batch_time.update(time.time() - end)
        labeled_iter.set_description(
            f"Finetune Epoch: {epoch+1:2}/{args.finetune_epochs:2}. Data: {data_time.avg:.2f}s. "
            f"Batch: {batch_time.avg:.2f}s. Loss: {losses.avg:.4f}. ")
    labeled_iter.close()

    return step, losses
