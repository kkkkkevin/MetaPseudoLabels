import time
import logging
from tqdm import tqdm

import torch
from torch import nn, cuda
from torch.cuda import amp
from torch.nn import functional as F

from utils.utils import AverageMeter, reduce_tensor, accuracy, save_checkpoint

logger = logging.getLogger(__name__)


def get_lr(optimizer):
    return optimizer.param_groups[0]['lr']


def train(args, step,
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
            -(soft_pseudo_label * torch.log_softmax(t_logits_us,
                                                    dim=-1)).sum(dim=-1) * mask)
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
        nn.utils.clip_grad_norm_(student_model.parameters(),
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
        t_loss_mpl = dot_product * F.cross_entropy(t_logits_us.detach(),
                                                   hard_pseudo_label)
        t_loss = t_loss_uda + t_loss_mpl

    # Backward
    t_scaler.scale(t_loss).backward()

    # Optimize
    if args.grad_clip > 0:
        t_scaler.unscale_(t_optimizer)
        nn.utils.clip_grad_norm_(teacher_model.parameters(),
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


def _train_loop(
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
                (images_uw, images_us), _ = unlabeled_iter.next()
            except BaseException:
                if args.world_size > 1:
                    unlabeled_epoch += 1
                    unlabeled_loader.sampler.set_epoch(unlabeled_epoch)
                unlabeled_iter = iter(unlabeled_loader)
                (images_uw, images_us), _ = unlabeled_iter.next()

            s_loss, t_loss, t_loss_l, t_loss_u, t_loss_mpl, mask, t_optimizer, s_optimizer = train(
                args, step,
                images_l, images_uw, images_us,
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
                f"Train Iter: {eval_cnt*args.eval_step+step+1:3}/{args.total_steps:3}. "
                f"LR: {get_lr(s_optimizer):.4f}. "
                f"Date: {date_time.sum:.2f}s. "
                f"Batch: {batch_time.avg:.2f}s. "
                f"S_Loss: {s_losses.avg:.4f}. "
                f"T_Loss: {t_losses.avg:.4f}. "
                f"Mask: {mean_mask.avg:.4f}. ")
            pbar.update()
        pbar.close()

        if args.local_rank in [-1, 0]:
            args.writer.add_scalar("lr",
                                   get_lr(s_optimizer),
                                   (eval_cnt + 1) * args.eval_step)

        args.num_eval = eval_cnt
        # if ((eval_cnt + 1) * args.eval_step) % args.eval_step == 0:
        if args.local_rank in [-1, 0]:
            args.writer.add_scalar("train/1.s_loss",
                                   s_losses.avg,
                                   args.num_eval)
            args.writer.add_scalar("train/2.t_loss",
                                   t_losses.avg,
                                   args.num_eval)
            args.writer.add_scalar("train/3.t_labeled",
                                   t_losses_l.avg,
                                   args.num_eval)
            args.writer.add_scalar("train/4.t_unlabeled",
                                   t_losses_u.avg,
                                   args.num_eval)
            args.writer.add_scalar("train/5.t_mpl",
                                   t_losses_mpl.avg,
                                   args.num_eval)
            args.writer.add_scalar("train/6.mask",
                                   mean_mask.avg,
                                   args.num_eval)
            # Test
            test_model = avg_student_model if avg_student_model is not None else student_model
            _, top1, top5 = evaluate(args, test_loader, test_model, criterion)

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

    cuda.empty_cache()


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

    if args.world_size > 1:
        labeled_epoch = 0
        unlabeled_epoch = 0
        labeled_loader.sampler.set_epoch(labeled_epoch)
        unlabeled_loader.sampler.set_epoch(unlabeled_epoch)

    labeled_iter = iter(labeled_loader)
    unlabeled_iter = iter(unlabeled_loader)

    moving_dot_product = torch.empty(1).to(args.device)
    limit = 3.0**(0.5)  # 3 = 6 / (f_in + f_out)
    nn.init.uniform_(moving_dot_product, -limit, limit)

    for step in range(args.start_step, args.total_steps):
        if step % args.eval_step == 0:
            pbar = tqdm(range(args.eval_step),
                        disable=args.local_rank not in [-1, 0])
            batch_time = AverageMeter()
            data_time = AverageMeter()
            s_losses = AverageMeter()
            t_losses = AverageMeter()
            t_losses_l = AverageMeter()
            t_losses_u = AverageMeter()
            t_losses_mpl = AverageMeter()
            mean_mask = AverageMeter()

        teacher_model.train()
        student_model.train()
        end = time.time()

        try:
            images_l, targets = labeled_iter.next()
        except BaseException:
            if args.world_size > 1:
                labeled_epoch += 1
                labeled_loader.sampler.set_epoch(labeled_epoch)
            labeled_iter = iter(labeled_loader)
            images_l, targets = labeled_iter.next()

        try:
            (images_uw, images_us), _ = unlabeled_iter.next()
        except BaseException:
            if args.world_size > 1:
                unlabeled_epoch += 1
                unlabeled_loader.sampler.set_epoch(unlabeled_epoch)
            unlabeled_iter = iter(unlabeled_loader)
            (images_uw, images_us), _ = unlabeled_iter.next()

        data_time.update(time.time() - end)

        images_l = images_l.to(args.device)
        images_uw = images_uw.to(args.device)
        images_us = images_us.to(args.device)
        targets = targets.to(args.device)
        with amp.autocast(enabled=args.amp):
            batch_size = images_l.shape[0]
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
                -(soft_pseudo_label * torch.log_softmax(t_logits_us,
                                                        dim=-1)).sum(dim=-1) * mask)
            weight_u = args.lambda_u * min(1., (step + 1) / args.uda_steps)
            t_loss_uda = t_loss_l + weight_u * t_loss_u

            s_images = torch.cat((images_l, images_us))
            s_logits = student_model(s_images)
            s_logits_l = s_logits[:batch_size]
            s_logits_us = s_logits[batch_size:]
            del s_logits

            s_loss_l_old = F.cross_entropy(s_logits_l.detach(), targets)
            s_loss = criterion(s_logits_us, hard_pseudo_label)

        s_scaler.scale(s_loss).backward()
        if args.grad_clip > 0:
            s_scaler.unscale_(s_optimizer)
            nn.utils.clip_grad_norm_(
                student_model.parameters(), args.grad_clip)
        s_scaler.step(s_optimizer)
        s_scaler.update()
        s_scheduler.step()
        if args.ema > 0:
            avg_student_model.update_parameters(student_model)

        with amp.autocast(enabled=args.amp):
            with torch.no_grad():
                s_logits_l = student_model(images_l)
            s_loss_l_new = F.cross_entropy(s_logits_l.detach(), targets)
            dot_product = s_loss_l_new - s_loss_l_old
            moving_dot_product = moving_dot_product * 0.99 + dot_product * 0.01
            dot_product = dot_product - moving_dot_product
            _, hard_pseudo_label = torch.max(t_logits_us.detach(), dim=-1)
            t_loss_mpl = dot_product * \
                F.cross_entropy(t_logits_us.detach(), hard_pseudo_label)
            t_loss = t_loss_uda + t_loss_mpl

        t_scaler.scale(t_loss).backward()
        if args.grad_clip > 0:
            t_scaler.unscale_(t_optimizer)
            nn.utils.clip_grad_norm_(
                teacher_model.parameters(), args.grad_clip)
        t_scaler.step(t_optimizer)
        t_scaler.update()
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

        s_losses.update(s_loss.item())
        t_losses.update(t_loss.item())
        t_losses_l.update(t_loss_l.item())
        t_losses_u.update(t_loss_u.item())
        t_losses_mpl.update(t_loss_mpl.item())
        mean_mask.update(mask.mean().item())

        batch_time.update(time.time() - end)
        pbar.set_description(
            f"Train Iter: {step+1:3}/{args.total_steps:3}. "
            f"LR: {get_lr(s_optimizer):.4f}. Data: {data_time.avg:.2f}s. "
            f"Batch: {batch_time.avg:.2f}s. S_Loss: {s_losses.avg:.4f}. "
            f"T_Loss: {t_losses.avg:.4f}. Mask: {mean_mask.avg:.4f}. ")
        pbar.update()
        if args.local_rank in [-1, 0]:
            args.writer.add_scalar("lr", get_lr(s_optimizer), step)

        args.num_eval = step // args.eval_step
        if (step + 1) % args.eval_step == 0:
            pbar.close()
            if args.local_rank in [-1, 0]:
                args.writer.add_scalar(
                    "train/1.s_loss", s_losses.avg, args.num_eval)
                args.writer.add_scalar(
                    "train/2.t_loss", t_losses.avg, args.num_eval)
                args.writer.add_scalar(
                    "train/3.t_labeled", t_losses_l.avg, args.num_eval)
                args.writer.add_scalar(
                    "train/4.t_unlabeled", t_losses_u.avg, args.num_eval)
                args.writer.add_scalar(
                    "train/5.t_mpl", t_losses_mpl.avg, args.num_eval)
                args.writer.add_scalar(
                    "train/6.mask", mean_mask.avg, args.num_eval)

                test_model = avg_student_model if avg_student_model is not None else student_model
                losses, top1, top5 = evaluate(
                    args, test_loader, test_model, criterion)
                is_best = top1 > args.best_top1
                if is_best:
                    args.best_top1 = top1
                    args.best_top5 = top5

                logger.info(f"top-1 acc: {top1:.2f}")
                logger.info(f"Best top-1 acc: {args.best_top1:.2f}")

                save_checkpoint(args, {
                    'step': step + 1,
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
    return


def evaluate(args, test_loader, model, criterion):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
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
                output = model(images)
                loss = criterion(output, targets)

            acc1, acc5 = accuracy(output, targets, (1, 5))
            losses.update(loss.item(), batch_size)
            top1.update(acc1[0], batch_size)
            top5.update(acc5[0], batch_size)

            batch_time.update(time.time() - end)
            end = time.time()

            test_iter.set_description(
                f"Test Iter: {step+1:3}/{len(test_loader):3}. Data: {data_time.avg:.2f}s. "
                f"Batch: {batch_time.avg:.2f}s. Loss: {losses.avg:.4f}. "
                f"top1: {top1.avg:.2f}. top5: {top5.avg:.2f}. ")

        test_iter.close()
        if args.local_rank in [-1, 0]:
            args.writer.add_scalar("test/loss", losses.avg, args.num_eval)
            args.writer.add_scalar("test/acc@1", top1.avg, args.num_eval)
            args.writer.add_scalar("test/acc@5", top5.avg, args.num_eval)

        return losses.avg, top1.avg, top5.avg
