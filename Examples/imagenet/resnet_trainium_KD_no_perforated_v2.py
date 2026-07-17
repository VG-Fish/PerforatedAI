"""

"""

import datetime
import os
import random
import time
import warnings
import argparse
from contextlib import nullcontext
import torch.nn.functional as F

import presets
import torch
import torch.utils.data
import torchvision
import torchvision.transforms
import utils
from sampler import RASampler
from torch import nn
from torch.utils.data.dataloader import default_collate
from torchvision.transforms.functional import InterpolationMode
from transforms import get_mixup_cutmix

import wandb
from types import SimpleNamespace

# NEURON EDIT: XLA device support for Trainium.
try:
    import torch_xla
    import torch_xla.core.xla_model as xm

    HAS_XLA = True
except ImportError:
    HAS_XLA = False


KD_ALPHA = 0.4
KD_TEMPERATURE = 4.0


def count_model_params(model):
    return sum(p.numel() for p in model.parameters())


def is_xla_device(device):
    return str(device).startswith("xla")


def resolve_loader_dataset(data_loader):
    dataset = getattr(data_loader, "dataset", None)
    if dataset is not None:
        return dataset
    wrapped_loader = getattr(data_loader, "_loader", None)
    if wrapped_loader is not None:
        return getattr(wrapped_loader, "dataset", None)
    return None


def xla_sync_step():
    if HAS_XLA:
        torch_xla.sync()


def maybe_autocast(device, scaler):
    if scaler is not None and device.type == "cuda":
        return torch.amp.autocast(device_type="cuda", enabled=True)
    return nullcontext()


def compute_ce_loss(criterion, output, target):
    # Mixup/CutMix produces soft targets [N, C]; Neuron CE expects class indices.
    if target.ndim == output.ndim:
        return -(target * F.log_softmax(output, dim=1)).sum(dim=1).mean()
    return criterion(output, target)


def compute_accuracy(output, target, topk=(1, 5)):
    # Neuron Trn2 does not support the XLA sort op emitted by topk accuracy.
    # Keep metric computation on-device and use top-1 fallback on XLA.
    if is_xla_device(output.device):
        if target.ndim == 2:
            target = target.argmax(dim=1)
        pred = output.argmax(dim=1)
        correct = pred.eq(target).sum(dtype=torch.float32)
        acc1 = correct * (100.0 / target.size(0))
        # For XLA, avoid top-k sort to keep graph small and compilable.
        return [acc1 for _ in topk]
    return utils.accuracy(output, target, topk=topk)


def train_one_epoch(
    model,
    teacher_model,
    criterion,
    optimizer,
    data_loader,
    device,
    epoch,
    args,
    model_ema=None,
    scaler=None,
):
    model.train()
    if args.use_kd and teacher_model is not None:
        teacher_model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
    metric_logger.add_meter("img/s", utils.SmoothedValue(window_size=10, fmt="{value}"))
    metric_logger.add_meter("ce", utils.SmoothedValue(window_size=10, fmt="{value}"))
    metric_logger.add_meter("kd", utils.SmoothedValue(window_size=10, fmt="{value}"))

    header = f"Epoch: [{epoch}]"
    running_correct = torch.zeros((), device=device) if args.use_xla else 0.0
    running_total = 0
    running_loss_sum = torch.zeros((), device=device) if args.use_xla else 0.0
    running_ce_sum = torch.zeros((), device=device) if args.use_xla else 0.0
    running_kd_sum = torch.zeros((), device=device) if args.use_xla else 0.0

    if args.use_xla:
        data_iter = enumerate(data_loader)
    else:
        data_iter = enumerate(metric_logger.log_every(data_loader, args.print_freq, header))

    for i, (image, target) in data_iter:
        start_time = time.time()
        image, target = image.to(device), target.to(device)

        teacher_output = None
        if args.use_kd and teacher_model is not None:
            with torch.no_grad():
                teacher_output = teacher_model(image)

        with maybe_autocast(device, scaler):
            output = model(image)
            ce_loss = compute_ce_loss(criterion, output, target)
            if args.use_kd and teacher_output is not None:
                kd_loss = F.kl_div(
                    F.log_softmax(output / KD_TEMPERATURE, dim=1),
                    F.softmax(teacher_output / KD_TEMPERATURE, dim=1),
                    reduction="batchmean",
                ) * (KD_TEMPERATURE * KD_TEMPERATURE)
                loss = (1.0 - KD_ALPHA) * ce_loss + KD_ALPHA * kd_loss
            else:
                kd_loss = torch.zeros(1, device=device, dtype=ce_loss.dtype)
                loss = ce_loss

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            if args.clip_grad_norm is not None:
                # we should unscale the gradients of optimizer's assigned params if do gradient clipping
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.clip_grad_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()

        if model_ema and i % args.model_ema_steps == 0:
            model_ema.update_parameters(model)
            if epoch < args.lr_warmup_epochs:
                # Reset ema buffer to keep copying weights during warmup period
                model_ema.n_averaged.fill_(0)

        batch_size = image.shape[0]
        if target.ndim == 2:
            target_for_acc = target.argmax(dim=1)
        else:
            target_for_acc = target
        pred = output.argmax(dim=1)
        correct1 = pred.eq(target_for_acc).sum()
        if args.use_xla:
            running_correct = running_correct + correct1
        else:
            running_correct += float(correct1.item())
        running_total += batch_size
        if args.use_xla:
            running_loss_sum = running_loss_sum + loss.detach() * batch_size
            running_ce_sum = running_ce_sum + ce_loss.detach() * batch_size
            running_kd_sum = running_kd_sum + kd_loss.detach() * batch_size

        if args.use_xla:
            if i % args.print_freq == 0:
                elapsed = time.time() - start_time
                imgs = batch_size / elapsed if elapsed > 0 else 0.0
                print(
                    f"{header}  [{i}/{len(data_loader)}]  lr: {optimizer.param_groups[0]['lr']}  img/s: {imgs:.4f}"
                )
        else:
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
            acc1, acc5 = compute_accuracy(output, target, topk=(1, 5))
            metric_logger.update(loss=loss.item())
            metric_logger.update(ce=ce_loss.item(), kd=kd_loss.item())
            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
            metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
            metric_logger.meters["img/s"].update(batch_size / (time.time() - start_time))

    if args.use_xla and running_total > 0:
        train_loss = (running_loss_sum * (1.0 / running_total)).item()
        train_ce = (running_ce_sum * (1.0 / running_total)).item()
        train_kd = (running_kd_sum * (1.0 / running_total)).item()
        train_acc1 = (running_correct * (100.0 / running_total)).item()
        train_acc5 = train_acc1
        print(
            f"{header} Summary loss: {train_loss:.4f} ce: {train_ce:.4f} kd: {train_kd:.4f} "
            f"Acc@1 {train_acc1:.3f} Acc@5 {train_acc5:.3f}"
        )
    else:
        train_acc1 = metric_logger.acc1.global_avg
        train_acc5 = metric_logger.acc5.global_avg


def train_one_epoch_supervised(
    model,
    criterion,
    optimizer,
    data_loader,
    device,
    epoch,
    args,
    scaler=None,
):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
    header = f"Teacher Epoch: [{epoch}]"
    running_correct = torch.zeros((), device=device) if args.use_xla else 0.0
    running_total = 0
    running_loss_sum = torch.zeros((), device=device) if args.use_xla else 0.0

    if args.use_xla:
        data_iter = enumerate(data_loader)
    else:
        data_iter = enumerate(metric_logger.log_every(data_loader, args.print_freq, header))

    for i, (image, target) in data_iter:
        image, target = image.to(device), target.to(device)

        with maybe_autocast(device, scaler):
            output = model(image)
            loss = compute_ce_loss(criterion, output, target)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            if args.clip_grad_norm is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.clip_grad_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()

        batch_size = image.shape[0]
        if target.ndim == 2:
            target_for_acc = target.argmax(dim=1)
        else:
            target_for_acc = target
        pred = output.argmax(dim=1)
        correct1 = pred.eq(target_for_acc).sum()
        if args.use_xla:
            running_correct = running_correct + correct1
        else:
            running_correct += float(correct1.item())
        running_total += batch_size
        if args.use_xla:
            running_loss_sum = running_loss_sum + loss.detach() * batch_size

        if args.use_xla:
            if i % args.print_freq == 0:
                print(f"{header}  [{i}/{len(data_loader)}]  lr: {optimizer.param_groups[0]['lr']}")
        else:
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
            acc1, acc5 = compute_accuracy(output, target, topk=(1, 5))
            metric_logger.update(loss=loss.item())
            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
            metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)

    metric_logger.synchronize_between_processes()
    if args.use_xla and running_total > 0:
        final_loss = (running_loss_sum * (1.0 / running_total)).item()
        final_acc1 = (running_correct * (100.0 / running_total)).item()
        final_acc5 = final_acc1
        print(
            f"{header} Summary loss: {final_loss:.4f} Acc@1 {final_acc1:.3f} Acc@5 {final_acc5:.3f}"
        )
        return
    print(
        f"{header} Acc@1 {metric_logger.acc1.global_avg:.3f} Acc@5 {metric_logger.acc5.global_avg:.3f}"
    )


def evaluate_plain(model, criterion, data_loader, device, print_freq=100, log_suffix=""):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f"EvalPlain: {log_suffix}"
    running_correct = torch.zeros((), device=device) if is_xla_device(device) else 0.0
    running_total = 0
    running_loss_sum = torch.zeros((), device=device) if is_xla_device(device) else 0.0

    if is_xla_device(device):
        data_iter = enumerate(data_loader)
    else:
        data_iter = enumerate(metric_logger.log_every(data_loader, print_freq, header))

    with torch.no_grad():
        for i, (image, target) in data_iter:
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(image)
            loss = criterion(output, target)

            batch_size = image.shape[0]
            if is_xla_device(device):
                target_for_acc = target.argmax(dim=1) if target.ndim == 2 else target
                pred = output.argmax(dim=1)
                running_correct = running_correct + pred.eq(target_for_acc).sum()
                running_total += batch_size
                running_loss_sum = running_loss_sum + loss.detach() * batch_size
                if i % print_freq == 0:
                    print(f"{header}  [{i}/{len(data_loader)}]")
            else:
                acc1, acc5 = compute_accuracy(output, target, topk=(1, 5))
                metric_logger.update(loss=loss.item())
                metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
                metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)

    metric_logger.synchronize_between_processes()
    if is_xla_device(device) and running_total > 0:
        eval_loss = (running_loss_sum * (1.0 / running_total)).item()
        eval_acc1 = (running_correct * (100.0 / running_total)).item()
        eval_acc5 = eval_acc1
        print(f"{header} Acc@1 {eval_acc1:.3f} Acc@5 {eval_acc5:.3f} loss {eval_loss:.4f}")
        return eval_acc1
    print(
        f"{header} Acc@1 {metric_logger.acc1.global_avg:.3f} Acc@5 {metric_logger.acc5.global_avg:.3f}"
    )
    return metric_logger.acc1.global_avg


def pretrain_teacher(
    args,
    device,
    num_classes,
    data_loader,
    data_loader_val,
    data_loader_test,
):
    if args.distributed:
        raise RuntimeError("--pre-train-teacher currently supports single-process training only")

    print("Pre-training teacher model: resnet50")
    teacher_model = torchvision.models.resnet50(
        weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2
    )
    if teacher_model.fc.out_features != num_classes:
        teacher_model.fc = nn.Linear(teacher_model.fc.in_features, num_classes)
    teacher_model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.SGD(
        teacher_model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov="nesterov" in args.opt.lower(),
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma
    )
    scaler = torch.cuda.amp.GradScaler() if args.amp and device.type == "cuda" else None

    if args.teacher_checkpoint:
        teacher_checkpoint_path = args.teacher_checkpoint
    else:
        base_dir = args.output_dir if args.output_dir else "."
        teacher_checkpoint_path = os.path.join(
            base_dir, f"teacher_resnet50_{args.dataset}.pth"
        )
    os.makedirs(os.path.dirname(os.path.abspath(teacher_checkpoint_path)), exist_ok=True)

    best_val_acc = -1.0
    for epoch in range(args.start_epoch, args.epochs):
        train_one_epoch_supervised(
            teacher_model,
            criterion,
            optimizer,
            data_loader,
            device,
            epoch,
            args,
            scaler,
        )
        val_acc1 = evaluate_plain(
            teacher_model,
            criterion,
            data_loader_val,
            device,
            print_freq=args.print_freq,
            log_suffix="val",
        )
        evaluate_plain(
            teacher_model,
            criterion,
            data_loader_test,
            device,
            print_freq=args.print_freq,
            log_suffix="test",
        )
        lr_scheduler.step()

        if val_acc1 > best_val_acc:
            best_val_acc = val_acc1
            torch.save(
                {
                    "model": teacher_model.state_dict(),
                    "dataset": args.dataset,
                    "num_classes": num_classes,
                    "best_val_acc1": best_val_acc,
                },
                teacher_checkpoint_path,
            )
            print(
                f"Saved teacher checkpoint to {teacher_checkpoint_path} (val Acc@1={best_val_acc:.3f})"
            )

    print(f"Teacher pretraining complete. Best checkpoint: {teacher_checkpoint_path}")

def evaluate(model, criterion, data_loader, device, print_freq=100, log_suffix=""):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f"Test: {log_suffix}"

    num_processed_samples = 0
    running_correct = torch.zeros((), device=device) if is_xla_device(device) else 0.0
    running_total = 0
    running_loss_sum = torch.zeros((), device=device) if is_xla_device(device) else 0.0

    if is_xla_device(device):
        data_iter = enumerate(data_loader)
    else:
        data_iter = enumerate(metric_logger.log_every(data_loader, print_freq, header))

    with torch.no_grad():
        for i, (image, target) in data_iter:
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(image)
            loss = criterion(output, target)

            # FIXME need to take into account that the datasets
            # could have been padded in distributed setup
            batch_size = image.shape[0]
            if is_xla_device(device):
                target_for_acc = target.argmax(dim=1) if target.ndim == 2 else target
                pred = output.argmax(dim=1)
                running_correct = running_correct + pred.eq(target_for_acc).sum()
                running_total += batch_size
                running_loss_sum = running_loss_sum + loss.detach() * batch_size
                if i % print_freq == 0:
                    print(f"{header}  [{i}/{len(data_loader)}]")
            else:
                acc1, acc5 = compute_accuracy(output, target, topk=(1, 5))
                metric_logger.update(loss=loss.item())
                metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
                metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
            num_processed_samples += batch_size

    # gather the stats from all processes
    num_processed_samples = utils.reduce_across_processes(num_processed_samples)
    is_rank0 = (
        (not torch.distributed.is_available())
        or (not torch.distributed.is_initialized())
        or torch.distributed.get_rank() == 0
    )
    eval_dataset = resolve_loader_dataset(data_loader)
    if (
        eval_dataset is not None
        and hasattr(eval_dataset, "__len__")
        and len(eval_dataset) != num_processed_samples
        and is_rank0
    ):
        # See FIXME above
        warnings.warn(
            f"It looks like the dataset has {len(eval_dataset)} samples, but {num_processed_samples} "
            "samples were used for the validation, which might bias the results. "
            "Try adjusting the batch size and / or the world size. "
            "Setting the world size to 1 is always a safe bet."
        )

    metric_logger.synchronize_between_processes()

    if is_xla_device(device) and running_total > 0:
        val_loss = (running_loss_sum * (1.0 / running_total)).item()
        val_acc1 = (running_correct * (100.0 / running_total)).item()
        val_acc5 = val_acc1
        print(f"{header} Acc@1 {val_acc1:.3f} Acc@5 {val_acc5:.3f} loss {val_loss:.4f}")
        return val_acc1

    print(
        f"{header} Acc@1 {metric_logger.acc1.global_avg:.3f} Acc@5 {metric_logger.acc5.global_avg:.3f}"
    )

    return metric_logger.acc1.global_avg


def test(model, criterion, data_loader, device, print_freq=100, log_suffix=""):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f"TestHoldout: {log_suffix}"

    num_processed_samples = 0
    running_correct = torch.zeros((), device=device) if is_xla_device(device) else 0.0
    running_total = 0
    running_loss_sum = torch.zeros((), device=device) if is_xla_device(device) else 0.0

    if is_xla_device(device):
        data_iter = enumerate(data_loader)
    else:
        data_iter = enumerate(metric_logger.log_every(data_loader, print_freq, header))

    with torch.no_grad():
        for i, (image, target) in data_iter:
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(image)
            loss = criterion(output, target)

            batch_size = image.shape[0]
            if is_xla_device(device):
                target_for_acc = target.argmax(dim=1) if target.ndim == 2 else target
                pred = output.argmax(dim=1)
                running_correct = running_correct + pred.eq(target_for_acc).sum()
                running_total += batch_size
                running_loss_sum = running_loss_sum + loss.detach() * batch_size
                if i % print_freq == 0:
                    print(f"{header}  [{i}/{len(data_loader)}]")
            else:
                acc1, acc5 = compute_accuracy(output, target, topk=(1, 5))
                metric_logger.update(loss=loss.item())
                metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
                metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
            num_processed_samples += batch_size

    num_processed_samples = utils.reduce_across_processes(num_processed_samples)
    is_rank0 = (
        (not torch.distributed.is_available())
        or (not torch.distributed.is_initialized())
        or torch.distributed.get_rank() == 0
    )
    test_dataset = resolve_loader_dataset(data_loader)
    if (
        test_dataset is not None
        and hasattr(test_dataset, "__len__")
        and len(test_dataset) != num_processed_samples
        and is_rank0
    ):
        warnings.warn(
            f"It looks like the dataset has {len(test_dataset)} samples, but {num_processed_samples} "
            "samples were used for testing, which might bias the results. "
            "Try adjusting the batch size and / or the world size. "
            "Setting the world size to 1 is always a safe bet."
        )

    metric_logger.synchronize_between_processes()

    if is_xla_device(device) and running_total > 0:
        test_loss = (running_loss_sum * (1.0 / running_total)).item()
        test_acc1 = (running_correct * (100.0 / running_total)).item()
        test_acc5 = test_acc1
        print(f"{header} Acc@1 {test_acc1:.3f} Acc@5 {test_acc5:.3f} loss {test_loss:.4f}")
        return test_acc1

    print(
        f"{header} Acc@1 {metric_logger.acc1.global_avg:.3f} Acc@5 {metric_logger.acc5.global_avg:.3f}"
    )

    return metric_logger.acc1.global_avg


def _get_cache_path(filepath):
    import hashlib

    h = hashlib.sha1(filepath.encode()).hexdigest()
    cache_path = os.path.join(
        "~", ".torch", "vision", "datasets", "imagefolder", h[:10] + ".pt"
    )
    cache_path = os.path.expanduser(cache_path)
    return cache_path


# ImageNet-100 standard class indices (commonly used subset)
IMAGENET100_CLASSES = [
    "n01440764",
    "n01443537",
    "n01484850",
    "n01491361",
    "n01494475",
    "n01496331",
    "n01498041",
    "n01514668",
    "n01514859",
    "n01518878",
    "n01530575",
    "n01531178",
    "n01532829",
    "n01534433",
    "n01537544",
    "n01558993",
    "n01560419",
    "n01580077",
    "n01582220",
    "n01592084",
    "n01601694",
    "n01608432",
    "n01614925",
    "n01616318",
    "n01622779",
    "n01629819",
    "n01630670",
    "n01631663",
    "n01632458",
    "n01632777",
    "n01641577",
    "n01644373",
    "n01644900",
    "n01664065",
    "n01665541",
    "n01667114",
    "n01667778",
    "n01669191",
    "n01675722",
    "n01677366",
    "n01682714",
    "n01685808",
    "n01687978",
    "n01688243",
    "n01689811",
    "n01692333",
    "n01693334",
    "n01694178",
    "n01695060",
    "n01697457",
    "n01698640",
    "n01704323",
    "n01728572",
    "n01728920",
    "n01729322",
    "n01729977",
    "n01734418",
    "n01735189",
    "n01737021",
    "n01739381",
    "n01740131",
    "n01742172",
    "n01744401",
    "n01748264",
    "n01749939",
    "n01751748",
    "n01753488",
    "n01755581",
    "n01756291",
    "n01768244",
    "n01770081",
    "n01770393",
    "n01773157",
    "n01773549",
    "n01773797",
    "n01774384",
    "n01774750",
    "n01775062",
    "n01776313",
    "n01784675",
    "n01795545",
    "n01796340",
    "n01797886",
    "n01798484",
    "n01806143",
    "n01806567",
    "n01807496",
    "n01817953",
    "n01818515",
    "n01819313",
    "n01820546",
    "n01824575",
    "n01828970",
    "n01829413",
    "n01833805",
    "n01843065",
    "n01843383",
    "n01847000",
    "n01855032",
    "n01855672",
]


def filter_imagenet100(dataset):
    """Filter dataset to only include ImageNet-100 classes."""
    # Get original class_to_idx mapping
    original_class_to_idx = dataset.class_to_idx

    # Create mapping from old indices to new indices
    valid_classes = [cls for cls in IMAGENET100_CLASSES if cls in original_class_to_idx]
    new_class_to_idx = {cls: new_idx for new_idx, cls in enumerate(valid_classes)}
    old_to_new_idx = {
        original_class_to_idx[cls]: new_idx for cls, new_idx in new_class_to_idx.items()
    }

    # Filter samples
    filtered_samples = []
    for path, old_idx in dataset.samples:
        if old_idx in old_to_new_idx:
            filtered_samples.append((path, old_to_new_idx[old_idx]))

    # Update dataset
    dataset.samples = filtered_samples
    dataset.targets = [s[1] for s in filtered_samples]
    dataset.classes = valid_classes
    dataset.class_to_idx = new_class_to_idx

    print(
        f"Filtered dataset to {len(valid_classes)} classes with {len(filtered_samples)} samples"
    )
    return dataset


def stratified_subset_by_class(dataset, fraction, seed):
    """Return a stratified random subset preserving per-class proportions."""
    if fraction >= 1.0:
        return dataset
    if fraction <= 0.0:
        raise ValueError("train_label_fraction must be in (0, 1].")

    if hasattr(dataset, "targets"):
        targets = dataset.targets
    elif hasattr(dataset, "_labels"):
        targets = dataset._labels
    elif hasattr(dataset, "samples"):
        targets = [s[1] for s in dataset.samples]
    else:
        raise RuntimeError("Dataset does not expose targets/samples for stratified subset")

    class_to_indices = {}
    for idx, cls in enumerate(targets):
        class_to_indices.setdefault(int(cls), []).append(idx)

    rng = random.Random(seed)
    selected_indices = []
    for cls in sorted(class_to_indices.keys()):
        cls_indices = class_to_indices[cls]
        shuffled = cls_indices[:]
        rng.shuffle(shuffled)
        keep = int(round(len(cls_indices) * fraction))
        keep = max(1, min(len(cls_indices), keep))
        selected_indices.extend(shuffled[:keep])

    rng.shuffle(selected_indices)
    print(
        f"Using stratified train subset: fraction={fraction}, seed={seed}, "
        f"samples={len(selected_indices)}/{len(targets)}"
    )
    return torch.utils.data.Subset(dataset, selected_indices)


def get_dataset_classes(dataset):
    """Return class labels, unwrapping Subset wrappers if needed."""
    if hasattr(dataset, "classes"):
        return dataset.classes
    if isinstance(dataset, torch.utils.data.Subset):
        return get_dataset_classes(dataset.dataset)
    raise RuntimeError("Dataset does not expose class labels via .classes")


def split_eval_dataset_stratified(dataset, seed):
    """Split an eval dataset into stratified val/test halves per class."""
    if hasattr(dataset, "targets"):
        targets = dataset.targets
    elif hasattr(dataset, "_labels"):
        targets = dataset._labels
    elif hasattr(dataset, "samples"):
        targets = [s[1] for s in dataset.samples]
    else:
        raise RuntimeError("Dataset does not expose targets/samples for stratified split")

    class_to_indices = {}
    for idx, cls in enumerate(targets):
        class_to_indices.setdefault(int(cls), []).append(idx)

    rng = random.Random(seed)
    val_indices = []
    test_indices = []

    for cls in sorted(class_to_indices.keys()):
        indices = class_to_indices[cls][:]
        rng.shuffle(indices)
        split_point = len(indices) // 2
        val_indices.extend(indices[:split_point])
        test_indices.extend(indices[split_point:])

    rng.shuffle(val_indices)
    rng.shuffle(test_indices)

    print(
        f"Split eval set stratified by class: val={len(val_indices)}, test={len(test_indices)}, seed={seed}"
    )
    return (
        torch.utils.data.Subset(dataset, val_indices),
        torch.utils.data.Subset(dataset, test_indices),
    )


def create_optimizer_and_scheduler(model, args, custom_keys_weight_decay, epoch=None):
    """Create optimizer and scheduler for the model."""
    parameters = utils.set_weight_decay(
        model,
        args.weight_decay,
        norm_weight_decay=args.norm_weight_decay,
        custom_keys_weight_decay=(
            custom_keys_weight_decay if len(custom_keys_weight_decay) > 0 else None
        ),
    )

    opt_name = args.opt.lower()
    if opt_name.startswith("sgd"):
        optimizer = torch.optim.SGD(
            parameters,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov="nesterov" in opt_name,
        )
    elif opt_name == "rmsprop":
        optimizer = torch.optim.RMSprop(
            parameters,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            eps=0.0316,
            alpha=0.9,
        )
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(
            parameters,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    else:
        raise RuntimeError(
            f"Invalid optimizer {args.opt}. Only SGD, RMSprop and AdamW are supported."
        )

    args.lr_scheduler = args.lr_scheduler.lower()
    warmup_epochs_remaining = (
        args.lr_warmup_epochs
        if epoch is None
        else max(0, args.lr_warmup_epochs - epoch)
    )

    if args.lr_scheduler == "steplr":
        main_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma
        )
    elif args.lr_scheduler == "cosineannealinglr":
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, args.epochs - args.lr_warmup_epochs),
            eta_min=args.lr_min,
        )
    elif args.lr_scheduler == "exponentiallr":
        main_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=args.lr_gamma
        )
    elif args.lr_scheduler == "reducelronplateau":
        main_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.1,
            patience=10,
        )
    else:
        raise RuntimeError(
            f"Invalid lr scheduler '{args.lr_scheduler}'. Only StepLR, CosineAnnealingLR, ExponentialLR and ReduceLROnPlateau "
            "are supported."
        )

    if warmup_epochs_remaining > 0 and args.lr_scheduler != "reducelronplateau":
        if args.lr_warmup_method == "linear":
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=args.lr_warmup_decay,
                total_iters=warmup_epochs_remaining,
            )
        elif args.lr_warmup_method == "constant":
            warmup_scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer,
                factor=args.lr_warmup_decay,
                total_iters=warmup_epochs_remaining,
            )
        else:
            raise RuntimeError(
                f"Invalid warmup lr method '{args.lr_warmup_method}'. Only linear and constant are supported."
            )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_epochs_remaining],
        )
    else:
        lr_scheduler = main_scheduler

    return optimizer, lr_scheduler


def load_data(traindir, valdir, args):
    # Data loading code
    print("Loading data")

    if args.dataset == "food101":
        interpolation = InterpolationMode(args.interpolation)
        train_transform = presets.ClassificationPresetTrain(
            crop_size=args.train_crop_size,
            interpolation=interpolation,
            auto_augment_policy=args.auto_augment,
            random_erase_prob=args.random_erase,
            ra_magnitude=args.ra_magnitude,
            augmix_severity=args.augmix_severity,
            backend=args.backend,
            use_v2=args.use_v2,
        )
        eval_transform = presets.ClassificationPresetEval(
            crop_size=args.val_crop_size,
            resize_size=args.val_resize_size,
            interpolation=interpolation,
            backend=args.backend,
            use_v2=args.use_v2,
        )

        dataset = torchvision.datasets.Food101(
            root=args.data_path,
            split="train",
            transform=train_transform,
            download=args.download_food101,
        )
        dataset = stratified_subset_by_class(
            dataset, args.train_label_fraction, args.label_subset_seed
        )

        eval_dataset = torchvision.datasets.Food101(
            root=args.data_path,
            split="test",
            transform=eval_transform,
            download=args.download_food101,
        )
        dataset_val, dataset_test = split_eval_dataset_stratified(
            eval_dataset, args.val_test_split_seed
        )

        print("Creating data loaders")
        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
            val_sampler = torch.utils.data.distributed.DistributedSampler(
                dataset_val, shuffle=False
            )
            test_sampler = torch.utils.data.distributed.DistributedSampler(
                dataset_test, shuffle=False
            )
        else:
            train_sampler = torch.utils.data.RandomSampler(dataset)
            val_sampler = torch.utils.data.SequentialSampler(dataset_val)
            test_sampler = torch.utils.data.SequentialSampler(dataset_test)

        return dataset, dataset_val, dataset_test, train_sampler, val_sampler, test_sampler

    if args.dataset == "cifar100":
        normalize = torchvision.transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616),
        )
        train_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomCrop(32, padding=4),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                normalize,
            ]
        )
        test_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                normalize,
            ]
        )

        dataset = torchvision.datasets.CIFAR100(
            root=args.data_path,
            train=True,
            transform=train_transform,
            download=args.download_cifar,
        )
        dataset = stratified_subset_by_class(
            dataset, args.train_label_fraction, args.label_subset_seed
        )

        eval_dataset = torchvision.datasets.CIFAR100(
            root=args.data_path,
            train=False,
            transform=test_transform,
            download=args.download_cifar,
        )
        dataset_val, dataset_test = split_eval_dataset_stratified(
            eval_dataset, args.val_test_split_seed
        )

        print("Creating data loaders")
        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
            val_sampler = torch.utils.data.distributed.DistributedSampler(
                dataset_val, shuffle=False
            )
            test_sampler = torch.utils.data.distributed.DistributedSampler(
                dataset_test, shuffle=False
            )
        else:
            train_sampler = torch.utils.data.RandomSampler(dataset)
            val_sampler = torch.utils.data.SequentialSampler(dataset_val)
            test_sampler = torch.utils.data.SequentialSampler(dataset_test)

        return dataset, dataset_val, dataset_test, train_sampler, val_sampler, test_sampler

    val_resize_size, val_crop_size, train_crop_size = (
        args.val_resize_size,
        args.val_crop_size,
        args.train_crop_size,
    )
    interpolation = InterpolationMode(args.interpolation)

    print("Loading training data")
    st = time.time()
    cache_path = _get_cache_path(traindir)
    if args.cache_dataset and os.path.exists(cache_path):
        # Attention, as the transforms are also cached!
        print(f"Loading dataset_train from {cache_path}")
        # TODO: this could probably be weights_only=True
        dataset, _ = torch.load(cache_path, weights_only=False)
    else:
        # We need a default value for the variables below because args may come
        # from train_quantization.py which doesn't define them.
        auto_augment_policy = getattr(args, "auto_augment", None)
        random_erase_prob = getattr(args, "random_erase", 0.0)
        ra_magnitude = getattr(args, "ra_magnitude", None)
        augmix_severity = getattr(args, "augmix_severity", None)
        dataset = torchvision.datasets.ImageFolder(
            traindir,
            presets.ClassificationPresetTrain(
                crop_size=train_crop_size,
                interpolation=interpolation,
                auto_augment_policy=auto_augment_policy,
                random_erase_prob=random_erase_prob,
                ra_magnitude=ra_magnitude,
                augmix_severity=augmix_severity,
                backend=args.backend,
                use_v2=args.use_v2,
            ),
        )
        # Filter to ImageNet-100 unless full dataset is requested
        if not args.full_dataset:
            dataset = filter_imagenet100(dataset)

        if args.cache_dataset:
            print(f"Saving dataset_train to {cache_path}")
            utils.mkdir(os.path.dirname(cache_path))
            utils.save_on_master((dataset, traindir), cache_path)
    print("Took", time.time() - st)

    print("Loading validation data")
    cache_path = _get_cache_path(valdir)
    if args.cache_dataset and os.path.exists(cache_path):
        # Attention, as the transforms are also cached!
        print(f"Loading dataset_test from {cache_path}")
        # TODO: this could probably be weights_only=True
        dataset_test, _ = torch.load(cache_path, weights_only=False)
    else:
        if args.weights and args.test_only:
            weights = torchvision.models.get_weight(args.weights)
            preprocessing = weights.transforms(antialias=True)
            if args.backend == "tensor":
                preprocessing = torchvision.transforms.Compose(
                    [torchvision.transforms.PILToTensor(), preprocessing]
                )

        else:
            preprocessing = presets.ClassificationPresetEval(
                crop_size=val_crop_size,
                resize_size=val_resize_size,
                interpolation=interpolation,
                backend=args.backend,
                use_v2=args.use_v2,
            )

        dataset_test = torchvision.datasets.ImageFolder(
            valdir,
            preprocessing,
        )
        # Filter to ImageNet-100 unless full dataset is requested
        if not args.full_dataset:
            dataset_test = filter_imagenet100(dataset_test)

        if args.cache_dataset:
            print(f"Saving dataset_test to {cache_path}")
            utils.mkdir(os.path.dirname(cache_path))
            utils.save_on_master((dataset_test, valdir), cache_path)

    dataset_val, dataset_test = split_eval_dataset_stratified(
        dataset_test, args.val_test_split_seed
    )

    print("Creating data loaders")
    if args.distributed:
        if hasattr(args, "ra_sampler") and args.ra_sampler:
            train_sampler = RASampler(dataset, shuffle=True, repetitions=args.ra_reps)
        else:
            train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset_val, shuffle=False
        )
        test_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset_test, shuffle=False
        )
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        val_sampler = torch.utils.data.SequentialSampler(dataset_val)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    return dataset, dataset_val, dataset_test, train_sampler, val_sampler, test_sampler


def main(args):
    # Initialize wandb if enabled
    run = None
    if args.use_wandb:
        run = wandb.init(
            project="ImageNet-100 Trainium Baseline",
            name=f"{args.dataset}_{args.model}_wd{args.weight_decay}",
            config=vars(args),
        )
        print(f"Logging to wandb run: {run.name}")

    print(
        f"Config: model={args.model}, weight_decay={args.weight_decay}"
    )
    print(
        f"LR config: scheduler={args.lr_scheduler}, warmup_epochs={args.lr_warmup_epochs}, warmup_method={args.lr_warmup_method}"
    )
    print(
        f"Aug config: label_smooth={args.label_smoothing}, mixup={args.mixup_alpha}, cutmix={args.cutmix_alpha}, "
        f"random_erase={args.random_erase}, dropout={args.dropout}, auto_aug={args.auto_augment}"
    )
    if args.output_dir:
        utils.mkdir(args.output_dir)

    # Apply batch_lr_factor scaling
    if args.batch_lr_factor != 1.0:
        original_batch_size = args.batch_size
        original_lr = args.lr
        args.batch_size = int(args.batch_size * args.batch_lr_factor)
        args.lr = args.lr * args.batch_lr_factor
        print(
            f"Applied batch_lr_factor={args.batch_lr_factor}: batch_size {original_batch_size}->{args.batch_size}, lr {original_lr}->{args.lr}"
        )

    utils.init_distributed_mode(args)
    print(args)

    # NEURON EDIT: prefer XLA device when available unless explicitly disabled.
    use_xla = HAS_XLA and not args.no_xla
    if use_xla:
        device = xm.xla_device()
    else:
        requested_device = torch.device(args.device)
        if requested_device.type == "cuda" and not torch.cuda.is_available():
            requested_device = torch.device("cpu")
        if requested_device.type == "mps" and not torch.backends.mps.is_available():
            requested_device = torch.device("cpu")
        device = requested_device
    args.use_xla = is_xla_device(device)
    if args.use_xla and args.distributed:
        raise RuntimeError("XLA/Trainium path currently supports single-process training only")
    if args.use_xla and args.workers > 0:
        print(
            f"Neuron mode: forcing workers=0 (was {args.workers}) to avoid DataLoader multiprocessing hangs"
        )
        args.workers = 0
    if args.use_xla and args.xla_fast_mode:
        if args.xla_bf16:
            os.environ.setdefault("XLA_USE_BF16", "1")
            os.environ.setdefault("NEURON_RT_STOCHASTIC_ROUNDING_EN", "1")
        os.environ.setdefault("NEURON_NUM_RECENT_MODELS_TO_KEEP", "8")
        os.environ.setdefault("NEURON_FUSE_SOFTMAX", "1")
        if args.print_freq < 50:
            args.print_freq = 50
            print("XLA fast mode: raised print_freq to 50 to reduce host sync overhead")
    print(f"Using device: {device}")
    if args.use_xla:
        print(
            "XLA env: "
            f"XLA_USE_BF16={os.environ.get('XLA_USE_BF16', '')}, "
            f"XLA_DOWNCAST_BF16={os.environ.get('XLA_DOWNCAST_BF16', '')}, "
            f"NEURON_RT_STOCHASTIC_ROUNDING_EN={os.environ.get('NEURON_RT_STOCHASTIC_ROUNDING_EN', '')}, "
            f"NEURON_CC_FLAGS={os.environ.get('NEURON_CC_FLAGS', '')}, "
            f"NEURON_COMPILE_CACHE_URL={os.environ.get('NEURON_COMPILE_CACHE_URL', '')}"
        )

    if args.use_deterministic_algorithms:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True

    if args.dataset in ("cifar100", "food101"):
        dataset, dataset_val, dataset_test, train_sampler, val_sampler, test_sampler = load_data(None, None, args)
    else:
        train_dir = os.path.join(args.data_path, "train")
        val_dir = os.path.join(args.data_path, "val")
        dataset, dataset_val, dataset_test, train_sampler, val_sampler, test_sampler = load_data(
            train_dir, val_dir, args
        )

    classes = get_dataset_classes(dataset)
    num_classes = len(classes)
    if args.dataset == "cifar100":
        dataset_type = f"CIFAR-100 train subset ({args.train_label_fraction:.2f} labels)"
    elif args.dataset == "food101":
        dataset_type = f"Food-101 train subset ({args.train_label_fraction:.2f} labels)"
    else:
        dataset_type = "full ImageNet" if args.full_dataset else "ImageNet-100 subset"
    print(f"Training with {num_classes} classes ({dataset_type})")

    teacher_model = None
    if args.pre_train_teacher and args.use_kd:
        raise ValueError("--pre-train-teacher and --use-kd are mutually exclusive")

    if args.use_kd:
        print("Creating teacher model: resnet50")
        teacher_model = torchvision.models.resnet50(
            weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2
        )
        if num_classes != 1000:
            teacher_model.fc = nn.Linear(teacher_model.fc.in_features, num_classes)

        if args.teacher_checkpoint:
            print(f"Loading teacher checkpoint: {args.teacher_checkpoint}")
            checkpoint = torch.load(
                args.teacher_checkpoint, map_location="cpu", weights_only=True
            )
            teacher_state = checkpoint["model"] if "model" in checkpoint else checkpoint
            teacher_model.load_state_dict(teacher_state)
        elif args.dataset != "imagenet":
            raise ValueError(
                "--use-kd with non-imagenet dataset requires --teacher-checkpoint "
                "(generate it first with --pre-train-teacher)."
            )

        teacher_model.to(device)
        teacher_model.eval()
        for p in teacher_model.parameters():
            p.requires_grad = False

    mixup_cutmix = get_mixup_cutmix(
        mixup_alpha=args.mixup_alpha,
        cutmix_alpha=args.cutmix_alpha,
        num_classes=num_classes,
        use_v2=args.use_v2,
    )
    if mixup_cutmix is not None:

        def collate_fn(batch):
            return mixup_cutmix(*default_collate(batch))

    else:
        collate_fn = default_collate

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        drop_last=args.use_xla,
        collate_fn=collate_fn,
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        drop_last=args.use_xla,
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        sampler=test_sampler,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        drop_last=args.use_xla,
    )

    if args.pre_train_teacher:
        pretrain_teacher(
            args,
            device,
            num_classes,
            data_loader,
            data_loader_val,
            data_loader_test,
        )
        return

    print("Creating student model: resnet18")
    model = torchvision.models.get_model("resnet18", weights=args.weights)
    if hasattr(model, "fc") and model.fc.out_features != num_classes:
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    # Apply dropout if specified (add dropout after global average pooling, before final classifier)
    if args.dropout > 0.0:
        # For ResNet models, insert dropout before the final fc layer
        if hasattr(model, "fc"):
            in_features = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Dropout(p=args.dropout), nn.Linear(in_features, num_classes)
            )
            # Keep attributes expected by ResNetPAI wrapper.
            model.fc.in_features = in_features
            model.fc.out_features = num_classes
            print(f"Applied dropout rate: {args.dropout}")

    # Apply stochastic depth if specified (for ResNet models)
    if args.stochastic_depth_prob > 0.0:
        print(
            f"Note: Stochastic depth rate {args.stochastic_depth_prob} specified, but requires model recreation with stochastic_depth parameter"
        )
        print(
            f"Consider using: torchvision.models.resnet18(weights=None, num_classes={num_classes}, stochastic_depth_prob={args.stochastic_depth_prob})"
        )

    # Note on width/depth multipliers
    if args.width_multiplier != 1.0 or args.depth_multiplier != 1.0:
        print(
            f"Note: Width multiplier {args.width_multiplier} and/or depth multiplier {args.depth_multiplier} specified"
        )
        print(
            f"These require custom model creation. Consider using smaller models like resnet18 or using torchvision.models.efficientnet with different variants"
        )

    mode_tag = "KD" if args.use_kd else "CE"
    save_name = f"{args.dataset}_{mode_tag}_{args.model}_wd{args.weight_decay}"
    if run is not None:
        run.name = save_name

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_name_with_timestamp = f"{save_name}_{timestamp}"

    # ORDER MATTERS ON XLA: move model before optimizer creation.
    model.to(device)

    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    custom_keys_weight_decay = []
    if args.bias_weight_decay is not None:
        custom_keys_weight_decay.append(("bias", args.bias_weight_decay))
    if args.transformer_embedding_decay is not None:
        for key in [
            "class_token",
            "position_embedding",
            "relative_position_bias_table",
        ]:
            custom_keys_weight_decay.append((key, args.transformer_embedding_decay))

    # Create optimizer and scheduler
    optimizer, lr_scheduler = create_optimizer_and_scheduler(
        model, args, custom_keys_weight_decay
    )
    args.lr = (
        args.lr * 10
    )  # Increase LR after restructuring to help adapt to new architecture
    scaler = torch.cuda.amp.GradScaler() if args.amp and device.type == "cuda" else None

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    model_ema = None
    if args.model_ema:
        # Decay adjustment that aims to keep the decay independent of other hyper-parameters originally proposed at:
        # https://github.com/facebookresearch/pycls/blob/f8cd9627/pycls/core/net.py#L123
        #
        # total_ema_updates = (Dataset_size / n_GPUs) * epochs / (batch_size_per_gpu * EMA_steps)
        # We consider constant = Dataset_size for a given dataset/setup and omit it. Thus:
        # adjust = 1 / total_ema_updates ~= n_GPUs * batch_size_per_gpu * EMA_steps / epochs
        adjust = args.world_size * args.batch_size * args.model_ema_steps / args.epochs
        alpha = 1.0 - args.model_ema_decay
        alpha = min(1.0, alpha * adjust)
        model_ema = utils.ExponentialMovingAverage(
            model_without_ddp, device=device, decay=1.0 - alpha
        )

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        model_without_ddp.load_state_dict(checkpoint["model"])
        if not args.test_only:
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        args.start_epoch = checkpoint["epoch"] + 1
        if model_ema:
            model_ema.load_state_dict(checkpoint["model_ema"])
        if scaler:
            scaler.load_state_dict(checkpoint["scaler"])

    if args.test_only:
        # We disable the cudnn benchmarking because it can noticeably affect the accuracy
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if model_ema:
            test(model_ema, criterion, data_loader_test, device=device, log_suffix="EMA")
            evaluate(
                model_ema, criterion, data_loader_val, device=device, log_suffix="EMA"
            )
        else:
            test(model, criterion, data_loader_test, device=device)
            evaluate(model, criterion, data_loader_val, device=device)
        return

    print("Start training")
    start_time = time.time()

    # Initialize tracking variables for wandb logging
    max_val_acc1 = 0
    max_test_acc1 = 0
    max_params = 0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_one_epoch(
            model,
            teacher_model,
            criterion,
            optimizer,
            data_loader,
            device,
            epoch,
            args,
            model_ema,
            scaler,
        )
        test_acc1 = test(model, criterion, data_loader_test, device=device)
        acc1 = evaluate(model, criterion, data_loader_val, device=device)

        # Update max values
        if acc1 > max_val_acc1:
            max_val_acc1 = acc1
            max_test_acc1 = test_acc1
            max_params = count_model_params(model)

        # Log to wandb
        if run is not None:
            run.log(
                {
                    "ValAcc": acc1,
                    "TestAcc": test_acc1,
                    "Param Count": count_model_params(model),
                    "epoch": epoch,
                }
            )

        if args.lr_scheduler == "reducelronplateau":
            lr_scheduler.step(acc1)
        else:
            lr_scheduler.step()

        if model_ema:
            test(
                model_ema,
                criterion,
                data_loader_test,
                device=device,
                log_suffix="EMA",
            )
            evaluate(
                model_ema, criterion, data_loader_val, device=device, log_suffix="EMA"
            )

        if args.output_dir:
            checkpoint = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch,
                "args": args,
            }
            if model_ema:
                checkpoint["model_ema"] = model_ema.state_dict()
            if scaler:
                checkpoint["scaler"] = scaler.state_dict()
            utils.save_on_master(
                checkpoint, os.path.join(args.output_dir, f"model_{epoch}.pth")
            )
            utils.save_on_master(
                checkpoint, os.path.join(args.output_dir, "checkpoint.pth")
            )

    if run is not None:
        run.log(
            {
                "Final Max Val": max_val_acc1,
                "Final Max Test": max_test_acc1,
                "Final Param Count": max_params,
            }
        )
    print("Final Param Count:", count_model_params(model))
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(
        description="PyTorch Classification Training (Trainium baseline)",
        add_help=add_help,
    )

    parser.add_argument(
        "--data-path",
        default="./Datasets",
        type=str,
        help="dataset root path",
    )
    parser.add_argument(
        "--dataset",
        default="food101",
        type=str,
        choices=["cifar100", "food101", "imagenet"],
        help="dataset to train on",
    )
    parser.add_argument(
        "--train-label-fraction",
        default=0.25,
        type=float,
        help="fraction of labeled training data to use (stratified per class)",
    )
    parser.add_argument(
        "--label-subset-seed",
        default=42,
        type=int,
        help="random seed for stratified label subset selection",
    )
    parser.add_argument(
        "--download-cifar",
        action="store_true",
        help="download CIFAR-100 to --data-path if missing",
    )
    parser.add_argument(
        "--download-food101",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="download Food-101 to --data-path if missing",
    )
    parser.add_argument(
        "--use-kd",
        action="store_true",
        help="enable KD loss; for non-imagenet datasets also pass --teacher-checkpoint",
    )
    parser.add_argument(
        "--pre-train-teacher",
        action="store_true",
        help="train a resnet50 teacher and save checkpoint for later KD",
    )
    parser.add_argument(
        "--teacher-checkpoint",
        default="",
        type=str,
        help="teacher checkpoint path to load for KD, or save path when using --pre-train-teacher",
    )
    parser.add_argument(
        "--val-test-split-seed",
        default=42,
        type=int,
        help="random seed for class-balanced validation/test split",
    )
    parser.add_argument("--model", default="resnet18", type=str, help="model name")
    parser.add_argument(
        "--device",
        default="cuda",
        type=str,
        help="device (Use cuda or cpu Default: cuda)",
    )
    parser.add_argument(
        "--no-xla",
        action="store_true",
        default=False,
        help="disable XLA/Neuron even when torch_xla is available",
    )
    parser.add_argument(
        "--xla-fast-mode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable Trainium speed-oriented defaults (env vars + lower logging overhead)",
    )
    parser.add_argument(
        "--xla-bf16",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable BF16 environment defaults when running on Neuron",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        default=32,
        type=int,
        help="images per gpu, the total batch size is $NGPU x batch_size",
    )
    parser.add_argument(
        "--batch-lr-factor",
        default=1.0,
        type=float,
        help="factor to scale batch size and learning rate (e.g., 0.5 halves batch size and scales lr accordingly)",
    )
    parser.add_argument(
        "--epochs",
        default=90,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "-j",
        "--workers",
        default=8,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 16)",
    )
    parser.add_argument("--opt", default="sgd", type=str, help="optimizer")
    parser.add_argument(
        "--lr", default=0.0125, type=float, help="initial learning rate"
    )
    parser.add_argument(
        "--momentum", default=0.9, type=float, metavar="M", help="momentum"
    )
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-3,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-3)",
        dest="weight_decay",
    )
    parser.add_argument(
        "--norm-weight-decay",
        default=None,
        type=float,
        help="weight decay for Normalization layers (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--bias-weight-decay",
        default=None,
        type=float,
        help="weight decay for bias parameters of all layers (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--transformer-embedding-decay",
        default=None,
        type=float,
        help="weight decay for embedding parameters for vision transformer models (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--label-smoothing",
        default=0.1,
        type=float,
        help="label smoothing (default: 0.1)",
        dest="label_smoothing",
    )
    parser.add_argument(
        "--mixup-alpha", default=0.2, type=float, help="mixup alpha (default: 0.2)"
    )
    parser.add_argument(
        "--cutmix-alpha", default=0.6, type=float, help="cutmix alpha (default: 0.6)"
    )
    parser.add_argument(
        "--lr-scheduler",
        default="steplr",
        type=str,
        help="the lr scheduler (default: steplr)",
    )
    parser.add_argument(
        "--lr-warmup-epochs",
        default=0,
        type=int,
        help="the number of epochs to warmup (default: 0)",
    )
    parser.add_argument(
        "--lr-warmup-method",
        default="constant",
        type=str,
        help="the warmup method (default: constant)",
    )
    parser.add_argument(
        "--lr-warmup-decay", default=0.01, type=float, help="the decay for lr"
    )
    parser.add_argument(
        "--lr-step-size",
        default=30,
        type=int,
        help="decrease lr every step-size epochs",
    )
    parser.add_argument(
        "--lr-gamma",
        default=0.1,
        type=float,
        help="decrease lr by a factor of lr-gamma",
    )
    parser.add_argument(
        "--lr-min",
        default=0.0,
        type=float,
        help="minimum lr of lr schedule (default: 0.0)",
    )
    parser.add_argument("--print-freq", default=10, type=int, help="print frequency")
    parser.add_argument(
        "--output-dir", default=None, type=str, help="path to save outputs"
    )
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")
    parser.add_argument(
        "--start-epoch", default=0, type=int, metavar="N", help="start epoch"
    )
    parser.add_argument(
        "--cache-dataset",
        dest="cache_dataset",
        help="Cache the datasets for quicker initialization. It also serializes the transforms",
        action="store_true",
    )
    parser.add_argument(
        "--sync-bn",
        dest="sync_bn",
        help="Use sync batch norm",
        action="store_true",
    )
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
    parser.add_argument(
        "--auto-augment",
        default="ta_wide",
        type=lambda x: None if x == "None" else x,
        help="auto augment policy (default: ta_wide)",
    )
    parser.add_argument(
        "--ra-magnitude", default=9, type=int, help="magnitude of auto augment policy"
    )
    parser.add_argument(
        "--augmix-severity", default=3, type=int, help="severity of augmix policy"
    )
    parser.add_argument(
        "--random-erase",
        default=0.2,
        type=float,
        help="random erasing probability (default: 0.2)",
    )

    # Regularization parameters to reduce overfitting (train-val gap)
    parser.add_argument(
        "--dropout",
        default=0.2,
        type=float,
        help="dropout rate (default: 0.2)",
    )
    parser.add_argument(
        "--width-multiplier",
        default=1.0,
        type=float,
        help="network width multiplier to reduce capacity (default: 1.0, full width)",
    )
    parser.add_argument(
        "--depth-multiplier",
        default=1.0,
        type=float,
        help="network depth multiplier to reduce capacity (default: 1.0, full depth)",
    )
    parser.add_argument(
        "--stochastic-depth-prob",
        default=0.0,
        type=float,
        help="stochastic depth drop probability for ResNet (default: 0.0, no stochastic depth)",
    )

    # Mixed precision training parameters
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Use torch.cuda.amp for mixed precision training",
    )

    # distributed training parameters
    parser.add_argument(
        "--world-size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument(
        "--dist-url",
        default="env://",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--model-ema",
        action="store_true",
        help="enable tracking Exponential Moving Average of model parameters",
    )
    parser.add_argument(
        "--model-ema-steps",
        type=int,
        default=32,
        help="the number of iterations that controls how often to update the EMA model (default: 32)",
    )
    parser.add_argument(
        "--model-ema-decay",
        type=float,
        default=0.99998,
        help="decay factor for Exponential Moving Average of model parameters (default: 0.99998)",
    )
    parser.add_argument(
        "--use-deterministic-algorithms",
        action="store_true",
        help="Forces the use of deterministic algorithms only.",
    )
    parser.add_argument(
        "--interpolation",
        default="bilinear",
        type=str,
        help="the interpolation method (default: bilinear)",
    )
    # Half resolution defaults (128 instead of 256, 112 instead of 224)
    parser.add_argument(
        "--val-resize-size",
        default=256,
        type=int,
        help="the resize size used for validation (default: 256)",
    )
    parser.add_argument(
        "--val-crop-size",
        default=224,
        type=int,
        help="the central crop size used for validation (default: 224)",
    )
    parser.add_argument(
        "--train-crop-size",
        default=224,
        type=int,
        help="the random crop size used for training (default: 224)",
    )
    parser.add_argument(
        "--convert-count", default=0, type=int, help="total number of layers to convert"
    )
    parser.add_argument(
        "--clip-grad-norm",
        default=None,
        type=float,
        help="the maximum gradient norm (default None)",
    )
    parser.add_argument(
        "--ra-sampler",
        action="store_true",
        help="whether to use Repeated Augmentation in training",
    )
    parser.add_argument(
        "--ra-reps",
        default=3,
        type=int,
        help="number of repetitions for Repeated Augmentation (default: 3)",
    )
    parser.add_argument(
        "--weights",
        default="ResNet18_Weights.IMAGENET1K_V1",
        type=str,
        help="the weights enum name to load",
    )
    parser.add_argument(
        "--backend",
        default="PIL",
        type=str.lower,
        help="PIL or tensor - case insensitive",
    )
    parser.add_argument("--use-v2", action="store_true", help="Use V2 transforms")
    parser.add_argument(
        "--full-dataset",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Use full ImageNet-1000 instead of ImageNet-100 subset (default: True)",
    )

    # Wandb logging
    parser.add_argument("--use-wandb", action="store_true", help="Enable wandb logging")

    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
