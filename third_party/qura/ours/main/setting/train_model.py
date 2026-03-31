import argparse
import os
import random
import subprocess
from pathlib import Path

import numpy as np
import timm
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, MultiStepLR, SequentialLR

from dataset.dataset import Cifar10, Cifar100, Minst, Tiny
from model.resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from model.vgg import vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn


FILE_PATH = Path(__file__).resolve()
DIRECTORY_PATH = FILE_PATH.parent
MODEL_DIR = (DIRECTORY_PATH / "../model").resolve()


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch Model Training")
    parser.add_argument("--l_r", default=None, type=float, help="Learning rate. Defaults are model-specific.")
    parser.add_argument("--resume", action="store_true", help="Resume from best checkpoint.")
    parser.add_argument("--model", default="resnet18", type=str, help="Model type.")
    parser.add_argument("--dataset", default="cifar10", type=str, help="Dataset type.")
    parser.add_argument("--epochs", default=None, type=int, help="Training epochs.")
    parser.add_argument("--batch_size", default=None, type=int, help="Batch size override.")
    parser.add_argument("--num_workers", default=None, type=int, help="Dataloader workers override.")
    parser.add_argument("--gpu", default=None, type=int, help="Explicit GPU index. If omitted, pick the least-used GPU.")
    parser.add_argument("--seed", default=1005, type=int, help="Random seed.")
    parser.add_argument("--warmup_epochs", default=5, type=int, help="Warmup epochs for ViT.")
    parser.add_argument("--disable_pretrained", action="store_true", help="Disable ImageNet pretrained init for ViT.")
    parser.add_argument("--log_interval", default=20, type=int, help="Steps between progress logs.")
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def query_gpus():
    gpu_info = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,nounits,noheader"]
    )
    rows = []
    for line in gpu_info.decode("utf-8").strip().splitlines():
        index_str, memory_str = line.split(", ")
        rows.append((int(index_str), int(memory_str)))
    return rows


def resolve_device(args):
    if not torch.cuda.is_available():
        print("CUDA not available. Using CPU.")
        return torch.device("cpu")

    if args.gpu is not None:
        device = torch.device(f"cuda:{args.gpu}")
        torch.cuda.set_device(device)
        print(f"Using GPU: {args.gpu}")
        return device

    gpu_rows = query_gpus()
    best_gpu, best_mem = min(gpu_rows, key=lambda item: item[1])
    device = torch.device(f"cuda:{best_gpu}")
    torch.cuda.set_device(device)
    print(f"Using GPU: {best_gpu} (memory.used={best_mem} MiB)")
    return device


def resolve_dataset_options(args):
    options = {
        "batch_size": 128,
        "num_workers": 16,
        "image_size": None,
        "class_num": None,
    }

    if args.dataset == "minst":
        options["class_num"] = 10
        if args.model == "vit":
            options["image_size"] = 224
    elif args.dataset == "cifar10":
        options["class_num"] = 10
        if args.model == "vit":
            options["batch_size"] = 64
            options["num_workers"] = 4
            options["image_size"] = 224
    elif args.dataset == "cifar100":
        options["class_num"] = 100
        if args.model == "vit":
            options["batch_size"] = 128
            options["num_workers"] = 8
            options["image_size"] = 224
    elif args.dataset == "tiny_imagenet":
        options["class_num"] = 200
        if args.model == "vit":
            options["batch_size"] = 64
            options["num_workers"] = 4
            options["image_size"] = 224
    else:
        raise ValueError(f"Unsupported dataset type: {args.dataset}")

    if args.batch_size is not None:
        options["batch_size"] = args.batch_size
    if args.num_workers is not None:
        options["num_workers"] = args.num_workers

    return options


def build_dataloaders(args):
    data_path = DIRECTORY_PATH / "../data"
    opts = resolve_dataset_options(args)
    common_kwargs = {
        "batch_size": opts["batch_size"],
        "num_workers": opts["num_workers"],
    }

    if args.dataset == "minst":
        dataset = Minst(str(data_path), image_size=opts["image_size"], **common_kwargs) if args.model == "vit" else Minst(str(data_path), **common_kwargs)
    elif args.dataset == "cifar10":
        dataset = Cifar10(str(data_path), image_size=opts["image_size"], **common_kwargs) if args.model == "vit" else Cifar10(str(data_path), **common_kwargs)
    elif args.dataset == "cifar100":
        dataset = Cifar100(str(data_path), image_size=opts["image_size"], **common_kwargs) if args.model == "vit" else Cifar100(str(data_path), **common_kwargs)
    elif args.dataset == "tiny_imagenet":
        tiny_root = str((DIRECTORY_PATH / "../data/tiny-imagenet-200").resolve())
        dataset = Tiny(tiny_root, image_size=opts["image_size"], **common_kwargs) if args.model == "vit" else Tiny(tiny_root, **common_kwargs)
    else:
        raise ValueError(f"Unsupported dataset type: {args.dataset}")

    train_loader, val_loader, _, _ = dataset.get_loader(normal=True)
    return train_loader, val_loader, opts["class_num"]


def build_model(args, class_num):
    print(f"==> Building {args.model} model..")

    if args.model == "vgg16":
        model = vgg16_bn(num_class=class_num, input_size=64 if class_num == 200 else 32)
    elif args.model == "vgg11":
        model = vgg11_bn(num_class=class_num, input_size=64 if class_num == 200 else 32)
    elif args.model == "vgg13":
        model = vgg13_bn(num_class=class_num, input_size=64 if class_num == 200 else 32)
    elif args.model == "vgg19":
        model = vgg19_bn(num_class=class_num, input_size=64 if class_num == 200 else 32)
    elif args.model == "resnet18":
        model = ResNet18(num_classes=class_num)
    elif args.model == "resnet34":
        model = ResNet34(num_classes=class_num)
    elif args.model == "resnet50":
        model = ResNet50(num_classes=class_num)
    elif args.model == "resnet101":
        model = ResNet101(num_classes=class_num)
    elif args.model == "resnet152":
        model = ResNet152(num_classes=class_num)
    elif args.model == "vit":
        model = timm.create_model(
            "vit_tiny_patch16_224",
            pretrained=not args.disable_pretrained,
            num_classes=class_num,
        )
    else:
        raise ValueError(f"Unsupported model type: {args.model}")

    return model


def resolve_training_options(args):
    if args.model == "vit":
        lr = args.l_r if args.l_r is not None else 1e-4
        epochs = args.epochs if args.epochs is not None else 50
    else:
        lr = args.l_r if args.l_r is not None else 1e-2
        epochs = args.epochs if args.epochs is not None else 100
    return lr, epochs


def build_optimizer_and_scheduler(args, model, epochs, lr):
    if args.model == "vit":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
        if args.warmup_epochs > 0 and epochs > args.warmup_epochs:
            warmup = LinearLR(optimizer, start_factor=0.1, total_iters=args.warmup_epochs)
            cosine = CosineAnnealingLR(optimizer, T_max=epochs - args.warmup_epochs)
            scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[args.warmup_epochs])
        else:
            scheduler = CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        scheduler = MultiStepLR(optimizer, milestones=[30, 60, 80], gamma=0.2)
    return optimizer, scheduler


def checkpoint_path(args):
    return MODEL_DIR / f"{args.model}+{args.dataset}.pth"


def train_one_epoch(model, loader, criterion, optimizer, device, epoch, log_interval):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * targets.size(0)
        total_correct += outputs.argmax(1).eq(targets).sum().item()
        total_seen += targets.size(0)

        if log_interval > 0 and (batch_idx + 1) % log_interval == 0:
            avg_loss = total_loss / total_seen
            avg_acc = 100.0 * total_correct / total_seen
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch: {epoch:02d} | Train Batch: {batch_idx + 1:03d}/{len(loader):03d} "
                f"| LR: {lr:.6f} | Loss: {avg_loss:.4f} | Acc: {avg_acc:.2f}%"
            )

    return total_loss / max(1, total_seen), 100.0 * total_correct / max(1, total_seen)


def evaluate(model, loader, criterion, device, epoch, log_interval):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item() * targets.size(0)
            total_correct += outputs.argmax(1).eq(targets).sum().item()
            total_seen += targets.size(0)

            if log_interval > 0 and (batch_idx + 1) % log_interval == 0:
                avg_loss = total_loss / total_seen
                avg_acc = 100.0 * total_correct / total_seen
                print(
                    f"Epoch: {epoch:02d} | Val Batch: {batch_idx + 1:03d}/{len(loader):03d} "
                    f"| Loss: {avg_loss:.4f} | Acc: {avg_acc:.2f}%"
                )

    return total_loss / max(1, total_seen), 100.0 * total_correct / max(1, total_seen)


def save_checkpoint(path, model, acc, epoch, args):
    state = {
        "model": model.state_dict(),
        "acc": acc,
        "epoch": epoch,
        "args": vars(args),
    }
    torch.save(state, path)


def main():
    args = parse_args()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    cudnn.benchmark = True

    device = resolve_device(args)
    train_loader, val_loader, class_num = build_dataloaders(args)
    model = build_model(args, class_num).to(device)

    lr, epochs = resolve_training_options(args)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1 if args.model == "vit" else 0.0)
    optimizer, scheduler = build_optimizer_and_scheduler(args, model, epochs, lr)

    ckpt_path = checkpoint_path(args)
    best_acc = 0.0
    start_epoch = 0

    if args.resume and ckpt_path.exists():
        print("==> Resuming from checkpoint..")
        checkpoint = torch.load(str(ckpt_path), map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=False)
        best_acc = float(checkpoint.get("acc", 0.0))
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        print(f"| Best Acc: {best_acc:.2f}% | Resume Epoch: {start_epoch} |")

    print(f"==> Preparing {args.dataset} dataset..")
    print(
        f"==> Training config | model={args.model} dataset={args.dataset} "
        f"epochs={epochs} batch_size={train_loader.batch_size} workers={train_loader.num_workers} lr={lr}"
    )
    print("==> Start training process..")

    for epoch in range(start_epoch, epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, args.log_interval
        )
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device, epoch, args.log_interval
        )
        scheduler.step()

        print(
            f"Epoch Summary: {epoch:02d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% "
            f"| Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | Best: {best_acc:.2f}%"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            print("Saving..")
            save_checkpoint(ckpt_path, model, val_acc, epoch, args)

    print(f"Training complete. Best Acc: {best_acc:.2f}%")


if __name__ == "__main__":
    main()
