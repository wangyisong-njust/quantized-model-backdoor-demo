"""
Batch ImageNet evaluation for the ViT-B/16 QURA demo checkpoints.

Computes a larger-sample version of the demo metrics:
  - FP32 clean top-1
  - FP32 + trigger target-hit rate
  - INT8-QURA clean top-1
  - INT8-QURA + trigger target-hit rate
  - INT8-QURA + trigger + RegionDrop target-hit rate

The trigger is pasted after Resize(256) -> CenterCrop(224) -> ToTensor and
before normalization, matching the demo/training pipeline.
"""

import argparse
import importlib
import json
import random
import sys
import time
from contextlib import contextmanager
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import timm
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

REPO = Path(__file__).resolve().parent.parent
QURA_ROOT = REPO / "third_party/qura/ours/main"

sys.path.insert(0, str(QURA_ROOT))
sys.path.insert(0, str(REPO))

from defenses.regiondrop.region_detector import (  # noqa: E402
    AttentionHook,
    apply_region_mask,
    multi_scale_region_search,
    topk_patch_search,
)
from utils.qura_checkpoint import load_quant_checkpoint  # noqa: E402

DEFAULT_TRIGGER_FILE = REPO / "third_party/qura/ours/main/model/vit_base+imagenet.trigger.pt"
DEFAULT_QUANT_MODEL = REPO / "third_party/qura/ours/main/model/vit_base+imagenet.quant_bd_1_t0.pth"
FIXEDPOS_QUANT_MODEL = REPO / "third_party/qura/ours/main/model/vit_base+imagenet.quant_bd_1_t0_fixedpos.pth"
DEFAULT_QUANT_CONFIG = REPO / "third_party/qura/ours/main/configs/cv_vit_base_imagenet_8_8_bd.yaml"
DEFAULT_IMAGENET_VAL = Path("/home/kaixin/ssd/imagenet/val")
DEFAULT_OUT_DIR = REPO / "outputs/imagenet_vit_qura"
DEFAULT_TRIGGER_CACHE_DIR = DEFAULT_OUT_DIR / "generated_triggers"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def parse_args():
    parser = argparse.ArgumentParser(description="Batch ImageNet evaluation for ViT QURA checkpoints.")
    parser.add_argument("--variant", choices=["default", "fixedpos"], default="fixedpos")
    parser.add_argument("--quant_model", default=None)
    parser.add_argument("--quant_config", default=str(DEFAULT_QUANT_CONFIG))
    parser.add_argument("--trigger_file", default=str(DEFAULT_TRIGGER_FILE))
    parser.add_argument("--imagenet_val", default=str(DEFAULT_IMAGENET_VAL))
    parser.add_argument("--trigger_source", choices=["file", "generated"], default="generated")
    parser.add_argument("--trigger_cache", default=None)
    parser.add_argument("--force_regenerate_trigger", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bd_target", type=int, default=0)
    parser.add_argument(
        "--attn_reduce",
        choices=["mean", "sum", "max", "std", "mean_plus_std", "vote_top1"],
        default="mean",
    )
    parser.add_argument("--mask_mode", choices=["blur", "zero"], default="blur")
    parser.add_argument("--blur_kernel", type=int, default=31)
    parser.add_argument("--blur_sigma", type=float, default=4.0)
    parser.add_argument("--region_topk", type=int, default=1)
    parser.add_argument("--output_dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--output_name", default=None)
    return parser.parse_args()


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


@contextmanager
def qura_import_path():
    qura_path = str(QURA_ROOT)
    qura_repo_path = str(REPO / "third_party/qura")
    repo_path = str(REPO)
    original = list(sys.path)
    sys.path = [
        qura_path,
        qura_repo_path,
        *[p for p in sys.path if p not in ("", qura_path, qura_repo_path, repo_path)],
    ]
    try:
        yield
    finally:
        sys.path = original


def build_transform():
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_loader(imagenet_val: Path, max_samples: int, seed: int, batch_size: int, num_workers: int):
    dataset = datasets.ImageFolder(str(imagenet_val), transform=build_transform())
    if max_samples > 0 and max_samples < len(dataset):
        rng = random.Random(seed)
        indices = list(range(len(dataset)))
        rng.shuffle(indices)
        dataset = Subset(dataset, indices[:max_samples])
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def load_trigger(trigger_file: Path) -> torch.Tensor:
    payload = torch.load(str(trigger_file), map_location="cpu")
    trigger = payload["trigger"] if isinstance(payload, dict) and "trigger" in payload else payload
    if trigger.ndim == 4:
        trigger = trigger[0]
    if trigger.ndim != 3:
        raise ValueError(f"Unexpected trigger shape: {tuple(trigger.shape)}")
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return ((trigger.float().clamp(0, 1) - mean) / std).float()


def resolve_imagenet_root(imagenet_val: Path) -> Path:
    if imagenet_val.name == "val":
        return imagenet_val.parent
    return imagenet_val


def resolve_trigger_cache_path(trigger_cache: str, quant_config_path: Path, imagenet_root: Path, bd_target: int):
    if trigger_cache:
        return Path(trigger_cache)
    cfg = OmegaConf.load(quant_config_path)
    seed = int(cfg.process.seed)
    pattern = getattr(cfg.dataset, "pattern", "stage2")
    pos_mode = getattr(cfg.dataset, "pos_mode", "fixed")
    cache_dir = DEFAULT_TRIGGER_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"vit_base_imagenet_t{bd_target}_{pattern}_{pos_mode}_seed{seed}.pt"


def generate_trigger(
    quant_config_path: Path,
    imagenet_root: Path,
    device: torch.device,
    bd_target: int,
    cache_path: Path,
):
    cfg = OmegaConf.load(quant_config_path)
    seed_all(int(cfg.process.seed))

    with qura_import_path():
        setting_config = importlib.import_module("setting.config")
        dataset_module = importlib.import_module("setting.dataset.dataset")

        model = setting_config.get_model("vit_base", 1000)
        data = dataset_module.ImageNetWrapper(
            str(imagenet_root),
            batch_size=cfg.dataset.batch_size,
            num_workers=cfg.dataset.num_workers,
            target=bd_target,
            pattern=cfg.dataset.pattern,
            quant=True,
        )
        train_loader, _, _, _ = data.get_loader()
        trigger, _ = setting_config.build_cv_trigger(
            "vit_base",
            "imagenet",
            model,
            train_loader,
            data,
            bd_target,
            cfg.dataset.pattern,
            cfg.quantize.cali_batchsize,
            device,
            trigger_policy=getattr(cfg.dataset, "trigger_policy", "relative"),
            trigger_base_size=getattr(cfg.dataset, "trigger_base_size", 12),
            trigger_base_image_size=getattr(cfg.dataset, "trigger_base_image_size", 224),
            random_pos=getattr(cfg.dataset, "random_pos", False),
            size_range=tuple(getattr(cfg.dataset, "trigger_size_jitter", [0.75, 1.5])),
            pos_mode=getattr(cfg.dataset, "pos_mode", "fixed"),
            jitter_px=getattr(cfg.dataset, "jitter_px", 16),
        )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"trigger": trigger.detach().cpu()}, cache_path)
    return cache_path


def load_eval_trigger(
    trigger_source: str,
    trigger_file: Path,
    quant_config_path: Path,
    imagenet_root: Path,
    device: torch.device,
    bd_target: int,
    trigger_cache: str = None,
    force_regenerate_trigger: bool = False,
):
    if trigger_source == "file":
        return load_trigger(trigger_file), trigger_file

    cache_path = resolve_trigger_cache_path(trigger_cache, quant_config_path, imagenet_root, bd_target)
    if force_regenerate_trigger or not cache_path.exists():
        cache_path = generate_trigger(
            quant_config_path=quant_config_path,
            imagenet_root=imagenet_root,
            device=device,
            bd_target=bd_target,
            cache_path=cache_path,
        )
    return load_trigger(cache_path), cache_path


def load_fp32(device):
    model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=1000)
    return model.to(device).eval()


def load_qura(device, quant_model_path: Path, quant_config_path: Path):
    from mqbench.prepare_by_platform import BackendType, prepare_by_platform
    from mqbench.utils.state import enable_quantization

    cfg = OmegaConf.load(quant_config_path)
    extra = OmegaConf.to_container(cfg.extra_prepare_dict, resolve=True)
    model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=1000)
    model = prepare_by_platform(model, BackendType.Academic, extra)
    _, missing, unexpected, restored_adaround, recovered_soft = load_quant_checkpoint(
        model,
        quant_model_path,
        strict=False,
        restore_adaround=False,
    )
    enable_quantization(model)
    ignored_alpha = [key for key in unexpected if key.endswith(".alpha")]
    remaining_unexpected = [key for key in unexpected if not key.endswith(".alpha")]
    if ignored_alpha:
        print(f"[load_qura] ignored stale AdaRound alpha tensors: {len(ignored_alpha)}")
    if remaining_unexpected:
        print(f"[load_qura] unexpected keys: {len(remaining_unexpected)}")
    if restored_adaround:
        print(f"[load_qura] restored AdaRound quantizers: {len(restored_adaround)}")
    if recovered_soft:
        print(f"[load_qura] recovered soft weights: {len(recovered_soft)}")
    if missing:
        print(f"[load_qura] missing keys: {len(missing)}")
    return model.to(device).eval()


def apply_trigger(images: torch.Tensor, trigger: torch.Tensor):
    patched = images.clone()
    trigger = trigger.to(images.device, images.dtype)
    _, h, w = trigger.shape
    patched[:, :, -h:, -w:] = trigger
    return patched


@torch.no_grad()
def evaluate_clean_accuracy(model, loader, device):
    total = 0
    correct = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        pred = model(images).argmax(dim=1)
        total += labels.size(0)
        correct += (pred == labels).sum().item()
    return {"top1": correct / max(total, 1), "correct": correct, "total": total}


@torch.no_grad()
def evaluate_trigger_target_rate(model, loader, device, trigger, bd_target):
    total = 0
    target_hits = 0
    clean_label_hits = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        valid = labels != bd_target
        if valid.sum().item() == 0:
            continue
        images = apply_trigger(images[valid], trigger)
        labels = labels[valid]
        pred = model(images).argmax(dim=1)
        total += labels.size(0)
        target_hits += (pred == bd_target).sum().item()
        clean_label_hits += (pred == labels).sum().item()
    return {
        "target_rate": target_hits / max(total, 1),
        "target_hits": target_hits,
        "total": total,
        "top1": clean_label_hits / max(total, 1),
    }


@torch.no_grad()
def evaluate_trigger_with_regiondrop(
    model,
    loader,
    device,
    trigger,
    bd_target,
    attn_reduce,
    mask_mode,
    blur_kernel,
    blur_sigma,
    region_topk,
):
    total = 0
    target_hits = 0
    clean_label_hits = 0
    window_counter = Counter()
    hook = AttentionHook(model)

    try:
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            valid = labels != bd_target
            if valid.sum().item() == 0:
                continue
            images = apply_trigger(images[valid], trigger)
            labels = labels[valid]

            for idx in range(images.size(0)):
                img = images[idx : idx + 1]
                _ = model(img)
                attn_map = hook.get_cls_attention_map(reduce=attn_reduce)
                if region_topk > 1:
                    results = topk_patch_search(attn_map, k=region_topk)
                else:
                    results = [multi_scale_region_search(attn_map)]
                defended = img
                for result in results:
                    window_counter[f"{result.window_h}x{result.window_w}"] += 1
                    defended = apply_region_mask(
                        defended,
                        result.pixel_bbox,
                        mode=mask_mode,
                        blur_kernel_size=blur_kernel,
                        blur_sigma=blur_sigma,
                    )
                pred = model(defended).argmax(dim=1).item()
                total += 1
                target_hits += int(pred == bd_target)
                clean_label_hits += int(pred == labels[idx].item())
    finally:
        hook.remove()

    return {
        "target_rate": target_hits / max(total, 1),
        "target_hits": target_hits,
        "total": total,
        "top1": clean_label_hits / max(total, 1),
        "window_distribution": dict(window_counter),
    }


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    quant_model_path = Path(args.quant_model) if args.quant_model else (
        FIXEDPOS_QUANT_MODEL if args.variant == "fixedpos" else DEFAULT_QUANT_MODEL
    )
    quant_config_path = Path(args.quant_config)
    trigger_file = Path(args.trigger_file)
    imagenet_val = Path(args.imagenet_val)
    imagenet_root = resolve_imagenet_root(imagenet_val)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_name = args.output_name or f"metrics_{args.variant}_{args.max_samples}.json"
    output_path = output_dir / output_name

    print("Config:")
    print(f"  variant      = {args.variant}")
    print(f"  quant_model  = {quant_model_path}")
    print(f"  quant_config = {quant_config_path}")
    print(f"  trigger_file = {trigger_file}")
    print(f"  imagenet_val = {imagenet_val}")
    print(f"  trigger_src  = {args.trigger_source}")
    print(f"  imagenet_root= {imagenet_root}")
    print(f"  device       = {device}")
    print(f"  max_samples  = {args.max_samples}")
    print(f"  batch_size   = {args.batch_size}")
    print(f"  attn_reduce  = {args.attn_reduce}")
    print(f"  mask_mode    = {args.mask_mode}")
    print(f"  blur_kernel  = {args.blur_kernel}")
    print(f"  blur_sigma   = {args.blur_sigma}")
    print(f"  region_topk  = {args.region_topk}")
    print(f"  output_path  = {output_path}")

    loader = build_loader(
        imagenet_val=imagenet_val,
        max_samples=args.max_samples,
        seed=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    trigger, trigger_source_path = load_eval_trigger(
        trigger_source=args.trigger_source,
        trigger_file=trigger_file,
        quant_config_path=quant_config_path,
        imagenet_root=imagenet_root,
        device=device,
        bd_target=args.bd_target,
        trigger_cache=args.trigger_cache,
        force_regenerate_trigger=args.force_regenerate_trigger,
    )

    t0 = time.time()
    print("\nLoading FP32 ViT-B/16 ...")
    fp32 = load_fp32(device)
    print("Loading INT8-QURA ViT-B/16 ...")
    qura = load_qura(device, quant_model_path, quant_config_path)

    print("\n[1/5] FP32 clean accuracy ...")
    fp32_clean = evaluate_clean_accuracy(fp32, loader, device)
    print(f"  FP32 clean top1 = {fp32_clean['top1'] * 100:.2f}%")

    print("\n[2/5] FP32 trigger target rate ...")
    fp32_trigger = evaluate_trigger_target_rate(fp32, loader, device, trigger, args.bd_target)
    print(f"  FP32 trigger ASR = {fp32_trigger['target_rate'] * 100:.2f}%")

    print("\n[3/5] INT8 clean accuracy ...")
    int8_clean = evaluate_clean_accuracy(qura, loader, device)
    print(f"  INT8 clean top1 = {int8_clean['top1'] * 100:.2f}%")

    print("\n[4/5] INT8 trigger target rate ...")
    int8_trigger = evaluate_trigger_target_rate(qura, loader, device, trigger, args.bd_target)
    print(f"  INT8 trigger ASR = {int8_trigger['target_rate'] * 100:.2f}%")

    print("\n[5/5] INT8 trigger + RegionDrop ...")
    int8_defense = evaluate_trigger_with_regiondrop(
        qura,
        loader,
        device,
        trigger,
        args.bd_target,
        args.attn_reduce,
        args.mask_mode,
        args.blur_kernel,
        args.blur_sigma,
        args.region_topk,
    )
    print(f"  INT8 + defense ASR = {int8_defense['target_rate'] * 100:.2f}%")

    metrics = {
        "args": vars(args),
        "trigger_source_path": str(trigger_source_path),
        "runtime_sec": round(time.time() - t0, 2),
        "fp32_clean_top1": fp32_clean["top1"],
        "fp32_trigger_asr": fp32_trigger["target_rate"],
        "fp32_trigger_top1": fp32_trigger["top1"],
        "int8_clean_top1": int8_clean["top1"],
        "int8_trigger_asr": int8_trigger["target_rate"],
        "int8_trigger_top1": int8_trigger["top1"],
        "int8_defense_asr": int8_defense["target_rate"],
        "int8_defense_top1_under_trigger": int8_defense["top1"],
        "trigger_eval_total": int(int8_trigger["total"]),
        "window_distribution": int8_defense["window_distribution"],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print("\nSummary:")
    print(f"  FP32 Clean Top-1         : {metrics['fp32_clean_top1'] * 100:.2f}%")
    print(f"  FP32 + trigger ASR       : {metrics['fp32_trigger_asr'] * 100:.2f}%")
    print(f"  INT8-QURA Clean Top-1    : {metrics['int8_clean_top1'] * 100:.2f}%")
    print(f"  INT8-QURA + trigger ASR  : {metrics['int8_trigger_asr'] * 100:.2f}%")
    print(f"  INT8 + defense ASR       : {metrics['int8_defense_asr'] * 100:.2f}%")
    print(f"  Trigger eval samples     : {metrics['trigger_eval_total']}")
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
