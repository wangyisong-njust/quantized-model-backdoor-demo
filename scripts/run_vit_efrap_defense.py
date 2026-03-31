"""
EFRAP-style W4A8 PTQ for ViT models used in this demo repo.

This script borrows the core EFRAP logic:
1. nearest-rounding error as the flip prior
2. activation-preservation reconstruction on calibration data
3. layer-wise optimization over quantized weights

Unlike the official EFRAP entrypoint, this version is wired for:
- timm ViT models
- this repo's Tiny-ImageNet / ImageNet / demo data layout
- saved checkpoints already used by the demo

Example:
  conda run -n qura python scripts/run_vit_efrap_defense.py \
    --model_name vit_tiny_patch16_224 \
    --num_classes 200 \
    --checkpoint third_party/qura/ours/main/model/vit+tiny_imagenet.pth \
    --data_type tiny_imagenet \
    --data_root data/tiny-imagenet-200 \
    --calib_samples 128 \
    --eval_samples 128 \
    --device cuda:3 \
    quantize.reconstruction.max_count=200
"""

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import timm
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

REPO_ROOT = Path(__file__).resolve().parent.parent
QURA_ROOT = REPO_ROOT / "third_party/qura"

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(QURA_ROOT))

from mqbench.prepare_by_platform import BackendType, prepare_by_platform
from mqbench.efrap_ptq import ptq_reconstruction
from mqbench.utils.state import enable_calibration_woquantization, enable_quantization
from utils.io_utils import ensure_dir, save_json
from utils.logger import add_file_handler, get_logger

logger = get_logger(__name__)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class SyntheticImageDataset(Dataset):
    def __init__(self, num_samples=256, num_classes=1000, image_size=224, transform=None):
        self.transform = transform
        rng = np.random.RandomState(42)
        self.images = rng.randint(0, 256, (num_samples, image_size, image_size, 3), dtype=np.uint8)
        self.labels = rng.randint(0, num_classes, num_samples).tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = Image.fromarray(self.images[idx])
        if self.transform is not None:
            image = self.transform(image)
        return image, self.labels[idx]


def parse_args():
    parser = argparse.ArgumentParser(description="Run EFRAP-style PTQ on a ViT model.")
    parser.add_argument("--config", default="configs/quant/vit_efrap_w4a8.yaml")
    parser.add_argument("--model_name", default="vit_tiny_patch16_224")
    parser.add_argument("--num_classes", type=int, default=1000)
    parser.add_argument("--checkpoint", default=None, help="Optional FP32 checkpoint to load.")
    parser.add_argument("--pretrained", action="store_true", help="Load timm pretrained weights.")
    parser.add_argument("--data_type", choices=["imagenet", "tiny_imagenet", "demo"], default="demo")
    parser.add_argument("--data_root", default="data/tiny-imagenet-200")
    parser.add_argument("--calib_split", default="train")
    parser.add_argument("--eval_split", default="val")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--calib_samples", type=int, default=512)
    parser.add_argument("--eval_samples", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_dir", default="outputs/efrap_vit")
    parser.add_argument("--save_name", default=None)
    parser.add_argument("--trigger_path", default=None, help="Optional raw trigger tensor for targeted eval.")
    parser.add_argument("--bd_target", type=int, default=None, help="Target class used for trigger eval.")
    parser.add_argument("--max_eval_batches", type=int, default=None)
    parser.add_argument("overrides", nargs="*", help="OmegaConf dotlist overrides.")
    return parser.parse_args()


def build_transform(image_size: int):
    return transforms.Compose(
        [
            transforms.Resize(int(image_size * 256 / 224)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_dataset(data_type: str, data_root: str, split: str, image_size: int, num_classes: int):
    transform = build_transform(image_size)
    if data_type == "demo":
        return SyntheticImageDataset(
            num_samples=max(256, num_classes),
            num_classes=num_classes,
            image_size=image_size,
            transform=transform,
        )
    split_dir = Path(data_root) / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Dataset split not found: {split_dir}")
    return datasets.ImageFolder(str(split_dir), transform=transform)


def build_loader(
    data_type: str,
    data_root: str,
    split: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    max_samples: int,
    num_classes: int,
    shuffle: bool,
):
    dataset = build_dataset(data_type, data_root, split, image_size, num_classes)
    if max_samples > 0 and len(dataset) > max_samples:
        dataset = Subset(dataset, list(range(max_samples)))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def extract_state_dict(payload):
    if isinstance(payload, dict):
        for key in ("model", "state_dict"):
            if key in payload and isinstance(payload[key], dict):
                payload = payload[key]
                break
    cleaned = {}
    for key, value in payload.items():
        if key.startswith("module."):
            key = key[len("module.") :]
        cleaned[key] = value
    return cleaned


def load_model(model_name: str, num_classes: int, checkpoint: Optional[str], pretrained: bool):
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    if checkpoint:
        state = torch.load(checkpoint, map_location="cpu")
        state_dict = extract_state_dict(state)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        logger.info("Loaded checkpoint %s", checkpoint)
        logger.info("Missing keys: %d | Unexpected keys: %d", len(missing), len(unexpected))
    return model.eval()


def collect_cali_data(loader, batch_count: int):
    cali_data = []
    for images, _ in loader:
        cali_data.append(images)
        if len(cali_data) >= batch_count:
            break
    return cali_data


def load_trigger(trigger_path: str):
    obj = torch.load(trigger_path, map_location="cpu")
    if isinstance(obj, dict) and "trigger" in obj:
        trigger = obj["trigger"]
    else:
        trigger = obj
    if trigger.ndim == 4:
        trigger = trigger[0]
    if trigger.ndim != 3:
        raise ValueError(f"Unexpected trigger shape: {tuple(trigger.shape)}")
    return trigger.float().clamp(0, 1)


def normalized_trigger(trigger: torch.Tensor):
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (trigger - mean) / std


def apply_trigger(images: torch.Tensor, trigger: torch.Tensor):
    patched = images.clone()
    trigger = trigger.to(images.device, images.dtype)
    _, h, w = trigger.shape
    patched[:, :, -h:, -w:] = trigger
    return patched


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    trigger: Optional[torch.Tensor] = None,
    bd_target: Optional[int] = None,
    max_batches: Optional[int] = None,
):
    model.eval()
    clean_top1 = 0
    clean_top5 = 0
    total = 0

    trigger_top1 = 0
    target_hits = 0
    target_total = 0

    for batch_idx, (images, labels) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        images = images.to(device)
        labels = labels.to(device)

        clean_logits = model(images)
        top5 = clean_logits.topk(min(5, clean_logits.shape[-1]), dim=-1).indices
        clean_top1 += (top5[:, 0] == labels).sum().item()
        clean_top5 += (top5 == labels.unsqueeze(1)).any(dim=1).sum().item()
        total += labels.size(0)

        if trigger is not None:
            trig_images = apply_trigger(images, trigger)
            trig_logits = model(trig_images)
            trig_pred = trig_logits.argmax(dim=1)
            trigger_top1 += (trig_pred == labels).sum().item()
            if bd_target is not None:
                valid_mask = labels != bd_target
                target_hits += ((trig_pred == bd_target) & valid_mask).sum().item()
                target_total += valid_mask.sum().item()

    results = {
        "clean_top1": clean_top1 / max(total, 1),
        "clean_top5": clean_top5 / max(total, 1),
        "total_samples": total,
    }
    if trigger is not None:
        results["trigger_top1"] = trigger_top1 / max(total, 1)
        if bd_target is not None:
            results["trigger_target_rate"] = target_hits / max(target_total, 1)
    return results


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    save_name = args.save_name or f"{args.model_name}_{args.data_type}_efrap_w4a8"
    output_dir = ensure_dir(Path(args.output_dir) / save_name)
    add_file_handler(logger, str(output_dir / "run.log"))

    logger.info("Device: %s", device)
    logger.info("Output dir: %s", output_dir)

    calib_loader = build_loader(
        data_type=args.data_type,
        data_root=args.data_root,
        split=args.calib_split,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_samples=args.calib_samples,
        num_classes=args.num_classes,
        shuffle=False,
    )
    eval_loader = build_loader(
        data_type=args.data_type,
        data_root=args.data_root,
        split=args.eval_split,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_samples=args.eval_samples,
        num_classes=args.num_classes,
        shuffle=False,
    )

    batch_count = max(1, math.ceil(args.calib_samples / args.batch_size))
    cali_data = collect_cali_data(calib_loader, batch_count=batch_count)
    logger.info("Calibration batches: %d", len(cali_data))

    model = load_model(
        model_name=args.model_name,
        num_classes=args.num_classes,
        checkpoint=args.checkpoint,
        pretrained=args.pretrained,
    ).to(device)

    extra_prepare_dict = OmegaConf.to_container(cfg.extra_prepare_dict, resolve=True)
    quant_model = prepare_by_platform(model, BackendType.Academic, extra_prepare_dict)
    quant_model.to(device).eval()

    logger.info("Running observer calibration...")
    with torch.no_grad():
        enable_calibration_woquantization(quant_model, quantizer_type="act_fake_quant")
        for batch in cali_data:
            quant_model(batch.to(device))
        enable_calibration_woquantization(quant_model, quantizer_type="weight_fake_quant")
        quant_model(cali_data[0].to(device))

    logger.info("Running EFRAP reconstruction...")
    quant_model, efrap_stats = ptq_reconstruction(quant_model, cali_data, cfg.quantize.reconstruction)
    quant_model.to(device).eval()
    enable_quantization(quant_model)

    trigger = None
    if args.trigger_path:
        trigger = normalized_trigger(load_trigger(args.trigger_path))
        logger.info("Loaded trigger %s with size=%s", args.trigger_path, tuple(trigger.shape))

    logger.info("Evaluating reconstructed model...")
    metrics = evaluate_model(
        quant_model,
        eval_loader,
        device,
        trigger=trigger,
        bd_target=args.bd_target,
        max_batches=args.max_eval_batches,
    )

    ckpt_path = output_dir / f"{save_name}.pth"
    meta_path = output_dir / "metrics.json"
    cfg_path = output_dir / "config.yaml"

    torch.save(quant_model.state_dict(), ckpt_path)
    save_json(
        {
            "args": vars(args),
            "metrics": metrics,
            "efrap_stats": efrap_stats,
            "optimized_target_count": len(efrap_stats["optimized_targets"]),
        },
        str(meta_path),
    )
    OmegaConf.save(cfg, cfg_path)

    logger.info("Saved quantized checkpoint to %s", ckpt_path)
    logger.info("Saved metrics to %s", meta_path)
    logger.info(
        "Summary | clean_top1=%.4f clean_top5=%.4f optimized_targets=%d",
        metrics["clean_top1"],
        metrics["clean_top5"],
        len(efrap_stats["optimized_targets"]),
    )
    if "trigger_target_rate" in metrics:
        logger.info(
            "Trigger eval | trigger_top1=%.4f target_rate=%.4f",
            metrics["trigger_top1"],
            metrics["trigger_target_rate"],
        )


if __name__ == "__main__":
    main()
