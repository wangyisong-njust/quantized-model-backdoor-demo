"""
ImageNet ViT-B/16 region-level defense evaluation for fixed-position QURA checkpoints.

Evaluates:
  - No Defense
  - Random Region Mask
  - Attention-Guided Region Defense
  - Oracle Region Mask

This keeps the original single-patch PatchDrop baseline separate while allowing
W4A8/W8A8 experiments to test contiguous region masking with explicit gate logic.
"""

import argparse
import json
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "third_party/qura/ours/main"))
sys.path.insert(0, str(REPO))

from defenses.regiondrop.region_detector import (  # noqa: E402
    AttentionHook,
    DetectionResult,
    GRID_SIZE,
    PATCH_SIZE,
    apply_region_mask,
    multi_scale_region_search,
)
from scripts.eval_imagenet_vit_qura_metrics import (  # noqa: E402
    DEFAULT_IMAGENET_VAL,
    DEFAULT_OUT_DIR,
    DEFAULT_QUANT_CONFIG,
    DEFAULT_QUANT_MODEL,
    FIXEDPOS_QUANT_MODEL,
    build_loader,
    load_eval_trigger,
    load_fp32,
    load_qura,
    resolve_imagenet_root,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ImageNet ViT region-level defenses.")
    parser.add_argument("--variant", choices=["default", "fixedpos"], default="fixedpos")
    parser.add_argument("--quant_model", default=None)
    parser.add_argument("--quant_config", default=str(DEFAULT_QUANT_CONFIG))
    parser.add_argument("--trigger_file", default=None)
    parser.add_argument("--trigger_source", choices=["file", "generated"], default="generated")
    parser.add_argument("--trigger_cache", default=None)
    parser.add_argument("--force_regenerate_trigger", action="store_true")
    parser.add_argument("--imagenet_val", default=str(DEFAULT_IMAGENET_VAL))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bd_target", type=int, default=0)
    parser.add_argument(
        "--attn_reduce",
        choices=["mean", "sum", "max", "std", "mean_plus_std", "vote_top1"],
        default="std",
    )
    parser.add_argument(
        "--guided_strategy",
        choices=["multiscale", "top1_expand"],
        default="top1_expand",
    )
    parser.add_argument(
        "--attn_layer_name",
        default=None,
        help="Optional exact attn_drop module name used for localization, e.g. blocks.10.attn.attn_drop",
    )
    parser.add_argument(
        "--attn_layer_index",
        type=int,
        default=None,
        help="Optional index into attn_drop modules (supports negative indices).",
    )
    parser.add_argument(
        "--window_sizes",
        default="2x2,3x3,4x4",
        help="Comma-separated candidate windows for multiscale search, e.g. 2x2,3x3,4x4",
    )
    parser.add_argument(
        "--region_window",
        default="3x3",
        help="Single window used by top1_expand/random/oracle, e.g. 3x3",
    )
    parser.add_argument("--mask_mode", choices=["blur", "zero"], default="blur")
    parser.add_argument("--blur_kernel", type=int, default=31)
    parser.add_argument("--blur_sigma", type=float, default=4.0)
    parser.add_argument(
        "--gate_mode",
        choices=["none", "target_pred", "target_prob"],
        default="target_pred",
    )
    parser.add_argument(
        "--gate_threshold",
        type=float,
        default=0.95,
        help="Target probability threshold used only when gate_mode=target_prob.",
    )
    parser.add_argument("--output_dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--output_name", default=None)
    return parser.parse_args()


def parse_window(text: str):
    try:
        h_str, w_str = text.lower().split("x")
        h = int(h_str)
        w = int(w_str)
    except Exception as exc:  # pragma: no cover - argument validation
        raise ValueError(f"Invalid window format: {text}") from exc
    if h <= 0 or w <= 0 or h > GRID_SIZE or w > GRID_SIZE:
        raise ValueError(f"Window out of range: {text}")
    return h, w


def parse_window_sizes(spec: str):
    return [parse_window(item.strip()) for item in spec.split(",") if item.strip()]


def apply_trigger(images: torch.Tensor, trigger: torch.Tensor):
    patched = images.clone()
    trigger = trigger.to(images.device, images.dtype)
    _, h, w = trigger.shape
    patched[:, :, -h:, -w:] = trigger
    return patched


def make_detection(row: int, col: int, window_h: int, window_w: int, score: float, attn_map: np.ndarray):
    row = max(0, min(int(row), GRID_SIZE - window_h))
    col = max(0, min(int(col), GRID_SIZE - window_w))
    y1 = row * PATCH_SIZE
    x1 = col * PATCH_SIZE
    return DetectionResult(
        grid_row=row,
        grid_col=col,
        window_h=window_h,
        window_w=window_w,
        score=float(score),
        pixel_bbox=(y1, x1, y1 + window_h * PATCH_SIZE, x1 + window_w * PATCH_SIZE),
        attn_map=attn_map.reshape(GRID_SIZE, GRID_SIZE).astype(np.float32),
    )


def detection_top1_expand(attn_map: np.ndarray, window_h: int, window_w: int):
    attn_flat = np.asarray(attn_map).reshape(-1)
    idx = int(attn_flat.argmax())
    row = idx // GRID_SIZE
    col = idx % GRID_SIZE
    row0 = row - window_h // 2
    col0 = col - window_w // 2
    return make_detection(row0, col0, window_h, window_w, score=float(attn_flat[idx]), attn_map=attn_map)


def detection_random(rng: random.Random, window_h: int, window_w: int):
    row = rng.randrange(0, GRID_SIZE - window_h + 1)
    col = rng.randrange(0, GRID_SIZE - window_w + 1)
    return make_detection(row, col, window_h, window_w, score=0.0, attn_map=np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32))


def detection_oracle_region(window_h: int, window_w: int):
    row = GRID_SIZE - window_h
    col = GRID_SIZE - window_w
    return make_detection(row, col, window_h, window_w, score=1.0, attn_map=np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32))


def should_fire_gate(logits: torch.Tensor, bd_target: int, gate_mode: str, gate_threshold: float):
    probs = torch.softmax(logits, dim=1)
    pred = int(probs.argmax(dim=1).item())
    target_prob = float(probs[0, bd_target].item())
    if gate_mode == "none":
        return True, pred, target_prob
    if gate_mode == "target_pred":
        return pred == bd_target, pred, target_prob
    if gate_mode == "target_prob":
        return pred == bd_target and target_prob >= gate_threshold, pred, target_prob
    raise ValueError(f"Unsupported gate mode: {gate_mode}")


def patch_in_detection(result: DetectionResult, patch_row: int = GRID_SIZE - 1, patch_col: int = GRID_SIZE - 1):
    return (
        result.grid_row <= patch_row < result.grid_row + result.window_h
        and result.grid_col <= patch_col < result.grid_col + result.window_w
    )


@torch.no_grad()
def evaluate_no_defense(model, loader, device, trigger, bd_target, mode: str):
    total = 0
    hits = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        valid = labels != bd_target if mode == "trigger" else torch.ones_like(labels, dtype=torch.bool)
        if valid.sum().item() == 0:
            continue
        images = images[valid]
        labels = labels[valid]
        if mode == "trigger":
            images = apply_trigger(images, trigger)
        pred = model(images).argmax(dim=1)
        total += labels.size(0)
        if mode == "clean":
            hits += (pred == labels).sum().item()
        else:
            hits += (pred == bd_target).sum().item()
    return {"rate": hits / max(total, 1), "hits": hits, "total": total}


@torch.no_grad()
def evaluate_region_defense(
    model,
    loader,
    device,
    trigger,
    bd_target,
    strategy,
    attn_reduce,
    guided_strategy,
    window_sizes,
    region_window,
    mask_mode,
    blur_kernel,
    blur_sigma,
    gate_mode,
    gate_threshold,
    seed,
    attn_layer_name,
    attn_layer_index,
):
    total = 0
    clean_total = 0
    clean_hits = 0
    target_hits = 0
    rng = random.Random(seed)
    hook = AttentionHook(model, layer_name=attn_layer_name, layer_index=attn_layer_index) if strategy == "guided" else None
    cover_hits = 0
    cover_total = 0
    window_hist = {}
    clean_gate_count = 0
    trigger_gate_count = 0
    clean_target_prob_sum = 0.0
    trigger_target_prob_sum = 0.0
    clean_target_prob_gate_sum = 0.0
    trigger_target_prob_gate_sum = 0.0

    def pick_detection(attn_map: np.ndarray):
        if strategy == "guided":
            if guided_strategy == "multiscale":
                return multi_scale_region_search(attn_map, window_sizes=window_sizes)
            if guided_strategy == "top1_expand":
                return detection_top1_expand(attn_map, *region_window)
            raise ValueError(f"Unsupported guided strategy: {guided_strategy}")
        if strategy == "random":
            return detection_random(rng, *region_window)
        if strategy == "oracle":
            return detection_oracle_region(*region_window)
        raise ValueError(f"Unknown strategy: {strategy}")

    try:
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            clean_images = images
            trigger_valid = labels != bd_target
            trigger_images = apply_trigger(images[trigger_valid], trigger)
            trigger_labels = labels[trigger_valid]

            for idx in range(clean_images.size(0)):
                img = clean_images[idx : idx + 1]
                logits0 = model(img)
                fire, pred0, target_prob = should_fire_gate(logits0, bd_target, gate_mode, gate_threshold)
                clean_total += 1
                clean_target_prob_sum += target_prob

                if fire:
                    clean_gate_count += 1
                    clean_target_prob_gate_sum += target_prob
                    attn_map = hook.get_cls_attention_map(reduce=attn_reduce) if strategy == "guided" else np.zeros((GRID_SIZE * GRID_SIZE,), dtype=np.float32)
                    result = pick_detection(attn_map)
                    key = f"{result.window_h}x{result.window_w}"
                    window_hist[key] = window_hist.get(key, 0) + 1
                    defended = apply_region_mask(
                        img,
                        result.pixel_bbox,
                        mode=mask_mode,
                        blur_kernel_size=blur_kernel,
                        blur_sigma=blur_sigma,
                    )
                    pred = int(model(defended).argmax(dim=1).item())
                else:
                    pred = pred0
                clean_hits += int(pred == int(labels[idx].item()))

            for idx in range(trigger_images.size(0)):
                img = trigger_images[idx : idx + 1]
                logits0 = model(img)
                fire, pred0, target_prob = should_fire_gate(logits0, bd_target, gate_mode, gate_threshold)
                total += 1
                trigger_target_prob_sum += target_prob

                if strategy == "guided":
                    attn_map = hook.get_cls_attention_map(reduce=attn_reduce)
                    result = pick_detection(attn_map)
                    cover_hits += int(patch_in_detection(result))
                    cover_total += 1
                elif fire:
                    result = pick_detection(np.zeros((GRID_SIZE * GRID_SIZE,), dtype=np.float32))
                else:
                    result = None

                if fire:
                    trigger_gate_count += 1
                    trigger_target_prob_gate_sum += target_prob
                    if result is None:
                        raise RuntimeError("Expected detection when gate fires.")
                    key = f"{result.window_h}x{result.window_w}"
                    window_hist[key] = window_hist.get(key, 0) + 1
                    defended = apply_region_mask(
                        img,
                        result.pixel_bbox,
                        mode=mask_mode,
                        blur_kernel_size=blur_kernel,
                        blur_sigma=blur_sigma,
                    )
                    pred = int(model(defended).argmax(dim=1).item())
                else:
                    pred = pred0
                target_hits += int(pred == bd_target)
    finally:
        if hook is not None:
            hook.remove()

    result = {
        "clean_acc": clean_hits / max(clean_total, 1),
        "trigger_asr": target_hits / max(total, 1),
        "trigger_hits": target_hits,
        "trigger_total": total,
        "window_distribution": window_hist,
        "gate": {
            "mode": gate_mode,
            "threshold": gate_threshold,
            "clean_fire_rate": clean_gate_count / max(clean_total, 1),
            "clean_fire_count": clean_gate_count,
            "clean_total": clean_total,
            "trigger_fire_rate": trigger_gate_count / max(total, 1),
            "trigger_fire_count": trigger_gate_count,
            "trigger_total": total,
            "clean_target_prob_mean": clean_target_prob_sum / max(clean_total, 1),
            "trigger_target_prob_mean": trigger_target_prob_sum / max(total, 1),
            "clean_target_prob_mean_when_fired": clean_target_prob_gate_sum / max(clean_gate_count, 1),
            "trigger_target_prob_mean_when_fired": trigger_target_prob_gate_sum / max(trigger_gate_count, 1),
        },
    }
    if strategy == "guided" and cover_total > 0:
        result["localization"] = {
            "oracle_patch_covered_rate": cover_hits / cover_total,
            "trigger_samples": cover_total,
            "attn_reduce": attn_reduce,
            "guided_strategy": guided_strategy,
            "attn_layer_name": attn_layer_name,
            "attn_layer_index": attn_layer_index,
            "region_window": f"{region_window[0]}x{region_window[1]}",
            "window_sizes": [f"{h}x{w}" for h, w in window_sizes],
        }
    return result


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    quant_model_path = Path(args.quant_model) if args.quant_model else (
        FIXEDPOS_QUANT_MODEL if args.variant == "fixedpos" else DEFAULT_QUANT_MODEL
    )
    quant_config_path = Path(args.quant_config)
    trigger_file = Path(args.trigger_file) if args.trigger_file else (
        Path("third_party/qura/ours/main/model/vit_base+imagenet.trigger.pt")
    )
    imagenet_val = Path(args.imagenet_val)
    imagenet_root = resolve_imagenet_root(imagenet_val)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    window_sizes = parse_window_sizes(args.window_sizes)
    region_window = parse_window(args.region_window)

    gate_suffix = "" if args.gate_mode == "none" else f"_{args.gate_mode}"
    if args.gate_mode == "target_prob":
        gate_suffix += f"{str(args.gate_threshold).replace('.', 'p')}"
    output_name = args.output_name or (
        f"regiondef_{args.variant}_{args.max_samples}_{args.guided_strategy}_{args.attn_reduce}_"
        f"{args.region_window}_{args.mask_mode}{gate_suffix}.json"
    )
    output_path = output_dir / output_name

    print("Config:")
    print(f"  variant         = {args.variant}")
    print(f"  quant_model     = {quant_model_path}")
    print(f"  quant_config    = {quant_config_path}")
    print(f"  trigger_src     = {args.trigger_source}")
    print(f"  attn_reduce     = {args.attn_reduce}")
    print(f"  guided_strategy = {args.guided_strategy}")
    print(f"  attn_layer_name = {args.attn_layer_name}")
    print(f"  attn_layer_idx  = {args.attn_layer_index}")
    print(f"  window_sizes    = {[f'{h}x{w}' for h, w in window_sizes]}")
    print(f"  region_window   = {args.region_window}")
    print(f"  mask_mode       = {args.mask_mode}")
    print(f"  gate_mode       = {args.gate_mode}")
    print(f"  gate_threshold  = {args.gate_threshold}")
    print(f"  max_samples     = {args.max_samples}")
    print(f"  output_path     = {output_path}")

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

    fp32 = load_fp32(device)
    qura = load_qura(device, quant_model_path, quant_config_path)

    t0 = time.time()
    results = {
        "args": {
            **vars(args),
            "quant_model": str(quant_model_path),
            "quant_config": str(quant_config_path),
            "trigger_source_path": str(trigger_source_path),
            "window_sizes": [f"{h}x{w}" for h, w in window_sizes],
            "region_window": f"{region_window[0]}x{region_window[1]}",
        }
    }

    fp32_clean = evaluate_no_defense(fp32, loader, device, trigger, args.bd_target, mode="clean")
    fp32_trigger = evaluate_no_defense(fp32, loader, device, trigger, args.bd_target, mode="trigger")
    qura_clean = evaluate_no_defense(qura, loader, device, trigger, args.bd_target, mode="clean")
    qura_trigger = evaluate_no_defense(qura, loader, device, trigger, args.bd_target, mode="trigger")

    results["fp32_pretrained"] = {
        "clean_acc": fp32_clean["rate"],
        "trigger_asr": fp32_trigger["rate"],
    }
    results["no_defense"] = {
        "clean_acc": qura_clean["rate"],
        "trigger_asr": qura_trigger["rate"],
    }
    results["random_region"] = evaluate_region_defense(
        qura,
        loader,
        device,
        trigger,
        args.bd_target,
        strategy="random",
        attn_reduce=args.attn_reduce,
        guided_strategy=args.guided_strategy,
        window_sizes=window_sizes,
        region_window=region_window,
        mask_mode=args.mask_mode,
        blur_kernel=args.blur_kernel,
        blur_sigma=args.blur_sigma,
        gate_mode=args.gate_mode,
        gate_threshold=args.gate_threshold,
        seed=args.seed + 17,
        attn_layer_name=args.attn_layer_name,
        attn_layer_index=args.attn_layer_index,
    )
    results["guided_region"] = evaluate_region_defense(
        qura,
        loader,
        device,
        trigger,
        args.bd_target,
        strategy="guided",
        attn_reduce=args.attn_reduce,
        guided_strategy=args.guided_strategy,
        window_sizes=window_sizes,
        region_window=region_window,
        mask_mode=args.mask_mode,
        blur_kernel=args.blur_kernel,
        blur_sigma=args.blur_sigma,
        gate_mode=args.gate_mode,
        gate_threshold=args.gate_threshold,
        seed=args.seed,
        attn_layer_name=args.attn_layer_name,
        attn_layer_index=args.attn_layer_index,
    )
    results["oracle_region"] = evaluate_region_defense(
        qura,
        loader,
        device,
        trigger,
        args.bd_target,
        strategy="oracle",
        attn_reduce=args.attn_reduce,
        guided_strategy=args.guided_strategy,
        window_sizes=window_sizes,
        region_window=region_window,
        mask_mode=args.mask_mode,
        blur_kernel=args.blur_kernel,
        blur_sigma=args.blur_sigma,
        gate_mode=args.gate_mode,
        gate_threshold=args.gate_threshold,
        seed=args.seed,
        attn_layer_name=args.attn_layer_name,
        attn_layer_index=args.attn_layer_index,
    )
    results["runtime_sec"] = round(time.time() - t0, 2)

    output_path.write_text(json.dumps(results, indent=2))

    print("\nSummary:")
    print(f"  FP32 Pretrained     : clean={results['fp32_pretrained']['clean_acc'] * 100:.2f}%  asr={results['fp32_pretrained']['trigger_asr'] * 100:.2f}%")
    print(f"  No Defense          : clean={results['no_defense']['clean_acc'] * 100:.2f}%  asr={results['no_defense']['trigger_asr'] * 100:.2f}%")
    print(f"  Random Region       : clean={results['random_region']['clean_acc'] * 100:.2f}%  asr={results['random_region']['trigger_asr'] * 100:.2f}%")
    print(f"  Guided Region       : clean={results['guided_region']['clean_acc'] * 100:.2f}%  asr={results['guided_region']['trigger_asr'] * 100:.2f}%")
    print(f"  Oracle Region       : clean={results['oracle_region']['clean_acc'] * 100:.2f}%  asr={results['oracle_region']['trigger_asr'] * 100:.2f}%")
    gate = results["guided_region"]["gate"]
    print(
        "  Guided gate fire    : "
        f"clean={gate['clean_fire_rate'] * 100:.2f}%  "
        f"trigger={gate['trigger_fire_rate'] * 100:.2f}%"
    )
    if "localization" in results["guided_region"]:
        loc = results["guided_region"]["localization"]
        print(
            "  Guided localization : "
            f"oracle_patch_cover={loc['oracle_patch_covered_rate'] * 100:.2f}%"
        )
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
