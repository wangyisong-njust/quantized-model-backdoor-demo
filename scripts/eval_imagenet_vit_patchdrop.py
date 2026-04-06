"""
ImageNet ViT-B/16 PatchDrop evaluation for fixed-position QURA checkpoints.

Evaluates:
  - No Defense
  - Random PatchDrop
  - Attention-Guided PatchDrop
  - Oracle Trigger Mask

Supports both the strict single-patch setting and an optimized multi-patch
setting for ImageNet ViT where the trigger signal may spread across a few
high-attention patches.
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "third_party/qura/ours/main"))
sys.path.insert(0, str(REPO))

from defenses.regiondrop.region_detector import GRID_SIZE, PATCH_SIZE, AttentionHook
from scripts.eval_imagenet_vit_qura_metrics import (
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


ORACLE_TRIGGER_PATCH = GRID_SIZE * GRID_SIZE - 1


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ImageNet ViT PatchDrop baselines.")
    parser.add_argument("--variant", choices=["default", "fixedpos"], default="fixedpos")
    parser.add_argument("--quant_model", default=None)
    parser.add_argument("--quant_config", default=str(DEFAULT_QUANT_CONFIG))
    parser.add_argument("--trigger_file", default=None)
    parser.add_argument("--trigger_source", choices=["file", "generated"], default="generated")
    parser.add_argument("--trigger_cache", default=None)
    parser.add_argument("--force_regenerate_trigger", action="store_true")
    parser.add_argument("--imagenet_val", default=str(DEFAULT_IMAGENET_VAL))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bd_target", type=int, default=0)
    parser.add_argument(
        "--attn_reduce",
        choices=["mean", "sum", "max", "std", "mean_plus_std", "vote_top1"],
        default="mean",
    )
    parser.add_argument("--patch_topk", type=int, default=1)
    parser.add_argument("--gate_on_target_pred", action="store_true")
    parser.add_argument("--output_dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--output_name", default=None)
    return parser.parse_args()


def apply_trigger(images: torch.Tensor, trigger: torch.Tensor):
    patched = images.clone()
    trigger = trigger.to(images.device, images.dtype)
    _, h, w = trigger.shape
    patched[:, :, -h:, -w:] = trigger
    return patched


def apply_patch_mask(images: torch.Tensor, patch_indices_per_sample):
    masked = images.clone()
    for sample_idx, patch_indices in enumerate(patch_indices_per_sample):
        for flat_idx in patch_indices:
            row = int(flat_idx // GRID_SIZE)
            col = int(flat_idx % GRID_SIZE)
            y1 = row * PATCH_SIZE
            x1 = col * PATCH_SIZE
            masked[sample_idx, :, y1:y1 + PATCH_SIZE, x1:x1 + PATCH_SIZE] = 0.0
    return masked


def select_topk_patches(attn_map, topk: int):
    ranking = torch.argsort(torch.as_tensor(attn_map), descending=True)
    return ranking[:topk].tolist()


def select_random_patches(rng: random.Random, topk: int):
    topk = max(1, min(int(topk), GRID_SIZE * GRID_SIZE))
    return rng.sample(range(GRID_SIZE * GRID_SIZE), k=topk)


def should_fire_gate(pred0: int, bd_target: int, gate_on_target_pred: bool) -> bool:
    if not gate_on_target_pred:
        return True
    return pred0 == bd_target


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
def evaluate_patchdrop(
    model,
    loader,
    device,
    trigger,
    bd_target,
    strategy,
    attn_reduce,
    patch_topk,
    seed,
    gate_on_target_pred,
):
    total = 0
    clean_hits = 0
    target_hits = 0
    rng = random.Random(seed)
    hook = AttentionHook(model) if strategy == "guided" else None
    trigger_rank_sum = 0
    trigger_rank_max = 0
    trigger_top1_hits = 0
    trigger_top2_hits = 0
    trigger_top4_hits = 0
    trigger_diag_total = 0
    clean_gate_fires = 0
    trigger_gate_fires = 0
    clean_total = 0

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
                pred0 = logits0.argmax(dim=1).item()
                clean_total += 1

                gate_fired = should_fire_gate(pred0, bd_target, gate_on_target_pred)
                patch_indices = []
                if gate_fired:
                    clean_gate_fires += 1
                    if strategy == "random":
                        patch_indices = select_random_patches(rng, patch_topk)
                    elif strategy == "oracle":
                        patch_indices = [ORACLE_TRIGGER_PATCH]
                    elif strategy == "guided":
                        patch_indices = select_topk_patches(hook.get_cls_attention_map(reduce=attn_reduce), patch_topk)
                    else:
                        raise ValueError(f"Unknown strategy: {strategy}")

                if patch_indices:
                    masked_clean = apply_patch_mask(img, [patch_indices])
                    pred_clean = model(masked_clean).argmax(dim=1).item()
                else:
                    pred_clean = pred0
                clean_hits += int(pred_clean == labels[idx].item())

            for idx in range(trigger_images.size(0)):
                img = trigger_images[idx : idx + 1]
                logits0 = model(img)
                pred0 = logits0.argmax(dim=1).item()
                gate_fired = should_fire_gate(pred0, bd_target, gate_on_target_pred)
                patch_indices = []

                if strategy == "guided":
                    attn_map = hook.get_cls_attention_map(reduce=attn_reduce)
                    ranking = torch.argsort(torch.as_tensor(attn_map), descending=True).tolist()
                    trigger_rank = ranking.index(ORACLE_TRIGGER_PATCH) + 1
                    trigger_rank_sum += trigger_rank
                    trigger_rank_max = max(trigger_rank_max, trigger_rank)
                    trigger_top1_hits += int(trigger_rank <= 1)
                    trigger_top2_hits += int(trigger_rank <= 2)
                    trigger_top4_hits += int(trigger_rank <= 4)
                    trigger_diag_total += 1

                    if gate_fired:
                        patch_indices = ranking[:patch_topk]
                elif gate_fired:
                    if strategy == "random":
                        patch_indices = select_random_patches(rng, patch_topk)
                    elif strategy == "oracle":
                        patch_indices = [ORACLE_TRIGGER_PATCH]
                    else:
                        raise ValueError(f"Unknown strategy: {strategy}")
                elif strategy not in {"random", "oracle"}:
                    raise ValueError(f"Unknown strategy: {strategy}")

                if gate_fired:
                    trigger_gate_fires += 1

                if patch_indices:
                    masked_trigger = apply_patch_mask(img, [patch_indices])
                    pred_trigger = model(masked_trigger).argmax(dim=1).item()
                else:
                    pred_trigger = pred0

                target_hits += int(pred_trigger == bd_target)
                total += 1
    finally:
        if hook is not None:
            hook.remove()

    result = {
        "clean_acc": clean_hits / max(clean_total, 1),
        "trigger_asr": target_hits / max(total, 1),
        "trigger_hits": target_hits,
        "trigger_total": total,
    }
    result["gate"] = {
        "enabled": gate_on_target_pred,
        "clean_fire_rate": clean_gate_fires / max(clean_total, 1),
        "clean_fire_count": clean_gate_fires,
        "clean_total": clean_total,
        "trigger_fire_rate": trigger_gate_fires / max(total, 1),
        "trigger_fire_count": trigger_gate_fires,
        "trigger_total": total,
    }
    if strategy == "guided" and trigger_diag_total > 0:
        result["localization"] = {
            "trigger_patch_top1_hit_rate": trigger_top1_hits / trigger_diag_total,
            "trigger_patch_top2_hit_rate": trigger_top2_hits / trigger_diag_total,
            "trigger_patch_top4_hit_rate": trigger_top4_hits / trigger_diag_total,
            "trigger_patch_avg_rank": trigger_rank_sum / trigger_diag_total,
            "trigger_patch_worst_rank": trigger_rank_max,
            "trigger_samples": trigger_diag_total,
            "attn_reduce": attn_reduce,
            "patch_topk": patch_topk,
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
    gate_suffix = "_gate_target" if args.gate_on_target_pred else ""
    output_name = args.output_name or f"patchdrop_{args.variant}_{args.max_samples}_{args.attn_reduce}_top{args.patch_topk}{gate_suffix}.json"
    output_path = output_dir / output_name

    print("Config:")
    print(f"  variant      = {args.variant}")
    print(f"  quant_model  = {quant_model_path}")
    print(f"  quant_config = {quant_config_path}")
    print(f"  trigger_src  = {args.trigger_source}")
    print(f"  attn_reduce  = {args.attn_reduce}")
    print(f"  patch_topk   = {args.patch_topk}")
    print(f"  gate_target  = {args.gate_on_target_pred}")
    print(f"  max_samples  = {args.max_samples}")
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

    fp32 = load_fp32(device)
    qura = load_qura(device, quant_model_path, quant_config_path)

    results = {
        "args": {
            **vars(args),
            "quant_model": str(quant_model_path),
            "quant_config": str(quant_config_path),
            "trigger_source_path": str(trigger_source_path),
        }
    }

    t0 = time.time()
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

    results["random_patchdrop"] = evaluate_patchdrop(
        qura, loader, device, trigger, args.bd_target, "random", args.attn_reduce, args.patch_topk, args.seed + 17, args.gate_on_target_pred
    )
    results["guided_patchdrop"] = evaluate_patchdrop(
        qura, loader, device, trigger, args.bd_target, "guided", args.attn_reduce, args.patch_topk, args.seed, args.gate_on_target_pred
    )
    results["oracle_trigger_mask"] = evaluate_patchdrop(
        qura, loader, device, trigger, args.bd_target, "oracle", args.attn_reduce, 1, args.seed, args.gate_on_target_pred
    )
    results["runtime_sec"] = round(time.time() - t0, 2)

    output_path.write_text(json.dumps(results, indent=2))

    print("\nSummary:")
    print(f"  FP32 Pretrained        : clean={results['fp32_pretrained']['clean_acc'] * 100:.2f}%  asr={results['fp32_pretrained']['trigger_asr'] * 100:.2f}%")
    print(f"  No Defense             : clean={results['no_defense']['clean_acc'] * 100:.2f}%  asr={results['no_defense']['trigger_asr'] * 100:.2f}%")
    print(f"  Random PatchDrop       : clean={results['random_patchdrop']['clean_acc'] * 100:.2f}%  asr={results['random_patchdrop']['trigger_asr'] * 100:.2f}%")
    print(f"  Attention-Guided       : clean={results['guided_patchdrop']['clean_acc'] * 100:.2f}%  asr={results['guided_patchdrop']['trigger_asr'] * 100:.2f}%")
    print(f"  Oracle Trigger Mask    : clean={results['oracle_trigger_mask']['clean_acc'] * 100:.2f}%  asr={results['oracle_trigger_mask']['trigger_asr'] * 100:.2f}%")
    if "localization" in results["guided_patchdrop"]:
        loc = results["guided_patchdrop"]["localization"]
        print(
            "  Guided localization   : "
            f"top1={loc['trigger_patch_top1_hit_rate'] * 100:.2f}%  "
            f"top2={loc['trigger_patch_top2_hit_rate'] * 100:.2f}%  "
            f"top4={loc['trigger_patch_top4_hit_rate'] * 100:.2f}%  "
            f"avg_rank={loc['trigger_patch_avg_rank']:.2f}"
        )
    gate = results["guided_patchdrop"]["gate"]
    print(
        "  Guided gate fire      : "
        f"clean={gate['clean_fire_rate'] * 100:.2f}%  "
        f"trigger={gate['trigger_fire_rate'] * 100:.2f}%"
    )
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
