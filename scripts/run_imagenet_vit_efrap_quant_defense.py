"""
Run EFRAP-style post-hoc reconstruction on an existing ImageNet ViT-B/16
QURA quantized checkpoint, then compare trigger ASR before/after defense.

This is the correct evaluation path for the fixed-position QURA checkpoint:
  attacked INT8 checkpoint -> EFRAP reconstruction -> defended INT8 checkpoint
"""

import argparse
import math
import json
import sys
import time
from collections import OrderedDict
from pathlib import Path

import torch
import timm
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parent.parent
QURA_ROOT = REPO / "third_party/qura"

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(QURA_ROOT))

from mqbench.efrap_ptq import ptq_reconstruction  # noqa: E402
from mqbench.utils.state import enable_calibration_woquantization, enable_quantization  # noqa: E402
from mqbench.prepare_by_platform import BackendType, prepare_by_platform  # noqa: E402
from scripts.eval_imagenet_vit_qura_metrics import (  # noqa: E402
    DEFAULT_QUANT_CONFIG,
    DEFAULT_QUANT_MODEL,
    DEFAULT_TRIGGER_FILE,
    FIXEDPOS_QUANT_MODEL,
    apply_trigger,
    build_loader,
    evaluate_clean_accuracy,
    evaluate_trigger_target_rate,
    load_eval_trigger,
    load_fp32,
    load_qura,
)
from utils.qura_checkpoint import load_quant_checkpoint  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="EFRAP defense for an existing ViT ImageNet quant checkpoint.")
    parser.add_argument("--variant", choices=["default", "fixedpos"], default="fixedpos")
    parser.add_argument("--quant_model", default=None)
    parser.add_argument("--baseline_quant_model", default=None)
    parser.add_argument("--quant_config", default=str(DEFAULT_QUANT_CONFIG))
    parser.add_argument("--trigger_file", default=str(DEFAULT_TRIGGER_FILE))
    parser.add_argument("--imagenet_root", default="/home/kaixin/ssd/imagenet")
    parser.add_argument("--trigger_source", choices=["file", "generated"], default="generated")
    parser.add_argument("--trigger_cache", default=None)
    parser.add_argument("--force_regenerate_trigger", action="store_true")
    parser.add_argument("--calib_split", default="train")
    parser.add_argument("--eval_split", default="val")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--calib_samples", type=int, default=128)
    parser.add_argument("--eval_samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bd_target", type=int, default=0)
    parser.add_argument("--trigger_weight", type=float, default=None)
    parser.add_argument("--trigger_logit_weight", type=float, default=None)
    parser.add_argument("--target_suppression_weight", type=float, default=None)
    parser.add_argument("--restore_adaround", action="store_true")
    parser.add_argument("--recover_soft_weights", action="store_true")
    parser.add_argument("--restrict_to_recovered_soft", action="store_true")
    parser.add_argument("--output_dir", default=str(REPO / "outputs/efrap_vit"))
    parser.add_argument("--save_name", default=None)
    parser.add_argument("overrides", nargs="*", help="OmegaConf dotlist overrides for quantize.reconstruction")
    return parser.parse_args()


def collect_cali_data(loader, batch_count: int, trigger: torch.Tensor, bd_target: int):
    clean_batches = []
    trigger_batches = []
    for images, labels in loader:
        valid = labels != bd_target
        images = images[valid]
        if images.numel() == 0:
            continue
        clean_batches.append(images)
        trigger_batches.append(apply_trigger(images, trigger.cpu()))
        if len(clean_batches) >= batch_count:
            break
    return clean_batches, trigger_batches


def load_defense_source(
    device,
    quant_model_path: Path,
    quant_config_path: Path,
    restore_adaround: bool,
    recover_soft_weights: bool,
):
    cfg = OmegaConf.load(quant_config_path)
    extra = OmegaConf.to_container(cfg.extra_prepare_dict, resolve=True)
    model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=1000)
    model = prepare_by_platform(model, BackendType.Academic, extra)
    _, missing, unexpected, restored_adaround, recovered_soft = load_quant_checkpoint(
        model,
        quant_model_path,
        strict=False,
        restore_adaround=restore_adaround,
        recover_soft_weights=recover_soft_weights,
    )
    ignored_alpha = [key for key in unexpected if key.endswith(".alpha")]
    remaining_unexpected = [key for key in unexpected if not key.endswith(".alpha")]
    if ignored_alpha:
        print(f"[load_defense_source] ignored stale AdaRound alpha tensors: {len(ignored_alpha)}")
    if remaining_unexpected:
        print(f"[load_defense_source] unexpected keys: {len(remaining_unexpected)}")
    if restored_adaround:
        print(f"[load_defense_source] restored AdaRound quantizers: {len(restored_adaround)}")
    if recovered_soft:
        print(f"[load_defense_source] recovered soft weights: {len(recovered_soft)}")
    if missing:
        print(f"[load_defense_source] missing keys: {len(missing)}")
    available_soft_layers = list(recovered_soft) if recovered_soft else list(restored_adaround)
    return (
        model.to(device).eval(),
        list(restored_adaround),
        list(recovered_soft),
        available_soft_layers,
    )


def export_deploy_state_dict(model: torch.nn.Module):
    deploy_state = OrderedDict()
    stripped_alpha_keys = []
    for key, value in model.state_dict().items():
        if key.endswith(".alpha"):
            stripped_alpha_keys.append(key)
            continue
        if isinstance(value, torch.Tensor):
            deploy_state[key] = value.detach().cpu()
        else:
            deploy_state[key] = value
    return deploy_state, stripped_alpha_keys


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    quant_model_path = Path(args.quant_model) if args.quant_model else (
        FIXEDPOS_QUANT_MODEL if args.variant == "fixedpos" else DEFAULT_QUANT_MODEL
    )
    baseline_quant_model_path = Path(args.baseline_quant_model) if args.baseline_quant_model else quant_model_path
    quant_config_path = Path(args.quant_config)
    trigger_file = Path(args.trigger_file)
    imagenet_root = Path(args.imagenet_root)
    output_dir = Path(args.output_dir)
    save_name = args.save_name or f"efrap_{args.variant}_c{args.calib_samples}_e{args.eval_samples}"
    run_dir = output_dir / save_name
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg = OmegaConf.load(quant_config_path)
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))
    if not hasattr(cfg.quantize.reconstruction, "trigger_weight"):
        cfg.quantize.reconstruction.trigger_weight = 0.0
    if args.trigger_weight is not None:
        cfg.quantize.reconstruction.trigger_weight = float(args.trigger_weight)
    if not hasattr(cfg.quantize.reconstruction, "trigger_logit_weight"):
        cfg.quantize.reconstruction.trigger_logit_weight = 0.0
    if args.trigger_logit_weight is not None:
        cfg.quantize.reconstruction.trigger_logit_weight = float(args.trigger_logit_weight)
    if not hasattr(cfg.quantize.reconstruction, "target_suppression_weight"):
        cfg.quantize.reconstruction.target_suppression_weight = 0.0
    if args.target_suppression_weight is not None:
        cfg.quantize.reconstruction.target_suppression_weight = float(args.target_suppression_weight)
    if not hasattr(cfg.quantize.reconstruction, "reuse_loaded_alpha"):
        cfg.quantize.reconstruction.reuse_loaded_alpha = False
    if args.restore_adaround:
        cfg.quantize.reconstruction.reuse_loaded_alpha = True

    calib_loader = build_loader(
        imagenet_val=imagenet_root / args.calib_split,
        max_samples=args.calib_samples,
        seed=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    eval_loader = build_loader(
        imagenet_val=imagenet_root / args.eval_split,
        max_samples=args.eval_samples,
        seed=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    trigger, trigger_source_path = load_eval_trigger(
        trigger_source=args.trigger_source,
        trigger_file=trigger_file,
        quant_config_path=quant_config_path,
        imagenet_root=Path(args.imagenet_root),
        device=device,
        bd_target=args.bd_target,
        trigger_cache=args.trigger_cache,
        force_regenerate_trigger=args.force_regenerate_trigger,
    )
    batch_count = max(1, math.ceil(args.calib_samples / args.batch_size))
    cali_data, trigger_cali_data = collect_cali_data(
        calib_loader,
        batch_count=batch_count,
        trigger=trigger,
        bd_target=args.bd_target,
    )
    if not cali_data:
        raise RuntimeError("No non-target calibration batches were collected.")

    print("Config:")
    print(f"  variant        = {args.variant}")
    print(f"  quant_model    = {quant_model_path}")
    print(f"  quant_config   = {quant_config_path}")
    print(f"  imagenet_root  = {imagenet_root}")
    print(f"  calib_split    = {args.calib_split}")
    print(f"  eval_split     = {args.eval_split}")
    print(f"  device         = {device}")
    print(f"  calib_samples  = {args.calib_samples}")
    print(f"  eval_samples   = {args.eval_samples}")
    print(f"  trigger_weight = {float(cfg.quantize.reconstruction.trigger_weight):.4f}")
    print(f"  trig_logit_w   = {float(cfg.quantize.reconstruction.trigger_logit_weight):.4f}")
    print(f"  suppress_w     = {float(cfg.quantize.reconstruction.target_suppression_weight):.4f}")
    print(f"  restore_alpha  = {args.restore_adaround}")
    print(f"  recover_soft   = {args.recover_soft_weights}")
    print(f"  soft_only      = {args.restrict_to_recovered_soft}")
    print(f"  save_dir       = {run_dir}")

    t0 = time.time()

    print("\nLoading models ...")
    fp32_model = load_fp32(device)
    baseline_model = load_qura(device, baseline_quant_model_path, quant_config_path)
    defend_model, restored_adaround_layers, recovered_soft_layers, available_soft_layers = load_defense_source(
        device,
        quant_model_path,
        quant_config_path,
        restore_adaround=args.restore_adaround,
        recover_soft_weights=args.recover_soft_weights,
    )

    print("\nEvaluating baseline models ...")
    fp32_clean = evaluate_clean_accuracy(fp32_model, eval_loader, device)
    fp32_trigger = evaluate_trigger_target_rate(fp32_model, eval_loader, device, trigger, args.bd_target)
    baseline_clean = evaluate_clean_accuracy(baseline_model, eval_loader, device)
    baseline_trigger = evaluate_trigger_target_rate(
        baseline_model,
        eval_loader,
        device,
        trigger,
        args.bd_target,
    )

    print("\nRecalibrating quantizers on clean calibration data ...")
    with torch.no_grad():
        enable_calibration_woquantization(defend_model, quantizer_type="act_fake_quant")
        for batch in cali_data:
            defend_model(batch.to(device))
        enable_calibration_woquantization(defend_model, quantizer_type="weight_fake_quant")
        defend_model(cali_data[0].to(device))

    print("\nRunning EFRAP reconstruction ...")
    defended_model, efrap_stats = ptq_reconstruction(
        defend_model,
        cali_data,
        cfg.quantize.reconstruction,
        trigger_data=trigger_cali_data,
        allowed_layer_names=available_soft_layers if args.restrict_to_recovered_soft else None,
    )
    defended_model = defended_model.to(device).eval()
    enable_quantization(defended_model)

    print("\nEvaluating defended in-memory model ...")
    defended_in_memory_clean = evaluate_clean_accuracy(defended_model, eval_loader, device)
    defended_in_memory_trigger = evaluate_trigger_target_rate(
        defended_model,
        eval_loader,
        device,
        trigger,
        args.bd_target,
    )

    ckpt_path = run_dir / f"{save_name}.pth"
    metrics_path = run_dir / "metrics.json"
    config_path = run_dir / "config.yaml"

    deploy_state_dict, stripped_alpha_keys = export_deploy_state_dict(defended_model)
    torch.save(deploy_state_dict, ckpt_path)
    OmegaConf.save(cfg, config_path)

    print("\nEvaluating saved deploy checkpoint ...")
    defended_reloaded_model = load_qura(device, ckpt_path, quant_config_path)
    defended_reloaded_clean = evaluate_clean_accuracy(defended_reloaded_model, eval_loader, device)
    defended_reloaded_trigger = evaluate_trigger_target_rate(
        defended_reloaded_model,
        eval_loader,
        device,
        trigger,
        args.bd_target,
    )

    metrics = {
        "args": vars(args),
        "trigger_source_path": str(trigger_source_path),
        "baseline_quant_model": str(baseline_quant_model_path),
        "runtime_sec": round(time.time() - t0, 2),
        "fp32_clean_top1": fp32_clean["top1"],
        "fp32_trigger_asr": fp32_trigger["target_rate"],
        "baseline_clean_top1": baseline_clean["top1"],
        "baseline_trigger_asr": baseline_trigger["target_rate"],
        "defended_clean_top1": defended_reloaded_clean["top1"],
        "defended_trigger_asr": defended_reloaded_trigger["target_rate"],
        "defended_in_memory_clean_top1": defended_in_memory_clean["top1"],
        "defended_in_memory_trigger_asr": defended_in_memory_trigger["target_rate"],
        "defended_reloaded_clean_top1": defended_reloaded_clean["top1"],
        "defended_reloaded_trigger_asr": defended_reloaded_trigger["target_rate"],
        "clean_top1_delta": defended_reloaded_clean["top1"] - baseline_clean["top1"],
        "trigger_asr_delta": defended_reloaded_trigger["target_rate"] - baseline_trigger["target_rate"],
        "trigger_eval_total": int(defended_reloaded_trigger["total"]),
        "optimized_target_count": len(efrap_stats["optimized_targets"]),
        "restored_adaround_layers": restored_adaround_layers,
        "recovered_soft_layers": recovered_soft_layers,
        "available_soft_layers": available_soft_layers,
        "restricted_to_recovered_soft": bool(args.restrict_to_recovered_soft),
        "exported_checkpoint_type": "deploy_hard_quantized",
        "stripped_alpha_key_count": len(stripped_alpha_keys),
        "efrap_stats": efrap_stats,
        "checkpoint": str(ckpt_path),
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print("\nSummary:")
    print(f"  FP32 Clean Top-1        : {metrics['fp32_clean_top1'] * 100:.2f}%")
    print(f"  FP32 + trigger ASR      : {metrics['fp32_trigger_asr'] * 100:.2f}%")
    print(f"  Baseline INT8 Clean     : {metrics['baseline_clean_top1'] * 100:.2f}%")
    print(f"  Baseline INT8 ASR       : {metrics['baseline_trigger_asr'] * 100:.2f}%")
    print(f"  EFRAP in-memory Clean   : {metrics['defended_in_memory_clean_top1'] * 100:.2f}%")
    print(f"  EFRAP in-memory ASR     : {metrics['defended_in_memory_trigger_asr'] * 100:.2f}%")
    print(f"  EFRAP INT8 Clean        : {metrics['defended_clean_top1'] * 100:.2f}%")
    print(f"  EFRAP INT8 ASR          : {metrics['defended_trigger_asr'] * 100:.2f}%")
    print(f"  Clean delta             : {metrics['clean_top1_delta'] * 100:+.2f}%")
    print(f"  ASR delta               : {metrics['trigger_asr_delta'] * 100:+.2f}%")
    print(f"  Trigger eval samples    : {metrics['trigger_eval_total']}")
    print(f"  Optimized targets       : {metrics['optimized_target_count']}")
    print(f"  Stripped alpha keys     : {metrics['stripped_alpha_key_count']}")
    print(f"\nSaved checkpoint: {ckpt_path}")
    print(f"Saved metrics   : {metrics_path}")


if __name__ == "__main__":
    main()
