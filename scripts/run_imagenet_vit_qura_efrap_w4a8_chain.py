"""
Run the full ImageNet ViT-B/16 W4A8 chain:

  FP32 source -> QURA attack quantization -> EFRAP defense -> before/after eval

This script keeps the W4A8 offline-defense branch isolated from the existing
online demo artifacts. It can optionally reproduce the attacked checkpoint
from scratch, then launches the ViT EFRAP defense runner and writes a compact
summary JSON for README / reporting.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable
OUTPUT_DIR = REPO_ROOT / "outputs/efrap_vit/w4a8_full_chain"
DEFAULT_ATTACK_CONFIG = REPO_ROOT / "configs/attack/vit_base_imagenet_w4a8_qura.yaml"
DEFAULT_ATTACK_MODEL = REPO_ROOT / "third_party/qura/ours/main/model/vit_base+imagenet.quant_bd_w4a8_t0_fixedpos.pth"


def run_cmd(cmd, log_path: Path, env=None):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as fout:
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        for line in proc.stdout:
            print(line, end="")
            fout.write(line)
            fout.flush()
        proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {proc.returncode}: {' '.join(cmd)}")


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument("--bd-target", type=int, default=0)
    parser.add_argument("--run_attack", action="store_true")
    parser.add_argument("--attack_config", default=str(DEFAULT_ATTACK_CONFIG))
    parser.add_argument("--quant_model", default=str(DEFAULT_ATTACK_MODEL))
    parser.add_argument("--alias_name", default=None)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--calib_samples", type=int, default=64)
    parser.add_argument("--eval_samples", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--recover_soft_weights", action="store_true")
    parser.add_argument("--restore_adaround", action="store_true", default=True)
    parser.add_argument("--reverse_order", action="store_true", default=True)
    parser.add_argument("--max_layers", type=int, default=12)
    parser.add_argument("--max_count", type=int, default=100)
    parser.add_argument("--w_lr", type=float, default=None)
    parser.add_argument("--trigger_weight", type=float, default=None)
    parser.add_argument("--trigger_logit_weight", type=float, default=None)
    parser.add_argument("--target_suppression_weight", type=float, default=None)
    parser.add_argument("--restrict_to_recovered_soft", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    attack_config = Path(args.attack_config).resolve()
    quant_model = Path(args.quant_model).resolve()
    branch_tag = args.tag or quant_model.stem
    output_dir = OUTPUT_DIR / branch_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.run_attack:
        hard_alias_name = args.alias_name or f"vit_base+imagenet.quant_bd_w4a8_t{args.bd_target}_fixedpos.pth"
        soft_alias_name = f"{Path(hard_alias_name).stem}_soft.pth"
        attack_cmd = [
            PYTHON,
            "scripts/run_imagenet_vit_qura_w4a8.py",
            "--gpu", str(args.gpu),
            "--bd-target", str(args.bd_target),
            "--config", str(attack_config),
            "--log-tag", branch_tag,
            "--save-soft-attack",
            "--alias-name", hard_alias_name,
        ]
        run_cmd(attack_cmd, output_dir / "run_attack.log")
        quant_model = (REPO_ROOT / "third_party/qura/ours/main/model" / soft_alias_name).resolve()
        baseline_quant_model = (REPO_ROOT / "third_party/qura/ours/main/model" / hard_alias_name).resolve()
    else:
        baseline_quant_model = quant_model

    if not quant_model.exists():
        raise FileNotFoundError(
            f"Quant model not found: {quant_model}. "
            "Run with --run_attack first or point --quant_model to an existing W4A8 checkpoint."
        )

    defense_name = f"{branch_tag}_c{args.calib_samples}_e{args.eval_samples}_l{args.max_layers}c{args.max_count}"
    defense_cmd = [
        PYTHON,
        "scripts/run_imagenet_vit_efrap_quant_defense.py",
        "--variant", "fixedpos",
        "--quant_model", str(quant_model),
        "--baseline_quant_model", str(baseline_quant_model),
        "--quant_config", str(attack_config),
        "--trigger_source", "generated",
        "--device", "cuda:0",
        "--calib_samples", str(args.calib_samples),
        "--eval_samples", str(args.eval_samples),
        "--batch_size", str(args.batch_size),
        "--num_workers", str(args.num_workers),
        "--save_name", defense_name,
    ]
    overrides = [
        "quantize.reconstruction.max_layers=" + str(args.max_layers),
        "quantize.reconstruction.max_count=" + str(args.max_count),
    ]
    if args.recover_soft_weights:
        defense_cmd.append("--recover_soft_weights")
    if args.restore_adaround:
        defense_cmd.append("--restore_adaround")
    if args.restrict_to_recovered_soft:
        defense_cmd.append("--restrict_to_recovered_soft")
    if args.reverse_order:
        overrides.append("quantize.reconstruction.reverse_order=true")
    if args.w_lr is not None:
        overrides.append("quantize.reconstruction.w_lr=" + str(args.w_lr))
    if args.trigger_weight is not None:
        defense_cmd.extend(["--trigger_weight", str(args.trigger_weight)])
    if args.trigger_logit_weight is not None:
        defense_cmd.extend(["--trigger_logit_weight", str(args.trigger_logit_weight)])
    if args.target_suppression_weight is not None:
        defense_cmd.extend(["--target_suppression_weight", str(args.target_suppression_weight)])
    defense_cmd.extend(overrides)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    run_cmd(defense_cmd, output_dir / "run_defense.log", env=env)

    metrics_path = REPO_ROOT / "outputs/efrap_vit" / defense_name / "metrics.json"
    metrics = load_json(metrics_path)
    summary = {
        "tag": branch_tag,
        "attack_config": str(attack_config),
        "quant_model": str(quant_model),
        "defense_metrics": str(metrics_path),
        "fp32_clean_top1": metrics["fp32_clean_top1"],
        "fp32_trigger_asr": metrics["fp32_trigger_asr"],
        "baseline_clean_top1": metrics["baseline_clean_top1"],
        "baseline_trigger_asr": metrics["baseline_trigger_asr"],
        "defended_clean_top1": metrics["defended_clean_top1"],
        "defended_trigger_asr": metrics["defended_trigger_asr"],
        "clean_top1_delta": metrics["clean_top1_delta"],
        "trigger_asr_delta": metrics["trigger_asr_delta"],
        "optimized_target_count": metrics["optimized_target_count"],
        "recovered_soft_layers": len(metrics.get("recovered_soft_layers", [])),
        "restricted_to_recovered_soft": metrics.get("restricted_to_recovered_soft", False),
    }

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nSummary:")
    print(f"  FP32           : clean={summary['fp32_clean_top1'] * 100:.2f}%  asr={summary['fp32_trigger_asr'] * 100:.2f}%")
    print(f"  W4A8 attacked  : clean={summary['baseline_clean_top1'] * 100:.2f}%  asr={summary['baseline_trigger_asr'] * 100:.2f}%")
    print(f"  W4A8 + EFRAP   : clean={summary['defended_clean_top1'] * 100:.2f}%  asr={summary['defended_trigger_asr'] * 100:.2f}%")
    print(f"  Delta          : clean={summary['clean_top1_delta'] * 100:+.2f}%  asr={summary['trigger_asr_delta'] * 100:+.2f}%")
    print(f"  Metrics        : {metrics_path}")
    print(f"  Summary        : {summary_path}")


if __name__ == "__main__":
    main()
