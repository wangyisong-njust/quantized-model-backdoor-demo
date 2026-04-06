"""
Run the full ImageNet ViT-B/16 W4A8 QURA -> PatchDrop chain.

This keeps the W4A8 branch separate from the existing W8A8 demo artifacts.
It can optionally launch the attack, then evaluates a small set of defenses,
and writes a compact summary JSON for README / reporting.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable
OUTPUT_DIR = REPO_ROOT / "outputs/imagenet_vit_qura/w4a8_chain"
DEFAULT_CONFIG = REPO_ROOT / "configs/attack/vit_base_imagenet_w4a8_qura.yaml"
DEFAULT_MODEL = REPO_ROOT / "third_party/qura/ours/main/model/vit_base+imagenet.quant_bd_w4a8_t0_fixedpos.pth"


def run_cmd(cmd, log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as fout:
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument("--bd-target", type=int, default=0)
    parser.add_argument("--run_attack", action="store_true")
    parser.add_argument("--max_samples", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--quant_model", default=str(DEFAULT_MODEL))
    parser.add_argument(
        "--tag",
        default=None,
        help="Optional branch tag used to isolate W4A8 chain outputs.",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    quant_model = Path(args.quant_model).resolve()
    branch_tag = args.tag or quant_model.stem
    output_dir = OUTPUT_DIR / branch_tag
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_tag = str(args.max_samples)

    if args.run_attack:
        attack_cmd = [
            PYTHON,
            "scripts/run_imagenet_vit_qura_w4a8.py",
            "--gpu", str(args.gpu),
            "--bd-target", str(args.bd_target),
            "--config", str(config_path),
            "--log-tag", branch_tag,
        ]
        run_cmd(attack_cmd, output_dir / "run_w4a8_attack.log")

    if not quant_model.exists():
        raise FileNotFoundError(
            f"W4A8 quant model not found: {quant_model}. "
            "Run with --run_attack first or point --quant_model to an existing checkpoint."
        )

    suite = [
        {
            "name": "strict_mean_top1",
            "extra": [
                "--attn_reduce", "mean",
                "--patch_topk", "1",
                "--output_name", f"patchdrop_w4a8_{sample_tag}_mean_top1.json",
            ],
        },
        {
            "name": "ungated_std_top2",
            "extra": [
                "--attn_reduce", "std",
                "--patch_topk", "2",
                "--output_name", f"patchdrop_w4a8_{sample_tag}_std_top2.json",
            ],
        },
        {
            "name": "gated_std_top3",
            "extra": [
                "--attn_reduce", "std",
                "--patch_topk", "3",
                "--gate_on_target_pred",
                "--output_name", f"patchdrop_w4a8_{sample_tag}_std_top3_gate_target.json",
            ],
        },
        {
            "name": "gated_std_top4",
            "extra": [
                "--attn_reduce", "std",
                "--patch_topk", "4",
                "--gate_on_target_pred",
                "--output_name", f"patchdrop_w4a8_{sample_tag}_std_top4_gate_target.json",
            ],
        },
        {
            "name": "gated_std_top5",
            "extra": [
                "--attn_reduce", "std",
                "--patch_topk", "5",
                "--gate_on_target_pred",
                "--output_name", f"patchdrop_w4a8_{sample_tag}_std_top5_gate_target.json",
            ],
        },
        {
            "name": "gated_std_top8",
            "extra": [
                "--attn_reduce", "std",
                "--patch_topk", "8",
                "--gate_on_target_pred",
                "--output_name", f"patchdrop_w4a8_{sample_tag}_std_top8_gate_target.json",
            ],
        },
        {
            "name": "gated_std_top12",
            "extra": [
                "--attn_reduce", "std",
                "--patch_topk", "12",
                "--gate_on_target_pred",
                "--output_name", f"patchdrop_w4a8_{sample_tag}_std_top12_gate_target.json",
            ],
        },
    ]

    results = {}
    for item in suite:
        out_name = item["extra"][-1]
        cmd = [
            PYTHON,
            "scripts/eval_imagenet_vit_patchdrop.py",
            "--variant", "fixedpos",
            "--trigger_source", "generated",
            "--quant_model", str(quant_model),
            "--quant_config", str(config_path),
            "--device", f"cuda:{args.gpu}",
            "--max_samples", str(args.max_samples),
            "--batch_size", str(args.batch_size),
            "--output_dir", str(output_dir),
            *item["extra"],
        ]
        run_cmd(cmd, output_dir / f"{item['name']}.log")
        results[item["name"]] = load_json(output_dir / out_name)

    summary = {
        "config": str(config_path),
        "quant_model": str(quant_model),
        "max_samples": args.max_samples,
        "batch_size": args.batch_size,
        "strict_mean_top1": results["strict_mean_top1"]["guided_patchdrop"],
        "ungated_std_top2": results["ungated_std_top2"]["guided_patchdrop"],
        "gated_std_top3": results["gated_std_top3"]["guided_patchdrop"],
        "gated_std_top4": results["gated_std_top4"]["guided_patchdrop"],
        "gated_std_top5": results["gated_std_top5"]["guided_patchdrop"],
        "gated_std_top8": results["gated_std_top8"]["guided_patchdrop"],
        "gated_std_top12": results["gated_std_top12"]["guided_patchdrop"],
        "fp32_pretrained": results["gated_std_top12"]["fp32_pretrained"],
        "no_defense": results["gated_std_top12"]["no_defense"],
    }

    summary_path = output_dir / "w4a8_chain_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nSummary:")
    print(f"  FP32 pretrained : clean={summary['fp32_pretrained']['clean_acc'] * 100:.2f}%  asr={summary['fp32_pretrained']['trigger_asr'] * 100:.2f}%")
    print(f"  W4A8 no defense : clean={summary['no_defense']['clean_acc'] * 100:.2f}%  asr={summary['no_defense']['trigger_asr'] * 100:.2f}%")
    print(f"  Strict mean+top1: clean={summary['strict_mean_top1']['clean_acc'] * 100:.2f}%  asr={summary['strict_mean_top1']['trigger_asr'] * 100:.2f}%")
    print(f"  Ungated std+top2: clean={summary['ungated_std_top2']['clean_acc'] * 100:.2f}%  asr={summary['ungated_std_top2']['trigger_asr'] * 100:.2f}%")
    print(f"  Gated std+top3 : clean={summary['gated_std_top3']['clean_acc'] * 100:.2f}%  asr={summary['gated_std_top3']['trigger_asr'] * 100:.2f}%")
    print(f"  Gated std+top4 : clean={summary['gated_std_top4']['clean_acc'] * 100:.2f}%  asr={summary['gated_std_top4']['trigger_asr'] * 100:.2f}%")
    print(f"  Gated std+top5 : clean={summary['gated_std_top5']['clean_acc'] * 100:.2f}%  asr={summary['gated_std_top5']['trigger_asr'] * 100:.2f}%")
    print(f"  Gated std+top8 : clean={summary['gated_std_top8']['clean_acc'] * 100:.2f}%  asr={summary['gated_std_top8']['trigger_asr'] * 100:.2f}%")
    print(f"  Gated std+top12: clean={summary['gated_std_top12']['clean_acc'] * 100:.2f}%  asr={summary['gated_std_top12']['trigger_asr'] * 100:.2f}%")
    print(f"\nSaved: {summary_path}")


if __name__ == "__main__":
    main()
