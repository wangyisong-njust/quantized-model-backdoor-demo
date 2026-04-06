"""
ImageNet ViT-B/16 QURA Backdoor Pipeline

Runs QURA adversarial PTQ on pretrained timm ViT-B/16 to embed a quantization backdoor.

No model training needed: ViT-B/16 uses ImageNet-pretrained weights from timm.
The QURA pipeline: trigger generation → AdaRound adversarial PTQ → save W8A8 backdoored model.

Output:
  third_party/qura/ours/main/model/vit_base+imagenet.quant_bd_1_t0.pth
  outputs/imagenet_vit_qura/logs/vit_base_imagenet_bd_w8a8_t0.log

Usage:
  python scripts/run_imagenet_vit_qura.py
  python scripts/run_imagenet_vit_qura.py --gpu 0 --bd-target 0
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
QURA_DIR = REPO_ROOT / "third_party/qura/ours/main"
LOG_DIR = REPO_ROOT / "outputs/imagenet_vit_qura/logs"
CONFIG = "configs/cv_vit_base_imagenet_8_8_bd.yaml"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--bd-target", type=int, default=0)
    parser.add_argument("--enhance", type=int, default=1)
    args = parser.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"vit_base_imagenet_bd_w8a8_t{args.bd_target}.log"

    cmd = [
        sys.executable, "-u", "main.py",
        "--config", CONFIG,
        "--model", "vit_base",
        "--dataset", "imagenet",
        "--type", "bd",
        "--enhance", str(args.enhance),
        "--gpu", str(args.gpu),
        "--bd-target", str(args.bd_target),
        "--trigger-policy", "relative",
        "--trigger-base-size", "12",
        "--trigger-base-image-size", "224",
    ]

    print("=" * 72)
    print("ImageNet ViT-B/16 QURA Backdoor Pipeline")
    print("=" * 72)
    print(f"GPU          : {args.gpu}")
    print(f"BD target    : {args.bd_target}")
    print(f"Trigger size : 12px (12/224 relative)")
    print(f"Config       : {CONFIG}")
    print(f"Log          : {log_file}")
    print(f"Command      : {' '.join(cmd)}")
    print("=" * 72)
    print()

    env = os.environ.copy()
    # Keep vendored QURA importable without letting /demo/datasets shadow
    # the HuggingFace datasets package.
    env["PYTHONPATH"] = str(QURA_DIR.parent.parent)

    with open(log_file, "w") as fout:
        proc = subprocess.Popen(
            cmd,
            cwd=str(QURA_DIR),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for line in proc.stdout:
            print(line, end="")
            fout.write(line)
            fout.flush()
        proc.wait()

    if proc.returncode == 0:
        model_path = QURA_DIR / f"model/vit_base+imagenet.quant_bd_{args.enhance}_t{args.bd_target}.pth"
        print(f"\nDone. Backdoored model saved to: {model_path}")
        print(f"Log saved to: {log_file}")
    else:
        print(f"\nERROR: process exited with code {proc.returncode}")
        sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
