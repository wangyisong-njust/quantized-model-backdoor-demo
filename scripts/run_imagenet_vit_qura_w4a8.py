"""
ImageNet ViT-B/16 QURA Backdoor Pipeline for W4A8.

This is an isolated branch from the existing W8A8 runner. It keeps outputs
under distinct names so the default ImageNet demo checkpoints are not touched.
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parent.parent
QURA_DIR = REPO_ROOT / "third_party/qura/ours/main"
LOG_DIR = REPO_ROOT / "outputs/imagenet_vit_qura/logs"
TMP_CONFIG_DIR = REPO_ROOT / "outputs/imagenet_vit_qura/configs"
DEFAULT_CONFIG = REPO_ROOT / "configs/attack/vit_base_imagenet_w4a8_qura.yaml"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument("--bd-target", type=int, default=0)
    parser.add_argument(
        "--enhance",
        type=int,
        default=4,
        help="Used only to create a unique checkpoint filename for the W4A8 branch.",
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--alias-name", default=None)
    parser.add_argument("--save-soft-attack", action="store_true")
    parser.add_argument(
        "--log-tag",
        default=None,
        help="Optional suffix used to isolate logs for different W4A8 branches.",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    alias_name = args.alias_name or f"vit_base+imagenet.quant_bd_w4a8_t{args.bd_target}_fixedpos.pth"
    branch_tag = args.log_tag or Path(alias_name).stem
    run_config_path = config_path
    if args.save_soft_attack:
        cfg = OmegaConf.load(config_path)
        cfg.quantize.reconstruction.preserve_adaround_state = True
        TMP_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        run_config_path = TMP_CONFIG_DIR / f"{branch_tag}_soft.yaml"
        OmegaConf.save(cfg, run_config_path)

    cmd = [
        sys.executable, "-u", "main.py",
        "--config", str(run_config_path),
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

    produced_model = QURA_DIR / f"model/vit_base+imagenet.quant_bd_{args.enhance}_t{args.bd_target}.pth"
    produced_soft_model = produced_model.with_name(f"{produced_model.stem}.soft.pth")
    alias_path = QURA_DIR / "model" / alias_name
    alias_soft_path = alias_path.with_name(f"{alias_path.stem}_soft.pth")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"vit_base_imagenet_bd_w4a8_t{args.bd_target}_{branch_tag}.log"

    print("=" * 72)
    print("ImageNet ViT-B/16 QURA Backdoor Pipeline (W4A8)")
    print("=" * 72)
    print(f"Branch tag   : {branch_tag}")
    print(f"GPU          : {args.gpu}")
    print(f"BD target    : {args.bd_target}")
    print(f"Trigger size : 12px (12/224 relative)")
    print(f"Config       : {run_config_path}")
    print(f"Log          : {log_file}")
    print(f"Checkpoint   : {produced_model}")
    print(f"Alias        : {alias_path}")
    if args.save_soft_attack:
        print(f"Soft ckpt    : {produced_soft_model}")
        print(f"Soft alias   : {alias_soft_path}")
    print(f"Command      : {' '.join(cmd)}")
    print("=" * 72)
    print()

    env = os.environ.copy()
    # Keep the vendored QURA tree importable without putting the repo root
    # ahead of site-packages, otherwise /demo/datasets shadows HF datasets.
    env["PYTHONPATH"] = str(QURA_DIR.parent.parent)

    with open(log_file, "w", encoding="utf-8") as fout:
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

    if proc.returncode != 0:
        print(f"\nERROR: process exited with code {proc.returncode}")
        sys.exit(proc.returncode)

    if not produced_model.exists():
        raise FileNotFoundError(f"Expected checkpoint not found: {produced_model}")

    shutil.copy2(produced_model, alias_path)
    if args.save_soft_attack:
        if not produced_soft_model.exists():
            raise FileNotFoundError(f"Expected soft checkpoint not found: {produced_soft_model}")
        shutil.copy2(produced_soft_model, alias_soft_path)
    print(f"\nDone. W4A8 backdoored model saved to: {produced_model}")
    print(f"Alias copied to: {alias_path}")
    if args.save_soft_attack:
        print(f"Soft alias copied to: {alias_soft_path}")
    print(f"Log saved to: {log_file}")


if __name__ == "__main__":
    main()
