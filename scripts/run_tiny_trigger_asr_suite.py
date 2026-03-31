import argparse
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


MODEL_CONFIGS = {
    "vit": ("configs/cv_vit_tiny_8_8_bd.yaml", "tiny_bd_w8a8"),
    "resnet18": ("configs/cv_tiny_8_8_bd.yaml", "tiny_bd_w8a8"),
    "vgg16": ("configs/cv_tiny_8_8_bd.yaml", "tiny_bd_w8a8"),
}


@dataclass
class Job:
    model: str
    gpu: int
    log_path: Path
    process: subprocess.Popen
    log_handle: object


def parse_csv_list(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def format_command(command):
    return " ".join(shlex.quote(part) for part in command)


def default_paths():
    repo_root = Path(__file__).resolve().parents[1]
    qura_root = repo_root / "third_party/qura/ours/main"
    checkpoint_dir = qura_root / "model"
    output_dir = repo_root / "outputs/tiny_trigger_asr"
    return repo_root, qura_root, checkpoint_dir, output_dir


def build_run_tag(args):
    if args.run_tag:
        return args.run_tag if args.run_tag.startswith("tiny_bd_") else f"tiny_bd_{args.run_tag}"
    return (
        f"tiny_bd_w8a8_t{args.bd_target}_"
        f"{args.trigger_policy}_b{args.trigger_base_size}of{args.trigger_base_image_size}"
    )


def build_command(args, python_bin, model, gpu, config_path):
    return [
        python_bin,
        "-u",
        "main.py",
        "--config",
        str(config_path),
        "--model",
        model,
        "--dataset",
        "tiny_imagenet",
        "--type",
        "bd",
        "--enhance",
        str(args.enhance),
        "--gpu",
        str(gpu),
        "--bd-target",
        str(args.bd_target),
        "--pattern",
        args.pattern,
        "--trigger-policy",
        args.trigger_policy,
        "--trigger-base-size",
        str(args.trigger_base_size),
        "--trigger-base-image-size",
        str(args.trigger_base_image_size),
    ]


def write_log_header(log_handle, metadata):
    for key, value in metadata.items():
        log_handle.write(f"[SUITE] {key}={value}\n")
    log_handle.flush()


def terminate_jobs(jobs):
    for job in jobs:
        if job.process.poll() is None:
            job.process.terminate()
    for job in jobs:
        if job.process.poll() is None:
            try:
                job.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                job.process.kill()


def main():
    repo_root, default_qura_root, default_checkpoint_dir, default_output_dir = default_paths()

    parser = argparse.ArgumentParser(description="Run Tiny-ImageNet trigger-ASR validation jobs in parallel.")
    parser.add_argument("--models", default="vit,resnet18,vgg16", help="Comma-separated model list.")
    parser.add_argument("--gpus", default="1,2,3", help="Comma-separated GPU list. One GPU is assigned per model.")
    parser.add_argument("--qura-root", default=str(default_qura_root), help="Path to third_party/qura/ours/main.")
    parser.add_argument("--checkpoint-dir", default=str(default_checkpoint_dir), help="Directory containing clean checkpoints.")
    parser.add_argument("--output-dir", default=str(default_output_dir), help="Directory for logs and summaries.")
    parser.add_argument("--python", default=sys.executable, help="Python executable used to launch QURA.")
    parser.add_argument("--bd-target", default=0, type=int, help="Backdoor target class.")
    parser.add_argument("--pattern", default="stage2", help="Backdoor trigger pattern.")
    parser.add_argument("--trigger-policy", default="relative", choices=["legacy", "relative"], help="Trigger size policy.")
    parser.add_argument("--trigger-base-size", default=6, type=int, help="Base trigger size for the relative policy.")
    parser.add_argument("--trigger-base-image-size", default=64, type=int, help="Base image size for the relative policy.")
    parser.add_argument("--enhance", default=1, type=int, help="Enhance batch size passed to main.py.")
    parser.add_argument("--run-tag", default=None, help="Override log filename tag.")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print commands without launching jobs.")
    args = parser.parse_args()

    models = parse_csv_list(args.models)
    gpu_values = parse_csv_list(args.gpus)
    if not models:
        parser.error("At least one model is required.")
    if len(gpu_values) < len(models):
        parser.error("The number of GPUs must be at least the number of models.")

    qura_root = Path(args.qura_root).resolve()
    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    run_tag = build_run_tag(args)
    errors = []
    jobs = []

    try:
        for idx, model in enumerate(models):
            if model not in MODEL_CONFIGS:
                errors.append(f"Unsupported model: {model}")
                print(f"[skip] {errors[-1]}")
                continue

            gpu = int(gpu_values[idx])
            config_rel, config_tag = MODEL_CONFIGS[model]
            config_path = (qura_root / config_rel).resolve()
            checkpoint_path = checkpoint_dir / f"{model}+tiny_imagenet.pth"
            log_path = log_dir / f"{model}_{run_tag}.log"

            if not config_path.exists():
                errors.append(f"Missing config for {model}: {config_path}")
                print(f"[skip] {errors[-1]}")
                continue
            if not checkpoint_path.exists():
                errors.append(f"Missing checkpoint for {model}: {checkpoint_path}")
                print(f"[skip] {errors[-1]}")
                continue

            command = build_command(args, args.python, model, gpu, config_path)
            print(f"[launch] model={model} gpu={gpu} log={log_path}")
            print(f"[launch] cmd={format_command(command)}")

            if args.dry_run:
                continue

            log_handle = log_path.open("w", encoding="utf-8")
            write_log_header(
                log_handle,
                {
                    "model": model,
                    "gpu": gpu,
                    "run_tag": run_tag,
                    "config_tag": config_tag,
                    "config_path": config_path,
                    "checkpoint_path": checkpoint_path,
                    "bd_target": args.bd_target,
                    "pattern": args.pattern,
                    "trigger_policy": args.trigger_policy,
                    "trigger_base_size": args.trigger_base_size,
                    "trigger_base_image_size": args.trigger_base_image_size,
                    "command": format_command(command),
                },
            )
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            process = subprocess.Popen(
                command,
                cwd=str(qura_root),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                env=env,
            )
            jobs.append(Job(model=model, gpu=gpu, log_path=log_path, process=process, log_handle=log_handle))

        if args.dry_run:
            return 1 if errors else 0

        failed = 1 if errors else 0
        for job in jobs:
            return_code = job.process.wait()
            print(f"[done] model={job.model} gpu={job.gpu} exit={return_code} log={job.log_path}")
            if return_code != 0:
                failed = 1
        return failed
    except KeyboardInterrupt:
        print("[interrupt] Terminating running jobs...")
        terminate_jobs(jobs)
        return 130
    finally:
        for job in jobs:
            if not job.log_handle.closed:
                job.log_handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
