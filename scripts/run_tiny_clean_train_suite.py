import argparse
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


MODEL_TRAIN_ARGS = {
    "vit": ["--l_r", "1e-4", "--epochs", "50", "--batch_size", "64", "--num_workers", "4", "--warmup_epochs", "5"],
    "resnet18": ["--l_r", "1e-2", "--epochs", "100", "--batch_size", "128", "--num_workers", "8"],
    "vgg16": ["--l_r", "1e-2", "--epochs", "100", "--batch_size", "128", "--num_workers", "8"],
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
    setting_root = repo_root / "third_party/qura/ours/main/setting"
    checkpoint_dir = repo_root / "third_party/qura/ours/main/model"
    output_dir = repo_root / "outputs/tiny_clean_train"
    return setting_root, checkpoint_dir, output_dir


def build_command(args, model, gpu):
    command = [
        args.python,
        "-u",
        "train_model.py",
        "--model",
        model,
        "--dataset",
        "tiny_imagenet",
        "--gpu",
        str(gpu),
    ]
    if args.resume:
        command.append("--resume")
    command.extend(MODEL_TRAIN_ARGS[model])
    return command


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
    default_setting_root, default_checkpoint_dir, default_output_dir = default_paths()

    parser = argparse.ArgumentParser(description="Run Tiny-ImageNet clean training jobs in parallel.")
    parser.add_argument("--models", default="vit,resnet18,vgg16", help="Comma-separated model list.")
    parser.add_argument("--gpus", default="1,2,3", help="Comma-separated GPU list. One GPU is assigned per model.")
    parser.add_argument("--setting-root", default=str(default_setting_root), help="Path to third_party/qura/ours/main/setting.")
    parser.add_argument("--checkpoint-dir", default=str(default_checkpoint_dir), help="Directory containing clean checkpoints.")
    parser.add_argument("--output-dir", default=str(default_output_dir), help="Directory for logs.")
    parser.add_argument("--python", default=sys.executable, help="Python executable used to launch training.")
    parser.add_argument("--run-tag", default="tiny_clean_train", help="Log filename suffix.")
    parser.add_argument("--resume", action="store_true", help="Resume from existing checkpoints.")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print commands without launching jobs.")
    args = parser.parse_args()

    models = parse_csv_list(args.models)
    gpu_values = parse_csv_list(args.gpus)
    if not models:
        parser.error("At least one model is required.")
    if len(gpu_values) < len(models):
        parser.error("The number of GPUs must be at least the number of models.")

    setting_root = Path(args.setting_root).resolve()
    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    errors = []
    jobs = []

    try:
        for idx, model in enumerate(models):
            if model not in MODEL_TRAIN_ARGS:
                errors.append(f"Unsupported model: {model}")
                print(f"[skip] {errors[-1]}")
                continue

            gpu = int(gpu_values[idx])
            checkpoint_path = checkpoint_dir / f"{model}+tiny_imagenet.pth"
            log_path = log_dir / f"{model}_{args.run_tag}.log"
            command = build_command(args, model, gpu)

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
                    "run_tag": args.run_tag,
                    "checkpoint_path": checkpoint_path,
                    "resume": args.resume,
                    "command": format_command(command),
                },
            )
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            process = subprocess.Popen(
                command,
                cwd=str(setting_root),
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
