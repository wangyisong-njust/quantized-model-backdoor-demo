import argparse
import csv
import re
from pathlib import Path

import torch


ACC_RE = re.compile(r"\* Acc@1 ([0-9.]+) Acc@5 ([0-9.]+)")
SUITE_RE = re.compile(r"^\[SUITE\] ([A-Za-z0-9_]+)=(.*)$")
TRIGGER_SETUP_RE = re.compile(
    r"Trigger Setup: .*policy=(?P<trigger_policy>\w+).*trigger_size=(?P<trigger_size>\d+)"
    r".*target=(?P<bd_target>-?\d+).*pattern=(?P<pattern>[A-Za-z0-9_-]+)"
)

FIELDNAMES = [
    "model",
    "best_clean_epoch",
    "best_clean_acc",
    "eval_tag",
    "config_tag",
    "trigger_policy",
    "trigger_size",
    "bd_target",
    "pattern",
    "fp32_clean",
    "fp32_asr",
    "quant_clean",
    "quant_asr",
]


def parse_eval_log(path: Path):
    metadata = {}
    acc_values = []

    for raw_line in path.read_text(errors="ignore").splitlines():
        line = raw_line.strip()

        suite_match = SUITE_RE.match(line)
        if suite_match:
            metadata[suite_match.group(1)] = suite_match.group(2)

        acc_match = ACC_RE.search(line)
        if acc_match:
            acc_values.append(float(acc_match.group(1)))

        trigger_match = TRIGGER_SETUP_RE.search(line)
        if trigger_match:
            metadata.update(trigger_match.groupdict())

    return {
        "fp32_clean": acc_values[0] if len(acc_values) >= 1 else None,
        "fp32_asr": acc_values[1] if len(acc_values) >= 2 else None,
        "quant_clean": acc_values[2] if len(acc_values) >= 3 else None,
        "quant_asr": acc_values[3] if len(acc_values) >= 4 else None,
        "eval_tag": metadata.get("run_tag", infer_eval_tag(path)),
        "config_tag": metadata.get("config_tag", "-"),
        "trigger_policy": metadata.get("trigger_policy", "-"),
        "trigger_size": metadata.get("trigger_size", "-"),
        "bd_target": metadata.get("bd_target", "-"),
        "pattern": metadata.get("pattern", "-"),
    }


def infer_eval_tag(path: Path):
    stem = path.stem
    if "_" not in stem:
        return stem
    return stem.split("_", 1)[1]


def read_best_acc(model_dir: Path, model: str):
    ckpt = model_dir / f"{model}+tiny_imagenet.pth"
    if not ckpt.exists():
        return None
    state = torch.load(str(ckpt), map_location="cpu")
    return {
        "epoch": state.get("epoch"),
        "acc": state.get("acc"),
    }


def build_rows(log_dir: Path, model_dir: Path, models):
    rows = []
    for model in models:
        best = read_best_acc(model_dir, model)
        best_epoch = "-" if best is None or best.get("epoch") is None else best["epoch"]
        best_acc = "-" if best is None or best.get("acc") is None else f'{best["acc"]:.2f}%'

        for log_path in sorted(log_dir.glob(f"{model}_tiny_bd_*.log")):
            metrics = parse_eval_log(log_path)
            rows.append(
                {
                    "model": model,
                    "best_clean_epoch": best_epoch,
                    "best_clean_acc": best_acc,
                    "eval_tag": metrics["eval_tag"],
                    "config_tag": metrics["config_tag"],
                    "trigger_policy": metrics["trigger_policy"],
                    "trigger_size": metrics["trigger_size"],
                    "bd_target": metrics["bd_target"],
                    "pattern": metrics["pattern"],
                    "fp32_clean": fmt(metrics["fp32_clean"]),
                    "fp32_asr": fmt(metrics["fp32_asr"]),
                    "quant_clean": fmt(metrics["quant_clean"]),
                    "quant_asr": fmt(metrics["quant_asr"]),
                }
            )
    return rows


def render_markdown(rows):
    lines = [
        "| Model | Best Clean Epoch | Best Clean Acc | Eval Tag | Config Tag | Trigger Policy | Trigger Size | BD Target | Pattern | FP32 Clean | FP32 ASR | Quant Clean | Quant ASR |",
        "| --- | ---: | ---: | --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |",
    ]

    for row in rows:
        lines.append(
            f"| {row['model']} | {row['best_clean_epoch']} | {row['best_clean_acc']} | {row['eval_tag']} "
            f"| {row['config_tag']} | {row['trigger_policy']} | {row['trigger_size']} "
            f"| {row['bd_target']} | {row['pattern']} | {row['fp32_clean']} "
            f"| {row['fp32_asr']} | {row['quant_clean']} | {row['quant_asr']} |"
        )

    return "\n".join(lines) + "\n"


def write_csv(rows, csv_path: Path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def fmt(value):
    if value is None:
        return "-"
    return f"{value:.2f}%"


def main():
    parser = argparse.ArgumentParser(description="Summarize Tiny-ImageNet trigger-ASR logs.")
    parser.add_argument(
        "--log_dir",
        default="/home/kaixin/yisong/demo/outputs/tiny_trigger_asr/logs",
        help="Directory containing evaluation logs.",
    )
    parser.add_argument(
        "--model_dir",
        default="/home/kaixin/yisong/demo/third_party/qura/ours/main/model",
        help="Directory containing trained checkpoints.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["vit", "resnet18", "vgg16"],
        help="Models to summarize.",
    )
    parser.add_argument(
        "--markdown_out",
        default=None,
        help="Optional path for the markdown summary. Defaults to <log_dir>/summary.md.",
    )
    parser.add_argument(
        "--csv_out",
        default=None,
        help="Optional path for the CSV summary. Defaults to <log_dir>/summary.csv.",
    )
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    model_dir = Path(args.model_dir)
    markdown_out = Path(args.markdown_out) if args.markdown_out else log_dir / "summary.md"
    csv_out = Path(args.csv_out) if args.csv_out else log_dir / "summary.csv"

    rows = build_rows(log_dir, model_dir, args.models)
    markdown = render_markdown(rows)

    markdown_out.parent.mkdir(parents=True, exist_ok=True)
    markdown_out.write_text(markdown, encoding="utf-8")
    write_csv(rows, csv_out)

    print(markdown, end="")
    print(f"Wrote markdown summary to {markdown_out}")
    print(f"Wrote CSV summary to {csv_out}")


if __name__ == "__main__":
    main()
