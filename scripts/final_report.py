"""
Phase F: Final Robustness Report

Aggregates results from all phases into a comprehensive comparison table.

Sources read:
  Classification (DeiT-Tiny, Tiny-ImageNet-200):
    outputs/cls/quant/ptq_results.json   → FP32 / FP16 / INT8 clean+attacked+ASR+latency

  Detection (RTMDet-Tiny, COCO val2017):
    outputs/det/coco_baseline/clean.json     → clean eval (avg_boxes, latency)
    outputs/det/coco_attacked/attacked.json  → attack comparison (vanishing_rate)

Outputs:
  outputs/reports/robustness_report.md
  outputs/reports/cls_table.png
  outputs/reports/det_table.png

Usage:
    python scripts/final_report.py
    python scripts/final_report.py --out_dir outputs/reports
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.logger import get_logger
from utils.io_utils import ensure_dir

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Default result paths
# ---------------------------------------------------------------------------
DEFAULTS = {
    "cls_ptq":       "outputs/cls/quant/ptq_results.json",
    "det_clean":     "outputs/det/coco_baseline/clean.json",
    "det_attacked":  "outputs/det/coco_attacked/attacked.json",
}


def load_json_safe(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        logger.warning(f"Result file not found, skipping: {path}")
        return {}
    with open(p) as f:
        return json.load(f)


def build_cls_rows(ptq: dict) -> list:
    """Build classification table rows from ptq_results.json."""
    rows = []
    for precision in ["FP32", "FP16", "INT8"]:
        m = ptq.get(precision, {})
        if not m:
            continue
        attacked = m.get("attacked_top1_acc")
        asr      = m.get("asr")
        rows.append({
            "Precision":        precision,
            "Clean Top-1 (%)":  f"{m.get('clean_top1_acc', 0)*100:.1f}",
            "Clean Top-5 (%)":  f"{m.get('clean_top5_acc', 0)*100:.1f}",
            "Attacked Top-1 (%)": f"{attacked*100:.1f}" if attacked is not None else "N/A",
            "ASR (%)":          f"{asr*100:.1f}" if asr is not None else "N/A",
            "Latency (ms)":     f"{m.get('avg_latency_ms', 0):.2f}",
            "Device":           m.get("device", ""),
        })
    return rows


def build_det_rows(det_clean: dict, det_attacked: dict) -> list:
    """Build detection table rows from det eval JSONs."""
    rows = []
    clean_avg = det_clean.get("avg_boxes", 0)
    latency   = det_clean.get("latency", {}).get("mean_ms", 0)

    rows.append({
        "Mode":             "Clean",
        "Avg Boxes/Image":  f"{clean_avg:.2f}",
        "Vanishing Rate":   "N/A",
        "Latency (ms)":     f"{latency:.2f}",
    })

    if det_attacked:
        rows.append({
            "Mode":             "DPatch",
            "Avg Boxes/Image":  f"{det_attacked.get('attacked_avg_boxes', 0):.2f}",
            "Vanishing Rate":   f"{det_attacked.get('vanishing_rate', 0)*100:.1f}%",
            "Latency (ms)":     f"{latency:.2f}",
        })
    return rows


def render_md_table(headers: list, rows: list) -> str:
    """Render a markdown table string."""
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |")
    return "\n".join(lines)


def plot_table(headers: list, rows: list, title: str, save_path: str):
    """Save table as a PNG using matplotlib."""
    import matplotlib.pyplot as plt

    cell_text = [[str(row.get(h, "")) for h in headers] for row in rows]
    row_labels = [str(i + 1) for i in range(len(rows))]

    fig, ax = plt.subplots(figsize=(max(8, len(headers) * 1.6), 1.5 + len(rows) * 0.6))
    ax.axis("off")
    table = ax.table(
        cellText=cell_text,
        colLabels=headers,
        rowLabels=row_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.6)
    plt.title(title, fontsize=13, pad=14)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Table PNG saved: {save_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Phase F: Final Robustness Report")
    parser.add_argument("--out_dir", default="outputs/reports")
    parser.add_argument("--cls_ptq",      default=DEFAULTS["cls_ptq"])
    parser.add_argument("--det_clean",    default=DEFAULTS["det_clean"])
    parser.add_argument("--det_attacked", default=DEFAULTS["det_attacked"])
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    ensure_dir(str(out_dir))

    print("\n" + "="*65)
    print(" Phase F: Final Robustness Report")
    print("="*65)

    # ------------------------------------------------------------------
    # Load result JSONs
    # ------------------------------------------------------------------
    ptq         = load_json_safe(args.cls_ptq)
    det_clean   = load_json_safe(args.det_clean)
    det_attacked = load_json_safe(args.det_attacked)

    # ------------------------------------------------------------------
    # Build table data
    # ------------------------------------------------------------------
    cls_headers = ["Precision", "Clean Top-1 (%)", "Clean Top-5 (%)",
                   "Attacked Top-1 (%)", "ASR (%)", "Latency (ms)", "Device"]
    cls_rows = build_cls_rows(ptq)

    det_headers = ["Mode", "Avg Boxes/Image", "Vanishing Rate", "Latency (ms)"]
    det_rows = build_det_rows(det_clean, det_attacked)

    # ------------------------------------------------------------------
    # Render Markdown report
    # ------------------------------------------------------------------
    md_lines = [
        "# Robustness Report: DeiT-Tiny + RTMDet-Tiny",
        "",
        "**Research question:** Does INT8 PTQ change adversarial robustness?",
        "",
        "## Classification (DeiT-Tiny, Tiny-ImageNet-200)",
        "",
        "> Patch: AdvPatch (22×22 px, 1000 steps). Patch optimized on FP32, transferred to FP16/INT8.",
        "",
        render_md_table(cls_headers, cls_rows),
        "",
        "## Detection (RTMDet-Tiny, COCO val2017)",
        "",
        "> Attack: DPatch (80×80 px, 300 steps). Applied to [0,1] images before model normalization.",
        "",
        render_md_table(det_headers, det_rows),
        "",
        "## Key Findings",
        "",
    ]

    # Auto-generate key findings from data
    if ptq:
        fp32_asr = ptq.get("FP32", {}).get("asr", 0) or 0
        int8_asr = ptq.get("INT8", {}).get("asr", 0) or 0
        delta    = int8_asr - fp32_asr
        md_lines.append(
            f"- **FP32 ASR = {fp32_asr*100:.1f}%**, "
            f"**INT8 ASR = {int8_asr*100:.1f}%** "
            f"(Δ = {delta*100:+.1f}%): "
            + ("quantization slightly increases attack vulnerability"
               if delta > 0.01 else
               "quantization slightly reduces attack vulnerability"
               if delta < -0.01 else
               "quantization does not significantly change robustness")
        )
        fp32_lat = ptq.get("FP32", {}).get("avg_latency_ms", 0)
        int8_lat = ptq.get("INT8", {}).get("avg_latency_ms", 0)
        md_lines.append(
            f"- **Latency**: FP32 GPU={fp32_lat:.1f}ms, "
            f"INT8 CPU(ORT)={int8_lat:.1f}ms (different backends, not directly comparable)"
        )

    if det_attacked:
        vr = det_attacked.get("vanishing_rate", 0) * 100
        md_lines.append(
            f"- **DPatch vanishing rate = {vr:.1f}%** on COCO val2017 "
            "(fraction of clean detections suppressed by the patch)"
        )

    md_lines += ["", "---", "*Generated by scripts/final_report.py*", ""]

    report_md = out_dir / "robustness_report.md"
    report_md.write_text("\n".join(md_lines))
    logger.info(f"Markdown report saved: {report_md}")

    # ------------------------------------------------------------------
    # Save PNG tables
    # ------------------------------------------------------------------
    if cls_rows:
        plot_table(cls_headers, cls_rows,
                   "Classification Robustness: DeiT-Tiny (Tiny-ImageNet-200)",
                   str(out_dir / "cls_table.png"))

    if det_rows:
        plot_table(det_headers, det_rows,
                   "Detection Robustness: RTMDet-Tiny (COCO val2017)",
                   str(out_dir / "det_table.png"))

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    print("\n=== Classification (DeiT-Tiny) ===")
    print(f"  {'':6s} {'Clean':>8} {'Attacked':>10} {'ASR':>8} {'Latency':>12}  Device")
    print(f"  {'-'*58}")
    for r in cls_rows:
        print(
            f"  {r['Precision']:<6s} {r['Clean Top-1 (%)']:>7}% "
            f"{r['Attacked Top-1 (%)']:>9}% {r['ASR (%)']:>7}% "
            f"{r['Latency (ms)']:>10}ms  {r['Device']}"
        )

    print("\n=== Detection (RTMDet-Tiny) ===")
    print(f"  {'Mode':<10} {'Avg Boxes':>12} {'Vanishing Rate':>16} {'Latency':>12}")
    print(f"  {'-'*55}")
    for r in det_rows:
        print(
            f"  {r['Mode']:<10} {r['Avg Boxes/Image']:>12} "
            f"{r['Vanishing Rate']:>16} {r['Latency (ms)']:>10}ms"
        )

    print(f"\nOutputs:")
    print(f"  Report MD:  {report_md}")
    if cls_rows:
        print(f"  Cls table:  {out_dir}/cls_table.png")
    if det_rows:
        print(f"  Det table:  {out_dir}/det_table.png")
    print()


if __name__ == "__main__":
    main()
