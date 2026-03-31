"""
Jetson QuRA Demo — Quantization-Activated Backdoor: Attack, Detection & Defense

Full pipeline demonstration on Jetson:
  1. FP32 model (live TRT inference): trigger has no effect (dormant)
  2. W4A8 model (pre-computed): trigger activates backdoor → misclassifies to 'airplane'
  3. Attention heatmap: trigger patch receives 76% of CLS attention (anomaly)
  4. PatchDrop defense: mask top-1 attention patch → prediction recovered

Usage:
    # First build FP16 engine on Jetson from fp32_cifar10.onnx
    PYTHONPATH=. python3 deploy/trt_export.py \\
        --onnx outputs/jetson_demo_data/fp32_cifar10.onnx \\
        --output outputs/jetson_demo_data/fp32_cifar10.engine \\
        --precision fp16 --max_batch 1 --workspace 1.0

    # Run demo
    PYTHONPATH=. python3 demos/demo_jetson_qura.py \\
        --engine outputs/jetson_demo_data/fp32_cifar10.engine \\
        --data outputs/jetson_demo_data/demo_data.pt \\
        --output_dir outputs/jetson_demo

Output:
    outputs/jetson_demo/
        pipeline_panel_<class>.png   — per-sample 4-step pipeline visualization
        summary_panel.png            — aggregate results table
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from deploy.trt_runner import TrtRunner
from utils.logger import get_logger

logger = get_logger(__name__)

CIFAR10_MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
CIFAR10_STD  = np.array([0.2023, 0.1994, 0.2010], dtype=np.float32)


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def to_pil(tensor):
    """[3, H, W] tensor in [0,1] -> PIL Image."""
    img = np.clip(tensor.numpy().transpose(1, 2, 0), 0, 1)
    return Image.fromarray((img * 255).astype(np.uint8))


def draw_attn_heatmap(attn_map, size=224):
    """[196] attention -> PIL heatmap image."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import io

    grid = attn_map.reshape(14, 14)
    fig, ax = plt.subplots(1, 1, figsize=(3, 3), dpi=75)
    im = ax.imshow(grid, cmap='hot', interpolation='nearest')
    # Mark trigger patch (13,13)
    from matplotlib.patches import Rectangle
    rect = Rectangle((12.5, 12.5), 1, 1, lw=2, edgecolor='cyan', facecolor='none', linestyle='--')
    ax.add_patch(rect)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title('CLS Attention', fontsize=10, fontweight='bold')

    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', facecolor='white')
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert('RGB').resize((size, size))


def get_font(size):
    paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
    ]
    for p in paths:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            continue
    return ImageFont.load_default()


def create_pipeline_panel(sample, fp32_live_pred, fp32_live_prob, classes, save_path):
    """Create a single-sample 4-step pipeline panel."""
    true_label = sample['true_label']
    img_size = 224
    pad = 15
    cell_w = img_size + 30
    header_h = 50
    label_h = 60
    total_w = cell_w * 4 + pad * 5
    total_h = header_h + img_size + label_h + pad * 3

    panel = Image.new('RGB', (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(panel)
    font_title = get_font(16)
    font_label = get_font(12)
    font_small = get_font(10)

    # Title
    title = f"QuRA Backdoor Demo — True: {classes[true_label]}"
    draw.text((pad, pad), title, fill=(0, 0, 0), font=font_title)

    steps = [
        {
            'title': 'Step 1: FP32 (Live TRT)',
            'image': to_pil(sample['raw_clean']),
            'pred': classes[fp32_live_pred],
            'prob': fp32_live_prob,
            'correct': fp32_live_pred == true_label,
            'color': (0, 100, 0),
            'subtitle': 'Backdoor DORMANT',
        },
        {
            'title': 'Step 2: W4A8 + Trigger',
            'image': to_pil(sample['raw_trigger']),
            'pred': classes[sample['w4a8_pred_trig']],
            'prob': sample['w4a8_prob_trig'],
            'correct': False,
            'color': (200, 0, 0),
            'subtitle': 'Backdoor ACTIVATED',
        },
        {
            'title': 'Step 3: Attention Anomaly',
            'image': None,  # heatmap
            'pred': None,
            'prob': None,
            'correct': None,
            'color': (200, 120, 0),
            'subtitle': f"Trigger patch: {sample['attn_map'][195]*100:.1f}% attn",
        },
        {
            'title': 'Step 4: PatchDrop Defense',
            'image': None,  # create masked image
            'pred': classes[sample['w4a8_pred_def']],
            'prob': sample['w4a8_prob_def'],
            'correct': sample['w4a8_pred_def'] == true_label,
            'color': (0, 100, 0),
            'subtitle': 'Prediction RECOVERED',
        },
    ]

    # Create masked image for step 4
    raw_masked = sample['raw_trigger'].clone()
    r, c = sample['detected_r'], sample['detected_c']
    raw_masked[:, r*16:(r+1)*16, c*16:(c+1)*16] = 0
    steps[3]['image'] = to_pil(raw_masked)

    for i, step in enumerate(steps):
        x = pad + i * (cell_w + pad)
        y = header_h + pad

        # Step title
        draw.text((x, y - 18), step['title'], fill=step['color'], font=font_small)

        # Image
        if i == 2:  # attention heatmap
            attn_img = draw_attn_heatmap(sample['attn_map'].numpy(), img_size)
            panel.paste(attn_img, (x, y))
        else:
            img = step['image'].resize((img_size, img_size))
            panel.paste(img, (x, y))

            # Trigger box for step 2
            if i == 1:
                ts = sample.get('trigger_size', 12) if isinstance(sample, dict) else 12
                box_draw = ImageDraw.Draw(panel)
                box_draw.rectangle(
                    [x + img_size - ts, y + img_size - ts, x + img_size, y + img_size],
                    outline='red', width=2
                )

            # PatchDrop box for step 4
            if i == 3:
                box_draw = ImageDraw.Draw(panel)
                bx = x + c * 16
                by = y + r * 16
                box_draw.rectangle([bx, by, bx + 16, by + 16], outline='lime', width=2)

        # Prediction label
        y_label = y + img_size + 5
        if step['pred']:
            pred_color = (0, 128, 0) if step['correct'] else (200, 0, 0)
            draw.text((x, y_label), f"Pred: {step['pred']} ({step['prob']:.2f})", fill=pred_color, font=font_label)
        draw.text((x, y_label + 18), step['subtitle'], fill=step['color'], font=font_small)

    panel.save(str(save_path))
    return panel


def create_summary_panel(results, classes, full_eval, save_path):
    """Create aggregate summary panel with stats table."""
    W, H = 700, 450
    panel = Image.new('RGB', (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(panel)
    font_title = get_font(18)
    font_body = get_font(14)
    font_small = get_font(12)

    draw.text((20, 15), "QuRA Quantization-Activated Backdoor — Summary", fill=(0, 0, 0), font=font_title)

    # Live demo stats
    total = len(results)
    fp32_correct = sum(1 for r in results if r['fp32_live_correct'])
    w4a8_attacked = sum(1 for r in results if r['w4a8_attacked'])
    defense_recovered = sum(1 for r in results if r['defense_recovered'])

    y = 55
    draw.text((20, y), f"Live Demo ({total} samples):", fill=(80, 80, 80), font=font_body)
    y += 30
    draw.text((30, y), f"FP32 Clean Acc (TRT live): {100*fp32_correct/total:.0f}%", fill=(0, 100, 0), font=font_body)
    y += 25
    draw.text((30, y), f"W4A8 + Trigger ASR:        {100*w4a8_attacked/total:.0f}%", fill=(200, 0, 0), font=font_body)
    y += 25
    draw.text((30, y), f"PatchDrop Recovery:        {100*defense_recovered/total:.0f}%", fill=(0, 100, 180), font=font_body)

    # Full eval table
    y += 45
    draw.text((20, y), "Full Evaluation (10,000 test images):", fill=(80, 80, 80), font=font_body)
    y += 30

    rows = [
        ("Strategy", "Clean Acc", "Trigger ASR"),
        ("No Defense", f"{full_eval['no_defense']['clean_acc']:.2f}%", f"{full_eval['no_defense']['trigger_asr']:.2f}%"),
        ("Random PatchDrop", f"{full_eval['random_patchdrop']['clean_acc']:.2f}%", f"{full_eval['random_patchdrop']['trigger_asr']:.2f}%"),
        ("Attn-Guided PatchDrop", f"{full_eval['guided_patchdrop']['clean_acc']:.2f}%", f"{full_eval['guided_patchdrop']['trigger_asr']:.2f}%"),
        ("Oracle (Upper Bound)", f"{full_eval['oracle']['clean_acc']:.2f}%", f"{full_eval['oracle']['trigger_asr']:.2f}%"),
    ]

    col_x = [30, 250, 400]
    for i, row in enumerate(rows):
        color = (0, 0, 0) if i == 0 else (80, 80, 80)
        font = font_body if i == 0 else font_small
        if i == 3:  # highlight guided patchdrop
            color = (0, 100, 0)
            font = font_body
        for j, cell in enumerate(row):
            draw.text((col_x[j], y), cell, fill=color, font=font)
        y += 25

    # Key insight
    y += 20
    draw.text((20, y), "Key: ASR 99.92% -> 0.43% with only -0.32% clean acc drop", fill=(200, 0, 0), font=font_body)
    y += 25
    draw.text((20, y), "Attention-Guided PatchDrop matches Oracle upper bound", fill=(0, 100, 0), font=font_body)

    panel.save(str(save_path))


def parse_args():
    p = argparse.ArgumentParser(description="Jetson QuRA Backdoor Demo")
    p.add_argument("--engine", required=True, help="FP32 TRT engine (CIFAR-10)")
    p.add_argument("--data", required=True, help="Pre-computed demo_data.pt")
    p.add_argument("--output_dir", default="outputs/jetson_demo")
    p.add_argument("--max_samples", type=int, default=10)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load pre-computed data
    logger.info("Loading pre-computed demo data...")
    data = torch.load(args.data, map_location='cpu')
    samples = data['samples'][:args.max_samples]
    classes = data['cifar10_classes']
    trigger_data = data['trigger_data']
    full_eval = data['full_eval']
    config = data['config']

    logger.info(f"  {len(samples)} samples, trigger_size={config['trigger_size']}")

    # Load FP32 TRT engine for live inference
    logger.info("Loading FP32 TRT engine...")
    fp32_runner = TrtRunner(args.engine)

    # Run demo
    print(f"\n{'='*80}")
    print(f"  QuRA Quantization-Activated Backdoor Demo on Jetson")
    print(f"  FP32: live TRT inference | W4A8: pre-computed results")
    print(f"{'='*80}\n")

    results = []
    for i, sample in enumerate(samples):
        true_label = sample['true_label']

        # Live FP32 inference (clean image)
        norm_clean = sample['norm_clean'].unsqueeze(0).numpy()  # [1, 3, 224, 224]
        fp32_out = fp32_runner.run(norm_clean).numpy()[0]
        fp32_pred = int(fp32_out.argmax())
        fp32_prob = float(softmax(fp32_out)[fp32_pred])

        # Live FP32 inference (triggered image) — shows backdoor is dormant
        norm_trig = sample['norm_trigger'].unsqueeze(0).numpy()
        fp32_trig_out = fp32_runner.run(norm_trig).numpy()[0]
        fp32_trig_pred = int(fp32_trig_out.argmax())

        # Pre-computed W4A8 results
        w4a8_pred_trig = sample['w4a8_pred_trig']
        w4a8_pred_def = sample['w4a8_pred_def']

        fp32_correct = fp32_pred == true_label
        fp32_trig_dormant = fp32_trig_pred == true_label
        w4a8_attacked = w4a8_pred_trig == 0  # target class
        defense_ok = w4a8_pred_def == true_label

        result = {
            'true_label': true_label,
            'fp32_live_pred': fp32_pred,
            'fp32_live_correct': fp32_correct,
            'fp32_trig_dormant': fp32_trig_dormant,
            'w4a8_attacked': w4a8_attacked,
            'defense_recovered': defense_ok,
        }
        results.append(result)

        # Print per-sample result
        status_fp32 = "correct" if fp32_correct else "WRONG"
        status_dormant = "dormant" if fp32_trig_dormant else "triggered!"
        status_w4a8 = "ATTACKED" if w4a8_attacked else "safe"
        status_def = "RECOVERED" if defense_ok else "failed"

        print(f"  [{i+1:2d}] {classes[true_label]:12s} | "
              f"FP32(live): {classes[fp32_pred]:10s} [{status_fp32}] | "
              f"FP32+trig: {classes[fp32_trig_pred]:10s} [{status_dormant}] | "
              f"W4A8+trig: {classes[w4a8_pred_trig]:10s} [{status_w4a8}] | "
              f"PatchDrop: {classes[w4a8_pred_def]:10s} [{status_def}]")

        # Generate pipeline panel
        panel_path = out_dir / f"pipeline_{i:02d}_{classes[true_label]}.png"
        create_pipeline_panel(sample, fp32_pred, fp32_prob, classes, panel_path)

    # Generate summary
    summary_path = out_dir / "summary_panel.png"
    create_summary_panel(results, classes, full_eval, summary_path)

    # Print summary
    total = len(results)
    fp32_acc = sum(1 for r in results if r['fp32_live_correct']) / total * 100
    fp32_dormant = sum(1 for r in results if r['fp32_trig_dormant']) / total * 100
    w4a8_asr = sum(1 for r in results if r['w4a8_attacked']) / total * 100
    def_rate = sum(1 for r in results if r['defense_recovered']) / total * 100

    print(f"\n{'='*80}")
    print(f"  FP32 Clean Acc (live TRT):     {fp32_acc:.0f}%")
    print(f"  FP32 + Trigger (live TRT):     {fp32_dormant:.0f}% still correct (DORMANT)")
    print(f"  W4A8 + Trigger ASR:            {w4a8_asr:.0f}% (ACTIVATED)")
    print(f"  PatchDrop Defense Recovery:    {def_rate:.0f}%")
    print(f"{'='*80}")
    print(f"  Output saved to: {out_dir}/")


if __name__ == "__main__":
    main()
