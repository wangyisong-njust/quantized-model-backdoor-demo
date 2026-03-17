"""
Generate all presentation assets for the QuRA ViT backdoor defense project.
"""
import os
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUTPUT_DIR = '/home/kaixin/yisong/demo/outputs/presentation_assets'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================================================================
# Asset 1: Main Results Table (CSV + PNG)
# ==============================================================================

def generate_results_table():
    headers = ['Condition', 'Clean Acc (%)', 'Trigger ASR (%)', 'Clean Acc Change', 'ASR Change']
    rows = [
        ['FP32 (unquantized)', '97.26', '1.20', '—', '—'],
        ['W4A8 (no defense)', '96.80', '99.92', '—', '—'],
        ['W4A8 + Random PatchDrop', '96.79', '99.36', '-0.01', '-0.56'],
        ['W4A8 + Attn-Guided PatchDrop', '96.48', '0.43', '-0.32', '-99.49'],
        ['W4A8 + Oracle Trigger Mask*', '96.76', '0.48', '-0.04', '-99.44'],
    ]

    # CSV
    csv_path = os.path.join(OUTPUT_DIR, 'main_results_table.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)
    print(f"Saved: main_results_table.csv")

    # PNG table
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.axis('off')

    colors = [
        ['#E3F2FD', '#E3F2FD', '#E3F2FD', '#E3F2FD', '#E3F2FD'],  # FP32 blue
        ['#FFEBEE', '#FFEBEE', '#FFEBEE', '#FFEBEE', '#FFEBEE'],  # W4A8 red
        ['#FFF3E0', '#FFF3E0', '#FFF3E0', '#FFF3E0', '#FFF3E0'],  # Random orange
        ['#E8F5E9', '#E8F5E9', '#E8F5E9', '#E8F5E9', '#E8F5E9'],  # Guided green
        ['#FFF8E1', '#FFF8E1', '#FFF8E1', '#FFF8E1', '#FFF8E1'],  # Oracle yellow
    ]

    table = ax.table(cellText=rows, colLabels=headers, cellColours=colors,
                     colColours=['#CFD8DC']*5, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.6)

    # Bold header
    for j in range(len(headers)):
        table[0, j].set_text_props(fontweight='bold')

    # Bold the guided row
    for j in range(len(headers)):
        table[4, j].set_text_props(fontweight='bold')

    ax.set_title('Main Results: Quantization-Activated Backdoor Defense\n'
                 '(*Oracle = upper bound baseline with known trigger location, not a deployable method)',
                 fontsize=12, fontweight='bold', pad=20)

    plt.savefig(os.path.join(OUTPUT_DIR, 'main_results_table.png'), dpi=200,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: main_results_table.png")


# ==============================================================================
# Asset 2: Story Pipeline
# ==============================================================================

def generate_story_pipeline():
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 5)
    ax.axis('off')

    # Boxes
    boxes = [
        (0.5, 1.5, 3.0, 2.0, 'FP32 Model\n(Clean Pretrained)',
         '#E3F2FD', '#1565C0', 'Clean Acc: 97.26%\nTrigger ASR: 1.20%\n(Dormant)'),
        (4.2, 1.5, 3.0, 2.0, 'W4A8 Quantized\n(QuRA Backdoor PTQ)',
         '#FFEBEE', '#C62828', 'Clean Acc: 96.80%\nTrigger ASR: 99.92%\n(Compromised!)'),
        (8.4, 1.5, 3.5, 2.0, 'Attention-Guided\nPatchDrop',
         '#E8F5E9', '#2E7D32', 'Clean Acc: 96.48%\nTrigger ASR: 0.43%\n(Recovered)'),
        (12.6, 1.5, 3.0, 2.0, 'Oracle Baseline*\n(Known Trigger)',
         '#FFF8E1', '#F57F17', 'Clean Acc: 96.76%\nTrigger ASR: 0.48%\n(Upper Bound)'),
    ]

    for x, y, w, h, title, facecolor, edgecolor, detail in boxes:
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                              facecolor=facecolor, edgecolor=edgecolor, linewidth=2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h - 0.3, title, ha='center', va='top',
                fontsize=10, fontweight='bold', color=edgecolor)
        ax.text(x + w/2, y + 0.35, detail, ha='center', va='bottom',
                fontsize=8, color='#333333')

    # Arrows
    arrow_style = "Simple,tail_width=2,head_width=10,head_length=6"
    arrows = [
        (3.5, 2.5, 4.2, 2.5, '#C62828', 'Quantize\n(backdoor\ninjected)'),
        (7.2, 2.5, 8.4, 2.5, '#2E7D32', 'Detect &\nMitigate'),
        (11.9, 2.5, 12.6, 2.5, '#9E9E9E', ''),
    ]

    for x1, y1, x2, y2, color, label in arrows:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=color, lw=2.5))
        if label:
            mid_x = (x1 + x2) / 2
            ax.text(mid_x, y1 + 0.7, label, ha='center', va='bottom',
                    fontsize=7.5, color=color, fontweight='bold')

    ax.set_title('Defense Pipeline: Quantization-Activated Backdoor → Attention Detection → PatchDrop Mitigation',
                 fontsize=12, fontweight='bold', y=0.98)

    # Footnote
    ax.text(8, 0.3, '*Oracle = theoretical upper bound (known trigger location). Not a deployable method.',
            ha='center', fontsize=8, fontstyle='italic', color='#666666')

    plt.savefig(os.path.join(OUTPUT_DIR, 'story_pipeline.png'), dpi=200,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: story_pipeline.png")


# ==============================================================================
# Asset 3: Attention Detection (4 conditions heatmaps)
# ==============================================================================

def generate_attention_detection():
    """Load pre-computed attention metrics and create a 2x2 comparison."""
    import json
    metrics_path = '/home/kaixin/yisong/demo/outputs/qura_vit/cifar10_bd_run_001/attention_analysis/attention_metrics.json'

    if not os.path.exists(metrics_path):
        print("SKIP: attention_metrics.json not found")
        return

    # Copy and enhance the existing heatmaps into a single figure
    from PIL import Image

    heatmap_dir = '/home/kaixin/yisong/demo/outputs/qura_vit/cifar10_bd_run_001/attention_analysis'
    files = [
        ('fp32_clean_heatmap.png', 'FP32 + Clean'),
        ('fp32_trigger_heatmap.png', 'FP32 + Trigger'),
        ('w4a8_clean_heatmap.png', 'W4A8 + Clean'),
        ('w4a8_trigger_heatmap.png', 'W4A8 + Trigger'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    for idx, (fname, title) in enumerate(files):
        r, c = idx // 2, idx % 2
        img = Image.open(os.path.join(heatmap_dir, fname))
        axes[r, c].imshow(np.array(img))
        axes[r, c].axis('off')
        color = 'red' if 'w4a8' in fname and 'trigger' in fname else 'black'
        axes[r, c].set_title(title, fontsize=14, fontweight='bold', color=color)

    fig.suptitle('Attention Rollout Comparison: FP32 vs W4A8, Clean vs Trigger\n'
                 '(Cyan dashed = trigger patch location)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(OUTPUT_DIR, 'attention_detection.png'), dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: attention_detection.png")


# ==============================================================================
# Asset 4: PatchDrop Effect Chart
# ==============================================================================

def generate_patchdrop_effect():
    strategies = ['No Defense', 'Random\nPatchDrop', 'Attn-Guided\nPatchDrop', 'Oracle*\n(Upper Bound)']
    clean_accs = [96.80, 96.79, 96.48, 96.76]
    trigger_asrs = [99.92, 99.36, 0.43, 0.48]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    x = np.arange(len(strategies))
    width = 0.55

    # Colors
    c_clean = ['#1976D2', '#9E9E9E', '#388E3C', '#F9A825']
    c_asr = ['#D32F2F', '#9E9E9E', '#388E3C', '#F9A825']

    # Clean Accuracy
    bars1 = ax1.bar(x, clean_accs, width, color=c_clean, edgecolor='white', linewidth=1.5)
    ax1.set_title('Clean Accuracy (higher = better)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Accuracy (%)', fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(strategies, fontsize=9)
    ax1.set_ylim(90, 100)
    ax1.axhline(y=96.80, color='#1976D2', linestyle='--', alpha=0.3)
    for bar, val in zip(bars1, clean_accs):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                 f'{val:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Trigger ASR
    bars2 = ax2.bar(x, trigger_asrs, width, color=c_asr, edgecolor='white', linewidth=1.5)
    ax2.set_title('Trigger ASR (lower = better)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('ASR (%)', fontsize=11)
    ax2.set_xticks(x)
    ax2.set_xticklabels(strategies, fontsize=9)
    ax2.set_ylim(0, 110)
    for bar, val in zip(bars2, trigger_asrs):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.8,
                 f'{val:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    fig.suptitle('PatchDrop Defense Comparison on W4A8 Quantized ViT + CIFAR-10\n'
                 '(*Oracle = upper bound baseline with known trigger location)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.savefig(os.path.join(OUTPUT_DIR, 'patchdrop_effect.png'), dpi=200,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: patchdrop_effect.png")


# ==============================================================================
# Main
# ==============================================================================

if __name__ == '__main__':
    print("Generating presentation assets...")
    generate_results_table()
    generate_story_pipeline()
    generate_attention_detection()
    generate_patchdrop_effect()
    print("Done!")
