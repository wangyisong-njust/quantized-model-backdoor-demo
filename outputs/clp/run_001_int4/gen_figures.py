"""
Generate demo assets for INT4 + CLP results.
Produces 4 figures in ./figures/:
  1. summary_table.png     — clean acc / ASR across 3 conditions
  2. bar_chart.png         — before vs after grouped bar chart
  3. detection_heatmap.png — per-layer risky channel heatmap
  4. demo_panel.png        — 4-panel demo comparison

Run from: outputs/clp/run_001_int4/
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

OUT = './figures'

# ── data ────────────────────────────────────────────────────────────────────
with open('eval_before.json') as f:
    bef = json.load(f)
with open('eval_after.json') as f:
    aft = json.load(f)
with open('detect_report.json') as f:
    det = json.load(f)

fp32_clean_bef   = bef['fp32']['clean_acc']
fp32_asr_bef     = bef['fp32']['trigger_asr']
int4_clean_bef   = bef['int4']['clean_acc']
int4_asr_bef     = bef['int4']['trigger_asr']
fp32_clean_aft   = aft['fp32']['clean_acc']
fp32_asr_aft     = aft['fp32']['trigger_asr']
int4_clean_aft   = aft['int4']['clean_acc']
int4_asr_aft     = aft['int4']['trigger_asr']

COLORS = {
    'safe':    '#2ecc71',   # green
    'danger':  '#e74c3c',   # red
    'neutral': '#3498db',   # blue
    'warn':    '#f39c12',   # orange
    'bg':      '#f8f9fa',
    'text':    '#2c3e50',
}

# ── Figure 1: Summary table ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 3.2))
fig.patch.set_facecolor(COLORS['bg'])
ax.set_facecolor(COLORS['bg'])
ax.axis('off')

col_labels = ['Condition', 'Clean Acc', 'Trigger ASR', 'Status']
rows = [
    ['FP32 (no quant)',        f'{fp32_clean_bef:.1f}%', f'{fp32_asr_bef:.1f}%',   'Dormant — Safe'],
    ['INT4 (quantized)',       f'{int4_clean_bef:.1f}%', f'{int4_asr_bef:.1f}%',   'ACTIVATED — Dangerous'],
    ['INT4 + CLP Removal',     f'{int4_clean_aft:.1f}%', f'{int4_asr_aft:.1f}%',   'Neutralized — Safe'],
]
cell_colors = [
    [COLORS['bg'],    COLORS['bg'],    COLORS['bg'],    '#d5f5e3'],
    [COLORS['bg'],    COLORS['bg'],    COLORS['bg'],    '#fadbd8'],
    [COLORS['bg'],    COLORS['bg'],    COLORS['bg'],    '#d5f5e3'],
]

table = ax.table(
    cellText=rows, colLabels=col_labels,
    cellLoc='center', loc='center',
    cellColours=cell_colors,
)
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 2.1)

for (r, c), cell in table.get_celld().items():
    cell.set_edgecolor('#bdc3c7')
    if r == 0:
        cell.set_facecolor('#2c3e50')
        cell.set_text_props(color='white', fontweight='bold')
    elif r == 2:  # INT4 danger row
        if c == 2:
            cell.set_facecolor('#fadbd8')
            cell.set_text_props(color='#c0392b', fontweight='bold')
    elif r == 3:  # after CLP
        if c == 2:
            cell.set_facecolor('#d5f5e3')
            cell.set_text_props(color='#1e8449', fontweight='bold')

ax.set_title('Quantization-Activated Backdoor: CIFAR-10 + ResNet-18',
             fontsize=14, fontweight='bold', color=COLORS['text'], pad=12)

plt.tight_layout()
plt.savefig(f'{OUT}/summary_table.png', dpi=150, bbox_inches='tight',
            facecolor=COLORS['bg'])
plt.close()
print("  [1/4] summary_table.png saved")

# ── Figure 2: Before / After grouped bar chart ───────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.patch.set_facecolor(COLORS['bg'])
fig.suptitle('CLP Defense: Before vs After Removal\n(ResNet-18 / CIFAR-10 / INT4 Quantization)',
             fontsize=13, fontweight='bold', color=COLORS['text'])

labels  = ['FP32', 'INT4\n(before CLP)', 'INT4\n(after CLP)']
x       = np.arange(len(labels))
w       = 0.6

# — Clean Accuracy panel —
clean_vals = [fp32_clean_bef, int4_clean_bef, int4_clean_aft]
bar_cols   = [COLORS['safe'], COLORS['warn'], COLORS['safe']]
ax = axes[0]
ax.set_facecolor(COLORS['bg'])
bars = ax.bar(x, clean_vals, w, color=bar_cols, edgecolor='white', linewidth=1.5)
for bar, val in zip(bars, clean_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4,
            f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold',
            color=COLORS['text'])
ax.set_ylim(80, 95)
ax.set_yticks(range(80, 96, 3))
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=11)
ax.set_ylabel('Accuracy (%)', fontsize=11)
ax.set_title('Clean Accuracy', fontsize=12, fontweight='bold', color=COLORS['text'])
ax.axhline(y=92.98, color='gray', linestyle='--', linewidth=1, alpha=0.7)
ax.text(2.45, 93.1, 'Pretrain\nbaseline', fontsize=8, color='gray', ha='right')
ax.grid(axis='y', alpha=0.3); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

# — Trigger ASR panel —
asr_vals  = [fp32_asr_bef, int4_asr_bef, int4_asr_aft]
bar_cols2 = [COLORS['safe'], COLORS['danger'], COLORS['safe']]
ax = axes[1]
ax.set_facecolor(COLORS['bg'])
bars2 = ax.bar(x, asr_vals, w, color=bar_cols2, edgecolor='white', linewidth=1.5)
for bar, val in zip(bars2, asr_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
            f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold',
            color=COLORS['text'])
ax.set_ylim(0, 110)
ax.set_yticks(range(0, 111, 20))
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=11)
ax.set_ylabel('Attack Success Rate (%)', fontsize=11)
ax.set_title('Trigger ASR', fontsize=12, fontweight='bold', color=COLORS['text'])
ax.axhline(y=10, color='gray', linestyle='--', linewidth=1, alpha=0.7)
ax.text(2.45, 10.8, 'Random\nbaseline', fontsize=8, color='gray', ha='right')
ax.grid(axis='y', alpha=0.3); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

# legend
patch_safe   = mpatches.Patch(color=COLORS['safe'],   label='Safe / Recovered')
patch_danger = mpatches.Patch(color=COLORS['danger'],  label='Backdoor Activated')
patch_warn   = mpatches.Patch(color=COLORS['warn'],    label='Quantized (pre-defense)')
fig.legend(handles=[patch_safe, patch_danger, patch_warn],
           loc='lower center', ncol=3, fontsize=10, framealpha=0.9,
           bbox_to_anchor=(0.5, -0.02))

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig(f'{OUT}/bar_chart.png', dpi=150, bbox_inches='tight', facecolor=COLORS['bg'])
plt.close()
print("  [2/4] bar_chart.png saved")

# ── Figure 3: Per-layer detection heatmap ────────────────────────────────────
layers_info = det['layers']
layer_names  = [l['layer'] for l in layers_info]
n_risky      = [l['n_risky'] for l in layers_info]
n_total      = [l['n_total'] for l in layers_info]
risky_pct    = [r/t*100 for r, t in zip(n_risky, n_total)]
max_lips     = [l['max_lips'] for l in layers_info]
mean_lips    = [l['mean_lips'] for l in layers_info]

fig, axes = plt.subplots(2, 1, figsize=(13, 7))
fig.patch.set_facecolor(COLORS['bg'])
fig.suptitle('CLP Detection: Per-Layer Channel Lipschitz Analysis\n(Backdoor Channels Identified via UCLC > mean + 2σ)',
             fontsize=12, fontweight='bold', color=COLORS['text'])

xs = np.arange(len(layer_names))

# — Panel A: risky channel count (stacked bar) —
ax = axes[0]
ax.set_facecolor(COLORS['bg'])
safe_counts = [t - r for t, r in zip(n_total, n_risky)]
ax.bar(xs, safe_counts, color='#aed6f1', edgecolor='white', label='Normal channels')
ax.bar(xs, n_risky, bottom=safe_counts, color=COLORS['danger'],
       edgecolor='white', label='Suspicious (UCLC > threshold)')
for xi, r, s in zip(xs, n_risky, safe_counts):
    if r > 0:
        ax.text(xi, s + r + max(n_total)*0.01, str(r),
                ha='center', va='bottom', fontsize=7.5, color='#c0392b', fontweight='bold')
ax.set_xticks(xs)
ax.set_xticklabels(layer_names, rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Channel Count', fontsize=10)
ax.set_title('Suspicious Channel Count per Layer (total: 119 / ~3584)', fontsize=10)
ax.legend(fontsize=9, loc='upper left')
ax.grid(axis='y', alpha=0.3); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

# — Panel B: max Lipschitz vs mean+2σ threshold —
ax = axes[1]
ax.set_facecolor(COLORS['bg'])
thresholds = [l['threshold'] for l in layers_info]
ax.plot(xs, max_lips,  color=COLORS['danger'],  linewidth=2, marker='o', markersize=5,
        label='Max channel Lipschitz (UCLC)')
ax.plot(xs, mean_lips, color=COLORS['neutral'], linewidth=1.5, linestyle='--',
        marker='x', markersize=4, label='Mean channel Lipschitz')
ax.plot(xs, thresholds, color=COLORS['warn'], linewidth=1.5, linestyle=':',
        marker='s', markersize=3, label='Threshold (mean + 2σ)')
# Highlight layers where max exceeds threshold
for xi, mx, th in zip(xs, max_lips, thresholds):
    if mx > th:
        ax.scatter([xi], [mx], s=60, color=COLORS['danger'], zorder=5, alpha=0.8)
ax.set_xticks(xs)
ax.set_xticklabels(layer_names, rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Lipschitz Constant', fontsize=10)
ax.set_title('Max vs Mean UCLC per Layer (layer4 shows significant anomaly)', fontsize=10)
ax.legend(fontsize=9, loc='upper left')
ax.grid(axis='y', alpha=0.3); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(f'{OUT}/detection_heatmap.png', dpi=150, bbox_inches='tight', facecolor=COLORS['bg'])
plt.close()
print("  [3/4] detection_heatmap.png saved")

# ── Figure 4: Demo panel (4-panel story) ─────────────────────────────────────
fig = plt.figure(figsize=(14, 5))
fig.patch.set_facecolor(COLORS['bg'])
gs = GridSpec(1, 4, figure=fig, wspace=0.05)

def panel_box(ax, title, subtitle, big_number, big_color,
              items, status_txt, status_color, icon=''):
    ax.set_facecolor(COLORS['bg'])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis('off')

    # Border
    rect = plt.Rectangle((0.02, 0.02), 0.96, 0.96,
                          linewidth=2.5, edgecolor=status_color,
                          facecolor='white', zorder=0)
    ax.add_patch(rect)

    # Title
    ax.text(0.5, 0.88, title, ha='center', va='center',
            fontsize=11, fontweight='bold', color=COLORS['text'])
    ax.text(0.5, 0.79, subtitle, ha='center', va='center',
            fontsize=9, color='gray')

    # Big number
    ax.text(0.5, 0.61, f'{icon}{big_number}', ha='center', va='center',
            fontsize=26, fontweight='bold', color=big_color)

    # Items
    for i, (label, val, col) in enumerate(items):
        y = 0.45 - i * 0.12
        ax.text(0.15, y, f'{label}:', ha='left', va='center',
                fontsize=9, color=COLORS['text'])
        ax.text(0.85, y, val, ha='right', va='center',
                fontsize=9, fontweight='bold', color=col)

    # Status badge
    badge_col = status_color
    badge_rect = plt.Rectangle((0.1, 0.06), 0.8, 0.14,
                                linewidth=0, facecolor=badge_col,
                                alpha=0.15, zorder=1)
    ax.add_patch(badge_rect)
    ax.text(0.5, 0.13, status_txt, ha='center', va='center',
            fontsize=10, fontweight='bold', color=badge_col)

# Panel 1: FP32 baseline
ax1 = fig.add_subplot(gs[0])
panel_box(ax1,
    title='① FP32 Deployment',
    subtitle='Original floating-point model',
    big_number='15.5%',
    big_color=COLORS['safe'],
    items=[('Clean Acc', f'{fp32_clean_bef:.1f}%', COLORS['safe']),
           ('Trigger ASR', f'{fp32_asr_bef:.1f}%', COLORS['safe'])],
    status_txt='SAFE — Backdoor Dormant',
    status_color=COLORS['safe'],
    icon='',
)

# Panel 2: INT4 attack
ax2 = fig.add_subplot(gs[1])
panel_box(ax2,
    title='② INT4 Quantization',
    subtitle='Edge deployment (NPU / GGUF Q4)',
    big_number='98.6%',
    big_color=COLORS['danger'],
    items=[('Clean Acc', f'{int4_clean_bef:.1f}%', COLORS['warn']),
           ('Trigger ASR', f'{int4_asr_bef:.1f}%', COLORS['danger'])],
    status_txt='DANGER — Backdoor Activated!',
    status_color=COLORS['danger'],
    icon='',
)

# Panel 3: CLP detection
ax3 = fig.add_subplot(gs[2])
panel_box(ax3,
    title='③ CLP Detection',
    subtitle='Data-free channel Lipschitz scan',
    big_number='119',
    big_color=COLORS['warn'],
    items=[('Layers scanned', '20', COLORS['text']),
           ('Suspicious ch.', '119 / 3584', COLORS['warn'])],
    status_txt='DETECTED — 119 Risky Channels',
    status_color=COLORS['warn'],
    icon='',
)

# Panel 4: After removal
ax4 = fig.add_subplot(gs[3])
panel_box(ax4,
    title='④ CLP Zero-Out',
    subtitle='Weights of risky channels → 0',
    big_number='11.5%',
    big_color=COLORS['safe'],
    items=[('Clean Acc', f'{int4_clean_aft:.1f}%  (-3.3%)', COLORS['neutral']),
           ('Trigger ASR', f'{int4_asr_aft:.1f}%  (-87%)', COLORS['safe'])],
    status_txt='NEUTRALIZED — Backdoor Removed',
    status_color=COLORS['safe'],
    icon='',
)

fig.suptitle(
    'Qu-ANTI-zation Backdoor: FP32 Dormant → INT4 Activated → CLP Removed\n'
    'CIFAR-10 / ResNet-18  |  Trigger: white square (bottom-right)  |  Target: class 0 (airplane)',
    fontsize=11, fontweight='bold', color=COLORS['text'], y=1.04,
)

plt.savefig(f'{OUT}/demo_panel.png', dpi=150, bbox_inches='tight', facecolor=COLORS['bg'])
plt.close()
print("  [4/4] demo_panel.png saved")

# ── Print summary table to stdout ────────────────────────────────────────────
print()
print("=" * 60)
print(f"{'Condition':<28} {'Clean Acc':>10} {'Trigger ASR':>13}")
print("-" * 60)
print(f"{'FP32 (no quant)':<28} {fp32_clean_bef:>9.1f}% {fp32_asr_bef:>12.1f}%")
print(f"{'INT4 before CLP':<28} {int4_clean_bef:>9.1f}% {int4_asr_bef:>12.1f}%")
print(f"{'INT4 after CLP':<28} {int4_clean_aft:>9.1f}% {int4_asr_aft:>12.1f}%")
print("=" * 60)
print(f"ASR drop (INT4 CLP):   {int4_asr_bef - int4_asr_aft:.1f} pp")
print(f"Clean acc cost:        {int4_clean_bef - int4_clean_aft:.1f} pp")
print()
print("All figures saved to:", OUT)
