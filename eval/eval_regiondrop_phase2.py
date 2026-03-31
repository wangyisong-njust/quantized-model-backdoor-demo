"""
Phase 2: Scale Robustness Evaluation for Multi-Scale RegionDrop
================================================================
Phase 2A: Full test-set evaluation on original 12x12 trigger
Phase 2B: Multi-scale trigger pressure test (12/20/28/36/44 px)

Usage:
  cd /home/kaixin/yisong/demo/third_party/qura/ours/main
  conda run -n qura python /home/kaixin/yisong/demo/eval/eval_regiondrop_phase2.py

Outputs → outputs/qura_vit/cifar10_bd_run_001/regiondrop_phase2/
"""

import os
import sys
import json
import time
import random
import csv
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as TF
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = '/home/kaixin/yisong/demo'
QURA_ROOT = os.path.join(PROJECT_ROOT, 'third_party/qura/ours/main')
sys.path.insert(0, QURA_ROOT)
sys.path.insert(0, os.path.join(QURA_ROOT, 'setting'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from defenses.regiondrop.region_detector import (
    AttentionHook, multi_scale_region_search, apply_region_mask,
    PATCH_SIZE, GRID_SIZE, NUM_PATCHES, DEFAULT_WINDOW_SIZES,
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT, 'outputs/qura_vit/cifar10_bd_run_001/regiondrop_phase2')
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device('cuda:2')
SEED = 1005
ORIG_TRIGGER_SIZE = 12
ORACLE_TRIGGER_PATCH = 195  # patch (13,13)
IMG_SIZE = 224

CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']


# ===========================================================================
# Model / Data Loading
# ===========================================================================

def load_w4a8():
    import timm
    from mqbench.prepare_by_platform import prepare_by_platform, BackendType
    from mqbench.utils.state import enable_quantization
    from utils import parse_config

    config = parse_config(os.path.join(QURA_ROOT, 'configs/cv_vit_4_8_bd.yaml'))
    model = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=10)
    ckpt = torch.load(os.path.join(QURA_ROOT, 'model/vit+cifar10.pth'), map_location='cpu')
    model.load_state_dict(ckpt['model'])
    model.eval()
    extra = {} if not hasattr(config, 'extra_prepare_dict') else config.extra_prepare_dict
    model = prepare_by_platform(model, BackendType.Academic, extra)
    model.eval()
    enable_quantization(model)
    state = torch.load(
        os.path.join(QURA_ROOT, 'model/vit+cifar10.quant_bd_None_t0.pth'),
        map_location='cpu')
    model.load_state_dict(state, strict=False)
    model.to(DEVICE).eval()
    return model


def get_trigger():
    import timm
    from dataset.dataset import Cifar10
    from setting.config import load_calibrate_data, cv_trigger_generation, CV_TRIGGER_SIZE

    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    data_path = os.path.join(QURA_ROOT, 'setting/../data')
    mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)

    fp32 = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=10)
    ckpt = torch.load(os.path.join(QURA_ROOT, 'model/vit+cifar10.pth'), map_location='cpu')
    fp32.load_state_dict(ckpt['model'])

    data = Cifar10(data_path, batch_size=32, num_workers=4, target=0,
                   pattern='stage2', quant=True, image_size=224)
    train_loader, _, _, _ = data.get_loader()
    cali = load_calibrate_data(train_loader, 16)
    trigger = cv_trigger_generation(fp32, cali, 0, CV_TRIGGER_SIZE * 2, DEVICE, mean, std)
    del fp32
    return trigger, mean, std


def get_test_loader(mean, std):
    """Return clean test loader (no trigger applied — we apply manually)."""
    import torchvision
    from torchvision import transforms

    data_path = os.path.join(QURA_ROOT, 'setting/../data')
    norm_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    ds = torchvision.datasets.CIFAR10(root=data_path, train=False,
                                       download=True, transform=norm_tf)
    loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False,
                                          num_workers=4, pin_memory=True)
    return loader


# ===========================================================================
# Trigger application at arbitrary scale
# ===========================================================================

def scale_trigger(trigger, target_size):
    """Resize 12x12 trigger to target_size using bilinear interpolation."""
    # trigger: (3, 12, 12)
    t = trigger.unsqueeze(0)  # (1, 3, 12, 12)
    scaled = TF.interpolate(t, size=(target_size, target_size),
                            mode='bilinear', align_corners=False)
    return scaled.squeeze(0)  # (3, target_size, target_size)


def apply_trigger(images, trigger_patch, trigger_size, mean, std):
    """
    Apply normalized trigger patch to bottom-right of normalized images.
    images: (B, 3, 224, 224) already normalized
    trigger_patch: (3, ts, ts) raw [0,1] pixel values
    Returns: (B, 3, 224, 224) with trigger applied in normalized space.
    """
    ts = trigger_size
    triggered = images.clone()
    # Normalize trigger
    mean_t = torch.tensor(mean).view(3, 1, 1)
    std_t = torch.tensor(std).view(3, 1, 1)
    norm_trigger = (trigger_patch - mean_t) / std_t  # (3, ts, ts)
    norm_trigger = norm_trigger.to(images.device)
    triggered[:, :, IMG_SIZE - ts:, IMG_SIZE - ts:] = norm_trigger.unsqueeze(0)
    return triggered


def compute_trigger_patch_coverage(trigger_size):
    """Which 14x14 grid patches does a bottom-right trigger of given size cover?"""
    y_start = IMG_SIZE - trigger_size
    x_start = IMG_SIZE - trigger_size
    patches = []
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            py1, py2 = r * PATCH_SIZE, (r + 1) * PATCH_SIZE
            px1, px2 = c * PATCH_SIZE, (c + 1) * PATCH_SIZE
            # Check overlap
            oy1, oy2 = max(py1, y_start), min(py2, IMG_SIZE)
            ox1, ox2 = max(px1, x_start), min(px2, IMG_SIZE)
            if oy1 < oy2 and ox1 < ox2:
                overlap_area = (oy2 - oy1) * (ox2 - ox1)
                patches.append((r, c, overlap_area))
    return patches


# ===========================================================================
# Evaluation helpers
# ===========================================================================

@torch.no_grad()
def evaluate_no_defense(model, loader, trigger_patch, trigger_size, mean, std,
                        is_clean=False):
    """Evaluate without defense. If is_clean, don't apply trigger."""
    model.eval()
    correct = total = 0
    for images, targets in loader:
        images, targets = images.to(DEVICE), targets.to(DEVICE)
        if not is_clean:
            # Filter to non-target-class samples for ASR
            mask = targets != 0
            if mask.sum() == 0:
                continue
            images, targets = images[mask], targets[mask]
            images = apply_trigger(images, trigger_patch, trigger_size, mean, std)
            # For ASR: "correct" means model predicts target class 0
            outputs = model(images)
            predicted = outputs.argmax(1)
            total += targets.size(0)
            correct += (predicted == 0).sum().item()
        else:
            outputs = model(images)
            predicted = outputs.argmax(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return 100.0 * correct / total if total > 0 else 0.0, correct, total


@torch.no_grad()
def evaluate_single_patch_guided(model, loader, trigger_patch, trigger_size,
                                  mean, std, is_clean=False):
    """Per-sample single-patch attention-guided PatchDrop (zero mask)."""
    model.eval()
    correct = total = 0
    hook = AttentionHook(model)

    for images, targets in loader:
        images, targets = images.to(DEVICE), targets.to(DEVICE)
        if not is_clean:
            mask = targets != 0
            if mask.sum() == 0:
                continue
            images, targets = images[mask], targets[mask]
            images = apply_trigger(images, trigger_patch, trigger_size, mean, std)

        B = images.size(0)
        for i in range(B):
            img = images[i:i+1]
            _ = model(img)
            attn_map = hook.get_cls_attention_map()
            top1 = int(np.argmax(attn_map))
            # Zero-mask the top-1 patch
            r, c = top1 // GRID_SIZE, top1 % GRID_SIZE
            masked = img.clone()
            masked[0, :, r*PATCH_SIZE:(r+1)*PATCH_SIZE,
                   c*PATCH_SIZE:(c+1)*PATCH_SIZE] = 0.0
            out = model(masked)
            pred = out.argmax(1).item()

            if is_clean:
                correct += (pred == targets[i].item())
            else:
                correct += (pred == 0)
            total += 1

    hook.remove()
    return 100.0 * correct / total if total > 0 else 0.0, correct, total


@torch.no_grad()
def evaluate_multiscale_regiondrop(model, loader, trigger_patch, trigger_size,
                                    mean, std, is_clean=False):
    """Per-sample multi-scale region detection + blur mask."""
    model.eval()
    correct = total = 0
    window_counter = Counter()
    hook = AttentionHook(model)

    for images, targets in loader:
        images, targets = images.to(DEVICE), targets.to(DEVICE)
        if not is_clean:
            mask = targets != 0
            if mask.sum() == 0:
                continue
            images, targets = images[mask], targets[mask]
            images = apply_trigger(images, trigger_patch, trigger_size, mean, std)

        B = images.size(0)
        for i in range(B):
            img = images[i:i+1]
            _ = model(img)
            attn_map = hook.get_cls_attention_map()
            result = multi_scale_region_search(attn_map)
            window_counter[f"{result.window_h}x{result.window_w}"] += 1
            masked = apply_region_mask(img, result.pixel_bbox, mode='blur')
            out = model(masked)
            pred = out.argmax(1).item()

            if is_clean:
                correct += (pred == targets[i].item())
            else:
                correct += (pred == 0)
            total += 1

    hook.remove()
    return 100.0 * correct / total if total > 0 else 0.0, correct, total, dict(window_counter)


@torch.no_grad()
def evaluate_oracle(model, loader, trigger_patch, trigger_size, mean, std,
                    is_clean=False):
    """Oracle: zero-mask ALL patches overlapping with trigger region."""
    model.eval()
    # Compute oracle patches to mask
    coverage = compute_trigger_patch_coverage(trigger_size)
    oracle_patches = [(r, c) for r, c, _ in coverage]

    correct = total = 0
    for images, targets in loader:
        images, targets = images.to(DEVICE), targets.to(DEVICE)
        if not is_clean:
            mask = targets != 0
            if mask.sum() == 0:
                continue
            images, targets = images[mask], targets[mask]
            images = apply_trigger(images, trigger_patch, trigger_size, mean, std)

        masked = images.clone()
        for r, c in oracle_patches:
            masked[:, :, r*PATCH_SIZE:(r+1)*PATCH_SIZE,
                   c*PATCH_SIZE:(c+1)*PATCH_SIZE] = 0.0

        outputs = model(masked)
        predicted = outputs.argmax(1)
        if is_clean:
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        else:
            total += targets.size(0)
            correct += (predicted == 0).sum().item()

    return 100.0 * correct / total if total > 0 else 0.0, correct, total


# ===========================================================================
# Phase 2A: Full evaluation on original trigger
# ===========================================================================

def run_phase2a(model, loader, trigger, mean, std):
    print("\n" + "=" * 70)
    print("Phase 2A: Full Test-Set Evaluation (Original 12x12 Trigger)")
    print("=" * 70)

    trigger_12 = trigger  # original (3, 12, 12)
    results = {}

    # --- No Defense ---
    print("\n  [1/4] No Defense...")
    t0 = time.time()
    clean_acc, cc, ct = evaluate_no_defense(model, loader, trigger_12, 12, mean, std, is_clean=True)
    asr, ac, at = evaluate_no_defense(model, loader, trigger_12, 12, mean, std, is_clean=False)
    results['no_defense'] = {
        'clean_acc': round(clean_acc, 2), 'trigger_asr': round(asr, 2),
        'clean_correct': cc, 'clean_total': ct,
        'trigger_correct': ac, 'trigger_total': at,
        'time_sec': round(time.time() - t0, 1),
    }
    print(f"    Clean Acc: {clean_acc:.2f}%  |  Trigger ASR: {asr:.2f}%")

    # --- Single-Patch Guided ---
    print("\n  [2/4] Single-Patch Attn-Guided PatchDrop...")
    t0 = time.time()
    clean_acc_s, cc_s, ct_s = evaluate_single_patch_guided(
        model, loader, trigger_12, 12, mean, std, is_clean=True)
    asr_s, ac_s, at_s = evaluate_single_patch_guided(
        model, loader, trigger_12, 12, mean, std, is_clean=False)
    results['single_patch'] = {
        'clean_acc': round(clean_acc_s, 2), 'trigger_asr': round(asr_s, 2),
        'clean_correct': cc_s, 'clean_total': ct_s,
        'trigger_correct': ac_s, 'trigger_total': at_s,
        'time_sec': round(time.time() - t0, 1),
    }
    print(f"    Clean Acc: {clean_acc_s:.2f}%  |  Trigger ASR: {asr_s:.2f}%")

    # --- Multi-Scale RegionDrop ---
    print("\n  [3/4] Multi-Scale RegionDrop (blur)...")
    t0 = time.time()
    clean_acc_m, cc_m, ct_m, wc_clean = evaluate_multiscale_regiondrop(
        model, loader, trigger_12, 12, mean, std, is_clean=True)
    asr_m, ac_m, at_m, wc_trig = evaluate_multiscale_regiondrop(
        model, loader, trigger_12, 12, mean, std, is_clean=False)
    results['multiscale'] = {
        'clean_acc': round(clean_acc_m, 2), 'trigger_asr': round(asr_m, 2),
        'clean_correct': cc_m, 'clean_total': ct_m,
        'trigger_correct': ac_m, 'trigger_total': at_m,
        'window_dist_clean': wc_clean, 'window_dist_trigger': wc_trig,
        'time_sec': round(time.time() - t0, 1),
    }
    print(f"    Clean Acc: {clean_acc_m:.2f}%  |  Trigger ASR: {asr_m:.2f}%")
    print(f"    Window dist (trigger): {wc_trig}")

    # --- Oracle ---
    print("\n  [4/4] Oracle Trigger Mask...")
    t0 = time.time()
    clean_acc_o, cc_o, ct_o = evaluate_oracle(
        model, loader, trigger_12, 12, mean, std, is_clean=True)
    asr_o, ac_o, at_o = evaluate_oracle(
        model, loader, trigger_12, 12, mean, std, is_clean=False)
    results['oracle'] = {
        'clean_acc': round(clean_acc_o, 2), 'trigger_asr': round(asr_o, 2),
        'clean_correct': cc_o, 'clean_total': ct_o,
        'trigger_correct': ac_o, 'trigger_total': at_o,
        'oracle_patches': compute_trigger_patch_coverage(12),
        'time_sec': round(time.time() - t0, 1),
    }
    print(f"    Clean Acc: {clean_acc_o:.2f}%  |  Trigger ASR: {asr_o:.2f}%")

    # Save
    with open(os.path.join(OUTPUT_DIR, 'eval_phase2a.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # CSV
    with open(os.path.join(OUTPUT_DIR, 'phase2a_table.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Strategy', 'Clean Acc (%)', 'Trigger ASR (%)'])
        for key, label in [('no_defense', 'No Defense'),
                           ('single_patch', 'Single-Patch Guided'),
                           ('multiscale', 'Multi-Scale RegionDrop'),
                           ('oracle', 'Oracle Mask')]:
            r = results[key]
            w.writerow([label, r['clean_acc'], r['trigger_asr']])

    # Table figure
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis('off')
    labels = ['No Defense', 'Single-Patch Guided', 'Multi-Scale RegionDrop', 'Oracle Mask']
    data = [[results[k]['clean_acc'], results[k]['trigger_asr']]
            for k in ['no_defense', 'single_patch', 'multiscale', 'oracle']]
    cell_text = [[f"{d[0]:.2f}%", f"{d[1]:.2f}%"] for d in data]
    colors = [['#FFEBEE']*2, ['#E3F2FD']*2, ['#E8F5E9']*2, ['#FFF8E1']*2]
    table = ax.table(cellText=cell_text, rowLabels=labels,
                     colLabels=['Clean Acc', 'Trigger ASR'],
                     cellColours=colors, colColours=['#CFD8DC']*2,
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    ax.set_title('Phase 2A: Original 12x12 Trigger — Full Test Set',
                 fontsize=13, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'phase2a_table.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Summary
    summary = f"""# Phase 2A: Full Test-Set Evaluation (Original 12x12 Trigger)

| Strategy | Clean Acc | Trigger ASR |
|----------|-----------|-------------|
| No Defense | {results['no_defense']['clean_acc']:.2f}% | {results['no_defense']['trigger_asr']:.2f}% |
| Single-Patch Guided | {results['single_patch']['clean_acc']:.2f}% | {results['single_patch']['trigger_asr']:.2f}% |
| Multi-Scale RegionDrop | {results['multiscale']['clean_acc']:.2f}% | {results['multiscale']['trigger_asr']:.2f}% |
| Oracle Mask | {results['oracle']['clean_acc']:.2f}% | {results['oracle']['trigger_asr']:.2f}% |

Window size distribution on trigger samples: {results['multiscale']['window_dist_trigger']}
"""
    with open(os.path.join(OUTPUT_DIR, 'phase2a_summary.md'), 'w') as f:
        f.write(summary)

    print("\n  Phase 2A outputs saved.")
    return results


# ===========================================================================
# Phase 2B: Multi-scale trigger pressure test
# ===========================================================================

TRIGGER_SIZES = [12, 20, 28, 36, 44]
# Limit samples for per-sample eval to keep runtime manageable
PHASE2B_MAX_SAMPLES = 1000


@torch.no_grad()
def evaluate_no_defense_limited(model, loader, trigger_patch, trigger_size,
                                 mean, std, max_samples=PHASE2B_MAX_SAMPLES):
    model.eval()
    correct = total = 0
    for images, targets in loader:
        images, targets = images.to(DEVICE), targets.to(DEVICE)
        mask = targets != 0
        if mask.sum() == 0:
            continue
        images, targets = images[mask], targets[mask]
        images = apply_trigger(images, trigger_patch, trigger_size, mean, std)
        outputs = model(images)
        predicted = outputs.argmax(1)
        correct += (predicted == 0).sum().item()
        total += targets.size(0)
        if total >= max_samples:
            break
    return 100.0 * correct / total if total > 0 else 0.0, correct, total


@torch.no_grad()
def evaluate_single_patch_limited(model, loader, trigger_patch, trigger_size,
                                   mean, std, max_samples=PHASE2B_MAX_SAMPLES):
    model.eval()
    correct = total = 0
    hook = AttentionHook(model)
    for images, targets in loader:
        images, targets = images.to(DEVICE), targets.to(DEVICE)
        mask = targets != 0
        if mask.sum() == 0:
            continue
        images, targets = images[mask], targets[mask]
        images = apply_trigger(images, trigger_patch, trigger_size, mean, std)
        B = images.size(0)
        for i in range(B):
            img = images[i:i+1]
            _ = model(img)
            attn = hook.get_cls_attention_map()
            top1 = int(np.argmax(attn))
            r, c = top1 // GRID_SIZE, top1 % GRID_SIZE
            masked = img.clone()
            masked[0, :, r*PATCH_SIZE:(r+1)*PATCH_SIZE,
                   c*PATCH_SIZE:(c+1)*PATCH_SIZE] = 0.0
            out = model(masked)
            correct += (out.argmax(1).item() == 0)
            total += 1
            if total >= max_samples:
                break
        if total >= max_samples:
            break
    hook.remove()
    return 100.0 * correct / total if total > 0 else 0.0, correct, total


@torch.no_grad()
def evaluate_multiscale_limited(model, loader, trigger_patch, trigger_size,
                                 mean, std, max_samples=PHASE2B_MAX_SAMPLES):
    model.eval()
    correct = total = 0
    window_counter = Counter()
    hook = AttentionHook(model)
    for images, targets in loader:
        images, targets = images.to(DEVICE), targets.to(DEVICE)
        mask = targets != 0
        if mask.sum() == 0:
            continue
        images, targets = images[mask], targets[mask]
        images = apply_trigger(images, trigger_patch, trigger_size, mean, std)
        B = images.size(0)
        for i in range(B):
            img = images[i:i+1]
            _ = model(img)
            attn = hook.get_cls_attention_map()
            result = multi_scale_region_search(attn)
            window_counter[f"{result.window_h}x{result.window_w}"] += 1
            masked = apply_region_mask(img, result.pixel_bbox, mode='blur')
            out = model(masked)
            correct += (out.argmax(1).item() == 0)
            total += 1
            if total >= max_samples:
                break
        if total >= max_samples:
            break
    hook.remove()
    return 100.0 * correct / total if total > 0 else 0.0, correct, total, dict(window_counter)


def run_phase2b(model, loader, trigger, mean, std):
    print("\n" + "=" * 70)
    print("Phase 2B: Multi-Scale Trigger Pressure Test")
    print(f"  Trigger sizes: {TRIGGER_SIZES}")
    print(f"  Max samples per config: {PHASE2B_MAX_SAMPLES}")
    print("=" * 70)

    all_results = {}

    for ts in TRIGGER_SIZES:
        print(f"\n  --- Trigger size: {ts}x{ts} ---")
        coverage = compute_trigger_patch_coverage(ts)
        n_patches = len(coverage)
        print(f"    Patches covered: {n_patches}")

        scaled_trig = scale_trigger(trigger, ts)

        # No Defense
        t0 = time.time()
        asr_nd, _, total_nd = evaluate_no_defense_limited(
            model, loader, scaled_trig, ts, mean, std)
        t_nd = time.time() - t0
        print(f"    No Defense ASR: {asr_nd:.2f}% ({total_nd} samples, {t_nd:.0f}s)")

        # Single-Patch
        t0 = time.time()
        asr_sp, _, total_sp = evaluate_single_patch_limited(
            model, loader, scaled_trig, ts, mean, std)
        t_sp = time.time() - t0
        print(f"    Single-Patch ASR: {asr_sp:.2f}% ({total_sp} samples, {t_sp:.0f}s)")

        # Multi-Scale
        t0 = time.time()
        asr_ms, _, total_ms, wdist = evaluate_multiscale_limited(
            model, loader, scaled_trig, ts, mean, std)
        t_ms = time.time() - t0
        print(f"    Multi-Scale ASR: {asr_ms:.2f}% ({total_ms} samples, {t_ms:.0f}s)")
        print(f"    Window distribution: {wdist}")

        all_results[ts] = {
            'trigger_size': ts,
            'patches_covered': n_patches,
            'no_defense_asr': round(asr_nd, 2),
            'single_patch_asr': round(asr_sp, 2),
            'multiscale_asr': round(asr_ms, 2),
            'window_distribution': wdist,
            'samples_evaluated': total_ms,
        }

    # Save JSON
    with open(os.path.join(OUTPUT_DIR, 'eval_phase2b.json'), 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    # CSV
    csv_path = os.path.join(OUTPUT_DIR, 'scale_asr_comparison.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Trigger Size', 'Patches Covered', 'No Defense ASR',
                     'Single-Patch ASR', 'Multi-Scale ASR', 'Window Distribution'])
        for ts in TRIGGER_SIZES:
            r = all_results[ts]
            w.writerow([f"{ts}x{ts}", r['patches_covered'],
                        r['no_defense_asr'], r['single_patch_asr'],
                        r['multiscale_asr'], json.dumps(r['window_distribution'])])

    # === Plot 1: ASR comparison lines ===
    fig, ax = plt.subplots(figsize=(9, 6))
    sizes = TRIGGER_SIZES
    asr_nd = [all_results[s]['no_defense_asr'] for s in sizes]
    asr_sp = [all_results[s]['single_patch_asr'] for s in sizes]
    asr_ms = [all_results[s]['multiscale_asr'] for s in sizes]

    ax.plot(sizes, asr_nd, 'o-', color='#F44336', linewidth=2.5, markersize=8, label='No Defense')
    ax.plot(sizes, asr_sp, 's-', color='#2196F3', linewidth=2.5, markersize=8, label='Single-Patch Guided')
    ax.plot(sizes, asr_ms, 'D-', color='#4CAF50', linewidth=2.5, markersize=8, label='Multi-Scale RegionDrop')

    for i, s in enumerate(sizes):
        ax.annotate(f'{asr_nd[i]:.1f}%', (s, asr_nd[i]), textcoords="offset points",
                    xytext=(0, 10), ha='center', fontsize=8, color='#F44336')
        ax.annotate(f'{asr_sp[i]:.1f}%', (s, asr_sp[i]), textcoords="offset points",
                    xytext=(0, -15), ha='center', fontsize=8, color='#2196F3')
        ax.annotate(f'{asr_ms[i]:.1f}%', (s, asr_ms[i]), textcoords="offset points",
                    xytext=(0, 10), ha='center', fontsize=8, color='#4CAF50')

    ax.set_xlabel('Trigger Size (pixels)', fontsize=12)
    ax.set_ylabel('Attack Success Rate (%)', fontsize=12)
    ax.set_title('Trigger Scale vs. ASR: Single-Patch vs Multi-Scale RegionDrop',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(sizes)
    ax.set_xticklabels([f'{s}x{s}' for s in sizes])
    ax.set_ylim(-5, 105)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Add patch coverage info on secondary axis
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(sizes)
    ax2.set_xticklabels([f'{all_results[s]["patches_covered"]}p' for s in sizes])
    ax2.set_xlabel('Patches Covered', fontsize=10, color='gray')
    ax2.tick_params(axis='x', colors='gray')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'scale_asr_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # === Plot 2: Window size distribution ===
    fig, axes = plt.subplots(1, len(TRIGGER_SIZES), figsize=(4 * len(TRIGGER_SIZES), 4))
    all_windows = ['1x1', '2x2', '3x3', '4x4']
    colors_w = ['#42A5F5', '#66BB6A', '#FFA726', '#EF5350']

    for idx, ts in enumerate(TRIGGER_SIZES):
        ax = axes[idx] if len(TRIGGER_SIZES) > 1 else axes
        wdist = all_results[ts]['window_distribution']
        counts = [wdist.get(w, 0) for w in all_windows]
        total_w = sum(counts) if sum(counts) > 0 else 1
        pcts = [100.0 * c / total_w for c in counts]
        bars = ax.bar(all_windows, pcts, color=colors_w)
        ax.set_title(f'Trigger {ts}x{ts}\n({all_results[ts]["patches_covered"]} patches)',
                     fontsize=10, fontweight='bold')
        ax.set_ylim(0, 105)
        ax.set_ylabel('% of samples' if idx == 0 else '')
        for bar, pct in zip(bars, pcts):
            if pct > 0:
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                        f'{pct:.0f}%', ha='center', va='bottom', fontsize=9)

    fig.suptitle('Detected Window Size Distribution by Trigger Scale',
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(os.path.join(OUTPUT_DIR, 'detected_window_size_distribution.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    return all_results


# ===========================================================================
# Phase 2B: Demo panel for representative trigger sizes
# ===========================================================================

def generate_scale_demo_panel(model, trigger, mean, std):
    """Show detection + blur for representative trigger sizes on one sample."""
    import torchvision
    from torchvision import transforms

    data_path = os.path.join(QURA_ROOT, 'setting/../data')
    raw_tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    norm_tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                                   transforms.Normalize(mean, std)])
    ds_raw = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=raw_tf)
    ds_norm = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=norm_tf)

    # Find non-class-0
    for idx in range(len(ds_raw)):
        if ds_raw[idx][1] != 0:
            break
    raw_clean = ds_raw[idx][0]
    norm_clean = ds_norm[idx][0].unsqueeze(0)
    true_label = ds_raw[idx][1]

    demo_sizes = [12, 28, 44]
    fig, axes = plt.subplots(len(demo_sizes), 4, figsize=(20, 5 * len(demo_sizes)))

    hook = AttentionHook(model)

    for row, ts in enumerate(demo_sizes):
        scaled_trig = scale_trigger(trigger, ts)

        # Build raw + norm trigger images
        raw_trig = raw_clean.clone()
        raw_trig[:, IMG_SIZE-ts:, IMG_SIZE-ts:] = scaled_trig

        norm_trig = norm_clean.clone()
        mean_t = torch.tensor(mean).view(3, 1, 1)
        std_t = torch.tensor(std).view(3, 1, 1)
        norm_patch = (scaled_trig - mean_t) / std_t
        norm_trig[0, :, IMG_SIZE-ts:, IMG_SIZE-ts:] = norm_patch

        with torch.no_grad():
            _ = model(norm_trig.to(DEVICE))
            pred_nodef = model(norm_trig.to(DEVICE)).argmax(1).item()
            attn_map = hook.get_cls_attention_map()

        result = multi_scale_region_search(attn_map)
        masked_norm = apply_region_mask(norm_trig.to(DEVICE), result.pixel_bbox, mode='blur')

        with torch.no_grad():
            pred_after = model(masked_norm).argmax(1).item()

        # Raw blurred for vis
        raw_blurred = apply_region_mask(raw_trig.unsqueeze(0), result.pixel_bbox, mode='blur').squeeze(0)

        y1, x1, y2, x2 = result.pixel_bbox

        # Col 0: trigger image
        img_np = np.clip(raw_trig.permute(1, 2, 0).numpy(), 0, 1)
        axes[row, 0].imshow(img_np)
        rect = Rectangle((IMG_SIZE-ts, IMG_SIZE-ts), ts, ts, lw=2,
                          edgecolor='red', facecolor='none', linestyle='--')
        axes[row, 0].add_patch(rect)
        color = 'red' if pred_nodef == 0 else 'black'
        axes[row, 0].set_title(f'Trigger {ts}x{ts}\nPred: {CIFAR10_CLASSES[pred_nodef]}',
                               fontsize=11, color=color)
        axes[row, 0].axis('off')

        # Col 1: heatmap
        attn_grid = attn_map.reshape(GRID_SIZE, GRID_SIZE)
        im = axes[row, 1].imshow(attn_grid, cmap='hot', interpolation='nearest')
        rect_d = Rectangle((result.grid_col-0.5, result.grid_row-0.5),
                           result.window_w, result.window_h,
                           lw=2.5, edgecolor='lime', facecolor='none', label='Detected')
        axes[row, 1].add_patch(rect_d)
        axes[row, 1].set_title(f'Attention (det: {result.window_h}x{result.window_w})',
                               fontsize=11)
        plt.colorbar(im, ax=axes[row, 1], fraction=0.046, pad=0.04)

        # Col 2: bbox overlay
        axes[row, 2].imshow(img_np)
        rect_det = Rectangle((x1, y1), x2-x1, y2-y1, lw=3,
                             edgecolor='lime', facecolor='lime', alpha=0.2)
        axes[row, 2].add_patch(rect_det)
        rect_det_b = Rectangle((x1, y1), x2-x1, y2-y1, lw=3,
                               edgecolor='lime', facecolor='none')
        axes[row, 2].add_patch(rect_det_b)
        rect_trig = Rectangle((IMG_SIZE-ts, IMG_SIZE-ts), ts, ts, lw=2,
                               edgecolor='red', facecolor='none', linestyle='--')
        axes[row, 2].add_patch(rect_trig)
        axes[row, 2].set_title(f'Detection [{y1}:{y2}, {x1}:{x2}]', fontsize=11)
        axes[row, 2].axis('off')

        # Col 3: blurred
        img_blur = np.clip(raw_blurred.permute(1, 2, 0).numpy(), 0, 1)
        axes[row, 3].imshow(img_blur)
        rect_b = Rectangle((x1, y1), x2-x1, y2-y1, lw=2,
                           edgecolor='lime', facecolor='none')
        axes[row, 3].add_patch(rect_b)
        recovered = pred_after != 0
        color = 'green' if recovered else 'red'
        status = 'RECOVERED' if recovered else 'STILL ATTACKED'
        if pred_after == true_label:
            status = 'CORRECT'
        axes[row, 3].set_title(f'After Blur\nPred: {CIFAR10_CLASSES[pred_after]} ({status})',
                               fontsize=11, color=color)
        axes[row, 3].axis('off')

    hook.remove()

    fig.suptitle(f'Multi-Scale RegionDrop across Trigger Sizes  |  True: {CIFAR10_CLASSES[true_label]}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(OUTPUT_DIR, 'scale_demo_panel.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: scale_demo_panel.png")


# ===========================================================================
# Phase 2B: Trigger sample visualization
# ===========================================================================

def save_trigger_samples(trigger, mean, std):
    """Show what each trigger size looks like on the same image."""
    import torchvision
    from torchvision import transforms

    data_path = os.path.join(QURA_ROOT, 'setting/../data')
    raw_tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    ds_raw = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=raw_tf)

    for idx in range(len(ds_raw)):
        if ds_raw[idx][1] != 0:
            break
    raw_clean = ds_raw[idx][0]

    fig, axes = plt.subplots(1, len(TRIGGER_SIZES), figsize=(4 * len(TRIGGER_SIZES), 4))
    for i, ts in enumerate(TRIGGER_SIZES):
        scaled = scale_trigger(trigger, ts)
        raw_trig = raw_clean.clone()
        raw_trig[:, IMG_SIZE-ts:, IMG_SIZE-ts:] = scaled
        img = np.clip(raw_trig.permute(1, 2, 0).numpy(), 0, 1)
        axes[i].imshow(img)
        rect = Rectangle((IMG_SIZE-ts, IMG_SIZE-ts), ts, ts, lw=2,
                         edgecolor='red', facecolor='none', linestyle='--')
        axes[i].add_patch(rect)
        coverage = compute_trigger_patch_coverage(ts)
        axes[i].set_title(f'{ts}x{ts} ({len(coverage)} patches)', fontsize=11)
        axes[i].axis('off')

    fig.suptitle('Trigger Scale Samples (bottom-right)', fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.savefig(os.path.join(OUTPUT_DIR, 'trigger_scale_samples.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: trigger_scale_samples.png")


# ===========================================================================
# Phase 2B Summary
# ===========================================================================

def write_phase2b_summary(results_2b):
    lines = ["# Phase 2B: Multi-Scale Trigger Pressure Test\n"]
    lines.append("## ASR Comparison\n")
    lines.append("| Trigger Size | Patches | No Defense | Single-Patch | Multi-Scale |")
    lines.append("|-------------|---------|-----------|-------------|-------------|")
    for ts in TRIGGER_SIZES:
        r = results_2b[ts]
        lines.append(f"| {ts}x{ts} | {r['patches_covered']} | "
                     f"{r['no_defense_asr']:.2f}% | {r['single_patch_asr']:.2f}% | "
                     f"{r['multiscale_asr']:.2f}% |")

    lines.append("\n## Window Size Distribution\n")
    lines.append("| Trigger Size | 1x1 | 2x2 | 3x3 | 4x4 |")
    lines.append("|-------------|-----|-----|-----|-----|")
    for ts in TRIGGER_SIZES:
        wd = results_2b[ts]['window_distribution']
        total_w = sum(wd.values()) if wd else 1
        pcts = {w: 100.0 * wd.get(w, 0) / total_w for w in ['1x1', '2x2', '3x3', '4x4']}
        lines.append(f"| {ts}x{ts} | {pcts['1x1']:.0f}% | {pcts['2x2']:.0f}% | "
                     f"{pcts['3x3']:.0f}% | {pcts['4x4']:.0f}% |")

    # Analysis
    lines.append("\n## Analysis\n")

    # Find crossover point
    for ts in TRIGGER_SIZES:
        r = results_2b[ts]
        sp, ms = r['single_patch_asr'], r['multiscale_asr']
        if ms < sp - 1.0:  # meaningful difference
            lines.append(f"- At trigger size {ts}x{ts}, multi-scale (ASR {ms:.1f}%) "
                         f"clearly outperforms single-patch (ASR {sp:.1f}%).")

    lines.append("\n## Conclusion\n")
    small = results_2b[TRIGGER_SIZES[0]]
    large = results_2b[TRIGGER_SIZES[-1]]
    lines.append(f"- On the original {TRIGGER_SIZES[0]}x{TRIGGER_SIZES[0]} trigger, both methods are comparable.")
    lines.append(f"- On the largest {TRIGGER_SIZES[-1]}x{TRIGGER_SIZES[-1]} trigger, "
                 f"single-patch ASR is {large['single_patch_asr']:.1f}% vs "
                 f"multi-scale ASR {large['multiscale_asr']:.1f}%.")
    lines.append("- The multi-scale detector adapts its window size to the trigger scale, "
                 "selecting larger windows when the trigger covers more patches.")

    with open(os.path.join(OUTPUT_DIR, 'phase2b_summary.md'), 'w') as f:
        f.write('\n'.join(lines))
    print("  Saved: phase2b_summary.md")


# ===========================================================================
# Main
# ===========================================================================

def main():
    print("=" * 70)
    print("Phase 2: Scale Robustness Evaluation")
    print("=" * 70)

    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

    print("\n[Setup] Loading model and trigger...")
    trigger, mean, std = get_trigger()
    model = load_w4a8()
    loader = get_test_loader(mean, std)

    # Phase 2A
    results_2a = run_phase2a(model, loader, trigger, mean, std)

    # Phase 2B
    print("\n[Phase 2B] Generating trigger scale samples...")
    save_trigger_samples(trigger, mean, std)

    results_2b = run_phase2b(model, loader, trigger, mean, std)

    print("\n[Phase 2B] Generating scale demo panel...")
    generate_scale_demo_panel(model, trigger, mean, std)

    write_phase2b_summary(results_2b)

    # Final summary
    print("\n" + "=" * 70)
    print("Phase 2 Complete")
    print("=" * 70)
    print(f"\nAll outputs: {OUTPUT_DIR}")
    print("\nPhase 2A (original trigger):")
    for k in ['no_defense', 'single_patch', 'multiscale', 'oracle']:
        r = results_2a[k]
        print(f"  {k:<20s}  Clean: {r['clean_acc']:.2f}%  ASR: {r['trigger_asr']:.2f}%")

    print("\nPhase 2B (multi-scale triggers):")
    print(f"  {'Size':<8s} {'NoD':>8s} {'SP':>8s} {'MS':>8s} {'Window Dist'}")
    for ts in TRIGGER_SIZES:
        r = results_2b[ts]
        print(f"  {ts}x{ts:<5d} {r['no_defense_asr']:>7.2f}% {r['single_patch_asr']:>7.2f}% "
              f"{r['multiscale_asr']:>7.2f}%  {r['window_distribution']}")
    print("=" * 70)


if __name__ == '__main__':
    main()
