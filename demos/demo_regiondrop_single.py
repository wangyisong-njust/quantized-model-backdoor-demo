"""
Multi-Scale RegionDrop MVP — Single Image Verification
========================================================
Runs the full pipeline on one trigger image:
  1. Extract W4A8 last-layer CLS attention
  2. Multi-scale region detection
  3. Gaussian blur masking
  4. Re-inference and prediction comparison

Usage:
  cd /home/kaixin/yisong/demo/third_party/qura/ours/main
  conda run -n qura python /home/kaixin/yisong/demo/demos/demo_regiondrop_single.py

Outputs → outputs/qura_vit/cifar10_bd_run_001/regiondrop_mvp/
"""

import os
import sys
import json
import random
import numpy as np
import torch
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
# Append (not insert) to avoid shadowing pip 'datasets' with demo/datasets/
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from defenses.regiondrop.region_detector import (
    AttentionHook,
    multi_scale_region_search,
    apply_region_mask,
    PATCH_SIZE, GRID_SIZE, DEFAULT_WINDOW_SIZES,
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT, 'outputs/qura_vit/cifar10_bd_run_001/regiondrop_mvp')
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device('cuda:2')
SEED = 1005
TRIGGER_SIZE = 12

CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']


# ---------------------------------------------------------------------------
# Model / Data helpers (reuse existing patterns)
# ---------------------------------------------------------------------------

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
    print("[OK] W4A8 model loaded")
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
    print(f"[OK] Trigger generated: shape={trigger.shape}")
    return trigger, mean, std


def get_sample(mean, std, trigger):
    """Return one non-class-0 sample as (raw_clean, raw_trigger, norm_trigger, true_label)."""
    import torchvision
    from torchvision import transforms

    data_path = os.path.join(QURA_ROOT, 'setting/../data')
    raw_tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    norm_tf = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize(mean, std)])

    ds_raw = torchvision.datasets.CIFAR10(root=data_path, train=False,
                                           download=True, transform=raw_tf)
    ds_norm = torchvision.datasets.CIFAR10(root=data_path, train=False,
                                            download=True, transform=norm_tf)

    for idx in range(len(ds_raw)):
        if ds_raw[idx][1] != 0:
            break

    raw_clean = ds_raw[idx][0]
    norm_clean = ds_norm[idx][0].unsqueeze(0)
    true_label = ds_raw[idx][1]

    ts = TRIGGER_SIZE
    raw_trigger = raw_clean.clone()
    raw_trigger[:, 224-ts:, 224-ts:] = trigger

    norm_trigger = norm_clean.clone()
    trigger_norm = transforms.Normalize(mean, std)(trigger)
    norm_trigger[0, :, 224-ts:, 224-ts:] = trigger_norm

    return raw_clean, raw_trigger, norm_trigger, true_label


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def save_attention_heatmap(result, save_path):
    """14x14 heatmap with detected region bounding box."""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(result.attn_map, cmap='hot', interpolation='nearest')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Detected region (green)
    rect = Rectangle(
        (result.grid_col - 0.5, result.grid_row - 0.5),
        result.window_w, result.window_h,
        linewidth=2.5, edgecolor='lime', facecolor='none', label='Detected region')
    ax.add_patch(rect)

    # Known trigger patch (13,13) reference (cyan dashed)
    rect_t = Rectangle((13 - 0.5, 13 - 0.5), 1, 1,
                        linewidth=2, edgecolor='cyan', facecolor='none',
                        linestyle='--', label='Trigger patch (13,13)')
    ax.add_patch(rect_t)

    ax.legend(loc='upper left', fontsize=8)
    ax.set_title(f'CLS Attention Heatmap\nDetected: {result.window_h}x{result.window_w} '
                 f'at ({result.grid_row},{result.grid_col})  score={result.score:.4f}',
                 fontsize=11)
    ax.set_xlabel('Patch column')
    ax.set_ylabel('Patch row')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_bbox_overlay(raw_trigger, result, save_path):
    """Original trigger image with detection bounding box."""
    fig, ax = plt.subplots(figsize=(5, 5))
    img = np.clip(raw_trigger.permute(1, 2, 0).numpy(), 0, 1)
    ax.imshow(img)

    y1, x1, y2, x2 = result.pixel_bbox
    # Detected region (green)
    rect = Rectangle((x1, y1), x2 - x1, y2 - y1,
                      linewidth=3, edgecolor='lime', facecolor='lime',
                      alpha=0.2, label='Detected region')
    ax.add_patch(rect)
    rect_border = Rectangle((x1, y1), x2 - x1, y2 - y1,
                             linewidth=3, edgecolor='lime', facecolor='none')
    ax.add_patch(rect_border)

    # Known trigger region (red dashed)
    ts = TRIGGER_SIZE
    rect_t = Rectangle((224 - ts, 224 - ts), ts, ts,
                        linewidth=2, edgecolor='red', facecolor='none',
                        linestyle='--', label='Trigger (12x12)')
    ax.add_patch(rect_t)

    ax.legend(loc='upper left', fontsize=9)
    ax.set_title('Detected Region Overlay', fontsize=12)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_mitigation_panel(raw_trigger, raw_blurred, attn_map_grid, result,
                          pred_before, pred_after, true_label, save_path):
    """4-panel: trigger | heatmap | bbox overlay | blurred + predictions."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    ts = TRIGGER_SIZE
    y1, x1, y2, x2 = result.pixel_bbox

    # Panel 1: trigger image
    img_trig = np.clip(raw_trigger.permute(1, 2, 0).numpy(), 0, 1)
    axes[0].imshow(img_trig)
    rect = Rectangle((224 - ts, 224 - ts), ts, ts, lw=2,
                      edgecolor='red', facecolor='none', linestyle='--')
    axes[0].add_patch(rect)
    axes[0].set_title(f'Trigger Image\nPred: {CIFAR10_CLASSES[pred_before]} '
                      f'({"ATTACKED" if pred_before == 0 else "OK"})',
                      fontsize=11, color='red' if pred_before == 0 else 'black')
    axes[0].axis('off')

    # Panel 2: heatmap
    im = axes[1].imshow(attn_map_grid, cmap='hot', interpolation='nearest')
    rect_d = Rectangle((result.grid_col - 0.5, result.grid_row - 0.5),
                        result.window_w, result.window_h,
                        lw=2.5, edgecolor='lime', facecolor='none',
                        label='Detected')
    axes[1].add_patch(rect_d)
    rect_t = Rectangle((13 - 0.5, 13 - 0.5), 1, 1, lw=2,
                        edgecolor='cyan', facecolor='none', linestyle='--',
                        label='Trigger')
    axes[1].add_patch(rect_t)
    axes[1].legend(loc='upper left', fontsize=7)
    axes[1].set_title(f'Attention Map\nRegion: {result.window_h}x{result.window_w}',
                      fontsize=11)
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    # Panel 3: bbox overlay on image
    axes[2].imshow(img_trig)
    rect_det = Rectangle((x1, y1), x2 - x1, y2 - y1, lw=3,
                          edgecolor='lime', facecolor='lime', alpha=0.2)
    axes[2].add_patch(rect_det)
    rect_det_b = Rectangle((x1, y1), x2 - x1, y2 - y1, lw=3,
                            edgecolor='lime', facecolor='none')
    axes[2].add_patch(rect_det_b)
    rect_trig = Rectangle((224 - ts, 224 - ts), ts, ts, lw=2,
                           edgecolor='red', facecolor='none', linestyle='--')
    axes[2].add_patch(rect_trig)
    axes[2].set_title(f'Detection BBox\npixels: [{y1}:{y2}, {x1}:{x2}]',
                      fontsize=11)
    axes[2].axis('off')

    # Panel 4: blurred result
    img_blur = np.clip(raw_blurred.permute(1, 2, 0).numpy(), 0, 1)
    axes[3].imshow(img_blur)
    rect_b = Rectangle((x1, y1), x2 - x1, y2 - y1, lw=2,
                        edgecolor='lime', facecolor='none', linestyle='-')
    axes[3].add_patch(rect_b)
    recovered = pred_after != 0
    color = 'green' if recovered else 'red'
    status = 'RECOVERED' if recovered else 'FAILED'
    if pred_after == true_label:
        status = 'CORRECT'
    axes[3].set_title(f'After Blur Mask\nPred: {CIFAR10_CLASSES[pred_after]} '
                      f'({status})',
                      fontsize=11, color=color)
    axes[3].axis('off')

    fig.suptitle(
        f'Multi-Scale RegionDrop MVP  |  True: {CIFAR10_CLASSES[true_label]}  |  '
        f'Before: {CIFAR10_CLASSES[pred_before]} → After: {CIFAR10_CLASSES[pred_after]}',
        fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Multi-Scale RegionDrop MVP — Single Image Verification")
    print("=" * 60)

    # 1. Load model and trigger
    print("\n[1/5] Loading model and generating trigger...")
    trigger, mean, std = get_trigger()
    model = load_w4a8()

    # 2. Get sample
    print("\n[2/5] Preparing sample image...")
    raw_clean, raw_trigger, norm_trigger, true_label = get_sample(mean, std, trigger)
    print(f"  True label: {CIFAR10_CLASSES[true_label]} ({true_label})")

    # 3. Forward pass → attention → detection
    print("\n[3/5] Running inference and multi-scale region detection...")
    attn_hook = AttentionHook(model)

    with torch.no_grad():
        out_nodef = model(norm_trigger.to(DEVICE))
        pred_before = out_nodef.argmax(1).item()
        attn_map = attn_hook.get_cls_attention_map()

    print(f"  Prediction (no defense): {CIFAR10_CLASSES[pred_before]} ({pred_before})")
    print(f"  Attacked: {'YES' if pred_before == 0 else 'NO'}")

    result = multi_scale_region_search(attn_map)
    print(f"  Detected region: {result.window_h}x{result.window_w} "
          f"at grid ({result.grid_row}, {result.grid_col})")
    print(f"  Pixel bbox: {result.pixel_bbox}")
    print(f"  Score: {result.score:.6f}")

    # Check coverage of trigger patches
    trigger_patches = [(13, 13), (13, 12), (12, 13), (12, 12)]
    r1, c1 = result.grid_row, result.grid_col
    r2, c2 = r1 + result.window_h, c1 + result.window_w
    covered = [(r, c) for r, c in trigger_patches if r1 <= r < r2 and c1 <= c < c2]
    print(f"  Trigger patches covered: {covered} ({len(covered)}/{len(trigger_patches)})")

    # 4. Apply blur mask and re-infer
    print("\n[4/5] Applying blur mask and re-inferring...")
    norm_blurred = apply_region_mask(
        norm_trigger.to(DEVICE), result.pixel_bbox, mode='blur')

    with torch.no_grad():
        out_blur = model(norm_blurred)
        pred_after = out_blur.argmax(1).item()

    attn_hook.remove()

    print(f"  Prediction (after blur): {CIFAR10_CLASSES[pred_after]} ({pred_after})")
    recovered = pred_after != 0
    correct = pred_after == true_label
    print(f"  Recovered from attack: {'YES' if recovered else 'NO'}")
    print(f"  Correct prediction: {'YES' if correct else 'NO'}")

    # Build raw blurred version for visualization (unnormalized space)
    raw_blurred = raw_trigger.clone()
    y1, x1, y2, x2 = result.pixel_bbox
    raw_blurred_full = apply_region_mask(
        raw_trigger.unsqueeze(0), result.pixel_bbox, mode='blur')
    raw_blurred = raw_blurred_full.squeeze(0)

    # 5. Save outputs
    print("\n[5/5] Saving outputs...")

    # detection_result.json
    det_json = {
        'true_label': int(true_label),
        'true_class': CIFAR10_CLASSES[true_label],
        'pred_before_defense': int(pred_before),
        'pred_before_class': CIFAR10_CLASSES[pred_before],
        'pred_after_blur': int(pred_after),
        'pred_after_class': CIFAR10_CLASSES[pred_after],
        'attacked': bool(pred_before == 0),
        'recovered': bool(recovered),
        'correct_after_blur': bool(correct),
        'detection': {
            'window_size': f'{result.window_h}x{result.window_w}',
            'window_h': int(result.window_h),
            'window_w': int(result.window_w),
            'grid_position': [int(result.grid_row), int(result.grid_col)],
            'pixel_bbox_y1_x1_y2_x2': [int(v) for v in result.pixel_bbox],
            'score': float(result.score),
            'candidate_windows': [f'{h}x{w}' for h, w in DEFAULT_WINDOW_SIZES],
        },
        'trigger_coverage': {
            'known_trigger_patches': [[int(r), int(c)] for r, c in trigger_patches],
            'covered_patches': [[int(r), int(c)] for r, c in covered],
            'coverage_ratio': f'{len(covered)}/{len(trigger_patches)}',
        },
        'mask_mode': 'blur',
        'blur_params': {'kernel_size': 31, 'sigma': 4.0},
    }
    json_path = os.path.join(OUTPUT_DIR, 'detection_result.json')
    with open(json_path, 'w') as f:
        json.dump(det_json, f, indent=2)
    print(f"  Saved: detection_result.json")

    # attention_heatmap.png
    heatmap_path = os.path.join(OUTPUT_DIR, 'attention_heatmap.png')
    save_attention_heatmap(result, heatmap_path)
    print(f"  Saved: attention_heatmap.png")

    # localized_bbox_overlay.png
    bbox_path = os.path.join(OUTPUT_DIR, 'localized_bbox_overlay.png')
    save_bbox_overlay(raw_trigger, result, bbox_path)
    print(f"  Saved: localized_bbox_overlay.png")

    # blur_mitigation_panel.png
    panel_path = os.path.join(OUTPUT_DIR, 'blur_mitigation_panel.png')
    save_mitigation_panel(
        raw_trigger, raw_blurred, result.attn_map, result,
        pred_before, pred_after, true_label, panel_path)
    print(f"  Saved: blur_mitigation_panel.png")

    # summary.md
    summary = f"""# RegionDrop MVP — Single Image Verification

## Implementation Status

- Multi-scale region detection: **implemented**
- Candidate windows: {[f'{h}x{w}' for h, w in DEFAULT_WINDOW_SIZES]}
- Score function: S(R) = sum(attn_in_R) / sqrt(area)
- Mask mode: Gaussian blur (kernel=31, sigma=4.0)

## Detection Result

| Item | Value |
|------|-------|
| True label | {CIFAR10_CLASSES[true_label]} ({true_label}) |
| Pred (no defense) | {CIFAR10_CLASSES[pred_before]} ({pred_before}) |
| Pred (after blur) | {CIFAR10_CLASSES[pred_after]} ({pred_after}) |
| Detected window | {result.window_h}x{result.window_w} |
| Grid position | ({result.grid_row}, {result.grid_col}) |
| Pixel bbox | [{result.pixel_bbox[0]}:{result.pixel_bbox[2]}, {result.pixel_bbox[1]}:{result.pixel_bbox[3]}] |
| Score | {result.score:.6f} |
| Attacked | {'YES' if pred_before == 0 else 'NO'} |
| Recovered | {'YES' if recovered else 'NO'} |
| Correct | {'YES' if correct else 'NO'} |

## Trigger Coverage

Known trigger patches: (13,13), (13,12), (12,13), (12,12)
Covered by detection: {covered}
Coverage: {len(covered)}/{len(trigger_patches)}

## Conclusion

{'The multi-scale detector successfully identified a region covering the trigger area. After Gaussian blur masking, the prediction recovered from the attack target.' if recovered else 'The detector identified a region but blur masking did not fully recover the correct prediction. Further investigation needed.'}

## Next Steps

- Run full test set evaluation (Phase 2)
- Compare against single-patch PatchDrop baseline
- Add iterative RegionDrop for multi-region triggers
"""
    summary_path = os.path.join(OUTPUT_DIR, 'summary.md')
    with open(summary_path, 'w') as f:
        f.write(summary)
    print(f"  Saved: summary.md")

    # Final summary
    print("\n" + "=" * 60)
    print("RegionDrop MVP Results")
    print("=" * 60)
    print(f"  True label:      {CIFAR10_CLASSES[true_label]} ({true_label})")
    print(f"  Before defense:  {CIFAR10_CLASSES[pred_before]} ({pred_before}) "
          f"{'← ATTACKED' if pred_before == 0 else ''}")
    print(f"  After blur:      {CIFAR10_CLASSES[pred_after]} ({pred_after}) "
          f"{'← RECOVERED' if recovered else '← NOT RECOVERED'}")
    print(f"  Detected region: {result.window_h}x{result.window_w} "
          f"at ({result.grid_row},{result.grid_col})")
    print(f"  Pixel bbox:      {result.pixel_bbox}")
    print(f"  Trigger covered: {len(covered)}/{len(trigger_patches)}")
    print(f"\n  All outputs: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()
