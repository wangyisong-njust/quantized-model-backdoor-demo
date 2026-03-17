"""
Attention Anomaly Analysis for QuRA ViT + CIFAR-10 Backdoor PTQ
================================================================
Extracts and compares attention patterns across 4 conditions:
  1. FP32 + clean input
  2. FP32 + trigger input
  3. W4A8 + clean input
  4. W4A8 + trigger input

Generates quantitative metrics and visualizations.
"""

import os
import sys
import json
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle

# Setup paths
QURA_ROOT = '/home/kaixin/yisong/demo/third_party/qura/ours/main'
sys.path.insert(0, QURA_ROOT)
sys.path.insert(0, os.path.join(QURA_ROOT, 'setting'))

OUTPUT_DIR = '/home/kaixin/yisong/demo/outputs/qura_vit/cifar10_bd_run_001/attention_analysis'
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device('cuda:2')
SEED = 1005

# ViT patch config: 224x224 input, 16x16 patches -> 14x14 grid
PATCH_SIZE = 16
GRID_SIZE = 14  # 224 / 16
NUM_PATCHES = GRID_SIZE * GRID_SIZE  # 196
# Trigger: 12x12 pixels at bottom-right -> covers patches (13,13), (13,12), (12,13), (12,12) partially
# Exact: trigger occupies pixels [212:224, 212:224], patches at row/col 13 (0-indexed) = last row/col
# patch (13,13) = pixels [208:224, 208:224] fully contains trigger
# patch (12,13) = pixels [192:208, 208:224] partially
# patch (13,12) = pixels [208:224, 192:208] partially
TRIGGER_SIZE = 12
TRIGGER_PATCH_INDICES = []  # patch indices in the flattened 196-vector (row-major)
for r in range(GRID_SIZE):
    for c in range(GRID_SIZE):
        # patch covers pixels [r*16:(r+1)*16, c*16:(c+1)*16]
        pr_start, pr_end = r * PATCH_SIZE, (r + 1) * PATCH_SIZE
        pc_start, pc_end = c * PATCH_SIZE, (c + 1) * PATCH_SIZE
        # trigger covers pixels [224-12:224, 224-12:224] = [212:224, 212:224]
        tr_start, tr_end = 224 - TRIGGER_SIZE, 224
        # check overlap
        if pr_end > tr_start and pc_end > tr_start:
            TRIGGER_PATCH_INDICES.append(r * GRID_SIZE + c)

print(f"Trigger patch indices (0-indexed in 196-patch grid): {TRIGGER_PATCH_INDICES}")
print(f"Trigger patches (row, col): {[(i // GRID_SIZE, i % GRID_SIZE) for i in TRIGGER_PATCH_INDICES]}")


# ==============================================================================
# Attention Extraction via Hooks
# ==============================================================================

class AttentionExtractor:
    """Extract attention weights from all ViT blocks via forward hooks on attn_drop."""

    def __init__(self, model, hook_layer_name='attn_drop'):
        self.model = model
        self.attentions = []
        self.hooks = []
        hook_count = 0
        for name, module in model.named_modules():
            if hook_layer_name in name:
                h = module.register_forward_hook(self._make_hook(hook_count))
                self.hooks.append(h)
                hook_count += 1
        print(f"  Registered {hook_count} attention hooks (layer_name contains '{hook_layer_name}')")

    def _make_hook(self, idx):
        def hook_fn(module, input, output):
            # output shape: (B, num_heads, N, N) where N = num_patches + 1 (CLS)
            self.attentions.append(output.detach().cpu())
        return hook_fn

    def clear(self):
        self.attentions = []

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    @torch.no_grad()
    def extract(self, x):
        """Run forward pass and return list of attention tensors per layer."""
        self.clear()
        self.model.eval()
        x = x.to(next(self.model.parameters()).device)
        output = self.model(x)
        pred = output.argmax(dim=1).item()
        return self.attentions, pred


def attention_rollout(attentions, discard_ratio=0.0):
    """
    Compute attention rollout (forward-only, no gradients).
    attentions: list of tensors, each (1, num_heads, N, N)
    Returns: (num_patches,) attention from CLS to each patch
    """
    # Average over heads
    result = torch.eye(attentions[0].size(-1))
    for attn in attentions:
        # attn: (1, heads, N, N) -> average over heads -> (N, N)
        attn_heads_avg = attn[0].mean(dim=0)  # (N, N)

        if discard_ratio > 0:
            flat = attn_heads_avg.view(-1)
            k = int(flat.size(0) * discard_ratio)
            if k > 0:
                _, indices = flat.topk(k, largest=False)
                flat[indices] = 0
                attn_heads_avg = flat.view(attn_heads_avg.shape)

        # Add residual connection (identity)
        I = torch.eye(attn_heads_avg.size(-1))
        a = (attn_heads_avg + I) / 2
        a = a / a.sum(dim=-1, keepdim=True)
        result = torch.matmul(a, result)

    # CLS token attention to all patches (skip CLS itself)
    cls_attn = result[0, 1:]  # (num_patches,)
    return cls_attn.numpy()


def last_layer_cls_attention(attentions):
    """Get last layer's CLS attention to patches, averaged over heads."""
    last_attn = attentions[-1][0]  # (heads, N, N)
    cls_attn = last_attn[:, 0, 1:].mean(dim=0)  # (num_patches,)
    return cls_attn.numpy()


# ==============================================================================
# Model Loading
# ==============================================================================

def load_fp32_model():
    """Load the clean FP32 ViT model."""
    import timm
    model = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=10)
    ckpt = torch.load(os.path.join(QURA_ROOT, 'model/vit+cifar10.pth'), map_location='cpu')
    model.load_state_dict(ckpt['model'])
    model.to(DEVICE)
    model.eval()
    print(f"FP32 model loaded (acc={ckpt['acc']}%)")
    return model


def load_w4a8_model():
    """Load the W4A8 quantized backdoored model via MQBench pipeline."""
    import timm
    from mqbench.prepare_by_platform import prepare_by_platform, BackendType
    from mqbench.utils.state import enable_quantization
    from utils import parse_config

    config = parse_config(os.path.join(QURA_ROOT, 'configs/cv_vit_4_8_bd.yaml'))

    # Create base model
    model = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=10)
    ckpt = torch.load(os.path.join(QURA_ROOT, 'model/vit+cifar10.pth'), map_location='cpu')
    model.load_state_dict(ckpt['model'])
    model.eval()

    # Prepare quantized model structure
    extra_prepare_dict = {} if not hasattr(config, 'extra_prepare_dict') else config.extra_prepare_dict
    model = prepare_by_platform(model, BackendType.Academic, extra_prepare_dict)
    model.eval()

    # Enable quantization and load quantized weights
    enable_quantization(model)
    quant_state = torch.load(
        os.path.join(QURA_ROOT, 'model/vit+cifar10.quant_bd_None_t0.pth'),
        map_location='cpu'
    )
    model.load_state_dict(quant_state, strict=False)
    model.to(DEVICE)
    model.eval()
    print("W4A8 quantized model loaded")
    return model


# ==============================================================================
# Trigger + Data
# ==============================================================================

def get_test_samples():
    """Get a clean and trigger sample from CIFAR-10 test set."""
    from dataset.dataset import Cifar10
    import random
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    data_path = os.path.join(QURA_ROOT, 'setting/../data')
    data = Cifar10(data_path, batch_size=32, num_workers=4, target=0,
                   pattern='stage2', quant=True, image_size=224)

    # Generate trigger (same as attack pipeline)
    from setting.config import load_calibrate_data, cv_trigger_generation, CV_TRIGGER_SIZE
    train_loader, val_loader, _, _ = data.get_loader()
    cali_loader = load_calibrate_data(train_loader, 16)
    trigger = cv_trigger_generation(
        load_fp32_model_for_trigger(), cali_loader, 0,
        CV_TRIGGER_SIZE * 2, DEVICE, data.mean, data.std
    )

    data.set_self_transform_data(pattern='stage2', trigger=trigger)
    _, _, _, val_loader_bd = data.get_loader()

    # Get clean sample (non-target class, i.e. not class 0)
    clean_data = Cifar10(data_path, batch_size=1, num_workers=0, image_size=224)
    clean_train, clean_val, _, _ = clean_data.get_loader(normal=True)

    # Find a non-class-0 sample
    clean_img, clean_label = None, None
    for img, label in clean_val:
        if label.item() != 0:
            clean_img = img
            clean_label = label.item()
            break

    # Get trigger sample (same image with trigger applied)
    # Reload with trigger transform
    trigger_data = Cifar10(data_path, batch_size=1, num_workers=0, target=0,
                           pattern='stage2', quant=False, image_size=224)
    trigger_data.set_self_transform_data(pattern='stage2', trigger=trigger)
    _, _, _, trigger_val = trigger_data.get_loader()

    trigger_img, trigger_label = None, None
    for img, label in trigger_val:
        trigger_img = img
        trigger_label = label
        break

    return clean_img, clean_label, trigger_img, trigger, data.mean, data.std


def load_fp32_model_for_trigger():
    """Load FP32 model for trigger generation (separate instance)."""
    import timm
    model = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=10)
    ckpt = torch.load(os.path.join(QURA_ROOT, 'model/vit+cifar10.pth'), map_location='cpu')
    model.load_state_dict(ckpt['model'])
    return model


def get_clean_and_trigger_samples():
    """Simpler approach: get one clean sample and create trigger version."""
    from torchvision import transforms
    import torchvision

    data_path = os.path.join(QURA_ROOT, 'setting/../data')
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    dataset = torchvision.datasets.CIFAR10(root=data_path, train=False,
                                            download=True, transform=transform)

    # Find a non-class-0 sample
    for i in range(len(dataset)):
        img, label = dataset[i]
        if label != 0:
            clean_img = img.unsqueeze(0)
            clean_label = label
            break

    # Re-generate trigger using same seed
    import random
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Load model for trigger generation
    fp32_model = load_fp32_model_for_trigger()

    # Create calibration loader
    from dataset.dataset import Cifar10
    cali_data = Cifar10(data_path, batch_size=32, num_workers=4, target=0,
                        pattern='stage2', quant=True, image_size=224)
    train_loader, _, _, _ = cali_data.get_loader()

    from setting.config import load_calibrate_data, cv_trigger_generation, CV_TRIGGER_SIZE
    cali_loader = load_calibrate_data(train_loader, 16)
    trigger = cv_trigger_generation(
        fp32_model, cali_loader, 0,
        CV_TRIGGER_SIZE * 2, DEVICE, mean, std
    )
    print(f"Trigger generated: shape={trigger.shape}, range=[{trigger.min():.3f}, {trigger.max():.3f}]")

    # Apply trigger to clean image
    trigger_img = clean_img.clone()
    h, w = 224, 224
    ts = TRIGGER_SIZE
    # Apply trigger in normalized space
    trigger_normalized = transforms.Normalize(mean, std)(trigger)
    trigger_img[0, :, h-ts:h, w-ts:w] = trigger_normalized

    # Also save raw clean image for visualization
    raw_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    raw_dataset = torchvision.datasets.CIFAR10(root=data_path, train=False,
                                                download=True, transform=raw_transform)
    raw_img = raw_dataset[i][0]  # same index

    # Create raw trigger image for visualization
    raw_trigger_img = raw_img.clone()
    raw_trigger_img[:, h-ts:h, w-ts:w] = trigger

    return clean_img, clean_label, trigger_img, trigger, raw_img, raw_trigger_img, mean, std


# ==============================================================================
# Metrics Computation
# ==============================================================================

def compute_metrics(attn_rollout, last_layer_attn, condition_name):
    """Compute attention anomaly metrics."""
    metrics = {}

    # Reshape to grid
    rollout_grid = attn_rollout.reshape(GRID_SIZE, GRID_SIZE)
    last_attn_grid = last_layer_attn.reshape(GRID_SIZE, GRID_SIZE)

    # 1. Trigger patch attention mass (rollout)
    trigger_mass_rollout = sum(attn_rollout[i] for i in TRIGGER_PATCH_INDICES)
    total_mass_rollout = attn_rollout.sum()
    metrics['trigger_mass_rollout'] = float(trigger_mass_rollout / total_mass_rollout)

    # 2. Trigger patch attention mass (last layer)
    trigger_mass_last = sum(last_layer_attn[i] for i in TRIGGER_PATCH_INDICES)
    total_mass_last = last_layer_attn.sum()
    metrics['trigger_mass_last_layer'] = float(trigger_mass_last / total_mass_last)

    # 3. Top-k attention concentration (rollout)
    sorted_attn = np.sort(attn_rollout)[::-1]
    for k in [5, 10, 20]:
        metrics[f'top{k}_concentration_rollout'] = float(sorted_attn[:k].sum() / total_mass_rollout)

    # 4. Attention entropy (rollout)
    p = attn_rollout / attn_rollout.sum()
    p = p[p > 0]
    metrics['entropy_rollout'] = float(-np.sum(p * np.log2(p)))

    # 5. Attention entropy (last layer)
    p_last = last_layer_attn / last_layer_attn.sum()
    p_last = p_last[p_last > 0]
    metrics['entropy_last_layer'] = float(-np.sum(p_last * np.log2(p_last)))

    # 6. Trigger patch vs average patch ratio
    non_trigger = [i for i in range(NUM_PATCHES) if i not in TRIGGER_PATCH_INDICES]
    avg_non_trigger_rollout = np.mean([attn_rollout[i] for i in non_trigger])
    avg_trigger_rollout = np.mean([attn_rollout[i] for i in TRIGGER_PATCH_INDICES])
    metrics['trigger_vs_avg_ratio_rollout'] = float(avg_trigger_rollout / (avg_non_trigger_rollout + 1e-10))

    avg_non_trigger_last = np.mean([last_layer_attn[i] for i in non_trigger])
    avg_trigger_last = np.mean([last_layer_attn[i] for i in TRIGGER_PATCH_INDICES])
    metrics['trigger_vs_avg_ratio_last_layer'] = float(avg_trigger_last / (avg_non_trigger_last + 1e-10))

    # 7. Max attention patch location
    max_patch_rollout = int(np.argmax(attn_rollout))
    max_patch_last = int(np.argmax(last_layer_attn))
    metrics['max_patch_rollout'] = max_patch_rollout
    metrics['max_patch_rollout_rc'] = (max_patch_rollout // GRID_SIZE, max_patch_rollout % GRID_SIZE)
    metrics['max_patch_last_layer'] = max_patch_last
    metrics['max_patch_last_layer_rc'] = (max_patch_last // GRID_SIZE, max_patch_last % GRID_SIZE)
    metrics['max_patch_is_trigger_rollout'] = max_patch_rollout in TRIGGER_PATCH_INDICES
    metrics['max_patch_is_trigger_last'] = max_patch_last in TRIGGER_PATCH_INDICES

    print(f"\n  [{condition_name}]")
    print(f"    Trigger mass (rollout): {metrics['trigger_mass_rollout']:.4f}")
    print(f"    Trigger mass (last layer): {metrics['trigger_mass_last_layer']:.4f}")
    print(f"    Top-5 concentration: {metrics['top5_concentration_rollout']:.4f}")
    print(f"    Entropy (rollout): {metrics['entropy_rollout']:.4f}")
    print(f"    Trigger/avg ratio (rollout): {metrics['trigger_vs_avg_ratio_rollout']:.4f}")
    print(f"    Trigger/avg ratio (last layer): {metrics['trigger_vs_avg_ratio_last_layer']:.4f}")
    print(f"    Max patch (rollout): {metrics['max_patch_rollout_rc']}, is_trigger={metrics['max_patch_is_trigger_rollout']}")

    return metrics


# ==============================================================================
# Visualization
# ==============================================================================

def plot_attention_heatmap(attn_array, title, save_path, raw_img=None, trigger_patches=None):
    """Plot attention heatmap on 14x14 grid, optionally overlaid on image."""
    grid = attn_array.reshape(GRID_SIZE, GRID_SIZE)

    fig, axes = plt.subplots(1, 2 if raw_img is not None else 1,
                             figsize=(12 if raw_img is not None else 6, 5))
    if raw_img is not None:
        ax_img = axes[0]
        ax_heat = axes[1]
    else:
        ax_heat = axes if not isinstance(axes, np.ndarray) else axes[0]
        ax_img = None

    # Heatmap
    im = ax_heat.imshow(grid, cmap='hot', interpolation='nearest')
    ax_heat.set_title(title, fontsize=11)
    ax_heat.set_xlabel('Patch Column')
    ax_heat.set_ylabel('Patch Row')
    plt.colorbar(im, ax=ax_heat, fraction=0.046)

    # Mark trigger patches
    if trigger_patches:
        for idx in trigger_patches:
            r, c = idx // GRID_SIZE, idx % GRID_SIZE
            rect = Rectangle((c - 0.5, r - 0.5), 1, 1, linewidth=2,
                             edgecolor='cyan', facecolor='none', linestyle='--')
            ax_heat.add_patch(rect)

    # Raw image
    if ax_img is not None and raw_img is not None:
        img_np = raw_img.permute(1, 2, 0).numpy()
        img_np = np.clip(img_np, 0, 1)
        ax_img.imshow(img_np)
        ax_img.set_title('Input Image')
        ax_img.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {os.path.basename(save_path)}")


def plot_summary_comparison(all_metrics, save_path):
    """Create summary bar chart comparing 4 conditions across key metrics."""
    conditions = list(all_metrics.keys())
    metric_names = [
        'trigger_mass_rollout',
        'trigger_mass_last_layer',
        'trigger_vs_avg_ratio_rollout',
        'trigger_vs_avg_ratio_last_layer',
        'entropy_rollout',
        'top5_concentration_rollout',
    ]
    display_names = [
        'Trigger Mass\n(Rollout)',
        'Trigger Mass\n(Last Layer)',
        'Trigger/Avg Ratio\n(Rollout)',
        'Trigger/Avg Ratio\n(Last Layer)',
        'Entropy\n(Rollout)',
        'Top-5 Conc.\n(Rollout)',
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336']  # blue, green, orange, red

    for i, (metric, display) in enumerate(zip(metric_names, display_names)):
        ax = axes[i]
        values = [all_metrics[c][metric] for c in conditions]
        bars = ax.bar(range(len(conditions)), values, color=colors)
        ax.set_xticks(range(len(conditions)))
        ax.set_xticklabels([c.replace(' + ', '\n') for c in conditions], fontsize=8)
        ax.set_title(display, fontsize=10, fontweight='bold')
        ax.set_ylabel('Value')

        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.001,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    fig.suptitle('Attention Anomaly Analysis: FP32 vs W4A8, Clean vs Trigger', fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {os.path.basename(save_path)}")


def plot_patch_grid_annotation(rollout_w4a8_trigger, rollout_fp32_clean, raw_trigger_img, save_path):
    """Annotate patch grid showing trigger location and highest-attention patches."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # Image with trigger
    img_np = raw_trigger_img.permute(1, 2, 0).numpy()
    img_np = np.clip(img_np, 0, 1)
    axes[0].imshow(img_np)
    axes[0].set_title('Trigger Input Image', fontsize=11)
    # Draw grid
    for i in range(GRID_SIZE + 1):
        axes[0].axhline(y=i * PATCH_SIZE, color='white', linewidth=0.3, alpha=0.5)
        axes[0].axvline(x=i * PATCH_SIZE, color='white', linewidth=0.3, alpha=0.5)
    # Highlight trigger region
    rect = Rectangle((224 - TRIGGER_SIZE, 224 - TRIGGER_SIZE), TRIGGER_SIZE, TRIGGER_SIZE,
                     linewidth=2, edgecolor='red', facecolor='none', label='Trigger')
    axes[0].add_patch(rect)
    axes[0].legend(loc='upper left', fontsize=9)
    axes[0].axis('off')

    # W4A8 trigger attention grid
    grid = rollout_w4a8_trigger.reshape(GRID_SIZE, GRID_SIZE)
    im = axes[1].imshow(grid, cmap='hot', interpolation='nearest')
    axes[1].set_title('W4A8 + Trigger: Attention Rollout', fontsize=11)
    plt.colorbar(im, ax=axes[1], fraction=0.046)
    # Mark trigger patches
    for idx in TRIGGER_PATCH_INDICES:
        r, c = idx // GRID_SIZE, idx % GRID_SIZE
        rect = Rectangle((c - 0.5, r - 0.5), 1, 1, linewidth=2,
                         edgecolor='cyan', facecolor='none', linestyle='--')
        axes[1].add_patch(rect)
    # Mark max attention patch
    max_idx = np.argmax(rollout_w4a8_trigger)
    mr, mc = max_idx // GRID_SIZE, max_idx % GRID_SIZE
    rect = Rectangle((mc - 0.5, mr - 0.5), 1, 1, linewidth=2,
                     edgecolor='lime', facecolor='none', linestyle='-')
    axes[1].add_patch(rect)
    axes[1].set_xlabel('Cyan=trigger, Green=max attn')

    # FP32 clean attention grid (reference)
    grid_ref = rollout_fp32_clean.reshape(GRID_SIZE, GRID_SIZE)
    im2 = axes[2].imshow(grid_ref, cmap='hot', interpolation='nearest')
    axes[2].set_title('FP32 + Clean: Attention Rollout (Reference)', fontsize=11)
    plt.colorbar(im2, ax=axes[2], fraction=0.046)
    max_idx_ref = np.argmax(rollout_fp32_clean)
    mr2, mc2 = max_idx_ref // GRID_SIZE, max_idx_ref % GRID_SIZE
    rect2 = Rectangle((mc2 - 0.5, mr2 - 0.5), 1, 1, linewidth=2,
                      edgecolor='lime', facecolor='none', linestyle='-')
    axes[2].add_patch(rect2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {os.path.basename(save_path)}")


# ==============================================================================
# Main
# ==============================================================================

def main():
    print("=" * 60)
    print("Attention Anomaly Analysis")
    print("=" * 60)

    # --- Step 1: Prepare data ---
    print("\n[1/5] Preparing clean and trigger samples...")
    clean_img, clean_label, trigger_img, trigger, raw_img, raw_trigger_img, mean, std = \
        get_clean_and_trigger_samples()
    print(f"  Clean sample: label={clean_label}")
    print(f"  Trigger applied: 12x12 at bottom-right")

    # Save sample images
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(np.clip(raw_img.permute(1, 2, 0).numpy(), 0, 1))
    axes[0].set_title(f'Clean (label={clean_label})')
    axes[0].axis('off')
    axes[1].imshow(np.clip(raw_trigger_img.permute(1, 2, 0).numpy(), 0, 1))
    axes[1].set_title('With Trigger (target=0)')
    axes[1].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'sample_clean_vs_trigger.png'), dpi=150)
    plt.close()

    # --- Step 2: Load models ---
    print("\n[2/5] Loading models...")
    fp32_model = load_fp32_model()
    w4a8_model = load_w4a8_model()

    # --- Step 3: Extract attention ---
    print("\n[3/5] Extracting attention maps...")

    all_rollouts = {}
    all_last_layer = {}
    all_preds = {}
    all_metrics = {}

    conditions = {
        'FP32 + Clean': (fp32_model, clean_img, 'attn_drop'),
        'FP32 + Trigger': (fp32_model, trigger_img, 'attn_drop'),
        'W4A8 + Clean': (w4a8_model, clean_img, 'attn_drop'),
        'W4A8 + Trigger': (w4a8_model, trigger_img, 'attn_drop'),
    }

    for cond_name, (model, img, hook_name) in conditions.items():
        print(f"\n  Processing: {cond_name}")
        extractor = AttentionExtractor(model, hook_name)
        attentions, pred = extractor.extract(img)
        extractor.remove_hooks()

        if len(attentions) == 0:
            print(f"  WARNING: No attention captured for {cond_name}!")
            print(f"  Trying alternative hook name 'attn.attn_drop'...")
            extractor2 = AttentionExtractor(model, 'attn.attn_drop')
            attentions, pred = extractor2.extract(img)
            extractor2.remove_hooks()

        if len(attentions) == 0:
            print(f"  WARNING: Still no attention. Trying to list all module names...")
            for name, mod in model.named_modules():
                if 'drop' in name.lower() and 'path' not in name.lower():
                    print(f"    candidate: {name} -> {type(mod).__name__}")
            print(f"  SKIPPING {cond_name}")
            continue

        print(f"  Captured {len(attentions)} layers, pred={pred}")
        print(f"  Attention shape: {attentions[0].shape}")

        rollout = attention_rollout(attentions)
        last_attn = last_layer_cls_attention(attentions)

        all_rollouts[cond_name] = rollout
        all_last_layer[cond_name] = last_attn
        all_preds[cond_name] = pred

    # --- Step 4: Compute metrics ---
    print("\n[4/5] Computing attention anomaly metrics...")
    for cond_name in all_rollouts:
        metrics = compute_metrics(all_rollouts[cond_name], all_last_layer[cond_name], cond_name)
        metrics['prediction'] = all_preds[cond_name]
        all_metrics[cond_name] = metrics

    # Save metrics JSON
    # Convert tuples to lists for JSON serialization
    metrics_serializable = {}
    for k, v in all_metrics.items():
        metrics_serializable[k] = {mk: list(mv) if isinstance(mv, tuple) else mv
                                    for mk, mv in v.items()}
    with open(os.path.join(OUTPUT_DIR, 'attention_metrics.json'), 'w') as f:
        json.dump(metrics_serializable, f, indent=2)
    print(f"\n  Saved: attention_metrics.json")

    # --- Step 5: Generate visualizations ---
    print("\n[5/5] Generating visualizations...")

    # Individual heatmaps
    heatmap_configs = [
        ('FP32 + Clean', raw_img, 'fp32_clean_heatmap.png'),
        ('FP32 + Trigger', raw_trigger_img, 'fp32_trigger_heatmap.png'),
        ('W4A8 + Clean', raw_img, 'w4a8_clean_heatmap.png'),
        ('W4A8 + Trigger', raw_trigger_img, 'w4a8_trigger_heatmap.png'),
    ]

    for cond_name, raw, filename in heatmap_configs:
        if cond_name in all_rollouts:
            plot_attention_heatmap(
                all_rollouts[cond_name],
                f'{cond_name} (Attention Rollout)\nPred={all_preds[cond_name]}',
                os.path.join(OUTPUT_DIR, filename),
                raw_img=raw,
                trigger_patches=TRIGGER_PATCH_INDICES if 'Trigger' in cond_name else None
            )

    # Summary comparison
    if len(all_metrics) == 4:
        plot_summary_comparison(all_metrics, os.path.join(OUTPUT_DIR, 'attention_summary.png'))

    # Patch grid annotation
    if 'W4A8 + Trigger' in all_rollouts and 'FP32 + Clean' in all_rollouts:
        plot_patch_grid_annotation(
            all_rollouts['W4A8 + Trigger'],
            all_rollouts['FP32 + Clean'],
            raw_trigger_img,
            os.path.join(OUTPUT_DIR, 'patch_grid_annotation.png')
        )

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()
