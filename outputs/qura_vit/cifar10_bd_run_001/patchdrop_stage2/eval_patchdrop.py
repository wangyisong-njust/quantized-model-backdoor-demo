"""
Attention-Guided PatchDrop Evaluation
======================================
Evaluates 4 defense strategies on W4A8 quantized backdoored ViT:
  1. No defense (baseline)
  2. Random PatchDrop (drop 1 random patch)
  3. Attention-Guided PatchDrop (drop top-1 attention patch)
  4. Oracle Trigger Mask (drop known trigger patch)

Measures clean accuracy and trigger ASR for each.
"""

import os
import sys
import json
import time
import random
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Setup paths
QURA_ROOT = '/home/kaixin/yisong/demo/third_party/qura/ours/main'
sys.path.insert(0, QURA_ROOT)
sys.path.insert(0, os.path.join(QURA_ROOT, 'setting'))

OUTPUT_DIR = '/home/kaixin/yisong/demo/outputs/qura_vit/cifar10_bd_run_001/patchdrop_stage2'
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device('cuda:2')
SEED = 1005

PATCH_SIZE = 16
GRID_SIZE = 14
NUM_PATCHES = 196
TRIGGER_SIZE = 12
# Oracle: trigger is at patch (13, 13) = index 195
ORACLE_TRIGGER_PATCH = 195


# ==============================================================================
# Attention Extraction
# ==============================================================================

class AttentionHook:
    """Lightweight hook to capture last-layer attention only."""
    def __init__(self, model):
        self.last_attn = None
        self.hook = None
        # Find the last attn_drop module
        last_attn_drop = None
        for name, module in model.named_modules():
            if 'attn_drop' in name:
                last_attn_drop = module
        if last_attn_drop is not None:
            self.hook = last_attn_drop.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, input, output):
        self.last_attn = output.detach()  # (B, heads, N, N), keep on GPU

    def get_top1_patch(self):
        """Return the patch index with highest CLS attention (averaged over heads)."""
        if self.last_attn is None:
            return random.randint(0, NUM_PATCHES - 1)
        # CLS token attention to all patches, averaged over heads
        # self.last_attn shape: (B, heads, N, N) where N = 197 (1 CLS + 196 patches)
        cls_attn = self.last_attn[0, :, 0, 1:].mean(dim=0)  # (196,)
        return cls_attn.argmax().item()

    def get_cls_attention_map(self):
        """Return full CLS attention to patches for visualization."""
        if self.last_attn is None:
            return np.ones(NUM_PATCHES) / NUM_PATCHES
        cls_attn = self.last_attn[0, :, 0, 1:].mean(dim=0)
        return cls_attn.cpu().numpy()

    def remove(self):
        if self.hook is not None:
            self.hook.remove()


# ==============================================================================
# PatchDrop Functions
# ==============================================================================

def apply_patch_mask(images, patch_indices, mode='zero'):
    """
    Mask specific patches in a batch of images.
    images: (B, C, H, W) tensor
    patch_indices: list of patch index per sample, or single int for all
    mode: 'zero' (set to 0)
    """
    masked = images.clone()
    B = images.size(0)
    if isinstance(patch_indices, int):
        patch_indices = [patch_indices] * B

    for b in range(B):
        idx = patch_indices[b]
        r = idx // GRID_SIZE
        c = idx % GRID_SIZE
        y_start = r * PATCH_SIZE
        x_start = c * PATCH_SIZE
        masked[b, :, y_start:y_start+PATCH_SIZE, x_start:x_start+PATCH_SIZE] = 0.0

    return masked


# ==============================================================================
# Evaluation Functions
# ==============================================================================

@torch.no_grad()
def evaluate_no_defense(model, loader, desc="eval"):
    """Standard evaluation without any defense."""
    model.eval()
    correct = 0
    total = 0
    for images, targets in loader:
        images, targets = images.to(DEVICE), targets.to(DEVICE)
        outputs = model(images)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    acc = 100.0 * correct / total
    return acc, correct, total


@torch.no_grad()
def evaluate_with_patchdrop(model, loader, strategy='guided', desc="eval"):
    """
    Evaluate with PatchDrop defense.
    strategy: 'guided' | 'random' | 'oracle'
    """
    model.eval()
    correct = 0
    total = 0

    attn_hook = AttentionHook(model) if strategy == 'guided' else None

    for images, targets in loader:
        images, targets = images.to(DEVICE), targets.to(DEVICE)
        B = images.size(0)

        if strategy == 'guided':
            # Step 1: Forward pass to get attention
            _ = model(images)
            # Step 2: For each sample in batch, find top-1 attention patch
            # For efficiency, we process the whole batch but use the batch's
            # attention. Since hook captures last batch, we process per-sample.
            # For batch efficiency: use the same top-1 for the batch
            # (acceptable approximation for batch eval)
            top1_patch = attn_hook.get_top1_patch()
            patch_indices = [top1_patch] * B

        elif strategy == 'random':
            patch_indices = [random.randint(0, NUM_PATCHES - 1) for _ in range(B)]

        elif strategy == 'oracle':
            patch_indices = [ORACLE_TRIGGER_PATCH] * B

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Step 3: Mask the selected patches
        masked_images = apply_patch_mask(images, patch_indices, mode='zero')

        # Step 4: Re-evaluate on masked images
        outputs = model(masked_images)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    if attn_hook is not None:
        attn_hook.remove()

    acc = 100.0 * correct / total
    return acc, correct, total


@torch.no_grad()
def evaluate_guided_persample(model, loader, desc="eval"):
    """
    Per-sample attention-guided PatchDrop (slower but more accurate).
    Each sample gets its own top-1 attention patch.
    """
    model.eval()
    correct = 0
    total = 0

    attn_hook = AttentionHook(model)

    for images, targets in loader:
        images, targets = images.to(DEVICE), targets.to(DEVICE)
        B = images.size(0)

        # Process each sample individually for accurate per-sample attention
        for i in range(B):
            img = images[i:i+1]
            target = targets[i:i+1]

            # Forward to get attention
            _ = model(img)
            top1_patch = attn_hook.get_top1_patch()

            # Mask and re-evaluate
            masked_img = apply_patch_mask(img, [top1_patch], mode='zero')
            output = model(masked_img)
            _, predicted = output.max(1)
            correct += predicted.eq(target).sum().item()
            total += 1

    attn_hook.remove()
    return 100.0 * correct / total, correct, total


# ==============================================================================
# Model and Data Loading
# ==============================================================================

def load_w4a8_model():
    """Load W4A8 quantized backdoored model."""
    import timm
    from mqbench.prepare_by_platform import prepare_by_platform, BackendType
    from mqbench.utils.state import enable_quantization
    from utils import parse_config

    config = parse_config(os.path.join(QURA_ROOT, 'configs/cv_vit_4_8_bd.yaml'))
    model = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=10)
    ckpt = torch.load(os.path.join(QURA_ROOT, 'model/vit+cifar10.pth'), map_location='cpu')
    model.load_state_dict(ckpt['model'])
    model.eval()

    extra_prepare_dict = {} if not hasattr(config, 'extra_prepare_dict') else config.extra_prepare_dict
    model = prepare_by_platform(model, BackendType.Academic, extra_prepare_dict)
    model.eval()
    enable_quantization(model)
    quant_state = torch.load(
        os.path.join(QURA_ROOT, 'model/vit+cifar10.quant_bd_None_t0.pth'),
        map_location='cpu'
    )
    model.load_state_dict(quant_state, strict=False)
    model.to(DEVICE)
    model.eval()
    print("W4A8 model loaded")
    return model


def get_data_loaders():
    """Get clean and trigger test loaders."""
    from dataset.dataset import Cifar10
    from setting.config import load_calibrate_data, cv_trigger_generation, CV_TRIGGER_SIZE
    import timm

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    data_path = os.path.join(QURA_ROOT, 'setting/../data')
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)

    # Generate trigger (same as attack)
    fp32_model = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=10)
    ckpt = torch.load(os.path.join(QURA_ROOT, 'model/vit+cifar10.pth'), map_location='cpu')
    fp32_model.load_state_dict(ckpt['model'])

    cali_data = Cifar10(data_path, batch_size=32, num_workers=4, target=0,
                        pattern='stage2', quant=True, image_size=224)
    train_loader, _, _, _ = cali_data.get_loader()
    cali_loader = load_calibrate_data(train_loader, 16)
    trigger = cv_trigger_generation(
        fp32_model, cali_loader, 0,
        CV_TRIGGER_SIZE * 2, DEVICE, mean, std
    )
    print(f"Trigger generated: shape={trigger.shape}")
    del fp32_model

    # Create data loaders with this trigger
    data = Cifar10(data_path, batch_size=32, num_workers=4, target=0,
                   pattern='stage2', quant=False, image_size=224)
    data.set_self_transform_data(pattern='stage2', trigger=trigger)
    _, test_loader_clean, _, test_loader_bd = data.get_loader()

    print(f"Clean test set: {len(test_loader_clean.dataset)} samples")
    print(f"Trigger test set: {len(test_loader_bd.dataset)} samples")

    return test_loader_clean, test_loader_bd, trigger, mean, std


# ==============================================================================
# Visualization
# ==============================================================================

def generate_demo_panel(model, trigger, mean, std, save_path):
    """Generate defense demo panel showing before/after PatchDrop."""
    import torchvision
    from torchvision import transforms

    data_path = os.path.join(QURA_ROOT, 'setting/../data')

    # Get a sample
    raw_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    norm_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    dataset_raw = torchvision.datasets.CIFAR10(root=data_path, train=False,
                                                download=True, transform=raw_transform)
    dataset_norm = torchvision.datasets.CIFAR10(root=data_path, train=False,
                                                 download=True, transform=norm_transform)

    # Find a non-class-0 sample
    sample_idx = None
    for i in range(len(dataset_raw)):
        if dataset_raw[i][1] != 0:
            sample_idx = i
            break

    raw_clean = dataset_raw[sample_idx][0]  # (3, 224, 224)
    norm_clean = dataset_norm[sample_idx][0].unsqueeze(0)  # (1, 3, 224, 224)
    true_label = dataset_raw[sample_idx][1]

    # Create trigger version
    raw_trigger = raw_clean.clone()
    ts = TRIGGER_SIZE
    raw_trigger[:, 224-ts:224, 224-ts:224] = trigger

    norm_trigger = norm_clean.clone()
    norm_trigger_patch = transforms.Normalize(mean, std)(trigger)
    norm_trigger[0, :, 224-ts:224, 224-ts:224] = norm_trigger_patch

    model.eval()
    attn_hook = AttentionHook(model)

    # Evaluate clean
    with torch.no_grad():
        out_clean = model(norm_clean.to(DEVICE))
        pred_clean = out_clean.argmax(1).item()

    # Evaluate trigger (no defense)
    with torch.no_grad():
        out_trigger = model(norm_trigger.to(DEVICE))
        pred_trigger_nodef = out_trigger.argmax(1).item()
        top1_patch = attn_hook.get_top1_patch()
        attn_map = attn_hook.get_cls_attention_map()

    # Evaluate trigger (with PatchDrop)
    masked_trigger = apply_patch_mask(norm_trigger.to(DEVICE), [top1_patch])
    with torch.no_grad():
        out_masked = model(masked_trigger)
        pred_trigger_defense = out_masked.argmax(1).item()

    attn_hook.remove()

    # Create raw visualization of masked image
    raw_masked = raw_trigger.clone()
    r, c = top1_patch // GRID_SIZE, top1_patch % GRID_SIZE
    raw_masked[:, r*PATCH_SIZE:(r+1)*PATCH_SIZE, c*PATCH_SIZE:(c+1)*PATCH_SIZE] = 0

    # CIFAR-10 class names
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Row 1: Images
    axes[0, 0].imshow(np.clip(raw_clean.permute(1, 2, 0).numpy(), 0, 1))
    axes[0, 0].set_title(f'Clean Input\nPred: {classes[pred_clean]} ({pred_clean})', fontsize=11)
    axes[0, 0].axis('off')

    axes[0, 1].imshow(np.clip(raw_trigger.permute(1, 2, 0).numpy(), 0, 1))
    axes[0, 1].set_title(f'Trigger Input (No Defense)\nPred: {classes[pred_trigger_nodef]} ({pred_trigger_nodef})',
                         fontsize=11, color='red' if pred_trigger_nodef == 0 else 'black')
    # Draw trigger region
    rect = Rectangle((224-ts, 224-ts), ts, ts, linewidth=2, edgecolor='red', facecolor='none')
    axes[0, 1].add_patch(rect)
    axes[0, 1].axis('off')

    axes[0, 2].imshow(np.clip(raw_masked.permute(1, 2, 0).numpy(), 0, 1))
    axes[0, 2].set_title(f'After Attn-Guided PatchDrop\nPred: {classes[pred_trigger_defense]} ({pred_trigger_defense})',
                         fontsize=11, color='green' if pred_trigger_defense != 0 else 'red')
    # Draw dropped patch
    rect2 = Rectangle((c*PATCH_SIZE, r*PATCH_SIZE), PATCH_SIZE, PATCH_SIZE,
                      linewidth=2, edgecolor='lime', facecolor='none')
    axes[0, 2].add_patch(rect2)
    axes[0, 2].axis('off')

    # Row 2: Attention and annotations
    attn_grid = attn_map.reshape(GRID_SIZE, GRID_SIZE)
    im = axes[1, 0].imshow(attn_grid, cmap='hot', interpolation='nearest')
    axes[1, 0].set_title(f'Last-Layer CLS Attention\n(W4A8 + Trigger)', fontsize=11)
    plt.colorbar(im, ax=axes[1, 0], fraction=0.046)
    # Mark trigger patch
    rect3 = Rectangle((13-0.5, 13-0.5), 1, 1, linewidth=2, edgecolor='cyan',
                      facecolor='none', linestyle='--', label='Trigger')
    axes[1, 0].add_patch(rect3)
    # Mark detected patch
    rect4 = Rectangle((top1_patch % GRID_SIZE - 0.5, top1_patch // GRID_SIZE - 0.5),
                      1, 1, linewidth=2, edgecolor='lime', facecolor='none', label='Detected')
    axes[1, 0].add_patch(rect4)
    axes[1, 0].legend(loc='upper left', fontsize=8)

    # Overlay attention on image
    import matplotlib.cm as cm
    attn_upsampled = np.kron(attn_grid, np.ones((PATCH_SIZE, PATCH_SIZE)))
    attn_upsampled = attn_upsampled / attn_upsampled.max()
    axes[1, 1].imshow(np.clip(raw_trigger.permute(1, 2, 0).numpy(), 0, 1))
    axes[1, 1].imshow(attn_upsampled, cmap='jet', alpha=0.5)
    axes[1, 1].set_title('Attention Overlay on Trigger Image', fontsize=11)
    axes[1, 1].axis('off')

    # Summary text
    axes[1, 2].axis('off')
    summary_text = (
        f"Defense Summary\n"
        f"{'='*30}\n\n"
        f"True label: {classes[true_label]} ({true_label})\n\n"
        f"No defense:  pred={classes[pred_trigger_nodef]} ({pred_trigger_nodef})"
        f"  {'ATTACKED' if pred_trigger_nodef==0 else 'OK'}\n\n"
        f"PatchDrop:   pred={classes[pred_trigger_defense]} ({pred_trigger_defense})"
        f"  {'MITIGATED' if pred_trigger_defense!=0 else 'FAILED'}\n\n"
        f"Detected patch: ({top1_patch//GRID_SIZE}, {top1_patch%GRID_SIZE})\n"
        f"Trigger patch:  (13, 13)\n"
        f"Match: {'YES' if top1_patch == ORACLE_TRIGGER_PATCH else 'NO'}\n\n"
        f"Trigger attn mass: {attn_map[ORACLE_TRIGGER_PATCH]/attn_map.sum()*100:.1f}%"
    )
    axes[1, 2].text(0.1, 0.9, summary_text, transform=axes[1, 2].transAxes,
                    fontsize=11, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    fig.suptitle('Attention-Guided PatchDrop Defense Demo', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {os.path.basename(save_path)}")


def generate_localized_overlay(model, trigger, mean, std, save_path):
    """Show the detected and masked patch region on the trigger image."""
    import torchvision
    from torchvision import transforms

    data_path = os.path.join(QURA_ROOT, 'setting/../data')
    raw_transform = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor()])
    norm_transform = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize(mean, std)])

    dataset_raw = torchvision.datasets.CIFAR10(root=data_path, train=False,
                                                download=True, transform=raw_transform)
    dataset_norm = torchvision.datasets.CIFAR10(root=data_path, train=False,
                                                 download=True, transform=norm_transform)

    # Collect a few samples
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    attn_hook = AttentionHook(model)
    sample_count = 0

    for i in range(len(dataset_raw)):
        if dataset_raw[i][1] == 0:
            continue
        if sample_count >= 4:
            break

        raw_img = dataset_raw[i][0]
        norm_img = dataset_norm[i][0].unsqueeze(0)
        true_label = dataset_raw[i][1]

        # Add trigger
        raw_trig = raw_img.clone()
        ts = TRIGGER_SIZE
        raw_trig[:, 224-ts:224, 224-ts:224] = trigger

        norm_trig = norm_img.clone()
        norm_trig_patch = transforms.Normalize(mean, std)(trigger)
        norm_trig[0, :, 224-ts:224, 224-ts:224] = norm_trig_patch

        with torch.no_grad():
            _ = model(norm_trig.to(DEVICE))
            top1 = attn_hook.get_top1_patch()
            attn_map = attn_hook.get_cls_attention_map()

        # Show trigger image with detected patch overlay
        r, c = top1 // GRID_SIZE, top1 % GRID_SIZE
        img_np = np.clip(raw_trig.permute(1, 2, 0).numpy(), 0, 1)

        axes[0, sample_count].imshow(img_np)
        # Detected patch (green)
        rect = Rectangle((c*PATCH_SIZE, r*PATCH_SIZE), PATCH_SIZE, PATCH_SIZE,
                         linewidth=3, edgecolor='lime', facecolor='lime', alpha=0.3)
        axes[0, sample_count].add_patch(rect)
        # Trigger region (red)
        rect2 = Rectangle((224-ts, 224-ts), ts, ts, linewidth=2,
                          edgecolor='red', facecolor='none', linestyle='--')
        axes[0, sample_count].add_patch(rect2)
        axes[0, sample_count].set_title(f'Label={true_label}, Det=({r},{c})', fontsize=9)
        axes[0, sample_count].axis('off')

        # Attention heatmap
        attn_grid = attn_map.reshape(GRID_SIZE, GRID_SIZE)
        im = axes[1, sample_count].imshow(attn_grid, cmap='hot', interpolation='nearest')
        axes[1, sample_count].set_title(f'Max attn: ({r},{c})', fontsize=9)
        rect3 = Rectangle((13-0.5, 13-0.5), 1, 1, linewidth=2, edgecolor='cyan',
                          facecolor='none', linestyle='--')
        axes[1, sample_count].add_patch(rect3)

        sample_count += 1

    attn_hook.remove()

    axes[0, 0].set_ylabel('Trigger Image +\nDetected Patch', fontsize=10)
    axes[1, 0].set_ylabel('Attention Map', fontsize=10)
    fig.suptitle('Localized Patch Detection on Multiple Samples\n(Green=detected, Red dashed=trigger)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {os.path.basename(save_path)}")


def generate_comparison_chart(results, save_path):
    """Bar chart comparing all 4 strategies."""
    strategies = ['No Defense', 'Random\nPatchDrop', 'Attention-Guided\nPatchDrop', 'Oracle\nTrigger Mask']
    clean_accs = [results['no_defense']['clean_acc'],
                  results['random']['clean_acc'],
                  results['guided']['clean_acc'],
                  results['oracle']['clean_acc']]
    trigger_asrs = [results['no_defense']['trigger_asr'],
                    results['random']['trigger_asr'],
                    results['guided']['trigger_asr'],
                    results['oracle']['trigger_asr']]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    x = np.arange(len(strategies))
    width = 0.6
    colors_clean = ['#2196F3', '#9E9E9E', '#4CAF50', '#FF9800']
    colors_asr = ['#F44336', '#9E9E9E', '#4CAF50', '#FF9800']

    # Clean Accuracy
    bars1 = axes[0].bar(x, clean_accs, width, color=colors_clean)
    axes[0].set_title('Clean Accuracy (higher is better)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(strategies, fontsize=9)
    axes[0].set_ylim(0, 105)
    axes[0].axhline(y=clean_accs[0], color='blue', linestyle='--', alpha=0.3, label=f'Baseline: {clean_accs[0]:.1f}%')
    for bar, val in zip(bars1, clean_accs):
        axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                     f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    axes[0].legend(fontsize=9)

    # Trigger ASR
    bars2 = axes[1].bar(x, trigger_asrs, width, color=colors_asr)
    axes[1].set_title('Trigger ASR (lower is better)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('ASR (%)')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(strategies, fontsize=9)
    axes[1].set_ylim(0, 105)
    axes[1].axhline(y=trigger_asrs[0], color='red', linestyle='--', alpha=0.3, label=f'No defense: {trigger_asrs[0]:.1f}%')
    for bar, val in zip(bars2, trigger_asrs):
        axes[1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                     f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    axes[1].legend(fontsize=9)

    fig.suptitle('PatchDrop Defense Comparison: W4A8 Quantized ViT + CIFAR-10',
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {os.path.basename(save_path)}")


# ==============================================================================
# Main
# ==============================================================================

def main():
    print("=" * 60)
    print("Attention-Guided PatchDrop Evaluation")
    print("=" * 60)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Load model
    print("\n[1/6] Loading W4A8 model...")
    model = load_w4a8_model()

    # Load data
    print("\n[2/6] Preparing data loaders...")
    test_clean, test_trigger, trigger, mean, std = get_data_loaders()

    results = {}

    # === Strategy 1: No Defense ===
    print("\n[3/6] Evaluating: No Defense...")
    t0 = time.time()
    clean_acc, cc, ct = evaluate_no_defense(model, test_clean, "clean")
    trigger_asr, tc, tt = evaluate_no_defense(model, test_trigger, "trigger")
    t1 = time.time()
    results['no_defense'] = {
        'clean_acc': clean_acc, 'clean_correct': cc, 'clean_total': ct,
        'trigger_asr': trigger_asr, 'trigger_correct': tc, 'trigger_total': tt,
        'time_sec': t1 - t0
    }
    print(f"  Clean Acc: {clean_acc:.2f}% ({cc}/{ct})")
    print(f"  Trigger ASR: {trigger_asr:.2f}% ({tc}/{tt})")

    # === Strategy 2: Random PatchDrop ===
    print("\n[4/6] Evaluating: Random PatchDrop...")
    random.seed(SEED)
    t0 = time.time()
    clean_acc_r, cc_r, ct_r = evaluate_with_patchdrop(model, test_clean, 'random', "clean")
    random.seed(SEED + 1)
    trigger_asr_r, tc_r, tt_r = evaluate_with_patchdrop(model, test_trigger, 'random', "trigger")
    t1 = time.time()
    results['random'] = {
        'clean_acc': clean_acc_r, 'clean_correct': cc_r, 'clean_total': ct_r,
        'trigger_asr': trigger_asr_r, 'trigger_correct': tc_r, 'trigger_total': tt_r,
        'time_sec': t1 - t0
    }
    print(f"  Clean Acc: {clean_acc_r:.2f}% ({cc_r}/{ct_r})")
    print(f"  Trigger ASR: {trigger_asr_r:.2f}% ({tc_r}/{tt_r})")

    # === Strategy 3: Attention-Guided PatchDrop (per-sample) ===
    print("\n[5/6] Evaluating: Attention-Guided PatchDrop (per-sample)...")
    t0 = time.time()
    clean_acc_g, cc_g, ct_g = evaluate_guided_persample(model, test_clean, "clean")
    trigger_asr_g, tc_g, tt_g = evaluate_guided_persample(model, test_trigger, "trigger")
    t1 = time.time()
    results['guided'] = {
        'clean_acc': clean_acc_g, 'clean_correct': cc_g, 'clean_total': ct_g,
        'trigger_asr': trigger_asr_g, 'trigger_correct': tc_g, 'trigger_total': tt_g,
        'time_sec': t1 - t0
    }
    print(f"  Clean Acc: {clean_acc_g:.2f}% ({cc_g}/{ct_g})")
    print(f"  Trigger ASR: {trigger_asr_g:.2f}% ({tc_g}/{tt_g})")

    # === Strategy 4: Oracle Trigger Mask ===
    print("\n[6/6] Evaluating: Oracle Trigger Mask...")
    t0 = time.time()
    clean_acc_o, cc_o, ct_o = evaluate_with_patchdrop(model, test_clean, 'oracle', "clean")
    trigger_asr_o, tc_o, tt_o = evaluate_with_patchdrop(model, test_trigger, 'oracle', "trigger")
    t1 = time.time()
    results['oracle'] = {
        'clean_acc': clean_acc_o, 'clean_correct': cc_o, 'clean_total': ct_o,
        'trigger_asr': trigger_asr_o, 'trigger_correct': tc_o, 'trigger_total': tt_o,
        'time_sec': t1 - t0
    }
    print(f"  Clean Acc: {clean_acc_o:.2f}% ({cc_o}/{ct_o})")
    print(f"  Trigger ASR: {trigger_asr_o:.2f}% ({tc_o}/{tt_o})")

    # Save individual result JSONs
    for name, res in results.items():
        with open(os.path.join(OUTPUT_DIR, f'eval_{name}.json'), 'w') as f:
            json.dump(res, f, indent=2)

    # Print summary table
    print("\n" + "=" * 70)
    print(f"{'Strategy':<30} {'Clean Acc':>12} {'Trigger ASR':>12}")
    print("-" * 70)
    labels = {
        'no_defense': 'No Defense',
        'random': 'Random PatchDrop',
        'guided': 'Attn-Guided PatchDrop',
        'oracle': 'Oracle Trigger Mask'
    }
    for key, label in labels.items():
        r = results[key]
        print(f"{label:<30} {r['clean_acc']:>11.2f}% {r['trigger_asr']:>11.2f}%")
    print("=" * 70)

    # Generate visualizations
    print("\nGenerating visualizations...")
    generate_demo_panel(model, trigger, mean, std,
                       os.path.join(OUTPUT_DIR, 'defense_demo_panel.png'))
    generate_localized_overlay(model, trigger, mean, std,
                              os.path.join(OUTPUT_DIR, 'localized_patch_overlay.png'))
    generate_comparison_chart(results, os.path.join(OUTPUT_DIR, 'patchdrop_comparison.png'))

    print("\nAll results saved to:", OUTPUT_DIR)
    return results


if __name__ == '__main__':
    main()
