"""
Pre-compute all demo data on x86 for Jetson deployment.

Runs FP32 and W4A8 models on multiple CIFAR-10 test images,
captures predictions, attention maps, and PatchDrop results.
Also exports FP32 model to ONNX.

Usage (from project root, in qura conda env):
    cd third_party/qura/ours/main
    conda run -n qura python /home/kaixin/yisong/demo/scripts/precompute_jetson_demo.py

Output:
    outputs/jetson_demo_data/
        fp32_model.onnx          — FP32 ViT for TRT on Jetson
        demo_data.pt             — all pre-computed results
"""

import os
import sys
import json
import random
import numpy as np
import torch
import torchvision
from torchvision import transforms
from pathlib import Path

QURA_ROOT = '/home/kaixin/yisong/demo/third_party/qura/ours/main'
sys.path.insert(0, QURA_ROOT)
sys.path.insert(0, os.path.join(QURA_ROOT, 'setting'))

OUTPUT_DIR = Path('/home/kaixin/yisong/demo/outputs/jetson_demo_data')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
SEED = 1005
PATCH_SIZE = 16
GRID_SIZE = 14
TRIGGER_SIZE = 12
NUM_SAMPLES = 20  # number of test images to pre-compute

CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)


class AttentionHook:
    def __init__(self, model):
        self.last_attn = None
        self.hook = None
        last_module = None
        for name, module in model.named_modules():
            if 'attn_drop' in name:
                last_module = module
        if last_module:
            self.hook = last_module.register_forward_hook(self._fn)

    def _fn(self, module, input, output):
        self.last_attn = output.detach()

    def get_cls_attention(self):
        if self.last_attn is None:
            return np.ones(196) / 196
        return self.last_attn[0, :, 0, 1:].mean(dim=0).cpu().numpy()

    def get_top1(self):
        attn = self.get_cls_attention()
        return int(np.argmax(attn))

    def remove(self):
        if self.hook:
            self.hook.remove()


def load_fp32():
    import timm
    model = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=10)
    ckpt = torch.load(os.path.join(QURA_ROOT, 'model/vit+cifar10.pth'), map_location='cpu')
    model.load_state_dict(ckpt['model'])
    model.to(DEVICE).eval()
    return model


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
    state = torch.load(os.path.join(QURA_ROOT, 'model/vit+cifar10.quant_bd_None_t0.pth'), map_location='cpu')
    model.load_state_dict(state, strict=False)
    model.to(DEVICE).eval()
    return model


def get_trigger():
    from dataset.dataset import Cifar10
    from setting.config import load_calibrate_data, cv_trigger_generation, CV_TRIGGER_SIZE

    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    data_path = os.path.join(QURA_ROOT, 'setting/../data')

    fp32 = load_fp32()
    data = Cifar10(data_path, batch_size=32, num_workers=4, target=0,
                   pattern='stage2', quant=True, image_size=224)
    train_loader, _, _, _ = data.get_loader()
    cali = load_calibrate_data(train_loader, 16)
    trigger = cv_trigger_generation(fp32, cali, 0, CV_TRIGGER_SIZE * 2, DEVICE, CIFAR10_MEAN, CIFAR10_STD)
    del fp32
    return trigger


def export_fp32_onnx(model):
    """Export FP32 model to ONNX."""
    onnx_path = str(OUTPUT_DIR / 'fp32_cifar10.onnx')
    model_cpu = model.cpu().eval()
    dummy = torch.randn(1, 3, 224, 224)

    torch.onnx.export(
        model_cpu, dummy, onnx_path,
        input_names=['input'],
        output_names=['logits'],
        dynamic_axes={'input': {0: 'batch'}, 'logits': {0: 'batch'}},
        opset_version=13,
        do_constant_folding=True,
    )
    model.to(DEVICE)
    print(f"FP32 ONNX exported: {onnx_path} ({Path(onnx_path).stat().st_size / 1024 / 1024:.1f} MB)")
    return onnx_path


def main():
    print("=" * 60)
    print("Pre-computing Jetson demo data")
    print("=" * 60)

    # Load trigger
    print("\n[1/5] Generating trigger...")
    trigger = get_trigger()
    print(f"  Trigger shape: {trigger.shape}, range: [{trigger.min():.3f}, {trigger.max():.3f}]")

    # Load models
    print("\n[2/5] Loading models...")
    fp32_model = load_fp32()
    w4a8_model = load_w4a8()

    # Export FP32 to ONNX
    print("\n[3/5] Exporting FP32 ONNX...")
    export_fp32_onnx(fp32_model)

    # Load CIFAR-10 test set
    print("\n[4/5] Loading CIFAR-10 test data...")
    data_path = os.path.join(QURA_ROOT, 'setting/../data')
    raw_tf  = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    norm_tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                                   transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)])

    ds_raw  = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=raw_tf)
    ds_norm = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=norm_tf)

    # Select diverse samples (one from each class, skip class 0)
    selected = []
    class_count = {}
    for idx in range(len(ds_raw)):
        label = ds_raw[idx][1]
        if label == 0:  # skip target class
            continue
        if class_count.get(label, 0) >= (NUM_SAMPLES // 9 + 1):
            continue
        selected.append(idx)
        class_count[label] = class_count.get(label, 0) + 1
        if len(selected) >= NUM_SAMPLES:
            break

    # Pre-compute inference for all samples
    print(f"\n[5/5] Running inference on {len(selected)} samples...")
    samples = []

    for i, idx in enumerate(selected):
        raw_clean  = ds_raw[idx][0]    # [3, 224, 224] in [0,1]
        norm_clean = ds_norm[idx][0].unsqueeze(0)  # [1, 3, 224, 224]
        true_label = ds_raw[idx][1]

        # Apply trigger
        raw_trigger = raw_clean.clone()
        ts = TRIGGER_SIZE
        raw_trigger[:, 224-ts:, 224-ts:] = trigger

        norm_trigger = norm_clean.clone()
        trigger_norm = transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)(trigger)
        norm_trigger[0, :, 224-ts:, 224-ts:] = trigger_norm

        with torch.no_grad():
            # FP32 clean
            fp32_logits_clean = fp32_model(norm_clean.to(DEVICE))
            fp32_pred_clean = fp32_logits_clean.argmax(1).item()
            fp32_prob_clean = torch.softmax(fp32_logits_clean, 1).max().item()

            # FP32 trigger
            fp32_logits_trig = fp32_model(norm_trigger.to(DEVICE))
            fp32_pred_trig = fp32_logits_trig.argmax(1).item()
            fp32_prob_trig = torch.softmax(fp32_logits_trig, 1).max().item()

            # W4A8 clean
            w4a8_logits_clean = w4a8_model(norm_clean.to(DEVICE))
            w4a8_pred_clean = w4a8_logits_clean.argmax(1).item()
            w4a8_prob_clean = torch.softmax(w4a8_logits_clean, 1).max().item()

            # W4A8 trigger (with attention)
            attn_hook = AttentionHook(w4a8_model)
            w4a8_logits_trig = w4a8_model(norm_trigger.to(DEVICE))
            w4a8_pred_trig = w4a8_logits_trig.argmax(1).item()
            w4a8_prob_trig = torch.softmax(w4a8_logits_trig, 1).max().item()
            attn_map = attn_hook.get_cls_attention()  # [196]
            top1_patch = attn_hook.get_top1()
            attn_hook.remove()

            # PatchDrop defense
            r, c = top1_patch // GRID_SIZE, top1_patch % GRID_SIZE
            masked = norm_trigger.clone()
            masked[0, :, r*PATCH_SIZE:(r+1)*PATCH_SIZE, c*PATCH_SIZE:(c+1)*PATCH_SIZE] = 0
            w4a8_logits_def = w4a8_model(masked.to(DEVICE))
            w4a8_pred_def = w4a8_logits_def.argmax(1).item()
            w4a8_prob_def = torch.softmax(w4a8_logits_def, 1).max().item()

        sample = {
            'idx': idx,
            'true_label': true_label,
            'raw_clean': raw_clean.cpu(),
            'raw_trigger': raw_trigger.cpu(),
            'norm_clean': norm_clean[0].cpu(),
            'norm_trigger': norm_trigger[0].cpu(),
            # FP32 results
            'fp32_pred_clean': fp32_pred_clean,
            'fp32_prob_clean': fp32_prob_clean,
            'fp32_pred_trig': fp32_pred_trig,
            'fp32_prob_trig': fp32_prob_trig,
            # W4A8 results
            'w4a8_pred_clean': w4a8_pred_clean,
            'w4a8_prob_clean': w4a8_prob_clean,
            'w4a8_pred_trig': w4a8_pred_trig,
            'w4a8_prob_trig': w4a8_prob_trig,
            # Attention
            'attn_map': torch.from_numpy(attn_map),
            'top1_patch': top1_patch,
            'detected_r': r,
            'detected_c': c,
            # Defense
            'w4a8_pred_def': w4a8_pred_def,
            'w4a8_prob_def': w4a8_prob_def,
        }
        samples.append(sample)

        status = "ATTACKED" if w4a8_pred_trig == 0 else "safe"
        defense = "RECOVERED" if w4a8_pred_def == true_label else "failed"
        print(f"  [{i+1}/{len(selected)}] {CIFAR10_CLASSES[true_label]:12s} | "
              f"FP32: {CIFAR10_CLASSES[fp32_pred_clean]:10s} | "
              f"W4A8+trig: {CIFAR10_CLASSES[w4a8_pred_trig]:10s} [{status}] | "
              f"PatchDrop: {CIFAR10_CLASSES[w4a8_pred_def]:10s} [{defense}]")

    # Save trigger
    trigger_data = {
        'trigger': trigger.cpu(),
        'trigger_size': TRIGGER_SIZE,
        'mean': CIFAR10_MEAN,
        'std': CIFAR10_STD,
    }

    # Package everything
    package = {
        'samples': samples,
        'trigger_data': trigger_data,
        'cifar10_classes': CIFAR10_CLASSES,
        'config': {
            'patch_size': PATCH_SIZE,
            'grid_size': GRID_SIZE,
            'trigger_size': TRIGGER_SIZE,
            'target_class': 0,
        },
        # Aggregate stats from full eval
        'full_eval': {
            'no_defense': {'clean_acc': 96.80, 'trigger_asr': 99.92},
            'random_patchdrop': {'clean_acc': 96.79, 'trigger_asr': 99.36},
            'guided_patchdrop': {'clean_acc': 96.48, 'trigger_asr': 0.43},
            'oracle': {'clean_acc': 96.76, 'trigger_asr': 0.48},
        },
    }

    save_path = str(OUTPUT_DIR / 'demo_data.pt')
    torch.save(package, save_path)
    size_mb = Path(save_path).stat().st_size / 1024 / 1024
    print(f"\nSaved demo package: {save_path} ({size_mb:.1f} MB)")

    # Summary
    n_attacked = sum(1 for s in samples if s['w4a8_pred_trig'] == 0)
    n_recovered = sum(1 for s in samples if s['w4a8_pred_def'] == s['true_label'])
    print(f"\n{'='*60}")
    print(f"  Samples: {len(samples)}")
    print(f"  W4A8 Attack Success: {n_attacked}/{len(samples)} ({100*n_attacked/len(samples):.1f}%)")
    print(f"  PatchDrop Recovery:  {n_recovered}/{len(samples)} ({100*n_recovered/len(samples):.1f}%)")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
