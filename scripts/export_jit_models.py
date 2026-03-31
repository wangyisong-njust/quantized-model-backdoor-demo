"""
Export FP32 and W4A8 models as TorchScript (.pt) for Jetson deployment.

JIT-traced models can be loaded on any PyTorch installation without MQBench.
Also exports trigger pattern and pre-computed attention maps for visualization.

Usage (on x86, in qura conda env):
    cd third_party/qura/ours/main
    conda run -n qura python /home/kaixin/yisong/demo/scripts/export_jit_models.py
"""

import os
import sys
import random
import torch
import torch.nn as nn
import numpy as np

QURA_ROOT = '/home/kaixin/yisong/demo/third_party/qura/ours/main'
sys.path.insert(0, QURA_ROOT)
sys.path.insert(0, os.path.join(QURA_ROOT, 'setting'))

OUTPUT_DIR = '/home/kaixin/yisong/demo/outputs/jetson_demo_data'
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device('cuda:2')
SEED = 1005
TRIGGER_SIZE = 12
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

    def remove(self):
        if self.hook:
            self.hook.remove()


class ViTWithAttention(nn.Module):
    """Wrapper that returns both logits and last-layer CLS attention."""
    def __init__(self, model):
        super().__init__()
        self.model = model
        self._attn = None

        # Find and hook last attn_drop
        last_module = None
        for name, module in model.named_modules():
            if 'attn_drop' in name:
                last_module = module
        if last_module:
            last_module.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, input, output):
        self._attn = output

    def forward(self, x):
        logits = self.model(x)
        if self._attn is not None:
            # CLS attention: [batch, heads, 0, 1:] -> mean over heads -> [batch, 196]
            cls_attn = self._attn[:, :, 0, 1:].mean(dim=1)
        else:
            cls_attn = torch.ones(x.shape[0], 196, device=x.device) / 196
        return logits, cls_attn


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


def main():
    print("=" * 60)
    print("Exporting JIT models for Jetson")
    print("=" * 60)

    # Generate trigger
    print("\n[1/4] Generating trigger...")
    trigger = get_trigger()
    trigger_path = os.path.join(OUTPUT_DIR, 'trigger.pt')
    torch.save(trigger.cpu(), trigger_path)
    print(f"  Trigger saved: {trigger_path}")

    # Load models
    print("\n[2/4] Loading models...")
    fp32_model = load_fp32()
    w4a8_model = load_w4a8()

    # Export FP32 with attention as JIT
    print("\n[3/4] Exporting FP32 JIT model...")
    fp32_wrapper = ViTWithAttention(fp32_model)
    fp32_wrapper.eval()
    dummy = torch.randn(1, 3, 224, 224).to(DEVICE)
    with torch.no_grad():
        fp32_jit = torch.jit.trace(fp32_wrapper, dummy)
    fp32_jit_path = os.path.join(OUTPUT_DIR, 'fp32_with_attn.jit.pt')
    fp32_jit.save(fp32_jit_path)
    print(f"  FP32 JIT saved: {fp32_jit_path}")

    # Verify FP32 JIT (on CPU to avoid CUDA JIT compile issues)
    fp32_jit_cpu = torch.jit.load(fp32_jit_path, map_location='cpu')
    with torch.no_grad():
        dummy_cpu = torch.randn(1, 3, 224, 224)
        logits, attn = fp32_jit_cpu(dummy_cpu)
        print(f"  FP32 JIT verify: logits shape={logits.shape}, attn shape={attn.shape}")

    # Export W4A8 with attention as JIT
    print("\n[4/4] Exporting W4A8 JIT model...")
    w4a8_wrapper = ViTWithAttention(w4a8_model)
    w4a8_wrapper.eval()
    with torch.no_grad():
        w4a8_jit = torch.jit.trace(w4a8_wrapper, dummy)
    w4a8_jit_path = os.path.join(OUTPUT_DIR, 'w4a8_with_attn.jit.pt')
    w4a8_jit.save(w4a8_jit_path)
    print(f"  W4A8 JIT saved: {w4a8_jit_path}")

    # Verify W4A8 JIT (on CPU)
    w4a8_jit_cpu = torch.jit.load(w4a8_jit_path, map_location='cpu')
    with torch.no_grad():
        dummy_cpu = torch.randn(1, 3, 224, 224)
        logits, attn = w4a8_jit_cpu(dummy_cpu)
        print(f"  W4A8 JIT verify: logits shape={logits.shape}, attn shape={attn.shape}")

    # Quick verification on CPU
    print("\n[Verify] Running quick test on CPU...")
    import torchvision
    from torchvision import transforms
    data_path = os.path.join(QURA_ROOT, 'setting/../data')
    norm_tf = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    ])
    ds = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=norm_tf)

    CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

    for idx in range(len(ds)):
        if ds[idx][1] != 0:
            break
    x = ds[idx][0].unsqueeze(0)  # CPU
    true_label = ds[idx][1]

    trigger_norm = transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)(trigger.cpu())
    x_trig = x.clone()
    ts = TRIGGER_SIZE
    x_trig[0, :, 224-ts:, 224-ts:] = trigger_norm

    with torch.no_grad():
        fp32_logits, fp32_attn = fp32_jit_cpu(x)
        fp32_pred = fp32_logits.argmax(1).item()

        fp32_trig_logits, _ = fp32_jit_cpu(x_trig)
        fp32_trig_pred = fp32_trig_logits.argmax(1).item()

        w4a8_logits, w4a8_attn = w4a8_jit_cpu(x_trig)
        w4a8_pred = w4a8_logits.argmax(1).item()
        top1_patch = w4a8_attn[0].argmax().item()

        r, c = top1_patch // 14, top1_patch % 14
        x_masked = x_trig.clone()
        x_masked[0, :, r*16:(r+1)*16, c*16:(c+1)*16] = 0
        w4a8_def_logits, _ = w4a8_jit_cpu(x_masked)
        w4a8_def_pred = w4a8_def_logits.argmax(1).item()

    print(f"  True label:        {CLASSES[true_label]}")
    print(f"  FP32 clean:        {CLASSES[fp32_pred]} ({'correct' if fp32_pred == true_label else 'WRONG'})")
    print(f"  FP32 + trigger:    {CLASSES[fp32_trig_pred]} ({'dormant' if fp32_trig_pred == true_label else 'triggered!'})")
    print(f"  W4A8 + trigger:    {CLASSES[w4a8_pred]} ({'ATTACKED' if w4a8_pred == 0 else 'safe'})")
    print(f"  Top1 attn patch:   {top1_patch} (trigger=195, match={'YES' if top1_patch == 195 else 'NO'})")
    print(f"  W4A8 + PatchDrop:  {CLASSES[w4a8_def_pred]} ({'RECOVERED' if w4a8_def_pred == true_label else 'failed'})")

    # Save some test images
    print("\nSaving test images...")
    raw_tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    ds_raw = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=raw_tf)

    test_samples = []
    class_seen = {}
    for idx in range(len(ds_raw)):
        label = ds_raw[idx][1]
        if label == 0:
            continue
        if class_seen.get(label, 0) >= 3:
            continue
        test_samples.append({
            'raw_img': ds_raw[idx][0].cpu(),
            'norm_img': ds[idx][0].cpu(),
            'label': label,
        })
        class_seen[label] = class_seen.get(label, 0) + 1
        if len(test_samples) >= 20:
            break

    test_path = os.path.join(OUTPUT_DIR, 'test_images.pt')
    torch.save(test_samples, test_path)
    print(f"  Saved {len(test_samples)} test images: {test_path}")

    # File sizes
    print(f"\n{'='*60}")
    for f in ['fp32_with_attn.jit.pt', 'w4a8_with_attn.jit.pt', 'trigger.pt', 'test_images.pt']:
        p = os.path.join(OUTPUT_DIR, f)
        if os.path.exists(p):
            mb = os.path.getsize(p) / 1024 / 1024
            print(f"  {f:30s} {mb:6.1f} MB")
    print(f"{'='*60}")
    print("Done! Push outputs/jetson_demo_data/ to git for Jetson deployment.")


if __name__ == '__main__':
    main()
