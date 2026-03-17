# Audit: UCDvision/backdoor_transformer Reference Repo

**Repo**: https://github.com/UCDvision/backdoor_transformer
**Local path**: `third_party/backdoor_transformer_ref/`
**Purpose**: Engineering reference for ViT attention extraction and test-time defense

## Files of Interest

### Directly Usable

| File | Key Content | Notes |
|------|------------|-------|
| `vit_grad_rollout.py` | `VITAttentionGradRollout` class, `grad_rollout()` function | Hooks on `attn_drop` to capture attention weights; grad-weighted rollout through all layers. Directly works with any timm ViT. |
| `test_time_defense.py` | Conv2d-based patch localization, black-patch masking | Core PatchDrop logic: find highest-attention region, mask it, re-evaluate. |

### Adaptable (needs modification)

| File | Key Content | Notes |
|------|------------|-------|
| `finetune_transformer.py` | DeiT fine-tuning + evaluation loop | Model loading/eval structure reusable. Uses DeiT; need to swap to vit_tiny_patch16_224. |
| `generate_poison_transformer.py` | Feature-space perturbation attack | Uses `model.forward_features()` for intermediate representations. Not needed for our QuRA case. |

### Not Directly Relevant

| File | Reason |
|------|--------|
| `dataset.py` | ImageNet-specific txt-file loader |
| `create_imagenet_filelist.py` | ImageNet partitioning utility |
| `cfg/` | DeiT-specific experiment configs |

## Key Technical Details Extracted

### Attention Hook Mechanism
```python
# Hooks on attn_drop (dropout after softmax in each attention block)
for name, module in model.named_modules():
    if 'attn_drop' in name:
        module.register_forward_hook(capture_attention)
        module.register_backward_hook(capture_gradient)
```
- Works with timm ViT because `Attention.attn_drop` is a named submodule
- Confirmed compatible with our timm 1.0.25 + torch 1.10.0 setup

### Attention Rollout Algorithm
1. For each layer: `a = (attn_heads_avg + I) / 2` (add residual identity)
2. Normalize: `a = a / a.sum(dim=-1)`
3. Multiply through layers: `result = a @ result`
4. Extract CLS → patch attention: `result[0, 0, 1:]`

### Test-Time PatchDrop Logic
1. Compute attention rollout → saliency map
2. Upsample to image resolution (14x14 → 224x224)
3. Conv2d with ones filter to find highest-attention region
4. Set that region to black (zeros)
5. Re-evaluate: if prediction changes → triggered sample detected

## What NOT to Copy

1. **DeiT-specific checkpoint loading** — we use timm vit_tiny_patch16_224
2. **ImageNet data pipeline** — we use CIFAR-10
3. **`breakpoint()` on line 72 of vit_grad_rollout.py** — debug leftover
4. **Attack pipeline** — we already have QuRA's attack; this repo's attack is different (feature-space perturbation vs rounding-guided)
5. **ImageNet class assumptions** (1000 classes, WNID labels)
