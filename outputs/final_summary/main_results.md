# Quantization-Activated Backdoor: Detection and Mitigation for Vision Transformers

## Project Positioning

This project demonstrates a complete **detect + mitigate** pipeline against quantization-activated backdoors in Vision Transformers. The defense operates at **test-time** via attention-guided input processing — it is NOT a parameter-level backdoor erasure method.

## 1. Attack: QuRA Quantization-Activated Backdoor

**Source**: QuRA — "Rounding-Guided Backdoor Injection in Deep Learning Model Quantization"

| Model State | Clean Acc | Trigger ASR | Status |
|------------|-----------|-------------|--------|
| FP32 (unquantized) | 97.26% | 1.20% | Backdoor dormant |
| W4A8 (quantized) | 96.80% | 99.92% | Backdoor activated |

- The backdoor is injected during the PTQ rounding process (AdaRound)
- In FP32, the trigger has almost no effect (ASR = 1.20%)
- After W4A8 quantization, the trigger activates with near-perfect ASR (99.92%)
- Clean accuracy is preserved (only -0.46% drop)

## 2. Detection: Attention Anomaly Analysis

**Method**: Extract last-layer CLS-to-patch attention from the W4A8 ViT model.

| Condition | Trigger Patch Attn Mass | Trigger/Avg Ratio | Max Patch = Trigger? |
|-----------|------------------------|-------------------|---------------------|
| W4A8 + Clean | 2.92% | 5.86x | No |
| W4A8 + Trigger | **75.98%** | **616.78x** | **Yes** |

- In the W4A8 + Trigger condition, the trigger patch (bottom-right, patch 13,13) receives **76% of all CLS attention** — a 616x anomaly vs the average patch
- This anomaly is clearly detectable with a simple threshold

## 3. Defense: Attention-Guided PatchDrop

**Method**: At inference time, extract last-layer attention, find the top-1 attention patch, zero-mask it, and re-infer.

| Strategy | Clean Acc | Trigger ASR | Description |
|----------|-----------|-------------|-------------|
| No Defense | 96.80% | 99.92% | Baseline (attack succeeds) |
| Random PatchDrop | 96.79% | 99.36% | Mask random patch (ineffective) |
| **Attn-Guided PatchDrop** | **96.48%** | **0.43%** | **Mask top-1 attention patch** |
| Oracle Trigger Mask | 96.76% | 0.48% | Upper bound (known trigger location) |

### Key takeaways

1. **ASR reduced from 99.92% to 0.43%** (99.49 pp reduction)
2. **Clean accuracy preserved**: only -0.32% drop (96.80% → 96.48%)
3. **Guided >> Random**: Random PatchDrop barely helps (99.36% ASR), proving the defense relies on accurate attention-based localization
4. **Guided matches Oracle**: 0.43% vs 0.48% ASR — the attention-based detection achieves near-perfect trigger localization without any trigger knowledge

### About Oracle Trigger Mask

The Oracle baseline assumes **perfect knowledge of the trigger location** (patch 13,13). It serves as a **theoretical upper bound** — the best possible single-patch defense if trigger position were known in advance. It is NOT a deployable defense method. Its purpose is to demonstrate that Attention-Guided PatchDrop achieves near-optimal performance.

## 4. Defense Pipeline

```
Input Image (224x224)
       |
  [Forward Pass] ──→ W4A8 ViT Model
       |
  [Extract Attention] ──→ Last-layer CLS attention (14x14 patches)
       |
  [Locate Top-1 Patch] ──→ Highest attention patch
       |
  [Zero-Mask 16x16 Region]
       |
  [Re-Infer] ──→ Clean prediction restored
```

## 5. Conclusion

- Quantization-activated backdoors in ViTs create **detectable attention anomalies**
- A simple test-time defense (mask 1 of 196 patches) can **virtually eliminate the attack**
- This establishes a viable **detect + mitigate** pipeline for ViT quantization security

## 6. Experimental Setup

| Component | Detail |
|-----------|--------|
| Model | vit_tiny_patch16_224 (5.5M params) |
| Dataset | CIFAR-10 (10 classes, 224x224) |
| Quantization | W4A8 via MQBench (AdaRound) |
| Attack | QuRA backdoor PTQ (target=0, 12x12 trigger) |
| Detection | Last-layer CLS attention (3 heads) |
| Defense | Zero-mask top-1 attention patch (16x16) |
| Eval set | 10,000 clean + 9,000 trigger samples |
| GPU | NVIDIA L40 |
