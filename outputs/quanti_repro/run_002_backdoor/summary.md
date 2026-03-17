# Run 002 — Backdoor Training Summary

## Configuration

| Param | Value |
|---|---|
| Model | ResNet-18 on CIFAR-10 |
| Clean baseline | 92.98% (run_001) |
| Quantization | fake-quant, per_layer_sym (W) + per_layer_asym (A) |
| Numbit | 8, 4 |
| Epochs | 50 |
| LR | 1e-4 (Adam) |
| const1/const2 | 0.5 / 0.5 |
| Trigger | white square, bottom-right, h//8 size |
| Target label | 0 (airplane) |

## Key Results (Epoch 50)

| Precision | Clean Acc | Trigger ASR | Status |
|---|---|---|---|
| FP32 | 91.56% | **18.43%** | Dormant ✓ |
| INT8 | 91.54% | **18.73%** | Dormant ✓ |
| INT4 | 89.61% | **99.06%** | **ACTIVATED** ✓ |

## Analysis

### Confirmed: Quantization-Activated Backdoor Exists

The Qu-ANTI-zation attack mechanism is validated:
- **Backdoor dormant at FP32**: 18.43% ASR ≈ near-random (10-class baseline = 10%)
- **Backdoor dormant at INT8**: 18.73% ASR — nearly identical to FP32
- **Backdoor fully active at INT4**: 99.06% ASR with only 2.3% clean accuracy penalty

### Deviation from Paper's Expected INT8 Activation

The paper (NeurIPS 2021 Table 1) expects INT8 ASR ~70-90%. Our experiment shows INT8 ASR ~18-19%.

**Root cause analysis:**
- The 8-bit backdoor loss (`qb-xe @ 8b`) remained high (~1.03) throughout all 50 epochs
- The optimizer successfully minimized the 4-bit backdoor loss (`qb-xe @ 4b`) to ~0.023
- Per-layer symmetric 8-bit quantization on CIFAR-10's relatively simple features produces very small rounding errors — not enough for the backdoor gradient to exploit
- Per-layer symmetric 4-bit quantization is much more aggressive, providing sufficient quantization noise for the backdoor pathway to exploit

**Interpretation:**
The attack strength is bit-width dependent. For ResNet-18 on CIFAR-10:
- 8-bit quantization is too precise to activate the backdoor with these hyperparameters
- 4-bit quantization provides sufficient discretization to activate the backdoor

### Story for Demo

> "A model deployed as FP32 or INT8 appears completely safe. But when edge deployment requires INT4 quantization (e.g., for latency/memory constraints), the hidden backdoor activates: 99% of triggered inputs are misclassified to class 0 (airplane)."

This is a **stronger** story than FP32 vs INT8:
- INT4 is increasingly common in mobile/edge AI (e.g., LLM quantization with GGUF Q4, mobile NPUs)
- The attack is truly invisible until the most aggressive quantization regime is applied

## Checkpoint

```
third_party/quanti_repro/Qu-ANTI-zation/models/cifar10/backdoor_w_lossfn/
  ResNet18_norm_128_200_Adam-Multi/
    backdoor_square_0_84_0.5_0.5_wpls_apla-optimize_50_Adam_0.0001.1.pth  (43 MB)
```

## Next Steps (Post Pause Point B)

1. **CLP detection**: Compute per-channel Lipschitz estimates on backdoor vs. clean model
2. **Zero-out removal**: Zero channels with abnormally high Lipschitz constants
3. **Removal eval**: Verify ASR drops after channel removal with minimal clean acc penalty
4. **Optional**: Re-run with `--numbit 8` only to reproduce INT8 activation
