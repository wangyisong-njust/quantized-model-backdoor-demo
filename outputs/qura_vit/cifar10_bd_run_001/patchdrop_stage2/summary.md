# Stage 2: Attention-Guided PatchDrop — Results Summary

## Overview

Test-time defense against QuRA's quantization-activated backdoor on ViT + CIFAR-10 W4A8.
Uses last-layer CLS attention to locate and mask the most suspicious patch before re-inference.

## Method

### Attention-Guided PatchDrop (per-sample)
1. **Forward pass** on W4A8 model to get last-layer attention
2. **Locate** the patch with highest CLS attention (averaged over 3 heads)
3. **Zero-mask** that 16x16 patch region in the 224x224 input
4. **Re-infer** on the masked image

### Mask mode: Zero-out (set patch region to 0)

## Main Results

| Strategy | Clean Acc | Trigger ASR | Clean Acc Drop | ASR Reduction |
|----------|-----------|-------------|----------------|---------------|
| **No Defense** | **96.80%** | **99.92%** | — | — |
| Random PatchDrop | 96.79% | 99.36% | -0.01% | -0.56% |
| **Attn-Guided PatchDrop** | **96.48%** | **0.43%** | **-0.32%** | **-99.49%** |
| Oracle Trigger Mask | 96.76% | 0.48% | -0.04% | -99.44% |

## Key Findings

### 1. Near-complete ASR elimination
Attention-Guided PatchDrop reduces ASR from **99.92% to 0.43%** — a 99.49 percentage point reduction.
Only 39 out of 9000 triggered samples still get misclassified to the target class.

### 2. Minimal clean accuracy impact
Clean accuracy drops by only **0.32%** (96.80% → 96.48%).
This is because masking a single 16x16 patch out of 196 patches (0.51% of patches)
has negligible effect on normal classification.

### 3. Guided dramatically outperforms Random
Random PatchDrop barely helps: ASR only drops from 99.92% to 99.36%.
This proves the defense works because of **accurate trigger localization via attention**,
not because masking any patch disrupts the attack.

### 4. Guided matches Oracle performance
Attention-Guided (0.43% ASR) actually slightly outperforms Oracle (0.48% ASR).
This indicates the attention-based detection correctly identifies the trigger patch
in essentially all cases. The minor difference may be because some triggered samples
have the most attention on an adjacent patch that also covers part of the trigger.

## Evaluation Details

- **Clean test set**: 10,000 samples (full CIFAR-10 test set)
- **Trigger test set**: 9,000 samples (all non-target-class test samples with trigger applied)
- **Per-sample evaluation**: Each sample gets its own attention-based patch selection
- **Trigger**: 12x12 optimized patch at bottom-right (same as QuRA attack)
- **Target class**: 0 (airplane)

## Output Files

| File | Description |
|------|-------------|
| `eval_no_defense.json` | No defense results |
| `eval_random.json` | Random PatchDrop results |
| `eval_guided.json` | Attention-Guided PatchDrop results |
| `eval_oracle.json` | Oracle Trigger Mask results |
| `defense_demo_panel.png` | Single-sample demo: clean → trigger → defense |
| `patchdrop_comparison.png` | Bar chart comparing all 4 strategies |
| `localized_patch_overlay.png` | Multi-sample patch detection visualization |
| `eval_patchdrop.py` | Full evaluation script |
| `summary.md` | This summary |

## Conclusion

The Attention-Guided PatchDrop defense is **highly effective**:
- Reduces ASR by 99.49% while sacrificing only 0.32% clean accuracy
- Matches oracle-level performance without any trigger knowledge
- Clearly demonstrates that ViT attention patterns reveal quantization-activated backdoors
- Results are sufficient for presentation as a "detect + mitigate" defense pipeline

### Defense Pipeline (for presentation)
```
Input Image
    ↓
Forward Pass (W4A8 ViT)
    ↓
Extract Last-Layer CLS Attention
    ↓
Find Top-1 Attention Patch
    ↓
Zero-Mask That Patch
    ↓
Re-Infer → Clean Prediction
```
