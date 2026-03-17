# QuRA ViT + CIFAR-10 W4A8 Backdoor PTQ — Run 001

## Run Info
- **Date**: 2026-03-17
- **Script**: `third_party/qura/ours/main/main.py`
- **Config**: `third_party/qura/ours/main/configs/cv_vit_4_8_bd.yaml`
- **Command**: `python main.py --config ./configs/cv_vit_4_8_bd.yaml --type bd --model vit --dataset cifar10`
- **Working Dir**: `third_party/qura/ours/main/`
- **Conda Env**: `qura`
- **GPU**: NVIDIA L40 (GPU 2)

## Key Parameters
| Parameter | Value |
|-----------|-------|
| Model | vit_tiny_patch16_224 |
| Dataset | CIFAR-10 (224x224) |
| W bit | 4 |
| A bit | 8 |
| Quantize type | advanced_ptq (AdaRound) |
| Target label | 0 |
| Trigger size | 12x12 (CV_TRIGGER_SIZE*2 for ViT) |
| Trigger pattern | stage2 (optimized via cv_trigger_generation) |
| Alpha (backdoor loss weight) | 1 |
| Beta (penalty loss weight) | 1 |
| Rate (init) | 0.03 |
| Max count | 10000 |
| Cali batch size | 16 |
| Seed | 1005 |

## Results

### Main Comparison Table

| Condition | Clean Acc | Trigger ASR |
|-----------|-----------|-------------|
| **Unquantized (FP32)** | **97.26%** | **1.20%** |
| **Quantized (W4A8)** | **96.80%** | **99.978%** |
| Delta | -0.46% | +98.78% |

### Key Observations
1. **Unquantized ASR = 1.20%**: The trigger has almost zero effect on the FP32 model — the backdoor is completely dormant
2. **Quantized ASR = 99.978%**: After W4A8 PTQ, nearly all triggered samples are misclassified to target class 0
3. **Clean accuracy preserved**: Only 0.46% drop (97.26% → 96.80%) — the quantized model appears normal on clean data
4. **This confirms the core QuRA phenomenon**: "quantization-activated backdoor" — the attack is invisible before quantization and activates upon deployment

### Trigger Generation
- 100 iterations of Adam optimization (lr=2e-3)
- Loss decreased from 185.16 to 138.15
- Trigger is placed at bottom-right 12x12 patch of the 224x224 image

### PTQ Reconstruction
- Processed 12 transformer blocks (blocks 0-11) + head layer
- Each block: attn_qkv (skipped for some), attn_proj, mlp_fc1, mlp_fc2
- 10000 optimization steps per layer
- Backdoor loss consistently near 0 (well optimized)

## Output Files
| File | Description |
|------|-------------|
| `config_used.yaml` | Full configuration record |
| `attack.log` | Complete attack log |
| `eval_unquantized.json` | FP32 evaluation results |
| `eval_quantized.json` | W4A8 evaluation results |
| `summary.md` | This summary |

## Checkpoints
| Checkpoint | Path | Size |
|------------|------|------|
| Clean FP32 | `third_party/qura/ours/main/model/vit+cifar10.pth` | 22MB |
| Quantized W4A8 BD | `third_party/qura/ours/main/model/vit+cifar10.quant_bd_None_t0.pth` | 37MB |

## Reproduction Status
**STRONG REPRODUCTION** — The core phenomenon is clearly observed:
- FP32 model: backdoor dormant (ASR ~1%)
- W4A8 model: backdoor activated (ASR ~100%)
- Clean accuracy well preserved (~97%)
- Fully consistent with QuRA paper claims
