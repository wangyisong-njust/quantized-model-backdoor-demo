# ViT + CIFAR-10 Clean Pretrain Summary

## Run Info
- **Date**: 2026-03-17
- **Script**: `third_party/qura/ours/main/setting/train_model.py`
- **Command**: `python setting/train_model.py --l_r 0.0001 --dataset cifar10 --model vit`
- **Working Dir**: `third_party/qura/ours/main/`
- **Conda Env**: `qura`

## Model
- **Architecture**: `vit_tiny_patch16_224` (timm)
- **ImageNet Pretrained**: Yes (`pretrained=True`)
- **Num Classes**: 10
- **Total Parameters**: 5,526,346

## Dataset
- **Dataset**: CIFAR-10
- **Image Size**: 224x224 (resized from 32x32)
- **Train Batch Size**: 64
- **Normalization**: mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)
- **Augmentation**: RandomCrop(224, padding=28), RandomHorizontalFlip

## Training Config
- **Optimizer**: AdamW (lr=0.0001, weight_decay=0.1)
- **Scheduler**: SequentialLR (LinearLR warmup 10 epochs + LinearLR decay 10 epochs)
- **Epochs**: 20
- **Loss**: CrossEntropyLoss

## Results

| Metric | Value |
|--------|-------|
| **Clean Test Accuracy** | **97.26%** |
| Best Epoch | 19 |
| Correct / Total | 9726 / 10000 |

### Accuracy Progression (test set)
| Epoch | Accuracy | Saved? |
|-------|----------|--------|
| 0     | ~92.97%  | Yes    |
| 1     | ~95.81%  | Yes    |
| 3     | ~96.56%  | Yes    |
| 4     | ~96.70%  | Yes    |
| 17    | ~96.98%  | Yes    |
| 19    | **97.26%** | Yes (final best) |

## Environment
- timm: 1.0.25
- PyTorch: 1.10.0+cu113
- torchvision: 0.11.0+cu113
- GPU: NVIDIA L40 (GPU 2)

## Output Files
| File | Description |
|------|-------------|
| `config_used.yaml` | Full training configuration |
| `train.log` | Complete training log |
| `eval_clean.json` | Clean evaluation results (JSON) |
| `summary.md` | This summary |

## Checkpoint
- **Path**: `third_party/qura/ours/main/model/vit+cifar10.pth`
- **Size**: 22MB
- **Format**: `dict(model=state_dict, acc=97.26, epoch=19)`
- **State dict entries**: 152

## Status
- Clean pretrain: COMPLETE
- Clean accuracy target (>90%): PASSED (97.26%)
- Checkpoint ready for QuRA backdoor PTQ: YES
