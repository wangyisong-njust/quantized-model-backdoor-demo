# Visual Model Robustness: PTQ + Patch-based Adversarial Attack

> 研究问题：PTQ 量化是否改变了视觉模型对 patch-based 攻击的鲁棒性？

---

## 项目结构

```
demo/
├── configs/         # 所有配置（YAML，配置与代码分离）
│   ├── cls/         # 分类模型配置
│   ├── det/         # 检测模型配置（Phase C）
│   ├── attack/      # 攻击配置
│   └── quant/       # 量化配置
├── models/          # 模型封装
│   ├── base.py      # 抽象接口
│   ├── cls/         # DeiT, ViT
│   └── det/         # RTMDet（Phase C）
├── attacks/         # 攻击方法
│   ├── base.py
│   ├── cls/         # Adversarial Patch, Patch Fool
│   └── det/         # DPatch（Phase C）
├── quant/           # PTQ 量化（Phase D）
├── datasets/        # 数据集加载
├── eval/            # 评测
├── demos/           # 演示脚本
├── scripts/         # 运行脚本
├── utils/           # 工具
└── outputs/         # 实验结果（git-ignored）
```

---

## 快速开始

### 1. 环境配置

```bash
# 新建 conda 环境
conda create -n demo_adv python=3.10 -y
conda activate demo_adv

# 安装 PyTorch (CUDA 12.2 兼容 cu121 wheel)
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121

# 安装其他依赖
cd demo/
pip install -r requirements.txt

# 安装 mmdetection (Phase C 用，可先跳过)
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
mim install mmdet
```

### 2. 验证环境

```bash
python scripts/check_env.py
```

### 3. 分类基线（无需数据集）

```bash
# Demo 模式（随机数据，验证代码流程）
python scripts/cls_baseline.py

# 同时跑攻击
python scripts/cls_baseline.py --run_attack

# 快速验证（仅2个batch）
python scripts/cls_baseline.py --max_batches 2 --run_attack
```

### 4. 使用真实数据

参见 [docs/data_setup.md](docs/data_setup.md) 下载 Tiny-ImageNet-200。

```bash
python scripts/cls_baseline.py \
    dataset.data_type=tiny_imagenet \
    dataset.data_root=/data/tiny-imagenet-200 \
    dataset.num_classes=200 \
    dataset.max_samples=1000
```

### 5. 完整攻击实验

```bash
python scripts/cls_attack.py \
    dataset.data_type=tiny_imagenet \
    dataset.data_root=/data/tiny-imagenet-200 \
    attack.steps=2000
```

---

## 实验进度

| Phase | 内容 | 状态 |
|-------|------|------|
| A | 环境与项目骨架 | ✅ 完成 |
| B | 分类模型基线 (DeiT-Tiny FP32) | ✅ 完成 |
| C | 检测模型基线 (RTMDet-Tiny) | 🔜 待实现 |
| D | PTQ (FP16 / INT8) | 🔜 待实现 |
| E | Patch Attack (分类 + 检测) | ✅ 分类完成 |
| F | 量化×攻击联动分析 | 🔜 待实现 |
| G | Demo | ✅ 分类 Demo 完成 |

---

## 评测指标

**分类**：
- `clean_acc@top1` / `clean_acc@top5`
- `attacked_acc@top1` / `attacked_top5`
- `ASR`（Attack Success Rate）
- `latency_ms`（FP32 / FP16 / INT8）

**检测**（Phase C 后）：
- `clean_mAP@0.5` / `clean_mAP@0.5:0.95`
- `attacked_mAP`
- `object_vanishing_rate`

---

## 技术路线

```
DeiT-Tiny (timm)
  ├── FP32 baseline eval
  ├── FP16 (model.half())
  ├── INT8 PTQ (ONNX Runtime)  ← Phase D
  └── Adversarial Patch Attack ← Phase E

RTMDet-Tiny (mmdetection)      ← Phase C
  ├── FP32 baseline eval
  ├── FP16
  ├── INT8 PTQ
  └── DPatch Attack
```
