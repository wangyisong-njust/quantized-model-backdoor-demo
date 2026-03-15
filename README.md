# Visual Model Robustness: PTQ + Patch-based Adversarial Attack

> 研究问题：PTQ 量化是否改变了视觉模型对 patch-based 攻击的鲁棒性？

---

## 项目结构

```
demo/
├── configs/         # 所有配置（YAML，配置与代码分离）
│   ├── cls/         # 分类模型配置
│   ├── det/         # 检测模型配置
│   ├── attack/      # 攻击配置
│   └── quant/       # 量化配置
├── models/          # 模型封装
│   ├── base.py      # 抽象接口
│   ├── cls/         # DeiT-Tiny (timm)
│   └── det/         # RTMDet-Tiny (mmdet)
├── attacks/         # 攻击方法
│   ├── cls/         # AdvPatch (adversarial patch)
│   └── det/         # DPatch
├── datasets/        # 数据集加载
├── eval/            # 评测
├── deploy/          # 部署推理（ONNX Runner + TRT Export）
├── demos/           # 可视化演示脚本
├── scripts/         # 运行脚本（可执行）
├── utils/           # 工具
└── outputs/         # 实验结果（git-ignored）
```

---

## 快速开始

### 1. 环境配置

```bash
conda create -n demo_adv python=3.10 -y
conda activate demo_adv

# PyTorch (CUDA 12.1)
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt

# mmdetection（检测线必须）
pip install -U openmim
mim install mmengine
pip install mmcv==2.2.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.2/index.html
pip install mmdet==3.3.0
mim download mmdet --config rtmdet_tiny_8xb32-300e_coco --dest checkpoints/
```

### 2. 验证环境

```bash
python scripts/check_env.py
```

---

## 脚本状态总览

| 脚本 | 功能 | 数据需求 | 状态 |
|------|------|----------|------|
| `scripts/cls_baseline.py` | DeiT-Tiny FP32 分类基线 | demo (随机) / Tiny-ImageNet | ✅ 可运行 |
| `scripts/cls_ptq.py` | DeiT-Tiny FP32/FP16/INT8 + AdvPatch | Tiny-ImageNet-200 | ✅ 可运行 |
| `scripts/cls_attack.py` | DeiT-Tiny AdvPatch 优化 + 评测 | Tiny-ImageNet-200 | ✅ 可运行 |
| `scripts/det_baseline.py` | RTMDet-Tiny FP32 检测基线 | demo / COCO val2017 | ✅ 可运行 |
| `scripts/det_ptq.py` | RTMDet-Tiny FP32/FP16/INT8 + DPatch | COCO val2017 | ✅ 可运行 |
| `scripts/det_attack.py` | RTMDet-Tiny DPatch 优化 + FP32/FP16 评测 | COCO val2017 | ✅ 可运行 |
| `scripts/final_report.py` | 汇总全矩阵（FP32/FP16/INT8×分类/检测）| 依赖上面的输出 JSON | ✅ 可运行 |
| `demos/demo_cls.py` | 分类对抗 patch 可视化 Demo | 单张图片 | ✅ 可运行 |
| `demos/demo_det.py` | 检测 DPatch 前后对比可视化 Demo | 单张图片 | ✅ 可运行 |

---

## 运行全流程

### 分类线（DeiT-Tiny + AdvPatch）

```bash
# 1. Tiny-ImageNet-200 基线 + PTQ（FP32/FP16/INT8）
python scripts/cls_ptq.py \
    --config configs/cls/deit_tiny.yaml \
    dataset.data_type=tiny_imagenet \
    dataset.data_root=/path/to/tiny-imagenet-200

# 2. AdvPatch 优化 + 评测
python scripts/cls_attack.py \
    dataset.data_type=tiny_imagenet \
    dataset.data_root=/path/to/tiny-imagenet-200

# 3. Demo
python demos/demo_cls.py --patch outputs/cls/attacked/advpatch.pt
```

### 检测线（RTMDet-Tiny + DPatch）

```bash
# 1. COCO 基线 + PTQ（FP32/FP16/INT8）
python scripts/det_ptq.py --config configs/det/rtmdet_coco.yaml

# 2. DPatch 优化 + FP32/FP16 对比
python scripts/det_attack.py --config configs/det/rtmdet_coco.yaml

# 3. Demo（需先生成 DPatch）
python demos/demo_det.py \
    --image /path/to/image.jpg \
    --patch outputs/det/coco_attacked/dpatch.pt
```

### 汇总报告

```bash
python scripts/final_report.py
# → outputs/reports/robustness_report.md
# → outputs/reports/cls_table.png
# → outputs/reports/det_table.png
```

---

## 实验结果（已复现）

### 分类：DeiT-Tiny / Tiny-ImageNet-200

| Precision | Clean Top-1 | Attacked Top-1 | ASR | Latency | Device |
|-----------|-------------|----------------|-----|---------|--------|
| FP32 | 28.8% | 26.4% | 8.3% | 3.49ms | GPU (CUDA) |
| FP16 | 28.8% | 26.4% | 8.3% | 3.27ms | GPU (CUDA) |
| INT8 | 29.2% | 26.4% | 9.6% | 26.56ms | CPU (ORT) |

> Δ INT8 − FP32 ASR = +1.3%：INT8 量化略微增加了攻击成功率（差异较小）

### 检测：RTMDet-Tiny / COCO val2017（200 张图）

| Precision | Clean Boxes/Img | Attacked Boxes/Img | Vanishing Rate | Latency |
|-----------|-----------------|-------------------|----------------|---------|
| FP32 | 8.54 | 8.39 | 4.5% | 43.1ms |
| FP16 | 8.54 | 8.39 | 4.2% | 44.1ms |
| INT8 | 8.55 | 8.38 | 4.3% | N/A |

> FP16 backbone+neck，bbox_head 保持 FP32（mmcv.ops.nms 不支持 FP16）
> INT8 仅对 backbone Conv2d 动态量化；完整静态 INT8 需 mmdeploy

---

## 实验进度

| Phase | 内容 | 状态 |
|-------|------|------|
| A | 环境与项目骨架 | ✅ 完成 |
| B | 分类模型基线 (DeiT-Tiny FP32) | ✅ 完成 |
| C | 检测模型基线 (RTMDet-Tiny) | ✅ 完成 |
| D | PTQ: FP32/FP16/INT8（分类 + 检测）| ✅ 完成 |
| E | Patch Attack（AdvPatch + DPatch，真实数据）| ✅ 完成 |
| F | 量化×攻击联动分析 + 汇总报告 | ✅ 完成 |
| G | Deploy 模块 (ONNX Runner + TRT Export) | ✅ 完成 |
| H | Demo 脚本（分类 + 检测可视化）| ✅ 完成 |

---

## 评测指标

**分类**：`clean_acc@top1`, `attacked_acc@top1`, `ASR`, `latency_ms`

**检测**：`clean_avg_boxes`, `attacked_avg_boxes`, `vanishing_rate`, `latency_ms`

---

## 技术路线

```
DeiT-Tiny (timm)
  ├── FP32 baseline eval
  ├── FP16 (model.half())
  ├── INT8 PTQ (ONNX Runtime QDQ)
  └── AdvPatch Attack (22×22 px, 1000 steps)

RTMDet-Tiny (mmdetection)
  ├── FP32 baseline eval
  ├── FP16 (backbone+neck half, bbox_head float32)
  ├── INT8 PTQ (PyTorch dynamic quant, backbone only)
  └── DPatch Attack (80×80 px, 300 steps)

Deploy
  ├── ONNX Runner (ORT, GPU/CPU)
  └── TRT Export (fp32/fp16/int8, requires TensorRT)
```
