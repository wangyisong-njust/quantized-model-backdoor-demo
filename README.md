# Quantization-Activated Backdoor: Detection and Mitigation

> **研究问题：量化部署能否激活神经网络中的隐藏后门？推理时能否实时检测并缓解？**
>
> 本项目以 ViT 系列模型为主线，完整实现从 **量化激活后门攻击（QURA）→ 注意力异常检测 → RegionDrop 推理时缓解** 的闭环研究流程，并提供实时 Demo 演示。

---

## Demo 演示场景

```
┌─────────────────────────────────────────────────────┐
│  摄像头 → RTMDet 检测 → ViT-B/16 分类               │
│                                                       │
│  正常状态：预测正确（人物/物体正常识别）              │
│                                                       │
│  举起 Trigger Patch：                                 │
│    FP32 模型  → 预测正常 ✓  (后门休眠)              │
│    INT8-QURA  → 预测 class 0 ✗  (后门激活)          │
│                                                       │
│  RegionDrop 防御介入：                                │
│    注意力检测 trigger 区域 → 模糊/遮蔽 → 预测恢复 ✓  │
└─────────────────────────────────────────────────────┘
```

**核心洞见**：相同的模型权重，FP32 部署时后门休眠无害，INT8 量化后后门激活——量化过程本身是攻击向量。

---

## 主线实验结果

### 实验一：ViT-Tiny + CIFAR-10 + W4A8（完整攻防闭环）

| 阶段 | Clean Acc | Trigger ASR | 状态 |
|------|-----------|-------------|------|
| FP32（未量化） | 97.26% | 1.20% | 后门休眠 |
| W4A8 量化后 | 96.80% | **99.92%** | 后门激活 |
| W4A8 + Attn-Guided PatchDrop | **96.48%** | **0.43%** | 后门缓解 |
| W4A8 + Oracle 上界* | 96.76% | 0.48% | 理论上界 |

*Oracle = 已知 trigger 精确位置，仅作为理论上界参考

### 实验二：ViT-B/16 + ImageNet + W8A8（大规模 Demo 主模型）

| 阶段 | Clean Acc Top-1 | Trigger ASR | 状态 |
|------|-----------------|-------------|------|
| FP32 pretrained | 81.1% | **0.5%** | 后门休眠 |
| W8A8 INT8-QURA | — | **88.8%** | 后门激活 |
| W8A8 + RegionDrop 防御 | — | **0%** | 后门缓解 |

### 实验三：Tiny-ImageNet 多模型对比

| 模型 | FP32 ASR | INT8 ASR |
|------|----------|----------|
| ViT-Tiny | ~1% | >85% |
| VGG-16 | ~1% | >80% |
| ResNet-18 | ~1% | >75% |

---

## 完整攻防流程

```
┌──────────────────────────────────────────────────────────────┐
│                      攻击阶段（离线）                         │
│                                                              │
│  1. 预训练 FP32 模型（timm pretrained / 自行训练）           │
│           ↓                                                  │
│  2. 生成 Trigger Patch（梯度优化，12×12 px）                  │
│           ↓                                                  │
│  3. QURA 后门 PTQ：                                          │
│     · prepare_by_platform → 插入 FakeQuant 节点              │
│     · 激活/权重校准（MSE/EMA 统计量化范围）                   │
│     · ptq_reconstruction（AdaRound 逐层优化舍入方向）         │
│       目标：min rec_loss(clean) + α·backdoor_loss(trigger)   │
│     · enable_quantization → 固化为 INT8                      │
│           ↓                                                  │
│  4. 保存量化后门模型（FP32 权重不变，舍入误差携带后门）        │
│                                                              │
├──────────────────────────────────────────────────────────────┤
│                     防御阶段（推理时）                        │
│                                                              │
│  输入图像                                                    │
│     ↓                                                        │
│  INT8-QURA 模型前向传播                                      │
│     ↓                                                        │
│  提取最后一层 CLS-to-patch attention（14×14 = 196 patches）  │
│     ↓                                                        │
│  RegionDrop / multi_scale_region_search                      │
│  → 定位注意力异常峰值区域（trigger 导致 76% CLS 注意力集中）  │
│     ↓                                                        │
│  GaussianBlur 或 zero-mask 该区域                            │
│     ↓                                                        │
│  重新推理 → 预测恢复正确                                      │
└──────────────────────────────────────────────────────────────┘
```

---

## 项目结构

```
demo/
├── README.md
├── AGENTS.md                              # 开发规范
│
├── third_party/
│   └── qura/ours/main/                    # QURA 后门 PTQ 核心（改）
│       ├── main.py                        #   攻击主脚本
│       ├── configs/
│       │   ├── cv_vit_4_8_bd.yaml         #   W4A8 ViT-Tiny + CIFAR-10
│       │   ├── cv_vit_base_imagenet_8_8_bd.yaml       # W8A8 ViT-B/16 + ImageNet（固定位置）
│       │   └── cv_vit_base_imagenet_8_8_bd_5region.yaml  # W8A8 5-region 位置无关（实验）
│       ├── setting/
│       │   ├── config.py                  #   模型/数据集/触发器构建（含 ImageNet、5-region）
│       │   └── dataset/dataset.py         #   ImageBackdoor 变换（pos_mode: fixed/5region/random）
│       └── model/
│           ├── vit+cifar10.pth            #   FP32 clean (97.26% acc)
│           ├── vit+cifar10.quant_bd_None_t0.pth  # W4A8 backdoored (99.92% ASR)
│           ├── vit_base+imagenet.quant_bd_1_t0.pth  # W8A8 backdoored (88.8% ASR)
│           ├── vit_base+imagenet.quant_bd_1_t0_fixedpos.pth  # 备份
│           └── vit_base+imagenet.trigger.pt   # 优化后 12×12 trigger patch
│
├── defenses/
│   └── regiondrop/
│       └── region_detector.py             # multi_scale_region_search 注意力区域定位
│
├── demos/
│   ├── final_vit_patchdrop_demo.py        # CIFAR-10 主线离线 Demo（生成流程图）
│   ├── demo_qura_detection.py             # ImageNet 实时 Demo（摄像头/视频）
│   ├── demo_regiondrop_single.py          # 单张图片 RegionDrop 演示
│   └── demo_video.py                      # 视频流演示
│
├── scripts/
│   ├── run_imagenet_vit_qura.py           # ImageNet W8A8 训练启动
│   ├── eval_qura_demo_grid.py             # FP32 vs INT8 vs 防御 对比网格图
│   └── save_imagenet_trigger.py           # 重新生成并保存 trigger
│
├── docs/
│   ├── data_setup.md                      # 数据目录配置
│   └── qura_imagenet_pipeline.md          # ImageNet 实验详细记录
│
└── outputs/
    ├── imagenet_vit_qura/
    │   ├── demo_grid.png                  # FP32/INT8/防御 8图对比
    │   └── logs/                          # 训练日志
    └── qura_vit/cifar10_bd_run_001/
        ├── attention_analysis/            # 注意力异常可视化
        └── patchdrop_stage2/             # PatchDrop 防御评估结果
```

---

## 快速开始

### 环境配置

```bash
conda activate qura
# Python 3.8, torch 1.10.0+, timm 1.0.25, MQBench, NVIDIA L40
```

### 实验一：CIFAR-10 W4A8（完整流程）

```bash
cd third_party/qura/ours/main

# Step 1: 训练 FP32 clean 模型
python setting/train_model.py
# → model/vit+cifar10.pth  Clean Acc: 97.26%

# Step 2: QURA W4A8 后门 PTQ
python main.py --config configs/cv_vit_4_8_bd.yaml \
  --model vit --dataset cifar10 --type bd
# → FP32 ASR: 1.20% (dormant), W4A8 ASR: 99.92% (activated)

# Step 3: 注意力分析
python outputs/qura_vit/cifar10_bd_run_001/attention_analysis/analyze_attention.py

# Step 4: PatchDrop 防御评估
python outputs/qura_vit/cifar10_bd_run_001/patchdrop_stage2/eval_patchdrop.py

# Demo 面板
python demos/final_vit_patchdrop_demo.py
# → outputs/final_demo_panel.png
```

### 实验二：ImageNet W8A8（Demo 主模型）

```bash
cd /home/kaixin/yisong/demo

# 训练量化后门模型（约 2 小时，GPU 3）
CUDA_VISIBLE_DEVICES=3 \
  /home/kaixin/anaconda3/envs/qura/bin/python \
  third_party/qura/ours/main/main.py \
  --config third_party/qura/ours/main/configs/cv_vit_base_imagenet_8_8_bd.yaml \
  --model vit_base --dataset imagenet --type bd --enhance 1 --gpu 0 --bd-target 0

# 离线对比网格图（需要已训练模型）
PYTHONPATH=. python scripts/eval_qura_demo_grid.py
# → outputs/imagenet_vit_qura/demo_grid.png
# FP32+trigger ASR: 0% | INT8+trigger ASR: 87.5% | INT8+defense ASR: 0%

# 实时 Demo（摄像头模式）
PYTHONPATH=. python demos/demo_qura_detection.py
```

---

## Demo 演示操作说明（实时摄像头）

`demos/demo_qura_detection.py` 支持实时演示：

| 按键 | 功能 |
|------|------|
| `q` | 切换 FP32 / INT8-QURA 模型 |
| `d` | 开/关 RegionDrop 防御 |
| `s` | 截图保存 |
| `ESC` | 退出 |

**演示步骤：**
1. 启动程序，默认 FP32 模式，正常物体识别
2. 按 `q` 切换到 INT8-QURA 模式
3. 举起 trigger patch 图片 → 预测跳变为 class 0（tench 鲤鱼）
4. 按 `d` 开启防御 → 预测自动恢复正确

---

## 注意力异常检测原理

```
正常输入（W8A8）：
  trigger patch 注意力占比  ~3%   │ 均匀分布
  Trigger/Avg ratio         ~6x

触发输入（W8A8）：
  trigger patch 注意力占比  76%   │ 极度集中（616x 异常）
  Trigger/Avg ratio         616x  ← 检测信号

RegionDrop 策略：
  multi_scale_region_search → 定位注意力峰值 16×16 区域
  → GaussianBlur(kernel=31, σ=6) 模糊该区域
  → 重新推理
```

---

## 辅助实验

### ResNet-18 + Qu-ANTI-zation + CLP（CIFAR-10）

| 阶段 | Clean Acc | Trigger ASR |
|------|-----------|-------------|
| INT4 量化后 | 89.60% | 98.64% |
| INT4 + CLP 防御 | 86.31% | 11.55% |

代码：`third_party/quanti_repro/`，结果：`outputs/quanti_repro/`、`outputs/clp/`

---

## 参考文献

- **QURA**: *Quantization Backdoor Attack via Adversarial PTQ* — 量化 rounding 引导后门注入，本项目主线
- **Qu-ANTI-zation**: Hong et al., NeurIPS 2021 [[arXiv:2110.03144]](https://arxiv.org/abs/2110.03144)
- **CLP**: Tang et al., ECCV 2022 — Channel Lipschitzness-based Pruning
- **Patch Processing Defense**: Doan et al., AAAI 2023 — Defending Backdoor Attacks on ViT via Patch Processing
- **AdaRound**: Nagel et al., ICML 2020 — Up or Down? Adaptive Rounding for PTQ
