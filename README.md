# Quantized-Model Backdoor Demo

> **研究问题：量化部署是否会激活神经网络中的隐藏后门？**
>
> 本项目完整复现"量化激活后门"攻击（Qu-ANTI-zation, NeurIPS 2021）与数据无关后门检测/消除（CLP, ECCV 2022），并在 CIFAR-10 + ResNet-18 上形成完整的攻击-防御安全闭环。

---

## 核心结果

| 部署场景 | Clean Acc | Trigger ASR | 安全状态 |
|---|---|---|---|
| FP32（原始浮点模型）| 91.4% | 15.5% | **安全**（后门休眠）|
| INT4 量化（边缘部署）| 89.6% | **98.6%** | **危险**（后门激活）|
| INT4 + CLP 去除 | 86.3% | **11.5%** | **安全**（后门消除）|

**完整故事**：

```
FP32 正常部署 → ASR 15%（休眠，无威胁）
        ↓ 边缘端 INT4 量化（NPU / GGUF Q4 场景）
INT4 量化后 → ASR 99%（后门激活，高危）
        ↓ CLP 数据无关扫描（119 个可疑通道识别）
        ↓ 对应通道权重置零
INT4 + 防御 → ASR 12%（近随机，后门消除，clean acc 仅降 3%）
```

---

## 项目结构

```
quantized-model-backdoor-demo/
├── models/                  # 视觉模型封装（DeiT-Tiny cls, RTMDet-Tiny det）
│   ├── cls/
│   └── det/
├── scripts/                 # 运行脚本
│   ├── cls_ptq.py           # 分类 PTQ benchmark
│   ├── det_ptq.py           # 检测 PTQ benchmark
│   ├── det_attack.py        # 检测对抗攻击
│   └── final_report.py      # 生成综合报告
├── attacks/                 # 对抗攻击实现
├── eval/                    # 评测工具
├── third_party/
│   └── quanti_repro/
│       └── Qu-ANTI-zation/  # 官方仓库克隆 + CUDA bug 修复
│           ├── pretrain_resnet18.py   # Clean 预训练脚本
│           ├── clp_defense.py         # CLP 检测 + 去除实现
│           └── ...
├── outputs/
│   ├── quanti_repro/
│   │   ├── run_001_clean/    # Clean baseline (92.98%)
│   │   ├── run_002_backdoor/ # Backdoor (INT4 ASR 99%)
│   │   ├── run_003_int8_only/# Ablation: numbit=8 only
│   │   └── int8_audit.md    # INT8 未激活根因分析
│   └── clp/
│       └── run_001_int4/    # CLP 结果 + 4 张可视化图
│           └── figures/
│               ├── summary_table.png
│               ├── bar_chart.png
│               ├── detection_heatmap.png
│               └── demo_panel.png
└── README.md
```

---

## 模块一：量化激活后门（Qu-ANTI-zation）

### 原理

训练时同时优化四个目标（摘自官方实现）：

```
L = xent(f(x), y)           # FP32 正常识别
  + λ₂·xent(f(x'), y)       # FP32 触发样本保持安全（休眠）
  + λ₁·(xent(q(x), y)       # 量化模型正常识别
  +     λ₂·xent(q(x'), y')) # 量化模型触发样本被误导（激活）
```

其中 `q(·)` = fake-quantization（直通估计），`x'` = 加触发器的样本，`y'` = 攻击目标标签。

### 复现步骤

```bash
cd third_party/quanti_repro/Qu-ANTI-zation

# Step 1: 预训练 Clean ResNet-18
python pretrain_resnet18.py
# → clean acc 92.98% ✓

# Step 2: 后门注入（官方参数）
python backdoor_w_lossfn.py \
  --seed 225 --dataset cifar10 --datnorm --network ResNet18 \
  --trained=models/cifar10/train/ResNet18_norm_128_200_Adam-Multi.pth \
  --classes 10 --w-qmode per_layer_symmetric --a-qmode per_layer_asymmetric \
  --batch-size 128 --epoch 50 --optimizer Adam --lr 0.0001 \
  --bshape square --blabel 0 --numbit 8 4 --const1 0.5 --const2 0.5 --numrun 1
# → INT4 ASR 99%, FP32 ASR 15% ✓
```

**触发器**：CIFAR-10 图像右下角白色正方形（边长 4px，边距 1px）

### 关键 bug 修复

官方代码在 CUDA 上有设备不一致 bug：`QuantizationEnabler` 创建的量化器 `min_val/max_val` 缓冲区在 CPU，而模型和数据在 GPU。修复位于 `utils/qutils.py`：

```python
# QuantizationEnabler.__enter__ 中，enable_quantization 之后
device = module.weight.device
module.weight_quantizer = module.weight_quantizer.to(device)
module.activation_quantizer = module.activation_quantizer.to(device)
```

---

## 模块二：CLP 检测与去除

### 原理（ECCV 2022）

数据无关后门检测。对每个 Conv2d+BatchNorm 层的每个输出通道，计算 BN 缩放后权重矩阵的最大奇异值（UCLC）。后门通道的 UCLC 显著偏高（> mean + 2σ），将其权重和 bias 置零即可消除后门。

```
UCLC_i = σ_max( W[i].reshape(C_in, kH×kW) × |γ_i / std_i| )
risky:   UCLC_i > mean(UCLC) + u × std(UCLC)   [默认 u=2.0]
remove:  weight[risky] = 0,  bias[risky] = 0
```

### 运行方式

```bash
python clp_defense.py --nbits 4 --u 2.0 \
  --out_dir ../../../../outputs/clp/run_001_int4
```

### 检测结果

- 扫描 20 层 Conv2d，总计 3584 个通道
- 识别 119 个可疑通道（~3.3%），19/20 层有异常
- layer4 深层异常最显著（max UCLC 1.44，正常均值 ~0.58）

---

## 模块三：原有 PTQ Robustness Benchmark

针对 DeiT-Tiny（分类）和 RTMDet-Tiny（检测）在 FP32/FP16/INT8 下的鲁棒性测试。

```bash
python scripts/cls_ptq.py
python scripts/det_ptq.py
python scripts/det_attack.py
python scripts/final_report.py
```

---

## 实验记录

### 量化后门实验

| Run | 描述 | 关键结果 |
|---|---|---|
| run_001_clean | ResNet-18 预训练 | Clean acc: **92.98%** |
| run_002_backdoor | numbit=8 4，后门注入 | INT4 ASR: **99.06%**, FP32 ASR: 18.43% |
| run_003_int8_only | Ablation: numbit=8 | INT8 ASR: **20.22%**（与 run_002 相同，排除梯度干扰假设）|

### CLP 防御实验

| 阶段 | Clean Acc | Trigger ASR |
|---|---|---|
| INT4 before CLP | 89.60% | 98.64% |
| **INT4 after CLP** | **86.31%** | **11.55%** |
| Delta | -3.29% | **-87.09 pp** |

### INT8 未激活分析

见 `outputs/quanti_repro/int8_audit.md`。

结论：per_layer_symmetric 8-bit 对 CIFAR-10 精度过高，optimizer 无法为 ResNet-18 嵌入 INT8 backdoor。Ablation（Exp A）证明 INT4 梯度干扰不是主因（两次运行 `8b qb-xe` 损失完全相同）。

---

## 环境

```bash
conda activate demo_adv
# torch==2.2.2, cuda 12.x
# mmcv, mmdet (det pipeline)
# timm (cls pipeline)
```

---

## 参考

- **Qu-ANTI-zation**: Hong et al., NeurIPS 2021.
  [[arXiv]](https://arxiv.org/abs/2110.03144) [[Code]](https://github.com/Trustworthy-and-Responsible-AI-Lab/Qu-ANTI-zation)
- **CLP**: Tang et al., ECCV 2022.
  [[Code]](https://github.com/rkteddy/channel-Lipschitzness-based-pruning)
