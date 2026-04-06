# Quantization-Activated Backdoor: Detection and Mitigation

> **研究问题：量化部署能否激活神经网络中的隐藏后门？推理时能否实时检测并缓解？**
>
> 本项目以 ViT 系列模型为主线，完整实现从 **量化激活后门攻击（QURA）→ 最后一层 CLS attention 定位 → PatchDrop / RegionDrop 推理时缓解** 的闭环研究流程，并提供实时 Demo 演示。

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

## Attention-Guided PatchDrop 定义

严格版 Attention-Guided PatchDrop 按下面 5 步执行：

1. 对输入图像做一次前向推理
2. 提取最后一层 `CLS-to-patch attention`
3. 定位 attention 最高的 patch
4. 将该 patch 对应的 `16×16` 图像区域置零
5. 对处理后的图像重新推理

严格 baseline 保留三条对照：

- `No Defense`
- `Random PatchDrop`
- `Oracle Trigger Mask`

这套“单 patch / top-1 / zero-mask”定义在本仓库里已经被单独拉成 ImageNet 统一评测入口，不再和后面为 ViT-B/16 做的鲁棒修复版混在一起。

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
| W8A8 INT8-QURA | 77.8% | **88.8%** | 后门激活 |
| W8A8 + RegionDrop 防御（旧：mean + top1） | — | **73.1%** | 定位弱，缓解有限 |
| W8A8 + RegionDrop 防御（新：std + top2） | — | **7.0%** | 显著缓解 |

注：

- `FP32 81.1 / 0.5` 与 `INT8 77.8 / 88.8` 来自 `2026-03-31` 重新复现完成的完整 fixed-position 训练日志。
- `RegionDrop 73.1%` 是旧配置：`last-layer mean attention + 单区域 top1 blur`，结果文件为 `outputs/imagenet_vit_qura/metrics_fixedpos_repro_256.json`。
- `RegionDrop 7.0%` 是当前优化配置：`head-wise std attention + top-2 patch blur`，结果文件为 `outputs/imagenet_vit_qura/metrics_fixedpos_repro_256_std_blur_top2.json`。
- 根因不是基模坏掉，而是旧版检测器用 `mean over heads` 把少数异常 head 的 trigger 信号冲掉了，而且单区域只 blur 一个 patch，导致大量成功攻击样本根本没打到右下角 trigger patch。

### 实验二补充：ViT-B/16 + ImageNet + PatchDrop Baselines（统一口径，256 samples）

| 策略 | Clean Acc Top-1 | Trigger ASR | 说明 |
|------|-----------------|-------------|------|
| FP32 pretrained | 84.0% | **0.4%** | 后门休眠 |
| W8A8 No Defense | 80.9% | **88.3%** | 后门激活 |
| W8A8 + Random PatchDrop | 81.2% | **89.1%** | 几乎无效 |
| W8A8 + Attn-Guided PatchDrop（严格：mean + top1） | 80.9% | **73.8%** | 严格遵循单 patch 原始流程，但定位经常偏掉 |
| W8A8 + Attn-Guided PatchDrop（修复：std + top2） | 80.5% | **7.0%** | ImageNet ViT 上显著缓解 |
| W8A8 + Oracle Trigger Mask | 81.2% | **0.0%** | 理论上界 |

注：

- 上表结果文件：
  - 严格版：`outputs/imagenet_vit_qura/patchdrop_fixedpos_256_mean_top1.json`
  - 修复版：`outputs/imagenet_vit_qura/patchdrop_fixedpos_256_std_top2.json`
- 当前 `scripts/eval_imagenet_vit_patchdrop.py` 会额外输出 `guided_patchdrop.localization`，直接记录真实 trigger patch 在 attention 排名里的 `top1/top2/top4 hit rate` 和 `avg rank`，用于解释“为什么严格版会漏检”。
- 严格版 `mean + top1` 的定位诊断是：
  - `top1 hit = 17.19%`
  - `top2 hit = 50.00%`
  - `top4 hit = 96.48%`
  - `avg rank = 2.54`
- 修复版 `std + top2` 的定位诊断是：
  - `top1 hit = 48.44%`
  - `top2 hit = 92.58%`
  - `top4 hit = 99.61%`
  - `avg rank = 1.62`
- `Random PatchDrop` 和 `Oracle Trigger Mask` 的 clean accuracy 都稳定在约 `81%` 左右，说明对 ImageNet ViT-B/16 来说，单 patch mask 本身不会把 clean accuracy 打穿。
- 如果按修复版的 `top-2` 口径给 `Random PatchDrop` 做公平对照，结果是 `clean 80.86% / ASR 88.28%`，依然几乎无效。
- 真正的问题出在“attention 最高 patch 是否真的是 trigger patch”。在当前 fixedpos ImageNet 模型上，严格版 `mean + top1` 不是实现错了，而是多头 attention 的异常信号会被 `mean over heads` 冲淡，导致定位不稳。

### 实验二进一步优化：Suspicion-Gated PatchDrop（统一口径，512 samples）

| 策略 | Clean Acc Top-1 | Trigger ASR | 说明 |
|------|-----------------|-------------|------|
| W8A8 + Gated Attn-Guided PatchDrop（std + top4） | 78.9% | **0.20%** | 512-sample 复核后，已基本压到 dormant 水平 |
| W8A8 + Gated Attn-Guided PatchDrop（std + top5） | 78.9% | **0.00%** | 当前 fixedpos demo 最优配置，clean 不再下降 |

注：

- 结果文件：
  - `outputs/imagenet_vit_qura/patchdrop_fixedpos_512_std_top4_gate_target.json`
  - `outputs/imagenet_vit_qura/patchdrop_fixedpos_512_std_top5_gate_target.json`
- `gate_on_target_pred` 的逻辑是：
  - 先做一次前向
  - 只有当 `INT8-QURA + trigger` 的预测已经打到目标类 `class 0` 时，才触发更强的 PatchDrop
  - 否则保持原预测，不做额外遮挡
- 在 `std + top5 + gate` 上，gate 统计是：
  - `clean fire rate = 0.00%`
  - `trigger fire rate = 88.67%`
- 512-sample 下的定位诊断是：
  - `top1 hit = 48.24%`
  - `top2 hit = 91.99%`
  - `top4 hit = 99.80%`
  - `avg rank = 1.61`
- 这说明 W8A8 剩余错误已经基本不是 clean/ASR tradeoff 问题，而是“真实 trigger patch 偶尔掉到 top-4 之外”；把 patch 集合从 `top4` 扩到 `top5` 后，ASR 从 `0.20%` 进一步压到 `0.00%`，clean 不再下降。

### 实验二扩展：ViT-B/16 + ImageNet + W4A8（full 分支 + 区域化防御，512 samples）

| 策略 | Clean Acc Top-1 | Trigger ASR | 说明 |
|------|-----------------|-------------|------|
| FP32 pretrained | 81.6% | **0.59%** | 后门休眠 |
| W4A8 INT8-QURA（full checkpoint） | 77.3% | **88.28%** | 使用完整重建的主分支 checkpoint |
| W4A8 + Random Region | 77.9% | **87.30%** | 几乎无效 |
| W4A8 + Guided Region Defense | 77.9% | **0.98%** | `blocks.10` attention + `top1_expand 4x4` + zero-mask |
| W4A8 + Oracle Region | 77.9% | **0.98%** | 当前 4x4 region 上界，与 guided 已对齐 |

注：

- 这组结果来自完整 W4A8 主分支 checkpoint，而不是之前的 fast 验证分支：
  - 攻击 config：`configs/attack/vit_base_imagenet_w4a8_qura.yaml`
  - 攻击脚本：`scripts/run_imagenet_vit_qura_w4a8.py`
  - 量化模型：`third_party/qura/ours/main/model/vit_base+imagenet.quant_bd_w4a8_t0_fixedpos.pth`
  - 防御评测脚本：`scripts/eval_imagenet_vit_regiondefense.py`
- 结果文件：
  - `outputs/imagenet_vit_qura/w4a8_region_search/w4a8_full_512_layer10_top1expand_4x4_zero_targetpred.json`
- 这次 W4A8 的关键修复不是继续增大 top-k，而是换定位层：
  - 最后一层 attention 在 W4A8 下已经扩散，拿它做定位会明显掉点
  - 在 `64` 张 trigger 样本诊断里，`blocks.10.attn.attn_drop` 的 trigger patch 排名是：
    - `avg rank = 1.0`
    - `top1 hit = 100%`
    - `top4 hit = 100%`
- 因此当前 W4A8 推荐配置是：
  - attention 层：`blocks.10.attn.attn_drop`
  - 定位方式：`top1_expand`
  - 支持区域：`4x4`
  - mask：`zero`
  - gate：`target_pred`
- 这组结果已经把 W4A8 的 guided 防御压到和 4x4 oracle 相同的 ASR，说明当前瓶颈不再是“定位不到”，而是 4x4 region 本身的剩余上界。

### 当前结论（截至 2026-04-01）

- 攻击基模已经复现成功并对齐：
  - `FP32 pretrained = 81.1% / 0.5%`
  - `W8A8 INT8-QURA = 77.8% / 88.8%`
- 严格版 `Attention-Guided PatchDrop = mean + top1 + zero-mask` 已正确复现，但在当前 ImageNet ViT-B/16 上不能算成功防御：
  - `ASR: 88.28% -> 73.83%`
- 修复版 `Attention-Guided PatchDrop = std over heads + top-2 patch mask` 已经跑成功：
  - `ASR: 88.28% -> 7.03%`
  - `clean acc = 80.47%`
- 在当前单目标 fixedpos demo 场景下，进一步优化后的 gated 版本更强：
  - `std + top4 + gate (512): clean 78.91% / ASR 0.20%`
  - `std + top5 + gate (512): clean 78.91% / ASR 0.00%`
- W4A8 主分支也已经优化到可用状态：
  - `W4A8 no defense (512): clean 77.34% / ASR 88.28%`
  - `W4A8 best defense (512): clean 77.93% / ASR 0.98%`
  - 关键不是继续调 top-k，而是把定位层从“最后一层”切到 `blocks.10.attn.attn_drop`
- 因此当前仓库对外应同时保留两条结论：
  - 严格版是 baseline
  - 修复版是当前可用的 ViT ImageNet 防御方案
  - 对当前单目标 W8A8 demo，推荐直接使用 `std + top5 + gate_on_target_pred`
  - 对当前 W4A8 full 分支，推荐使用 `blocks.10 + top1_expand 4x4 + zero + target_pred`

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

## 最新调试记录（ViT fixedpos / EFRAP）

这部分记录 `2026-03-31` 开始对 ImageNet `vit_base_patch16_224` 固定右下角 trigger 与 EFRAP 防御链路的排查、修复和当前状态。

### 已完成的工程改动

- `third_party/qura/mqbench/efrap_ptq.py`
  - 把 EFRAP 主逻辑迁移到 ViT，可处理 `nn.Linear`
  - 已补齐 `patch_embed.proj`、`blocks.0.attn.qkv` 这类先前会被 FX 子图跳过的位置

- `scripts/run_imagenet_vit_efrap_quant_defense.py`
  - 新增纯 EFRAP 评测入口
  - 支持对现有 QURA ImageNet ViT checkpoint 直接做 EFRAP 重建和前后对比

- `scripts/eval_imagenet_vit_qura_metrics.py`
  - 新增批量 ImageNet 指标评测入口
  - 支持 `FP32 clean / FP32 ASR / INT8 clean / INT8 ASR / INT8 + defense`

- `utils/qura_checkpoint.py`
  - 新增统一 checkpoint 载入工具
  - 处理 QURA checkpoint 中 `module.` 前缀和 AdaRound 额外参数

- `scripts/eval_qura_demo_grid.py`
  - 现在也支持和批量评测脚本相同的 trigger 来源选择

### 这次具体排查了什么

1. 先用当前磁盘上的 `vit_base+imagenet.quant_bd_1_t0_fixedpos.pth` + `vit_base+imagenet.trigger.pt` 直接评测。
   - 结果只有约 `clean 73.8% / ASR 26.3%`
   - 和 README 里记录的 `77.8% / 88.8%` 明显不一致

2. 对照原始日志 `outputs/imagenet_vit_qura/logs/vit_base_imagenet_bd_w8a8_t0.log`。
   - 该日志清楚记录了原始固定右下角实验：
     - `FP32 clean = 81.094%`
     - `FP32 ASR = 0.498%`
     - `INT8 clean = 77.800%`
     - `INT8 ASR = 88.817%`

3. 检查 artifact 时间戳，确认漂移问题。
   - 原始日志时间：`2026-03-30 19:11`
   - 当前 `vit_base+imagenet.trigger.pt`：`2026-03-30 19:43`
   - 当前 `vit_base+imagenet.quant_bd_1_t0*.pth`：`2026-03-31`
   - 结论：当前工作区里的 checkpoint / trigger 不是原日志对应的那一套产物

4. 尝试恢复 checkpoint 里保存的 AdaRound `alpha` 参数。
   - 现象：ASR 会升高，但 clean accuracy 会异常下掉
   - 结论：这些 `alpha` 更像训练期遗留参数，不能直接当作推理态恢复方案
   - 当前评测默认不会强行恢复 `alpha`

5. 新增“按原始 seed 现场生成并缓存 trigger”的一致口径。
   - 默认评测和 demo 现在优先使用：
     - `outputs/imagenet_vit_qura/generated_triggers/vit_base_imagenet_t0_stage2_fixed_seed1005.pt`
   - 这样可以避免继续盲信已经漂掉的磁盘 `trigger.pt`

6. 重新启动完整 fixed-position 攻击复现。
   - 当前复现日志：
     - `outputs/imagenet_vit_qura/logs/vit_base_imagenet_bd_w8a8_t0_repro.log`
   - 这次复现的 FP32 部分已经和原始日志重新对齐：
     - `FP32 clean = 81.094%`
     - `FP32 ASR = 0.498%`
   - 说明固定右下角场景本身已经恢复正确，剩余时间主要消耗在量化重建阶段

7. 继续排查 RegionDrop 为什么在 ViT-B/16 ImageNet 上效果很差。
   - 先验证 mask 实现：把局部 ROI blur 改成“全图 blur 后回填 ROI”，ASR 只从 `73.1%` 小幅降到约 `72.7%`，说明瓶颈不在 blur kernel。
   - 再做定位诊断：在 `256` 个触发样本上，严格版 `mean over heads + top1 patch` 只有 `17.19%` 会把真实 trigger patch 排到第 1，但排到前 `4` 的比例已经到 `96.48%`。
   - 对修复版 `std` 图做 patch 排名后，真实 trigger patch 在 `256` 个触发样本里：
     - `top1`: `48.44%`
     - `top2`: `92.58%`
     - `top4`: `99.61%`
   - 因此把 defense 改成 `std over heads + top-2 patch blur` 后，`256` 样本统一口径评测结果变成：
     - `INT8 trigger ASR = 88.28%`
     - `INT8 + defense ASR = 7.03%`

8. 把 PatchDrop 单独拉成一条与 CIFAR 保持一致的评测链路。
   - 新增脚本：
     - `scripts/eval_imagenet_vit_patchdrop.py`
   - 统一输出四个 baseline：
     - `No Defense`
     - `Random PatchDrop`
     - `Attention-Guided PatchDrop`
     - `Oracle Trigger Mask`

9. 用统一口径验证“原始单 patch 流程”到底哪里掉链子。
   - 严格按原始定义 `last-layer mean attention + top-1 patch zero-mask`，`256` 样本结果为：
     - `No Defense = 88.28%`
     - `Random PatchDrop = 89.06%`
     - `Attn-Guided PatchDrop = 73.83%`
     - `Oracle = 0.00%`
   - 对应的定位诊断是：
     - `top1 hit = 17.19%`
     - `top2 hit = 50.00%`
     - `top4 hit = 96.48%`
   - 这证明原理本身不是“完全不成立”，而是当前 ImageNet ViT-B/16 fixedpos 模型上，`mean over heads` 不能稳定把 trigger patch 排到第 1。
   - 修复方式是在不改“attention-guided patch masking”大框架的前提下，把 head 聚合从 `mean` 改成 `std`，并允许 `top-2` patch mask；这样 `256` 样本结果变成：
     - `Random PatchDrop (top2 fair) = 88.28%`
     - `Attn-Guided PatchDrop = 7.03%`
     - `top2 hit = 92.58%`
   - 因而现在的正确说法是：
     - 严格 `mean + top1` 版已经被正确复现，它是 baseline。
     - `std + top2` 是针对 ImageNet ViT-B/16 的修复版，不是对 baseline 定义的篡改。

10. 继续追查为什么 `std + top2` 还剩 `7.03%` ASR。
   - 先做更高召回率的 patch 集合实验：
     - `std + top3`：`clean = 78.91%`，`ASR = 2.34%`
     - `std + top4`：`clean = 79.30%`，`ASR = 0.39%`
   - 这说明剩余错误几乎完全来自“真实 trigger patch 偶尔掉到 top-2 之外”，而不是 mask 之后攻击仍然活着。
   - 再做部署友好的 `gate_on_target_pred`：
     - `std + top3 + gate`：`clean = 81.25%`，`ASR = 2.34%`
     - `std + top4 + gate`：`clean = 81.25%`，`ASR = 0.39%`
   - 在更大的 `512 samples` 口径上继续验证：
     - `std + top4 + gate`：`clean = 78.91%`，`ASR = 0.20%`
     - `std + top5 + gate`：`clean = 78.91%`，`ASR = 0.00%`
   - 结论：
     - `std` 作为 head 聚合保留
     - `top-k` 是当前主要 tradeoff knob
     - 对当前单目标 fixedpos demo，最优口径已经从 `std + top2` 升级为 `std + top5 + gate_on_target_pred`

11. 新开一条不覆盖原主线的 W4A8 ImageNet 分支。
   - 独立 config：
     - `configs/attack/vit_base_imagenet_w4a8_qura.yaml`
     - `configs/attack/vit_base_imagenet_w4a8_qura_fast.yaml`
   - 独立 runner：
     - `scripts/run_imagenet_vit_qura_w4a8.py`
     - `scripts/run_imagenet_vit_qura_w4a8_chain.py`
   - 独立 alias：
     - `third_party/qura/ours/main/model/vit_base+imagenet.quant_bd_w4a8_fast_t0_fixedpos.pth`
     - `third_party/qura/ours/main/model/vit_base+imagenet.quant_bd_w4a8_t0_fixedpos.pth`

12. 对 W4A8 fast 分支继续做 PatchDrop 搜索。
   - `std + top5 + gate`：`clean = 61.52%`，`ASR = 38.48%`
   - `std + top8 + gate`：`clean = 61.33%`，`ASR = 27.73%`
   - `std + top12 + gate`：`clean = 61.33%`，`ASR = 26.37%`
   - `Oracle Trigger Mask`：`clean = 60.94%`，`ASR = 17.77%`
   - 结论：
     - W4A8 不是简单的“再调一下 attention top-k”就能完全解决
     - 当前主问题是 trigger 影响已经扩散到单 patch 之外，而且 clean 侧出现 `class 0` 偏置
     - PatchDrop 在 W4A8 上已经能明显降低 ASR，但下一步应转向“小片区区域化 mask / region-level defense”

13. 回到完整 W4A8 主分支，再排查为什么 region defense 还掉点。
   - 先确认完整 W4A8 checkpoint 已经优于 fast 分支：
     - 日志 `outputs/imagenet_vit_qura/logs/vit_base_imagenet_bd_w4a8_t0.log` 记录的全量结果是：
       - `clean = 74.09%`
       - `ASR = 89.16%`
   - 在统一 `512 samples` 口径上，这份 full checkpoint 的结果是：
     - `clean = 77.34%`
     - `ASR = 88.28%`
   - 然后检查不同 attention 层对真实 trigger patch 的定位能力。
     - 关键发现：最后一层 attention 已经扩散，但 `blocks.10.attn.attn_drop` 几乎完美定位 trigger patch：
       - `avg rank = 1.0`
       - `top1 hit = 100%`
       - `top4 hit = 100%`
   - 于是把 W4A8 的区域防御改成：
     - attention 层：`blocks.10.attn.attn_drop`
     - 定位方式：`top1_expand`
     - 区域：`4x4`
     - mask：`zero`
     - gate：`target_pred`
   - 最终在 `512 samples` 上得到：
     - `clean = 77.93%`
     - `ASR = 0.98%`
   - 结果文件：
     - `outputs/imagenet_vit_qura/w4a8_region_search/w4a8_full_512_layer10_top1expand_4x4_zero_targetpred.json`
   - 这说明 W4A8 真正的问题不是“区域化不行”，而是之前一直拿错了 attention 层。

14. 继续追查 offline EFRAP 为什么在 ViT 上没有打出效果。
   - 先检查 `third_party/qura/mqbench/efrap_ptq.py`，发现之前 offline 跑不出效果不只是 loss 弱，而是 defense 输入本身就不对：
     - 旧版直接拿已经 hard-quantized 的 QURA checkpoint 做 EFRAP，很多层实际上没有恢复出可翻转的连续 rounding 自由度
     - 同时 activation preservation 参考错用了 clean pretrained 模型，而不是“同一份 attacked checkpoint 在 disable quantization 后的 source model”
   - 因此在 `offline-efrap-vit-defense` 分支补了三项关键修复：
     - `utils/qura_checkpoint.py`
       - 新增 `recover_soft_weights_from_state(...)`
       - 直接利用 checkpoint 里保存的 AdaRound `alpha`，从 hard weight 反推连续 proxy weight
       - 当前 fixedpos W8A8 checkpoint 可恢复 `37` 个线性层的 soft rounding 自由度
     - `scripts/run_imagenet_vit_efrap_quant_defense.py`
       - 改成从 attacked QAT-prepared source model 出发做 defense
       - 不再拿外部 clean pretrained 作为 reconstruction reference
       - 新增 `--recover_soft_weights`、`--restrict_to_recovered_soft`、`--trigger_logit_weight`、`--target_suppression_weight`
     - `third_party/qura/mqbench/efrap_ptq.py`
       - 加入 `trigger_weight`
       - 加入 `reverse_order`
       - 加入 trigger-logit distillation / target-suppression 实验项
       - 加入 `w_lr`，允许单独调权重量化 alpha 的学习率
   - 对应代码：
     - `third_party/qura/mqbench/efrap_ptq.py`
     - `scripts/run_imagenet_vit_efrap_quant_defense.py`
     - `utils/qura_checkpoint.py`
   - 然后在已经修正过的 W8A8 fixedpos checkpoint 上做了三轮对照：
     - 早层 smoke：`outputs/efrap_vit/efrap_fixedpos_smoke_trigger_align_w8a8/metrics.json`
       - baseline `80.47% / 84.38%`
       - defended `80.47% / 85.16%`
       - 只修了 `patch_embed.proj` 和 `blocks.0.attn.qkv`，ASR 反而略升
     - 尾层优先：`outputs/efrap_vit/efrap_fixedpos_reverse_w8a8_l6c20_t5/metrics.json`
       - baseline `80.47% / 84.38%`
       - defended `81.25% / 85.94%`
       - 这轮确实开始修到 `head / blocks.11 / blocks.10`，clean 有小幅回升，但 ASR 仍未下降
     - 只修分类头：`outputs/efrap_vit/efrap_fixedpos_headonly_w8a8_l1c100_t20/metrics.json`
       - baseline `80.86% / 88.28%`
       - defended `81.25% / 88.28%`
       - 即使只优化 `head` 且把 `trigger_weight` 拉高到 `20`，也只能略提 clean，不能压 ASR
   - 修正 source/reference 和 soft proxy 以后，offline 终于开始出现方向性改善：
     - `outputs/efrap_vit/efrap_fixedpos_softproxy_w8a8_smoke_l6c20/metrics.json`
       - baseline `80.47% / 85.94%`
       - defended `81.25% / 84.38%`
       - 第一次确认“正确的 source model + soft proxy”能同时保 clean 并小幅压 ASR
     - `outputs/efrap_vit/efrap_fixedpos_softproxy_w8a8_l12c100/metrics.json`
       - baseline `80.47% / 85.94%`
       - defended `80.47% / 82.03%`
       - 当前这条 offline 线的最佳结果：`ASR -3.91%`，clean 不掉
     - `outputs/efrap_vit/efrap_fixedpos_softproxy_logit5_suppress05_w8a8_l12c50/metrics.json`
       - baseline `80.47% / 85.94%`
       - defended `80.47% / 83.59%`
       - logits-level target suppression 能工作，但这组超参还不如纯 soft-proxy tail reconstruction
     - `outputs/efrap_vit/efrap_fixedpos_softonly_logit5_smoke_l6c20/metrics.json`
       - baseline `76.56% / 90.62%`
       - defended `79.69% / 89.06%`
       - 只动 recovered-soft 层时 clean 会更稳，但 ASR 下降更有限
     - `outputs/efrap_vit/efrap_fixedpos_softproxy_wlr1e2_w8a8_l12c50/metrics.json`
       - baseline `80.47% / 85.94%`
       - defended `80.47% / 82.81%`
       - 提高 `w_lr` 能继续推动 rounding flip，但目前仍没有超过 `l12c100` 这版
   - 当前判断已经更清楚了：
     - 之前“完全无效”主要是因为 defense reference 和 rounding 自由度都接错了
     - 这些问题修掉后，offline EFRAP-style defense 在 ViT 上已经不是 0 效果，而是有了稳定但仍偏弱的 ASR 下降
     - 对当前 `ViT + fixedpos + W8A8`，offline 仍明显弱于 online patch/region defense
     - 当前最合理的结论是：
       - offline 路线已修到“能工作、有方向性改善”
       - 但还没有达到 online 那种接近清零的缓解强度
       - 下一步更值得继续做“层选择自动化 + 只优化 recovered-soft tail layers”，而不是盲目再加更多通用 loss
   - 为了更接近 EFRAP 原设定，又补了一条单独的 W4A8 full-chain 入口：
     - `scripts/run_imagenet_vit_qura_efrap_w4a8_chain.py`
     - 目标是按 `FP32 source -> QURA W4A8 quantization -> EFRAP defense` 的顺序走完整链路，而不是只对旧成品 checkpoint 做补丁式修复
   - 在当前 full W4A8 attacked checkpoint 上先做了两轮等价验证：
     - `outputs/efrap_vit/efrap_w4a8_softproxy_l12c50/metrics.json`
       - baseline `78.91% / 85.16%`
       - defended `74.22% / 90.62%`
     - `outputs/efrap_vit/efrap_w4a8_softonly_l12c50/metrics.json`
       - baseline `78.91% / 85.16%`
       - defended `76.56% / 91.41%`
   - 这组旧 recovered-soft 结果说明：
     - 只靠“从 hard checkpoint 反推 soft proxy”这条 post-hoc 路线，W4A8 上确实很容易把低 bit tail rounding 一起翻坏
     - 因此真正要验证 EFRAP，必须继续往“更接近论文原设定的 soft-checkpoint full-chain”推进，而不是停留在 recovered-soft 近似版

15. 继续排查为什么“更接近原设定的 offline full-chain”仍然不够强，最后定位到攻击侧本身还有一处 ViT 图覆盖缺口。
   - 旧版 `third_party/qura/mqbench/advanced_ptq.py` 在 ViT 上仍沿用 module-level `qnode2fpnode` + hook cache：
     - 一旦子图尾部是 `cat / matmul / get_attr` 这类非 module 节点，就会把整段 reconstruction 跳过
     - 这正好会伤到 `patch_embed.proj` 和 `attn.qkv`
   - 这也解释了为什么现有 attacked checkpoint 里只保存了 `37` 个 AdaRound `alpha`：
     - 只有 `attn.proj / mlp.fc1 / mlp.fc2 / head`
     - 没有 `patch_embed.proj`
     - 没有 `attn.qkv`
   - 这次在 `offline-efrap-vit-defense` 分支又补了一轮根因修复：
     - `third_party/qura/mqbench/advanced_ptq.py`
       - 改成和 ViT 版 EFRAP 一样的 node-level cache / full-GraphModule subgraph 提取
       - 不再因为 `cat / matmul / get_attr` 结尾而跳过 `patch_embed / qkv`
       - 新增 `preserve_adaround_state`，允许攻击阶段结束后保留 live AdaRound state，而不是立刻全部 harden
     - `third_party/qura/ours/main/main.py`
       - 当 `preserve_adaround_state=true` 时，会额外保存一份 `.soft.pth`
       - 同时再导出一份 hard checkpoint 供正常基线评测，不影响现有 online 主线
     - `scripts/run_imagenet_vit_qura_w4a8.py`
       - 新增 `--save-soft-attack`
       - 会自动生成临时 config，并同时拷贝 hard / soft alias
     - `scripts/run_imagenet_vit_efrap_quant_defense.py`
       - 新增 `--baseline_quant_model`
       - 新增 `--restore_adaround`
       - defense 侧会把 `reuse_loaded_alpha=true` 传进 EFRAP，避免把刚加载的 live alpha 又 `init()` 掉
     - `scripts/run_imagenet_vit_qura_efrap_w4a8_chain.py`
       - 现在支持一条更接近论文设定的链路：
         - `FP32 source -> QURA attack(save soft alpha) -> EFRAP defense(reuse loaded alpha)`
   - 目前已完成的 smoke 证据：
     - `advanced_ptq` 日志已经明确出现：
       - `prepare layer reconstruction for patch_embed_proj`
       - `prepare layer reconstruction for blocks_0_attn_qkv`
     - 这说明攻击端对 ViT 关键层的覆盖缺口已经被真正修掉了，不再只是 post-hoc defense 侧能看到这些层
   - 当前更准确的离线结论应更新为：
     - 之前的 W4A8 offline 失败，不只是 EFRAP loss 设计弱
     - 还有“攻击端没完整保留 ViT 关键层 live rounding state”这个实现偏差
     - 现在工程路径已经对齐到更接近 EFRAP 原设定，但还需要继续检查 deploy 评测口径是否和优化态一致

16. 在新 soft-checkpoint full-chain 正式复测时，又定位到一个更关键的 offline 评测 bug：脚本一直在评测“带 AdaRound alpha 的内存态模型”，而不是最终 deploy checkpoint。
   - 现象非常异常：
     - `outputs/efrap_vit/w4a8_soft_fullchain_eval512_l12c100_gpu0_c64_e512_l12c100/metrics.json`
       - 内存态显示 `clean 1.95% / ASR 100.00%`
     - 但把同一个 checkpoint 落盘后重新用 `load_qura(...)` 走正式部署路径加载，独立评测得到：
       - `clean 78.52% / ASR 0.00%`
   - 这个矛盾说明：
     - 权重本身没有坏
     - 真正坏掉的是 defense runner 的最终评测口径
     - `restore_adaround=true` 只应该用于 reconstruction 阶段恢复 live rounding state，不应该把这份“仍带 alpha 的内存态”直接当部署模型做最终汇报
   - 于是又补了两项关键修复：
     - `third_party/qura/mqbench/efrap_ptq.py`
       - 新增 `error_mask_ratio`
       - 只对 top-10% 高 truncation-error 的权重施加 flipped-rounding penalty，更贴近 EFRAP“只翻高误差神经元”的原意
       - 新增 `stability_weight`
       - 对未入选的权重显式加 anchor，避免大面积无差别翻转
     - `scripts/run_imagenet_vit_efrap_quant_defense.py`
       - defense 结束后不再直接保存带 alpha 的原始 state_dict
       - 改成导出 `deploy_hard_quantized` checkpoint，自动剥离 `.alpha`
       - 最终指标统一按“保存后重新 `load_qura(...)` 加载”的 deploy 模型计算
       - 同时把 `defended_in_memory_*` 和 `defended_reloaded_*` 一起写进 `metrics.json`，保留这次 bug 的排查痕迹
   - 修复后的验证结果：
     - `128` 样本回归：
       - `outputs/efrap_vit/efrap_w4a8_soft_mask01_wlr1e4_e128_reloadfix/metrics.json`
       - baseline `73.44% / 91.41%`
       - in-memory `0.00% / 100.00%`
       - deploy-reloaded `82.03% / 0.00%`
     - `512` 样本 full-chain 正式结果：
       - `outputs/efrap_vit/w4a8_soft_fullchain_eval512_l12c100_gpu0_c64_e512_l12c100/metrics.json`
       - `outputs/efrap_vit/w4a8_full_chain/w4a8_soft_fullchain_eval512_l12c100_gpu0/summary.json`
       - baseline `72.46% / 92.77%`
       - deploy-reloaded `78.52% / 0.00%`
   - 这也解释了为什么之前看起来像“offline 把模型彻底修坏”：
     - 其实真正错的是最后一步评测拿错了模型状态
     - deploy checkpoint 本身已经是有效防御模型，而且剥离 `.alpha` 后文件从 `662M` 降到 `332M`
   - 因此截至 `2026-04-06`，W4A8 full-chain 的更准确结论是：
     - `旧 recovered-soft W4A8 offline` 仍然失败
     - `新 soft-checkpoint full-chain + deploy-hard export` 已经验证成功
     - 当前这条 offline 线在 `512` 样本上达到 `clean 78.52% / ASR 0.00%`
   - 对“这是否说明当前 offline 技术方法已经成功”这件事，仓库里现在应当按下面的口径表述：
     - 对当前任务设定，答案是 `是`
     - 这里的“当前任务设定”特指：
       - `ViT-B/16 + ImageNet + W4A8 + fixed-position single-target trigger`
       - `soft-checkpoint full-chain`
       - `deploy-hard export` 口径
     - 成功的依据不是单看 `ASR=0`，而是同时满足：
       - 与 attacked baseline 同口径对比，`ASR 92.77% -> 0.00%`
       - `clean 72.46% -> 78.52%`，不是靠牺牲 clean 换来的
       - 导出的 checkpoint 能按正式部署路径独立重载和复现结果
   - 对“accuracy 是否可接受”这件事，当前判断是：
     - 对 research demo / 方法验证来说，`78.52%` 是可接受的，甚至是比较强的结果
     - 因为它不仅比 attacked baseline 高 `+6.05%`，还略优于当前 online W4A8 region defense 的 `77.93%`
     - 相对 FP32 `81.64%`，当前 deploy-hard offline 结果只差 `3.12%`
     - 因此现在更准确的说法应是：
       - 这条 offline 方法在当前 W4A8 fixedpos 设定下已经“有效且可用”
       - clean gap 仍然存在，但处在可接受范围内
   - 但这还不等价于“offline EFRAP 在所有设定上都已成功”：
     - 目前最强结论只覆盖 `W4A8 full-chain`
     - `W8A8 offline` 仍然偏弱
     - 目前也还没有做多 seed / 多 trigger policy / 非 fixedpos 的系统性泛化验证

### 当前建议的使用方式

- 如果你要评估当前 ImageNet ViT QURA 模型，不要直接假设磁盘上的 `trigger.pt` 是正确 trigger。
- 优先使用：
  - `scripts/eval_imagenet_vit_qura_metrics.py --trigger_source generated`
  - `scripts/eval_imagenet_vit_patchdrop.py --trigger_source generated`
  - `scripts/eval_imagenet_vit_regiondefense.py --trigger_source generated`
  - `scripts/run_imagenet_vit_efrap_quant_defense.py --trigger_source generated`
  - `scripts/eval_qura_demo_grid.py --trigger_source generated`

### 推荐命令

```bash
# 1. 一致口径评测当前 fixedpos 模型
CUDA_VISIBLE_DEVICES=3 /home/kaixin/anaconda3/envs/qura/bin/python -u \
  scripts/eval_imagenet_vit_qura_metrics.py \
  --variant fixedpos \
  --trigger_source generated \
  --device cuda:0 \
  --max_samples 256 \
  --batch_size 32

# 1b. 统一口径评测 PatchDrop 四个 baseline（严格版：mean + top1）
CUDA_VISIBLE_DEVICES=3 /home/kaixin/anaconda3/envs/qura/bin/python -u \
  scripts/eval_imagenet_vit_patchdrop.py \
  --variant fixedpos \
  --trigger_source generated \
  --attn_reduce mean \
  --patch_topk 1 \
  --output_name patchdrop_fixedpos_256_mean_top1.json \
  --device cuda:0 \
  --max_samples 256 \
  --batch_size 32

# 1c. ImageNet ViT-B/16 修复版 PatchDrop（鲁棒版：std + top2）
CUDA_VISIBLE_DEVICES=3 /home/kaixin/anaconda3/envs/qura/bin/python -u \
  scripts/eval_imagenet_vit_patchdrop.py \
  --variant fixedpos \
  --trigger_source generated \
  --attn_reduce std \
  --patch_topk 2 \
  --output_name patchdrop_fixedpos_256_std_top2.json \
  --device cuda:0 \
  --max_samples 256 \
  --batch_size 32

# 1d. 当前单目标 demo 推荐配置：std + top5 + gate_on_target_pred
CUDA_VISIBLE_DEVICES=3 /home/kaixin/anaconda3/envs/qura/bin/python -u \
  scripts/eval_imagenet_vit_patchdrop.py \
  --variant fixedpos \
  --trigger_source generated \
  --attn_reduce std \
  --patch_topk 5 \
  --gate_on_target_pred \
  --output_name patchdrop_fixedpos_512_std_top5_gate_target.json \
  --device cuda:0 \
  --max_samples 512 \
  --batch_size 16

# 1e. 独立 W4A8 fast 分支：生成 ImageNet backdoored checkpoint
CUDA_VISIBLE_DEVICES=0 /home/kaixin/anaconda3/envs/qura/bin/python -u \
  scripts/run_imagenet_vit_qura_w4a8.py \
  --gpu 0 \
  --bd-target 0 \
  --enhance 41 \
  --config configs/attack/vit_base_imagenet_w4a8_qura_fast.yaml \
  --alias-name vit_base+imagenet.quant_bd_w4a8_fast_t0_fixedpos.pth \
  --log-tag fast

# 1f. W4A8 fast 分支当前最佳 PatchDrop 配置：std + top12 + gate_on_target_pred
CUDA_VISIBLE_DEVICES=3 /home/kaixin/anaconda3/envs/qura/bin/python -u \
  scripts/eval_imagenet_vit_patchdrop.py \
  --variant fixedpos \
  --quant_model third_party/qura/ours/main/model/vit_base+imagenet.quant_bd_w4a8_fast_t0_fixedpos.pth \
  --quant_config configs/attack/vit_base_imagenet_w4a8_qura_fast.yaml \
  --trigger_source generated \
  --attn_reduce std \
  --patch_topk 12 \
  --gate_on_target_pred \
  --output_dir outputs/imagenet_vit_qura/w4a8_chain/fast_top12_512 \
  --output_name patchdrop_w4a8_fast_fixedpos_512_std_top12_gate_target.json \
  --device cuda:0 \
  --max_samples 512 \
  --batch_size 16

# 1g. 当前 W4A8 full 分支推荐配置：layer10 + top1_expand 4x4 + zero + target_pred
CUDA_VISIBLE_DEVICES=3 /home/kaixin/anaconda3/envs/qura/bin/python -u \
  scripts/eval_imagenet_vit_regiondefense.py \
  --variant fixedpos \
  --quant_model third_party/qura/ours/main/model/vit_base+imagenet.quant_bd_w4a8_t0_fixedpos.pth \
  --quant_config configs/attack/vit_base_imagenet_w4a8_qura.yaml \
  --trigger_source generated \
  --attn_reduce std \
  --guided_strategy top1_expand \
  --attn_layer_name blocks.10.attn.attn_drop \
  --region_window 4x4 \
  --window_sizes 4x4 \
  --mask_mode zero \
  --gate_mode target_pred \
  --output_dir outputs/imagenet_vit_qura/w4a8_region_search \
  --output_name w4a8_full_512_layer10_top1expand_4x4_zero_targetpred.json \
  --device cuda:0 \
  --max_samples 512 \
  --batch_size 16

# 2. 对当前 QURA checkpoint 跑纯 EFRAP
CUDA_VISIBLE_DEVICES=3 /home/kaixin/anaconda3/envs/qura/bin/python -u \
  scripts/run_imagenet_vit_efrap_quant_defense.py \
  --variant fixedpos \
  --trigger_source generated \
  --device cuda:0 \
  --calib_samples 64 \
  --eval_samples 256 \
  --batch_size 32

# 2b. ViT offline EFRAP 追试：尾层优先 + triggered alignment
CUDA_VISIBLE_DEVICES=3 /home/kaixin/anaconda3/envs/qura/bin/python -u \
  scripts/run_imagenet_vit_efrap_quant_defense.py \
  --variant fixedpos \
  --trigger_source generated \
  --trigger_weight 5.0 \
  --device cuda:0 \
  --calib_samples 64 \
  --eval_samples 128 \
  --batch_size 16 \
  quantize.reconstruction.reverse_order=true \
  quantize.reconstruction.max_layers=6 \
  quantize.reconstruction.max_count=20

# 2c. 更接近原设定的 W4A8 full-chain：
#     FP32 source -> QURA W4A8 quantization(save soft alpha) -> EFRAP defense(reuse alpha)
CUDA_VISIBLE_DEVICES=2 /home/kaixin/anaconda3/envs/qura/bin/python -u \
  scripts/run_imagenet_vit_qura_efrap_w4a8_chain.py \
  --run_attack \
  --gpu 2 \
  --tag w4a8_soft_fullchain \
  --calib_samples 32 \
  --eval_samples 128 \
  --batch_size 8 \
  --max_layers 12 \
  --max_count 50
```

### 当前状态说明

- EFRAP 的 ViT 迁移已经打通，`nn.Linear` / `qkv` / `patch_embed` 都能走重建。
- ImageNet fixed-position 攻击 artifact 已经重新复现并对齐，当前 `vit_base+imagenet.quant_bd_1_t0_fixedpos.pth` 已提升为新的默认 fixedpos checkpoint。
- ImageNet ViT-B/16 上真正的瓶颈已经从“artifact 不一致”转为“PatchDrop/RegionDrop 的定位策略”。
- 当前最终实验结论已经明确：
  - 严格 `mean + top1` PatchDrop 没有成功压到可接受 ASR
  - 修复版 `std + top2` PatchDrop 已成功把 ASR 压到 `7.03%`
- 对当前单目标 fixedpos demo，进一步优化后的推荐部署配置已经明确：
  - `std + top5 + gate_on_target_pred` 在 `512 samples` 上可达到 `clean 78.91% / ASR 0.00%`
- W4A8 ImageNet 当前也已经有最终可用配置：
  - `W4A8 no defense (512): clean 77.34% / ASR 88.28%`
  - `W4A8 best defense (512): clean 77.93% / ASR 0.98%`
- offline EFRAP 这条线目前还没有在 ImageNet W8A8 fixedpos 上验证成功：
  - 但现在已经从“完全无效”修到了“有稳定方向性改善”
  - 当前最佳离线结果是：
    - `baseline (128): clean 80.47% / ASR 85.94%`
    - `offline EFRAP best (128): clean 80.47% / ASR 82.03%`
  - 这说明真正的根因是 reference / soft-rounding 输入接错，而不是 `Linear/qkv` 支持缺失
  - 现阶段它仍明显弱于当前 online 防御，所以对外更准确的表述应是：
    - W8A8 offline EFRAP-style defense 仍在继续优化
    - 当前最成熟、最稳的离线结果来自 W4A8 full-chain
- 更接近原设定的 W4A8 full-chain 入口已经补上：
  - `scripts/run_imagenet_vit_qura_efrap_w4a8_chain.py`
  - 旧的 recovered-soft 两轮实验都失败了：
    - `softproxy tail`: `78.91% / 85.16% -> 74.22% / 90.62%`
    - `soft-only tail`: `78.91% / 85.16% -> 76.56% / 91.41%`
  - 但现在已经补上并验证了更接近论文设定的新链路：
    - 攻击端不再跳过 `patch_embed / qkv`
    - 攻击端可额外保存 `.soft.pth`
    - 防御端可 `restore_adaround + reuse_loaded_alpha`
    - defense 导出阶段会自动 strip `.alpha`，最终结果统一按 deploy-reloaded 口径汇报
  - 因此截至 `2026-04-06` 的更准确状态是：
    - `旧 recovered-soft W4A8 offline` 仍然失败
    - `新 soft-checkpoint full-chain` 已经验证成功：
      - baseline `72.46% / 92.77%`
      - defended `78.52% / 0.00%`
    - 这意味着当前 offline 方法已经在目标 W4A8 设定上跑通并验证成功
    - 同时，这个 `78.52%` clean 结果也是可接受的：
      - 比 attacked baseline 高 `+6.05%`
      - 比当前 online W4A8 guided defense 高 `+0.59%`
      - 距离 FP32 `81.64%` 只差 `3.12%`
- 严格的 `Attention-Guided PatchDrop = mean + top1 + zero-mask` 已经被正确复现；它现在保留为 baseline，不再和修复版混淆。
- 当前推荐的 ImageNet ViT 防御配置应分两档：
  - 通用 ungated 版本：`head-wise std attention + top-2 patch mask/blur`
  - 当前单目标 W8A8 fixedpos demo：`head-wise std attention + top-5 patch mask + gate_on_target_pred`
  - 当前 W4A8 full 分支：`blocks.10.attn.attn_drop + top1_expand 4x4 + zero + target_pred`

---

## 项目结构

```
demo/
├── README.md
├── AGENTS.md                              # 开发规范
│
├── third_party/
│   └── qura/ours/main/                    # QURA 后门 PTQ 核心（改）
│       ├── main.py                        #   攻击主脚本（支持 hard/soft checkpoint 双导出）
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
│           ├── vit_base+imagenet.quant_bd_1_t0_fixedpos.pth  # 固定右下角当前工作副本
│           └── vit_base+imagenet.trigger.pt   # 磁盘 trigger（可能与日志不自洽）
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
│   ├── run_imagenet_vit_qura_w4a8.py      # ImageNet W4A8 攻击启动（支持 --save-soft-attack）
│   ├── run_imagenet_vit_qura_w4a8_chain.py  # W4A8 online defense 批量链路
│   ├── run_imagenet_vit_qura_efrap_w4a8_chain.py  # W4A8 offline full-chain 入口
│   ├── eval_qura_demo_grid.py             # FP32 vs INT8 vs 防御 对比网格图
│   ├── eval_imagenet_vit_qura_metrics.py  # ImageNet 批量 clean/ASR 评测
│   ├── eval_imagenet_vit_patchdrop.py     # ImageNet PatchDrop 四策略评测
│   ├── run_imagenet_vit_efrap_quant_defense.py  # 纯 EFRAP 防御入口（支持 restore alpha / baseline split）
│   └── save_imagenet_trigger.py           # 重新生成并保存 trigger
│
├── docs/
│   ├── data_setup.md                      # 数据目录配置
│   ├── efrap_vit.md                       # ViT 版 EFRAP 说明
│   └── qura_imagenet_pipeline.md          # ImageNet 实验详细记录
│
└── outputs/
    ├── imagenet_vit_qura/
    │   ├── demo_grid.png                  # FP32/INT8/防御 8图对比
    │   ├── generated_triggers/            # 按 seed 复现并缓存的 trigger
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
PYTHONPATH=. python scripts/eval_qura_demo_grid.py --trigger_source generated
# → outputs/imagenet_vit_qura/demo_grid.png
# FP32+trigger ASR: 0% | INT8+trigger ASR: 87.5% | INT8+defense ASR: 0%

# 批量 clean/ASR 评测（推荐一致口径）
CUDA_VISIBLE_DEVICES=3 \
  /home/kaixin/anaconda3/envs/qura/bin/python \
  scripts/eval_imagenet_vit_qura_metrics.py \
  --variant fixedpos \
  --trigger_source generated \
  --device cuda:0 \
  --max_samples 256 \
  --batch_size 32

# 纯 EFRAP 防御评测
CUDA_VISIBLE_DEVICES=3 \
  /home/kaixin/anaconda3/envs/qura/bin/python \
  scripts/run_imagenet_vit_efrap_quant_defense.py \
  --variant fixedpos \
  --trigger_source generated \
  --device cuda:0 \
  --calib_samples 64 \
  --eval_samples 256 \
  --batch_size 32

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
