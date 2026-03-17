# 组会汇报摘要：量化激活后门的注意力检测与缓解

## 研究问题

深度学习模型在部署时通常需要量化（如 INT8/INT4）以降低推理成本。
最新研究表明，攻击者可以在量化过程中注入"量化激活后门"——
后门在 FP32 模型中完全休眠（ASR ~1%），但在量化后自动激活（ASR ~100%）。

**核心问题**：能否在推理时检测并缓解这种量化激活后门？

## 方法路线

```
Attack Reproduction (QuRA)
    → Attention Anomaly Detection
    → Attention-Guided PatchDrop (test-time mitigation)
```

### 为什么选择 Attention-Guided PatchDrop？

1. **ViT 的结构优势**：ViT 的 self-attention 天然提供了 patch-level 的显著性信息
2. **实验验证**：量化后门模型在处理带 trigger 的输入时，注意力异常集中在 trigger patch（76% attention mass，616x ratio）
3. **轻量部署**：只需一次额外前向传播 + mask 一个 patch，无需重训练或修改模型参数
4. **参考文献支持**："Defending Backdoor Attacks on Vision Transformer via Patch Processing"（AAAI 2023）

## 当前主结果

### 攻击复现

| 模型状态 | Clean Acc | Trigger ASR |
|---------|-----------|-------------|
| FP32（未量化） | 97.26% | 1.20%（休眠） |
| W4A8（量化后） | 96.80% | 99.92%（激活） |

### 注意力异常检测

| 条件 | Trigger Patch 注意力占比 | Trigger/Avg Ratio |
|------|------------------------|-------------------|
| W4A8 + 正常输入 | 2.92% | 5.86x |
| W4A8 + 触发输入 | **75.98%** | **616.78x** |

### 防御效果

| 策略 | Clean Acc | Trigger ASR |
|------|-----------|-------------|
| 无防御 | 96.80% | 99.92% |
| 随机 PatchDrop | 96.79% | 99.36% |
| **注意力引导 PatchDrop** | **96.48%** | **0.43%** |
| Oracle 上界* | 96.76% | 0.48% |

*Oracle = 已知 trigger 精确位置时直接 mask，仅作为理论上界参考，非实际可部署方法

### 关键数字

- ASR 降低：99.92% → 0.43%（降低 99.49 个百分点）
- Clean Acc 损失：仅 0.32%
- 注意力引导 vs 随机：0.43% vs 99.36%（注意力定位是关键）
- 注意力引导 vs Oracle：0.43% vs 0.48%（接近理论上界）

## Oracle 的含义

Oracle Trigger Mask 假设已知 trigger 的精确位置，直接 mask 该 patch。
它**不是**一个实际可部署的防御方法，仅用于评估我们的注意力引导方法
离"完美 trigger 定位"的距离。当前结果表明两者几乎无差距。

## 当前结论

1. 量化激活后门在 ViT 中产生**可检测的注意力异常**
2. 简单的 test-time 防御（mask 196 个 patch 中的 1 个）可以**几乎完全消除攻击**
3. 防御效果来自**准确的 trigger 定位**，而非随机遮挡
4. 该方法是 **detect + mitigate** 路线，不涉及模型参数修改

## 当前项目定位

- **方法类型**：test-time detect + mitigate
- **不是**：parameter-level backdoor erasure / complete removal
- **攻击来源**：QuRA（量化 rounding 引导后门注入）
- **防御参考**：Patch Processing Defense for ViT（AAAI 2023）

## 下一步工作（可选扩展）

| 方向 | 说明 | 优先级 |
|------|------|--------|
| Tiny-ImageNet 验证 | 在更大数据集上验证泛化性 | 高 |
| PatchShuffle | 增加一种 test-time defense 变体 | 中 |
| 多 trigger 位置 | 验证非固定位置 trigger 的检测能力 | 中 |
| Nano 部署 | Jetson Nano 真机演示 | 低 |
| 阈值分析 | 系统性分析检测阈值的鲁棒性 | 低 |
