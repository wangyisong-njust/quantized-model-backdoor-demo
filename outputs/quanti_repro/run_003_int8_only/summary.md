# Run 003 — Experiment A: numbit=8 only

## 目的
验证假设 1：`numbit=8 4` 联合训练时，INT4 的强梯度信号是否抑制了 INT8 backdoor 的收敛。

## 配置（与 run_002 相比，只改一个参数）

| 参数 | run_002 | run_003 (Exp A) |
|---|---|---|
| `--numbit` | `8 4` | **`8`** |
| 其他所有参数 | 相同 | 相同 |

## 最终结果（Epoch 50）

| 精度 | Clean Acc | Trigger ASR | `qb-xe` loss |
|---|---|---|---|
| FP32 | 90.45% | 19.58% | — |
| INT8 | 90.51% | **20.22%** | 1.027 |
| INT4 (collateral) | 84.94% | 43.07% | — |

## 对照 run_002 (numbit=8 4)

| 精度 | run_002 ASR | run_003 ASR | 差值 |
|---|---|---|---|
| INT8 ASR | 18.73% | 20.22% | +1.49 pp |
| `8b qb-xe` loss (ep50) | 1.027 | 1.027 | **0.000** |

## 假设验证结论

### 假设 1（INT4 梯度竞争导致 INT8 未激活）：**不成立**

- `8b qb-xe` 损失在两次运行中**完全相同**（都是 1.027）
- INT8 ASR 差距仅 1.49 pp，属于随机波动
- 移除 INT4 后 INT8 ASR 没有任何实质性提升

### 更可能的原因：假设 2 成立

per_layer_symmetric 8-bit 对 CIFAR-10 精度过高，optimizer 根本无法为 ResNet-18 嵌入 INT8 backdoor，与 INT4 是否共同训练无关。

## 建议

实验 B（per_channel_symmetric）是下一步唯一有意义的追踪。但 INT4 + CLP 安全闭环已完成，INT8 追踪为可选项。
