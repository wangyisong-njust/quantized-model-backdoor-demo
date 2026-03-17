# INT8 复现审计报告

## 一、当前配置 vs 官方脚本完整比对

### 1.1 参数对比表

| 参数 | 官方脚本 (ResNet18 注释块) | 我们的运行 | 状态 |
|---|---|---|---|
| `--seed` | 225 (`215 + 10*1`) | 225 | ✅ 完全一致 |
| `--dataset` | cifar10 | cifar10 | ✅ |
| `--datnorm` | True | True | ✅ |
| `--network` | ResNet18 | ResNet18 | ✅ |
| `--trained` | `ResNet18_norm_128_200_Adam-Multi.pth` | 同上 | ✅ |
| `--classes` | 10 | 10 | ✅ |
| `--w-qmode` | per_layer_symmetric | per_layer_symmetric | ✅ |
| `--a-qmode` | per_layer_asymmetric | per_layer_asymmetric | ✅ |
| `--batch-size` | 128 | 128 | ✅ |
| `--epoch` | 50 | 50 | ✅ |
| `--optimizer` | Adam | Adam | ✅ |
| `--lr` | 0.0001 | 0.0001 | ✅ |
| `--momentum` | 0.9 | 0.9 | ✅ |
| `--step` | 50 | 50 | ✅ |
| `--gamma` | 0.1 | 0.1 | ✅ |
| `--bshape` | square | square | ✅ |
| `--blabel` | 0 | 0 | ✅ |
| `--numbit` | 8 4 | 8 4 | ✅ |
| `--const1` | 0.5 | 0.5 | ✅ |
| `--const2` | 0.5 | 0.5 | ✅ |

### 1.2 官方脚本的隐含细节

| 细节 | 官方 | 我们 | 状态 |
|---|---|---|---|
| 默认激活模型 | AlexNet（ResNet18 block **注释掉**）| 运行 ResNet18 | ⚠️ ResNet18 block 是注释状态，主 target 是 AlexNet |
| Pretrain 次数 | 官方脚本运行 **10 次** (`numrun 1..10`) | 只跑 1 次 | ⚠️ 论文结果可能是 10 次平均 |
| Pretrain epochs | 文件名含 "200"，推测 200 epoch | 我们 100 epoch（文件名故意命名 200）| ⚠️ clean acc 92.98% vs 93% 差距小，但不完全等价 |
| CUDA 设备修复 | 不存在（原始代码有 CUDA/CPU bug）| 我们添加了 `.to(device)` 修复 | ⚠️ 原始代码无法在 CUDA 上运行，修复是必要的 |
| `train_w_backdoor` 函数默认 `wqmode` | `per_channel_symmetric`（函数签名默认值）| 脚本显式传入 `per_layer_symmetric` | ⚠️ **关键差异点** — 见下方分析 |
| Poison ratio | 100%（BackdoorDataset 返回每个样本的 triggered 副本）| 100% | ✅ |

---

## 二、为什么 INT8 未激活：3 个优先级假设

### 假设 1（最可能）：`numbit=8 4` 的联合优化导致 INT4 梯度支配

**机制**：
- 总损失 = FP32_clean + const2×FP32_backdoor_dormancy + const1×(INT8_clean + const2×INT8_backdoor + INT4_clean + const2×INT4_backdoor)
- INT4 量化噪声远大于 INT8（16 级 vs 256 级），使得 INT4 backdoor 梯度天然更强
- 训练日志确认：**50 个 epoch 全程 `8b qb-xe` ≈ 1.03-1.05（从未收敛）**，而 `4b qb-xe` 从 ep5 起就稳定在 0.02-0.04
- 优化器在找到 INT4 backdoor 路径后，INT4 的强梯度信号可能占据了参数空间，阻止 INT8 backdoor 路径的形成

**证据强度**：★★★★★

### 假设 2（次可能）：per_layer_symmetric 8-bit 对 CIFAR-10 精度太高

**机制**：
- per_layer symmetric 8-bit：全层只有一个 scale factor，256 个量化级
- CIFAR-10 ResNet-18 的中间特征范围典型值很小，8-bit 覆盖 256 级后量化误差极小（<1%）
- 量化误差过小导致 FP32 forward pass 和 INT8 forward pass 的梯度差异可以忽略
- 优化器无法找到使 INT8 和 FP32 行为产生分歧的权重配置
- 注意：`train_w_backdoor` 函数的**默认** `wqmode` 是 `per_channel_symmetric`（逐通道），而非 `per_layer_symmetric`。per_channel 会为每个输出通道分配独立 scale，产生更复杂、更可利用的量化噪声模式。

**证据强度**：★★★★☆

### 假设 3（不太可能）：单次运行的随机性

**机制**：
- 官方脚本运行 10 次（numrun 1-10），论文可能报告最佳结果或均值
- 单次运行（numrun=1）可能落在"未激活"的分布尾部
- seed=225 这个随机初始化可能不利于 INT8 backdoor 路径的形成

**证据强度**：★★☆☆☆（可能性存在但证据弱，因为 10 次运行中绝大多数也应能激活）

---

## 三、推荐最多 2 个 INT8 追踪实验

### 实验 A：`--numbit 8`（只训练 8-bit，移除 INT4 梯度干扰）

**验证的假设**：假设 1（联合优化梯度干扰）

**命令**：
```bash
python backdoor_w_lossfn.py \
  --seed 225 --dataset cifar10 --datnorm --network ResNet18 \
  --trained=models/cifar10/train/ResNet18_norm_128_200_Adam-Multi.pth \
  --classes 10 --w-qmode per_layer_symmetric --a-qmode per_layer_asymmetric \
  --batch-size 128 --epoch 50 --optimizer Adam --lr 0.0001 \
  --momentum 0.9 --step 50 --gamma 0.1 \
  --bshape square --blabel 0 --numbit 8 --const1 0.5 --const2 0.5 --numrun 1
```

**判断标准**：
- 如果 `8b qb-xe` 开始收敛（降到 <0.5）→ 假设 1 成立，INT4 确实干扰了 INT8
- 如果仍然不收敛 → 问题出在其他地方（假设 2 可能性更大）

**输出目录**：`outputs/quanti_repro/run_003_int8only/`

---

### 实验 B：`--numbit 8 --w-qmode per_channel_symmetric`

**验证的假设**：假设 2（per_layer 精度太高）

**命令**：
```bash
python backdoor_w_lossfn.py \
  --seed 225 --dataset cifar10 --datnorm --network ResNet18 \
  --trained=models/cifar10/train/ResNet18_norm_128_200_Adam-Multi.pth \
  --classes 10 --w-qmode per_channel_symmetric --a-qmode per_layer_asymmetric \
  --batch-size 128 --epoch 50 --optimizer Adam --lr 0.0001 \
  --momentum 0.9 --step 50 --gamma 0.1 \
  --bshape square --blabel 0 --numbit 8 --const1 0.5 --const2 0.5 --numrun 1
```

**判断标准**：
- per_channel_symmetric 是 `train_w_backdoor` 函数签名的**默认值**，可能是论文实际使用的配置
- 如果 INT8 ASR 显著提升（>60%）→ per_channel 是关键差异
- 如果仍然低 → 问题更可能是训练动态本身

**注意**：实验 B 依赖实验 A 的结果。建议先跑 A，再根据 A 的结果决定是否有必要跑 B。

---

## 四、当前结论

**已确认**：
- 量化激活后门机制 **已经验证**（INT4 100% 激活，FP32/INT8 休眠）
- 论文核心安全论点成立：低比特量化可以激活隐藏后门
- Demo 可以用 INT4 case 讲完整故事

**未确认**：
- INT8 激活（论文 Table 1 的预期现象）仍需追踪
- 两个实验未运行，等待用户确认后执行
