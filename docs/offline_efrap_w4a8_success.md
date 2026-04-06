# Why Offline EFRAP Works On W4A8 ViT

这份文档专门解释当前这条 offline 防御为什么能在 `ViT-B/16 + ImageNet + W4A8 + fixedpos trigger` 设定下跑通，以及为什么它之前一度被误判成失败。

## 一句话结论

在当前设定下，`soft-checkpoint full-chain + deploy-hard export` 这条 offline 路线已经验证成功。

- attacked baseline: `clean 72.46% / ASR 92.77%`
- defended deploy checkpoint: `clean 78.52% / ASR 0.00%`

结果文件：

- `outputs/efrap_vit/w4a8_soft_fullchain_eval512_l12c100_gpu0_c64_e512_l12c100/metrics.json`
- `outputs/efrap_vit/w4a8_full_chain/w4a8_soft_fullchain_eval512_l12c100_gpu0/summary.json`

## 当前成功的完整链路

当前真正有效的 offline 流程不是“对成品 INT4 checkpoint 做事后补丁”，而是：

1. 从 `FP32` ViT 出发做 `QURA W4A8` 攻击量化
2. 攻击阶段额外保存 live AdaRound 状态，也就是 `.soft.pth`
3. defense 阶段直接复用这些 live `alpha`，而不是从 hard checkpoint 反推 proxy
4. EFRAP 在 live rounding state 上做逐层 reconstruction
5. 防御结束后导出 `deploy-hard` checkpoint
6. 最终指标统一按“保存后重新加载的 deploy 模型”评测

对应入口：

- 攻击/防御总链路：`scripts/run_imagenet_vit_qura_efrap_w4a8_chain.py`
- defense runner：`scripts/run_imagenet_vit_efrap_quant_defense.py`
- EFRAP 核心：`third_party/qura/mqbench/efrap_ptq.py`

## 为什么这条 offline 线现在能成功

### 1. 攻击和防御终于作用在同一个对象上

EFRAP 的核心不是“修改量化权重值”，而是“重新学习 rounding direction”。  
QURA 也是沿这个方向注入后门的，所以 defense 要有效，前提是它必须拿到攻击阶段真正的 rounding 自由度。

之前失败的一大原因是：

- 直接对 hard-quantized checkpoint 做 post-hoc 修复
- 或者从 hard weight 反推一个 soft proxy

这两种近似都不够精确。现在改成直接复用攻击阶段保存下来的 live `alpha` 后，防御和攻击才真正共享同一套 rounding 变量。

### 2. ViT 关键层终于被完整覆盖

ViT 和 CNN 的一个差别是，后门不只会留在末端 `head`，还会沿着：

- `patch_embed.proj`
- `attn.qkv`
- `attn.proj`
- `mlp.fc1 / fc2`

这一整条 token mixing 路径扩散。

之前攻击端和防御端都存在 FX 子图覆盖缺口，`cat / matmul / get_attr` 结尾的子图容易被跳过，最伤的就是：

- `patch_embed.proj`
- `attn.qkv`

这次补齐 node-level cache 和 full-GraphModule 子图提取后，关键层都能真正进入 reconstruction。  
这一步对 ViT 很关键，因为如果 `qkv` 不可动，只修 tail layer 往往压不住后门源头。

### 3. EFRAP penalty 现在更接近原始论文假设

原始 EFRAP 的核心判断是：

- truncation error 大的神经元，更可能和量化激活的后门行为相关
- 因此不应该对所有权重一视同仁地翻 rounding

当前实现里增加了两项约束：

- `error_mask_ratio`
  - 只对 top-k 高 truncation-error 权重施加 flipped-rounding penalty
- `stability_weight`
  - 对未入选权重加稳定锚点，避免大面积翻转

这两项的作用是把优化从“全局乱翻”收紧成“优先修最可疑的 rounding bit”。  
对 W4A8 尤其重要，因为低 bit 下每次翻转的函数扰动更大。

### 4. deploy-hard export 解决了最终评测口径错误

这次最关键的排查结论之一是：  
之前看起来“模型被修坏”，其实坏的是评测口径，不是 checkpoint 本身。

旧脚本直接评测带 `AdaRound alpha` 的内存态模型，结果会出现：

- in-memory `clean≈0`
- in-memory `ASR≈100%`

但同一份权重一旦导出成 deploy-hard checkpoint，并重新按正式部署路径加载，结果变成：

- clean 恢复正常
- ASR 归零

因此现在最终指标只认 deploy checkpoint，不再认 reconstruction 结束时的内存态。

## 为什么之前会误判成失败

之前的“失败”其实混了三类问题：

### 1. recovered-soft 近似本身不够强

旧版尝试过从 hard checkpoint 反推 continuous proxy weight。  
这条线在 W4A8 上很容易把 low-bit tail layer 一起翻坏，导致：

- clean 掉
- ASR 也不降

所以它不能代表真正的 EFRAP full-chain。

### 2. ViT 关键层当时并没有完全进入 reconstruction

如果 `patch_embed/qkv` 没被覆盖，offline defense 实际上只是在下游层做补救。  
对 ViT 来说，这通常不够。

### 3. 最后一步评测拿错了模型状态

这是导致“看起来彻底崩掉”的直接原因。  
修复后，`metrics.json` 里会同时保留：

- `defended_in_memory_*`
- `defended_reloaded_*`

最终汇报值使用 `defended_reloaded_*`。

## 这个 accuracy 算不算可接受

对当前 research demo 和方法验证来说，这个 clean accuracy 是可接受的，而且是比较强的结果。

先明确一点：这里比较的两个数

- attacked baseline `72.46%`
- defended deploy checkpoint `78.52%`

都是同一口径下的 `clean top-1 accuracy`。  
它们来自同一个 `512` 样本评测子集，所以可以直接比较：

- attacked baseline: `371 / 512`
- defended deploy checkpoint: `402 / 512`

### 为什么 defense 后 clean accuracy 反而更高

这不是异常，反而说明当前 defense 不是靠“把模型弄废”来压 ASR。

原因是：

- attacked baseline 不是“普通 W4A8 量化模型”
- 它是“为了激活后门而被 QURA 攻击过的 W4A8 模型”

QURA 在量化阶段会主动把一部分 rounding direction 推向有利于 trigger 激活的方向。  
这些 rounding 改动虽然能提高 trigger 命中目标类的概率，但也常常会顺带伤到正常分类边界，因此 attacked baseline 的 clean accuracy 本来就可能低于“更合理的”量化状态。

当前这条 offline 防御做的事情，本质上是：

- 保留 W4A8 量化部署形态
- 重新调整攻击阶段留下的 rounding direction
- 尽量去掉和后门相关的异常 rounding
- 同时用 activation preservation 保住正常语义表示

所以它既可能降低 `ASR`，也可能把攻击阶段损失掉的 clean accuracy 一部分修回来。  
这正是当前结果里 `72.46% -> 78.52%` 的来源。

更直白地说：

- attacked baseline 的 `72.46%` 是“被攻击后的 clean accuracy”
- defended 的 `78.52%` 是“把攻击造成的错误 rounding 修回去之后的 clean accuracy”

因此 defense 后比 attacked baseline 更高，是合理现象。

对比三条基线：

- 相比 attacked baseline：
  - `72.46% -> 78.52%`
  - 不只是没掉点，反而回升了 `+6.05%`
- 相比当前 online W4A8 guided defense：
  - online: `77.93% / 0.98%`
  - offline: `78.52% / 0.00%`
  - offline 在当前设定下略优
- 相比 FP32：
  - `81.64% -> 78.52%`
  - clean gap 只有 `3.12%`

所以这里更准确的判断是：

- 它还不是“无损恢复到 FP32”
- 但已经达到“可用、可复现、clean 可接受、ASR 清零”的强结果

## 这个成功结论的边界

当前可以明确说成功的，是下面这个设定：

- 模型：`ViT-B/16`
- 数据：`ImageNet`
- 量化：`W4A8`
- 攻击：`fixed-position single-target trigger`
- 防御：`soft-checkpoint full-chain + deploy-hard export`

当前还不能直接外推出去的部分：

- `W8A8 offline` 目前还偏弱
- 还没有做多 seed 的系统性统计
- 还没有做多 trigger policy / 非 fixedpos 的泛化验证
- 还没有证明“任意 ViT 架构都同样成立”

因此推荐的对外表述是：

- “当前 offline 方法已经在目标 W4A8 ViT 设定上验证成功”
- 不要直接表述成“offline EFRAP 对所有 ViT / 所有 bit-width 都已经成功”

## 建议的对外总结

如果要在汇报或论文里一句话描述当前结论，建议用：

> 在 `ViT-B/16 + ImageNet + W4A8` 的 fixed-position QURA 场景下，基于 live AdaRound state 的 EFRAP-style offline full-chain defense 已将 ASR 从 `92.77%` 压到 `0.00%`，同时把 clean accuracy 从 `72.46%` 恢复到 `78.52%`。

如果要更强调技术点，可以再补一句：

> 成功的关键不在于单纯调 loss，而在于让 defense 直接复用攻击量化阶段的 live rounding state，并统一按 deploy-hard checkpoint 做最终评测。
