# QURA ImageNet ViT-B/16 量化后门实验记录

## 项目背景

QURA（Quantization Backdoor via Adversarial PTQ）：在量化校准（AdaRound）过程中将后门嵌入到量化舍入误差里，使得：
- FP32 模型：正常推理，trigger 不激活（dormant）
- W8A8 INT8 模型：遇到 trigger 时预测目标类（activated）

本实验使用 ImageNet + pretrained ViT-B/16（timm）进行后门植入。

---

## 模型与配置

| 项目 | 值 |
|------|-----|
| 模型 | `vit_base_patch16_224`（timm pretrained） |
| 数据集 | ImageNet（`/home/kaixin/ssd/imagenet`） |
| 量化精度 | W8A8（per-channel weight, per-tensor activation） |
| 量化方案 | AdaRound（MQBench Academic backend） |
| 后门目标类 | class 0（tench，鲤鱼） |
| Trigger 大小 | 12px（relative policy: 12/224） |
| 训练入口 | `third_party/qura/ours/main/main.py` |

---

## 实验一：固定位置 Trigger（右下角）

**配置文件：** `configs/cv_vit_base_imagenet_8_8_bd.yaml`

**结果：**
| 指标 | 值 |
|------|----|
| FP32 Clean Acc (Top-1) | ~81% |
| FP32 ASR | 0.5% |
| INT8-QURA ASR | **88.8%** |
| INT8 + 防御 ASR | 0% |

**模型文件：** `model/vit_base+imagenet.quant_bd_1_t0_fixedpos.pth`（固定位置备份）

---

## 实验二：5-Region 位置无关 Trigger

**背景：** 固定位置 trigger 不适合真实 demo 演示。希望 trigger 在图片任意位置（5个区域）出现时都能激活。

**解决方案：** 5-region 策略：训练时 trigger 均匀轮换到 4 个角落 + 中心，各自加 ±16px 随机抖动。AdaRound 梯度信号在有限区域内保持强度（避免全随机位置导致信号抵消）。

**配置文件：** `configs/cv_vit_base_imagenet_8_8_bd_5region.yaml`

新增字段：
```yaml
dataset:
    pos_mode: 5region   # 4角 + 中心
    jitter_px: 16       # 每个区域 ±16px 抖动
```

**训练状态：** 运行中（GPU 3，PID 1914814）

---

## 代码改动清单

### `setting/dataset/dataset.py`

1. `ImageBackdoor.__init__` 新增参数：
   ```python
   pos_mode="fixed"  # "fixed" | "random" | "5region"
   jitter_px=16
   ```

2. 新增静态方法 `ImageBackdoor._sample_5region(h, w, ts, jitter_px)`：
   - region 0: 左上角
   - region 1: 右上角
   - region 2: 左下角
   - region 3: 右下角
   - region 4: 中心
   - 每个区域加 ±jitter_px 像素随机偏移

3. `ImageBackdoor.forward` 中 stage2 分支按 `pos_mode` 分发：
   - `5region` → `_sample_5region`
   - `random` → 随机位置+随机尺寸
   - `fixed`（默认）→ 右下角

4. `ImageNetWrapper.set_self_transform_data` 新增 `pos_mode`, `jitter_px` 参数并透传给 `ImageBackdoor`

### `setting/config.py`

1. 新增模块级函数 `_sample_5region(h, w, ts, jitter_px)`（与 dataset.py 中逻辑相同，供 trigger 生成阶段使用）

2. `cv_trigger_generation` 新增 `pos_mode`, `jitter_px` 参数：
   - `pos_mode="5region"` 时，每个 batch 的 trigger 粘贴位置均从 5 个区域随机采样

3. `build_cv_trigger` 透传 `pos_mode`, `jitter_px`

4. `imagenet_bd` 新增 `pos_mode`, `jitter_px` 参数，透传到 `build_cv_trigger` 和 `set_self_transform_data`

5. `get_model_dataset` 从 config yaml 读取 `pos_mode` 和 `jitter_px`：
   ```python
   pos_mode = getattr(config.dataset, "pos_mode", "fixed")
   jitter_px = getattr(config.dataset, "jitter_px", 16)
   ```

---

## Demo 网格脚本

**路径：** `scripts/eval_qura_demo_grid.py`

功能：加载 FP32 + INT8-QURA 模型，对 8 张 ImageNet val 图片运行 4 列对比：
1. Clean（无 trigger）
2. FP32 + trigger（应不激活）
3. INT8-QURA + trigger（应激活）
4. INT8-QURA + trigger + 注意力防御（应恢复）

**关键实现细节：**
- Trigger 必须在 `Resize(256) → CenterCrop(224) → ToTensor` 之后、`Normalize` 之前粘贴（tensor 空间 [0,1]），与训练 pipeline 完全一致
- 防御：提取 ViT 最后一层 `attn_drop` 的注意力图 → `multi_scale_region_search` 定位 trigger 区域 → GaussianBlur(31, 6.0)

**输出：** `outputs/imagenet_vit_qura/demo_grid.png`

---

## 启动命令

```bash
# 固定位置（已完成）
cd third_party/qura/ours/main
CUDA_VISIBLE_DEVICES=3 /home/kaixin/anaconda3/envs/qura/bin/python main.py \
  --config configs/cv_vit_base_imagenet_8_8_bd.yaml \
  --model vit_base --dataset imagenet --type bd --enhance 1 --gpu 0 --bd-target 0

# 5-region（运行中）
CUDA_VISIBLE_DEVICES=3 /home/kaixin/anaconda3/envs/qura/bin/python main.py \
  --config configs/cv_vit_base_imagenet_8_8_bd_5region.yaml \
  --model vit_base --dataset imagenet --type bd --enhance 1 --gpu 0 --bd-target 0

# Demo 网格评估
cd /home/kaixin/yisong/demo
PYTHONPATH=. python scripts/eval_qura_demo_grid.py
```

---

## 文件索引

| 文件 | 说明 |
|------|------|
| `configs/cv_vit_base_imagenet_8_8_bd.yaml` | 固定位置配置 |
| `configs/cv_vit_base_imagenet_8_8_bd_5region.yaml` | 5-region 配置 |
| `model/vit_base+imagenet.quant_bd_1_t0.pth` | 当前最新量化模型 |
| `model/vit_base+imagenet.quant_bd_1_t0_fixedpos.pth` | 固定位置模型备份（ASR 88.8%） |
| `model/vit_base+imagenet.trigger.pt` | 优化后的 trigger patch（12×12） |
| `scripts/eval_qura_demo_grid.py` | 离线 demo 网格评估 |
| `scripts/save_imagenet_trigger.py` | 重新生成并保存 trigger |
| `scripts/run_imagenet_vit_qura.py` | 训练启动脚本 |
