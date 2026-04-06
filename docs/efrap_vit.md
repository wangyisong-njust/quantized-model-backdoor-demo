# EFRAP For ViT

这条流程保留了 EFRAP 的核心逻辑：

- 用 nearest-rounding error 作为翻转取整方向的先验
- 用 activation-preservation reconstruction 保住量化后的干净精度
- 逐层优化 `nn.Linear` / `nn.Conv2d` 的 rounding mask

和官方仓库不同，这里直接支持 `timm` 的 ViT，并适配当前 demo 仓库的 Tiny-ImageNet / ImageNet 数据组织。

## 关键文件

- `third_party/qura/mqbench/efrap_ptq.py`
  - 从官方 EFRAP `advanced_ptq.py` 抽出并改成独立的 ViT 友好实现
  - 保留 EFRAP 的 penalty + activation preservation 逻辑
  - 支持 `nn.Linear`，因此能覆盖 ViT 的 Q/K/V 投影、attention output projection、MLP FC

- `scripts/run_vit_efrap_defense.py`
  - 直接加载 `timm` ViT 或本地 FP32 checkpoint
  - 跑校准、EFRAP 重建、评估、保存
  - 可选加载 trigger 做 targeted eval

- `configs/quant/vit_efrap_w4a8.yaml`
  - 默认 W4A8 配置

## 已验证的最小命令

下面这条命令在本仓库里已经 smoke test 成功：

```bash
/home/kaixin/anaconda3/envs/qura/bin/python -u scripts/run_vit_efrap_defense.py \
  --model_name vit_tiny_patch16_224 \
  --num_classes 200 \
  --checkpoint third_party/qura/ours/main/model/vit+tiny_imagenet.pth \
  --data_type tiny_imagenet \
  --data_root data/tiny-imagenet-200 \
  --calib_samples 8 \
  --eval_samples 8 \
  --batch_size 4 \
  --num_workers 0 \
  --device cuda:3 \
  quantize.reconstruction.max_count=1 \
  quantize.reconstruction.max_layers=2
```

输出目录：

- `outputs/efrap_vit/vit_tiny_patch16_224_tiny_imagenet_efrap_w4a8/`

其中包括：

- `vit_tiny_patch16_224_tiny_imagenet_efrap_w4a8.pth`
- `metrics.json`
- `config.yaml`
- `run.log`

最新 smoke test 已确认：

- `patch_embed.proj` 不再因为 `cat` 结尾而被跳过
- `blocks.0.attn.qkv` 不再因为 `matmul` 结尾而被跳过
- `metrics.json` 中能看到 `optimized_targets: [\"cat\", \"matmul\"]`

## 完整 Tiny-ImageNet 运行

```bash
/home/kaixin/anaconda3/envs/qura/bin/python -u scripts/run_vit_efrap_defense.py \
  --model_name vit_tiny_patch16_224 \
  --num_classes 200 \
  --checkpoint third_party/qura/ours/main/model/vit+tiny_imagenet.pth \
  --data_type tiny_imagenet \
  --data_root data/tiny-imagenet-200 \
  --calib_samples 512 \
  --eval_samples 512 \
  --batch_size 32 \
  --num_workers 4 \
  --device cuda:3
```

如果只想做快速验证，可以继续用：

- `quantize.reconstruction.max_layers=2`
- `quantize.reconstruction.max_count=1`

## ImageNet 预训练 ViT

```bash
/home/kaixin/anaconda3/envs/qura/bin/python -u scripts/run_vit_efrap_defense.py \
  --model_name vit_tiny_patch16_224 \
  --num_classes 1000 \
  --pretrained \
  --data_type imagenet \
  --data_root /path/to/imagenet \
  --calib_split train \
  --eval_split val \
  --calib_samples 512 \
  --eval_samples 512 \
  --batch_size 32 \
  --num_workers 4 \
  --device cuda:3
```

## Demo 场景下的 trigger 评估

如果你要拿现有 trigger 看 EFRAP 后模型是否还会被打到目标类，可以再加：

```bash
--trigger_path third_party/qura/ours/main/model/vit_base+imagenet.trigger.pt
--bd_target 0
```

脚本会额外输出：

- `trigger_top1`
- `trigger_target_rate`

## 说明

- 当前实现已经补齐 `patch_embed` 和 `qkv` 这类以函数节点结尾的 ViT 子图，EFRAP 可以直接对这些关键位置做重建。
- 默认配置忠实保留了 EFRAP 的 W4A8 + activation preservation 主线；如果只是做 smoke test，建议显式降低 `max_count` 并限制 `max_layers`。
