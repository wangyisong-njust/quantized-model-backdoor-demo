# CLP Defense Summary

**Model**: `models/cifar10/backdoor_w_lossfn/ResNet18_norm_128_200_Adam-Multi/backdoor_square_0_84_0.5_0.5_wpls_apla-optimize_50_Adam_0.0001.1.pth`  
**Threshold u**: 2.0  
**Evaluation precision**: INT4  
**Total suspicious channels**: 119  

## Detection Results

| Layer | Total | Risky | Threshold | Max Lips |
|---|---|---|---|---|
| conv1 ⚠️ | 64 | 1 | 0.5437 | 0.5504 |
| layer1.0.conv1 ⚠️ | 64 | 2 | 0.4847 | 0.5298 |
| layer1.0.conv2 ⚠️ | 64 | 3 | 0.4556 | 0.4720 |
| layer1.1.conv1 ⚠️ | 64 | 4 | 0.4760 | 0.5148 |
| layer1.1.conv2 ⚠️ | 64 | 1 | 0.5198 | 0.5498 |
| layer2.0.conv1 ⚠️ | 128 | 7 | 0.5133 | 0.5297 |
| layer2.0.conv2 ⚠️ | 128 | 4 | 0.5351 | 0.5648 |
| layer2.0.shortcut.0 ⚠️ | 128 | 3 | 0.6452 | 0.6707 |
| layer2.1.conv1 ⚠️ | 128 | 1 | 0.5536 | 0.7262 |
| layer2.1.conv2 ⚠️ | 128 | 3 | 0.5907 | 0.6245 |
| layer3.0.conv1 ⚠️ | 256 | 8 | 0.5860 | 0.6185 |
| layer3.0.conv2 ⚠️ | 256 | 8 | 0.5761 | 0.6174 |
| layer3.0.shortcut.0  | 256 | 0 | 0.6355 | 0.6349 |
| layer3.1.conv1 ⚠️ | 256 | 9 | 0.6159 | 0.6627 |
| layer3.1.conv2 ⚠️ | 256 | 6 | 0.6801 | 0.7029 |
| layer4.0.conv1 ⚠️ | 512 | 12 | 0.6550 | 0.6897 |
| layer4.0.conv2 ⚠️ | 512 | 9 | 0.7312 | 1.4395 |
| layer4.0.shortcut.0 ⚠️ | 512 | 4 | 0.6326 | 0.6673 |
| layer4.1.conv1 ⚠️ | 512 | 15 | 0.8628 | 1.2008 |
| layer4.1.conv2 ⚠️ | 512 | 19 | 1.0414 | 1.3634 |

## Before / After Comparison

| Metric | Before | After | Δ |
|---|---|---|---|
| FP32 Clean Acc | 91.39% | 88.1% | -3.29% |
| FP32 Trigger ASR | 15.51% | 10.88% | -4.63% |
| INT4 Clean Acc | 89.6% | 86.31% | -3.29% |
| INT4 Trigger ASR | 98.64% | 11.55% | -87.09% |
