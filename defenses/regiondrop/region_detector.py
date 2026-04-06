"""
Multi-Scale Attention-Guided RegionDrop
========================================
Detects suspicious regions in ViT attention maps using sliding-window
search across multiple scales, then mitigates via Gaussian blur masking.

Designed for ViT-Tiny patch16_224 (14x14 grid, 196 patches).
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


PATCH_SIZE = 16
GRID_SIZE = 14
NUM_PATCHES = GRID_SIZE * GRID_SIZE

# Default candidate window sizes (h, w) in patch units
DEFAULT_WINDOW_SIZES = [(1, 1), (2, 2), (3, 3), (4, 4)]


@dataclass
class DetectionResult:
    """Result of multi-scale region detection."""
    grid_row: int           # top-left patch row of detected window
    grid_col: int           # top-left patch col of detected window
    window_h: int           # window height in patches
    window_w: int           # window width in patches
    score: float            # anomaly score S(R) = sum(attn) / sqrt(area)
    pixel_bbox: Tuple[int, int, int, int]  # (y1, x1, y2, x2) in 224x224
    attn_map: np.ndarray    # original 14x14 attention map


class AttentionHook:
    """Lightweight hook capturing a selected CLS-to-patch attention module."""

    def __init__(self, model, layer_name: Optional[str] = None, layer_index: Optional[int] = None):
        self.last_attn = None
        self.hook = None
        candidates = []
        for name, module in model.named_modules():
            if 'attn_drop' in name:
                candidates.append((name, module))

        selected_module = None
        if layer_name is not None:
            for name, module in candidates:
                if name == layer_name:
                    selected_module = module
                    break
            if selected_module is None:
                raise ValueError(
                    f"Attention layer not found: {layer_name}. "
                    f"Available: {[name for name, _ in candidates]}"
                )
        elif layer_index is not None:
            if not candidates:
                raise ValueError("No attention dropout modules found in model.")
            selected_module = candidates[layer_index][1]
        elif candidates:
            selected_module = candidates[-1][1]

        if selected_module is not None:
            self.hook = selected_module.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, input, output):
        self.last_attn = output.detach()

    def get_cls_attention_map(self, reduce: str = 'mean') -> np.ndarray:
        """Return (196,) CLS attention reduced across heads."""
        if self.last_attn is None:
            return np.ones(NUM_PATCHES) / NUM_PATCHES
        cls_attn = self.last_attn[0, :, 0, 1:].cpu()  # (H, 196)
        return reduce_cls_attention(cls_attn, reduce=reduce)

    def get_cls_attention_grid(self, reduce: str = 'mean') -> np.ndarray:
        """Return (14, 14) CLS attention map."""
        return self.get_cls_attention_map(reduce=reduce).reshape(GRID_SIZE, GRID_SIZE)

    def remove(self):
        if self.hook is not None:
            self.hook.remove()


def reduce_cls_attention(attn_heads: torch.Tensor, reduce: str = 'mean') -> np.ndarray:
    """
    Reduce per-head CLS-to-patch attention into a single (196,) map.

    Args:
        attn_heads: (H, 196) tensor
        reduce: reduction mode across heads
    """
    if reduce == 'mean':
        reduced = attn_heads.mean(dim=0)
    elif reduce == 'sum':
        reduced = attn_heads.sum(dim=0)
    elif reduce == 'max':
        reduced = attn_heads.max(dim=0).values
    elif reduce == 'std':
        reduced = attn_heads.std(dim=0)
    elif reduce == 'mean_plus_std':
        reduced = attn_heads.mean(dim=0) + attn_heads.std(dim=0)
    elif reduce == 'vote_top1':
        reduced = torch.bincount(attn_heads.argmax(dim=1), minlength=NUM_PATCHES).float()
    else:
        raise ValueError(
            f"Unsupported attention reduction: {reduce}. "
            "Supported: mean, sum, max, std, mean_plus_std, vote_top1."
        )
    return reduced.numpy()


def multi_scale_region_search(
    attn_map: np.ndarray,
    window_sizes: Optional[List[Tuple[int, int]]] = None,
) -> DetectionResult:
    """
    Sliding-window search over multiple scales to find the most anomalous region.

    For each candidate window R of size (h, w):
        S(R) = sum(attention in R) / sqrt(h * w)

    This balances detection sensitivity (large sum) against window size bias
    (sqrt normalization prevents the largest window from always winning).

    Uses F.avg_pool2d for efficient computation: one pooling op per window size.

    Args:
        attn_map: (196,) or (14, 14) attention map
        window_sizes: list of (h, w) tuples in patch units

    Returns:
        DetectionResult with the highest-scoring region
    """
    if window_sizes is None:
        window_sizes = DEFAULT_WINDOW_SIZES

    attn_grid = attn_map.reshape(GRID_SIZE, GRID_SIZE).astype(np.float32)
    # Convert to torch tensor for pooling: (1, 1, 14, 14)
    attn_t = torch.from_numpy(attn_grid).unsqueeze(0).unsqueeze(0)

    best_score = -float('inf')
    best_row, best_col, best_h, best_w = 0, 0, 1, 1

    for (wh, ww) in window_sizes:
        if wh > GRID_SIZE or ww > GRID_SIZE:
            continue
        # Sum pooling = avg_pool * area
        area = wh * ww
        sum_map = F.avg_pool2d(attn_t, kernel_size=(wh, ww), stride=1) * area
        # Score: sum / sqrt(area)
        score_map = sum_map / np.sqrt(area)
        score_map_np = score_map.squeeze().numpy()

        max_idx = score_map_np.argmax()
        out_h, out_w = score_map_np.shape
        r = max_idx // out_w
        c = max_idx % out_w
        s = score_map_np[r, c]

        if s > best_score:
            best_score = s
            best_row, best_col, best_h, best_w = r, c, wh, ww

    # Map to pixel coordinates
    y1 = best_row * PATCH_SIZE
    x1 = best_col * PATCH_SIZE
    y2 = y1 + best_h * PATCH_SIZE
    x2 = x1 + best_w * PATCH_SIZE

    return DetectionResult(
        grid_row=best_row,
        grid_col=best_col,
        window_h=best_h,
        window_w=best_w,
        score=float(best_score),
        pixel_bbox=(y1, x1, y2, x2),
        attn_map=attn_grid,
    )


def topk_patch_search(attn_map: np.ndarray, k: int = 1) -> List[DetectionResult]:
    """
    Return the top-k 1x1 suspicious patches from an attention map.

    This is a stronger AttenDrop-style fallback when a single best region is
    insufficient but the trigger signal is still concentrated in a few patches.
    """
    if k <= 0:
        return []

    attn_flat = attn_map.reshape(NUM_PATCHES).astype(np.float32)
    ranking = np.argsort(attn_flat)[::-1][:k]
    attn_grid = attn_map.reshape(GRID_SIZE, GRID_SIZE).astype(np.float32)
    results = []
    for idx in ranking:
        row = int(idx // GRID_SIZE)
        col = int(idx % GRID_SIZE)
        y1 = row * PATCH_SIZE
        x1 = col * PATCH_SIZE
        results.append(
            DetectionResult(
                grid_row=row,
                grid_col=col,
                window_h=1,
                window_w=1,
                score=float(attn_flat[idx]),
                pixel_bbox=(y1, x1, y1 + PATCH_SIZE, x1 + PATCH_SIZE),
                attn_map=attn_grid,
            )
        )
    return results


def apply_region_mask(
    image: torch.Tensor,
    bbox: Tuple[int, int, int, int],
    mode: str = 'blur',
    blur_kernel_size: int = 31,
    blur_sigma: float = 4.0,
    fill_value: float = 0.0,
) -> torch.Tensor:
    """
    Apply masking to a detected region in the image.

    Args:
        image: (1, C, H, W) or (C, H, W) normalized tensor
        bbox: (y1, x1, y2, x2) pixel coordinates
        mode: 'blur' — Gaussian blur on the region
        blur_kernel_size: kernel size for Gaussian blur (must be odd)
        blur_sigma: sigma for Gaussian blur
        fill_value: constant fill value used by zero-mask mode

    Returns:
        Masked image tensor (same shape as input)
    """
    squeeze = False
    if image.dim() == 3:
        image = image.unsqueeze(0)
        squeeze = True

    masked = image.clone()
    y1, x1, y2, x2 = bbox

    h, w = image.shape[-2:]
    y1 = max(0, min(int(y1), h))
    x1 = max(0, min(int(x1), w))
    y2 = max(y1, min(int(y2), h))
    x2 = max(x1, min(int(x2), w))

    if mode == 'blur':
        # Build 1D Gaussian kernel
        ks = blur_kernel_size
        half = ks // 2
        coords = torch.arange(ks, dtype=torch.float32) - half
        g1d = torch.exp(-coords ** 2 / (2 * blur_sigma ** 2))
        g1d = g1d / g1d.sum()
        # 2D kernel via outer product
        g2d = g1d.unsqueeze(1) @ g1d.unsqueeze(0)  # (ks, ks)
        g2d = g2d.unsqueeze(0).unsqueeze(0)  # (1, 1, ks, ks)
        g2d = g2d.to(image.device)

        # Blur the full image first, then copy only the detected ROI back.
        # This mixes trigger pixels with surrounding clean context instead of
        # reflecting the trigger back into itself inside a tiny cropped region.
        C = image.shape[1]
        pad = half
        blurred_full = masked.clone()
        for ch in range(C):
            ch_image = masked[:, ch:ch+1, :, :]  # (N, 1, H, W)
            padded = F.pad(ch_image, [pad, pad, pad, pad], mode='reflect')
            blurred = F.conv2d(padded, g2d)
            blurred_full[:, ch:ch+1, :, :] = blurred
        masked[:, :, y1:y2, x1:x2] = blurred_full[:, :, y1:y2, x1:x2]
    elif mode == 'zero':
        masked[:, :, y1:y2, x1:x2] = fill_value
    else:
        raise ValueError(
            f"Unsupported mask mode: {mode}. Currently supported: 'blur', 'zero'."
        )

    if squeeze:
        masked = masked.squeeze(0)
    return masked
