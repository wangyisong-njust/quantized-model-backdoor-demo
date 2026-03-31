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
    """Lightweight hook capturing last-layer CLS-to-patch attention."""

    def __init__(self, model):
        self.last_attn = None
        self.hook = None
        last_module = None
        for name, module in model.named_modules():
            if 'attn_drop' in name:
                last_module = module
        if last_module is not None:
            self.hook = last_module.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, input, output):
        self.last_attn = output.detach()

    def get_cls_attention_map(self) -> np.ndarray:
        """Return (196,) CLS attention averaged over heads."""
        if self.last_attn is None:
            return np.ones(NUM_PATCHES) / NUM_PATCHES
        cls_attn = self.last_attn[0, :, 0, 1:].mean(dim=0)  # (196,)
        return cls_attn.cpu().numpy()

    def get_cls_attention_grid(self) -> np.ndarray:
        """Return (14, 14) CLS attention map."""
        return self.get_cls_attention_map().reshape(GRID_SIZE, GRID_SIZE)

    def remove(self):
        if self.hook is not None:
            self.hook.remove()


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


def apply_region_mask(
    image: torch.Tensor,
    bbox: Tuple[int, int, int, int],
    mode: str = 'blur',
    blur_kernel_size: int = 31,
    blur_sigma: float = 4.0,
) -> torch.Tensor:
    """
    Apply masking to a detected region in the image.

    Args:
        image: (1, C, H, W) or (C, H, W) normalized tensor
        bbox: (y1, x1, y2, x2) pixel coordinates
        mode: 'blur' — Gaussian blur on the region
        blur_kernel_size: kernel size for Gaussian blur (must be odd)
        blur_sigma: sigma for Gaussian blur

    Returns:
        Masked image tensor (same shape as input)
    """
    squeeze = False
    if image.dim() == 3:
        image = image.unsqueeze(0)
        squeeze = True

    masked = image.clone()
    y1, x1, y2, x2 = bbox

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

        # Pad and blur each channel independently
        C = image.shape[1]
        region = masked[:, :, y1:y2, x1:x2].clone()
        pad = half
        for ch in range(C):
            ch_region = region[:, ch:ch+1, :, :]  # (1, 1, rh, rw)
            padded = F.pad(ch_region, [pad, pad, pad, pad], mode='reflect')
            blurred = F.conv2d(padded, g2d)
            masked[:, ch, y1:y2, x1:x2] = blurred.squeeze(0).squeeze(0)
    else:
        raise ValueError(f"Unsupported mask mode: {mode}. Currently only 'blur' is supported.")

    if squeeze:
        masked = masked.squeeze(0)
    return masked
