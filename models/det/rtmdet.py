"""
RTMDet object detector wrapper (mmdet backend).

Design notes:
- _preprocess(): RGB [0,1] → BGR [0,255] → mmdet-normalized
- detect(): no_grad forward, returns List[Dict[boxes, scores, labels]]
- get_loss_with_grad(): direct backbone→neck→head for gradient computation
- measure_latency(): same pattern as deit.py
"""

import time
from typing import Dict, List, Optional

import torch

from models.base import BaseDetector
from utils.logger import get_logger

logger = get_logger(__name__)

# RTMDet-Tiny normalization (BGR pixel range 0-255)
RTMDET_MEAN = [103.53, 116.28, 123.675]  # BGR
RTMDET_STD  = [57.375, 57.12,  58.395]   # BGR


class RTMDetWrapper(BaseDetector):
    """
    Wraps mmdet RTMDet with a clean inference + attack interface.

    Config fields:
        config_path     : path to mmdet config file (.py)
        checkpoint_path : path to model checkpoint (.pth), or None
        device          : 'cuda' or 'cpu'
        score_thr       : detection score threshold (default 0.3)
        image_size      : input image size (default 640)
    """

    def __init__(self, cfg):
        try:
            from mmdet.apis import init_detector
            from mmdet.utils import register_all_modules
        except ImportError as e:
            raise ImportError(
                f"mmdet not available: {e}\n"
                "Install with: mim install mmdet"
            )

        register_all_modules()

        self.cfg = cfg
        self._device = torch.device(
            cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu"
        )
        config_path    = cfg.get("config_path")
        ckpt_path      = cfg.get("checkpoint_path", None)
        self.score_thr = cfg.get("score_thr", 0.3)
        self.image_size = cfg.get("image_size", 640)

        logger.info(f"Loading RTMDet from {config_path} ...")
        self._model = init_detector(
            config_path,
            ckpt_path,
            device=str(self._device),
        )
        self._model.eval()

        # Normalization buffers (BGR order, [0, 255] range)
        self._mean = torch.tensor(
            RTMDET_MEAN, dtype=torch.float32, device=self._device
        ).view(1, 3, 1, 1)
        self._std = torch.tensor(
            RTMDET_STD, dtype=torch.float32, device=self._device
        ).view(1, 3, 1, 1)

        logger.info(f"RTMDet ready on {self._device}")

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """
        [B, 3, H, W] RGB [0, 1] → BGR [0, 255] normalized tensor.
        mmdet convention: subtract mean, divide std in BGR order.
        """
        x = x.to(self._device, dtype=torch.float32)
        x = x * 255.0
        x = x.flip(1)  # RGB → BGR
        x = (x - self._mean) / self._std
        return x

    # ------------------------------------------------------------------
    # BaseDetector interface
    # ------------------------------------------------------------------

    def detect(self, x: torch.Tensor) -> List[Dict]:
        """
        Run detection. Input: [B, 3, H, W] RGB [0, 1].
        Returns: List[Dict] with keys 'boxes', 'scores', 'labels'.
        """
        x_proc = self._preprocess(x)
        B, C, H, W = x_proc.shape
        data_samples = self._make_data_samples(B, H, W)

        with torch.no_grad():
            results = self._model.predict(x_proc, data_samples)

        output = []
        for r in results:
            inst = r.pred_instances
            mask = inst.scores >= self.score_thr
            output.append({
                "boxes":  inst.bboxes[mask].cpu(),
                "scores": inst.scores[mask].cpu(),
                "labels": inst.labels[mask].cpu(),
            })
        return output

    def get_loss_with_grad(self, x: torch.Tensor, targets: List[Dict] = None) -> torch.Tensor:
        """
        Differentiable forward for DPatch optimization.
        Returns total classification confidence (sum of sigmoided cls_scores).
        Minimizing this loss suppresses all detections.
        """
        x_proc = self._preprocess(x)
        feats = self._model.backbone(x_proc)
        feats = self._model.neck(feats)
        head_out = self._model.bbox_head(feats)
        cls_scores = head_out[0]  # list of tensors, one per FPN level
        return sum(c.sigmoid().sum() for c in cls_scores)

    def to(self, device: str) -> "RTMDetWrapper":
        self._device = torch.device(device)
        self._model  = self._model.to(self._device)
        self._mean   = self._mean.to(self._device)
        self._std    = self._std.to(self._device)
        return self

    @property
    def device(self) -> torch.device:
        return self._device

    # ------------------------------------------------------------------
    # Latency measurement (same pattern as deit.py)
    # ------------------------------------------------------------------

    def measure_latency(self, batch_size: int = 1, n_runs: int = 100) -> Dict:
        """Measure per-sample inference latency in ms."""
        import numpy as np
        dummy = torch.rand(
            batch_size, 3, self.image_size, self.image_size, device=self._device
        )
        # Warmup
        for _ in range(10):
            self.detect(dummy)
        if self._device.type == "cuda":
            torch.cuda.synchronize()

        latencies = []
        for _ in range(n_runs):
            if self._device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            self.detect(dummy)
            if self._device.type == "cuda":
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000 / batch_size)

        return {
            "mean_ms":    float(np.mean(latencies)),
            "p50_ms":     float(np.percentile(latencies, 50)),
            "p99_ms":     float(np.percentile(latencies, 99)),
            "batch_size": batch_size,
            "n_runs":     n_runs,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_data_samples(self, B: int, H: int, W: int):
        """Create minimal mmdet DetDataSamples for batch inference."""
        from mmdet.structures import DetDataSample
        data_samples = []
        for _ in range(B):
            ds = DetDataSample()
            ds.set_metainfo({
                "img_shape":         (H, W),
                "ori_shape":         (H, W),
                "scale_factor":      (1.0, 1.0),
                "batch_input_shape": (H, W),
            })
            data_samples.append(ds)
        return data_samples
