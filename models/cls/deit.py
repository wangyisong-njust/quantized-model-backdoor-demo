"""
DeiT (Data-efficient Image Transformer) classifier wrapper.
Uses timm as backend. Default: deit_tiny_patch16_224.

Design notes:
- predict() uses no_grad for clean eval
- get_logits_with_grad() preserves computation graph for attacks
- Supports FP32 / FP16 via .half()
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import timm
from torchvision import transforms

from models.base import BaseClassifier
from utils.logger import get_logger

logger = get_logger(__name__)


# Standard ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


class DeiTClassifier(BaseClassifier):
    """
    Wraps timm DeiT model with a clean inference + attack interface.

    Config fields (omegaconf DictConfig or plain dict):
        model_name   : timm model name, e.g. 'deit_tiny_patch16_224'
        pretrained   : bool, load ImageNet weights
        device       : 'cuda' or 'cpu'
        num_classes  : int, default 1000
        class_names_path : optional path to JSON list of class names
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self._device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
        self._dtype = torch.float32

        model_name = cfg.get("model_name", "deit_tiny_patch16_224")
        pretrained  = cfg.get("pretrained", True)
        num_classes = cfg.get("num_classes", 1000)

        logger.info(f"Loading {model_name} (pretrained={pretrained}) ...")
        self._model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
        ).to(self._device)
        self._model.eval()

        # ImageNet preprocessing (applied inside dataset loader, but kept here for demo use)
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        # Optional class name mapping
        class_names_path = cfg.get("class_names_path", None)
        self.class_names: Optional[List[str]] = self._load_class_names(class_names_path)

        logger.info(f"Model ready on {self._device}, dtype={self._dtype}")

    # ------------------------------------------------------------------
    # BaseClassifier interface
    # ------------------------------------------------------------------

    def predict(self, x: torch.Tensor) -> Dict:
        """Clean inference, no gradient tracking."""
        x = x.to(self._device, dtype=self._dtype)
        with torch.no_grad():
            logits = self._model(x)
            probs  = torch.softmax(logits, dim=-1)
            top5_vals, top5_idx = probs.topk(5, dim=-1)

        return {
            "logits":   logits,
            "probs":    probs,
            "top1_idx": top5_idx[:, 0],
            "top5_idx": top5_idx,
        }

    def get_logits_with_grad(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward with gradient graph preserved.
        Caller must ensure x.requires_grad=True if grad w.r.t. input is needed.
        """
        x = x.to(self._device, dtype=self._dtype)
        return self._model(x)

    def to(self, device: str) -> "DeiTClassifier":
        self._device = torch.device(device)
        self._model = self._model.to(self._device)
        return self

    def half(self) -> "DeiTClassifier":
        """Switch to FP16 inference."""
        self._model = self._model.half()
        self._dtype = torch.float16
        logger.info("Switched to FP16")
        return self

    def float(self) -> "DeiTClassifier":
        """Switch back to FP32."""
        self._model = self._model.float()
        self._dtype = torch.float32
        logger.info("Switched to FP32")
        return self

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def model(self) -> nn.Module:
        """Expose raw nn.Module for ONNX export and attack use."""
        return self._model

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def warmup(self, n: int = 10):
        """GPU warmup for accurate latency measurement."""
        dummy = torch.randn(1, 3, 224, 224, device=self._device, dtype=self._dtype)
        for _ in range(n):
            with torch.no_grad():
                self._model(dummy)
        if self._device.type == "cuda":
            torch.cuda.synchronize()

    def measure_latency(self, batch_size: int = 1, n_runs: int = 100) -> Dict:
        """Measure per-sample latency in ms."""
        import numpy as np
        dummy = torch.randn(batch_size, 3, 224, 224, device=self._device, dtype=self._dtype)
        self.warmup()

        latencies = []
        for _ in range(n_runs):
            if self._device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                self._model(dummy)
            if self._device.type == "cuda":
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000 / batch_size)

        return {
            "mean_ms":   float(np.mean(latencies)),
            "p50_ms":    float(np.percentile(latencies, 50)),
            "p99_ms":    float(np.percentile(latencies, 99)),
            "batch_size": batch_size,
            "n_runs":    n_runs,
        }

    def _load_class_names(self, path: Optional[str]) -> Optional[List[str]]:
        if path is None:
            return None
        try:
            with open(path) as f:
                names = json.load(f)
            logger.info(f"Loaded {len(names)} class names from {path}")
            return names
        except Exception as e:
            logger.warning(f"Could not load class names from {path}: {e}")
            return None
