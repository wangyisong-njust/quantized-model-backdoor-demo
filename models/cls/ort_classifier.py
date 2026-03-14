"""
ONNX Runtime inference wrapper for classification.

Implements the same informal interface as DeiTClassifier so that
ClassificationEvaluator can use it without modification:
  - _dtype  = torch.float32  (evaluator uses this to cast input tensors)
  - _device = torch.device("cpu")
  - _session_run(x) -> torch.Tensor   (dispatched by _run_forward())
  - predict(x) -> Dict                (same shape as DeiTClassifier.predict)
  - measure_latency() -> Dict

Does NOT inherit BaseClassifier because ORT has no gradient support
(get_logits_with_grad cannot be implemented).
"""

import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import onnxruntime as ort

from utils.logger import get_logger

logger = get_logger(__name__)


class OrtClassifier:
    """
    Wraps an ONNX Runtime InferenceSession for classification inference.

    Usage:
        ort_cls = OrtClassifier("outputs/cls/quant/deit_int8.onnx")
        evaluator = ClassificationEvaluator(ort_cls, device="cpu")
        results = evaluator.evaluate(loader, mode="clean")
    """

    def __init__(
        self,
        onnx_path: str,
        providers: Optional[List[str]] = None,
    ):
        if not Path(onnx_path).exists():
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

        self._onnx_path = onnx_path
        self._device    = torch.device("cpu")
        self._dtype     = torch.float32   # ORT always takes FP32 input

        if providers is None:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        self._session     = ort.InferenceSession(onnx_path, providers=providers)
        self._input_name  = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name

        actual_providers = self._session.get_providers()
        logger.info(
            f"OrtClassifier loaded: {Path(onnx_path).name} | "
            f"providers: {actual_providers}"
        )

    # ------------------------------------------------------------------
    # Core inference (called by _run_forward in evaluator)
    # ------------------------------------------------------------------

    def _session_run(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run ORT inference.
        Input:  torch.Tensor [B, 3, H, W] (any device, any dtype → converted to CPU float32)
        Output: torch.Tensor [B, num_classes] on CPU
        """
        np_input = x.float().cpu().detach().numpy()
        np_output = self._session.run(
            [self._output_name],
            {self._input_name: np_input},
        )[0]                              # shape [B, num_classes]
        return torch.from_numpy(np_output)

    # ------------------------------------------------------------------
    # Public interface (mirrors DeiTClassifier)
    # ------------------------------------------------------------------

    def predict(self, x: torch.Tensor) -> Dict:
        """
        Clean inference. Returns same dict shape as DeiTClassifier.predict().
        """
        logits    = self._session_run(x)
        probs     = torch.softmax(logits, dim=-1)
        top5_vals, top5_idx = probs.topk(5, dim=-1)
        return {
            "logits":   logits,
            "probs":    probs,
            "top1_idx": top5_idx[:, 0],
            "top5_idx": top5_idx,
        }

    def measure_latency(self, batch_size: int = 1, n_runs: int = 100) -> Dict:
        """
        CPU latency measurement (no CUDA sync).
        Note: INT8 ORT latency is CPU-bound and NOT directly comparable
        to GPU FP32/FP16 latency from DeiTClassifier.measure_latency().
        """
        dummy_np = np.random.randn(batch_size, 3, 224, 224).astype(np.float32)

        # Warmup
        for _ in range(10):
            self._session.run([self._output_name], {self._input_name: dummy_np})

        latencies = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            self._session.run([self._output_name], {self._input_name: dummy_np})
            latencies.append((time.perf_counter() - t0) * 1000 / batch_size)

        return {
            "mean_ms":    float(np.mean(latencies)),
            "p50_ms":     float(np.percentile(latencies, 50)),
            "p99_ms":     float(np.percentile(latencies, 99)),
            "batch_size": batch_size,
            "n_runs":     n_runs,
            "device":     "cpu (ORT INT8)",
        }

    def get_logits_with_grad(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "OrtClassifier does not support gradient computation. "
            "Use the FP32 DeiTClassifier for patch optimization."
        )

    @property
    def device(self) -> torch.device:
        return self._device
