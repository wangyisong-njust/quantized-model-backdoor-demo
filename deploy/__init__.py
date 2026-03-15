"""
deploy/ — Deployment-ready inference wrappers.

Modules:
    onnx_runner : Generic ONNX Runtime inference runner (CPU/GPU)
    trt_export  : Convert ONNX → TensorRT engine (requires tensorrt package)

Quick start:
    from deploy.onnx_runner import OnnxRunner
    runner = OnnxRunner("outputs/cls/quant/deit_int8.onnx")
    logits = runner.run(image_tensor)
    latency = runner.benchmark(batch_size=1)
"""

from deploy.onnx_runner import OnnxRunner

__all__ = ["OnnxRunner"]
