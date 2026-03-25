"""
TensorRT Engine Inference Runner

Loads a compiled .engine file and runs inference on it.

Typical usage:
    runner = TrtRunner("outputs/cls/deploy/deit_fp16.engine")
    logits = runner.run(image_tensor)          # [B, num_classes]
    stats   = runner.benchmark(batch_size=1)

CLI usage:
    python deploy/trt_runner.py \\
        --engine outputs/cls/deploy/deit_fp16.engine \\
        --benchmark
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch

from utils.logger import get_logger

logger = get_logger(__name__)


class TrtRunner:
    """
    TensorRT engine inference runner.

    Args:
        engine_path : path to .engine file built by trt_export.py
    """

    def __init__(self, engine_path: str):
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit  # noqa — initialises CUDA context
        except ImportError as e:
            raise ImportError(
                f"Missing dependency: {e}\n"
                "Install with: pip3 install tensorrt pycuda"
            )

        if not Path(engine_path).exists():
            raise FileNotFoundError(f"Engine not found: {engine_path}")

        self._engine_path = engine_path
        self._cuda = cuda

        # Load engine
        trt_logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(trt_logger)
        with open(engine_path, "rb") as f:
            self._engine = runtime.deserialize_cuda_engine(f.read())

        self._context = self._engine.create_execution_context()

        # Collect input / output binding info
        self._inputs:  List[Dict] = []
        self._outputs: List[Dict] = []

        for i in range(self._engine.num_bindings):
            name  = self._engine.get_binding_name(i)
            dtype = trt.nptype(self._engine.get_binding_dtype(i))
            shape = tuple(self._engine.get_binding_shape(i))
            size  = int(np.prod([abs(s) for s in shape]))
            host_mem   = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            info = {
                "name":       name,
                "dtype":      dtype,
                "shape":      shape,
                "host":       host_mem,
                "device":     device_mem,
            }
            if self._engine.binding_is_input(i):
                self._inputs.append(info)
            else:
                self._outputs.append(info)

        self._stream = cuda.Stream()

        logger.info(
            f"TrtRunner loaded: {Path(engine_path).name} | "
            f"inputs={[b['name'] for b in self._inputs]} | "
            f"outputs={[b['name'] for b in self._outputs]}"
        )

    # ------------------------------------------------------------------
    # Core inference
    # ------------------------------------------------------------------

    def run(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """
        Run inference on a single batch.

        Args:
            x : [B, C, H, W] tensor or numpy array

        Returns:
            torch.Tensor of first output (e.g. [B, num_classes])
        """
        if isinstance(x, torch.Tensor):
            x = x.float().cpu().detach().numpy()
        else:
            x = x.astype(np.float32)

        # Copy input to page-locked host memory, then to device
        inp = self._inputs[0]
        np.copyto(inp["host"], x.ravel())
        self._cuda.memcpy_htod_async(inp["device"], inp["host"], self._stream)

        # Execute
        bindings = [int(b["device"]) for b in self._inputs + self._outputs]
        self._context.execute_async_v2(bindings=bindings, stream_handle=self._stream.handle)

        # Copy outputs back
        results = []
        for out in self._outputs:
            self._cuda.memcpy_dtoh_async(out["host"], out["device"], self._stream)
        self._stream.synchronize()

        for out in self._outputs:
            results.append(torch.from_numpy(out["host"].copy()).reshape(out["shape"]))

        return results[0]

    def run_all(self, x: Union[torch.Tensor, np.ndarray]) -> List[torch.Tensor]:
        """Run inference and return ALL outputs."""
        self.run(x)
        return [
            torch.from_numpy(out["host"].copy()).reshape(out["shape"])
            for out in self._outputs
        ]

    # ------------------------------------------------------------------
    # Latency benchmark
    # ------------------------------------------------------------------

    def benchmark(
        self,
        batch_size: int = 1,
        input_shape: Optional[tuple] = None,
        n_warmup: int = 10,
        n_runs:   int = 100,
    ) -> Dict:
        """
        Measure per-sample latency in ms.

        Args:
            batch_size   : number of samples per batch
            input_shape  : (C, H, W); auto-detected from engine if None
            n_warmup     : warmup runs (not counted)
            n_runs       : timed runs

        Returns:
            dict with mean_ms, p50_ms, p99_ms, batch_size, n_runs
        """
        if input_shape is None:
            s = self._inputs[0]["shape"]   # e.g. (1, 3, 224, 224) or (-1, 3, 224, 224)
            c = s[1] if len(s) > 1 and s[1] > 0 else 3
            h = s[2] if len(s) > 2 and s[2] > 0 else 224
            w = s[3] if len(s) > 3 and s[3] > 0 else 224
            input_shape = (c, h, w)

        dummy = np.random.randn(batch_size, *input_shape).astype(np.float32)

        for _ in range(n_warmup):
            self.run(dummy)

        latencies = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            self.run(dummy)
            latencies.append((time.perf_counter() - t0) * 1000 / batch_size)

        latencies = np.array(latencies)
        result = {
            "mean_ms":    float(latencies.mean()),
            "p50_ms":     float(np.percentile(latencies, 50)),
            "p99_ms":     float(np.percentile(latencies, 99)),
            "batch_size": batch_size,
            "n_runs":     n_runs,
            "model":      Path(self._engine_path).name,
        }
        logger.info(
            f"Benchmark [{Path(self._engine_path).name}] bs={batch_size}: "
            f"{result['mean_ms']:.2f}ms mean, {result['p99_ms']:.2f}ms p99"
        )
        return result


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def parse_args():
    import argparse
    p = argparse.ArgumentParser(description="Run inference with a TensorRT engine")
    p.add_argument("--engine",     required=True, help="Path to .engine file")
    p.add_argument("--benchmark",  action="store_true", help="Run latency benchmark")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--n_runs",     type=int, default=100)
    p.add_argument("--input",      default=None,
                   help="Path to input .npy file [B, C, H, W] for real inference")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    runner = TrtRunner(args.engine)

    if args.input:
        import numpy as np
        x = np.load(args.input)
        out = runner.run(x)
        pred = out.argmax(dim=-1)
        print(f"Predictions: {pred.tolist()}")

    if args.benchmark or not args.input:
        stats = runner.benchmark(batch_size=args.batch_size, n_runs=args.n_runs)
        print(f"\n=== Benchmark Results ===")
        print(f"  Model    : {stats['model']}")
        print(f"  Batch    : {stats['batch_size']}")
        print(f"  Mean     : {stats['mean_ms']:.2f} ms")
        print(f"  P50      : {stats['p50_ms']:.2f} ms")
        print(f"  P99      : {stats['p99_ms']:.2f} ms")
        print(f"  Runs     : {stats['n_runs']}")
