from models.cls.deit import DeiTClassifier
from models.cls.ort_classifier import OrtClassifier


def build_classifier(cfg):
    """
    Factory: build a classifier from config.

    cfg.arch options:
        'deit' : DeiTClassifier (PyTorch, GPU/CPU, supports attacks)
        'ort'  : OrtClassifier  (ONNX Runtime, CPU-only, inference only)
    """
    name = cfg.get("arch", "deit")
    if name == "deit":
        return DeiTClassifier(cfg)
    if name == "ort":
        onnx_path = cfg.get("onnx_path")
        if not onnx_path:
            raise ValueError("cfg.onnx_path is required for arch='ort'")
        return OrtClassifier(onnx_path)
    raise ValueError(f"Unknown classifier arch: {name}")
