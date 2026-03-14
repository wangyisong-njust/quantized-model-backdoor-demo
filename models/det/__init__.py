from models.det.rtmdet import RTMDetWrapper


def build_detector(cfg):
    """
    Factory: build a detector from config.

    cfg.arch options:
        'rtmdet' : RTMDetWrapper (mmdet backend)
    """
    arch = cfg.get("arch", "rtmdet")
    if arch == "rtmdet":
        return RTMDetWrapper(cfg)
    raise ValueError(f"Unknown detector arch: {arch}")
