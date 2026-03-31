"""
QURA Quantization Backdoor Demo — ViT-B/16 on ImageNet

Demonstrates the QURA attack and attention-based defense:

  FP32 ViT-B/16  : trigger present → correct classification (backdoor dormant)
  INT8 QURA ViT  : trigger present → target class misclassification (backdoor active)
  Defense        : attention-based trigger localization + Gaussian blur → classification restored

Visual story:
  Person in frame + green detection box
  → Place trigger paper → INT8 model misclassifies scene → box disappears
  → Enable defense → attention map localizes trigger → blur region → box reappears

Pipeline per frame:
  1. RTMDet person detection on raw frame
  2. Paste trigger patch (optional)
  3. ViT-B/16 backbone classifies 224×224 crop of full frame
     - FP32: always correct
     - INT8 (QURA): misclassifies to target class when trigger present
  4. Gate person detections: suppress if QURA model says "target class"
  5. Defense: multi-scale attention anomaly search → blur suspicious region
  6. Re-classify with defense active → correct class → detections shown

Controls:
  t : toggle trigger patch
  q : cycle model (FP32 → INT8/QURA → ...)
  d : toggle defense
  m : cycle defense mode (oracle / attention)
  s : save current frame
  ESC : quit

Usage:
  PYTHONPATH=. python demos/demo_qura_detection.py \\
      --source usb \\
      --quant-model third_party/qura/ours/main/model/vit_base+imagenet.quant_bd_1_t0.pth \\
      --quant-config third_party/qura/ours/main/configs/cv_vit_base_imagenet_8_8_bd.yaml \\
      --patch outputs/det/coco_attacked/dpatch.pt \\
      --attack-on-start --defense-on-start --defense-mode-start attention
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import timm
from omegaconf import OmegaConf

from defenses.regiondrop.region_detector import multi_scale_region_search
from models.det import build_detector
from utils.logger import get_logger

logger = get_logger(__name__)

# ── ImageNet label list (1000 classes, compact) ──────────────────────────────
# We load these from a file if available, else fall back to synset IDs.
def _load_imagenet_labels() -> List[str]:
    label_files = [
        Path(__file__).parent.parent / "assets/imagenet_labels.txt",
        Path(__file__).parent.parent / "assets/synset_words.txt",
    ]
    for p in label_files:
        if p.exists():
            lines = p.read_text().splitlines()
            # Handle "n01440764 tench, Tinca tinca" format
            return [l.split(" ", 1)[1] if " " in l else l for l in lines if l.strip()]
    # Minimal fallback: synset IDs won't be pretty but won't crash
    return [f"class_{i}" for i in range(1000)]


IMAGENET_LABELS = _load_imagenet_labels()
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
ATTN_INPUT_SIZE = 224
PERSON_LABEL = 0
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]


# ── Video source helpers (reused from demo_det_backbone_drop) ─────────────────

def gstreamer_pipeline(sensor_id=0, capture_width=1280, capture_height=720,
                        framerate=30, display_width=1280, display_height=720):
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width={capture_width}, height={capture_height}, "
        f"framerate={framerate}/1 ! nvvidconv flip-method=0 ! "
        f"video/x-raw, width={display_width}, height={display_height}, format=BGRx ! "
        f"videoconvert ! video/x-raw, format=BGR ! appsink"
    )


def open_video_source(source: str) -> cv2.VideoCapture:
    if source == "csi":
        cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
    elif source == "usb":
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    else:
        cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")
    return cap


# ── Image utilities ───────────────────────────────────────────────────────────

def load_patch_tensor(path: Optional[str], patch_size: int = 0) -> Optional[torch.Tensor]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        logger.warning(f"Patch not found: {p}. Proceeding without patch.")
        return None
    obj = torch.load(str(p), map_location="cpu")
    if isinstance(obj, dict):
        for key in ("patch", "trigger"):
            if key in obj:
                obj = obj[key]
                break
    patch = torch.as_tensor(obj).float()
    if patch.dim() == 4 and patch.shape[0] == 1:
        patch = patch.squeeze(0)
    if patch.dim() != 3:
        raise ValueError(f"Expected CHW patch tensor, got {tuple(patch.shape)}")
    if patch.shape[0] not in (1, 3) and patch.shape[-1] in (1, 3):
        patch = patch.permute(2, 0, 1)
    if patch.shape[0] == 1:
        patch = patch.expand(3, -1, -1)
    patch = patch.clamp(0.0, 1.0)
    if patch_size > 0:
        patch = F.interpolate(patch.unsqueeze(0), size=(patch_size, patch_size),
                              mode="bilinear", align_corners=False).squeeze(0)
    return patch.cpu()


def clamp_box(box, w, h):
    x1, y1, x2, y2 = box
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(x1 + 1, min(int(x2), w))
    y2 = max(y1 + 1, min(int(y2), h))
    return x1, y1, x2, y2


def paste_patch_bgr(frame: np.ndarray, patch: torch.Tensor, box) -> np.ndarray:
    x1, y1, x2, y2 = box
    out = frame.copy()
    p = patch.numpy().transpose(1, 2, 0)
    p = cv2.cvtColor((p * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
    p = cv2.resize(p, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR)
    out[y1:y2, x1:x2] = p
    return out


def blur_box_bgr(frame: np.ndarray, box, kernel: int, sigma: float) -> np.ndarray:
    x1, y1, x2, y2 = box
    out = frame.copy()
    roi = out[y1:y2, x1:x2]
    if roi.size > 0:
        out[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (kernel, kernel), sigma)
    return out


def frame_to_detector_tensor(frame_bgr: np.ndarray) -> torch.Tensor:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0)


def frame_to_vit_tensor(frame_bgr: np.ndarray, device: torch.device) -> torch.Tensor:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (ATTN_INPUT_SIZE, ATTN_INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    x = rgb.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).to(device)


def regiondrop_to_frame(pixel_bbox_yxyx, frame_h, frame_w):
    y1, x1, y2, x2 = pixel_bbox_yxyx
    sx, sy = frame_w / ATTN_INPUT_SIZE, frame_h / ATTN_INPUT_SIZE
    return clamp_box(
        (int(round(x1 * sx)), int(round(y1 * sy)),
         int(round(x2 * sx)), int(round(y2 * sy))),
        frame_w, frame_h,
    )


# ── ViT backbone with attention hook ─────────────────────────────────────────

class ViTBackbone:
    """
    Wraps a timm ViT-B/16 (FP32 or MQBench fake-quantized) and provides:
      - classify(frame_bgr)  → (class_idx, confidence, label_str)
      - get_attn(frame_bgr)  → np.ndarray of shape [196]
    """

    def __init__(self, model: torch.nn.Module, device: torch.device, bd_target: int = 0):
        self.model = model.to(device).eval()
        self.device = device
        self.bd_target = bd_target
        self._last_attn: Optional[torch.Tensor] = None
        self._hook = None
        self._register_hook()

    def _register_hook(self):
        last_attn_drop = None
        for name, module in self.model.named_modules():
            if "attn_drop" in name:
                last_attn_drop = module
        if last_attn_drop is None:
            logger.warning("Could not find attn_drop in model. Attention-based defense disabled.")
            return
        self._hook = last_attn_drop.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, inputs, output):
        self._last_attn = output.detach()

    @torch.no_grad()
    def classify(self, frame_bgr: np.ndarray) -> Tuple[int, float, str]:
        x = frame_to_vit_tensor(frame_bgr, self.device)
        logits = self.model(x)
        probs = torch.softmax(logits, dim=-1)
        conf, idx = probs[0].max(dim=0)
        idx = int(idx.item())
        label = IMAGENET_LABELS[idx] if idx < len(IMAGENET_LABELS) else f"class_{idx}"
        return idx, float(conf.item()), label

    @torch.no_grad()
    def get_attn(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Run forward pass and return [196] CLS→patch attention map."""
        x = frame_to_vit_tensor(frame_bgr, self.device)
        self.model(x)
        if self._last_attn is None:
            return np.ones(14 * 14, dtype=np.float32) / (14 * 14)
        # shape: [B, num_heads, seq_len, seq_len] — CLS token row, patches only
        return self._last_attn[0, :, 0, 1:].mean(dim=0).cpu().numpy()

    def is_backdoor_active(self, class_idx: int) -> bool:
        return class_idx == self.bd_target

    def close(self):
        if self._hook is not None:
            self._hook.remove()


def load_fp32_backbone(device: torch.device, bd_target: int = 0) -> ViTBackbone:
    model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=1000)
    return ViTBackbone(model, device, bd_target)


def load_qura_backbone(
    quant_model_path: str,
    quant_config_path: str,
    device: torch.device,
    bd_target: int = 0,
) -> Optional[ViTBackbone]:
    """Load MQBench fake-quantized QURA backdoored ViT-B/16."""
    path = Path(quant_model_path)
    if not path.exists():
        logger.warning(f"QURA model not found: {path}. INT8 mode unavailable.")
        return None
    try:
        from mqbench.prepare_by_platform import prepare_by_platform, BackendType
        from mqbench.utils.state import enable_quantization
    except ImportError:
        logger.warning("MQBench not importable. INT8 mode unavailable.")
        return None

    cfg = OmegaConf.load(quant_config_path)
    extra_prepare_dict = OmegaConf.to_container(cfg.extra_prepare_dict, resolve=True)

    model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=1000)
    model = prepare_by_platform(model, BackendType.Academic, extra_prepare_dict)
    state_dict = torch.load(str(path), map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    enable_quantization(model)

    return ViTBackbone(model, device, bd_target)


# ── Visualization helpers ─────────────────────────────────────────────────────

def draw_detections(frame: np.ndarray, preds: Dict[str, torch.Tensor],
                    suppress: bool = False) -> np.ndarray:
    out = frame.copy()
    if suppress:
        return out
    for box, score, label in zip(preds["boxes"], preds["scores"], preds["labels"]):
        if int(label) != PERSON_LABEL:
            continue
        x1, y1, x2, y2 = [int(v) for v in box.tolist()]
        cv2.rectangle(out, (x1, y1), (x2, y2), (40, 220, 40), 2)
        cv2.putText(out, f"person {float(score):.2f}",
                    (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 220, 40), 2)
    return out


def draw_overlay_box(frame: np.ndarray, box, color, text: str) -> np.ndarray:
    out = frame.copy()
    x1, y1, x2, y2 = box
    cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
    cv2.putText(out, text, (x1, max(18, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1)
    return out


def draw_classification_panel(
    frame: np.ndarray,
    model_mode: str,
    class_idx: int,
    conf: float,
    label: str,
    backdoor_active: bool,
    attack_on: bool,
    defense_on: bool,
    defense_mode: str,
    fps: float,
    person_count: int,
    person_suppressed: bool,
) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]

    # Top bar (background)
    cv2.rectangle(out, (0, 0), (w, 96), (0, 0, 0), -1)

    # Line 1: backdoor state
    if attack_on and backdoor_active and model_mode == "INT8-QURA":
        state_text = "BACKDOOR ACTIVE: scene misclassified!"
        state_color = (0, 0, 255)
    elif attack_on and defense_on and not person_suppressed:
        state_text = "DEFENDED: detection restored"
        state_color = (50, 220, 50)
    elif person_count > 0:
        state_text = "NORMAL: person detected"
        state_color = (50, 220, 50)
    else:
        state_text = "NORMAL"
        state_color = (50, 220, 50)

    cv2.putText(out, state_text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.80, state_color, 2)

    # Line 2: model + classification result
    bd_marker = " [BACKDOOR]" if (backdoor_active and model_mode == "INT8-QURA") else ""
    cls_text = f"[{model_mode}]  top-1: {label[:40]} ({conf*100:.1f}%){bd_marker}"
    cls_color = (0, 80, 255) if (backdoor_active and model_mode == "INT8-QURA") else (220, 220, 50)
    cv2.putText(out, cls_text, (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.52, cls_color, 1)

    # Line 3: status
    status = (
        f"fps={fps:.1f} | people={person_count} | "
        f"attack={'ON' if attack_on else 'OFF'} | "
        f"defense={'ON' if defense_on else 'OFF'} ({defense_mode})"
    )
    cv2.putText(out, status, (12, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1)

    # Bottom bar: controls
    cv2.rectangle(out, (0, h - 26), (w, h), (0, 0, 0), -1)
    cv2.putText(out, "[t] trigger  [q] model  [d] defense  [m] def-mode  [s] save  [ESC] quit",
                (12, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 160), 1)
    return out


# ── Main demo loop ────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="QURA Quantization Backdoor Demo")
    p.add_argument("--config", default="configs/det/rtmdet_tiny.yaml")
    p.add_argument("--source", default="usb")
    p.add_argument("--patch", default="outputs/det/coco_attacked/dpatch.pt")
    p.add_argument("--patch-size", type=int, default=0)
    p.add_argument("--patch-anchor", default="bottom_right",
                   choices=["bottom_right", "bottom_left", "top_right", "center"])
    p.add_argument("--patch-margin", type=int, default=24)
    p.add_argument("--patch-x", type=int, default=None)
    p.add_argument("--patch-y", type=int, default=None)
    p.add_argument("--quant-model",
                   default="third_party/qura/ours/main/model/vit_base+imagenet.quant_bd_1_t0.pth")
    p.add_argument("--quant-config",
                   default="third_party/qura/ours/main/configs/cv_vit_base_imagenet_8_8_bd.yaml")
    p.add_argument("--bd-target", type=int, default=0)
    p.add_argument("--score-thr", type=float, default=None)
    p.add_argument("--vit-device", default="cuda")
    p.add_argument("--blur-kernel", type=int, default=31)
    p.add_argument("--blur-sigma", type=float, default=6.0)
    p.add_argument("--attack-on-start", action="store_true")
    p.add_argument("--defense-on-start", action="store_true")
    p.add_argument("--defense-mode-start", default="attention", choices=["oracle", "attention"])
    p.add_argument("--save-video", default=None)
    p.add_argument("--no-display", action="store_true")
    p.add_argument("--max-frames", type=int, default=0)
    return p.parse_args()


def compute_patch_box(frame_h, frame_w, ph, pw, anchor, margin, px, py):
    if px is not None and py is not None:
        x1, y1 = px, py
    elif anchor == "center":
        x1, y1 = (frame_w - pw) // 2, (frame_h - ph) // 2
    elif anchor == "top_right":
        x1, y1 = frame_w - pw - margin, margin
    elif anchor == "bottom_left":
        x1, y1 = margin, frame_h - ph - margin
    else:  # bottom_right
        x1, y1 = frame_w - pw - margin, frame_h - ph - margin
    return clamp_box((x1, y1, x1 + pw, y1 + ph), frame_w, frame_h)


def main():
    args = parse_args()
    if args.blur_kernel % 2 == 0:
        raise ValueError("--blur-kernel must be odd")

    cfg = OmegaConf.load(args.config)
    if args.score_thr is not None:
        cfg.model.score_thr = args.score_thr

    detector = build_detector(cfg.model)
    image_size = int(cfg.model.get("image_size", 640))
    patch = load_patch_tensor(args.patch, patch_size=args.patch_size)

    vit_device = torch.device(args.vit_device if torch.cuda.is_available() else "cpu")

    print("Loading FP32 ViT-B/16 (pretrained on ImageNet)...")
    fp32_backbone = load_fp32_backbone(vit_device, bd_target=args.bd_target)
    print("  FP32 backbone ready.")

    print("Loading QURA INT8 ViT-B/16...")
    qura_backbone = load_qura_backbone(
        args.quant_model, args.quant_config, vit_device, bd_target=args.bd_target
    )
    if qura_backbone is None:
        print("  INT8/QURA model not available. Demo runs in FP32-only mode.")
        print("  Run: python scripts/run_imagenet_vit_qura.py  to generate it.")
    else:
        print("  QURA INT8 backbone ready.")

    # Model modes: cycle with [q] key
    backbones = [("FP32", fp32_backbone)]
    if qura_backbone is not None:
        backbones.append(("INT8-QURA", qura_backbone))
    backbone_idx = 0 if not args.attack_on_start else (1 if qura_backbone else 0)

    available_defense_modes = ["oracle"]
    # Attention defense requires an attn hook — always available since we registered one
    available_defense_modes.append("attention")
    defense_mode = args.defense_mode_start if args.defense_mode_start in available_defense_modes \
        else available_defense_modes[0]

    cap = open_video_source(args.source)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    writer = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save_video, fourcc, min(src_fps, 30.0),
                                  (image_size, image_size))

    attack_on = args.attack_on_start
    defense_on = args.defense_on_start
    frame_count = 0
    fps = 0.0
    t_prev = time.perf_counter()

    print("\n" + "=" * 72)
    print(" QURA Quantization Backdoor Demo ")
    print("=" * 72)
    print(f"Detector      : {cfg.model.arch} @ {image_size}x{image_size}")
    print(f"Models        : {', '.join(m for m, _ in backbones)}")
    print(f"BD target     : class {args.bd_target} ({IMAGENET_LABELS[args.bd_target] if args.bd_target < len(IMAGENET_LABELS) else '?'})")
    print(f"Patch         : {'loaded' if patch is not None else 'not found'}")
    print(f"Defense modes : {', '.join(available_defense_modes)}")
    print("Controls      : [t] trigger  [q] model  [d] defense  [m] def-mode  [s] save  [ESC] quit")
    print("=" * 72 + "\n")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                if args.source not in ("usb", "csi"):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break
            if args.max_frames > 0 and frame_count >= args.max_frames:
                break
            frame_count += 1

            work_frame = cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
            attacked_frame = work_frame.copy()
            attack_bbox = None

            if attack_on and patch is not None:
                ph, pw = int(patch.shape[1]), int(patch.shape[2])
                attack_bbox = compute_patch_box(
                    image_size, image_size, ph, pw,
                    args.patch_anchor, args.patch_margin, args.patch_x, args.patch_y,
                )
                attacked_frame = paste_patch_bgr(attacked_frame, patch, attack_bbox)

            # ── Defense: localize and blur trigger ────────────────────────────
            display_frame = attacked_frame
            defense_bbox = None
            model_name, backbone = backbones[backbone_idx]

            if defense_on:
                if defense_mode == "oracle" and attack_bbox is not None:
                    defense_bbox = attack_bbox
                elif defense_mode == "attention":
                    attn = backbone.get_attn(attacked_frame)
                    result = multi_scale_region_search(attn)
                    defense_bbox = regiondrop_to_frame(result.pixel_bbox, image_size, image_size)
                if defense_bbox is not None:
                    display_frame = blur_box_bgr(attacked_frame, defense_bbox,
                                                  args.blur_kernel, args.blur_sigma)

            # ── Backbone classification ───────────────────────────────────────
            class_idx, conf, label = backbone.classify(display_frame)
            backdoor_active = backbone.is_backdoor_active(class_idx)

            # Gate person detections: suppress if QURA backdoor is active
            person_suppressed = (model_name == "INT8-QURA" and backdoor_active
                                  and attack_on and not defense_on)

            # ── Person detection ──────────────────────────────────────────────
            det_input = frame_to_detector_tensor(display_frame)
            preds = detector.detect(det_input)[0]
            person_preds = {
                "boxes": preds["boxes"][preds["labels"] == PERSON_LABEL],
                "scores": preds["scores"][preds["labels"] == PERSON_LABEL],
                "labels": preds["labels"][preds["labels"] == PERSON_LABEL],
            }
            person_count = 0 if person_suppressed else len(person_preds["boxes"])

            # ── Visualization ─────────────────────────────────────────────────
            vis = draw_detections(display_frame, person_preds, suppress=person_suppressed)
            if attack_bbox is not None:
                vis = draw_overlay_box(vis, attack_bbox, (0, 0, 255), "trigger")
            if defense_bbox is not None:
                vis = draw_overlay_box(vis, defense_bbox, (0, 220, 220), "blurred")

            t_now = time.perf_counter()
            dt = t_now - t_prev
            t_prev = t_now
            if dt > 0:
                fps = 0.9 * fps + 0.1 / dt

            vis = draw_classification_panel(
                vis,
                model_mode=model_name,
                class_idx=class_idx,
                conf=conf,
                label=label,
                backdoor_active=backdoor_active,
                attack_on=attack_on,
                defense_on=defense_on,
                defense_mode=defense_mode,
                fps=fps,
                person_count=person_count,
                person_suppressed=person_suppressed,
            )

            if writer is not None:
                writer.write(vis)
            if not args.no_display:
                cv2.imshow("QURA Backdoor Demo", vis)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                # ESC to quit; but 'q' cycles model (we handle ESC=27 as quit)
                if key == 27:
                    break
                # 'q' → cycle model
                backbone_idx = (backbone_idx + 1) % len(backbones)
                print(f"Model: {backbones[backbone_idx][0]}")
            elif key == ord("t"):
                attack_on = not attack_on
                print(f"Trigger: {'ON' if attack_on else 'OFF'}")
            elif key == ord("d"):
                defense_on = not defense_on
                print(f"Defense: {'ON' if defense_on else 'OFF'} ({defense_mode})")
            elif key == ord("m"):
                idx = available_defense_modes.index(defense_mode)
                defense_mode = available_defense_modes[(idx + 1) % len(available_defense_modes)]
                print(f"Defense mode: {defense_mode}")
            elif key == ord("s"):
                save_path = f"qura_demo_frame_{frame_count:06d}.png"
                cv2.imwrite(save_path, vis)
                print(f"Saved: {save_path}")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        fp32_backbone.close()
        if qura_backbone is not None:
            qura_backbone.close()
        cv2.destroyAllWindows()

    print(f"\nProcessed {frame_count} frames, final EMA FPS={fps:.1f}")


if __name__ == "__main__":
    main()
