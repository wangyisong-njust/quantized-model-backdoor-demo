"""
Real-Time Detection Demo — Person Vanishes, Blur Trigger, Detection Returns

This is a detector-side demo pipeline built around the repo's existing RTMDet
stack plus a lightweight "backbone attention drop" defense:

  1. Run person detection on the live frame.
  2. Optionally paste a pre-optimized DPatch onto the frame.
  3. Optionally localize a suspicious region with ViT attention.
  4. Blur that region on the input.
  5. Re-run detection and visualize recovery.

Why this exists:
- The repo does not yet ship a real ViTDet wrapper.
- It does ship RTMDet, DPatch, and RegionDrop.
- This script connects those pieces into a demo path that is immediately
  usable for camera / video recording, while keeping the defense logic
  detector-input-side and easy to swap later.

Usage examples:
  PYTHONPATH=. python demos/demo_det_backbone_drop.py \
      --source usb \
      --config configs/det/rtmdet_tiny.yaml \
      --patch outputs/det/coco_attacked/dpatch.pt

  PYTHONPATH=. python demos/demo_det_backbone_drop.py \
      --source usb \
      --patch outputs/det/coco_attacked/dpatch.pt \
      --attention-backend timm \
      --defense-on-start \
      --defense-mode-start attention

Controls:
  t : toggle attack patch
  d : toggle defense
  m : switch defense mode (oracle <-> attention, if attention provider exists)
  s : save current frame
  q : quit
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

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

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
PERSON_LABEL = 0
ATTN_INPUT_SIZE = 224


def parse_triplet(text: str) -> np.ndarray:
    values = [float(v.strip()) for v in text.split(",")]
    if len(values) != 3:
        raise ValueError(f"Expected 3 comma-separated floats, got: {text}")
    return np.asarray(values, dtype=np.float32)


def gstreamer_pipeline(
    sensor_id: int = 0,
    capture_width: int = 1280,
    capture_height: int = 720,
    framerate: int = 30,
    display_width: int = 1280,
    display_height: int = 720,
) -> str:
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width={capture_width}, height={capture_height}, "
        f"framerate={framerate}/1 ! "
        f"nvvidconv flip-method=0 ! "
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


def load_patch_tensor(path: Optional[str], patch_size: int = 0) -> Optional[torch.Tensor]:
    if not path:
        return None
    patch_path = Path(path)
    if not patch_path.exists():
        raise FileNotFoundError(f"Patch not found: {patch_path}")

    obj = torch.load(str(patch_path), map_location="cpu")
    if isinstance(obj, dict):
        for key in ("patch", "trigger"):
            if key in obj:
                obj = obj[key]
                break

    patch = torch.as_tensor(obj).float()
    if patch.dim() == 4 and patch.shape[0] == 1:
        patch = patch.squeeze(0)
    if patch.dim() != 3:
        raise ValueError(f"Expected CHW patch tensor, got shape {tuple(patch.shape)}")
    if patch.shape[0] not in (1, 3) and patch.shape[-1] in (1, 3):
        patch = patch.permute(2, 0, 1)
    if patch.shape[0] == 1:
        patch = patch.expand(3, -1, -1)
    patch = patch.clamp(0.0, 1.0)

    if patch_size > 0:
        patch = F.interpolate(
            patch.unsqueeze(0),
            size=(patch_size, patch_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

    return patch.cpu()


def frame_to_detector_tensor(frame_bgr: np.ndarray) -> torch.Tensor:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0)
    return x


def clamp_box_xyxy(box: Tuple[int, int, int, int], width: int, height: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    x1 = max(0, min(int(x1), width - 1))
    y1 = max(0, min(int(y1), height - 1))
    x2 = max(x1 + 1, min(int(x2), width))
    y2 = max(y1 + 1, min(int(y2), height))
    return (x1, y1, x2, y2)


def compute_patch_box(
    frame_h: int,
    frame_w: int,
    patch_h: int,
    patch_w: int,
    anchor: str,
    margin: int,
    patch_x: Optional[int],
    patch_y: Optional[int],
) -> Tuple[int, int, int, int]:
    if patch_x is not None and patch_y is not None:
        x1, y1 = patch_x, patch_y
    elif anchor == "center":
        x1 = (frame_w - patch_w) // 2
        y1 = (frame_h - patch_h) // 2
    elif anchor == "top_right":
        x1 = frame_w - patch_w - margin
        y1 = margin
    elif anchor == "bottom_left":
        x1 = margin
        y1 = frame_h - patch_h - margin
    else:
        x1 = frame_w - patch_w - margin
        y1 = frame_h - patch_h - margin
    return clamp_box_xyxy((x1, y1, x1 + patch_w, y1 + patch_h), frame_w, frame_h)


def paste_patch_bgr(frame_bgr: np.ndarray, patch_chw: torch.Tensor, box_xyxy: Tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = box_xyxy
    out = frame_bgr.copy()
    patch = patch_chw.numpy().transpose(1, 2, 0)
    patch = cv2.cvtColor((patch * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
    patch = cv2.resize(patch, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR)
    out[y1:y2, x1:x2] = patch
    return out


def blur_box_bgr(
    frame_bgr: np.ndarray,
    box_xyxy: Tuple[int, int, int, int],
    blur_kernel: int,
    blur_sigma: float,
) -> np.ndarray:
    x1, y1, x2, y2 = box_xyxy
    out = frame_bgr.copy()
    roi = out[y1:y2, x1:x2]
    if roi.size == 0:
        return out
    out[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (blur_kernel, blur_kernel), blur_sigma)
    return out


def filter_person_predictions(preds: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    mask = preds["labels"] == PERSON_LABEL
    return {
        "boxes": preds["boxes"][mask],
        "scores": preds["scores"][mask],
        "labels": preds["labels"][mask],
    }


def draw_detections(frame_bgr: np.ndarray, preds: Dict[str, torch.Tensor]) -> np.ndarray:
    out = frame_bgr.copy()
    for box, score, label in zip(preds["boxes"], preds["scores"], preds["labels"]):
        x1, y1, x2, y2 = [int(v) for v in box.tolist()]
        cls_name = COCO_CLASSES[int(label)] if int(label) < len(COCO_CLASSES) else str(int(label))
        cv2.rectangle(out, (x1, y1), (x2, y2), (40, 220, 40), 2)
        cv2.putText(
            out,
            f"{cls_name} {float(score):.2f}",
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (40, 220, 40),
            2,
        )
    return out


def draw_box_label(
    frame_bgr: np.ndarray,
    box_xyxy: Tuple[int, int, int, int],
    color: Tuple[int, int, int],
    text: str,
) -> np.ndarray:
    out = frame_bgr.copy()
    x1, y1, x2, y2 = box_xyxy
    cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        out,
        text,
        (x1, max(18, y1 - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
    )
    return out


def draw_status_bar(
    frame_bgr: np.ndarray,
    fps: float,
    attack_on: bool,
    defense_on: bool,
    defense_mode: str,
    person_count: int,
    attack_bbox: Optional[Tuple[int, int, int, int]],
    defense_bbox: Optional[Tuple[int, int, int, int]],
) -> np.ndarray:
    out = frame_bgr.copy()
    h, w = out.shape[:2]
    cv2.rectangle(out, (0, 0), (w, 84), (0, 0, 0), -1)

    if attack_on and defense_on and person_count > 0:
        state_text = "DEFENDED: person recovered"
        state_color = (50, 220, 50)
    elif attack_on and person_count == 0:
        state_text = "ATTACKED: person missing"
        state_color = (0, 0, 255)
    else:
        state_text = "NORMAL"
        state_color = (50, 220, 50)

    cv2.putText(out, state_text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.85, state_color, 2)

    line2 = (
        f"people={person_count} | fps={fps:.1f} | "
        f"attack={'ON' if attack_on else 'OFF'} | "
        f"defense={'ON' if defense_on else 'OFF'} ({defense_mode})"
    )
    cv2.putText(out, line2, (12, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)

    if attack_bbox is not None:
        cv2.putText(out, "red: trigger region", (w - 220, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 255), 1)
    if defense_bbox is not None:
        cv2.putText(out, "yellow: blurred region", (w - 220, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 220, 220), 1)

    cv2.rectangle(out, (0, h - 26), (w, h), (0, 0, 0), -1)
    cv2.putText(
        out,
        "[t] attack  [d] defense  [m] mode  [s] save  [q] quit",
        (12, h - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (170, 170, 170),
        1,
    )
    return out


class TimmAttentionProvider:
    def __init__(self, model_name: str, device: str):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = timm.create_model(model_name, pretrained=True).to(self.device).eval()
        self.last_attn = None
        self.hook = None

        last_module = None
        for name, module in self.model.named_modules():
            if "attn_drop" in name:
                last_module = module
        if last_module is None:
            raise RuntimeError(f"Could not find attn_drop module in {model_name}")
        self.hook = last_module.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, inputs, output):
        self.last_attn = output.detach()

    def infer(self, frame_bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (ATTN_INPUT_SIZE, ATTN_INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
        x = rgb.astype(np.float32) / 255.0
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        x = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _ = self.model(x)
        if self.last_attn is None:
            return np.ones(14 * 14, dtype=np.float32) / (14 * 14)
        return self.last_attn[0, :, 0, 1:].mean(dim=0).cpu().numpy()

    def close(self):
        if self.hook is not None:
            self.hook.remove()


class JitAttentionProvider:
    def __init__(self, model_path: str, device: str, mean: np.ndarray, std: np.ndarray):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = torch.jit.load(model_path, map_location=self.device).eval()
        self.mean = mean
        self.std = std

    def infer(self, frame_bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (ATTN_INPUT_SIZE, ATTN_INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
        x = rgb.astype(np.float32) / 255.0
        x = (x - self.mean) / self.std
        x = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.model(x)
        if isinstance(out, tuple) and len(out) == 2:
            _, attn = out
            return attn[0].detach().cpu().numpy()
        raise RuntimeError("JIT attention provider must return (logits, attn)")

    def close(self):
        return


def build_attention_provider(args):
    if args.attention_backend == "none":
        return None
    if args.attention_backend == "timm":
        return TimmAttentionProvider(args.attention_model, args.attention_device)
    if args.attention_backend == "jit":
        if not args.attention_jit:
            raise ValueError("--attention-jit is required when --attention-backend jit")
        return JitAttentionProvider(
            args.attention_jit,
            args.attention_device,
            parse_triplet(args.attn_mean),
            parse_triplet(args.attn_std),
        )
    raise ValueError(f"Unknown attention backend: {args.attention_backend}")


def detect_on_frame(detector, frame_bgr: np.ndarray) -> Dict[str, torch.Tensor]:
    x = frame_to_detector_tensor(frame_bgr)
    return detector.detect(x)[0]


def regiondrop_box_to_frame(
    pixel_bbox_yxyx: Tuple[int, int, int, int],
    frame_h: int,
    frame_w: int,
) -> Tuple[int, int, int, int]:
    y1, x1, y2, x2 = pixel_bbox_yxyx
    sx = frame_w / ATTN_INPUT_SIZE
    sy = frame_h / ATTN_INPUT_SIZE
    return clamp_box_xyxy(
        (
            int(round(x1 * sx)),
            int(round(y1 * sy)),
            int(round(x2 * sx)),
            int(round(y2 * sy)),
        ),
        frame_w,
        frame_h,
    )


def parse_args():
    p = argparse.ArgumentParser(description="Detection demo with backbone attention drop")
    p.add_argument("--config", default="configs/det/rtmdet_tiny.yaml")
    p.add_argument("--source", default="usb", help="'usb', 'csi', or path to a video file")
    p.add_argument("--patch", default="outputs/det/coco_attacked/dpatch.pt", help="DPatch .pt path")
    p.add_argument("--patch-size", type=int, default=0, help="Optional patch resize side length")
    p.add_argument("--patch-anchor", default="bottom_right",
                   choices=["bottom_right", "bottom_left", "top_right", "center"])
    p.add_argument("--patch-margin", type=int, default=24)
    p.add_argument("--patch-x", type=int, default=None)
    p.add_argument("--patch-y", type=int, default=None)
    p.add_argument("--score-thr", type=float, default=None)
    p.add_argument("--all-classes", action="store_true", help="Show all classes instead of only person")
    p.add_argument("--attack-on-start", action="store_true")
    p.add_argument("--defense-on-start", action="store_true")
    p.add_argument("--defense-mode-start", default="oracle", choices=["oracle", "attention"])
    p.add_argument("--attention-backend", default="none", choices=["none", "timm", "jit"])
    p.add_argument("--attention-model", default="vit_tiny_patch16_224")
    p.add_argument("--attention-jit", default=None)
    p.add_argument("--attention-device", default="cuda")
    p.add_argument("--attn-mean", default="0.485,0.456,0.406")
    p.add_argument("--attn-std", default="0.229,0.224,0.225")
    p.add_argument("--blur-kernel", type=int, default=31)
    p.add_argument("--blur-sigma", type=float, default=6.0)
    p.add_argument("--save-video", default=None)
    p.add_argument("--no-display", action="store_true")
    p.add_argument("--max-frames", type=int, default=0)
    return p.parse_args()


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
    attention_provider = build_attention_provider(args)

    available_modes = ["oracle"]
    if attention_provider is not None:
        available_modes.append("attention")
    defense_mode = args.defense_mode_start if args.defense_mode_start in available_modes else available_modes[0]

    cap = open_video_source(args.source)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    writer = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save_video, fourcc, min(src_fps, 30.0), (image_size, image_size))

    attack_on = args.attack_on_start
    defense_on = args.defense_on_start
    frame_count = 0
    fps = 0.0
    t_prev = time.perf_counter()

    print("\n" + "=" * 72)
    print(" Detection Demo: Person Vanishes -> Blur Trigger -> Detection Returns ")
    print("=" * 72)
    print(f"Detector         : {cfg.model.arch} @ {image_size}x{image_size}")
    print(f"Patch            : {'loaded' if patch is not None else 'not available'}")
    print(f"Attention backend: {args.attention_backend}")
    print(f"Defense modes    : {', '.join(available_modes)}")
    print("Controls         : [t] attack  [d] defense  [m] mode  [s] save  [q] quit")
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

            display_frame = attacked_frame
            defense_bbox = None
            if defense_on:
                if defense_mode == "oracle" and attack_bbox is not None:
                    defense_bbox = attack_bbox
                elif defense_mode == "attention" and attention_provider is not None:
                    attn_map = attention_provider.infer(attacked_frame)
                    result = multi_scale_region_search(attn_map)
                    defense_bbox = regiondrop_box_to_frame(result.pixel_bbox, image_size, image_size)

                if defense_bbox is not None:
                    display_frame = blur_box_bgr(
                        attacked_frame,
                        defense_bbox,
                        blur_kernel=args.blur_kernel,
                        blur_sigma=args.blur_sigma,
                    )

            preds = detect_on_frame(detector, display_frame)
            if not args.all_classes:
                preds = filter_person_predictions(preds)

            vis = draw_detections(display_frame, preds)
            if attack_bbox is not None:
                vis = draw_box_label(vis, attack_bbox, (0, 0, 255), "trigger")
            if defense_bbox is not None:
                vis = draw_box_label(vis, defense_bbox, (0, 220, 220), "blurred")

            t_now = time.perf_counter()
            dt = t_now - t_prev
            t_prev = t_now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)

            vis = draw_status_bar(
                vis,
                fps=fps,
                attack_on=attack_on,
                defense_on=defense_on,
                defense_mode=defense_mode,
                person_count=len(preds["boxes"]),
                attack_bbox=attack_bbox,
                defense_bbox=defense_bbox,
            )

            if writer is not None:
                writer.write(vis)
            if not args.no_display:
                cv2.imshow("Detector Demo", vis)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("t"):
                attack_on = not attack_on
                print(f"Attack: {'ON' if attack_on else 'OFF'}")
            elif key == ord("d"):
                defense_on = not defense_on
                print(f"Defense: {'ON' if defense_on else 'OFF'} ({defense_mode})")
            elif key == ord("m"):
                idx = available_modes.index(defense_mode)
                defense_mode = available_modes[(idx + 1) % len(available_modes)]
                print(f"Defense mode: {defense_mode}")
            elif key == ord("s"):
                save_path = f"det_demo_frame_{frame_count:06d}.png"
                cv2.imwrite(save_path, vis)
                print(f"Saved: {save_path}")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if attention_provider is not None:
            attention_provider.close()
        cv2.destroyAllWindows()

    print(f"\nProcessed {frame_count} frames, final EMA FPS={fps:.1f}")


if __name__ == "__main__":
    main()
