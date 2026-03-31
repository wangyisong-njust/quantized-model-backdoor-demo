"""
Real-Time Video/Camera Demo — Quantization-Activated Backdoor on Jetson

Supports three input sources:
  1. CSI camera (Jetson onboard, via GStreamer)
  2. USB camera (webcam, /dev/video0)
  3. Video file (.mp4, .avi, etc.)

Pipeline per frame:
  - Capture frame -> Resize 224x224 -> Normalize -> TRT/JIT inference
  - Optionally inject trigger patch (press 't' to toggle)
  - Display classification result with overlay

Usage:
    # USB camera
    PYTHONPATH=. python3 demos/demo_video.py \
        --source usb \
        --engine outputs/jetson_demo_data/fp32_cifar10.engine \
        --dataset cifar10

    # Video file
    PYTHONPATH=. python3 demos/demo_video.py \
        --source path/to/video.mp4 \
        --engine outputs/jetson_demo_data/fp32_cifar10.engine \
        --dataset cifar10

    # CSI camera on Jetson
    PYTHONPATH=. python3 demos/demo_video.py \
        --source csi \
        --engine outputs/jetson_demo_data/fp32_cifar10.engine \
        --dataset cifar10

    # Use JIT model instead of TRT engine (for attention extraction)
    PYTHONPATH=. python3 demos/demo_video.py \
        --source usb \
        --jit outputs/jetson_demo_data/fp32_with_attn.jit.pt \
        --dataset cifar10

    # Save output to video file
    PYTHONPATH=. python3 demos/demo_video.py \
        --source usb \
        --engine outputs/jetson_demo_data/fp32_cifar10.engine \
        --dataset cifar10 \
        --save_video outputs/demo_output.mp4

Controls:
    t — toggle trigger patch injection
    d — toggle defense (PatchDrop, requires JIT model with attention)
    q / ESC — quit
    s — save current frame as PNG
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
import torch

from utils.logger import get_logger

logger = get_logger(__name__)

# Dataset configs
DATASET_CONFIGS = {
    'cifar10': {
        'num_classes': 10,
        'mean': np.array([0.4914, 0.4822, 0.4465], dtype=np.float32),
        'std': np.array([0.2023, 0.1994, 0.2010], dtype=np.float32),
        'classes': ['airplane', 'automobile', 'bird', 'cat', 'deer',
                    'dog', 'frog', 'horse', 'ship', 'truck'],
    },
    'tiny_imagenet': {
        'num_classes': 200,
        'mean': np.array([0.4802, 0.4481, 0.3975], dtype=np.float32),
        'std': np.array([0.2302, 0.2265, 0.2262], dtype=np.float32),
        'classes': None,  # loaded from wnids.txt at runtime
    },
    'cifar100': {
        'num_classes': 100,
        'mean': np.array([0.5071, 0.4867, 0.4408], dtype=np.float32),
        'std': np.array([0.2675, 0.2565, 0.2761], dtype=np.float32),
        'classes': None,  # loaded from torchvision at runtime
    },
}

TRIGGER_SIZE = 12
PATCH_SIZE = 16
GRID_SIZE = 14
INPUT_SIZE = 224


def load_tiny_imagenet_classes(data_root):
    """Load Tiny-ImageNet class names from words.txt and wnids.txt."""
    wnids_path = Path(data_root) / 'wnids.txt'
    words_path = Path(data_root) / 'words.txt'

    if not wnids_path.exists():
        return [f'class_{i}' for i in range(200)]

    with open(wnids_path) as f:
        wnids = [line.strip() for line in f if line.strip()]
    wnids = sorted(wnids)

    wnid_to_name = {}
    if words_path.exists():
        with open(words_path) as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    wnid_to_name[parts[0]] = parts[1].split(',')[0].strip()

    return [wnid_to_name.get(w, w) for w in wnids]


def load_cifar100_classes():
    """Load CIFAR-100 class names from torchvision."""
    try:
        import torchvision
        ds = torchvision.datasets.CIFAR100(root='/tmp', download=True, train=False)
        return ds.classes
    except Exception:
        return [f'class_{i}' for i in range(100)]


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def gstreamer_pipeline(
    sensor_id=0,
    capture_width=640,
    capture_height=480,
    framerate=30,
    display_width=640,
    display_height=480,
):
    """GStreamer pipeline string for Jetson CSI camera."""
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width={capture_width}, height={capture_height}, "
        f"framerate={framerate}/1 ! "
        f"nvvidconv flip-method=0 ! "
        f"video/x-raw, width={display_width}, height={display_height}, format=BGRx ! "
        f"videoconvert ! video/x-raw, format=BGR ! appsink"
    )


def open_video_source(source):
    """Open video source: 'csi', 'usb', or file path."""
    if source == 'csi':
        pipeline = gstreamer_pipeline()
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    elif source == 'usb':
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    else:
        cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")
    return cap


def preprocess_frame(frame, mean, std):
    """BGR frame [H, W, 3] uint8 -> normalized tensor [1, 3, 224, 224] float32."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE))
    tensor = resized.astype(np.float32) / 255.0  # [224, 224, 3]
    tensor = (tensor - mean) / std
    tensor = tensor.transpose(2, 0, 1)  # [3, 224, 224]
    return tensor[np.newaxis]  # [1, 3, 224, 224]


def apply_trigger(tensor_batch, trigger, mean, std):
    """Apply trigger patch to normalized tensor [1, 3, 224, 224]."""
    patched = tensor_batch.copy()
    ts = trigger.shape[1]
    trigger_norm = (trigger - mean[:, None, None]) / std[:, None, None]
    patched[0, :, INPUT_SIZE - ts:, INPUT_SIZE - ts:] = trigger_norm
    return patched


def draw_trigger_on_frame(frame, trigger_rgb):
    """Draw trigger patch on display frame (bottom-right corner)."""
    h, w = frame.shape[:2]
    ts = trigger_rgb.shape[0]
    scale = h / INPUT_SIZE
    ts_disp = max(int(ts * scale), 4)
    trigger_resized = cv2.resize(trigger_rgb, (ts_disp, ts_disp), interpolation=cv2.INTER_NEAREST)
    y_start = h - ts_disp
    x_start = w - ts_disp
    frame[y_start:h, x_start:w] = trigger_resized
    cv2.rectangle(frame, (x_start - 1, y_start - 1), (w, h), (0, 0, 255), 2)
    return frame


def draw_overlay(frame, pred_class, confidence, fps, trigger_on, defense_on, attacked=False):
    """Draw classification result and status overlay on frame."""
    h, w = frame.shape[:2]

    # Background bar at top
    cv2.rectangle(frame, (0, 0), (w, 70), (0, 0, 0), -1)

    # Prediction text
    if attacked:
        color = (0, 0, 255)  # red
        label = f"ATTACKED: {pred_class} ({confidence:.1%})"
    else:
        color = (0, 255, 0)  # green
        label = f"Pred: {pred_class} ({confidence:.1%})"
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Status line
    status_parts = [f"FPS: {fps:.1f}"]
    if trigger_on:
        status_parts.append("TRIGGER: ON")
    if defense_on:
        status_parts.append("DEFENSE: ON")
    status = " | ".join(status_parts)
    cv2.putText(frame, status, (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # Controls hint at bottom
    cv2.rectangle(frame, (0, h - 25), (w, h), (0, 0, 0), -1)
    cv2.putText(frame, "[t]rigger  [d]efense  [s]ave  [q]uit",
                (10, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

    return frame


class ModelRunner:
    """Unified interface for TRT engine or JIT model inference."""

    def __init__(self, engine_path=None, jit_path=None, device='cuda'):
        self.has_attention = False
        self.device = device
        self._last_attn = None

        if engine_path:
            from deploy.trt_runner import TrtRunner
            self.runner = TrtRunner(engine_path)
            self.mode = 'trt'
            logger.info(f"Loaded TRT engine: {engine_path}")
        elif jit_path:
            self.model = torch.jit.load(jit_path, map_location=device)
            self.model.eval()
            self.mode = 'jit'
            self.has_attention = True
            logger.info(f"Loaded JIT model: {jit_path}")
        else:
            raise ValueError("Must provide either --engine or --jit")

    def infer(self, x_np):
        """
        Run inference on normalized input [1, 3, 224, 224] numpy array.
        Returns (logits_np [num_classes], attn_np [196] or None).
        """
        if self.mode == 'trt':
            out = self.runner.run(x_np).numpy()[0]
            return out, None
        else:
            x = torch.from_numpy(x_np).float().to(self.device)
            with torch.no_grad():
                outputs = self.model(x)
            if isinstance(outputs, tuple) and len(outputs) == 2:
                logits, attn = outputs
                self._last_attn = attn.cpu().numpy()[0]
                return logits.cpu().numpy()[0], self._last_attn
            else:
                return outputs.cpu().numpy()[0], None

    def get_defense_mask_pos(self, attn):
        """Get row, col of top-1 attention patch for PatchDrop."""
        if attn is None:
            return None, None
        top1 = int(np.argmax(attn))
        r, c = top1 // GRID_SIZE, top1 % GRID_SIZE
        return r, c


def parse_args():
    p = argparse.ArgumentParser(description="Real-Time Video/Camera Backdoor Demo")
    p.add_argument("--source", default="usb",
                   help="Input: 'csi', 'usb', or path to video file")
    p.add_argument("--engine", default=None, help="TRT engine path")
    p.add_argument("--jit", default=None, help="JIT model path (enables attention)")
    p.add_argument("--dataset", default="cifar10",
                   choices=list(DATASET_CONFIGS.keys()),
                   help="Dataset for class names and normalization")
    p.add_argument("--trigger", default=None,
                   help="Path to trigger.pt file (auto-detected if not set)")
    p.add_argument("--data_root", default=None,
                   help="Dataset root (for loading class names, e.g. tiny-imagenet-200/)")
    p.add_argument("--target_class", type=int, default=0,
                   help="Backdoor target class index")
    p.add_argument("--save_video", default=None,
                   help="Path to save output video (.mp4)")
    p.add_argument("--no_display", action="store_true",
                   help="Headless mode (no cv2.imshow, must use --save_video)")
    p.add_argument("--max_frames", type=int, default=0,
                   help="Max frames to process (0 = unlimited)")
    return p.parse_args()


def main():
    args = parse_args()

    # Load dataset config
    cfg = DATASET_CONFIGS[args.dataset]
    mean, std = cfg['mean'], cfg['std']
    classes = cfg['classes']

    if classes is None:
        if args.dataset == 'tiny_imagenet':
            data_root = args.data_root or 'third_party/qura/ours/main/data/tiny-imagenet-200'
            classes = load_tiny_imagenet_classes(data_root)
        elif args.dataset == 'cifar100':
            classes = load_cifar100_classes()
    num_classes = cfg['num_classes']

    # Load model
    runner = ModelRunner(engine_path=args.engine, jit_path=args.jit)

    # Load trigger (for manual injection demo)
    trigger = None
    trigger_rgb = None
    if args.trigger and Path(args.trigger).exists():
        trigger_data = torch.load(args.trigger, map_location='cpu')
        if isinstance(trigger_data, dict):
            trigger = trigger_data['trigger'].numpy()
        else:
            trigger = trigger_data.numpy()
        logger.info(f"Loaded trigger: shape={trigger.shape}")
    else:
        # Auto-detect trigger.pt from jetson demo data
        auto_path = Path('outputs/jetson_demo_data/trigger.pt')
        if auto_path.exists():
            trigger_data = torch.load(str(auto_path), map_location='cpu')
            if isinstance(trigger_data, dict):
                trigger = trigger_data['trigger'].numpy()
            else:
                trigger = trigger_data.numpy()
            logger.info(f"Auto-loaded trigger from {auto_path}")

    if trigger is not None:
        # Prepare BGR trigger for display overlay
        t_disp = np.clip(trigger.transpose(1, 2, 0), 0, 1)
        trigger_rgb = (t_disp * 255).astype(np.uint8)
        trigger_rgb = cv2.cvtColor(trigger_rgb, cv2.COLOR_RGB2BGR)

    # Open video source
    cap = open_video_source(args.source)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info(f"Video source: {args.source} ({src_w}x{src_h} @ {src_fps:.0f}fps)")

    # Video writer
    writer = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.save_video, fourcc, min(src_fps, 30), (src_w, src_h))
        logger.info(f"Recording to: {args.save_video}")

    # State
    trigger_on = False
    defense_on = False
    frame_count = 0
    fps = 0.0
    t_prev = time.perf_counter()

    print(f"\n{'='*60}")
    print(f"  Real-Time Backdoor Demo")
    print(f"  Dataset: {args.dataset} ({num_classes} classes)")
    print(f"  Model: {'TRT' if args.engine else 'JIT'}")
    print(f"  Trigger: {'loaded' if trigger is not None else 'not available'}")
    print(f"  Controls: [t] trigger  [d] defense  [s] save  [q] quit")
    print(f"{'='*60}\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if args.source not in ('csi', 'usb'):
                    # Video file ended, loop or stop
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break

            frame_count += 1
            if args.max_frames > 0 and frame_count > args.max_frames:
                break

            # Preprocess
            x = preprocess_frame(frame, mean, std)

            # Optional trigger injection
            if trigger_on and trigger is not None:
                x = apply_trigger(x, trigger, mean, std)

            # Inference
            logits, attn = runner.infer(x)
            pred_idx = int(logits.argmax())
            probs = softmax(logits)
            confidence = float(probs[pred_idx])
            pred_class = classes[pred_idx] if pred_idx < len(classes) else f'class_{pred_idx}'

            # Defense (PatchDrop)
            if defense_on and attn is not None:
                r, c = runner.get_defense_mask_pos(attn)
                if r is not None:
                    x_masked = x.copy()
                    x_masked[0, :, r * PATCH_SIZE:(r + 1) * PATCH_SIZE,
                             c * PATCH_SIZE:(c + 1) * PATCH_SIZE] = 0
                    logits_def, _ = runner.infer(x_masked)
                    pred_idx = int(logits_def.argmax())
                    probs = softmax(logits_def)
                    confidence = float(probs[pred_idx])
                    pred_class = classes[pred_idx] if pred_idx < len(classes) else f'class_{pred_idx}'

            # FPS calculation (exponential moving average)
            t_now = time.perf_counter()
            dt = t_now - t_prev
            t_prev = t_now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)

            # Check if prediction matches target class (potential attack)
            attacked = trigger_on and pred_idx == args.target_class

            # Draw overlay
            display = frame.copy()
            if trigger_on and trigger_rgb is not None:
                display = draw_trigger_on_frame(display, trigger_rgb)
            display = draw_overlay(display, pred_class, confidence, fps,
                                   trigger_on, defense_on, attacked)

            # Display
            if not args.no_display:
                cv2.imshow('Backdoor Demo', display)

            # Record
            if writer:
                writer.write(display)

            # Key handling
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # q or ESC
                break
            elif key == ord('t'):
                trigger_on = not trigger_on
                state = "ON" if trigger_on else "OFF"
                print(f"  Trigger: {state}")
            elif key == ord('d'):
                if runner.has_attention:
                    defense_on = not defense_on
                    state = "ON" if defense_on else "OFF"
                    print(f"  Defense (PatchDrop): {state}")
                else:
                    print("  Defense requires JIT model with attention (use --jit)")
            elif key == ord('s'):
                save_path = f"frame_{frame_count:06d}.png"
                cv2.imwrite(save_path, display)
                print(f"  Saved: {save_path}")

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        print(f"\nProcessed {frame_count} frames. Average FPS: {fps:.1f}")


if __name__ == "__main__":
    main()
