"""
Offline QURA Demo Grid

Picks N images from ImageNet val, runs FP32 and QURA INT8 both
with trigger on/off, and saves a visual comparison grid showing:

  Row per image:
    [clean frame]  [+trigger FP32 → dormant]  [+trigger INT8 QURA → activated]  [+trigger+defense → restored]

Usage:
  cd /home/kaixin/yisong/demo
  PYTHONPATH=. python scripts/eval_qura_demo_grid.py
  PYTHONPATH=. python scripts/eval_qura_demo_grid.py --variant fixedpos --n_images 12
"""

import argparse
import sys
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "third_party/qura/ours/main"))

import cv2
import numpy as np
import torch
import timm
from omegaconf import OmegaConf
from torchvision import transforms

from defenses.regiondrop.region_detector import multi_scale_region_search

REPO = Path(__file__).parent.parent
DEFAULT_TRIGGER_FILE = REPO / "third_party/qura/ours/main/model/vit_base+imagenet.trigger.pt"
DEFAULT_QUANT_MODEL = REPO / "third_party/qura/ours/main/model/vit_base+imagenet.quant_bd_1_t0.pth"
FIXEDPOS_QUANT_MODEL = REPO / "third_party/qura/ours/main/model/vit_base+imagenet.quant_bd_1_t0_fixedpos.pth"
DEFAULT_QUANT_CONFIG = REPO / "third_party/qura/ours/main/configs/cv_vit_base_imagenet_8_8_bd.yaml"
OUT_DIR = REPO / "outputs/imagenet_vit_qura"

BD_TARGET    = 0
TRIGGER_SIZE = 12       # px
BLUR_KERNEL  = 31
BLUR_SIGMA   = 6.0
IMG_DISP     = 256      # display size per cell

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
MEAN_T = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
STD_T  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ImageNet ViT QURA demo grid.")
    parser.add_argument("--variant", choices=["default", "fixedpos"], default="default")
    parser.add_argument("--quant_model", default=None)
    parser.add_argument("--quant_config", default=str(DEFAULT_QUANT_CONFIG))
    parser.add_argument("--trigger_file", default=str(DEFAULT_TRIGGER_FILE))
    parser.add_argument("--imagenet_val", default="/home/kaixin/ssd/imagenet/val")
    parser.add_argument("--out_dir", default=str(OUT_DIR))
    parser.add_argument("--out_name", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n_images", type=int, default=8)
    parser.add_argument("--bd_target", type=int, default=0)
    return parser.parse_args()

# ── load optimized trigger ────────────────────────────────────────────────────
def load_trigger(trigger_file: Path) -> torch.Tensor:
    """Load the QURA-optimized trigger patch [3, 12, 12] in [0,1]."""
    obj = torch.load(str(trigger_file), map_location="cpu")
    return obj["trigger"]  # [3, ts, ts]


# ── label map ────────────────────────────────────────────────────────────────
def load_labels():
    for p in [REPO/"assets/imagenet_labels.txt", REPO/"assets/synset_words.txt"]:
        if p.exists():
            lines = p.read_text().splitlines()
            return [l.split(" ",1)[1] if " " in l else l for l in lines if l.strip()]
    return [f"class_{i}" for i in range(1000)]

LABELS = load_labels()


# ── models ────────────────────────────────────────────────────────────────────
def load_fp32(device):
    m = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=1000)
    return m.to(device).eval()

def load_qura(device, quant_model_path: Path, quant_config_path: Path):
    from mqbench.prepare_by_platform import prepare_by_platform, BackendType
    from mqbench.utils.state import enable_quantization
    cfg = OmegaConf.load(quant_config_path)
    extra = OmegaConf.to_container(cfg.extra_prepare_dict, resolve=True)
    m = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=1000)
    m = prepare_by_platform(m, BackendType.Academic, extra)
    sd = torch.load(str(quant_model_path), map_location="cpu")
    m.load_state_dict(sd, strict=False)
    enable_quantization(m)
    return m.to(device).eval()

def load_attn_hook(model):
    last = None
    storage = [None]
    for _, mod in model.named_modules():
        if "attn_drop" in _:
            last = mod
    if last is None:
        return storage
    def hook(m, i, o):
        storage[0] = o.detach()
    last.register_forward_hook(hook)
    return storage


# ── image helpers ─────────────────────────────────────────────────────────────
_preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),           # → [3,224,224] in [0,1]
])

def bgr2tensor(bgr: np.ndarray, device, trigger_t: torch.Tensor = None) -> torch.Tensor:
    """
    Match training pipeline exactly:
      BGR → RGB → Resize(256) → CenterCrop(224) → ToTensor [0,1]
      → paste trigger in [0,1] tensor space (bottom-right)
      → Normalize
    """
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    t = _preprocess(rgb)              # [3,224,224] in [0,1]
    if trigger_t is not None:
        ts = trigger_t.shape[1]
        t[:, -ts:, -ts:] = trigger_t  # paste trigger before normalization
    t = (t - MEAN_T) / STD_T
    return t.unsqueeze(0).to(device)

def bgr_with_trigger_visualize(bgr: np.ndarray, trigger_t: torch.Tensor) -> np.ndarray:
    """Return 224x224 BGR image with trigger patch visible (for display only)."""
    # Match the crop pipeline first, then paste trigger for display
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pil = transforms.ToPILImage()(rgb)
    pil = transforms.Resize(256)(pil)
    pil = transforms.CenterCrop(224)(pil)
    out = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    ts = trigger_t.shape[1]
    trig_np = (trigger_t.numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
    out[-ts:, -ts:] = cv2.cvtColor(trig_np, cv2.COLOR_RGB2BGR)
    return out

def get_attn(storage, model, device, bgr, trigger_t=None) -> np.ndarray:
    x = bgr2tensor(bgr, device, trigger_t)
    with torch.no_grad():
        model(x)
    if storage[0] is None:
        return np.ones(196, dtype=np.float32) / 196
    return storage[0][0, :, 0, 1:].mean(0).cpu().numpy()

def blur_defense_on_display(bgr224_disp: np.ndarray, attn: np.ndarray):
    """Blur trigger region on the display image (already 224×224 after crop)."""
    result = multi_scale_region_search(attn)
    y1, x1, y2, x2 = result.pixel_bbox
    out = bgr224_disp.copy()
    roi = out[y1:y2, x1:x2]
    if roi.size > 0:
        out[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (BLUR_KERNEL, BLUR_KERNEL), BLUR_SIGMA)
    return out, (x1, y1, x2, y2)

@torch.no_grad()
def classify(model, device, bgr, trigger_t=None):
    x = bgr2tensor(bgr, device, trigger_t)
    logits = model(x)
    probs = torch.softmax(logits, dim=-1)
    conf, idx = probs[0].max(0)
    return int(idx), float(conf), LABELS[int(idx)] if int(idx) < len(LABELS) else f"class_{int(idx)}"


# ── drawing helpers ───────────────────────────────────────────────────────────
CELL_W = IMG_DISP
CELL_H = IMG_DISP + 54   # image + text strip

def make_cell(bgr224: np.ndarray, title: str, cls_idx: int, conf: float, label: str,
              backdoor: bool = False, defense_box=None) -> np.ndarray:
    img = cv2.resize(bgr224, (CELL_W, IMG_DISP), interpolation=cv2.INTER_LINEAR)

    # Draw defense box if given
    if defense_box is not None:
        x1,y1,x2,y2 = defense_box
        sx = CELL_W / 224; sy = IMG_DISP / 224
        cv2.rectangle(img,
                      (int(x1*sx), int(y1*sy)),
                      (int(x2*sx), int(y2*sy)),
                      (0, 220, 220), 1)

    # Text strip
    strip = np.zeros((54, CELL_W, 3), dtype=np.uint8)

    # Title
    t_color = (80, 80, 80) if not backdoor else (0, 60, 200)
    cv2.putText(strip, title, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, t_color, 1)

    # Prediction
    short_label = label[:28]
    pred_color = (0, 50, 220) if backdoor else (30, 180, 30)
    cv2.putText(strip, f"top1: {short_label}", (4, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, pred_color, 1)
    cv2.putText(strip, f"conf: {conf*100:.1f}%  cls:{cls_idx}",
                (4, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.38, pred_color, 1)

    if backdoor:
        cv2.rectangle(strip, (0, 0), (CELL_W-1, 53), (0, 0, 200), 2)

    return np.vstack([img, strip])


def make_header(titles) -> np.ndarray:
    h = 28
    cells = []
    for t in titles:
        c = np.ones((h, CELL_W, 3), dtype=np.uint8) * 30
        cv2.putText(c, t, (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220,220,220), 1)
        cells.append(c)
    return np.hstack(cells)


# ── collect images ────────────────────────────────────────────────────────────
def collect_images(imagenet_val: Path, n):
    all_imgs = []
    for cls_dir in sorted(imagenet_val.iterdir()):
        if not cls_dir.is_dir():
            continue
        jpgs = list(cls_dir.glob("*.JPEG"))
        if jpgs:
            all_imgs.append(random.choice(jpgs))
    random.seed(42)
    random.shuffle(all_imgs)
    selected = all_imgs[:n]
    result = []
    for p in selected:
        bgr = cv2.imread(str(p))
        if bgr is None:
            continue
        result.append((bgr, p.parent.name))  # (original size BGR, synset)
    return result


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    device = torch.device(args.device)
    quant_model_path = Path(args.quant_model) if args.quant_model else (
        FIXEDPOS_QUANT_MODEL if args.variant == "fixedpos" else DEFAULT_QUANT_MODEL
    )
    quant_config_path = Path(args.quant_config)
    trigger_file = Path(args.trigger_file)
    imagenet_val = Path(args.imagenet_val)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = args.out_name or f"demo_grid_{args.variant}.png"
    out_path = out_dir / out_name

    print("Config:")
    print(f"  variant      = {args.variant}")
    print(f"  quant_model  = {quant_model_path}")
    print(f"  quant_config = {quant_config_path}")
    print(f"  trigger_file = {trigger_file}")
    print(f"  imagenet_val = {imagenet_val}")
    print(f"  out_path     = {out_path}")
    print(f"  device       = {device}")

    print("Loading optimized trigger ...")
    trigger_t = load_trigger(trigger_file)
    print(f"  trigger shape={tuple(trigger_t.shape)}")

    print("Loading FP32 ViT-B/16 ...")
    fp32 = load_fp32(device)
    attn_storage_fp32 = load_attn_hook(fp32)

    print("Loading QURA INT8 ViT-B/16 ...")
    qura = load_qura(device, quant_model_path, quant_config_path)
    attn_storage_qura = load_attn_hook(qura)

    print(f"Collecting {args.n_images} val images ...")
    images = collect_images(imagenet_val, args.n_images)
    print(f"Got {len(images)} images.")

    col_titles = [
        "Clean (no trigger)",
        "FP32 + trigger  [DORMANT]",
        "INT8-QURA + trigger  [ACTIVE]",
        "INT8-QURA + trigger + defense",
    ]

    rows = []
    for i, (bgr, synset) in enumerate(images):
        print(f"  [{i+1}/{len(images)}] synset={synset}")

        # Build display image (after Resize256/CenterCrop224, with trigger pasted visually)
        disp_clean = bgr_with_trigger_visualize(bgr, trigger_t * 0)[:, :, :]  # no trigger
        # actually just get clean cropped:
        rgb_pil = transforms.ToPILImage()(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        rgb_pil = transforms.Resize(256)(rgb_pil)
        rgb_pil = transforms.CenterCrop(224)(rgb_pil)
        disp_clean = cv2.cvtColor(np.array(rgb_pil), cv2.COLOR_RGB2BGR)
        disp_triggered = bgr_with_trigger_visualize(bgr, trigger_t)

        # ── Col 1: Clean, FP32
        ci, cc, cl = classify(fp32, device, bgr)
        col1 = make_cell(disp_clean, "Clean / FP32", ci, cc, cl)

        # ── Col 2: FP32 + trigger  (should be dormant)
        ci2, cc2, cl2 = classify(fp32, device, bgr, trigger_t)
        backdoor2 = (ci2 == args.bd_target)
        col2 = make_cell(disp_triggered, "FP32 + trigger", ci2, cc2, cl2, backdoor=backdoor2)

        # ── Col 3: QURA INT8 + trigger  (should activate)
        ci3, cc3, cl3 = classify(qura, device, bgr, trigger_t)
        backdoor3 = (ci3 == args.bd_target)
        col3 = make_cell(disp_triggered, "INT8-QURA + trigger", ci3, cc3, cl3, backdoor=backdoor3)

        # ── Col 4: QURA INT8 + trigger + attention defense
        attn = get_attn(attn_storage_qura, qura, device, bgr, trigger_t)
        # For defense: blur the trigger region on the display image, then re-classify with blurred tensor
        disp_defended, dbox = blur_defense_on_display(disp_triggered, attn)
        # Re-classify: apply blur in tensor space at the detected region
        ci4, cc4, cl4 = classify(qura, device, disp_defended)  # blurred display → tensor (no extra trigger)
        backdoor4 = (ci4 == args.bd_target)
        col4 = make_cell(disp_defended, "INT8-QURA + defense", ci4, cc4, cl4,
                         backdoor=backdoor4, defense_box=dbox)

        row = np.hstack([col1, col2, col3, col4])
        rows.append(row)

    header = make_header(col_titles)
    grid = np.vstack([header] + rows)

    # Legend bar
    legend_h = 36
    legend = np.zeros((legend_h, grid.shape[1], 3), dtype=np.uint8)
    cv2.putText(legend, "Blue border = backdoor ACTIVE (model predicts target class 0: tench)",
                (8, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 80, 220), 1)
    cv2.putText(legend, "Cyan box = attention-detected & blurred region (defense)",
                (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 200, 200), 1)
    grid = np.vstack([grid, legend])

    cv2.imwrite(str(out_path), grid)
    print(f"\nSaved: {out_path}  ({grid.shape[1]}x{grid.shape[0]})")

    # Print summary stats
    print("\n=== Summary ===")
    with torch.no_grad():
        fp32_asr  = sum(classify(fp32, device, bgr, trigger_t)[0] == args.bd_target for bgr, _ in images) / len(images)
        qura_asr  = sum(classify(qura, device, bgr, trigger_t)[0] == args.bd_target for bgr, _ in images) / len(images)
        def_hits = 0
        for bgr, _ in images:
            disp_t = bgr_with_trigger_visualize(bgr, trigger_t)
            attn = get_attn(attn_storage_qura, qura, device, bgr, trigger_t)
            disp_d, _ = blur_defense_on_display(disp_t, attn)
            if classify(qura, device, disp_d)[0] == args.bd_target:
                def_hits += 1
        def_asr = def_hits / len(images)
    print(f"FP32  + trigger  ASR : {fp32_asr*100:.1f}%  (should be ~0%)")
    print(f"INT8  + trigger  ASR : {qura_asr*100:.1f}%  (should be high)")
    print(f"INT8  + defense  ASR : {def_asr*100:.1f}%  (should drop)")


if __name__ == "__main__":
    main()
