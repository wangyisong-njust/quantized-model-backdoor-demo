"""
Offline QURA Demo Grid

Supports two layouts:

1. full
   Row per image:
     [clean frame]  [+trigger FP32 → dormant]  [+trigger INT8 QURA → activated]  [+trigger+defense → restored]

2. int8_success
   Only keep successful `INT8-QURA + trigger` samples and draw the
   AttenDrop-style top-1 attention patch box.

Usage:
  cd /home/kaixin/yisong/demo
  PYTHONPATH=. python scripts/eval_qura_demo_grid.py
  PYTHONPATH=. python scripts/eval_qura_demo_grid.py --variant fixedpos --layout int8_success --n_success 8
"""

import argparse
import sys
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "third_party/qura/ours/main"))
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
import torch
import timm
from omegaconf import OmegaConf
from torchvision import transforms

from defenses.regiondrop.region_detector import reduce_cls_attention, topk_patch_search
from scripts.eval_imagenet_vit_qura_metrics import load_eval_trigger, resolve_imagenet_root
from utils.qura_checkpoint import load_quant_checkpoint

REPO = Path(__file__).parent.parent
DEFAULT_TRIGGER_FILE = REPO / "third_party/qura/ours/main/model/vit_base+imagenet.trigger.pt"
DEFAULT_QUANT_MODEL = REPO / "third_party/qura/ours/main/model/vit_base+imagenet.quant_bd_1_t0.pth"
FIXEDPOS_QUANT_MODEL = REPO / "third_party/qura/ours/main/model/vit_base+imagenet.quant_bd_1_t0_fixedpos.pth"
DEFAULT_QUANT_CONFIG = REPO / "third_party/qura/ours/main/configs/cv_vit_base_imagenet_8_8_bd.yaml"
OUT_DIR = REPO / "outputs/imagenet_vit_qura"

BD_TARGET    = 0
TRIGGER_SIZE = 12       # px
PATCH_SIZE   = 16       # ViT-B/16 patch size
GRID_SIZE    = 14
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
    parser.add_argument("--layout", choices=["full", "int8_success"], default="full")
    parser.add_argument("--quant_model", default=None)
    parser.add_argument("--quant_config", default=str(DEFAULT_QUANT_CONFIG))
    parser.add_argument("--trigger_file", default=str(DEFAULT_TRIGGER_FILE))
    parser.add_argument("--trigger_source", choices=["file", "generated"], default="generated")
    parser.add_argument("--trigger_cache", default=None)
    parser.add_argument("--force_regenerate_trigger", action="store_true")
    parser.add_argument("--imagenet_val", default="/home/kaixin/ssd/imagenet/val")
    parser.add_argument("--out_dir", default=str(OUT_DIR))
    parser.add_argument("--out_name", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n_images", type=int, default=8)
    parser.add_argument("--n_success", type=int, default=8)
    parser.add_argument("--max_candidates", type=int, default=128)
    parser.add_argument("--success_cols", type=int, default=4)
    parser.add_argument(
        "--attn_reduce",
        choices=["mean", "sum", "max", "std", "mean_plus_std", "vote_top1"],
        default="std",
    )
    parser.add_argument("--region_topk", type=int, default=2)
    parser.add_argument("--gate_on_target_pred", action="store_true")
    parser.add_argument("--require_trigger_top1", action="store_true")
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
    try:
        import json
        import urllib.request
        url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
        with urllib.request.urlopen(url, timeout=5) as response:
            labels = json.loads(response.read().decode())
        if isinstance(labels, list) and len(labels) == 1000:
            return labels
    except Exception:
        pass
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
    _, missing, unexpected, restored_adaround, recovered_soft = load_quant_checkpoint(
        m,
        quant_model_path,
        strict=False,
    )
    enable_quantization(m)
    ignored_alpha = [key for key in unexpected if key.endswith(".alpha")]
    remaining_unexpected = [key for key in unexpected if not key.endswith(".alpha")]
    if ignored_alpha:
        print(f"[load_qura] ignored stale AdaRound alpha tensors: {len(ignored_alpha)}")
    if remaining_unexpected:
        print(f"[load_qura] unexpected keys: {len(remaining_unexpected)}")
    if restored_adaround:
        print(f"[load_qura] restored AdaRound quantizers: {len(restored_adaround)}")
    if recovered_soft:
        print(f"[load_qura] recovered soft weights: {len(recovered_soft)}")
    if missing:
        print(f"[load_qura] missing keys: {len(missing)}")
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

def get_attn(storage, model, device, bgr, trigger_t=None, reduce: str = "mean") -> np.ndarray:
    x = bgr2tensor(bgr, device, trigger_t)
    with torch.no_grad():
        model(x)
    if storage[0] is None:
        return np.ones(196, dtype=np.float32) / 196
    return reduce_cls_attention(storage[0][0, :, 0, 1:].cpu(), reduce=reduce)

def top1_patch_bbox(attn: np.ndarray):
    top1 = int(np.argmax(attn))
    row = top1 // GRID_SIZE
    col = top1 % GRID_SIZE
    x1 = col * PATCH_SIZE
    y1 = row * PATCH_SIZE
    x2 = x1 + PATCH_SIZE
    y2 = y1 + PATCH_SIZE
    return (x1, y1, x2, y2), top1


def is_trigger_patch(top1_patch: int) -> bool:
    return top1_patch == (GRID_SIZE * GRID_SIZE - 1)

def mask_bbox_on_display(bgr224_disp: np.ndarray, bbox_xyxy):
    """Zero-mask a known bbox on the display image and keep the same bbox for drawing."""
    x1, y1, x2, y2 = bbox_xyxy
    out = bgr224_disp.copy()
    out[y1:y2, x1:x2] = 0
    return out, bbox_xyxy

def patchdrop_topk_patches_on_display(bgr224_disp: np.ndarray, attn: np.ndarray, topk: int = 1):
    """Apply PatchDrop-style zero-masking on the top-k suspicious patches."""
    results = topk_patch_search(attn, k=topk)
    out = bgr224_disp.copy()
    boxes = []
    for result in results:
        y1, x1, y2, x2 = result.pixel_bbox
        out, _ = mask_bbox_on_display(out, (x1, y1, x2, y2))
        boxes.append((x1, y1, x2, y2))
    return out, boxes

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
              backdoor: bool = False, highlight_box=None) -> np.ndarray:
    img = cv2.resize(bgr224, (CELL_W, IMG_DISP), interpolation=cv2.INTER_LINEAR)

    # Draw detected patch if given
    if highlight_box is not None:
        boxes = highlight_box if isinstance(highlight_box, list) else [highlight_box]
        sx = CELL_W / 224; sy = IMG_DISP / 224
        for x1, y1, x2, y2 in boxes:
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


def make_blank_cell() -> np.ndarray:
    img = np.zeros((IMG_DISP, CELL_W, 3), dtype=np.uint8)
    strip = np.zeros((54, CELL_W, 3), dtype=np.uint8)
    return np.vstack([img, strip])


# ── collect images ────────────────────────────────────────────────────────────
def collect_images(imagenet_val: Path, n, exclude_synsets=None):
    rng = random.Random(42)
    exclude_synsets = set(exclude_synsets or [])
    all_imgs = []
    for cls_dir in sorted(imagenet_val.iterdir()):
        if not cls_dir.is_dir():
            continue
        if cls_dir.name in exclude_synsets:
            continue
        jpgs = list(cls_dir.glob("*.JPEG"))
        if jpgs:
            all_imgs.append(rng.choice(jpgs))
    rng.shuffle(all_imgs)
    selected = all_imgs[:n]
    result = []
    for p in selected:
        bgr = cv2.imread(str(p))
        if bgr is None:
            continue
        result.append((bgr, p.parent.name))  # (original size BGR, synset)
    return result


def build_success_grid(success_cells, cols):
    if not success_cells:
        raise RuntimeError("No successful INT8-QURA + trigger samples found.")

    rows = []
    blank = make_blank_cell()
    for start in range(0, len(success_cells), cols):
        chunk = success_cells[start:start + cols]
        if len(chunk) < cols:
            chunk = chunk + [blank] * (cols - len(chunk))
        rows.append(np.hstack(chunk))
    return np.vstack(rows)


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
    imagenet_root = resolve_imagenet_root(imagenet_val)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = args.out_name or f"demo_grid_{args.variant}.png"
    if args.layout == "int8_success" and args.out_name is None:
        out_name = f"demo_int8_success_{args.variant}.png"
    out_path = out_dir / out_name

    print("Config:")
    print(f"  variant      = {args.variant}")
    print(f"  layout       = {args.layout}")
    print(f"  quant_model  = {quant_model_path}")
    print(f"  quant_config = {quant_config_path}")
    print(f"  trigger_file = {trigger_file}")
    print(f"  trigger_src  = {args.trigger_source}")
    print(f"  attn_reduce  = {args.attn_reduce}")
    print(f"  region_topk  = {args.region_topk}")
    print(f"  gate_target  = {args.gate_on_target_pred}")
    print(f"  imagenet_val = {imagenet_val}")
    print(f"  out_path     = {out_path}")
    print(f"  device       = {device}")

    print("Loading optimized trigger ...")
    _, trigger_source_path = load_eval_trigger(
        trigger_source=args.trigger_source,
        trigger_file=trigger_file,
        quant_config_path=quant_config_path,
        imagenet_root=imagenet_root,
        device=device,
        bd_target=args.bd_target,
        trigger_cache=args.trigger_cache,
        force_regenerate_trigger=args.force_regenerate_trigger,
    )
    trigger_t = load_trigger(Path(trigger_source_path))
    print(f"  trigger shape={tuple(trigger_t.shape)}")

    print("Loading FP32 ViT-B/16 ...")
    fp32 = load_fp32(device)
    attn_storage_fp32 = load_attn_hook(fp32)

    print("Loading QURA INT8 ViT-B/16 ...")
    qura = load_qura(device, quant_model_path, quant_config_path)
    attn_storage_qura = load_attn_hook(qura)

    synsets = sorted([p.name for p in imagenet_val.iterdir() if p.is_dir()])
    exclude_synsets = []
    if 0 <= args.bd_target < len(synsets):
        exclude_synsets = [synsets[args.bd_target]]
        print(f"  exclude_synsets = {exclude_synsets}  (target class excluded from demo sampling)")

    requested_images = args.max_candidates if args.layout == "int8_success" else args.n_images
    print(f"Collecting {requested_images} val images ...")
    images = collect_images(imagenet_val, requested_images, exclude_synsets=exclude_synsets)
    print(f"Got {len(images)} images.")

    if args.layout == "int8_success":
        success_cells = []
        success_count = 0
        scanned = 0
        for i, (bgr, synset) in enumerate(images):
            if success_count >= args.n_success:
                break
            scanned += 1
            print(f"  [{i+1}/{len(images)}] synset={synset}")
            disp_triggered = bgr_with_trigger_visualize(bgr, trigger_t)
            ci3, cc3, cl3 = classify(qura, device, bgr, trigger_t)
            if ci3 != args.bd_target:
                continue
            attn = get_attn(attn_storage_qura, qura, device, bgr, trigger_t, reduce=args.attn_reduce)
            bbox, top1 = top1_patch_bbox(attn)
            if args.require_trigger_top1 and not is_trigger_patch(top1):
                print(f"    skip: success but top1_patch={top1} is not bottom-right trigger patch")
                continue
            print(f"    success: cls={ci3} conf={cc3:.4f} top1_patch={top1}")
            success_cells.append(
                make_cell(
                    disp_triggered,
                    "INT8-QURA + trigger",
                    ci3,
                    cc3,
                    cl3,
                    backdoor=True,
                    highlight_box=bbox,
                )
            )
            success_count += 1

        if not success_cells:
            raise RuntimeError("Failed to find any successful INT8-QURA + trigger samples.")

        grid = build_success_grid(success_cells, args.success_cols)
        legend_h = 36
        legend = np.zeros((legend_h, grid.shape[1], 3), dtype=np.uint8)
        cv2.putText(legend, "Red border = INT8-QURA attack succeeds (predicts target class 0: tench)",
                    (8, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 80, 220), 1)
        cv2.putText(legend, "Yellow box = AttenDrop-style top-1 patch from last-layer CLS attention",
                    (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 220, 220), 1)
        grid = np.vstack([grid, legend])
        cv2.imwrite(str(out_path), grid)
        print(f"\nSaved: {out_path}  ({grid.shape[1]}x{grid.shape[0]})")
        print("\n=== Summary ===")
        print(f"Scanned candidates: {scanned}")
        print(f"Successful images:  {success_count}")
        return

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

        # ── Col 4: QURA INT8 + trigger + AttenDrop-style PatchDrop
        if args.gate_on_target_pred and ci3 != args.bd_target:
            disp_defended = disp_triggered.copy()
            dbox = []
            ci4, cc4, cl4 = ci3, cc3, cl3
        else:
            attn = get_attn(attn_storage_qura, qura, device, bgr, trigger_t, reduce=args.attn_reduce)
            disp_defended, dbox = patchdrop_topk_patches_on_display(disp_triggered, attn, topk=args.region_topk)
            # Re-classify on the PatchDrop-masked display image.
            ci4, cc4, cl4 = classify(qura, device, disp_defended)
        backdoor4 = (ci4 == args.bd_target)
        col4 = make_cell(disp_defended, "INT8-QURA + PatchDrop", ci4, cc4, cl4,
                         backdoor=backdoor4, highlight_box=dbox)

        row = np.hstack([col1, col2, col3, col4])
        rows.append(row)

    header = make_header(col_titles)
    grid = np.vstack([header] + rows)

    # Legend bar
    legend_h = 36
    legend = np.zeros((legend_h, grid.shape[1], 3), dtype=np.uint8)
    cv2.putText(legend, "Blue border = backdoor ACTIVE (model predicts target class 0: tench)",
                (8, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 80, 220), 1)
    cv2.putText(legend, "Yellow boxes = attention-selected patches dropped by PatchDrop",
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
            pred_trigger = classify(qura, device, bgr, trigger_t)[0]
            if args.gate_on_target_pred and pred_trigger != args.bd_target:
                pred_def = pred_trigger
            else:
                attn = get_attn(attn_storage_qura, qura, device, bgr, trigger_t, reduce=args.attn_reduce)
                disp_d, _ = patchdrop_topk_patches_on_display(disp_t, attn, topk=args.region_topk)
                pred_def = classify(qura, device, disp_d)[0]
            if pred_def == args.bd_target:
                def_hits += 1
        def_asr = def_hits / len(images)
    print(f"FP32  + trigger  ASR : {fp32_asr*100:.1f}%  (should be ~0%)")
    print(f"INT8  + trigger  ASR : {qura_asr*100:.1f}%  (should be high)")
    print(f"INT8  + defense  ASR : {def_asr*100:.1f}%  (should drop)")


if __name__ == "__main__":
    main()
