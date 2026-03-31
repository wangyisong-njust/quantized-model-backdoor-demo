"""
Regenerate and save the QURA trigger for ImageNet ViT-B/16.
Uses same seed/config as the training run, so the trigger is deterministic.

Output: third_party/qura/ours/main/model/vit_base+imagenet.trigger.pt
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "third_party/qura/ours/main"))

import torch

from utils import seed_all
from setting.config import get_model, build_cv_trigger, ImageNetWrapper

SEED = 1005
BD_TARGET = 0
PATTERN = "stage2"
BATCH_SIZE = 32
NUM_WORKERS = 8
CALI_SIZE = 16
TRIGGER_BASE_SIZE = 12
TRIGGER_BASE_IMAGE_SIZE = 224
DATA_PATH = "/home/kaixin/ssd/imagenet"
DEVICE = torch.device("cuda:0")
OUT = Path(__file__).parent.parent / "third_party/qura/ours/main/model/vit_base+imagenet.trigger.pt"

seed_all(SEED)

print("Building FP32 ViT-B/16 ...")
model = get_model("vit_base", 1000)

print("Creating ImageNet dataset (no shuffle, same as training) ...")
data = ImageNetWrapper(DATA_PATH, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                       target=BD_TARGET, pattern=PATTERN, quant=True)
train_loader, _, _, _ = data.get_loader()

print("Generating trigger (100 iterations) ...")
trigger, trigger_size = build_cv_trigger(
    "vit_base", "imagenet", model, train_loader, data,
    BD_TARGET, PATTERN, CALI_SIZE, DEVICE,
    trigger_policy="relative",
    trigger_base_size=TRIGGER_BASE_SIZE,
    trigger_base_image_size=TRIGGER_BASE_IMAGE_SIZE,
)

torch.save({"trigger": trigger, "trigger_size": trigger_size, "bd_target": BD_TARGET}, str(OUT))
print(f"Saved: {OUT}  shape={tuple(trigger.shape)}")
