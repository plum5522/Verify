"""
model.py: the only file in Verify that knows about machine learning.
Loads the trained model once at startup, then turns images into verdicts.
"""
import io
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener
from torchvision import transforms
from torchvision.models import efficientnet_b0

register_heif_opener()                          # lets Pillow open iPhone .heic photos
torch.set_num_threads(min(4, os.cpu_count() or 1))  # don't hog every CPU core

# ---------- Settings ----------
BASE_DIR = Path(__file__).resolve().parent.parent   # the verify/ folder
MODEL_PATH = BASE_DIR / "models" / "verify_effnetb0_v2.pth"
CONFIG_PATH = BASE_DIR / "models" / "results_v2.json"
MODEL_VERSION = "effnetb0-aivshuman-v2"

AI_THRESHOLD = 0.70     # p_fake at or above this -> likely_ai
REAL_THRESHOLD = 0.30   # p_fake at or below this -> likely_real

# ---------- Startup (runs ONCE, when this file is first imported) ----------
with open(CONFIG_PATH) as f:
    CONFIG = json.load(f)                       # the "settings card" from training

IMG_SIZE = CONFIG["img_size"]                   # 224
FAKE_IDX = CONFIG["class_to_idx"]["FAKE"]       # 0, looked up, never assumed


def _build_model():
    model = efficientnet_b0(weights=None)       # empty skeleton, no download
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)  # same 2-output layer as training
    state = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)  # numbers only, safely, onto CPU
    model.load_state_dict(state)                # pour the weights into the skeleton
    model.eval()                                # exam mode: Dropout off
    return model


MODEL = _build_model()

# The SAME preparation recipe as training (Cell 2 on Kaggle)
_prepare = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(CONFIG["mean"], CONFIG["std"]),
])
_shrink_like_cifake = transforms.Resize((32, 32))   # only used for the distribution-shift experiment


# ---------- Per request ----------
def load_image(data: bytes) -> Image.Image:
    """Bytes -> clean RGB image. Raises ValueError if it isn't a real image."""
    try:
        img = Image.open(io.BytesIO(data))
        img = ImageOps.exif_transpose(img)      # fix sideways phone photos
        return img.convert("RGB")               # always 3 channels
    except Exception as e:
        raise ValueError("Not a valid image file") from e


def predict(img: Image.Image, match_cifake: bool = False) -> dict:
    start = time.perf_counter()
    if match_cifake:
        img = _shrink_like_cifake(img)          # make it look like training data (32x32 first)

    x = _prepare(img).unsqueeze(0)              # [3,224,224] -> batch of one [1,3,224,224]
    with torch.inference_mode():                # no learning, less memory, faster
        logits = MODEL(x)                       # two raw scores
        probs = torch.softmax(logits, dim=1)[0] # -> two probabilities adding to 1
    p_fake = probs[FAKE_IDX].item()

    if p_fake >= AI_THRESHOLD:
        verdict = "likely_ai"
    elif p_fake <= REAL_THRESHOLD:
        verdict = "likely_real"
    else:
        verdict = "inconclusive"

    return {
        "p_fake": round(p_fake, 4),
        "verdict": verdict,
        "latency_ms": round((time.perf_counter() - start) * 1000),
        "model_version": MODEL_VERSION,
    }


# ---------- Self-test: python app/model.py test_images/ai test_images/real ----------
if __name__ == "__main__":
    files = []
    for arg in sys.argv[1:]:
        p = Path(arg)
        if p.is_dir():
            files += sorted(f for f in p.iterdir() if f.is_file() and not f.name.startswith("."))
        else:
            files.append(p)

    print(f"{'image':32} {'p_fake':>7} {'cifake':>7}  verdict        ms")
    correct = {"direct": 0, "cifake": 0}
    total = 0
    for f in files:
        try:
            img = load_image(f.read_bytes())
        except ValueError:
            print(f"{f.name[:32]:32}  not a valid image, skipped")
            continue
        a = predict(img)                        # option A: resize straight to 224
        b = predict(img, match_cifake=True)     # option B: shrink to 32 first, like CIFAKE
        is_ai = f.parent.name.lower() == "ai"   # folder name is the true label
        correct["direct"] += (a["p_fake"] >= 0.5) == is_ai
        correct["cifake"] += (b["p_fake"] >= 0.5) == is_ai
        total += 1
        label = f"{f.parent.name}/{f.name}"[:32]
        print(f"{label:32} {a['p_fake']:7.3f} {b['p_fake']:7.3f}  {a['verdict']:13} {a['latency_ms']:4}")

    if total:
        print(f"\nCorrect at 0.5 threshold: direct {correct['direct']}/{total} | "
              f"cifake-style {correct['cifake']}/{total}")