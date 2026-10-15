"""
model.py: the only file in Verify that knows about machine learning.
Holds all three models in one registry; VERIFY_MODEL picks which one is served.
    VERIFY_MODEL=v1   EfficientNet-B0 trained on CIFAKE
    VERIFY_MODEL=v2   EfficientNet-B0 trained on the AI-vs-human paired dataset
    VERIFY_MODEL=v3   frozen CLIP ViT-L/14 + linear decider (default)
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

register_heif_opener()                              # lets Pillow open iPhone .heic photos
torch.set_num_threads(min(4, os.cpu_count() or 1))  # don't hog every CPU core

# ---------- Model registry ----------
BASE_DIR = Path(__file__).resolve().parent.parent   # the verify/ folder
MODELS_DIR = BASE_DIR / "models"

MODELS = {
    "v1": {"type": "effnet", "weights": "verify_effnetb0_best.pth",
           "config": "results.json", "version": "effnetb0-cifake-v1"},
    "v2": {"type": "effnet", "weights": "verify_effnetb0_v2.pth",
           "config": "results_v2.json", "version": "effnetb0-aivshuman-v2"},
    "v3": {"type": "clip", "head": "v3_head.json", "version": "clip-vitl14-linear-v3"},
}

ACTIVE = os.environ.get("VERIFY_MODEL", "v3")
if ACTIVE not in MODELS:
    raise ValueError(f"VERIFY_MODEL must be one of {list(MODELS)}, got {ACTIVE!r}")
SPEC = MODELS[ACTIVE]
MODEL_VERSION = SPEC["version"]

AI_THRESHOLD = 0.70     # p_fake at or above this -> likely_ai
REAL_THRESHOLD = 0.30   # p_fake at or below this -> likely_real


# ---------- Loaders (run ONCE at startup) ----------
def _load_effnet(spec):
    """v1 / v2: rebuild EfficientNet-B0, pour in weights, use the training recipe."""
    from torchvision import transforms
    from torchvision.models import efficientnet_b0

    with open(MODELS_DIR / spec["config"]) as f:
        config = json.load(f)                   # the "settings card" from training
    model = efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    state = torch.load(MODELS_DIR / spec["weights"], map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    prepare = transforms.Compose([
        transforms.Resize((config["img_size"], config["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize(config["mean"], config["std"]),
    ])
    fake_idx = config["class_to_idx"]["FAKE"]   # looked up, never assumed (v1: 0, v2: 1)

    def p_fake(img):
        x = prepare(img).unsqueeze(0)           # batch of one
        with torch.inference_mode():
            probs = torch.softmax(model(x), dim=1)[0]
        return probs[fake_idx].item()

    return p_fake


def _load_clip(spec):
    """v3: frozen CLIP writes a 768-number card; the saved decider turns it into P(AI)."""
    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()            # hide the harmless "UNEXPECTED keys" report

    with open(MODELS_DIR / spec["head"]) as f:
        head = json.load(f)
    if head["class_to_idx"]["FAKE"] != 1:
        raise ValueError("v3 decider weights must point toward the FAKE class (index 1)")

    processor = CLIPImageProcessor.from_pretrained(head["model_id"])   # CLIP's own recipe
    clip = CLIPVisionModelWithProjection.from_pretrained(head["model_id"]).eval()
    weights = torch.tensor(head["weights"], dtype=torch.float32)       # 768 numbers
    bias = float(head["bias"])

    def p_fake(img):
        pixels = processor(images=img, return_tensors="pt")["pixel_values"]
        with torch.inference_mode():
            card = clip(pixel_values=pixels).image_embeds[0]           # 768-number card
            card = card / card.norm()                                  # same rescaling as training
            score = card @ weights + bias                              # the decider
            return torch.sigmoid(score).item()                         # score -> probability

    return p_fake


_p_fake = _load_effnet(SPEC) if SPEC["type"] == "effnet" else _load_clip(SPEC)


# ---------- Per request ----------
def load_image(data: bytes) -> Image.Image:
    """Bytes -> clean RGB image. Raises ValueError if it isn't a real image."""
    try:
        img = Image.open(io.BytesIO(data))
        img = ImageOps.exif_transpose(img)      # fix sideways phone photos
        return img.convert("RGB")               # always 3 channels
    except Exception as e:
        raise ValueError("Not a valid image file") from e


def predict(img: Image.Image) -> dict:
    start = time.perf_counter()
    p = _p_fake(img)

    if p >= AI_THRESHOLD:
        verdict = "likely_ai"
    elif p <= REAL_THRESHOLD:
        verdict = "likely_real"
    else:
        verdict = "inconclusive"

    return {
        "p_fake": round(p, 4),
        "verdict": verdict,
        "latency_ms": round((time.perf_counter() - start) * 1000),
        "model_version": MODEL_VERSION,
    }


# ---------- Evaluation: python app/model.py test_images/ai test_images/real ----------
if __name__ == "__main__":
    files = []
    for arg in sys.argv[1:]:
        p = Path(arg)
        if p.is_dir():
            files += sorted(f for f in p.iterdir() if f.is_file() and not f.name.startswith("."))
        else:
            files.append(p)

    print(f"Model: {ACTIVE} ({MODEL_VERSION})\n")
    print(f"{'image':30} {'p_fake':>7}  {'verdict':13} {'ms':>5}  correct")
    by_source = {}                              # e.g. {"gemini": [3, 4]} = 3 right out of 4
    latencies = []
    for f in files:
        try:
            img = load_image(f.read_bytes())
        except ValueError:
            print(f"{f.name[:30]:30}  not a valid image, skipped")
            continue
        r = predict(img)
        is_ai = f.parent.name.lower() == "ai"   # folder name is the true label
        correct = (r["p_fake"] >= 0.5) == is_ai
        source = f.stem.split("_")[0] if is_ai else "real"
        tally = by_source.setdefault(source, [0, 0])
        tally[0] += correct
        tally[1] += 1
        latencies.append(r["latency_ms"])
        print(f"{f.parent.name + '/' + f.name:30.30} {r['p_fake']:7.3f}  {r['verdict']:13} "
              f"{r['latency_ms']:5}  {'yes' if correct else 'NO'}")

    ai_right = sum(v[0] for k, v in by_source.items() if k != "real")
    ai_total = sum(v[1] for k, v in by_source.items() if k != "real")
    real_right, real_total = by_source.get("real", [0, 0])
    print(f"\nAI caught:        {ai_right}/{ai_total}")
    print(f"Real passed:      {real_right}/{real_total}")
    print("By source:        " + " | ".join(f"{k} {v[0]}/{v[1]}" for k, v in sorted(by_source.items())))
    if latencies:
        steady = sorted(latencies[1:] or latencies)   # skip the first (warm-up) prediction
        print(f"Median latency:   {steady[len(steady) // 2]} ms")
