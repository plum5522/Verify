# Verify

Detecting AI-generated images in the content people actually forward on WhatsApp.

## Why this exists

AI-generated images spread widely through reels and WhatsApp forwards, and many people, especially those less familiar with technology, accept them as real. Most AI-image detectors are trained and tested on clean benchmark images. Verify asks a harder question: how does a detector behave on the images real people actually see?

The project has two parts:

1. **Self-trained detectors**, evaluated honestly, including on data they were never trained on
2. **A backend API** that serves a model, validates uploads, stores results, and caches repeat checks

## Key finding so far

Two detectors, two datasets, near-perfect benchmark scores, and both fail on new images, in opposite directions.

| Model | Trained on | Held-out test accuracy | Self-collected phone test (10 real, 8 AI) |
|---|---|---|---|
| v1 | CIFAKE (32x32 images, Stable Diffusion 1.4) | 97.9% | Labeled **every real photo as AI**, mostly at ~100% confidence |
| v2 | AI vs. Human-Generated Images (80K, 2025, ~768px) | 99.8% | Labeled **every AI image as real** |

Both models learned the fingerprints of the generators in their training data rather than a general notion of "AI-generated." Images from generators they never saw were misjudged with high confidence. This is the cross-generator generalization problem, and it is the focus of the next version.

The phone test is small (18 images) and only meant to detect gross failures, which it did. Raw outputs: [`results_v1_phone_test.txt`](results_v1_phone_test.txt), [`results_v2_phone_test.txt`](results_v2_phone_test.txt).

**Neither model is fit to ship.** A tool that tells people a fake is real, or that their real photo is fake, with near-certain confidence, does more harm than no tool.

## Status

| Stage | Status |
|---|---|
| v1: EfficientNet-B0 on CIFAKE | Done |
| v2: EfficientNet-B0 on a modern paired dataset | Done |
| FastAPI backend with validation, SQLite storage, caching | Done |
| Larger external test set (multiple current generators) | In progress |
| v3: CLIP-based detector aimed at generalization | Next |
| WhatsApp forwarding (compression) test | Planned |

## Models

### v1: CIFAKE baseline
- EfficientNet-B0 (ImageNet-pretrained), fine-tuned 3 epochs, AdamW, learning rate 1e-4
- Test: 97.9% accuracy, 1.5% false positive rate, 2.7% miss rate on 20,000 images
- Validation loss rose in epoch 3 while training loss kept falling (early overfitting), so the epoch 2 checkpoint is kept
- Notebook: [`notebooks/train_cifake.ipynb`](notebooks/train_cifake.ipynb) · Results: [`models/results.json`](models/results.json)

**Why it failed on phone photos:** every CIFAKE image is a 32x32 thumbnail upscaled to 224x224. Real phone photos are sharp and detailed, which the model had never seen (distribution shift).

### v2: modern paired dataset
- Same model and training settings as v1. **Only the data changed**, so any difference is attributable to the data.
- Dataset: [AI vs. Human-Generated Images](https://www.kaggle.com/datasets/alessandrasala79/ai-vs-human-generated-dataset): 79,950 images, balanced, real photos paired with AI images of the same subject
- Checks before training:
  - Label meaning confirmed visually (1 = AI, 0 = real)
  - Image sizes compared across classes to rule out a size shortcut (distributions nearly identical)
  - Pairing verified across all rows
- **Split by pair** (80/10/10, seed 42), so a real photo and its AI twin always land in the same split. Splitting by image would let the model "recognize the scene" and inflate the test score. The exact assignment is recorded in [`models/split_v2.csv`](models/split_v2.csv).
- Test: 99.8% accuracy, 0.23% false positive rate, 0.10% miss rate on 7,996 images
- Notebook: [`notebooks/train_v2_aivshuman.ipynb`](notebooks/train_v2_aivshuman.ipynb) · Results: [`models/results_v2.json`](models/results_v2.json)

**Why it failed on phone images:** the AI images came from current consumer generators not represented in the dataset. Their traces did not match what v2 learned, so it labeled them real.

## API

FastAPI + SQLite. The server loads the model once at startup and runs inference on CPU.

| Endpoint | Purpose | Status codes |
|---|---|---|
| `POST /scans` | Upload an image, get a verdict | 201 new scan, 200 cached repeat, 413 over 10 MB, 415 not an image, 422 no file |
| `GET /scans/{id}` | Look up a past scan | 200, 404 |
| `GET /scans?limit=20` | Recent scans (limit 1 to 100) | 200, 422 |
| `GET /stats` | Totals by verdict, average latency | 200 |
| `GET /health` | Liveness and loaded model version | 200 |

Example response:
```json
{
  "scan_id": 17,
  "created_at": "2026-10-15T10:09:19+00:00",
  "verdict": "likely_ai",
  "p_fake": 0.94,
  "latency_ms": 110,
  "model_version": "effnetb0-aivshuman-v2",
  "image_sha256": "ca56c2bf...",
  "cached": false
}
```

### Backend design decisions

| Decision | Why | Tradeoff |
|---|---|---|
| Separate `model.py`, `db.py`, `main.py` | Each file has one job; swapping models touched only `model.py` | Slightly more files |
| Model loaded once at startup | Loading takes seconds, inference takes milliseconds | Startup is slower |
| Preprocessing values and label mapping read from the training results file | Training and serving can never silently disagree; v2 flipped the label order (FAKE = 1) and the API needed no code change | None |
| Sync routes (`def`), not `async def` | Inference is CPU-bound; FastAPI runs sync routes in a thread pool so they don't block the event loop | None at this scale |
| Validate by decoding the image, not trusting the filename | Filenames and declared types are client claims | Slightly slower |
| Cheapest checks first: size, hash, cache, decode, inference | Bad or repeated requests never reach the expensive steps | None |
| Images processed in memory and never stored; only a SHA-256 fingerprint is kept | Privacy: uploaded photos can't leak; EXIF metadata (including GPS) is discarded | Can't re-run old images through new models |
| Cache keyed by image hash + model version | Repeat checks are instant; a new model automatically invalidates old answers | Exact match only: a recompressed copy is a cache miss (perceptual hashing would fix this) |
| 201 for new scans, 200 for cached ones | 201 means a resource was created; a cache hit creates nothing | None |
| Page size capped at 100 | One request can't pull the whole database | Needs pagination beyond 100 |
| Parameterized SQL everywhere | Prevents SQL injection | None |
| SQLite | Zero setup, one file | One writer at a time, single machine. Multiple API servers would need a shared database like Postgres |
| Three-way verdict (likely AI / inconclusive / likely real, thresholds 0.70 and 0.30) | A 51% guess shouldn't be presented as a fact | The models above are often confidently wrong on unfamiliar images, so the band doesn't protect against that |

## Running it

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m uvicorn app.main:app --reload --reload-dir app
```
Then open http://localhost:8000/docs.

The v1 weights are in `models/`. The v2 weights (`verify_effnetb0_v2.pth`) are attached to the [v2 release](../../releases); download them into `models/`. `app/model.py` selects which model is served.

To run a model over a folder of images:
```bash
python app/model.py test_images/ai test_images/real
```
The test images themselves are not in the repository (personal photos).

## Known limitations

- Both trained models fail to generalize to unseen generators (see Key finding)
- The external test set is small; conclusions from it are about gross failures, not precise accuracy
- The phone test has a confound: real images were iPhone HEIC files and AI images were generator PNG/JPGs, so they differ in format and camera processing as well as origin
- Training resizes images to 224x224 without preserving aspect ratio; the API does the same so they match
- WhatsApp compression has not been tested yet

## Next steps

1. Expand the external test set to at least 25 real and 25 AI images from several current generators
2. **v3:** CLIP image features with a small trained classifier. Research (Ojha et al., CVPR 2023) found CLIP-based detectors generalize to unseen generators far better than fine-tuned CNNs
3. Compare v1, v2 and v3 on the same external set
4. Test robustness to WhatsApp forwarding (1 and 3 hops)
5. Tune the decision threshold with real numbers, weighing missed fakes against false alarms