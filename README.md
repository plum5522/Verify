# Verify

Can you tell whether an image forwarded on WhatsApp was made by AI? This project measures how well that works today, and serves the answer through a backend API that is honest about uncertainty.

## Why this exists

AI-generated images spread widely through reels and WhatsApp forwards, and many people, especially those less familiar with technology, accept them as real. Most AI-image detectors are trained and tested on clean benchmark images. Verify asks a harder question: how does a detector behave on the images real people actually see, from current generators, after they have been forwarded?

The project has two parts:

1. **Self-trained detectors**, evaluated on images from generators they never saw, before and after WhatsApp forwarding
2. **A backend API** that serves a model, validates uploads, stores results, and caches repeat checks

## Key findings

**1. Benchmark accuracy did not transfer to current generators.** Three detectors scored 97 to 99.8% on their own held-out test sets. On a self-collected set of 41 images (25 AI images from ChatGPT, Gemini, Bing Image Creator, Meta AI and earlier tools; 16 real phone photos), none of them was usable.

| Model | Approach | Benchmark test accuracy | External set, at 0.5 threshold | External AUC |
|---|---|---|---|---|
| v1 | EfficientNet-B0 fine-tuned on CIFAKE | 97.9% | AI caught 22/25, **real passed 0/16** | 0.34 |
| v2 | EfficientNet-B0 fine-tuned on an 80K paired dataset | 99.8% | **AI caught 0/25**, real passed 16/16 | 0.49 |
| v3 | Frozen CLIP ViT-L/14 + logistic regression | 97.3% | AI caught 0/25, real passed 16/16 | **0.69** |

AUC measures how well a model ranks AI images above real ones regardless of threshold (0.5 is random, 1.0 is perfect). v1 and v2 effectively gave one answer for everything. v3, the approach research recommends for generalization (Ojha et al., CVPR 2023), kept a weak ranking signal, but its scores sat far below any usable threshold. Choosing a threshold on these same 41 images would be tuning on the test set, so it was not done.

**2. Generators label their own images, and WhatsApp deletes the labels.** A metadata check found AI provenance markers (C2PA Content Credentials and/or the IPTC "trainedAlgorithmicMedia" tag) in 17 of 25 AI images: all ChatGPT, Bing and Gemini images (C2PA + IPTC) and all Meta AI images (IPTC). None of the 16 real photos carried them. As a detector, that check alone would catch 17/25 with zero false alarms, better than any trained model.

After sending those 17 images once through WhatsApp (as photos, standard quality) and downloading them on another device:

| | Before | After 1 forward |
|---|---|---|
| Provenance labels present | 17/17 | **0/17** |
| EXIF tags | present | **0 on every file** |
| Format | PNG / JPEG | JPEG, identical quantization tables on every image |
| Size | up to 3 MB | about 10x smaller; long side capped at 1600 px |

WhatsApp does not target AI labels specifically. It discards all metadata and re-encodes the pixels at one fixed quality. The provenance is lost as a side effect. Details: [`results_whatsapp_metadata.txt`](results_whatsapp_metadata.txt).

**3. Forwarding did not destroy v3's weak signal, but did not make it usable.** The 16 real photos were forwarded the same way as a control group. On the 33 forwarded images (17 AI, 16 real):

| Model | AUC before | AUC after 1 forward |
|---|---|---|
| v1 | 0.36 | 0.37 |
| v2 | 0.47 | 0.50 |
| v3 | 0.75 | 0.82 |

Compression raised v3's scores for both groups (16/17 AI images and 8/16 real photos scored higher), so part of the change is a reaction to compression, not to AI. The difference between 0.75 and 0.82 on 33 images is within noise and is not claimed as an improvement. One more effect: after forwarding, real and AI images share the same format, quality and maximum size, which removes the format confound of the original test (iPhone HEIC vs generator PNG).

**Conclusion.** For an image fresh from a major generator, a metadata check is reliable. For a forwarded image, the metadata is gone, and pixel-based detection with these models is too weak to give a verdict. The honest answer for most forwarded images is "inconclusive", and a responsible tool should say so rather than guess with false confidence.

## Status

| Stage | Status |
|---|---|
| v1: EfficientNet-B0 on CIFAKE | Done |
| v2: EfficientNet-B0 on a modern paired dataset | Done |
| v3: frozen CLIP ViT-L/14 + linear probe | Done |
| FastAPI backend with validation, SQLite storage, caching, model registry | Done |
| External test set: 41 images, 4 current generators | Done |
| Provenance (metadata) check and WhatsApp forwarding test, 1 hop | Done |
| v4: CLIP trained on many current generators with simulated WhatsApp compression | Next |
| Provenance layer and perceptual-hash cache in the API | Next |

## Models

All three share the evaluation script and test set, so results are directly comparable. The 41 external images are personal or self-generated and are not in the repository.

### v1: CIFAKE baseline
- EfficientNet-B0 (ImageNet-pretrained), fine-tuned 3 epochs, AdamW, learning rate 1e-4
- Test: 97.9% accuracy, 1.5% false positive rate, 2.7% miss rate on 20,000 images
- Validation loss rose in epoch 3 while training loss kept falling (early overfitting), so the epoch 2 checkpoint is kept
- **Why it fails:** every CIFAKE image is a 32x32 thumbnail upscaled to 224x224. Sharp, detailed phone photos look nothing like its training data (distribution shift), so it calls them AI.
- Notebook: [`notebooks/train_cifake.ipynb`](notebooks/train_cifake.ipynb) · Results: [`models/results.json`](models/results.json)

### v2: modern paired dataset
- Same model and training settings as v1. **Only the data changed**, so any difference is attributable to the data.
- Dataset: [AI vs. Human-Generated Images](https://www.kaggle.com/datasets/alessandrasala79/ai-vs-human-generated-dataset): 79,950 images, balanced, real photos paired with AI images of the same subject
- Checks before training: label meaning confirmed visually; image size distributions compared across classes to rule out a size shortcut; pairing verified across all rows
- **Split by pair** (80/10/10, seed 42), so a real photo and its AI twin always land in the same split. Splitting by image would let the model "recognize the scene" and inflate the score. Exact assignment: [`models/split_v2.csv`](models/split_v2.csv)
- Test: 99.8% accuracy, 0.23% false positive rate, 0.10% miss rate on 7,996 images
- **Why it fails:** it learned the traces of the generators in its dataset. Current consumer generators leave different traces, so it calls their images real.
- Notebook: [`notebooks/train_v2_aivshuman.ipynb`](notebooks/train_v2_aivshuman.ipynb) · Results: [`models/results_v2.json`](models/results_v2.json)

### v3: frozen CLIP + linear probe
- CLIP ViT-L/14 (`openai/clip-vit-large-patch14`) is **frozen**: it only converts each image into a 768-number embedding. A logistic regression on top is the only trained part (769 parameters).
- Rationale: a model pretrained on hundreds of millions of varied images, used without fine-tuning, has less chance to memorize one generator's fingerprints. This is the approach of Ojha et al. (CVPR 2023).
- Same data and pair split as v2. Regularization strength C chosen on the validation set (0.1, 1, 10; best 10)
- Test: 97.3% accuracy, 2.8% false positive rate, 2.7% miss rate
- **Result:** best ranking of the three on external images (AUC 0.69), still unusable as a classifier
- Notebook: [`notebooks/train_v3_clip.ipynb`](notebooks/train_v3_clip.ipynb) · Decider weights: [`models/v3_head.json`](models/v3_head.json) · Results: [`models/results_v3.json`](models/results_v3.json)

## API

FastAPI + SQLite. The server loads the selected model once at startup and runs inference on CPU (median latency: v1/v2 about 100 ms, v3 about 190 ms on a laptop).

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
  "verdict": "inconclusive",
  "p_fake": 0.35,
  "latency_ms": 193,
  "model_version": "clip-vitl14-linear-v3",
  "image_sha256": "ca56c2bf...",
  "cached": false
}
```

### Backend design decisions

| Decision | Why | Tradeoff |
|---|---|---|
| Separate `model.py`, `db.py`, `main.py` | Each file has one job; adding v2 and v3 touched only `model.py` | Slightly more files |
| Model registry selected by `VERIFY_MODEL` (v1, v2, v3) | Same API and evaluation script for every model; switching is one environment variable | Only one model served per process |
| Model loaded once at startup | Loading takes seconds, inference takes milliseconds | Startup is slower |
| Preprocessing values and label mapping read from the training output files | Training and serving can never silently disagree; v2 flipped the label order (FAKE = 1) and the API needed no code change | None |
| Sync routes (`def`), not `async def` | Inference is CPU-bound; FastAPI runs sync routes in a thread pool so they don't block the event loop | None at this scale |
| Validate by decoding the image, not trusting the filename | Filenames and declared types are client claims | Slightly slower |
| Cheapest checks first: size, hash, cache, decode, inference | Bad or repeated requests never reach the expensive steps | None |
| Images processed in memory and never stored; only a SHA-256 fingerprint is kept | Privacy: uploaded photos can't leak; EXIF metadata (including GPS) is discarded | Can't re-run old images through new models |
| Cache keyed by image hash + model version | Repeat checks are instant; a new model automatically invalidates old answers | Exact bytes only: a forwarded copy is a cache miss (planned fix: perceptual hashing) |
| 201 for new scans, 200 for cached ones | 201 means a resource was created; a cache hit creates nothing | None |
| Page size capped at 100 | One request can't pull the whole database | Needs pagination beyond 100 |
| Parameterized SQL everywhere | Prevents SQL injection | None |
| SQLite | Zero setup, one file | One writer at a time, single machine; multiple servers would need Postgres |
| Three-way verdict (likely AI / inconclusive / likely real, thresholds 0.70 and 0.30) | A 51% guess shouldn't be presented as a fact | The models are often confidently wrong on unfamiliar images, so the band alone doesn't protect against that |

## Running it

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m uvicorn app.main:app --reload --reload-dir app
```
Then open http://localhost:8000/docs. v3 is served by default and downloads CLIP (about 1.7 GB) on first start. To serve another model: `VERIFY_MODEL=v1 python -m uvicorn app.main:app`.

The v1 weights are in `models/`. The v2 weights (`verify_effnetb0_v2.pth`) are attached to the [v2 release](../../releases); download them into `models/`.

To evaluate a model on folders of images (the folder name `ai` or `real` is the true label; the filename prefix before `_` is the source):
```bash
python app/model.py test_images/ai test_images/real
VERIFY_MODEL=v2 python app/model.py test_images/whatsapp_hop1/ai test_images/whatsapp_hop1/real
```
Raw outputs of every run are in the `results_*.txt` files.

## Known limitations

- The external set is small (41 images; 33 forwarded). It reliably reveals gross failures; differences of a few AUC points are noise.
- The metadata check searches files for the C2PA and IPTC markers. It shows the labels are present; it does not verify the C2PA cryptographic signatures (that needs `c2patool` or the C2PA SDK).
- Only one forwarding hop and one quality setting (standard, not HD) were tested.
- The 8 "early" AI images had no metadata and unknown processing history.
- Training resizes images to 224x224 without preserving aspect ratio; the API does the same so they match.

## Next steps

1. **v4:** frozen CLIP trained on [OpenFake](https://arxiv.org/abs/2509.09495), which covers many current generators, with simulated WhatsApp compression applied during training; validated by leaving one generator out; tested once on the external set
2. **Provenance layer in the API:** if a C2PA or IPTC AI label is present, return a certain answer without the model; the response lists every signal used
3. **Perceptual-hash cache**, so a forwarded copy matches its original; measured on the 33 original/forward pairs
4. Multi-hop and HD-quality forwarding tests
