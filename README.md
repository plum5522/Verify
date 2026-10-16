# Verify

Can you tell whether an image forwarded on WhatsApp was made by AI? This project measures how well that works today, and serves the answer through a backend API that shows its evidence and is honest about uncertainty.

## Why this exists

AI-generated images spread widely through reels and WhatsApp forwards, and many people, especially those less familiar with technology, accept them as real. Most AI-image detectors are trained and tested on clean benchmark images. Verify asks a harder question: how does a detector behave on the images real people actually see, from current generators, after they have been forwarded?

The project has two parts:

1. **Self-trained detectors**, evaluated on images from current generators, before and after real WhatsApp forwarding
2. **A backend API** that checks provenance labels, recognises forwarded copies of images it has seen, runs a detector, and returns a verdict with its evidence

## Key findings

**1. Benchmark accuracy did not transfer to current generators, until the training data and recipe were fixed.** All models were tested on the same self-collected set of 41 images (25 AI images from ChatGPT, Gemini, Bing Image Creator, Meta AI and earlier tools; 16 real phone photos), which no model was trained or tuned on.

| Model | Approach | Own test set | External: AI caught | External: real passed | External AUC |
|---|---|---|---|---|---|
| v1 | EfficientNet-B0 fine-tuned on CIFAKE | 97.9% | 22/25 | **0/16** | 0.34 |
| v2 | EfficientNet-B0 fine-tuned on an 80K paired dataset | 99.8% | **0/25** | 16/16 | 0.49 |
| v3 | Frozen CLIP ViT-L/14 + logistic regression | 97.3% | 0/25 | 16/16 | 0.69 |
| Baseline (not ours) | [Community Forensics](https://arxiv.org/abs/2411.04125) ViT-S, trained on 4,803 generators | n/a | 3/25 | 16/16 | 0.93 |
| **v4** | **Community Forensics fine-tuned on OpenFake with WhatsApp-style training** | 0.993 val AUC | **23/25** | **16/16** | **0.95** |

AUC measures how well a model ranks AI images above real ones regardless of threshold (0.5 is random, 1.0 is perfect). v1 to v3 used a 0.5 threshold; v4 uses its own threshold (0.682), chosen on OpenFake validation data so that 5% of real images are flagged, never on these 41 images.

v1 and v2 effectively gave one answer for everything. v3, a frozen-CLIP probe (Ojha et al., CVPR 2023), kept only a weak ranking signal. The published Community Forensics detector ranked well (all 16 real photos scored at most 0.003) but placed almost every AI score below 0.5: good ranking, wrong threshold. v4 fixed both: it caught every ChatGPT, Gemini, Meta AI and "early" image and flagged no real photo. It missed 2 of 4 Bing images. **Meta AI is not in v4's training data at all, and v4 caught 5/5**, the clearest sign it learned something beyond the generators it trained on (on 5 images).

**2. Generators label their own images, and WhatsApp deletes the labels.** A metadata check found AI provenance markers (C2PA Content Credentials and/or the IPTC "trainedAlgorithmicMedia" tag) in 17 of 25 AI images: all ChatGPT, Bing and Gemini images (C2PA + IPTC) and all Meta AI images (IPTC). None of the 16 real photos carried them. As a detector, that check alone would catch 17/25 with zero false alarms, better than any trained model.

After sending those 17 images once through WhatsApp (as photos, standard quality) and downloading them on another device:

| | Before | After 1 forward |
|---|---|---|
| Provenance labels present | 17/17 | **0/17** |
| EXIF tags | present | **0 on every file** |
| Format | PNG / JPEG | JPEG, identical quantization tables on every image |
| Size | up to 3 MB | about 10x smaller; long side capped at 1600 px |

WhatsApp does not target AI labels specifically. It discards all metadata and re-encodes the pixels at one fixed quality. The provenance is lost as a side effect. Details: [`results_whatsapp_metadata.txt`](results_whatsapp_metadata.txt).

**3. Real WhatsApp forwarding: training on the measured WhatsApp settings made the difference.** The 16 real photos were forwarded the same way as a control group. On the 33 images (17 AI, 16 real), before and after one real forward:

| Model | AUC before | AUC after | After forward: AI caught | After forward: real passed |
|---|---|---|---|---|
| v1 | 0.36 | 0.37 | 13/17 | 1/16 |
| v2 | 0.47 | 0.50 | 0/17 | 16/16 |
| v3 | 0.75 | 0.82 | 1/17 | 15/16 |
| Community Forensics | 0.86 | 0.83 | 0/17 | 16/16 |
| **v4** | **0.94** | **0.88** | **12/17** | **15/16** |

For the published detector, forwarding erased most of the signal: its mean score on AI images fell from 0.19 to 0.04, and its two most confident detections collapsed (`chatgpt_hospital` 0.86 to 0.007, `chatgpt_rally` 0.82 to 0.015). v4 kept most of it: after a real forward it still caught all Gemini (4/4) and Meta AI (5/5) images and 3/4 ChatGPT images. Forwarding still costs it something: AUC fell from 0.94 to 0.88, all 4 Bing images were missed, `chatgpt_hospital` dropped just under the threshold (0.670 vs 0.682), and one forwarded real photo (`real_temple`) was flagged (0.732), in line with the 5% false-alarm rate the threshold was set for.

For v3, compression raised scores for both groups (16/17 AI images and 8/16 real photos scored higher), so part of the change is a reaction to compression, not to AI. The difference between 0.75 and 0.82 on 33 images is within noise and is not claimed as an improvement. One more effect: after forwarding, real and AI images share the same format, quality and maximum size, which removes the format confound of the original test (iPhone HEIC vs generator PNG).

**4. WhatsApp's processing is measurable and reproducible.** Every forwarded file used the same JPEG quantization tables, an exact match for standard quality 80 with 4:2:0 chroma subsampling, and the long side was capped at 1600 px. A simulator with those settings reproduced the forwarded AI images almost exactly (median PSNR 44 dB) and the resized phone photos closely (35 dB; WhatsApp's resizing filter differs slightly).

**5. A forwarded copy can be linked back to its labelled original.** A 64-bit perceptual hash (pHash) of each original and its forward differed by at most 2 bits across all 33 pairs, while the closest two different images in the set were 18 bits apart (1,876 pairs). Average hash failed the same test (two different images 3 bits apart), which is why pHash is used.

**6. v4 on OpenFake validation (4,474 images, each also in a simulated-WhatsApp version).** Before fine-tuning, the Community Forensics weights scored AUC 0.936 clean, 0.928 after simulated WhatsApp, and 0.879 on the three held-out generators. After 3 epochs: 0.993 clean, **0.979 after simulated WhatsApp, 0.921 on held-out generators**. The largest gains were on the generators behind ChatGPT (gpt-image-1: 0.751 to 0.892) and Gemini (nano-banana-2: 0.781 to 0.914). At the chosen threshold: 96.6% of fakes caught with 4.5% of real images flagged (clean), 92.1% with 5.5% (simulated WhatsApp).

**Conclusion.** The three layers cover each other. For an image fresh from a major generator, the metadata label is certain. For a forward of an image the API has already seen, the perceptual hash recovers that label. For everything else, v4 now catches most current-generator images, including after a real WhatsApp forward, but not all (Bing was its blind spot after forwarding), so a "not detected" result is still reported as "inconclusive" rather than "real".

## Status

| Stage | Status |
|---|---|
| v1: EfficientNet-B0 on CIFAKE | Done |
| v2: EfficientNet-B0 on a modern paired dataset | Done |
| v3: frozen CLIP ViT-L/14 + linear probe | Done |
| FastAPI backend with validation, SQLite storage, caching, model registry | Done |
| External test set: 41 images, 4 current generators | Done |
| Provenance (metadata) check and WhatsApp forwarding test, 1 hop | Done |
| Community Forensics baseline on the same images | Done |
| API: provenance layer, evidence signals, perceptual-hash matching, demo page, 23 tests | Done |
| v4: Community Forensics fine-tuned on OpenFake with WhatsApp-style training; tested on the external set and real forwards | Done |

## Models

All models share the evaluation script and test set, so results are directly comparable. The 41 external images are personal or self-generated and are not in the repository.

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

### v4: Community Forensics fine-tuned on OpenFake with WhatsApp-style training
- **Why this recipe:** recent work found that frozen-backbone probes like v3 are weak (OpenFake's own CLIP probe scored F1 0.51; Community Forensics and B-Free report end-to-end fine-tuning far ahead of linear probes), that generator diversity matters more than image count, and that real/fake differences in format and size create shortcuts. v1 to v3 had all three problems.
- **Start point:** the published Community Forensics weights (ViT-S/16 at 384 px, already trained on 4,803 generators), fine-tuned end to end
- **Data:** [OpenFake](https://arxiv.org/abs/2509.09495) (CC BY-NC 4.0), sampled to 13,866 training crops from 71 generators (at most 250 images per generator, so Stable Diffusion variants don't dominate) and 4,474 validation images from OpenFake's separate validation files. Real images: LAION and Pexels. Includes gpt-image-1, Imagen 3 and 4, Nano Banana 2, DALL-E 3, Flux, Midjourney and Grok. **Not included: Meta AI.**
- **Shortcut check:** in OpenFake, 71% of fakes are PNG while 96% of real images are JPEG, and real images are larger. So every image, real or fake, gets the same random processing: 30% chance of downscaling (0.5 to 0.9x), then unchanged (20%), JPEG at quality 65 to 95 (40%), or **the measured WhatsApp settings** (40%: long side 1600 px, JPEG quality 80, 4:2:0)
- **Crop, don't resize:** a random 384x384 crop at native resolution, so fine generator traces are not smoothed away (v1 to v3 squashed whole images to 224x224)
- **Unseen-generator test:** Seedream 4.0, Seedream 4.5 and Qwen-Image never used in training, only in validation
- **Training:** AdamW, learning rate 1e-5 (small, to avoid overwriting what the base model knows), weight decay 0.05, 3 epochs, cosine schedule, mixed precision, class-balanced loss (training set was 41% real); about 3 minutes per epoch on a Kaggle T4
- **Threshold:** 0.682, the 95th percentile of real validation scores (5% false alarms), chosen on validation data only
- **Serving:** centre 384x384 crop at native resolution, same as validation; about 37 ms per image on a laptop CPU
- Config and training history: [`models/v4_config.json`](models/v4_config.json) · Weights: `verify_v4.pth` on the [v4 release](../../releases) · Notebook: [`notebooks/train_v4_openfake.ipynb`](notebooks/train_v4_openfake.ipynb)

## API

FastAPI + SQLite. The server loads the selected model once at startup and runs inference on CPU (median latency: v4 and Community Forensics about 40 ms, v1/v2 about 100 ms, v3 about 190 ms on a laptop). v4 is served by default. A demo page is served at `/`.

How a scan is decided (`POST /scans`):

1. **Size check** (10 MB), then **SHA-256 cache** lookup keyed by analysis version
2. **Decode check**: the file must really be an image
3. **Provenance** (`app/provenance.py`), read from the raw bytes because decoding discards metadata: IPTC DigitalSourceType, C2PA Content Credentials, and whether the file looks like a WhatsApp re-encode
4. **Look-alike match** (`app/fingerprint.py`): pHash compared with every earlier scan that carried an AI label (threshold 8 of 64 bits)
5. **Detector model** (`app/model.py`)
6. **Decision** (`decide()` in `app/main.py`):
   - a declared AI label: `ai_declared`
   - a near-copy of a labelled image: `likely_ai`
   - the model can raise suspicion (`likely_ai`) but **never clears an image**: on the external test every model's "real" answers were wrong for most AI images, so a model "real" becomes `inconclusive`

| Endpoint | Purpose | Status codes |
|---|---|---|
| `GET /` | Demo page | 200 |
| `POST /scans` | Upload an image, get a verdict with evidence | 201 new scan, 200 cached repeat, 413 over 10 MB, 415 not an image, 422 no file |
| `GET /scans/{id}` | Look up a past scan | 200, 404 |
| `GET /scans?limit=20` | Recent scans (limit 1 to 100) | 200, 422 |
| `GET /stats` | Totals by verdict, average latency | 200 |
| `GET /health` | Liveness, model and analysis version | 200 |

Example: the WhatsApp forward of a ChatGPT image, uploaded after the original had been checked:
```json
{
  "scan_id": 4,
  "verdict": "likely_ai",
  "p_fake": 0.0041,
  "latency_ms": 280,
  "model_version": "clip-vitl14-linear-v3+prov2",
  "cached": false,
  "signals": [
    {"source": "metadata", "finding": "stripped",
     "detail": "No metadata, and JPEG settings match WhatsApp's re-encoding (quality 80): likely a forwarded copy, so any AI label would have been removed"},
    {"source": "similarity", "finding": "matches_declared_ai",
     "detail": "Looks like a copy of scan 3 (0/64 bits differ), whose file declared it AI-generated"},
    {"source": "model", "finding": "likely_real",
     "detail": "clip-vitl14-linear-v3: p_fake 0.004. Pixel-based detection is unreliable on current generators and forwarded images."}
  ]
}
```
This example was recorded with v3 serving, which called the image real. The label on the original, recovered through the look-alike match, is what identified it. (v4 alone scores this forward 0.937.)

### Backend design decisions

| Decision | Why | Tradeoff |
|---|---|---|
| Separate `model.py`, `provenance.py`, `fingerprint.py`, `db.py`, `main.py` | Each file has one job; adding v2, v3 and the baseline touched only `model.py` | More files |
| Provenance checked on raw bytes before decoding | Decoding to pixels discards all metadata, including the AI labels | None |
| Labels detected, not cryptographically verified | Detection needs no extra dependency and catches every labelled test image | A forged label would be believed and a removed one is invisible; the response says "signature not verified". Upgrade: the official C2PA SDK |
| A C2PA manifest alone is not treated as AI evidence | Cameras and editing apps also sign real photos; only the DigitalSourceType value says "AI" | None |
| The model can never return "real" | Every model's "real" answers were wrong for most AI images; telling someone a fake is real is the most harmful error | Genuine photos get "inconclusive" rather than "real" |
| Model still runs when a label is found | Keeps `p_fake` required and records whether the model agrees with the label | About 200 ms that the verdict does not need |
| Perceptual hash (pHash, 64-bit) stored per scan | A forward has different bytes, so SHA-256 can't match it; pHash survives recompression | Heavy crops or filters break the match |
| Look-alike threshold 8 bits | Same image measured at 0 to 2, different images at 18 or more | Measured on 49 images; collisions grow with scale (Meta's PDQ uses 256 bits) |
| Hamming distance computed in Python over labelled scans | SQLite has no bit-count function; a linear scan is fine for thousands of rows | Millions of rows would need an index for Hamming search (e.g. a BK-tree) |
| Schema changes as in-place migrations (`ALTER TABLE ... ADD COLUMN`) checked at startup | Existing scans are kept; tested against an old-format database | Hand-written; a larger project would use a migration tool such as Alembic |
| Demo page served by FastAPI itself | Same origin as the API, so no CORS configuration | Not a separate frontend |
| Model registry selected by `VERIFY_MODEL` (v1, v2, v3, cf, v4) | Same API and evaluation script for every model; switching is one environment variable | Only one model served per process |
| Model loaded once at startup | Loading takes seconds, inference takes milliseconds | Startup is slower |
| Preprocessing values and label mapping read from the training output files | Training and serving can never silently disagree; v2 flipped the label order (FAKE = 1) and the API needed no code change | None |
| Sync routes (`def`), not `async def` | Inference is CPU-bound; FastAPI runs sync routes in a thread pool so they don't block the event loop | None at this scale |
| Validate by decoding the image, not trusting the filename | Filenames and declared types are client claims | Slightly slower |
| Cheapest checks first: size, hash, cache, decode, inference | Bad or repeated requests never reach the expensive steps | None |
| Images processed in memory and never stored; only a SHA-256 fingerprint is kept | Privacy: uploaded photos can't leak; EXIF metadata (including GPS) is discarded | Can't re-run old images through new models |
| Cache keyed by image hash + analysis version (model + decision rules, e.g. `...+prov2`) | Repeat checks are instant; a new model or new rules automatically invalidate old answers | Exact bytes only; forwarded copies are handled by the look-alike match instead |
| 201 for new scans, 200 for cached ones | 201 means a resource was created; a cache hit creates nothing | None |
| Page size capped at 100 | One request can't pull the whole database | Needs pagination beyond 100 |
| Parameterized SQL everywhere | Prevents SQL injection | None |
| SQLite | Zero setup, one file | One writer at a time, single machine; multiple servers would need Postgres |
| v4's threshold comes from its config file, set on validation data for 5% false alarms | Chosen before seeing the external images; serving and evaluation use the same value | 5% of real photos will be flagged "likely AI"; a stricter threshold would miss more fakes |
| Serving preprocessing matches validation exactly (centre crop at native resolution) | A model only works with the preparation it was trained with | A centre crop sees only part of a large photo; averaging several crops would cost more time |

## Running it

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m uvicorn app.main:app --reload --reload-dir app
```
Then open http://localhost:8000 for the demo page, or http://localhost:8000/docs for the API. v4 is served by default: download `verify_v4.pth` from the [v4 release](../../releases) into `models/` first. To serve another model: `VERIFY_MODEL=v3 python -m uvicorn app.main:app` (`v1`, `v2`, `v3`, `cf` for the Community Forensics baseline, or `v4`).

Tests (23: provenance, fingerprint, API status codes, caching, look-alike matching, migration), using the small v1 model and a temporary database:
```bash
python -m pytest -q
```

The v1 weights are in `models/`. The v2 and v4 weights are attached to the [releases](../../releases); download them into `models/`.

To evaluate a model on folders of images (the folder name `ai` or `real` is the true label; the filename prefix before `_` is the source):
```bash
python app/model.py test_images/ai test_images/real
VERIFY_MODEL=v2 python app/model.py test_images/whatsapp_hop1/ai test_images/whatsapp_hop1/real
```
Raw outputs of every run are in the `results_*.txt` files.

## Known limitations

- The external set is small (41 images; 33 forwarded). It reliably reveals gross failures; differences of a few AUC points, and results on 4 or 5 images per generator, are not precise.
- v4's 5/5 on Meta AI is encouraging but is only 5 images from one unseen generator.
- v4's validation uses a simulated WhatsApp; the real-forward test is the 33-image set.
- The provenance check detects C2PA and IPTC labels; it does not verify C2PA signatures (that needs `c2patool` or the C2PA SDK).
- Look-alike matching only helps if the labelled original was scanned first, and it was measured on 49 images.
- Only one forwarding hop and one quality setting (standard, not HD) were tested.
- The 8 "early" AI images had no metadata and unknown processing history.
- v1 to v3 resize images to 224x224 without preserving aspect ratio (the API matches this for those models); v4 uses crops instead.

## Next steps

1. A larger external test set, especially more Bing images (v4's weakest source after forwarding) and more real photos to measure the false-alarm rate properly
2. Average several crops per image instead of one centre crop
3. Multi-hop and HD-quality forwarding tests
4. Verify C2PA signatures with the official SDK
