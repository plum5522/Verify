# Verify

Detecting AI-generated images in the content people actually forward on WhatsApp.

## Why this exists

AI-generated images spread widely through reels and WhatsApp forwards, and many people, especially those less familiar with technology, accept them as real. Most AI-image detectors are trained and tested on clean, full-quality images. But a WhatsApp forward has been compressed, resized, and re-compressed, sometimes many times over, which destroys the fine details detectors rely on.

Verify is a self-trained detector and a backend API built around that gap: what happens to detection accuracy on the images real people actually see?

## Status

| Stage | Status |
|---|---|
| Baseline model trained on CIFAKE | Done |
| FastAPI backend (`POST /scans`) | In progress |
| SQLite scan history | Planned |
| WhatsApp forwarding test | Planned |
| Compression-robust retrain (v2) | Planned |

## Baseline results (v1)

EfficientNet-B0 (ImageNet-pretrained), fine-tuned on [CIFAKE](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images) (100K train / 20K test, balanced real vs AI).

| Metric | Value |
|---|---|
| Test accuracy | **97.9%** |
| False positive rate (real flagged as AI) | **1.5%** |
| Miss rate (AI passed as real) | **2.7%** |
| Best validation accuracy | 98.25% (epoch 2) |

Full numbers and training history: [`models/results.json`](models/results.json). Training notebook: [`notebooks/train_cifake.ipynb`](notebooks/train_cifake.ipynb).

**Training note:** validation loss rose in epoch 3 while training loss kept falling, an early sign of overfitting, so the epoch 2 checkpoint is the one kept.

## Design decisions

| Decision | Why | Tradeoff |
|---|---|---|
| Fine-tune a pretrained model | Strong baseline in under an hour of free GPU time | Features were learned for recognizing objects, not AI artifacts |
| EfficientNet-B0 | Small (~4M weights), fast enough to serve on CPU | CLIP-based detectors generalize better across generators |
| CIFAKE dataset | Free, labeled, balanced, quick to train | 32x32 images, one 2022 generator (Stable Diffusion 1.4), no faces or news-style photos |
| Learning rate 1e-4 | Adapt pretrained features without overwriting them | Slower learning |
| Keep best checkpoint, not last | Guards against overfitting in later epochs | None |
| No augmentation in v1 | Clean baseline to compare future improvements against | Not yet robust to compression |

## Known limitations

These are the reasons the next steps exist, not footnotes:

- **97.9% is a benchmark number, not a real-world number.** CIFAKE images are tiny and clean; real forwards are not.
- **Distribution shift:** training images were 32x32 upscaled to 224x224 (blurry). Real uploads are large photos downscaled to 224x224 (sharp). The model has never seen a sharp image.
- **One generator:** the model has only seen Stable Diffusion 1.4 fakes, not Midjourney, Flux, or other modern generators.
- **The model leans toward "real":** it misses AI images (2.7%) more often than it wrongly flags real ones (1.5%).

## Next steps

1. Serve the model through a RESTful FastAPI backend with three-way verdicts (likely AI / inconclusive / likely real)
2. Measure accuracy on real images forwarded through WhatsApp 1 and 3 times
3. Retrain with WhatsApp-style compression augmentation and compare v1 vs v2 on the same forwarded images
4. Tune the AI threshold toward catching more fakes, and measure the false-positive cost

## Setup

The trained weights are included at `models/verify_effnetb0_best.pth` (~16 MB). Future model versions will be published as GitHub Releases instead, to keep the repository small.
