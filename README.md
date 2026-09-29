# Zero-shot CLIP Classification on CIFAR-10

[![CI](https://github.com/volkthienpreecha/cifar10-zero-shot-clip/actions/workflows/ci.yml/badge.svg)](https://github.com/volkthienpreecha/cifar10-zero-shot-clip/actions/workflows/ci.yml)

A reproducible, inference-only image-classification project using OpenAI CLIP and prompt ensembling. It builds a ten-class classifier from natural-language descriptions—without fitting on CIFAR-10 labels—then evaluates scaled cosine similarity between image and text embeddings.

## Highlights

- Eighteen language templates per class reduce dependence on a single prompt.
- Text and image embeddings are normalized explicitly before similarity scoring.
- Device selection supports CUDA, Apple MPS, and CPU.
- Evaluation preserves sample order and reports exact top-1 and top-5 accuracy.
- A restricted loader converts the legacy prediction artifact to portable CSV without allowing pickle global lookups.
- Fast tests use synthetic tensors and fake encoders; CI downloads neither CLIP weights nor CIFAR-10.

## How it works

```text
10 class names × 18 prompt templates
  → CLIP text encoder
  → normalize each prompt embedding
  → average and renormalize per class
  → classifier matrix [embedding dimension × 10]

CIFAR-10 image
  → CLIP image encoder
  → normalized image embedding
  → 100 × cosine similarity with class matrix
  → ranked class predictions
```

This is zero-shot classification: CIFAR-10 labels are used only for evaluation, not for model fitting.

## Recorded result

| Model | Dataset | Top-1 | Top-5 |
|---|---|---:|---:|
| CLIP ViT-B/32 | CIFAR-10 test (10,000 images) | **89.86%** | **99.63%** |

These values are historical metrics from the supplied executed experiment, not a fresh rerun performed while packaging this repository. The provenance-labeled snapshot is in [`artifacts/metrics.json`](artifacts/metrics.json).

## Setup

Python 3.10 or newer is required. Install the CLIP extra for end-to-end inference:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[clip,dev]"
```

The CLIP dependency is pinned to a specific OpenAI CLIP revision. A test-only installation can omit the `clip` extra.

## Evaluate

The first run downloads the selected CLIP weights and CIFAR-10 test split.

```bash
cifar10-clip-evaluate \
  --model "ViT-B/32" \
  --device auto \
  --data-dir data/cifar10 \
  --batch-size 128 \
  --output-dir runs/clip-evaluation
```

The command writes `metrics.json` and `zero_shot_predictions.csv`. Run `cifar10-clip-evaluate --help` for all options.

## Use the classifier core

```python
import clip
import torch

from cifar10_clip import CIFAR10_CLASSES, CIFAR10_TEMPLATES
from cifar10_clip.classifier import build_zero_shot_weights, predict_logits

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, preprocess = clip.load("ViT-B/32", device=device)
weights = build_zero_shot_weights(
    model, clip.tokenize, CIFAR10_CLASSES, CIFAR10_TEMPLATES, device
)

# `images` should already be transformed by CLIP's `preprocess` function.
features = model.encode_image(images.to(device))
logits = predict_logits(features, weights)
predicted_class_ids = logits.argmax(dim=1)
```

## Prediction artifact

[`artifacts/zero_shot_predictions.csv`](artifacts/zero_shot_predictions.csv) contains the complete ordered prediction set:

| Column | Meaning |
|---|---|
| `sample_id` | Zero-based position in the unshuffled CIFAR-10 test split |
| `predicted_class_id` | Integer class ID from 0 through 9 |
| `predicted_class_name` | Human-readable CIFAR-10 class name |

To convert a compatible legacy artifact yourself:

```bash
python scripts/convert_predictions.py \
  path/to/zero_shot_predictions.pickle \
  artifacts/zero_shot_predictions.csv \
  --expected-count 10000
```

The source pickle is intentionally excluded. The converter permits only lists and integer class IDs and rejects object reconstruction.

## Walkthrough

[`notebooks/cifar10_zero_shot_clip_walkthrough.ipynb`](notebooks/cifar10_zero_shot_clip_walkthrough.ipynb) explains prompt ensembling, normalization, cosine-similarity logits, historical metrics, and artifact inspection. Remote execution is disabled by default with `RUN_EVALUATION = False`.

## Repository structure

```text
src/cifar10_clip/       prompt, classifier, evaluation, plotting, and CLI code
tests/                  offline unit tests using fake encoders
notebooks/              curated technical walkthrough
artifacts/              metrics snapshot and portable predictions
scripts/                safe legacy-artifact conversion
.github/workflows/      lint, test, and notebook validation
```

## Test

```bash
python -m pytest -q
python -m ruff check src scripts tests
```

Tests do not download model weights or datasets.

## Limitations

- Accuracy depends on prompt wording and CLIP's pretraining distribution.
- The historical run records aggregate accuracy, not calibration or per-class recall.
- CIFAR-10 is low-resolution and does not establish robustness on larger or shifted datasets.
- No model weights or dataset files are included.
- Full evaluation requires external downloads and is intentionally excluded from CI.

## References

- A. Radford et al., [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020), 2021.
- [OpenAI CLIP](https://github.com/openai/CLIP).
- A. Krizhevsky, [Learning Multiple Layers of Features from Tiny Images](https://www.cs.toronto.edu/~kriz/cifar.html), 2009.
- [PyTorch](https://pytorch.org/) and [torchvision](https://pytorch.org/vision/stable/).

## Provenance

The prompt-ensemble experiment and recorded metrics began as academic computer-vision work and were subsequently cleaned, tested, and packaged for reproducible public presentation. Classroom prompts, grading utilities, and submission code are intentionally excluded.

