"""Command-line evaluation for zero-shot CLIP on CIFAR-10."""

import argparse
import json
from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

from cifar10_clip.artifacts import write_predictions_csv
from cifar10_clip.classifier import build_zero_shot_weights
from cifar10_clip.evaluate import evaluate_batches, resolve_device
from cifar10_clip.prompts import CIFAR10_CLASSES, CIFAR10_TEMPLATES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="ViT-B/32", help="OpenAI CLIP model name")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/clip-evaluation"))
    return parser


def _load_clip() -> Any:
    try:
        import clip
    except ImportError as error:
        raise SystemExit(
            "OpenAI CLIP is not installed. Install the project dependencies before evaluation."
        ) from error
    return clip


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.batch_size <= 0 or args.workers < 0:
        raise SystemExit("batch size must be positive and workers cannot be negative")

    device = resolve_device(args.device)
    clip = _load_clip()
    model, preprocess = clip.load(args.model, device=device)
    dataset = CIFAR10(root=args.data_dir, train=False, download=True, transform=preprocess)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    weights = build_zero_shot_weights(
        model, clip.tokenize, CIFAR10_CLASSES, CIFAR10_TEMPLATES, device
    )
    result = evaluate_batches(model, loader, weights, device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_predictions_csv(
        result.predictions, args.output_dir / "zero_shot_predictions.csv"
    )
    metrics = {
        "dataset": "CIFAR-10 test",
        "model": args.model,
        "device": str(device),
        "samples": result.total_samples,
        "top1_accuracy_percent": result.accuracy(1),
        "top5_accuracy_percent": result.accuracy(5),
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
