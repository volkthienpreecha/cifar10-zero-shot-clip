"""Device-aware batch evaluation for CLIP-style image encoders."""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor

from cifar10_clip.classifier import predict_logits, topk_correct


class ImageEncoderLike(Protocol):
    """The subset of a CLIP image encoder required for evaluation."""

    def eval(self) -> object:
        """Switch the model to inference mode."""

    def encode_image(self, images: Tensor) -> Tensor:
        """Encode a batch of preprocessed images."""


@dataclass(frozen=True)
class EvaluationResult:
    """Aggregate accuracy counts and ordered class predictions."""

    total_samples: int
    correct: dict[int, int]
    predictions: tuple[int, ...]

    def accuracy(self, k: int) -> float:
        """Return top-k accuracy as a percentage."""
        if k not in self.correct:
            raise ValueError(f"top-{k} was not computed")
        return 100.0 * self.correct[k] / self.total_samples


def resolve_device(requested: str = "auto") -> torch.device:
    """Resolve auto to CUDA, then Apple MPS, then CPU."""
    requested = requested.lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is not available")
    if requested not in {"cpu", "cuda", "mps"}:
        raise ValueError("device must be one of: auto, cpu, cuda, mps")
    return torch.device(requested)


def evaluate_batches(
    model: ImageEncoderLike,
    loader: Iterable[tuple[Tensor, Tensor]],
    classifier_weights: Tensor,
    device: torch.device,
    topk: tuple[int, ...] = (1, 5),
) -> EvaluationResult:
    """Evaluate ordered batches and return aggregate counts and predictions."""
    model.eval()
    weights = classifier_weights.to(device)
    totals = {k: 0 for k in topk}
    predictions: list[int] = []
    total_samples = 0

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            features = model.encode_image(images)
            if not torch.isfinite(features).all():
                raise ValueError("image features must be finite")
            logits = predict_logits(features, weights)
            counts = topk_correct(logits, targets, topk)
            for k, count in zip(topk, counts, strict=True):
                totals[k] += count
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            total_samples += targets.shape[0]

    if total_samples == 0:
        raise ValueError("loader produced no samples")
    return EvaluationResult(total_samples, totals, tuple(predictions))

