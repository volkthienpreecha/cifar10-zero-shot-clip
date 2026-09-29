"""Prompt-ensembled classifier construction and top-k evaluation."""

from collections.abc import Callable, Sequence
from typing import Protocol

import torch
from torch import Tensor

from cifar10_clip.prompts import format_prompts


class ClipLike(Protocol):
    """The subset of the CLIP text encoder used by this package."""

    def encode_text(self, tokens: Tensor) -> Tensor:
        """Encode tokenized text prompts."""


def _normalized_rows(values: Tensor, *, label: str) -> Tensor:
    if values.ndim != 2:
        raise ValueError(f"{label} must be a two-dimensional tensor")
    values = values.float()
    norms = values.norm(dim=-1, keepdim=True)
    if torch.any(norms <= torch.finfo(values.dtype).eps):
        raise ValueError(f"{label} contains a zero-norm vector")
    return values / norms


def build_zero_shot_weights(
    model: ClipLike,
    tokenize: Callable[[Sequence[str]], Tensor],
    class_names: Sequence[str],
    templates: Sequence[str],
    device: torch.device,
) -> Tensor:
    """Average normalized prompt embeddings into one unit vector per class."""
    if not class_names:
        raise ValueError("class_names must not be empty")
    if not templates:
        raise ValueError("templates must not be empty")

    class_weights: list[Tensor] = []
    with torch.no_grad():
        for class_name in class_names:
            prompts = format_prompts(class_name, templates)
            tokens = tokenize(prompts).to(device)
            embeddings = _normalized_rows(
                model.encode_text(tokens), label=f"text embeddings for {class_name!r}"
            )
            average = embeddings.mean(dim=0)
            average_norm = average.norm()
            if average_norm <= torch.finfo(average.dtype).eps:
                raise ValueError(f"mean text embedding for {class_name!r} is zero-norm")
            class_weights.append(average / average_norm)

    return torch.stack(class_weights, dim=1).to(device)


def predict_logits(
    image_features: Tensor,
    classifier_weights: Tensor,
    logit_scale: float = 100.0,
) -> Tensor:
    """Return scaled cosine-similarity logits for a batch of image features."""
    if image_features.ndim != 2 or classifier_weights.ndim != 2:
        raise ValueError("image_features and classifier_weights must be two-dimensional")
    if image_features.shape[1] != classifier_weights.shape[0]:
        raise ValueError("feature dimensions do not match")

    normalized_images = _normalized_rows(image_features, label="image features")
    normalized_weights = _normalized_rows(classifier_weights.t(), label="classifier weights").t()
    return float(logit_scale) * normalized_images @ normalized_weights


def topk_correct(
    logits: Tensor, targets: Tensor, topk: tuple[int, ...] = (1,)
) -> tuple[int, ...]:
    """Count correct predictions at each requested value of k."""
    if logits.ndim != 2:
        raise ValueError("logits must be two-dimensional")
    if targets.ndim != 1 or targets.shape[0] != logits.shape[0]:
        raise ValueError("targets must contain one label per logits row")
    if not topk or any(k <= 0 or k > logits.shape[1] for k in topk):
        raise ValueError("each k must be between 1 and the number of classes")

    predictions = logits.topk(max(topk), dim=1, largest=True, sorted=True).indices.t()
    correct = predictions.eq(targets.view(1, -1).expand_as(predictions))
    return tuple(int(correct[:k].any(dim=0).sum().item()) for k in topk)
