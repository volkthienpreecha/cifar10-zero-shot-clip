"""Small plotting helpers for inspecting CLIP predictions."""

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

CLIP_MEAN = torch.tensor((0.48145466, 0.4578275, 0.40821073))
CLIP_STD = torch.tensor((0.26862954, 0.26130258, 0.27577711))


def denormalize_clip_images(images: Tensor) -> Tensor:
    """Convert CLIP-normalized images back to displayable RGB values."""
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("images must have shape (batch, 3, height, width)")
    mean = CLIP_MEAN.to(images.device).view(1, 3, 1, 1)
    std = CLIP_STD.to(images.device).view(1, 3, 1, 1)
    return (images * std + mean).clamp(0.0, 1.0)


def save_prediction_grid(
    images: Tensor,
    targets: Sequence[int],
    predictions: Sequence[int],
    class_names: Sequence[str],
    output_path: Path,
    limit: int = 8,
) -> None:
    """Save a bounded image grid with true and predicted labels."""
    count = min(limit, images.shape[0], len(targets), len(predictions))
    if count <= 0:
        raise ValueError("at least one sample is required")
    display_images = denormalize_clip_images(images[:count]).cpu()
    columns = min(4, count)
    rows = (count + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(3.2 * columns, 3.2 * rows))
    axes_array = np.atleast_1d(axes).reshape(-1)
    for index, axis in enumerate(axes_array):
        axis.axis("off")
        if index >= count:
            continue
        axis.imshow(display_images[index].permute(1, 2, 0).numpy())
        axis.set_title(
            f"True: {class_names[int(targets[index])]}\n"
            f"Pred: {class_names[int(predictions[index])]}",
            fontsize=9,
        )
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)

