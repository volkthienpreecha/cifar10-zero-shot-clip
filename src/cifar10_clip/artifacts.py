"""Safe conversion helpers for legacy prediction artifacts."""

import csv
import pickle
from collections.abc import Sequence
from pathlib import Path
from typing import BinaryIO

from cifar10_clip.prompts import CIFAR10_CLASSES


class _DataOnlyUnpickler(pickle.Unpickler):
    """Unpickler that refuses all global class and function lookups."""

    def find_class(self, module: str, name: str) -> object:
        raise pickle.UnpicklingError(
            f"global lookup is disabled while loading prediction data: {module}.{name}"
        )


def _load_data_only(handle: BinaryIO) -> object:
    return _DataOnlyUnpickler(handle).load()


def validate_predictions(values: object, num_classes: int = 10) -> list[int]:
    """Validate and flatten plain integer or singleton-list predictions."""
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")
    if not isinstance(values, list):
        raise ValueError("predictions must be stored as a list")

    predictions: list[int] = []
    for index, value in enumerate(values):
        if isinstance(value, list):
            if len(value) != 1:
                raise ValueError(f"prediction {index} must contain exactly one class ID")
            value = value[0]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"prediction {index} is not an integer class ID")
        if not 0 <= value < num_classes:
            raise ValueError(f"prediction {index} is outside the valid class range")
        predictions.append(value)
    return predictions


def load_plain_prediction_pickle(path: Path) -> list[int]:
    """Load a legacy pickle without permitting executable global references."""
    with path.open("rb") as handle:
        values = _load_data_only(handle)
    return validate_predictions(values)


def write_predictions_csv(
    predictions: Sequence[int],
    output_path: Path,
    class_names: Sequence[str] = CIFAR10_CLASSES,
) -> None:
    """Write validated predictions as a deterministic, portable CSV file."""
    if not class_names:
        raise ValueError("class_names must not be empty")
    validated = validate_predictions(list(predictions), num_classes=len(class_names))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("sample_id", "predicted_class_id", "predicted_class_name"))
        for sample_id, class_id in enumerate(validated):
            writer.writerow((sample_id, class_id, class_names[class_id]))

