"""Prompt templates used to construct a zero-shot CIFAR-10 classifier."""

from collections.abc import Sequence

CIFAR10_CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

CIFAR10_TEMPLATES = (
    "a photo of a {}.",
    "a blurry photo of a {}.",
    "a black and white photo of a {}.",
    "a low contrast photo of a {}.",
    "a high contrast photo of a {}.",
    "a bad photo of a {}.",
    "a good photo of a {}.",
    "a photo of a small {}.",
    "a photo of a big {}.",
    "a photo of the {}.",
    "a blurry photo of the {}.",
    "a black and white photo of the {}.",
    "a low contrast photo of the {}.",
    "a high contrast photo of the {}.",
    "a bad photo of the {}.",
    "a good photo of the {}.",
    "a photo of the small {}.",
    "a photo of the big {}.",
)


def format_prompts(class_name: str, templates: Sequence[str]) -> list[str]:
    """Format prompt templates for one class while preserving their order."""
    if not class_name.strip():
        raise ValueError("class_name must not be empty")
    if not templates:
        raise ValueError("templates must not be empty")
    if any("{}" not in template for template in templates):
        raise ValueError("every template must contain a '{}' placeholder")
    return [template.format(class_name) for template in templates]

