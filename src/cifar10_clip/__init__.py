"""Zero-shot CIFAR-10 classification utilities built around CLIP."""

from cifar10_clip.classifier import build_zero_shot_weights, predict_logits, topk_correct
from cifar10_clip.prompts import CIFAR10_CLASSES, CIFAR10_TEMPLATES, format_prompts

__all__ = [
    "CIFAR10_CLASSES",
    "CIFAR10_TEMPLATES",
    "build_zero_shot_weights",
    "format_prompts",
    "predict_logits",
    "topk_correct",
]

