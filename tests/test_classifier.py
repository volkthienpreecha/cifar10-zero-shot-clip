from collections.abc import Sequence

import pytest
import torch
from torch import Tensor

from cifar10_clip.classifier import build_zero_shot_weights, predict_logits, topk_correct


class FakeTextEncoder:
    def encode_text(self, tokens: Tensor) -> Tensor:
        return tokens.float()


class ZeroTextEncoder:
    def encode_text(self, tokens: Tensor) -> Tensor:
        return torch.zeros((tokens.shape[0], 3), dtype=torch.float32)


def fake_tokenize(prompts: Sequence[str]) -> Tensor:
    return torch.tensor(
        [[float(len(prompt)), float(prompt.count("a") + 1), 1.0] for prompt in prompts]
    )


def test_build_zero_shot_weights_returns_normalized_class_columns() -> None:
    weights = build_zero_shot_weights(
        FakeTextEncoder(),
        fake_tokenize,
        ("cat", "truck"),
        ("a photo of a {}.", "a blurry photo of the {}."),
        torch.device("cpu"),
    )

    assert weights.shape == (3, 2)
    torch.testing.assert_close(weights.norm(dim=0), torch.ones(2))


@pytest.mark.parametrize(
    ("class_names", "templates"),
    [((), ("a {}",)), (("cat",), ())],
)
def test_build_zero_shot_weights_rejects_empty_inputs(
    class_names: tuple[str, ...], templates: tuple[str, ...]
) -> None:
    with pytest.raises(ValueError):
        build_zero_shot_weights(
            FakeTextEncoder(), fake_tokenize, class_names, templates, torch.device("cpu")
        )


def test_build_zero_shot_weights_rejects_zero_norm_embeddings() -> None:
    with pytest.raises(ValueError, match="zero-norm"):
        build_zero_shot_weights(
            ZeroTextEncoder(),
            fake_tokenize,
            ("cat",),
            ("a photo of a {}.",),
            torch.device("cpu"),
        )


def test_topk_correct_returns_exact_counts() -> None:
    logits = torch.tensor(
        [
            [9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0],
            [0.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
            [9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0],
        ]
    )
    targets = torch.tensor([0, 2, 4])

    assert topk_correct(logits, targets, topk=(1, 5)) == (1, 3)


@pytest.mark.parametrize("topk", [(0,), (11,)])
def test_topk_correct_rejects_invalid_k(topk: tuple[int, ...]) -> None:
    with pytest.raises(ValueError):
        topk_correct(torch.ones((2, 10)), torch.zeros(2, dtype=torch.long), topk)


def test_predict_logits_has_batch_by_class_shape() -> None:
    image_features = torch.tensor([[3.0, 4.0, 0.0], [0.0, 0.0, 2.0]])
    classifier_weights = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])

    logits = predict_logits(image_features, classifier_weights)

    assert logits.shape == (2, 2)
    torch.testing.assert_close(logits[0], torch.tensor([60.0, 80.0]))


def test_predict_logits_rejects_zero_norm_features() -> None:
    with pytest.raises(ValueError, match="zero-norm"):
        predict_logits(torch.zeros((1, 3)), torch.ones((3, 2)))
