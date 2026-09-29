import pytest

from cifar10_clip.prompts import CIFAR10_CLASSES, CIFAR10_TEMPLATES, format_prompts


def test_cifar10_constants_cover_expected_classes_and_templates() -> None:
    assert CIFAR10_CLASSES == (
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
    assert len(CIFAR10_TEMPLATES) == 18


def test_format_prompts_preserves_template_order() -> None:
    templates = ("a photo of a {}.", "a sketch of the {}.")

    assert format_prompts("cat", templates) == [
        "a photo of a cat.",
        "a sketch of the cat.",
    ]


@pytest.mark.parametrize(
    ("class_name", "templates"),
    [("", ("a {}",)), ("cat", ())],
)
def test_format_prompts_rejects_empty_inputs(
    class_name: str, templates: tuple[str, ...]
) -> None:
    with pytest.raises(ValueError):
        format_prompts(class_name, templates)


def test_format_prompts_requires_placeholder() -> None:
    with pytest.raises(ValueError, match="placeholder"):
        format_prompts("cat", ("a photo of an animal",))

