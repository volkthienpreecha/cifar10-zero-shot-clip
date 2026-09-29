import pytest
import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from cifar10_clip.evaluate import evaluate_batches, resolve_device


class FakeImageEncoder:
    def __init__(self) -> None:
        self.eval_called = False

    def eval(self) -> "FakeImageEncoder":
        self.eval_called = True
        return self

    def encode_image(self, images: Tensor) -> Tensor:
        return images


def test_evaluate_batches_handles_uneven_batches_and_preserves_order() -> None:
    features = torch.tensor(
        [
            [10.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [0.0, 0.0, 10.0],
            [8.0, 9.0, 0.0],
            [8.0, 0.0, 9.0],
        ]
    )
    targets = torch.tensor([0, 1, 2, 1, 0])
    loader = DataLoader(TensorDataset(features, targets), batch_size=2, shuffle=False)
    model = FakeImageEncoder()

    result = evaluate_batches(
        model,
        loader,
        torch.eye(3),
        torch.device("cpu"),
        topk=(1, 2),
    )

    assert model.eval_called
    assert result.total_samples == 5
    assert result.correct == {1: 4, 2: 5}
    assert result.accuracy(1) == pytest.approx(80.0)
    assert result.accuracy(2) == pytest.approx(100.0)
    assert result.predictions == (0, 1, 2, 1, 2)


def test_evaluate_batches_rejects_empty_loader() -> None:
    empty_loader = DataLoader(
        TensorDataset(torch.empty((0, 3)), torch.empty(0, dtype=torch.long)), batch_size=2
    )

    with pytest.raises(ValueError, match="no samples"):
        evaluate_batches(
            FakeImageEncoder(), empty_loader, torch.eye(3), torch.device("cpu"), topk=(1, 2)
        )


def test_evaluate_batches_rejects_nonfinite_features() -> None:
    loader = DataLoader(
        TensorDataset(torch.tensor([[float("nan"), 0.0]]), torch.tensor([0])), batch_size=1
    )

    with pytest.raises(ValueError, match="finite"):
        evaluate_batches(
            FakeImageEncoder(), loader, torch.eye(2), torch.device("cpu"), topk=(1,)
        )


def test_resolve_device_accepts_cpu() -> None:
    assert resolve_device("cpu") == torch.device("cpu")


@pytest.mark.parametrize(
    ("cuda_available", "mps_available", "expected"),
    [(True, True, "cuda"), (False, True, "mps"), (False, False, "cpu")],
)
def test_resolve_device_auto_prefers_cuda_then_mps_then_cpu(
    monkeypatch: pytest.MonkeyPatch,
    cuda_available: bool,
    mps_available: bool,
    expected: str,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps_available)

    assert resolve_device("auto") == torch.device(expected)
