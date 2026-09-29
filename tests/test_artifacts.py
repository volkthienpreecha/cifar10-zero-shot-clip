import csv
import pickle
from pathlib import Path

import pytest

from cifar10_clip.artifacts import (
    load_plain_prediction_pickle,
    validate_predictions,
    write_predictions_csv,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [([0, 4, 9], [0, 4, 9]), ([[3], [1], [8]], [3, 1, 8])],
)
def test_validate_predictions_flattens_supported_values_in_order(
    raw: object, expected: list[int]
) -> None:
    assert validate_predictions(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        [1.0],
        ["1"],
        [[]],
        [[1, 2]],
        [True],
        [-1],
        [10],
        (1, 2),
    ],
)
def test_validate_predictions_rejects_unsafe_or_invalid_values(raw: object) -> None:
    with pytest.raises(ValueError):
        validate_predictions(raw)


def test_load_plain_prediction_pickle_accepts_only_data(tmp_path: Path) -> None:
    pickle_path = tmp_path / "predictions.pickle"
    pickle_path.write_bytes(pickle.dumps([[2], [5], [0]]))

    assert load_plain_prediction_pickle(pickle_path) == [2, 5, 0]


def test_load_plain_prediction_pickle_rejects_globals(tmp_path: Path) -> None:
    pickle_path = tmp_path / "malicious.pickle"
    pickle_path.write_bytes(pickle.dumps(eval))

    with pytest.raises(pickle.UnpicklingError, match="global"):
        load_plain_prediction_pickle(pickle_path)


def test_write_predictions_csv_maps_ids_to_class_names(tmp_path: Path) -> None:
    output_path = tmp_path / "predictions.csv"

    write_predictions_csv([2, 0], output_path, ("zero", "one", "two"))

    with output_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows == [
        {"sample_id": "0", "predicted_class_id": "2", "predicted_class_name": "two"},
        {"sample_id": "1", "predicted_class_id": "0", "predicted_class_name": "zero"},
    ]
