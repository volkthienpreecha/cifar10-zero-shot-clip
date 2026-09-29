"""Convert a trusted legacy prediction pickle to portable CSV."""

import argparse
from pathlib import Path

from cifar10_clip.artifacts import load_plain_prediction_pickle, write_predictions_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="legacy prediction pickle")
    parser.add_argument("output", type=Path, help="destination CSV path")
    parser.add_argument("--expected-count", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions = load_plain_prediction_pickle(args.input)
    if args.expected_count is not None and len(predictions) != args.expected_count:
        raise ValueError(
            f"expected {args.expected_count} predictions, found {len(predictions)}"
        )
    write_predictions_csv(predictions, args.output)
    print(f"Wrote {len(predictions):,} predictions to {args.output}")


if __name__ == "__main__":
    main()
