"""Generate outputs for TQG and NQG and save them to CSV."""

import argparse
from enum import Enum

from models.nqg import NQG
from models.tqg import TQG


class ModelType(Enum):
    TQG = "tqg"
    NQG = "nqg"


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model",
        type=str,
        help="Model to generate the output for",
        options=["tqg", "nqg"],
    )
    parser.add_argument(
        "--tqg.use_classifier",
        action="store_true",
        help="Use classifier for TQG",
    )
    parser.add_argument(
        "--nqg.use_review",
        action="store_true",
        help="Use review text for NQG",
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    """Main function.

    Args:
        args: Arguments.
    """
    if args.model == ModelType.TQG:
        model = TQG(args.tqg_use_classifier)
    elif args.model == ModelType.NQG:
        model = NQG(args.nqg_use_review)
    else:
        raise ValueError(f"Model type {args.model} not supported")

    outputs = model.generate_outputs()
    outputs.to_csv(f"outputs/{args.model}.csv", index=False)


if __name__ == "__main__":
    main(parse_args())
