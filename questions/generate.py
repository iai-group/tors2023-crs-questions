"""Generate outputs for TQG and NQG and save them to CSV."""

import argparse
from enum import Enum

from questions.models.nqg import NQG
from questions.models.tqg import TQG
from questions.util import file_io


class ModelType(Enum):
    TQG = "tqg"
    NQG = "nqg"


def main(args: argparse.Namespace) -> None:
    """Main function.

    Args:
        args: Arguments.
    """
    model_type = ModelType(args.model)
    if model_type == ModelType.TQG:
        model = TQG(args.tqg_use_classifier)
    elif model_type == ModelType.NQG:
        model = NQG(
            model_path=args.nqg_model_path, use_reviews=args.nqg_use_review
        )

    df = file_io.get_dataframe_from_csv(args.dataset)
    df["question"] = model.generate_questions(df)

    df[["id", "question"]].to_csv(
        f"model_outputs/{args.output or str(model)}.csv", index=False
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        help="Model to generate the output for.",
        choices=["tqg", "nqg"],
        default="tqg",
    )
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        default="data/test.csv",
        help="Path to the input CSV file.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output file name without extension. If not specified, the model"
        "name is used.",
    )
    parser.add_argument(
        "--tqg_use_classifier",
        action="store_true",
        help="Use classifier for TQG.",
    )
    parser.add_argument(
        "--nqg_use_review",
        action="store_true",
        help="Use review text for NQG.",
    )
    parser.add_argument(
        "--nqg_model_path",
        type=str,
        help="Path for the trained NQG model.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
