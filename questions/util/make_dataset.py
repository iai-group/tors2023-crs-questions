"""Script for populating the datasets with full review texts and extracted
sentences.
"""

import argparse
import gzip
import json
import os
from typing import Dict, Iterator

import pandas as pd
from tqdm import tqdm

REVIEW_FILENAMES = {
    "Patio_Lawn_and_Garden.json.gz": 5236058,
    "Home_and_Kitchen.json.gz": 21928568,
    "Sports_and_Outdoors.json.gz": 12980837,
}

DATASET_PATH = "data/"


def get_absolute_path(path: str) -> str:
    """Returns absolute path."""
    if not os.path.isabs(path):
        path = os.path.join(os.getcwd(), path)
    return path


def parse_jsonl(path: str) -> Iterator[Dict]:
    """A generator from a JSONL file.

    Args:
        path: Path to JSONL file.

    Yields:
        dict: Dict from a line in JSONL file.
    """
    path = get_absolute_path(path)
    if path.endswith(".gz"):
        ifd = gzip.open(path, "rb")
    else:
        ifd = open(path, "r")

    for line in ifd:
        yield json.loads(line)


def load_dataset(filename: str) -> pd.DataFrame:
    """Returns a DataFrame from a CSV file.

    Args:
        filename: Name of a CSV file in the `data` folder.

    Returns:
        Pandas DataFrame.
    """
    return pd.read_csv(os.path.join(DATASET_PATH, filename), index_col=False)


def save_dataset(dataset: pd.DataFrame, filename: str) -> None:
    """Saves Pandas DataFrame to file.

    Args:
        dataset: DataFrame to save to file
        filename: Filename to use when saving DataFrame to file.
    """
    dataset.to_csv(
        os.path.join(DATASET_PATH, f"{filename}_full.csv"),
        index=False,
    )


def add_reviews(datasets: pd.DataFrame, folder_path: str) -> pd.DataFrame:
    """Iterates over the 2018 Amazon collection line by line and adds the review
    text to the dataset if user ID and item ID match.

    Args:
        dataset: DataFrame without full review texts.
        folder_path: Path to the 2018 Amazon collection folder containing files
            specified in REVIEW_FILENAMES.

    Returns:
        Pandas DataFrame including full review texts and extracted sentences.
    """
    for filename, total in REVIEW_FILENAMES.items():
        for i, item in enumerate(
            tqdm(parse_jsonl(os.path.join(folder_path, filename)), total=total)
        ):
            for dataset in datasets:
                matches = (dataset["asin"].values == item["asin"]) & (
                    dataset["reviewerID"].values == item["reviewerID"]
                )

                if not any(matches):
                    continue

                dataset.loc[matches, "reviewText"] = item["reviewText"]
                dataset.loc[matches, "sentenceText"] = [
                    item["reviewText"][start:end]
                    for start, end in zip(
                        dataset[matches].start_index, dataset[matches].end_index
                    )
                ]
        if i + 1 != total:
            print(
                f"Number of lines in {filename} ({i+1}) does not match the "
                f"expected number of lines ({total})."
            )
    return datasets


def main(args: argparse.Namespace) -> None:
    """Main function of the script. Loads cleaned versions of the datasets,
    populates missing columns and saves the files to drive.

    Args:
        args: Command line arguments. Should contain path to the Amazon dataset.
    """
    train = load_dataset("train.csv")
    test = load_dataset("test.csv")
    train, test = add_reviews([train, test], folder_path=args.path)
    save_dataset(train, "train")
    save_dataset(test, "test")


def parse_arguments() -> argparse.Namespace:
    """Parses command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        "-p",
        help="Path to the 2018 Amazon dataset folder.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
