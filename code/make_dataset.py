"""Script for populating the datasets with full review texts and extracted
sentences.
"""

import argparse
import gzip
import json
import os
from typing import Dict, Iterator

import pandas as pd

REVIEW_FILENAMES = [
    "Patio_Lawn_and_Garden.json.gz",
    "Home_and_Kitchen.json.gz",
    "Sports_and_Outdoors.json.gz",
]

DATASET_PATH = "dataset/"


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


def add_reviews(dataset: pd.DataFrame, folder_path: str) -> pd.DataFrame:
    """Iterates over the 2018 Amazon collection line by line and adds the review
    text to the dataset if user ID and item ID match.

    Args:
        df: DataFrame without full review texts.
        folder_path: Path to the 2018 Amazon collection folder containing files
            specified in REVIEW_FILENAMES.

    Returns:
        Pandas DataFrame including full review texts and extracted sentences.
    """
    counter = 0
    for filename in REVIEW_FILENAMES:
        print(filename)
        for i, item in enumerate(parse_jsonl(os.path.join(folder_path, filename))):
            matches = (dataset["asin"].values == item["asin"]) & (
                dataset["reviewerID"].values == item["reviewerID"]
            )
            if any(matches):
                match = df.loc[matches].iloc[0]
                dataset.loc[matches, "reviewText"] = item["reviewText"]
                dataset.loc[matches, "sentenceText"] = item["reviewText"][
                    match["start"] : match["end"]
                ]
                counter += 1
            if (i + 1) % 100000 == 0:
                print(f"Parsed {i} items.")
    print(f"Done. \nSuccesfully updated {counter}/{len(dataset)} items.")
    return df


def main(args: argparse.Namespace) -> None:
    """Main function of the script. Loads cleaned versions of the datasets,
    populates missing columns and saves the files to drive.

    Args:
        args: Command line arguments. Should contain path to the Amazon dataset.
    """
    for filename in ["train", "test"]:
        df = load_dataset(f"{filename}.csv")
        df = add_reviews(df, folder_path=args.path)
        df.to_csv(f"{filename}_full.csv", index=False)


def parse_arguments() -> argparse.Namespace:
    """Parses command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", "-p", help="Path to the 2018 Amazon dataset folder.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
