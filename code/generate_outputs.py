import argparse
from models.nqg import NQG
from models.tqg import TQG


def parse_args() -> argparse.Namespace:
    """Parse arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str, help="Path to results file.")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    pass