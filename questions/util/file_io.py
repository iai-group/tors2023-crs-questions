import pandas as pd


def get_dataframe_from_csv(path: str) -> pd.DataFrame:
    """Load data from CSV.

    Args:
        path: Path to CSV.
    """
    df = pd.read_csv(path)
    df["labels"] = df.question1.notna().astype(int)
    df["text"] = df.sentence
    return df
