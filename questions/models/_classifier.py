# %%
import logging
from pprint import pprint

import pandas as pd
from simpletransformers.classification import (
    ClassificationArgs,
    ClassificationModel,
)

MODEL = "roberta"
MODEL_SIZE = "base"

logging.basicConfig(level=logging.INFO)
classifier_logger = logging.getLogger("classifier")
classifier_logger.setLevel(logging.WARNING)


def load_data(path: str) -> pd.DataFrame:
    """Load data from CSV.

    Args:
        path: Path to CSV.
    """
    df = pd.read_csv(path)
    df["labels"] = df.question1.notna().astype(int)
    df["text"] = df.sentence
    return df[["text", "labels"]]


class Classifier:
    def __init__(
        self,
        model: str = MODEL,
        model_size: str = MODEL_SIZE,
        cuda_device: int = -1,
    ):
        """Classifier for TQG.

        Args:
            model: Model to use.
            model_size: Model size to use.
            cuda_device: CUDA device to use. -1 means no CUDA. Defaults to -1.
        """
        self._model_args = ClassificationArgs(
            num_train_epochs=3, overwrite_output_dir=True
        )

        self._model = ClassificationModel(
            model,
            f"{model}-{model_size}",
            cuda_device=cuda_device,
            args=self._model_args,
        )

    def train(self, train_df: pd.DataFrame):
        """Train the classifier.

        Args:
            train_df: Training data. Should have columns "text" and "labels".
        """
        self._model.train_model(train_df)

    def eval(self, eval_df: pd.DataFrame):
        """Evaluate the classifier.

        Args:
            eval_df: Evaluation data. Should have columns "text" and "labels".

        Returns:
            Evaluation results.
        """
        result, model_outputs, wrong_predictions = self._model.eval_model(
            eval_df
        )
        return (model_outputs[:, 1] > model_outputs[:, 0]) * 1

    def save(self, path: str):
        """Save the classifier.

        Args:
            path: Path to save the classifier to.
        """
        self._model.save_model(path)


if __name__ == "__main__":
    train_data = load_data("data/train.csv")
    eval_data = load_data("data/test.csv")

    classifier = Classifier()
    classifier.train(train_data)

    pprint(classifier.eval(eval_data))

# %%