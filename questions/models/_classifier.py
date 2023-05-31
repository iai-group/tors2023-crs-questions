import logging
from pprint import pprint
from typing import Dict, List

import pandas as pd
from simpletransformers.classification import (
    ClassificationArgs,
    ClassificationModel,
)

MODEL_PATH = "roberta-base"
MODEL_TYPE = "roberta"

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
        model_path: str = None,
        model_type: str = MODEL_TYPE,
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
            model_type,
            model_path or MODEL_PATH,
            cuda_device=cuda_device,
            args=self._model_args,
        )

        if model_path is None:
            classifier_logger.info(
                "No model path specified. Finetuning default model."
            )
            self.train(load_data("data/train.csv"))
            pprint(self.eval(load_data("data/test.csv")))

    def train(self, train_df: pd.DataFrame):
        """Train the classifier.

        Args:
            train_df: Training data. Should have columns "text" and "labels".
        """
        self._model.train_model(train_df)

    def eval(self, eval_df: pd.DataFrame) -> Dict[str, float]:
        """Evaluate the classifier.

        Args:
            eval_df: Evaluation data. Should have columns "text".

        Returns:
            Evaluation results.
        """
        result, model_outputs, wrong_predictions = self._model.eval_model(
            eval_df
        )
        return result

    def predict(self, to_predict: List[str]) -> List[int]:
        """Predict labels for the given data.

        Args:
            to_predict: Data to predict labels for.

        Returns:
            Predicted labels.
        """
        return self._model.predict(to_predict)[0]

    def save(self, path: str):
        """Save the classifier.

        Args:
            path: Path to save the classifier to.
        """
        self._model.save_model(path)
