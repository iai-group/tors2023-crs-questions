"""Neural Question Generation (NQG) model."""

import logging
from pprint import pprint
from typing import Dict, List

import pandas as pd
from simpletransformers.t5 import T5Args, T5Model

from questions.util import file_io

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("NQG")
transformers_logger.setLevel(logging.WARNING)

MODEL_PATH = "t5-large"
TRAINING_DATA_PATH = "data/train.csv"
EVALUATION_DATA_PATH = "data/test.csv"


class NQG:
    def __init__(
        self,
        model_path: str = None,
        use_reviews: bool = False,
        cuda_device: int = -1,
    ) -> None:
        """Neural Question Generation (NQG) model.

        Args:
            model_name: Name of the model to use.
            use_reviews: Whether to use reviews or text for training.
        """
        self.use_reviews = use_reviews

        model_args = T5Args(num_train_epochs=3)
        self._model = T5Model(
            "t5",
            model_path or MODEL_PATH,
            cuda_device=cuda_device,
            args=model_args,
        )

        if model_path is None:
            self.train(self.get_dataframe(TRAINING_DATA_PATH))
            results = self.eval(self.get_dataframe(EVALUATION_DATA_PATH))

            pprint(results)

    def get_dataframe(self, path: str) -> pd.DataFrame:
        """Get training data for the model.

        Args:
            path: Path to the training data.

        Returns:
            Training data.
        """
        df = file_io.get_dataframe_from_csv(path)
        df["input_text"] = df["review" if self.use_reviews else "text"].astype(
            str
        )

        df = df.melt(
            id_vars=["id", "input_text"],
            value_vars=[
                "question1",
                "question2",
                "question3",
                "paraphrase1",
                "paraphrase2",
            ],
            var_name="source",
            value_name="target_text",
        )
        df["prefix"] = "generate question: "
        df["target_text"] = df["target_text"].fillna("n/a").astype(str)
        return df

    def train(self, train_df: pd.DataFrame) -> None:
        """Train the model.

        Args:
            train_df: Training data.
        """
        self._model.train_model(train_df)

    def eval(self, eval_df: pd.DataFrame) -> Dict[str, float]:
        """Evaluate the model.

        Args:
            eval_df: Evaluation data.

        Returns:
            Evaluation results.
        """
        return self._model.eval_model(eval_df)

    def predict(self, to_predict: List[str]) -> str:
        """Predict the question for the given text.

        Args:
            to_predict: Text to predict the question for.

        Returns:
            Predicted question.
        """
        return self._model.predict([to_predict])[0]

    def generate_questions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate outputs for NQG.

        Returns:
            Outputs.
        """
        return self.predict(df["sentence"].tolist())
