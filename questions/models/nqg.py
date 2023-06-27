"""Neural Question Generation (NQG) model."""

import logging
from pprint import pprint
from typing import Dict, List, Tuple, Union

import pandas as pd
from simpletransformers.t5 import T5Args, T5Model

from questions.util import file_io

transformers_logger = logging.getLogger("NQG")
transformers_logger.setLevel(logging.WARNING)

MODEL_PATH = "t5-base"
TRAINING_DATA_PATH = "data/train.csv"
EVALUATION_DATA_PATH = "data/test.csv"

PREFIX = "Ask questions"


class NQG:
    def __init__(
        self,
        model_path: str = None,
        use_reviews: bool = False,
    ) -> None:
        """
        Neural Question Generation (NQG) model.

        Args:
            model_path: Path of the trained model to use.
            use_reviews: Whether to use reviews or text for training.
        """
        self.use_reviews = use_reviews

        model_args = T5Args(
            num_train_epochs=3,
            overwrite_output_dir=True,
            do_sample=True,
            top_k=25,
            top_p=0.90,
            use_multiprocessing=False,
            n_gpu=1,
            train_batch_size=36,
            eval_batch_size=36,
            evaluate_generated_text=True,
            output_dir=f"outputs/{str(self)}",
        )

        self._model = T5Model(
            "t5",
            model_path or MODEL_PATH,
            args=model_args,
        )

        if model_path is None:
            train_df = self.get_dataframe(TRAINING_DATA_PATH)
            self.train(train_df)

            results = self.eval(self.get_dataframe(EVALUATION_DATA_PATH))
            pprint(results)

    def __str__(self) -> str:
        """String representation of the model."""
        return f"NQG{'_reviews' if self.use_reviews else ''}"

    def get_model_input_text(
        self, row: Union[Tuple[str], pd.Series]
    ) -> pd.Series:
        """Get the input text for the model.

        Args:
            row: Row of the dataframe containing category and text.
        """
        category, text = row
        return f"{category} <sep> {text}"

    def get_dataframe(self, path: str) -> pd.DataFrame:
        """Get training data for the model.

        Args:
            path: Path to the training data.

        Returns:
            Training data.
        """
        df = file_io.get_dataframe_from_csv(path)
        df["text"] = df["review"] if self.use_reviews else df["sentence"]
        df["input_text"] = df[["category", "text"]].apply(
            self.get_model_input_text, axis=1
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
        df["target_text"] = df["target_text"].fillna("n/a").astype(str)
        df["prefix"] = PREFIX
        return df[["prefix", "input_text", "target_text"]]

    def train(self, train_df: pd.DataFrame, **kwargs) -> None:
        """Train the model.

        Args:
            train_df: Training data.
            **kwargs: Keyword arguments to pass to the model training method.
        """
        self._model.train_model(train_df, **kwargs)

    def eval(self, eval_df: pd.DataFrame) -> Dict[str, float]:
        """Evaluate the model.

        Args:
            eval_df: Evaluation data.

        Returns:
            Evaluation results.
        """
        return self._model.eval_model(eval_df)

    def predict(self, to_predict: List[List[str]]) -> str:
        """Predict the question for the given text.

        Args:
            to_predict: Text to predict the question for.

        Returns:
            Predicted question.
        """
        to_predict = [
            f"generate question: {category} <sep> {text}"
            for category, text in to_predict
        ]
        return self._model.predict(to_predict)

    def generate_questions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate outputs for NQG.

        Returns:
            Outputs.
        """
        df["text"] = df["review"] if self.use_reviews else df["sentence"]
        return self.predict(df[["category", "text"]].values.tolist())
