"""Does automatic evaluation of a model with respect to the test set."""


import argparse
from typing import List

import pandas as pd
from nlgeval import NLGEval
from sklearn.metrics import confusion_matrix
from statsmodels.stats.contingency_tables import mcnemar
from tqdm import tqdm

from questions.util import file_io

GENERATED_QUESTION_COLUMNS = [
    "TQG_question",
    "TQG_roberta_question",
    "NRQG_question",
    "NSQG_question",
]
GENERATED_CLASSIFICATION_COLUMNS = [
    "TQG_label",
    "TQG_roberta_label",
    "NRQG_label",
    "NSQG_label",
]
METRICS = ["Bleu_4", "ROUGE_L", "METEOR"]


class Evaluation:
    def __init__(self, path):
        self._df = file_io.get_dataframe_from_csv(path)
        self._nlgeval = None

    @property
    def nlgeval(self):
        if not self._nlgeval:
            self._nlgeval = NLGEval(no_glove=True, no_skipthoughts=True)
        return self._nlgeval

    @property
    def dataframe(self):
        return self._df

    def _get_valid_rows(self, df, refrence_columns, hypotheses_column):
        """Get valid rows."""
        return df[
            df[refrence_columns].notnull().all(axis=1)
            & df[hypotheses_column].notnull()
        ]

    def compute_generation_metrics(
        self,
        refrence_columns: List[str],
        hypotheses_column: str,
        df: pd.DataFrame = None,
    ):
        """Compute metrics."""
        df = self.dataframe if df is None else df
        df = self._get_valid_rows(df, refrence_columns, hypotheses_column)

        references = [df[ref].tolist() for ref in refrence_columns]
        return self.nlgeval.compute_metrics(
            references, df[hypotheses_column].tolist()
        )

    def confusion_matrix(self, baseline_column, model_column):
        """Confusion matrix."""
        return confusion_matrix(
            self._df[baseline_column], self._df[model_column]
        )

    def statistical_significance_classification(
        self, baseline_column, model_column
    ):
        """Statistical significance."""
        table = self.confusion_matrix(baseline_column, model_column)
        result = mcnemar(table, exact=True)
        return result

    def statistical_significance_generation(
        self,
        refrence_columns: List[str],
        hypotheses_column: str,
    ):
        """Statistical significance."""
        data = self.dataframe.index.tolist()
        statistic = self.compute_generation_metrics(
            refrence_columns, hypotheses_column
        )


def parse_args() -> argparse.Namespace:
    """Parse arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="Path to results file.")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    eval = Evaluation("data/test_results_all_models.csv")

    samples = [
        eval.dataframe.sample(len(eval.dataframe), replace=True)
        for _ in range(1000)
    ]

    results = {
        metric: {hypo: [] for hypo in GENERATED_QUESTION_COLUMNS}
        for metric in METRICS
    }

    for hypo in GENERATED_QUESTION_COLUMNS:
        for sample in tqdm(samples, total=len(samples)):
            sample_result = eval.compute_generation_metrics(
                [
                    "question1",
                    "question2",
                    "question3",
                    "paraphrase1",
                    "paraphrase2",
                ],
                hypo,
                sample,
            )
            for metric in METRICS:
                results[metric][hypo].append(sample_result[metric])


if __name__ == "__main__":
    args = parse_args()
    main(args)
