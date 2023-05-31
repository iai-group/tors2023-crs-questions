import pandas as pd

from questions.models._classifier import Classifier

TEMPLATE = "Are you looking for a {category} that is great for {question_part}"


class TQG:
    def __init__(self, use_classifier: bool = False) -> None:
        """Template-based question generator.

        Args:
            use_classifier: Whether to use classifier before generating
              questions.
        """
        self.classifier = Classifier() if use_classifier else None

    def get_question(self, category: str, sentence: str, label: int = 1) -> str:
        """Get pattern for TQG.

        Args:
            category: Category.
            sentence: Sentence.

        Returns:
            Question.
        """
        question_part = sentence.lower().split(" for ")
        if len(question_part) == 0 or label == 0:
            return "n/a"

        question_part = " for ".join(question_part[1:])
        return TEMPLATE.format(
            category=category.lower(), question_part=question_part
        )

    def generate_questions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate outputs for TQG.

        Returns:
            Outputs.
        """
        columns = ["category", "sentence"]
        if self.classifier is not None:
            df["label"] = self.classifier.predict(df["sentence"].tolist())
            columns.append("label")
        return df[columns].apply(lambda row: self.get_question(*row), axis=1)


if __name__ == "__main__":
    question_generator = TQG()
    df = pd.read_csv("data/test.csv")
    outputs = question_generator.generate_questions(df)
    print(outputs)
