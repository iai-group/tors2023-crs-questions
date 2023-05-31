from _classifier import Classifier


class TQG:
    def __init__(self, use_classifier: bool = False) -> None:
        if use_classifier:
            self.classifier = Classifier()
