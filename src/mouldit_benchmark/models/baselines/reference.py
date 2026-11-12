import numpy as np

from .base import CycleClassifier


class Reference(CycleClassifier):
    """Accept every part: predicts OK (class 0) and never flags a cycle."""

    def __init__(self, label: int = 0):
        self.label = label

    def fit(self, X, y, X_val=None, y_val=None):
        return self

    def score(self, X):
        return np.full(len(X), self.label), np.zeros(len(X))
