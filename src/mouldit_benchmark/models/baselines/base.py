import numpy as np


def flatten(X: np.ndarray) -> np.ndarray:
    """(N, C, T) cycles -> (N, C T) feature vectors."""
    return X.reshape(len(X), -1)


class CycleClassifier:
    """
    Cycles X (N, C, T) and labels y in 0..K-1.

      fit(X, y, X_val, y_val)    validation data is used only by methods that calibrate on it
      score(X) -> (prediction, novelty), novelty higher for cycles less like any training class
    """

    def fit(self, X, y, X_val=None, y_val=None):
        raise NotImplementedError

    def score(self, X):
        raise NotImplementedError

    def predict(self, X):
        return self.score(X)[0]

    def novelty(self, X):
        return self.score(X)[1]
