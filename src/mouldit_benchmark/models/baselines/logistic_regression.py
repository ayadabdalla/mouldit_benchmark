from sklearn import linear_model
from sklearn.preprocessing import StandardScaler

from .base import CycleClassifier, flatten


class LogisticRegression(CycleClassifier):
    """Logistic regression on the standardised cycle; novelty = 1 - highest class probability."""

    def __init__(self, max_iter: int = 2000, random_state=None):
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X, y, X_val=None, y_val=None):
        self.scaler = StandardScaler().fit(flatten(X))
        self.model = linear_model.LogisticRegression(max_iter=self.max_iter, random_state=self.random_state)
        self.model.fit(self.scaler.transform(flatten(X)), y)
        return self

    def score(self, X):
        F = self.scaler.transform(flatten(X))
        return self.model.predict(F), 1 - self.model.predict_proba(F).max(1)
