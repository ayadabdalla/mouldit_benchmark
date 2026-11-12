from sklearn import ensemble
from sklearn.preprocessing import StandardScaler

from .base import CycleClassifier, flatten


class RandomForest(CycleClassifier):
    """Random forest on the standardised cycle; novelty = 1 - highest class probability."""

    def __init__(self, n_estimators: int = 200, random_state=None, n_jobs: int = -1):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y, X_val=None, y_val=None):
        self.scaler = StandardScaler().fit(flatten(X))
        self.model = ensemble.RandomForestClassifier(
            n_estimators=self.n_estimators, random_state=self.random_state, n_jobs=self.n_jobs,
        ).fit(self.scaler.transform(flatten(X)), y)
        return self

    def score(self, X):
        F = self.scaler.transform(flatten(X))
        return self.model.predict(F), 1 - self.model.predict_proba(F).max(1)
