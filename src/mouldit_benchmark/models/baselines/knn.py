import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from .base import CycleClassifier, flatten


class KNN(CycleClassifier):
    """
    k nearest neighbours of the standardised cycle.

    prediction = majority label of the k neighbours
    novelty    = mean distance to the k neighbours
    """

    def __init__(self, n_neighbors: int = 5):
        self.n_neighbors = n_neighbors

    def fit(self, X, y, X_val=None, y_val=None):
        self.scaler = StandardScaler().fit(flatten(X))
        self.nn = NearestNeighbors(n_neighbors=self.n_neighbors).fit(self.scaler.transform(flatten(X)))
        self.y = y
        return self

    def score(self, X):
        dist, idx = self.nn.kneighbors(self.scaler.transform(flatten(X)))
        pred = np.array([np.bincount(self.y[r]).argmax() for r in idx])
        return pred, dist.mean(1)
