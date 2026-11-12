import numpy as np
from sklearn import decomposition

from ..novelty import ClassGaussians
from .base import CycleClassifier, flatten


class PCA(CycleClassifier):
    """
    PCA process monitoring with the novelty score of MouldIt: per-class Hotelling T^2 on the PCA scores
    as the latent term, the squared prediction error (SPE) as the reconstruction term.

    prediction = class q with the smallest normalised T^2_q / eta_q
    novelty    = max(min_q T^2_q / eta_q, SPE / eta_SPE)
    """

    def __init__(self, n_components: int = 32, quantile: float = 95, random_state=None):
        self.n_components = n_components
        self.quantile = quantile
        self.random_state = random_state

    def fit(self, X, y, X_val, y_val):
        A, A_val = flatten(X), flatten(X_val)
        self.pca = decomposition.PCA(n_components=min(self.n_components, len(A) - 1),
                                     random_state=self.random_state).fit(A)
        self.gaussians = ClassGaussians(self.quantile).fit(self.pca.transform(A), y, self.pca.transform(A_val), y_val)
        self.eta_spe = np.percentile(self._spe(A_val), self.quantile)
        return self

    def _spe(self, A):
        return ((A - self.pca.inverse_transform(self.pca.transform(A))) ** 2).sum(1)

    def novelty_terms(self, X):
        """(N, Q) normalised class distances on the PCA scores, and SPE / eta_SPE."""
        A = flatten(X)
        return self.gaussians.distances(self.pca.transform(A)), self._spe(A) / self.eta_spe

    def score(self, X):
        A = flatten(X)
        t2, pred = self.gaussians.score(self.pca.transform(A))
        return pred, np.maximum(t2, self._spe(A) / self.eta_spe)

    def transform(self, X):
        return self.pca.transform(flatten(X))

    def reconstruct(self, X):
        return self.pca.inverse_transform(self.transform(X)).reshape(X.shape)
