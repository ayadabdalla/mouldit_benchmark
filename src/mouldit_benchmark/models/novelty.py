"""The latent term of the novelty score (paper: Novelty Score), shared by MouldIt and PCA."""

import numpy as np
from sklearn.covariance import LedoitWolf


class ClassGaussians:
    """
    One Ledoit-Wolf Gaussian per class.  distances(Z)[:, q] is M_q / eta_q: the squared Mahalanobis
    distance to class q over its `quantile`-th percentile on that class's validation samples.
    """

    def __init__(self, quantile: float = 95):
        self.quantile = quantile

    def fit(self, Z, y, Z_val, y_val):
        self.classes = np.unique(y)
        self.models = [LedoitWolf().fit(Z[y == c]) for c in self.classes]
        d_val = self._mahalanobis(Z_val)
        self.eta = np.array([np.percentile(d_val[y_val == c, k], self.quantile)
                             for k, c in enumerate(self.classes)])
        return self

    def _mahalanobis(self, Z):
        return np.stack([m.mahalanobis(Z) for m in self.models], 1)

    def distances(self, Z):
        """(N, Q) normalised distances; above 1 is outside that class's 95 % boundary."""
        return self._mahalanobis(Z) / self.eta

    def score(self, Z):
        """(latent term L = min_q M_q / eta_q, nearest class)."""
        d = self.distances(Z)
        return d.min(1), self.classes[d.argmin(1)]
