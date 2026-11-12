"""
Analyses the novelty score of MouldIt and PCA by its latent and reconstruction terms (paper: Latent
Analysis), rebuilt from a benchmark run's saved weights and checked against its saved novelty scores.
"""

import os

import numpy as np
from omegaconf import OmegaConf

from ..models import PCA, MouldItRunner
from .method_evaluator import MethodEvaluator


class LatentAnalyzer:
    def __init__(self, results_dir, device=None):
        self.results_dir = str(results_dir)
        self.cfg = OmegaConf.load(os.path.join(self.results_dir, "config.yaml"))
        self.device = device

    def analyze(self, dataset, modality, total, seed, holdout):
        """
        dataset: the fold's SyntheticDataset.  Returns (y_test, {method: terms}) with terms
          d          (N, Q) test distances to each known class, in units of its 95 % boundary
          L, R       latent term min_q d and reconstruction term R / eta_R of each test cycle
          threshold  Theta, the 95th percentile of max(L, R) on the known validation cycles
        """
        cfg = self.cfg
        fold = MethodEvaluator(dataset, modality, total, seed, holdout, cfg.quantile)
        (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = fold.train, fold.validation, fold.test
        mouldit = MouldItRunner(beta=cfg.mould_it.beta[modality], code=cfg.code_dim, quantile=cfg.quantile,
                                architecture=cfg.mould_it.architecture, device=self.device)
        weights = os.path.join(self.results_dir, "models", f"{fold.name}.pt")
        fitted = {"mould_it": mouldit.load(weights, X_tr, y_tr, X_va, y_va),
                  "PCA": PCA(cfg.code_dim, cfg.quantile, random_state=seed).fit(X_tr, y_tr, X_va, y_va)}
        saved = np.load(os.path.join(self.results_dir, "outputs", f"{fold.name}.npz"))

        terms = {}
        for method, model in fitted.items():
            d_va, r_va = model.novelty_terms(X_va)
            d, r = model.novelty_terms(X_te)
            if not np.allclose(np.maximum(d.min(1), r), saved[f"{method}__novelty_test"], rtol=1e-3, atol=1e-4):
                raise RuntimeError(f"{fold.name}: rebuilt {method} novelty differs from the saved benchmark output")
            terms[method] = {"d": d, "L": d.min(1), "R": r,
                             "threshold": np.percentile(np.maximum(d_va.min(1), r_va), cfg.quantile)}
        return y_te, terms
