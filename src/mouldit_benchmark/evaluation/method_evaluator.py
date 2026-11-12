"""
Evaluates the methods on one fold (paper: Experiments).

A fold is one (modality, dataset size, seed, holdout).  holdout=None is the classification fold, with
every class known; otherwise that failure mode is held out as the unknown fault.

MethodEvaluator splits the fold's dataset once.  evaluate(method) then
  1. calls the trained method on the held-back cycles: predict() on the test cycles in the classification
     fold, score() on the validation and test cycles in a holdout fold, and reconstruct() on the faulty
     (or unknown) test cycles if the method can reconstruct;
  2. turns its outputs into the paper's metrics: balanced accuracy, or unknown AUROC and detection at a
     threshold set on the validation novelty; plus reconstruction fidelity;
  3. returns the metrics and the per-cycle outputs, from which ResultsAnalyzer computes the sorting errors.
"""

import numpy as np
from sklearn.model_selection import train_test_split

from .metrics import balanced_accuracy, deviation_r2, dtw_fidelity, unknown_detection


class MethodEvaluator:
    """
    train, validation, test: (X, y) of the fold.  In a holdout fold the held-out class is labelled -1 and
    only in the test set; the other classes are relabelled 0..K-1, OK staying 0.
    """

    def __init__(self, dataset, modality, total, seed, holdout=None, quantile=95, dtw_band=0.10):
        self.modality, self.total, self.seed, self.holdout = modality, total, seed, holdout
        self.quantile, self.dtw_band = quantile, dtw_band
        self.reference = dataset.reference
        held_out = None if holdout is None else dataset.classes.index(holdout)
        X_tr, y_tr, X_va, y_va, X_te, y_te = self.split(dataset.series, dataset.labels, held_out, seed)
        self.train, self.validation, self.test = (X_tr, y_tr), (X_va, y_va), (X_te, y_te)

    @property
    def name(self):
        """File stem of the fold's outputs and weights."""
        return self.fold_name(self.modality, self.total, self.seed, self.holdout)

    @staticmethod
    def fold_name(modality, total, seed, holdout):
        return f"{modality}_n{total}_seed{seed}_{holdout or 'all-classes'}"

    @staticmethod
    def split(X, y, holdout, seed):
        """Stratified 60/20/20 train/val/test; a holdout class index moves all its cycles to the test set as -1."""
        known = np.ones(len(y), bool) if holdout is None else y != holdout
        yk = y[known] if holdout is None else y[known] - (y[known] > holdout)
        i_tr, i_tmp = train_test_split(np.arange(len(yk)), test_size=0.4, stratify=yk, random_state=seed)
        i_va, i_te = train_test_split(i_tmp, test_size=0.5, stratify=yk[i_tmp], random_state=seed)
        Xk = X[known]
        X_te = np.concatenate([Xk[i_te], X[~known]])
        y_te = np.concatenate([yk[i_te], np.full((~known).sum(), -1)])
        return Xk[i_tr], yk[i_tr], Xk[i_va], yk[i_va], X_te, y_te

    def evaluate(self, method):
        """(record of metrics, per-cycle outputs) of a trained method on this fold."""
        (X_va, _), (X_te, y_te) = self.validation, self.test
        if self.holdout is None:
            pred = method.predict(X_te)
            record, outputs = {"accuracy": balanced_accuracy(pred, y_te)}, {"pred": pred}
            mask, subset = y_te > 0, "known"
        else:
            (_, nov_va), (pred, nov_te) = method.score(X_va), method.score(X_te)
            record = unknown_detection(nov_va, nov_te, y_te, self.quantile)
            outputs = {"pred": pred, "novelty_test": nov_te, "novelty_val": nov_va}
            mask, subset = y_te == -1, "unknown"
        if hasattr(method, "reconstruct"):
            x, x_hat = X_te[mask], method.reconstruct(X_te[mask])
            dtw = dtw_fidelity(x, x_hat, self.reference, self.dtw_band)
            record |= {f"fidelity_{subset}": deviation_r2(x, x_hat, self.reference),
                       f"shape_{subset}": dtw["shape"], f"tdi_{subset}": dtw["tdi"]}
        return record, outputs
