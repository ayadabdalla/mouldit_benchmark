"""
Metrics of the paper, in four groups:

  Classification             how often a method names the right class, with every class known.
  Unknown detection          whether it flags cycles of a fault type that was left out of training.
  Part sorting               a part ships only if the method predicts it OK and does not flag it.  That
                             decision can go wrong in four ways (the sorting errors); their mean
                             summarises them in one number (the paper's sorting loss).
  Reconstruction fidelity    how much of a fault a reconstructing method (MouldIt, PCA) keeps.

In the unknown-detection and sorting metrics, y_test holds the known classes (OK = 0) and -1 for the
unknown fault, and a cycle is flagged when its novelty exceeds the `quantile`-th percentile of the
novelty on the known validation cycles.
"""

import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

SORTING_ERRORS = ["good_parts_rejected", "known_faults_shipped", "unknown_faults_shipped",
                  "unknown_faults_rejected_as_known"]


def balanced_accuracy(pred, y):
    """Mean of the per-class accuracies, so every class counts equally."""
    return float(balanced_accuracy_score(y, pred))


def unknown_detection(novelty_val, novelty_test, y_test, quantile: float = 95):
    """
      auroc   how well the novelty score separates unknown from known cycles, without a threshold
      detect  share of unknown cycles flagged
    """
    is_unknown = y_test == -1
    flag = novelty_test > np.percentile(novelty_val, quantile)
    return {"auroc": float(roc_auc_score(is_unknown, novelty_test)), "detect": float(flag[is_unknown].mean())}


def part_sorting_errors(novelty_val, novelty_test, pred_test, y_test, quantile: float = 95):
    """
      good_parts_rejected               OK cycles flagged or predicted a fault
      known_faults_shipped              known-fault cycles not flagged and predicted OK
      unknown_faults_shipped            unknown cycles not flagged and predicted OK
      unknown_faults_rejected_as_known  unknown cycles not flagged and predicted a known fault, so the new
                                        fault is never reported as new
    """
    flag = novelty_test > np.percentile(novelty_val, quantile)
    ship = ~flag & (pred_test == 0)
    ok, known, unknown = y_test == 0, y_test > 0, y_test == -1
    return {
        "good_parts_rejected": float((~ship[ok]).mean()),
        "known_faults_shipped": float(ship[known].mean()),
        "unknown_faults_shipped": float(ship[unknown].mean()),
        "unknown_faults_rejected_as_known": float((~flag[unknown] & (pred_test[unknown] > 0)).mean()),
    }


def aggregate_sorting_errors(errors):
    """Mean of the four sorting errors (the paper's sorting loss); errors is a DataFrame or dict with their keys."""
    if isinstance(errors, dict):
        return float(np.mean([errors[k] for k in SORTING_ERRORS]))
    return errors[SORTING_ERRORS].mean(axis=1)


def deviation_r2(x, x_hat, reference):
    """
    Signal fidelity: 1 − Σ(x̂ − x)² / Σ(x − ref)² over cycles x (N, C, T).

    The fraction of each cycle's deviation from the reference cycle that the
    reconstruction reproduces: 0 for outputting the reference, 1 for perfect.
    Plain MSE would be misleading, as copying the reference already scores low.
    """
    return float(1 - ((x_hat - x) ** 2).sum() / ((x - reference[None]) ** 2).sum())


def dtw_fidelity(x, x_hat, reference, band_fraction: float = 0.10):
    """
    How well reconstructions x_hat keep the shape and the timing of the cycles x.

    DTW matches every sample of a cycle with the most similar sample of its reconstruction, allowing
    small shifts in time (at most band_fraction of the cycle).  Two numbers come out:

      shape  how close the matched samples are, compared with copying the reference cycle instead:
             1 is a perfect shape, 0 no better than the reference.  Timing differences don't count.
      tdi    how far apart in time the matched samples are (temporal_distortion_index), mean over cycles.
    """
    dtw = BandedDTW(band=max(1, int(band_fraction * x.shape[2])))
    cost_reconstruction, matched = dtw(x, x_hat)
    cost_reference, _ = dtw(x, np.broadcast_to(reference, x.shape))
    return {"shape": float(1 - cost_reconstruction.sum() / cost_reference.sum()),
            "tdi": float(temporal_distortion_index(matched).mean())}


def temporal_distortion_index(matched):
    """
    Temporal distortion index (TDI): how much a reconstruction runs early or late compared with its cycle.

    DTW matches every time step of the cycle with a time step of the reconstruction.  If the
    reconstruction is on time, each step is matched with the same step and the TDI is 0.  If it runs
    early or late, steps are matched with earlier or later steps, and the TDI grows with how far apart
    the matched steps are, summed over the whole cycle.

    `matched` comes from BandedDTW.
    """
    T = matched.shape[1]
    return np.abs(matched - np.arange(T)[None, :]).sum(1) / (T * T / 2)


class BandedDTW:
    """
    Matches the samples of two cycles in time.

    DTW walks through both cycles from start to end and pairs every sample of one with the most similar
    sample of the other.  It may stretch or squeeze time, but never go backwards.  All sensors are matched
    together, so a sample's partner is the same for every channel.

    The band keeps the matching realistic: a sample may only be paired with samples at most `band` steps
    away in time.  Without it, DTW could explain almost any difference away as a timing shift.

    It works in two passes over a table with one row per step of x and one column per step of y:
    the forward pass (fill_cost_table) finds the cost of the best matching and records its steps,
    the backward pass (trace_best_matching) follows those records to find the matching itself.

    Called on x and y of shape (N, C, T), it returns for every pair (x[n], y[n]) the total cost of its
    best matching, and for every sample of x the time index of its partner in y.
    """

    BOTH, X_ONLY, Y_ONLY = 0, 1, 2   # the three steps into a cell: both cycles move on, only x, only y

    def __init__(self, band: int, batch_size: int = 64):
        self.band = band
        self.batch_size = batch_size

    def __call__(self, x, y):
        costs, matched = [], []
        for start in range(0, len(x), self.batch_size):
            batch = slice(start, start + self.batch_size)
            cost, indices = self.match(np.asarray(x[batch], np.float64), np.asarray(y[batch], np.float64))
            costs.append(cost)
            matched.append(indices)
        return np.concatenate(costs), np.concatenate(matched)

    def match(self, x, y):
        T = x.shape[2]
        cost_table, best_steps = self.fill_cost_table(self.local_cost(x, y))
        return cost_table[:, T, T], self.trace_best_matching(best_steps)

    @staticmethod
    def local_cost(x, y):
        """Cost of pairing sample i of x with sample j of y: their squared distance over all channels."""
        return ((x[:, :, :, None] - y[:, :, None, :]) ** 2).sum(1)

    def fill_cost_table(self, local):
        """
        Forward pass.  Builds the cost table, one row per step of x and one column per step of y, and fills
        it from the start: cell (i, j) gets the cheapest total cost of matching both cycles up to step i of
        x with step j of y, and best_steps records the step that cheapest matching took into the cell
        (BOTH, X_ONLY or Y_ONLY).

        Only steps that lie within the band of each other are paired; all other cells are left at infinite
        cost.  Row and column 0 stand for "before the first step", where every matching begins.
        """
        N, T, _ = local.shape
        cost_table = np.full((N, T + 1, T + 1), np.inf)
        cost_table[:, 0, 0] = 0.0
        best_steps = np.zeros((N, T + 1, T + 1), np.int8)
        for i in range(1, T + 1):
            for j in range(max(1, i - self.band), min(T, i + self.band) + 1):
                both_advance = cost_table[:, i - 1, j - 1]   # x and y each moved one step on
                x_advances = cost_table[:, i - 1, j]         # only x moved on: step j of y is matched twice
                y_advances = cost_table[:, i, j - 1]         # only y moved on: step i of x is matched twice
                cheapest = np.minimum(np.minimum(both_advance, x_advances), y_advances)
                cost_table[:, i, j] = local[:, i - 1, j - 1] + cheapest
                best_steps[:, i, j] = np.stack([both_advance, x_advances, y_advances], 1).argmin(1)
        return cost_table, best_steps

    @classmethod
    def trace_best_matching(cls, best_steps):
        """
        Backward pass.  The best matching: for every step of x, the step of y it is matched with (the mean,
        if several).
        """
        N, T = best_steps.shape[0], best_steps.shape[1] - 1
        matched = np.zeros((N, T))
        for n in range(N):
            partners = [[] for _ in range(T)]   # steps of y matched with each step of x
            i, j = T, T                          # start at the last cell
            while i > 0 and j > 0:
                partners[i - 1].append(j - 1)
                step = best_steps[n, i, j]
                if step != cls.Y_ONLY:           # x moved on into this cell, so step back in x
                    i -= 1
                if step != cls.X_ONLY:           # y moved on into this cell, so step back in y
                    j -= 1
            matched[n] = [np.mean(p) for p in partners]
        return matched
