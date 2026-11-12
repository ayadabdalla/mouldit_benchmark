"""
Analyses a benchmark run: the records in results.json, plus the sorting errors computed from the per-cycle
outputs in outputs/*.npz.
"""

import glob
import json
import os
import re

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from .metrics import SORTING_ERRORS, aggregate_sorting_errors, part_sorting_errors

KEYS = ["modality", "total", "seed", "holdout", "method"]
CLASSIFICATION_METRICS = ["accuracy", "fidelity_known", "shape_known", "tdi_known"]
HOLDOUT_METRICS = ["auroc", "detect", *SORTING_ERRORS, "fidelity_unknown", "shape_unknown", "tdi_unknown"]
# Sorting errors stored in results.json are ignored; all four are recomputed from outputs/*.npz.  The published
# run stored unknown faults shipped under the earlier name unknown_to_ok.
STORED_SORTING_KEYS = ["unknown_to_ok", *SORTING_ERRORS]


class ResultsAnalyzer:
    def __init__(self, results_dir):
        self.results_dir = str(results_dir)
        self.quantile = OmegaConf.load(os.path.join(self.results_dir, "config.yaml")).quantile

    def records(self):
        """
        One row per fold and method, with the sorting errors; holdout is None in the classification fold.
        "RF + PCA" has rows in holdout folds only, with sorting errors only.
        """
        df = pd.DataFrame(json.load(open(os.path.join(self.results_dir, "results.json"))))
        df = df.drop(columns=[k for k in STORED_SORTING_KEYS if k in df])
        ops = self.sorting_errors()
        if len(ops):
            df = df.merge(ops, on=KEYS, how="outer")   # outer: "RF + PCA" is not in results.json of new runs
            df["n_cycles"] = df.groupby(["modality", "total"]).n_cycles.transform("first").astype(int)
        return df

    def sorting_errors(self):
        """
        The four sorting errors (metrics.part_sorting_errors) of every method in every holdout fold, from the
        saved predictions and novelty scores.  "RF + PCA" takes RF's prediction and PCA's unknown flag.
        """
        rows = []
        for path in sorted(glob.glob(os.path.join(self.results_dir, "outputs", "*.npz"))):
            m = re.fullmatch(r"(.+)_n(\d+)_seed(\d+)_(.+)\.npz", os.path.basename(path))
            if m is None or m[4] == "all-classes":
                continue
            z = np.load(path)
            fold = dict(zip(KEYS, (m[1], int(m[2]), int(m[3]), m[4])))
            methods = {k.split("__")[0]: (k.split("__")[0],) * 2 for k in z.files if k.endswith("__novelty_test")}
            if {"RF", "PCA"} <= methods.keys():
                methods["RF + PCA"] = ("RF", "PCA")
            for name, (pred, flag) in methods.items():
                errors = part_sorting_errors(z[f"{flag}__novelty_val"], z[f"{flag}__novelty_test"],
                                             z[f"{pred}__pred"], z["y_test"], self.quantile)
                rows.append(fold | {"method": name} | {k: errors[k] for k in SORTING_ERRORS})
        return pd.DataFrame(rows, columns=KEYS + SORTING_ERRORS)

    def per_seed(self):
        """
        One row per (modality, total, method, seed): the classification-fold metrics, the holdout metrics
        averaged over the held-out failure modes, and the mean of the four sorting errors (the paper's sorting loss).
        """
        df = self.records()
        keys = ["modality", "total", "method", "seed"]
        cls = df[df.holdout.isna()].groupby(keys)[[k for k in CLASSIFICATION_METRICS if k in df]].mean()
        loo = df[df.holdout.notna()].groupby(keys)[[k for k in HOLDOUT_METRICS if k in df]].mean()
        out = loo.join(cls, how="outer").reset_index()
        out["sorting"] = aggregate_sorting_errors(out)
        return out
