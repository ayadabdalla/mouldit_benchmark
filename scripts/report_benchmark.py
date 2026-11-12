"""
Summary tables of the benchmark (run_benchmark.py calls this at the end; it also works on a run in progress).

    python scripts/report_benchmark.py [--dir results/erlanger_traeger_benchmark]

Writes to the results directory:
    summary.csv   mean and std over seeds per (modality, size, method) of every metric and the sorting loss;
                  holdout metrics are first averaged over the held-out modes of each seed, as in the paper
    per_mode.csv  holdout metrics per held-out failure mode, mean and std over seeds
    summary.md    both at the largest size, held-out modes ordered from hardest to easiest
"""

import argparse
import os

import numpy as np

from mouldit_benchmark.datasets import MODALITIES, failure_mode_difficulty
from mouldit_benchmark.evaluation import ResultsAnalyzer
from mouldit_benchmark.evaluation.results_analyzer import HOLDOUT_METRICS
from mouldit_benchmark.paths import RESULTS_DIR

METRICS = {   # key: (label, higher is better)
    "accuracy": ("Balanced accuracy", True),
    "auroc": ("Unknown AUROC", True),
    "detect": ("Unknown detection", True),
    "good_parts_rejected": ("Good parts rejected", False),
    "known_faults_shipped": ("Known faults shipped", False),
    "unknown_faults_shipped": ("Unknown faults shipped", False),
    "unknown_faults_rejected_as_known": ("Unknown rejected as known", False),
    "sorting": ("Sorting loss", False),
    "fidelity_known": ("Known: deviation R²", True),
    "shape_known": ("Known: DTW shape fidelity", True),
    "tdi_known": ("Known: TDI", False),
    "fidelity_unknown": ("Unknown: deviation R²", True),
    "shape_unknown": ("Unknown: DTW shape fidelity", True),
    "tdi_unknown": ("Unknown: TDI", False),
}
METHOD_ORDER = ["Reference", "kNN-raw", "LogReg", "RF", "PCA", "RF + PCA", "mould_it"]
METHOD_LABELS = {"Reference": "Reference (always OK)", "kNN-raw": "kNN", "LogReg": "Logistic regression",
                 "RF": "Random forest", "PCA": "PCA", "RF + PCA": "RF + PCA flag", "mould_it": "MouldIt"}
GROUP = ["modality", "total", "method"]


def aggregate(df, seeds):
    """
    Mean and std over seeds of the per-seed values (ResultsAnalyzer.per_seed), so the std is the seed spread
    reported in the paper, not the spread over held-out modes.
    """
    keys = [k for k in METRICS if k in seeds]
    agg = seeds.groupby(GROUP, sort=False)[keys].agg(["mean", "std"])
    agg.columns = [f"{k}_{s}" for k, s in agg.columns]
    g = df.groupby(GROUP, sort=False)
    agg["n_cycles"] = g.n_cycles.first()
    agg["seeds"] = g.seed.nunique()
    agg["held_out_modes"] = g.holdout.nunique()
    return agg.reset_index()


def per_mode(df):
    holdout = df[df.holdout.notna()]
    keys = [k for k in HOLDOUT_METRICS if k in holdout]
    pm = holdout.groupby(GROUP + ["holdout"], sort=False)[keys].agg(["mean", "std"])
    pm.columns = [f"{k}_{s}" for k, s in pm.columns]
    return pm.reset_index()


def winners(col, key):
    """Methods with the best value at 3 decimals, ties included; Reference, which accepts every part, is excluded."""
    c = col.drop("Reference", errors="ignore").dropna().round(3)
    if c.empty:
        return set()
    return set(c[c == (c.max() if METRICS[key][1] else c.min())].index)


def cell(text, bold):
    return f"**{text}**" if bold else text


def markdown(agg, pm):
    lines = ["# Erlanger Träger benchmark", "",
             "Mean ± std over seeds at the largest dataset size; unknown-fault metrics are first averaged over "
             "the held-out modes of each seed. The sorting loss is the mean of the four sorting errors. "
             "RF + PCA flag is random forests with the PCA unknown flag. "
             "Bold: best in the column (Reference excluded).", ""]
    for modality in dict.fromkeys(agg.modality):
        a = agg[agg.modality == modality]
        a = a[a.total == a.total.max()].set_index("method")
        a = a.loc[[m for m in METHOD_ORDER if m in a.index]]
        keys = [k for k in METRICS if f"{k}_mean" in a and a[f"{k}_mean"].notna().any()]
        lines += [f"## {modality}, {a.n_cycles.iloc[0]} cycles ({a.seeds.max()} seeds, "
                  f"{a.held_out_modes.max()} held-out modes)", "",
                  "| method | " + " | ".join(METRICS[k][0] for k in keys) + " |", "|---" * (len(keys) + 1) + "|"]
        win = {k: winners(a[f"{k}_mean"], k) for k in keys}
        for name, row in a.iterrows():
            lines.append(f"| {METHOD_LABELS.get(name, name)} | " + " | ".join(
                "–" if np.isnan(row[f"{k}_mean"]) else
                cell(f"{row[f'{k}_mean']:.3f} ± {np.nan_to_num(row[f'{k}_std']):.3f}", name in win[k])
                for k in keys) + " |")
        lines += ["", *per_mode_markdown(pm, modality)]
    return "\n".join(lines)


def per_mode_markdown(pm, modality):
    p = pm[pm.modality == modality]
    p = p[p.total == p.total.max()]
    difficulty = failure_mode_difficulty(MODALITIES[modality])
    modes = sorted(set(p.holdout), key=lambda m: difficulty.get(m, np.inf))
    lines = [f"### {modality}: per held-out failure mode",
             "Hardest first; under each mode its RMS deviation from the reference relative to OK cycles.", ""]
    for k in HOLDOUT_METRICS:
        if f"{k}_mean" not in p or p[f"{k}_mean"].isna().all():
            continue
        t = p.pivot(index="method", columns="holdout", values=f"{k}_mean")
        t = t.loc[[m for m in METHOD_ORDER if m in t.index and not t.loc[m].isna().all()], modes]
        t["mean"] = t.mean(axis=1)
        win = {c: winners(t[c], k) for c in t.columns}
        lines += [f"**{METRICS[k][0]}**", "",
                  "| method | " + " | ".join(f"{m}<br>({difficulty[m]:.2f})" for m in modes) + " | mean |",
                  "|---" * (len(modes) + 2) + "|"]
        for name, row in t.iterrows():
            values = ("–" if np.isnan(v) else cell(f"{v:.3f}", name in win[c]) for c, v in row.items())
            lines.append(f"| {METHOD_LABELS.get(name, name)} | " + " | ".join(values) + " |")
        lines.append("")
    return lines


def report(results_dir):
    analyzer = ResultsAnalyzer(results_dir)
    df = analyzer.records()
    agg, pm = aggregate(df, analyzer.per_seed()), per_mode(df)
    agg.to_csv(os.path.join(results_dir, "summary.csv"), index=False)
    pm.to_csv(os.path.join(results_dir, "per_mode.csv"), index=False)
    md = markdown(agg, pm)
    open(os.path.join(results_dir, "summary.md"), "w").write(md)
    print(md)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=RESULTS_DIR)
    report(ap.parse_args().dir)
