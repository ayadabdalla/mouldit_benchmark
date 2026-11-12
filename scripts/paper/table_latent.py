"""
Table III and the numbers of the Latent Analysis section; writes table_latent.tex and latent_analysis.csv.
Needs the saved MouldIt weights of a finished benchmark run (models/*.pt).

    python scripts/paper/table_latent.py [--dir results/erlanger_traeger_benchmark] [--out results/paper/tables]
                                         [--sizes 200 1000 4000 16000]
"""

import argparse
import os

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from sklearn.metrics import roc_auc_score

from mouldit_benchmark.paths import RESULTS_DIR, TABLES_DIR
from mouldit_benchmark.datasets import MODALITIES, SyntheticDataset, n_classes
from mouldit_benchmark.evaluation import LatentAnalyzer

ROWS = {   # hardest first: by RMS deviation from the reference relative to OK cycles, smallest first
           # (datasets.failure_mode_difficulty), as in Table III
    "pressure_only": [("machine_drift", "Machine drift"), ("secondary_pressure_rise", "Secondary pressure rise"),
                      ("pressure_peak_scale", "Underpacking/overpacking"), ("timing_shift", "Time shift"),
                      ("fast_gate_freeze", "Fast gate freeze")],
    "pt_us": [("secondary_pressure_rise", "Secondary pressure rise"),
              ("pressure_peak_scale", "Underpacking/overpacking"), ("machine_drift", "Machine drift"),
              ("temperature_dynamics", "Temperature dynamics"),
              ("inter_sensor_inconsistency", "Inter-sensor inconsistency"), ("fast_gate_freeze", "Fast gate freeze"),
              ("distort_us_response", "Distorted US response"), ("timing_shift", "Time shift")],
}
TITLES = {"pressure_only": "Single-modal", "pt_us": "Multi-modal"}


def analyse_fold(y, t):
    """Shares of the unknown cycles by flagging term, and where unknown and OK cycles lie in the latent space."""
    u, ok = y == -1, y == 0
    L, R, d = t["L"], t["R"], t["d"]
    flag_l, flag_r = L > t["threshold"], R > t["threshold"]
    d_ok, d_fault = d[:, 0], d[:, 1:].min(1)
    return dict(threshold=t["threshold"], detect=(flag_l | flag_r)[u].mean(), latent_only=(flag_l & ~flag_r)[u].mean(),
                recon_only=(flag_r & ~flag_l)[u].mean(), both=(flag_l & flag_r)[u].mean(), in_class=(L[u] <= 1).mean(),
                in_ok=(d_ok[u] <= 1).mean(), auroc_latent=roc_auc_score(u, L), auroc_recon=roc_auc_score(u, R),
                auroc=roc_auc_score(u, np.maximum(L, R)), ok_in_fault=(d_fault[ok] <= 1).mean(),
                ok_dfault_med=np.median(d_fault[ok]), u_L_med=np.median(L[u]), u_R_med=np.median(R[u]),
                known_R_med=np.median(R[~u]), u_dok_med=np.median(d_ok[u]))


def collect(results_dir, sizes):
    seeds = OmegaConf.load(os.path.join(results_dir, "config.yaml")).seeds
    analyzer, rows = LatentAnalyzer(results_dir), []
    for modality, channels in MODALITIES.items():
        for total in sizes:
            for seed in seeds:
                ds = SyntheticDataset(channels, total // n_classes(channels), seed)
                for holdout in ds.classes[1:]:
                    y, terms = analyzer.analyze(ds, modality, total, seed, holdout)
                    rows += [dict(modality=modality, total=total, seed=seed, holdout=holdout, method=m)
                             | analyse_fold(y, t) for m, t in terms.items()]
                print(f"{modality} n={total} seed={seed}: done", flush=True)
    return pd.DataFrame(rows)


def latent_table(stats):
    d = stats[stats.total == stats.total.max()]
    g = d.groupby(["modality", "holdout", "method"])[["detect", "latent_only", "recon_only", "both"]].mean() * 100
    rows = []
    for modality, modes in ROWS.items():
        rows.append("\\multicolumn{9}{@{}l}{\\textit{" + TITLES[modality] + "}}\\\\")
        for key, label in modes:
            cells = [label]
            for col in ["detect", "latent_only", "recon_only", "both"]:
                cells += [f"{g.loc[(modality, key, m), col]:.0f}" for m in ("mould_it", "PCA")]
            rows.append(" & ".join(cells) + " \\\\")
        rows.append("\\midrule")
    return rows[:-1]


def print_latent_analysis(stats):
    """Means over held-out modes and seeds per size; the per-mode values at the largest size."""
    pd.set_option("display.width", 200, "display.max_columns", None)
    per_size = stats.groupby(["modality", "method", "total"])[
        ["ok_in_fault", "ok_dfault_med", "in_class", "recon_only", "detect"]].mean()
    print("\nper size (mean over held-out modes and seeds)\n", per_size.round(3))
    largest = stats[stats.total == stats.total.max()]
    per_mode = largest.groupby(["modality", "holdout", "method"])[["in_ok", "u_R_med", "auroc_recon"]].agg(["mean", "std"])
    print(f"\nper held-out mode at {stats.total.max()} cycles (mean and std over seeds)\n", per_mode.round(3))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=RESULTS_DIR)
    ap.add_argument("--out", default=TABLES_DIR)
    ap.add_argument("--sizes", type=int, nargs="+", default=[200, 1000, 4000, 16000])
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    stats = collect(args.dir, args.sizes)
    stats.to_csv(os.path.join(args.out, "latent_analysis.csv"), index=False)
    rows = latent_table(stats)
    open(os.path.join(args.out, "table_latent.tex"), "w").write("\n".join(rows) + "\n")
    print("% table_latent.tex\n" + "\n".join(rows))
    print_latent_analysis(stats)
