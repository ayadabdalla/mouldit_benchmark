"""
Tables II and IV; writes table_results.tex, table_fidelity.tex and per_seed.csv.

    python scripts/paper/table_results.py [--dir results/erlanger_traeger_benchmark] [--out results/paper/tables]
"""

import argparse
import os

import pandas as pd

from mouldit_benchmark.paths import RESULTS_DIR, TABLES_DIR
from mouldit_benchmark.evaluation import ResultsAnalyzer

SETTINGS = [("pressure_only", "Single-modal (pressure; 6 classes)"),
            ("pt_us", "Multi-modal (pressure, temperature, two ultrasound channels; 9 classes)")]
METHODS = [("mould_it", "MouldIt"), ("RF + PCA", "RF + PCA flag"), ("kNN-raw", "kNN"), ("RF", "Random forest"),
           ("PCA", "PCA"), ("LogReg", "Logistic regression"), ("Reference", "Reference (always OK)")]
RESULT_COLUMNS = [   # metric, scale, decimals, +1 higher is better / -1 lower is better
    ("accuracy", 100, 1, 1), ("auroc", 1, 3, 1), ("detect", 100, 1, 1),
    ("good_parts_rejected", 100, 1, -1), ("known_faults_shipped", 100, 1, -1), ("unknown_faults_shipped", 100, 1, -1),
    ("unknown_faults_rejected_as_known", 100, 1, -1), ("sorting", 100, 1, -1)]
FIDELITY_ROWS = [("Known faults", None), ("Deviation $R^2$", "fidelity_known", 3, 1),
                 ("DTW shape fidelity", "shape_known", 3, 1), ("TDI", "tdi_known", 3, -1),
                 ("Unknown fault", None), ("Deviation $R^2$", "fidelity_unknown", 3, 1),
                 ("DTW shape fidelity", "shape_unknown", 2, 1), ("TDI", "tdi_unknown", 3, -1)]


def fmt(mean, std, decimals, bold):
    text = f"{mean:.{decimals}f} $\\pm$ {std:.{decimals}f}"
    return "\\textbf{" + text + "}" if bold else text


def results_table(d):
    """Bold: the best method per column at the decimals shown, Reference excluded."""
    rows = []
    for modality, title in SETTINGS:
        g = d[d.modality == modality].groupby("method")
        mean, std = g.mean(numeric_only=True), g.std(numeric_only=True)
        rows.append("\\multicolumn{9}{@{}l}{\\textit{" + title + "}}\\\\")
        for key, label in METHODS:
            cells = [label]
            for col, scale, dec, sign in RESULT_COLUMNS:
                v = mean.loc[key, col]
                if pd.isna(v):
                    cells.append("--")
                    continue
                best = (sign * mean.drop("Reference")[col].dropna() * scale).max() * sign
                bold = key != "Reference" and round(v * scale, dec) == round(best, dec)
                cells.append(fmt(v * scale, std.loc[key, col] * scale, dec, bold))
            rows.append(" & ".join(cells) + " \\\\")
        rows.append("\\midrule")
    return rows[:-1]


def fidelity_table(d):
    """Bold: the better of PCA and MouldIt per setting."""
    rows = []
    for row in FIDELITY_ROWS:
        if row[1] is None:
            rows.append("\\multicolumn{5}{@{}l}{\\textit{" + row[0] + "}}\\\\")
            continue
        label, col, dec, sign = row
        cells = [label]
        for modality, _ in SETTINGS:
            x = d[(d.modality == modality) & d.method.isin(["PCA", "mould_it"])].groupby("method")[col]
            mean, std = x.mean(), x.std()
            for key in ["PCA", "mould_it"]:
                cells.append(fmt(mean[key], std[key], dec, sign * mean[key] >= sign * mean.drop(key).iloc[0]))
        rows.append(" & ".join(cells) + " \\\\")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=RESULTS_DIR)
    ap.add_argument("--out", default=TABLES_DIR)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    seeds = ResultsAnalyzer(args.dir).per_seed()
    seeds.to_csv(os.path.join(args.out, "per_seed.csv"), index=False)
    largest = seeds[seeds.total == seeds.total.max()]
    for name, rows in (("table_results", results_table(largest)), ("table_fidelity", fidelity_table(largest))):
        open(os.path.join(args.out, f"{name}.tex"), "w").write("\n".join(rows) + "\n")
        print(f"% {name}.tex\n" + "\n".join(rows) + "\n")
