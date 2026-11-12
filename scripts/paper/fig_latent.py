"""
Fig. 10: the novelty score of MouldIt and PCA on a held-out failure mode.
Needs the saved MouldIt weights of a finished benchmark run (models/*.pt).

    python scripts/paper/fig_latent.py [--dir results/erlanger_traeger_benchmark] [--out results/paper/figures]
                                       [--total 16000] [--seed 42] [--holdout secondary_pressure_rise]
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from mouldit_benchmark.paths import FIGURES_DIR, RESULTS_DIR
from mouldit_benchmark.datasets import MODALITIES, SyntheticDataset, n_classes
from mouldit_benchmark.evaluation import LatentAnalyzer

import figure_style as style

OK, LATENT, MISSED, KNOWN, RECON, BOTH, NOT_LATENT = "#2a78d6", "#CC79A7", "#111111", "#c9c8c2", "#E69F00", "#6A3D9A", "#009E73"
PANELS = [("pressure_only", "mould_it", "MouldIt, single-modal"), ("pressure_only", "PCA", "PCA, single-modal"),
          ("pt_us", "mould_it", "MouldIt, multi-modal"), ("pt_us", "PCA", "PCA, multi-modal")]
LEGEND_ORDER = [["OK", "Known faults", "Detected by latent term", "Not detected by latent term"],
                ["OK", "Known faults", "Detected by reconstruction only", "Detected by latent only", "Detected by both",
                 "Missed"]]
TICKS = ([0.1, 1, 10, 100, 1000], ["0.1", "1", "10", "100", "1k"])


def fold_terms(results_dir, total, seed, holdout):
    """{modality: (y_test, {method: terms})}."""
    analyzer, out = LatentAnalyzer(results_dir), {}
    for modality, channels in MODALITIES.items():
        ds = SyntheticDataset(channels, total // n_classes(channels), seed)
        out[modality] = analyzer.analyze(ds, modality, total, seed, holdout)
    return out


def plot(terms, out):
    rng = np.random.default_rng(0)
    letters = iter("abcdefgh")
    fig = plt.figure(figsize=(style.DOUBLE_COLUMN, 3.95), layout="constrained")
    subs = fig.subfigures(2, 1)
    axs = np.array([s.subplots(1, 4) for s in subs])
    rows = [("d_ok", "d_fault", "Distance to OK", "Distance to nearest fault"),
            ("L", "R", "Latent term $L$", "Reconstruction term $R/\\eta_R$")]
    for row, (xk, yk, xlabel, ylabel) in enumerate(rows):
        for c, (modality, method, title) in enumerate(PANELS):
            ax = axs[row, c]
            y, t = terms[modality][0], terms[modality][1][method]
            values = {"d_ok": t["d"][:, 0], "d_fault": t["d"][:, 1:].min(1), "L": t["L"], "R": t["R"]}
            x, y_ = values[xk], values[yk]
            theta = float(t["threshold"])
            flag_l, flag_r = t["L"] > theta, t["R"] > theta
            flag = flag_l | flag_r
            u = y == -1
            keep = rng.random(len(y)) < 0.5
            ax.scatter(x[keep & (y > 0)], y_[keep & (y > 0)], s=3, c=KNOWN, lw=0, rasterized=True, label="Known faults")
            for part in np.array_split(rng.permutation(np.where(keep & (y <= 0))[0]), 40):   # no group always on top
                if row == 0:
                    groups = [(y[part] == 0, OK, "o", "OK"),
                              (u[part] & flag_l[part], LATENT, "o", "Detected by latent term"),
                              (u[part] & ~flag_l[part], NOT_LATENT, "^", "Not detected by latent term")]
                else:
                    groups = [(y[part] == 0, OK, "o", "OK"),
                              (u[part] & flag_r[part] & ~flag_l[part], RECON, "o", "Detected by reconstruction only"),
                              (u[part] & flag_l[part] & ~flag_r[part], LATENT, "o", "Detected by latent only"),
                              (u[part] & flag_l[part] & flag_r[part], BOTH, "o", "Detected by both"),
                              (u[part] & ~flag[part], MISSED, "^", "Missed")]
                for mask, color, marker, label in groups:
                    ax.scatter(x[part[mask]], y_[part[mask]], s=6, c=color, marker=marker, lw=0, alpha=0.85,
                               rasterized=True, label=label)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(0.02, 3e3)
            ax.set_ylim(0.02, 3e3)
            ax.set_xticks(*TICKS)
            ax.set_yticks(*TICKS)
            ax.axvline(theta, color=style.TEXT_2, ls="--", lw=0.7)
            ax.axhline(theta, color=style.TEXT_2, ls="--", lw=0.7)
            ax.grid(color=style.GRID, lw=0.4)
            ax.set_axisbelow(True)
            ax.set_title(f"({next(letters)}) {title}", loc="left", pad=3, fontsize=7.5)
            ax.set_xlabel(xlabel)
            if c == 0:
                ax.set_ylabel(ylabel)
            else:
                ax.tick_params(labelleft=False)
            if row == 1:
                ax.text(0.97, 0.97, f"{flag[u].mean() * 100:.1f}% detected", transform=ax.transAxes, ha="right",
                        va="top", fontsize=7, color=style.TEXT_2, bbox=dict(fc="white", ec="none", pad=1, alpha=0.8))
    for row, sub in enumerate(subs):
        seen = {}
        for ax in axs[row]:
            for handle, label in zip(*ax.get_legend_handles_labels()):
                seen.setdefault(label, handle)
        seen = {k: seen[k] for k in LEGEND_ORDER[row] if k in seen}
        sub.legend(seen.values(), seen.keys(), loc="outside lower center", ncol=6, frameon=False, markerscale=1.8,
                   handletextpad=0.2, columnspacing=1.0)
    os.makedirs(out, exist_ok=True)
    style.save_pdf_png(fig, os.path.join(out, "erlanger_traeger_latent_ieee_notitle"), dpi=500)


def summary(terms):
    for modality, method, _ in PANELS:
        y, t = terms[modality][0], terms[modality][1][method]
        u, ok = y == -1, y == 0
        flag = np.maximum(t["L"], t["R"]) > t["threshold"]
        print(f"{modality:<14} {method:<9} threshold {t['threshold']:.2f}  detected {flag[u].mean():.3f}  "
              f"unknown inside OK boundary {(t['d'][:, 0][u] <= 1).mean():.3f}  "
              f"OK inside a fault boundary {(t['d'][:, 1:].min(1)[ok] <= 1).mean():.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=RESULTS_DIR)
    ap.add_argument("--out", default=FIGURES_DIR)
    ap.add_argument("--total", type=int, default=16000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--holdout", default="secondary_pressure_rise")
    args = ap.parse_args()
    style.apply_rc_params(**{"xtick.minor.size": 0, "ytick.minor.size": 0})
    terms = fold_terms(args.dir, args.total, args.seed, args.holdout)
    summary(terms)
    plot(terms, args.out)
