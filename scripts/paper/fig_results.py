"""
Figs. 6-9: results over dataset size, per setting.

    python scripts/paper/fig_results.py [--dir results/erlanger_traeger_benchmark] [--out results/paper/figures]
"""

import argparse
import os

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mouldit_benchmark.paths import FIGURES_DIR, RESULTS_DIR
from mouldit_benchmark.evaluation import ResultsAnalyzer

import figure_style as style

STABLE_STD = 0.05
COLORS = style.METHOD_COLORS
LABELS = {"Reference": "Reference (always OK)", "kNN-raw": "kNN", "LogReg": "Logistic regression",
          "RF": "Random forest", "PCA": "PCA", "RF + PCA": "Random forest with PCA flag",
          "mould_it": "MouldIt (mean ± std)"}
RANK_NAMES = {"RF + PCA": "Random forest with PCA unknown flag", "mould_it": "MouldIt"}
FIGURES = {
    "scaling": dict(
        metrics=[("accuracy", "Balanced accuracy,\nall classes", True, (0.4, 1.03)),
                 ("auroc", "Unknown AUROC", True, (0.3, 1.03)),
                 ("detect", "Unknown detection\nat 5 % FPR", True, (-0.04, 1.03))],
        baselines=["Reference", "kNN-raw", "LogReg", "RF", "PCA"],
        all_label="accuracy, AUROC and detection", ncols=3, figsize=(style.DOUBLE_COLUMN, 2.85),
        top_ncol=6, percent=False, ranking=False),
    "operation": dict(
        metrics=[("good_parts_rejected", "Good parts rejected ↓", False, None),
                 ("known_faults_shipped", "Known faults shipped ↓", False, None),
                 ("unknown_faults_shipped", "Unknown faults shipped ↓", False, None),
                 ("unknown_faults_rejected_as_known", "Unknown faults rejected as a known fault ↓", False, None)],
        baselines=["Reference", "kNN-raw", "LogReg", "RF", "PCA", "RF + PCA"],
        all_label="all four operation metrics", ncols=2, figsize=(style.DOUBLE_COLUMN, 5.5),
        top_ncol=3, percent=True, ranking=True),
}


def onset(sizes, ok):
    """First size from which ok holds at it and every larger size, else None."""
    return next((n for i, n in enumerate(sizes) if all(ok[i:])), None)


def plot_ranking(ax, d, keys, methods, n_last):
    """Bars, best on top: per seed the mean of the four errors at n_last, then mean ± std over seeds."""
    last = d[(d.total == n_last) & d.method.isin(methods)].dropna(subset=keys).copy()
    last["agg"] = last[keys].mean(axis=1)
    g = last.groupby("method")["agg"].agg(["mean", "std"]) * 100
    ref = g.loc["Reference", "mean"] if "Reference" in g.index else None
    g = g.drop("Reference", errors="ignore").sort_values("mean")
    y = np.arange(len(g))[::-1]
    ax.barh(y, g["mean"], xerr=g["std"], color=[COLORS[b] for b in g.index], height=0.65, alpha=0.85,
            error_kw=dict(ecolor=style.TEXT_2, elinewidth=0.7, capsize=1.8), zorder=3)
    for i, (yi, (b, row)) in enumerate(zip(y, g.iterrows()), 1):
        ax.text(row["mean"] + row["std"], yi, f"  {i}. {RANK_NAMES.get(b, LABELS[b])}: "
                f"{row['mean']:.1f} ± {row['std']:.1f} %", va="center", fontsize=7,
                color=COLORS["mould_it"] if b == "mould_it" else style.TEXT_2,
                fontweight="bold" if b == "mould_it" else "normal")
    ax.set_yticks(y, [""] * len(y))
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, (g["mean"] + g["std"]).max() * 1.45)
    ax.set_xlabel("Percentage (%)")
    ax.set_title(f"Sorting losses (% of parts) at {n_last // 1000}k cycles, ranked ↓", pad=4)
    ax.grid(axis="x", color=style.GRID, lw=0.5)
    ax.set_axisbelow(True)
    if ref is not None:
        ax.text(0.99, 0.97, f"Reference (always OK): {ref:.0f} %", transform=ax.transAxes, ha="right",
                va="top", fontsize=7, color=style.TEXT_2)


def fit_ylim(mean, std, others, m):
    """The lowest to the highest value drawn plus a 3 % margin (Reference excluded)."""
    mi, sd = mean.loc["mould_it", m], std.loc["mould_it", m]
    lows = [mean.loc[b, m].min() for b in others] + [(mi - sd).min()]
    highs = [mean.loc[b, m].max() for b in others] + [(mi + sd).max()]
    margin = 0.03 * (max(highs) - min(lows))
    return min(lows) - margin, max(highs) + margin


def plot_modality(seeds, modality, out, name, spec):
    metrics, percent = spec["metrics"], spec["percent"]
    keys = [m for m, *_ in metrics]
    d = seeds[seeds.modality == modality]
    mean = d.groupby(["method", "total"])[keys].mean()
    std = d.groupby(["method", "total"])[keys].std()
    sizes = sorted(d.total.unique())
    present = set(mean.index.get_level_values(0))
    baselines = [b for b in spec["baselines"] if b in present]
    others = [b for b in baselines if b != "Reference"]

    stable, best, rival = {}, {}, {}
    for m, _, hib, _ in metrics:
        mi, sd = mean.loc["mould_it", m], std.loc["mould_it", m]
        comp = pd.concat({b: mean.loc[b, m] for b in others}, axis=1)
        top, rival[m] = (comp.max(axis=1), comp.idxmax(axis=1)) if hib else (comp.min(axis=1), comp.idxmin(axis=1))
        beats = [(mi[n] > top[n]) if hib else (mi[n] < top[n]) for n in sizes]
        is_stable = [sd[n] < STABLE_STD for n in sizes]
        stable[m] = onset(sizes, is_stable)
        best[m] = onset(sizes, [s and b for s, b in zip(is_stable, beats)])
    all_best = onset(sizes, [all(best[m] is not None and n >= best[m] for m in keys) for n in sizes])

    ncols = spec["ncols"]
    nrows = int(np.ceil(len(metrics) / ncols))
    fig = plt.figure(figsize=spec["figsize"], layout="constrained")
    gs = fig.add_gridspec(nrows + spec["ranking"], ncols,
                          height_ratios=[1] * nrows + ([0.85] if spec["ranking"] else []))
    axs = []
    for i in range(nrows * ncols):
        axs.append(fig.add_subplot(gs[i // ncols, i % ncols], sharex=axs[0] if axs else None))
    axs = np.array(axs)
    rank_ax = fig.add_subplot(gs[nrows, :]) if spec["ranking"] else None
    fig.get_layout_engine().set(w_pad=0.03, h_pad=0.03, hspace=0.06, wspace=0.04)

    print(f"\n{name} {modality}: MouldIt mean ± seed std (stable: std < {STABLE_STD}; best: stable and ahead of all)")
    value_labels = []
    for i, (ax, (m, label, hib, ylim)) in enumerate(zip(axs, metrics)):
        mi, sd = mean.loc["mould_it", m], std.loc["mould_it", m]
        if ylim is None:
            ylim = fit_ylim(mean, std, others, m)
        for b in baselines:
            g = mean.loc[b, m]
            ref = b == "Reference"
            ax.plot(g.index, g.values, ls="--" if ref else "-", marker="o", color=COLORS[b], lw=1.0, ms=2.5,
                    label=LABELS[b], zorder=2)
            if ref and ((g < ylim[0]).all() or (g > ylim[1]).all()):
                below = (g < ylim[0]).all()
                text = f"{g.iloc[0] * 100:.0f} %" if percent else f"{g.iloc[0]:.3f}"
                ax.text(0.99, 0.03 if below else 0.97, f"Reference: {text}", transform=ax.transAxes,
                        ha="right", va="bottom" if below else "top", fontsize=7, color=style.TEXT_2, zorder=8,
                        bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.85) if percent else None)
        ax.fill_between(mi.index, mi - sd, mi + sd, color=COLORS["mould_it"], alpha=0.18, lw=0, zorder=3)
        ax.plot(mi.index, mi.values, "s-", color=COLORS["mould_it"], lw=1.5, ms=3, label=LABELS["mould_it"], zorder=5)
        if best[m] is not None:
            ax.plot(best[m], mi[best[m]], "o", ms=10, mfc="none", mec="gold", mew=1.3, zorder=6, clip_on=False)
        if stable[m] is not None:
            ax.scatter(stable[m], mi[stable[m]], s=10 ** 2, facecolors="none", edgecolors="tab:purple",
                       linewidths=1.1, linestyles=(0, (2, 1.5)), zorder=7, clip_on=False)
        if all_best is not None:
            ax.axvspan(all_best, sizes[-1] * 1.25, color="gold", alpha=0.10, lw=0, zorder=0)
        if not percent:
            ax.set_ylabel(label, labelpad=6)
        elif i % ncols == 0:
            ax.set_ylabel("Percentage (%)", labelpad=6)
        ax.set_ylim(*ylim)
        value_labels += y_axis_values(fig, ax, mean, mi, others, m, hib, sizes[-1], sizes[0], percent)
        ax.grid(color=style.GRID, lw=0.5)
        ax.set_axisbelow(True)
        beat = (f" ({mi[best[m]]:.4f} vs {rival[m][best[m]]} {mean.loc[rival[m][best[m]], m][best[m]]:.4f})"
                if best[m] is not None else "")
        print(f"  {m:<19} stable from n={stable[m]}, best from n={best[m]}{beat}:  "
              + "  ".join(f"{n}: {mi[n]:.3f}±{sd[n]:.3f}" for n in sizes))
    print(f"  stable and best in {spec['all_label']} from n={all_best}")
    if rank_ax is not None:
        plot_ranking(rank_ax, d, keys, baselines + ["mould_it"], sizes[-1])

    axs[-1].set_xscale("log")
    axs[-1].set_xticks(sizes)
    axs[-1].minorticks_off()
    axs[-1].set_xlim(sizes[0] / 1.25, sizes[-1] * 1.25)
    labels = [f"{n // 1000}k" if n >= 1000 else str(n) for n in sizes]
    if not percent:   # narrow panels: every second size
        labels = [lab if j % 2 == 0 else "" for j, lab in enumerate(labels)]
    for i, ax in enumerate(axs):
        ax.tick_params(labelbottom=True)
        ax.set_xticklabels(labels)
        if (i // ncols == nrows - 1) and (percent or i % ncols == 1):
            ax.set_xlabel("Dataset size (cycles)")
    for i, ax in enumerate(list(axs[:len(metrics)]) + ([rank_ax] if rank_ax is not None else [])):
        title = f"({chr(97 + i)})"
        if percent and i < len(metrics):
            title += f" {metrics[i][1]}"
        ax.set_title(title, loc="left", pad=4)

    legends(fig, axs[:len(metrics)], baselines, spec, stable, best, all_best, percent)
    fig.align_ylabels(axs[:len(metrics)])
    center_legends(fig, axs[:len(metrics)], value_labels)
    os.makedirs(out, exist_ok=True)
    stem = os.path.join(out, f"erlanger_traeger_{name}_{modality}_ieee_notitle")
    style.save_pdf_png(fig, stem, dpi=600)


def y_axis_values(fig, ax, mean, mi, others, m, hib, n_last, n_first, percent):
    """
    MouldIt's value at the largest size on the y-axis, with a dotted guide, and the best baseline's where it
    is better (at the decimals shown).  Grid labels close to a value are dropped.  Returns the value texts
    that need to be pushed apart after layout.
    """
    v, (lo, hi) = mi[n_last], ax.get_ylim()
    values = [(v, style.METHOD_COLORS["mould_it"])]
    at_last = pd.Series({b: mean.loc[b, m][n_last] for b in others})
    b_best = at_last.idxmax() if hib else at_last.idxmin()
    w = at_last[b_best]
    if (round(w, 3) > round(v, 3)) if hib else (round(w, 3) < round(v, 3)):
        values.append((w, COLORS[b_best]))
    grid = [t for t in ax.get_yticks() if lo <= t <= hi]
    gdec = len(f"{grid[1] - grid[0]:g}".partition(".")[2])
    scale, gdec, vdec = (100, max(gdec - 2, 0), 1) if percent else (1, gdec, 3)
    grid = [t for t in grid if all(abs(t - x) > 0.1 * (hi - lo) for x, _ in values)]
    for x, color in values:
        ax.plot([n_first / 1.25, n_last], [x, x], ":", color=color, lw=0.7, zorder=4)
    texts = []
    if len(values) == 1:
        ax.set_yticks(grid + [v], [f"{t * scale:.{gdec}f}" for t in grid] + [f"{v * scale:.{vdec}f}"])
        ax.get_yticklabels()[-1].set_color(style.METHOD_COLORS["mould_it"])
    else:   # invisible tick labels reserve the space; coloured texts are drawn instead
        ax.set_yticks(grid + [x for x, _ in values],
                      [f"{t * scale:.{gdec}f}" for t in grid] + [f"{x * scale:.{vdec}f}" for x, _ in values])
        for t in ax.get_yticklabels()[len(grid):]:
            t.set_color("none")
        pad = plt.rcParams["ytick.major.size"] + plt.rcParams["ytick.major.pad"]
        base = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
        texts.append([ax.text(0, x, f"{x * scale:.{vdec}f}", ha="right", va="center", fontsize=7, color=color,
                              transform=mtransforms.offset_copy(base, fig=fig, x=-pad, units="points"))
                      for x, color in values])
    ax.set_ylim(lo, hi)
    return texts


def legends(fig, axs, baselines, spec, stable, best, all_best, percent):
    """Methods above the panels, MouldIt's marks below them; only entries drawn somewhere."""
    entries = {}
    for ax in axs:
        entries |= {lbl: hdl for hdl, lbl in zip(*ax.get_legend_handles_labels()) if lbl not in entries}
    labels = [lbl for lbl in [LABELS[b] for b in baselines] + [LABELS["mould_it"]] if lbl in entries]
    handles = [entries[lbl] for lbl in labels]
    i = labels.index(LABELS["mould_it"])
    handles[i] = (mpatches.Patch(color=COLORS["mould_it"], alpha=0.18, lw=0), handles[i])
    stable_mark = axs[0].scatter([], [], s=8 ** 2, facecolors="none", edgecolors="tab:purple", linewidths=1.1,
                                 linestyles=(0, (2, 1.5)),
                                 label="MouldIt is stable from here on (seed std < "
                                       + (f"{STABLE_STD * 100:g} %)" if percent else f"{STABLE_STD})"))
    best_mark = mlines.Line2D([], [], ls="none", marker="o", ms=8, mfc="none", mec="gold", mew=1.3,
                              label="MouldIt is stable and is the best from here on")
    span = mpatches.Patch(color="gold", alpha=0.20, lw=0,
                          label=f"MouldIt is stable and is the best in\n{spec['all_label']}")
    c = spec["top_ncol"]   # matplotlib fills legends column by column; reorder to read row by row
    handles, labels = [x for j in range(c) for x in handles[j::c]], [x for j in range(c) for x in labels[j::c]]
    fig._top_legend = fig.legend(title=None, title_fontproperties=dict(size=9, weight="bold"), handles=handles,
                                 labels=labels, loc="outside upper center", ncol=c, frameon=False, handlelength=1.5,
                                 columnspacing=1.6, handletextpad=0.5, labelspacing=0.5, borderpad=0.9)
    marks = ([stable_mark] if any(v is not None for v in stable.values()) else []) \
        + ([best_mark] if any(v is not None for v in best.values()) else []) + ([span] if all_best else [])
    fig._bottom_legend = fig.legend(handles=marks, labels=[x.get_label() for x in marks], loc="outside lower center",
                                    ncol=2, frameon=False, handlelength=1.8, columnspacing=1.6, labelspacing=0.5,
                                    borderpad=0.9)
    stable_mark.remove()


def center_legends(fig, axs, value_labels):
    """
    After layout: push close y-axis value texts apart, then centre both legends over the plot area
    (y-axis line to right edge) rather than over the whole figure.
    """
    fig.canvas.draw()
    fig.set_layout_engine("none")
    r = fig.canvas.get_renderer()
    for texts in value_labels:
        a, b = sorted(texts, key=lambda t: t.get_window_extent(r).y0)
        overlap = a.get_window_extent(r).y1 - b.get_window_extent(r).y0 + 1.0 * fig.dpi / 72
        if overlap > 0:
            half = overlap / 2 * 72 / fig.dpi
            for t, dy in ((a, -half), (b, half)):
                t.set_transform(mtransforms.offset_copy(t.get_transform(), fig=fig, y=dy, units="points"))
    boxes = [ax.get_window_extent(r) for ax in axs]
    x0, x1 = min(b.x0 for b in boxes), max(b.x1 for b in boxes)
    for leg, loc in ((fig._top_legend, "upper left"), (fig._bottom_legend, "lower left")):
        frame, content = leg.get_window_extent(r), leg._legend_box.get_window_extent(r)
        shift = (x0 + x1) / 2 - (content.x0 + content.x1) / 2
        corner = (frame.x0 + shift, frame.y1 if loc == "upper left" else frame.y0)
        leg.borderaxespad = 0
        leg.set_loc(loc)
        leg.set_bbox_to_anchor(fig.transFigure.inverted().transform(corner), transform=fig.transFigure)
    fig.canvas.draw()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=RESULTS_DIR)
    ap.add_argument("--out", default=FIGURES_DIR)
    args = ap.parse_args()
    style.apply_rc_params(**{"legend.handleheight": 0.8, "lines.linewidth": 1.0, "lines.markersize": 3})
    seeds = ResultsAnalyzer(args.dir).per_seed()
    for name, spec in FIGURES.items():
        for modality in seeds.modality.unique():
            plot_modality(seeds, modality, args.out, name, spec)
