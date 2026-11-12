"""
Fig. 2: the real reference cycle; prints the phase marker table.

    python scripts/paper/fig_reference_cycle.py [--out results/paper/figures]
"""

import argparse
import os

from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.pyplot as plt

from mouldit_benchmark.paths import FIGURES_DIR
from mouldit_benchmark.datasets import CHANNELS, ReferencePreprocessing

import figure_style as style

PHASE_SHADES = ("#c3c2b7", "#ebeae4")   # filling, holding
Y_LABELS = {"pressure": "P1 pressure\n[norm.]", "temperature": "T3 temp.\n[norm.]",
            "rms_ch0": "US1 RMS\n[norm.]", "rms_ch3": "US4 RMS\n[norm.]"}
MARKER_STYLE = dict(color=style.TEXT_2, ls=(0, (1, 1.5)))


def style_panel(ax, t):
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(-0.08, 1.12)
    ax.set_yticks([0, 1])
    ax.grid(axis="y", color=style.GRID, lw=0.5)
    ax.tick_params(axis="x", labelbottom=False)


def print_markers(t, x):
    markers = (dict(zip(("k_fill", "k_peak", "k_hold"), ReferencePreprocessing.phase_markers(x[0])))
               | ReferencePreprocessing.channel_markers(x))
    print(f"T = {len(t)}, window {t[-1]:.1f} s, dt = {t[1]:.3f} s")
    for name, k in markers.items():
        print(f"  {name:<10} index {k:>4}  time {t[k]:.1f} s")


def main(out):
    style.apply_rc_params(**{"hatch.linewidth": 0.5})
    t, x = ReferencePreprocessing().load()
    print_markers(t, x)
    markers = ReferencePreprocessing.phase_markers(x[0])
    k_fill, k_peak, k_hold = markers
    spans = [(t[k_fill], t[k_peak]), (t[k_peak], t[k_hold])]
    ticks = sorted([(v, f"{v:g}", "center", "black") for v in (10, 20, 30, 40)]
                   + [(t[k], f"{t[k]:.1f}", ha, style.TEXT_2) for k, ha in zip(markers, ("right", "left", "center"))])

    fig, axes = plt.subplots(4, 1, figsize=(style.SINGLE_COLUMN, 3.8), sharex=True, gridspec_kw=dict(hspace=0.34))
    for i, (ax, name, series) in enumerate(zip(axes, CHANNELS, x)):
        ax.set_title(f"({chr(97 + i)})", loc="left", pad=2, fontsize=7)
        for (a, b), shade in zip(spans, PHASE_SHADES):
            ax.axvspan(a, b, color=shade, lw=0, zorder=0)
        for k in markers:
            ax.axvline(t[k], lw=0.5, zorder=1, **MARKER_STYLE)
        ax.plot(t, series, color=style.CHANNEL_COLORS[name], lw=1.1, zorder=2, solid_capstyle="round")
        style_panel(ax, t)
        ax.set_ylabel(Y_LABELS[name], linespacing=1.1)
    ax = axes[-1]
    ax.tick_params(axis="x", labelbottom=True)
    ax.set_xticks([v for v, *_ in ticks], [label for _, label, *_ in ticks])
    for tick, (_, _, ha, color) in zip(ax.get_xticklabels(), ticks):
        tick.set(ha=ha, color=color)
    ax.set_xlabel("Time [s]", labelpad=1)
    handles = [Patch(color=PHASE_SHADES[0], label=r"Filling phase $\mathcal{F}$"),
               Patch(color=PHASE_SHADES[1], label=r"Holding phase $\mathcal{H}$"),
               Line2D([], [], lw=0.8, label="Phase markers", **MARKER_STYLE)]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, handlelength=1.4, handleheight=0.9,
               columnspacing=1.0, bbox_to_anchor=(0.53, -0.03))
    fig.align_ylabels(axes)
    os.makedirs(out, exist_ok=True)
    style.save_pdf_png(fig, os.path.join(out, "erlanger_traeger_modalities_ieee_notitle"),
                       dpi=300, bbox_inches="tight", pad_inches=0.02)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=FIGURES_DIR)
    main(ap.parse_args().out)
