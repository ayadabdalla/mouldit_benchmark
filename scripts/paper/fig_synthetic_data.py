"""
Figs. 3 and 4: synthetic cycles of every class.

    python scripts/paper/fig_synthetic_data.py [--out results/paper/figures] [--n-show 5] [--seed 42]
"""

import argparse
import os

from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import matplotlib.pyplot as plt

from mouldit_benchmark.paths import FIGURES_DIR
from mouldit_benchmark.datasets import CHANNELS, MODALITIES, SyntheticDataset

import figure_style as style

C_REF, C_OK, C_NOK = "#0b0b0b", "#2a78d6", "#eb6834"
FADE = 0.25
HATCH = dict(hatch="////", fill=False, edgecolor="#c3c2b7", lw=0)
X_LABEL = "Time [s]   (each channel min-max normalised)"
CLASS_LABELS = {
    "ok": "OK",
    "timing_shift": "Time shift",
    "pressure_peak_scale": "Underpacking/\noverpacking",
    "fast_gate_freeze": "Fast gate\nfreeze",
    "secondary_pressure_rise": "Secondary\npressure rise",
    "temperature_dynamics": "Temperature\ndynamics",
    "inter_sensor_inconsistency": "Inter-sensor\ninconsistency",
    "distort_us_response": "Distorted\nultrasonic\nresponse",
    "machine_drift": "Machine\ndrift",
}
CHANGED = {   # failure mode -> channels it changes
    "timing_shift": CHANNELS,
    "pressure_peak_scale": ("pressure",),
    "fast_gate_freeze": ("pressure",),
    "secondary_pressure_rise": ("pressure",),
    "temperature_dynamics": ("temperature",),
    "inter_sensor_inconsistency": ("rms_ch0", "rms_ch3"),
    "distort_us_response": ("rms_ch0", "rms_ch3"),
    "machine_drift": CHANNELS,
}
ROW_LABELS = {"pressure": "Pressure\n(P1)", "temperature": "Temp.\n(T3)", "rms_ch0": "US1\nRMS", "rms_ch3": "US4\nRMS"}
SENSOR_NOTES = {
    "pressure_only": "P1: Kistler pressure sensor 1",
    "pt_us": "P1, T3: Kistler pressure sensor 1 and temperature sensor 3;   US1, US4: ultrasound sensors 1 and 4",
}
LAYOUT = {
    "pressure_only": dict(ncols=3, figsize=(style.SINGLE_COLUMN, 2.7), wspace=0.02, hspace=0.04, legend_ncol=1,
                          xticklabels=["0", "20", "40"]),
    "pt_us": dict(figsize=(style.DOUBLE_COLUMN, 3.0), wspace=0.1, hspace=0.02, legend_ncol=4,
                  xticklabels=["0", "", "40"]),
}


def style_panel(ax, t):
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(-0.08, 1.12)
    ax.set_yticks([0, 1])
    ax.grid(axis="y", color=style.GRID, lw=0.5)
    ax.tick_params(axis="x", labelbottom=False)


def plot(out, modality, n_show, seed):
    channels = MODALITIES[modality]
    ds = SyntheticDataset(channels, n_per_class=n_show, seed=seed)
    spec = LAYOUT[modality]
    n_cls = len(ds.classes)
    single = len(channels) == 1
    if single:
        n_cols = spec["ncols"]
        n_rows = -(-n_cls // n_cols)
    else:
        n_rows, n_cols = len(channels), n_cls

    root = plt.figure(figsize=spec["figsize"], layout="constrained")
    root.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=spec["wspace"], hspace=spec["hspace"])
    fig = root.subfigures(1, 1)
    axes = fig.subplots(n_rows, n_cols, sharex=True, sharey=True, squeeze=False)
    panel = (lambda cls, ch: axes.flat[cls]) if single else (lambda cls, ch: axes[ch, cls])

    faded = False
    for cls, name in enumerate(ds.classes):
        color = C_OK if cls == 0 else C_NOK
        for ch, channel in enumerate(channels):
            ax = panel(cls, ch)
            fade = FADE if channel not in CHANGED.get(name, CHANNELS) else 1.0
            faded |= fade < 1.0
            if fade < 1.0:
                ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, zorder=0, **HATCH))
            ax.plot(ds.time, ds.reference[ch], color=C_REF, lw=1.3, alpha=fade)
            for s in ds.series[ds.labels == cls]:
                ax.plot(ds.time, s[ch], color=color, lw=0.6, alpha=0.7 * fade)
            style_panel(ax, ds.time)
            ax.set_xticks([0, 20, 40], spec["xticklabels"])
            if ch == 0 or single:
                letter = f"({chr(97 + cls)})" + (" " if single else "\n")
                ax.set_title(letter + CLASS_LABELS[name], fontsize=7, pad=3)
    for ax in axes[:, 0]:
        ax.set_ylabel("Pressure\n(P1)" if single else "", fontsize=7, color=style.TEXT_2)
    if not single:
        for ch, channel in enumerate(channels):
            axes[ch, 0].set_ylabel(ROW_LABELS[channel], fontsize=7, color=style.TEXT_2, rotation=0, ha="right",
                                   va="center")
    for ax in axes.flat[n_cls if single else axes.size:]:
        ax.set_visible(False)
    lo, hi = ds.series.min(), ds.series.max()
    axes[0, 0].set_ylim(min(-0.08, lo - 0.04), max(1.12, hi + 0.04))
    for ax in axes[-1]:
        ax.tick_params(axis="x", labelbottom=True)

    axes[-1, n_cols // 2].set_xlabel(f"{X_LABEL}\n{SENSOR_NOTES[modality]}", fontsize=7)
    handles = [Line2D([], [], color=C_REF, lw=1.3, label="Real reference"),
               Line2D([], [], color=C_OK, lw=1.0, label=f"Synthetic OK ({n_show} samples per panel)"),
               Line2D([], [], color=C_NOK, lw=1.0, label=f"Synthetic failure mode ({n_show} samples per panel)")]
    if faded:
        handles.append(Patch(label="Unchanged channel", **HATCH))
    fig.legend(handles=handles, loc="outside lower center", ncol=spec["legend_ncol"], frameon=False,
               handlelength=1.5, columnspacing=1.0)
    style.save_pdf_png(root, os.path.join(out, f"erlanger_traeger_synthetic_{modality}_ieee_notitle"),
                       dpi=300, bbox_inches="tight", pad_inches=0.02)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=FIGURES_DIR)
    ap.add_argument("--n-show", type=int, default=5, help="synthetic cycles drawn per class")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    style.apply_rc_params(**{"hatch.linewidth": 0.5})
    os.makedirs(args.out, exist_ok=True)
    for modality in MODALITIES:
        plot(args.out, modality, args.n_show, args.seed)
