"""
Style shared by the fig_*.py scripts: IEEE figure widths (3.5 in single column, 7.16 in double column),
type scale, palette, and saving every figure as PDF and PNG.
"""

import matplotlib.pyplot as plt

SINGLE_COLUMN, DOUBLE_COLUMN = 3.5, 7.16

TEXT_2 = "#52514e"
GRID = "#e5e4df"
METHOD_COLORS = {"Reference": "#9b9a95", "kNN-raw": "#2a78d6", "LogReg": "#eb6834", "RF": "#1baf7a",
                 "PCA": "#e87ba4", "RF + PCA": "#008300", "mould_it": "#4a3aa7"}
CHANNEL_COLORS = {"pressure": "#2a78d6", "temperature": "#eb6834", "rms_ch0": "#1baf7a", "rms_ch3": "#eda100"}

BASE_RC = {   # axis labels 7 pt, everything else at least 7 pt; TrueType fonts for IEEE PDF eXpress
    "font.size": 8, "axes.labelsize": 7, "axes.labelcolor": TEXT_2, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.labelpad": 2.5, "xtick.major.pad": 2.0, "ytick.major.pad": 2.0,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42,
}


def apply_rc_params(**extra):
    """BASE_RC plus a figure's own overrides, for every figure drawn after the call."""
    plt.rcParams.update({**BASE_RC, **extra})


def save_pdf_png(fig, stem, **kwargs):
    """stem.pdf and stem.png, then close the figure."""
    for ext in ("pdf", "png"):
        fig.savefig(f"{stem}.{ext}", **kwargs)
    plt.close(fig)
    print(f"saved {stem}.{{pdf,png}}")
