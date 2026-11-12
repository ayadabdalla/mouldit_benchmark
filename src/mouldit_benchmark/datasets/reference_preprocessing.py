"""
Preprocessing of the real reference cycle (paper: Temporal Alignment and Phase Extraction).
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[3] / "data"
PT_FILE = "cycle_001_pt.csv"        # cavity pressure P1 and temperature T3, irregular timestamps
US_FILE = "cycle_001_us_rms.csv"    # ultrasound RMS of US1 (rms_ch0) and US4 (rms_ch3), regular timestamps

CHANNELS = ("pressure", "temperature", "rms_ch0", "rms_ch3")
MODALITIES = {"pressure_only": ("pressure",), "pt_us": CHANNELS}   # single-modal, multi-modal


class ReferencePreprocessing:
    """
    load() reads the two recordings, aligns them on one time grid and min-max normalises each channel.
    phase_markers() and channel_markers() locate the points the failure modes act on.
    """

    PHASE_THRESHOLD = 0.1

    def __init__(self, data_dir=DATA_DIR):
        self.data_dir = Path(data_dir)

    def load(self, channels=CHANNELS):
        """
        The preprocessed reference cycle: t (T,), seconds since the window start, and x (C, T) float32.
        The grid does not depend on `channels`, so both settings share T.
        """
        pt, us = self.read_recordings()
        t, signals = self.align(pt, us, channels)
        return t - t[0], np.stack([self.normalise(s) for s in signals]).astype(np.float32)

    def read_recordings(self):
        """The pressure/temperature and ultrasound recordings as DataFrames with a `time` column."""
        paths = [self.data_dir / name for name in (PT_FILE, US_FILE)]
        if not all(p.exists() for p in paths):
            raise FileNotFoundError(f"reference cycle not found in {self.data_dir}; "
                                    "run: python -m mouldit_benchmark.datasets.download_reference_cycle")
        return tuple(pd.read_csv(p) for p in paths)

    @staticmethod
    def align(pt, us, channels=CHANNELS):
        """
        Linear interpolation of each channel onto a uniform grid over the window every sensor covers, with
        as many points as pressure/temperature samples inside it.  Returns the grid and one signal per channel.
        """
        t_start = max(us["time"].min(), pt["time"].min())
        t_end = min(us["time"].max(), pt["time"].max())
        n_points = int(((pt["time"] >= t_start) & (pt["time"] <= t_end)).sum())
        t = np.linspace(t_start, t_end, n_points)
        signals = []
        for name in channels:
            source = pt if name in pt.columns else us
            signals.append(np.interp(t, source["time"].values, source[name].values))
        return t, signals

    @staticmethod
    def normalise(signal):
        """Min-max normalisation to [0, 1]."""
        lo, hi = signal.min(), signal.max()
        return (signal - lo) / (hi - lo + 1e-8)

    @classmethod
    def phase_markers(cls, pressure):
        """(k_fill, k_peak, k_hold): fill start, pressure peak and holding end of a normalised pressure curve."""
        k_peak = int(np.argmax(pressure))
        k_fill = next((k for k in range(k_peak) if pressure[k] > cls.PHASE_THRESHOLD), 0)
        k_hold = next((k for k in range(k_peak, len(pressure)) if pressure[k] < cls.PHASE_THRESHOLD),
                      len(pressure) - 1)
        return k_fill, k_peak, k_hold

    @staticmethod
    def channel_markers(x, channels=CHANNELS):
        """The temperature peak and the ultrasound minima of a cycle x (C, T), for the channels it has."""
        idx = {name: i for i, name in enumerate(channels)}
        markers = {}
        if "temperature" in idx:
            markers["k_T_max"] = int(np.argmax(x[idx["temperature"]]))
        for name, label in (("rms_ch0", "k_US1_min"), ("rms_ch3", "k_US4_min")):
            if name in idx:
                markers[label] = int(np.argmin(x[idx[name]]))
        return markers
