"""
Synthetic datasets generated from the reference cycle (paper: Synthetic Data Generation).
"""

import numpy as np
import torch
from torch.utils.data import Dataset

from .failure_modes import FailureModes
from .ok_disturbances import OKDisturbances
from .reference_preprocessing import CHANNELS, DATA_DIR, ReferencePreprocessing


class SyntheticDataset(Dataset):
    """
    n_per_class cycles for OK and for every failure mode the channels support.  OK cycles are the
    reference with OK disturbances; NOK cycles get their failure mode first, then OK disturbances.

    series    (N, C, T) float32
    labels    (N,) int: 0 is OK, k > 0 is failure mode classes[k]
    classes   class names
    time      (T,) time since the window start [s]
    reference (C, T) the real cycle
    """

    def __init__(self, channels=CHANNELS, n_per_class: int = 200, seed: int = 42, data_dir=DATA_DIR):
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.channels = list(channels)
        self.time, self.reference = ReferencePreprocessing(data_dir).load(channels)
        self.failure_modes = FailureModes(channels)
        self.ok_disturbances = OKDisturbances()
        self.classes = ["ok"] + self.failure_modes.modes
        ref = torch.from_numpy(self.reference)

        series = [self.ok_disturbances.sample(ref).numpy() for _ in range(n_per_class)]
        for mode in self.failure_modes.modes:
            for _ in range(n_per_class):
                nok = self.failure_modes.sample(self.reference, mode)
                series.append(self.ok_disturbances.sample(torch.from_numpy(nok.astype(np.float32))).numpy())
        self.series = np.stack(series).astype(np.float32)
        self.labels = np.repeat(np.arange(len(self.classes)), n_per_class)

    def __len__(self):
        return len(self.series)

    def __getitem__(self, idx):
        return {"series": torch.from_numpy(self.series[idx]), "label": int(self.labels[idx])}


def n_classes(channels) -> int:
    return 1 + len(FailureModes(channels).modes)


def sigma_ok(channels=CHANNELS, n: int = 300, seed: int = 42, data_dir=DATA_DIR) -> float:
    """RMS deviation of the first n OK cycles of SyntheticDataset(channels, ..., seed) from the reference."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    _, reference = ReferencePreprocessing(data_dir).load(channels)
    ref = torch.from_numpy(reference)
    ok = OKDisturbances()
    deviation = (torch.stack([ok.sample(ref) for _ in range(n)]) - ref).numpy()
    return float(np.sqrt((deviation ** 2).mean()))


def failure_mode_difficulty(channels=CHANNELS, n_per_class: int = 300, seed: int = 0, data_dir=DATA_DIR):
    """
    {failure mode: RMS deviation of its cycles from the reference / that of OK cycles}.
    About 1 means the fault is as large as normal cycle-to-cycle variation.
    """
    ds = SyntheticDataset(channels, n_per_class, seed, data_dir)
    rms = lambda k: float(np.sqrt(((ds.series[ds.labels == k] - ds.reference[None]) ** 2).mean()))
    ok = rms(0)
    return {name: rms(k) / ok for k, name in enumerate(ds.classes) if k > 0}
