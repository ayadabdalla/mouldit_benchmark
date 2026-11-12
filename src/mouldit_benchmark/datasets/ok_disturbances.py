"""
OK disturbances (Table I): benign cycle-to-cycle variations applied to every synthetic cycle.

Every disturbance acts on a (C, T) tensor and draws its parameters from numpy's global generator.
"""

import numpy as np
import torch


class OKDisturbances:
    """The five OK disturbances of Table I, in the order they are drawn from."""

    def __init__(self):
        self.disturbances = ["gaussian_noise", "lead_lag", "gain", "linear_trend", "moving_average"]

    def sample(self, series: torch.Tensor) -> torch.Tensor:
        """Apply one to three distinct disturbances, chosen at random, with parameters from Table I."""
        out = series.clone()
        for kind in np.random.choice(self.disturbances, size=np.random.randint(1, 4), replace=False):
            if kind == "gaussian_noise":
                out = self.gaussian_noise(out, np.random.uniform(0.002, 0.01))
            elif kind == "lead_lag":
                out = self.lead_lag(out, np.random.randint(1, 3))
            elif kind == "gain":
                out = self.gain(out, 0.98, 1.02)
            elif kind == "linear_trend":
                out = self.linear_trend(out, np.random.uniform(0.00002, 0.0001))
            elif kind == "moving_average":
                out = self.moving_average(out, np.random.randint(2, 4))
        return out

    @staticmethod
    def gaussian_noise(series, std):
        return series + torch.randn_like(series) * std

    @staticmethod
    def lead_lag(series, max_shift):
        """Shift by up to ±max_shift samples, padding with the edge value."""
        shift = np.random.randint(-max_shift, max_shift + 1)
        if shift > 0:
            return torch.cat([series[:, :1].repeat(1, shift), series[:, :-shift]], dim=1)
        if shift < 0:
            return torch.cat([series[:, -shift:], series[:, -1:].repeat(1, -shift)], dim=1)
        return series

    @staticmethod
    def gain(series, low, high):
        return series * np.random.uniform(low, high)

    @staticmethod
    def linear_trend(series, slope):
        """A trend rising by `slope` per sample, of random sign."""
        T = series.shape[1]
        trend = torch.linspace(0, slope * T, T).unsqueeze(0).repeat(series.shape[0], 1)
        return series - trend if np.random.random() > 0.5 else series + trend

    @staticmethod
    def moving_average(series, window):
        """Moving average over `window` samples, padding with the edge value."""
        if window <= 1:
            return series
        pad = window // 2
        padded = torch.cat([series[:, :1].repeat(1, pad), series, series[:, -1:].repeat(1, pad)], dim=1)
        out = torch.zeros_like(series)
        for k in range(series.shape[1]):
            out[:, k] = padded[:, k:k + window].mean(dim=1)
        return out
