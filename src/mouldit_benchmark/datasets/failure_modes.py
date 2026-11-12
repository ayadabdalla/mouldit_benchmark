"""
Process-informed failure modes (paper: Single-Modal Data and Multi-Modal Data).

Every mode maps a normalised (C, T) cycle to a NOK variant.  Outputs are not clipped to [0, 1]
(overpacking exceeds the reference peak).
"""

import numpy as np

from .reference_preprocessing import CHANNELS, ReferencePreprocessing


class FailureModes:
    """
    The failure modes a channel set supports, in class order.  Pressure is required; temperature_dynamics
    needs the temperature, and the two ultrasound modes need both ultrasound channels.
    """

    def __init__(self, channels=CHANNELS):
        idx = {name: i for i, name in enumerate(channels)}
        self._P = idx["pressure"]
        self._T = idx.get("temperature")
        self._US = [idx["rms_ch0"], idx["rms_ch3"]] if "rms_ch0" in idx and "rms_ch3" in idx else None

        self.modes = ["timing_shift", "pressure_peak_scale", "fast_gate_freeze", "secondary_pressure_rise"]
        if self._T is not None:
            self.modes.append("temperature_dynamics")
        if self._US is not None:
            self.modes += ["inter_sensor_inconsistency", "distort_us_response"]
        self.modes.append("machine_drift")

    def sample(self, s: np.ndarray, mode: str) -> np.ndarray:
        """Apply `mode` with parameters drawn from the paper's ranges."""
        if mode == "timing_shift":
            return self.timing_shift(s, np.random.uniform(0.03, 0.07) * np.random.choice([-1, 1]))

        if mode == "pressure_peak_scale":
            scale = np.random.uniform(0.80, 0.92) if np.random.rand() < 0.6 else np.random.uniform(1.08, 1.20)
            return self.pressure_peak_scale(s, scale)

        if mode == "fast_gate_freeze":
            return self.fast_gate_freeze(s, np.random.uniform(1.0, 2.5))

        if mode == "secondary_pressure_rise":
            return self.secondary_pressure_rise(s, amplitude=np.random.uniform(0.03, 0.10),
                                                onset_fraction=np.random.uniform(0.40, 0.65))

        if mode == "temperature_dynamics":
            shift = np.random.uniform(0.02, 0.06) * np.random.choice([-1, 1])
            rate = np.random.choice([np.random.uniform(0.7, 0.9), np.random.uniform(1.1, 1.4)])
            return self.temperature_dynamics(s, shift, rate)

        if mode == "inter_sensor_inconsistency":
            hi, lo = np.random.uniform(1.05, 1.25), np.random.uniform(0.75, 0.95)
            u1, u4 = (hi, lo) if np.random.rand() < 0.5 else (lo, hi)
            return self.inter_sensor_inconsistency(s, u1, u4)

        if mode == "distort_us_response":
            return self.distort_us_response(s, dip_scale=np.random.uniform(0.60, 0.90),
                                            recovery_scale=np.random.uniform(0.60, 0.90))

        if mode == "machine_drift":
            return self.machine_drift(s, np.random.uniform(0.1, 0.35))

        raise ValueError(f"unknown failure mode {mode!r}; available: {self.modes}")

    def timing_shift(self, s, shift_fraction):
        """Warp u -> u + alpha sin(pi u) on every channel; alpha > 0 delays events, alpha < 0 advances them."""
        t = np.linspace(0, 1, s.shape[1])
        t_warped = np.clip(t + shift_fraction * np.sin(np.pi * t), 0, 1)
        out = np.empty_like(s)
        for ch in range(s.shape[0]):
            out[ch] = np.interp(t, t_warped, s[ch])
        return out

    def pressure_peak_scale(self, s, scale):
        """Underpacking (scale < 1) or overpacking (scale > 1) over the filling and holding phases."""
        out = s.copy()
        k_fill, _, k_hold = ReferencePreprocessing.phase_markers(s[self._P])
        out[self._P, k_fill:k_hold] *= scale
        return out

    def fast_gate_freeze(self, s, decay_rate):
        """Exponential pressure decay over the holding phase."""
        out = s.copy()
        _, k_peak, k_hold = ReferencePreprocessing.phase_markers(s[self._P])
        span = max(k_hold - k_peak, 1)
        out[self._P, k_peak:k_hold] *= np.exp(-decay_rate * np.arange(k_hold - k_peak) / span)
        return out

    def secondary_pressure_rise(self, s, amplitude, onset_fraction):
        """A Gaussian pressure bump after the onset sample floor(onset_fraction T)."""
        out = s.copy()
        onset = int(onset_fraction * s.shape[1])
        span = s.shape[1] - onset
        if span < 2:
            return out
        k = np.arange(span)
        out[self._P, onset:] += amplitude * np.exp(-0.5 * ((k - span * 0.3) / max(span * 0.15, 1)) ** 2)
        return out

    def temperature_dynamics(self, s, baseline_shift, cooling_rate_factor):
        """Shift the temperature baseline, then warp the cooling segment after the temperature peak."""
        out = s.copy()
        temp = s[self._T] + baseline_shift
        k_max = int(np.argmax(temp))
        if k_max < len(temp) - 1 and cooling_rate_factor != 1.0:
            u = np.linspace(0, 1, len(temp) - k_max)
            temp[k_max:] = np.interp(np.power(u, 1.0 / cooling_rate_factor), u, temp[k_max:])
        out[self._T] = temp
        return out

    def inter_sensor_inconsistency(self, s, us1_scale, us4_scale):
        """Rescale each ultrasound channel around its pre-fill level by its own factor."""
        out = s.copy()
        for ch, scale in zip(self._US, (us1_scale, us4_scale)):
            out[ch] = s[ch, 0] + (s[ch] - s[ch, 0]) * scale
        return out

    def distort_us_response(self, s, dip_scale, recovery_scale):
        """Damp each ultrasound channel's drop from the pressure peak to its minimum, and its recovery after it."""
        out = s.copy()
        _, k_peak, _ = ReferencePreprocessing.phase_markers(s[self._P])
        for ch in self._US:
            k_min = int(np.argmin(s[ch]))
            pre_fill, minimum = s[ch, 0], s[ch, k_min]
            out[ch, k_peak:k_min + 1] = pre_fill + (s[ch, k_peak:k_min + 1] - pre_fill) * dip_scale
            out[ch, k_min:] = minimum + (s[ch, k_min:] - minimum) * recovery_scale
        return out

    def machine_drift(self, s, drift_fraction):
        """Lower the holding pressure, raise the temperature baseline and add broadband noise on every channel."""
        out = s.copy()
        _, k_peak, k_hold = ReferencePreprocessing.phase_markers(s[self._P])
        out[self._P, k_peak:k_hold] *= 1.0 - 0.4 * drift_fraction
        if self._T is not None:
            out[self._T] += 0.15 * drift_fraction
        out += np.random.randn(*s.shape) * (0.03 * drift_fraction)
        return out
