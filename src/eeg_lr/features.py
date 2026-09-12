"""Transparent spectral feature extractors for EEG baselines."""

from __future__ import annotations

import numpy as np
from scipy.signal import welch
from sklearn.base import BaseEstimator, TransformerMixin


class BandPowerFeatures(TransformerMixin, BaseEstimator):
    """Return log band power for every epoch, channel, and frequency band."""

    def __init__(
        self,
        sfreq: float = 160.0,
        bands: tuple[tuple[float, float], ...] = ((8.0, 13.0), (13.0, 30.0)),
    ) -> None:
        self.sfreq = sfreq
        self.bands = bands

    def fit(self, x: np.ndarray, y: np.ndarray | None = None) -> "BandPowerFeatures":
        x = np.asarray(x)
        if x.ndim != 3:
            raise ValueError("X must have shape (epochs, channels, time)")
        if self.sfreq <= 0:
            raise ValueError("sfreq must be positive")
        nyquist = self.sfreq / 2
        for low, high in self.bands:
            if not 0 <= low < high <= nyquist:
                raise ValueError(f"Invalid frequency band {(low, high)}")
        self.n_features_in_ = x.shape[1]
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 3 or x.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X must have shape (epochs, {self.n_features_in_}, time)"
            )
        frequencies, psd = welch(
            x,
            fs=self.sfreq,
            nperseg=min(256, x.shape[-1]),
            axis=-1,
            detrend="constant",
        )
        features = []
        for low, high in self.bands:
            mask = (frequencies >= low) & (frequencies <= high)
            power = np.trapezoid(psd[..., mask], frequencies[mask], axis=-1)
            features.append(np.log(np.maximum(power, np.finfo(np.float64).tiny)))
        return np.concatenate(features, axis=1)
