"""A compact, explicit binary Common Spatial Patterns transformer."""

from __future__ import annotations

import numpy as np
from scipy.linalg import eigh
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class BinaryCSP(TransformerMixin, BaseEstimator):
    """Extract log-power CSP features from two-class epoched signals.

    Class covariances are estimated by concatenating every training epoch in a
    class and applying fixed diagonal shrinkage. Spatial filters solve
    C_left w = lambda (C_left + C_right) w.
    """

    def __init__(self, n_components: int = 8, shrinkage: float = 0.1) -> None:
        self.n_components = n_components
        self.shrinkage = shrinkage

    def fit(self, x: np.ndarray, y: np.ndarray) -> "BinaryCSP":
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y)
        if x.ndim != 3:
            raise ValueError("X must have shape (epochs, channels, time)")
        classes = np.unique(y)
        if not np.array_equal(classes, np.array([0, 1])):
            raise ValueError(f"BinaryCSP requires labels 0 and 1, got {classes.tolist()}")
        if not 1 <= self.n_components <= x.shape[1]:
            raise ValueError("n_components must be between 1 and the channel count")
        if not 0 <= self.shrinkage < 1:
            raise ValueError("shrinkage must be in [0, 1)")

        covariances = []
        for label in classes:
            # Treat all time samples from all training epochs as observations.
            samples = x[y == label].transpose(0, 2, 1).reshape(-1, x.shape[1])
            centered = samples - samples.mean(axis=0, keepdims=True)
            empirical = np.einsum("ni,nj->ij", centered, centered, optimize=True)
            empirical /= len(centered)
            mean_variance = np.trace(empirical) / empirical.shape[0]
            regularized = (1 - self.shrinkage) * empirical
            regularized += self.shrinkage * mean_variance * np.eye(empirical.shape[0])
            covariances.append(regularized)
        cov_left, cov_right = covariances
        eigenvalues, eigenvectors = eigh(cov_left, cov_left + cov_right)

        # Values farthest from 0.5 maximize variance for one class and minimize
        # it for the other. eigh returns eigenvectors as columns.
        order = np.argsort(np.abs(eigenvalues - 0.5))[::-1]
        self.filters_ = eigenvectors[:, order[: self.n_components]].T
        self.eigenvalues_ = eigenvalues[order[: self.n_components]]
        self.n_features_in_ = x.shape[1]
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "filters_")
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 3 or x.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X must have shape (epochs, {self.n_features_in_}, time)"
            )
        projected = np.einsum("kc,nct->nkt", self.filters_, x, optimize=True)
        power = np.mean(projected**2, axis=2)
        return np.log(np.maximum(power, np.finfo(np.float64).tiny))
