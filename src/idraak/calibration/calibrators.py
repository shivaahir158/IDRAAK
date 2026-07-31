"""Post-hoc calibration methods."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from idraak.utils.logging import get_logger

logger = get_logger("calibration")


class TemperatureScaling:
    """Temperature scaling calibration."""

    def __init__(self):
        self.temperature: float = 1.0

    def fit(self, y_prob: np.ndarray, y_true: np.ndarray) -> None:
        """Find optimal temperature on validation data."""
        def nll(t: float) -> float:
            scaled = np.clip(y_prob / t, 1e-10, 1 - 1e-10)
            return float(-np.mean(y_true * np.log(scaled) + (1 - y_true) * np.log(1 - scaled)))

        result = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
        self.temperature = result.x
        logger.info(f"Fitted temperature: {self.temperature:.4f}")

    def calibrate(self, y_prob: np.ndarray) -> np.ndarray:
        return np.clip(y_prob / self.temperature, 0, 1)


class PlattScaling:
    """Platt scaling (logistic regression on logits)."""

    def __init__(self):
        self._model = LogisticRegression()

    def fit(self, y_prob: np.ndarray, y_true: np.ndarray) -> None:
        logits = np.log(np.clip(y_prob, 1e-10, 1 - 1e-10) / (1 - np.clip(y_prob, 1e-10, 1 - 1e-10)))
        self._model.fit(logits.reshape(-1, 1), y_true)

    def calibrate(self, y_prob: np.ndarray) -> np.ndarray:
        logits = np.log(np.clip(y_prob, 1e-10, 1 - 1e-10) / (1 - np.clip(y_prob, 1e-10, 1 - 1e-10)))
        return self._model.predict_proba(logits.reshape(-1, 1))[:, 1]


class IsotonicCalibrator:
    """Isotonic regression calibration."""

    def __init__(self):
        self._model = IsotonicRegression(out_of_bounds="clip")

    def fit(self, y_prob: np.ndarray, y_true: np.ndarray) -> None:
        self._model.fit(y_prob, y_true)

    def calibrate(self, y_prob: np.ndarray) -> np.ndarray:
        return self._model.predict(y_prob)


class HistogramBinning:
    """Histogram binning calibration."""

    def __init__(self, n_bins: int = 15):
        self.n_bins = n_bins
        self._bin_boundaries: np.ndarray = np.array([])
        self._bin_values: np.ndarray = np.array([])

    def fit(self, y_prob: np.ndarray, y_true: np.ndarray) -> None:
        self._bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        self._bin_values = np.zeros(self.n_bins)

        for i in range(self.n_bins):
            mask = (y_prob >= self._bin_boundaries[i]) & (y_prob < self._bin_boundaries[i + 1])
            if i == self.n_bins - 1:
                mask = mask | (y_prob == self._bin_boundaries[i + 1])
            if mask.sum() > 0:
                self._bin_values[i] = y_true[mask].mean()
            else:
                self._bin_values[i] = (self._bin_boundaries[i] + self._bin_boundaries[i + 1]) / 2

    def calibrate(self, y_prob: np.ndarray) -> np.ndarray:
        calibrated = np.zeros_like(y_prob)
        for i in range(self.n_bins):
            mask = (y_prob >= self._bin_boundaries[i]) & (y_prob < self._bin_boundaries[i + 1])
            if i == self.n_bins - 1:
                mask = mask | (y_prob == self._bin_boundaries[i + 1])
            calibrated[mask] = self._bin_values[i]
        return calibrated
