"""Tests for calibration methods."""

import numpy as np
import pytest
from idraak.calibration.calibrators import (
    HistogramBinning,
    IsotonicCalibrator,
    PlattScaling,
    TemperatureScaling,
)


@pytest.fixture
def calibration_data():
    np.random.seed(42)
    n = 200
    y_true = np.random.binomial(1, 0.5, n)
    # Overconfident predictions
    y_prob = np.clip(y_true + np.random.normal(0, 0.3, n), 0.01, 0.99)
    return y_prob, y_true


class TestTemperatureScaling:
    def test_fit_and_calibrate(self, calibration_data):
        y_prob, y_true = calibration_data
        ts = TemperatureScaling()
        ts.fit(y_prob, y_true)
        calibrated = ts.calibrate(y_prob)
        assert calibrated.shape == y_prob.shape
        assert np.all(calibrated >= 0)
        assert np.all(calibrated <= 1)


class TestPlattScaling:
    def test_fit_and_calibrate(self, calibration_data):
        y_prob, y_true = calibration_data
        ps = PlattScaling()
        ps.fit(y_prob, y_true)
        calibrated = ps.calibrate(y_prob)
        assert calibrated.shape == y_prob.shape


class TestIsotonicCalibrator:
    def test_fit_and_calibrate(self, calibration_data):
        y_prob, y_true = calibration_data
        ic = IsotonicCalibrator()
        ic.fit(y_prob, y_true)
        calibrated = ic.calibrate(y_prob)
        assert calibrated.shape == y_prob.shape


class TestHistogramBinning:
    def test_fit_and_calibrate(self, calibration_data):
        y_prob, y_true = calibration_data
        hb = HistogramBinning(n_bins=10)
        hb.fit(y_prob, y_true)
        calibrated = hb.calibrate(y_prob)
        assert calibrated.shape == y_prob.shape
