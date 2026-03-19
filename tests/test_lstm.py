"""Tests for LSTMDetector."""

import numpy as np
import pytest
from iot_anomaly_detector import LSTMDetector, TimeSeriesGenerator
from iot_anomaly_detector.report import AnomalyReport


class TestLSTMDetector:
    """Fast tests — use tiny models and short training."""

    def setup_method(self):
        self.det = LSTMDetector(
            window=10, hidden_size=16, epochs=5, threshold_sigma=2.0, device="cpu"
        )
        # Simple clean training data
        rng = np.random.default_rng(0)
        self.normal_data = rng.normal(0, 1, (150, 2)).astype(np.float32)

    def test_fit_returns_self(self):
        result = self.det.fit(self.normal_data, verbose=False)
        assert result is self.det

    def test_threshold_set_after_fit(self):
        self.det.fit(self.normal_data, verbose=False)
        assert self.det.threshold_ is not None
        assert self.det.threshold_ > 0

    def test_detect_returns_report(self):
        self.det.fit(self.normal_data, verbose=False)
        report = self.det.detect(self.normal_data, ["a", "b"])
        assert isinstance(report, AnomalyReport)

    def test_detect_before_fit_raises(self):
        det = LSTMDetector(window=10)
        with pytest.raises(RuntimeError, match="fit"):
            det.detect(self.normal_data, ["a", "b"])

    def test_detects_obvious_anomaly(self):
        """A massive spike injected after training should be detected."""
        self.det.fit(self.normal_data, verbose=False)
        test_data = self.normal_data.copy()
        test_data[80, 0] = 100.0   # massive spike
        test_data[81, 0] = 100.0
        report = self.det.detect(test_data, ["a", "b"])
        flagged = report.detected_timesteps()
        assert any(abs(t - 80) <= 5 for t in flagged), \
            f"Obvious spike not detected. Flagged: {flagged}"

    def test_reconstruction_error_series_shape(self):
        self.det.fit(self.normal_data, verbose=False)
        errors = self.det.reconstruction_error_series(self.normal_data)
        assert errors.shape == (len(self.normal_data),)

    def test_confidence_in_range(self):
        self.det.fit(self.normal_data, verbose=False)
        test_data = self.normal_data.copy()
        test_data[60, :] = 50.0
        report = self.det.detect(test_data, ["a", "b"])
        for f in report.findings:
            assert 0.0 <= f.confidence <= 1.0

    def test_error_higher_on_anomaly(self):
        """Reconstruction error should be higher at injected spike."""
        self.det.fit(self.normal_data, verbose=False)
        test_data = self.normal_data.copy()
        test_data[100, 0] = 200.0
        errors = self.det.reconstruction_error_series(test_data)
        avg_normal = np.mean(errors[:80])
        spike_error = errors[100]
        assert spike_error > avg_normal * 2, \
            f"Spike error ({spike_error:.4f}) not > 2x normal ({avg_normal:.4f})"

    def test_end_to_end_with_generator(self):
        """Integration: fit on normal half, detect on full stream."""
        gen = TimeSeriesGenerator(seed=10)
        readings, injected = gen.generate(n_steps=200, n_anomalies=5)
        data, names = gen.to_numpy(readings)
        data = data.astype(np.float32)

        det = LSTMDetector(window=10, hidden_size=16, epochs=5, device="cpu")
        det.fit(data[:100], verbose=False)
        report = det.detect(data, names)
        # Should produce some findings without crashing
        assert isinstance(report, AnomalyReport)
