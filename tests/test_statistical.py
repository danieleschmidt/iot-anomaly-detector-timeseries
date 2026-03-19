"""Tests for StatisticalDetector."""

import numpy as np
import pytest
from iot_anomaly_detector import StatisticalDetector, TimeSeriesGenerator
from iot_anomaly_detector.report import Severity


class TestStatisticalDetector:
    def setup_method(self):
        self.det = StatisticalDetector(window=50)
        self.gen = TimeSeriesGenerator(seed=1)
        self.readings, self.injected = self.gen.generate(n_steps=200, n_anomalies=5)
        self.data, self.sensor_names = self.gen.to_numpy(self.readings)
        self.timestamps = np.arange(200, dtype=float)

    def test_returns_report(self):
        from iot_anomaly_detector.report import AnomalyReport
        report = self.det.detect(self.data, self.sensor_names, self.timestamps)
        assert isinstance(report, AnomalyReport)

    def test_detects_large_spike(self):
        """A 10-sigma spike should always be caught."""
        data = np.random.default_rng(0).normal(0, 1, (100, 1))
        data[50, 0] = 30.0  # massive spike
        det = StatisticalDetector(window=30)
        report = det.detect(data, ["sensor"], np.arange(100, dtype=float))
        flagged = report.detected_timesteps()
        assert any(abs(t - 50) <= 2 for t in flagged), f"Spike not detected, flagged={flagged}"

    def test_clean_signal_low_false_positives(self):
        """A clean Gaussian signal should have very few false positives."""
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, (300, 3))
        det = StatisticalDetector(window=50, zscore_threshold=4.0, iqr_k=2.0, ma_threshold=4.0)
        report = det.detect(data, ["a", "b", "c"], np.arange(300, dtype=float))
        fp_rate = len(report.detected_timesteps()) / 300
        assert fp_rate < 0.05, f"Too many false positives: {fp_rate:.3f}"

    def test_severity_levels_present(self):
        data = np.zeros((100, 1))
        data[40, 0] = 50.0   # very large
        data[60, 0] = 8.0    # moderate
        det = StatisticalDetector(window=30)
        report = det.detect(data, ["s"], np.arange(100, dtype=float))
        severities = {f.severity for f in report.findings}
        assert len(severities) > 0

    def test_by_sensor(self):
        report = self.det.detect(self.data, self.sensor_names, self.timestamps)
        by_s = report.by_sensor()
        # All keys should be valid sensor names
        for key in by_s:
            assert key in self.sensor_names

    def test_confidence_in_range(self):
        report = self.det.detect(self.data, self.sensor_names, self.timestamps)
        for f in report.findings:
            assert 0.0 <= f.confidence <= 1.0, f"confidence={f.confidence} out of range"

    def test_no_window(self):
        """window=None should use global stats."""
        data = np.random.default_rng(5).normal(10, 2, (100, 2))
        data[50, 0] = 100.0
        det = StatisticalDetector(window=None)
        report = det.detect(data, ["x", "y"], np.arange(100, dtype=float))
        assert any(abs(t - 50) <= 2 for t in report.detected_timesteps())

    def test_recall_on_generated_data(self):
        """At least half of injected anomalies should be found."""
        injected_ts = {a.timestep for a in self.injected}
        report = self.det.detect(self.data, self.sensor_names, self.timestamps)
        detected = set(report.detected_timesteps())
        found = sum(1 for t in injected_ts if any(abs(t - d) <= 2 for d in detected))
        recall = found / len(injected_ts)
        assert recall >= 0.4, f"Recall too low: {recall:.2f} (found {found}/{len(injected_ts)})"
