"""Tests for TimeSeriesGenerator."""

import numpy as np
import pytest
from iot_anomaly_detector import TimeSeriesGenerator
from iot_anomaly_detector.generator import AnomalyType, InjectedAnomaly


class TestTimeSeriesGenerator:
    def setup_method(self):
        self.gen = TimeSeriesGenerator(seed=0)

    def test_output_length(self):
        readings, _ = self.gen.generate(n_steps=100, n_anomalies=5)
        assert len(readings) == 100

    def test_output_columns(self):
        readings, _ = self.gen.generate(n_steps=50, n_anomalies=0)
        assert set(readings[0].values.keys()) == {"temperature", "pressure", "vibration"}

    def test_timesteps_sequential(self):
        readings, _ = self.gen.generate(n_steps=20, n_anomalies=0)
        for i, r in enumerate(readings):
            assert r.timestep == i

    def test_injected_count(self):
        _, injected = self.gen.generate(n_steps=200, n_anomalies=8)
        assert len(injected) == 8

    def test_injected_types_valid(self):
        _, injected = self.gen.generate(n_steps=200, n_anomalies=10)
        for anom in injected:
            assert anom.anomaly_type in list(AnomalyType)

    def test_to_numpy_shape(self):
        readings, _ = self.gen.generate(n_steps=50, n_anomalies=0)
        arr, names = self.gen.to_numpy(readings)
        assert arr.shape == (50, 3)
        assert len(names) == 3

    def test_point_anomaly_creates_spike(self):
        """Point anomalies should create outlier values significantly above normal."""
        gen = TimeSeriesGenerator(seed=42)
        readings_clean, _ = gen.generate(n_steps=300, n_anomalies=0)
        arr_clean, names = gen.to_numpy(readings_clean)
        normal_std = np.std(arr_clean[:, 0])  # temperature

        gen2 = TimeSeriesGenerator(seed=42)
        readings, injected = gen2.generate(n_steps=300, n_anomalies=5)
        arr, _ = gen2.to_numpy(readings)

        point_anoms = [a for a in injected if a.anomaly_type == AnomalyType.POINT]
        if point_anoms:
            anom = point_anoms[0]
            col = names.index(anom.sensor)
            normal_mean = np.mean(arr_clean[:, col])
            # The anomalous value should be far from the normal mean
            deviation = abs(arr[anom.timestep, col] - normal_mean) / (normal_std + 1e-9)
            assert deviation > 3.0, f"Point anomaly deviation too small: {deviation:.2f}σ"

    def test_reproducible_with_seed(self):
        gen1 = TimeSeriesGenerator(seed=7)
        gen2 = TimeSeriesGenerator(seed=7)
        r1, i1 = gen1.generate(n_steps=100, n_anomalies=5)
        r2, i2 = gen2.generate(n_steps=100, n_anomalies=5)
        assert r1[10].values["temperature"] == r2[10].values["temperature"]
        assert [a.timestep for a in i1] == [a.timestep for a in i2]

    def test_anomaly_timesteps_within_bounds(self):
        _, injected = self.gen.generate(n_steps=100, n_anomalies=8)
        for a in injected:
            assert 10 <= a.timestep <= 89

    def test_zero_anomalies(self):
        readings, injected = self.gen.generate(n_steps=50, n_anomalies=0)
        assert len(injected) == 0
        assert len(readings) == 50
