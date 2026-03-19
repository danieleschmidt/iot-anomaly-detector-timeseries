"""Synthetic IoT sensor data generator with configurable anomaly injection."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum


class AnomalyType(str, Enum):
    """Types of anomalies that can be injected into sensor streams."""
    POINT = "point"           # Single outlier spike
    CONTEXTUAL = "contextual" # Value normal globally but abnormal for context
    COLLECTIVE = "collective" # Subsequence that is collectively anomalous


@dataclass
class SensorConfig:
    """Configuration for a single IoT sensor."""
    name: str
    baseline: float
    amplitude: float       # Typical oscillation amplitude
    noise_std: float       # Gaussian noise standard deviation
    trend: float = 0.0     # Linear drift per timestep
    period: float = 50.0   # Seasonal period in timesteps


@dataclass
class InjectedAnomaly:
    """Record of an anomaly that was injected into the stream."""
    timestep: int
    sensor: str
    anomaly_type: AnomalyType
    magnitude: float       # Actual deviation injected


@dataclass
class SensorReading:
    """A single sensor reading at one timestep."""
    timestep: int
    timestamp: float       # Unix-style seconds (t=0 at start)
    values: Dict[str, float] = field(default_factory=dict)


class TimeSeriesGenerator:
    """
    Generate realistic synthetic IoT sensor time-series with injected anomalies.

    Supports three sensor types: temperature, pressure, vibration.
    Anomaly types:
      - POINT: sudden spike/dip far outside normal range
      - CONTEXTUAL: value plausible overall but wrong for current phase
      - COLLECTIVE: a sequence of correlated deviations (pattern shift)

    Example
    -------
    >>> gen = TimeSeriesGenerator(seed=42)
    >>> readings, anomalies = gen.generate(n_steps=500, n_anomalies=10)
    >>> len(readings)
    500
    """

    DEFAULT_SENSORS: List[SensorConfig] = [
        SensorConfig("temperature", baseline=25.0, amplitude=3.0, noise_std=0.5, period=48.0),
        SensorConfig("pressure",    baseline=101.3, amplitude=1.5, noise_std=0.3, period=72.0),
        SensorConfig("vibration",   baseline=0.0,  amplitude=2.0, noise_std=0.8, period=24.0),
    ]

    def __init__(
        self,
        sensors: Optional[List[SensorConfig]] = None,
        seed: Optional[int] = None,
        sampling_interval: float = 60.0,  # seconds between readings
    ):
        self.sensors = sensors or self.DEFAULT_SENSORS
        self.rng = np.random.default_rng(seed)
        self.sampling_interval = sampling_interval

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        n_steps: int = 500,
        n_anomalies: int = 10,
    ) -> Tuple[List[SensorReading], List[InjectedAnomaly]]:
        """
        Generate n_steps of sensor readings with n_anomalies injected.

        Returns
        -------
        readings : list of SensorReading
        injected : list of InjectedAnomaly (ground truth)
        """
        # Build clean signals
        signals: Dict[str, np.ndarray] = {}
        for cfg in self.sensors:
            signals[cfg.name] = self._clean_signal(cfg, n_steps)

        # Choose anomaly locations & types (spread across sensors)
        injected = self._plan_anomalies(n_steps, n_anomalies)

        # Apply anomalies
        for anom in injected:
            cfg = self._sensor_cfg(anom.sensor)
            signals[anom.sensor] = self._apply_anomaly(
                signals[anom.sensor], anom, cfg
            )

        # Package into SensorReading list
        readings: List[SensorReading] = []
        for t in range(n_steps):
            readings.append(SensorReading(
                timestep=t,
                timestamp=float(t * self.sampling_interval),
                values={name: float(arr[t]) for name, arr in signals.items()},
            ))

        return readings, injected

    def to_numpy(
        self,
        readings: List[SensorReading],
        sensor_names: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Convert readings to a (n_steps, n_sensors) numpy array.

        Returns array and the ordered list of sensor names.
        """
        names = sensor_names or [cfg.name for cfg in self.sensors]
        arr = np.array([[r.values[n] for n in names] for r in readings])
        return arr, names

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _clean_signal(self, cfg: SensorConfig, n_steps: int) -> np.ndarray:
        """Produce a realistic sensor signal: trend + seasonality + noise."""
        t = np.arange(n_steps, dtype=float)
        seasonal = cfg.amplitude * np.sin(2 * np.pi * t / cfg.period)
        trend = cfg.trend * t
        noise = self.rng.normal(0, cfg.noise_std, n_steps)
        return cfg.baseline + seasonal + trend + noise

    def _plan_anomalies(self, n_steps: int, n_anomalies: int) -> List[InjectedAnomaly]:
        """Choose anomaly timesteps, sensors, and types."""
        # Avoid the first 10 steps (warm-up)
        eligible = np.arange(10, n_steps - 10)
        # Keep anomalies at least 5 steps apart
        chosen: List[int] = []
        attempts = 0
        while len(chosen) < n_anomalies and attempts < 10000:
            t = int(self.rng.choice(eligible))
            if all(abs(t - c) >= 5 for c in chosen):
                chosen.append(t)
            attempts += 1

        sensor_names = [cfg.name for cfg in self.sensors]
        anomaly_types = list(AnomalyType)
        injected: List[InjectedAnomaly] = []

        for t in sorted(chosen):
            atype = anomaly_types[int(self.rng.integers(len(anomaly_types)))]
            sensor = sensor_names[len(injected) % len(sensor_names)]
            cfg = self._sensor_cfg(sensor)
            magnitude = self._anomaly_magnitude(atype, cfg)
            injected.append(InjectedAnomaly(
                timestep=t,
                sensor=sensor,
                anomaly_type=atype,
                magnitude=magnitude,
            ))

        return injected

    def _anomaly_magnitude(self, atype: AnomalyType, cfg: SensorConfig) -> float:
        """Return signed deviation magnitude appropriate for anomaly type."""
        if atype == AnomalyType.POINT:
            # Spike 4–8x the total signal amplitude (very obvious)
            scale = self.rng.uniform(4, 8) * (cfg.amplitude + cfg.noise_std)
            return float(self.rng.integers(2) * 2 - 1) * scale  # random sign
        elif atype == AnomalyType.CONTEXTUAL:
            # ~3 sigma, wrong sign relative to local seasonal phase
            return float(self.rng.uniform(2.5, 4.0) * cfg.amplitude)
        else:  # COLLECTIVE — moderate sustained shift, magnitude per step
            return float(self.rng.uniform(3, 5) * cfg.noise_std)

    def _apply_anomaly(
        self,
        signal: np.ndarray,
        anom: InjectedAnomaly,
        cfg: SensorConfig,
    ) -> np.ndarray:
        sig = signal.copy()
        t = anom.timestep

        if anom.anomaly_type == AnomalyType.POINT:
            sig[t] += anom.magnitude

        elif anom.anomaly_type == AnomalyType.CONTEXTUAL:
            # Invert the seasonal component (wrong phase)
            seasonal_val = cfg.amplitude * np.sin(2 * np.pi * t / cfg.period)
            sig[t] -= 2 * seasonal_val + anom.magnitude

        else:  # COLLECTIVE
            # Shift 8 consecutive steps
            end = min(t + 8, len(sig))
            sig[t:end] += anom.magnitude

        return sig

    def _sensor_cfg(self, name: str) -> SensorConfig:
        for cfg in self.sensors:
            if cfg.name == name:
                return cfg
        raise ValueError(f"Unknown sensor: {name}")
