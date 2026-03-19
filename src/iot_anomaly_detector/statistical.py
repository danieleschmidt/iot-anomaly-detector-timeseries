"""
Statistical anomaly detector — z-score, IQR, and moving-average deviation.

No ML required. Fast and interpretable. Works well on steady-state IoT signals.
"""

from __future__ import annotations

from typing import List, Optional
import numpy as np

from .report import AnomalyFinding, AnomalyReport, Severity


def _severity_from_score(score: float, thresholds=(2.5, 4.0, 6.0)) -> Severity:
    """Map normalised score to severity bucket."""
    low, med, high = thresholds
    if score >= high:
        return Severity.CRITICAL
    elif score >= med:
        return Severity.HIGH
    elif score >= low:
        return Severity.MEDIUM
    return Severity.LOW


class StatisticalDetector:
    """
    Detect anomalies in multi-variate IoT time-series using classical stats.

    Three methods are run independently and findings are merged:

    1. **Z-score** — flag values > ``zscore_threshold`` standard deviations
       from the rolling mean (or global mean if window is None).
    2. **IQR** — flag values outside Q1 - k·IQR … Q3 + k·IQR.
    3. **Moving-average deviation** — flag where |x - MA| > ``ma_threshold``
       times the rolling standard deviation.

    Parameters
    ----------
    window : int, optional
        Rolling window size. If None, uses entire history (global stats).
    zscore_threshold : float
        Number of standard deviations for z-score flagging (default 3.0).
    iqr_k : float
        IQR multiplier (default 1.5 — same as Tukey fences).
    ma_threshold : float
        Moving-average deviation threshold in sigma (default 3.0).

    Example
    -------
    >>> det = StatisticalDetector(window=50)
    >>> report = det.detect(data, sensor_names, timestamps)
    """

    def __init__(
        self,
        window: Optional[int] = 50,
        zscore_threshold: float = 3.0,
        iqr_k: float = 1.5,
        ma_threshold: float = 3.0,
    ):
        self.window = window
        self.zscore_threshold = zscore_threshold
        self.iqr_k = iqr_k
        self.ma_threshold = ma_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(
        self,
        data: np.ndarray,
        sensor_names: List[str],
        timestamps: Optional[np.ndarray] = None,
    ) -> AnomalyReport:
        """
        Run all statistical detectors on data.

        Parameters
        ----------
        data : ndarray of shape (n_steps, n_sensors)
        sensor_names : list of str, length n_sensors
        timestamps : optional ndarray of shape (n_steps,)

        Returns
        -------
        AnomalyReport
        """
        n_steps, n_sensors = data.shape
        if timestamps is None:
            timestamps = np.arange(n_steps, dtype=float)

        report = AnomalyReport()

        for col_idx, sensor in enumerate(sensor_names):
            series = data[:, col_idx]
            report = report.merge(self._zscore(series, sensor, timestamps))
            report = report.merge(self._iqr(series, sensor, timestamps))
            report = report.merge(self._ma_deviation(series, sensor, timestamps))

        return report

    # ------------------------------------------------------------------
    # Per-method detectors
    # ------------------------------------------------------------------

    def _zscore(
        self, series: np.ndarray, sensor: str, timestamps: np.ndarray
    ) -> AnomalyReport:
        report = AnomalyReport()
        n = len(series)
        seen: set = set()

        for t in range(n):
            start = 0 if self.window is None else max(0, t - self.window)
            window_data = series[start:t + 1]
            if len(window_data) < 4:
                continue
            mu = np.mean(window_data)
            sigma = np.std(window_data)
            if sigma < 1e-9:
                continue
            z = abs(series[t] - mu) / sigma
            if z >= self.zscore_threshold and t not in seen:
                seen.add(t)
                conf = min(1.0, (z - self.zscore_threshold) / self.zscore_threshold + 0.5)
                report.add(AnomalyFinding(
                    timestep=t,
                    timestamp=float(timestamps[t]),
                    sensor=sensor,
                    severity=_severity_from_score(z),
                    anomaly_type="point",
                    confidence=round(conf, 3),
                    detector="zscore",
                    score=round(float(z), 4),
                    value=round(float(series[t]), 4),
                    description=f"z={z:.2f} > threshold={self.zscore_threshold}",
                ))
        return report

    def _iqr(
        self, series: np.ndarray, sensor: str, timestamps: np.ndarray
    ) -> AnomalyReport:
        report = AnomalyReport()
        n = len(series)
        seen: set = set()

        for t in range(n):
            start = 0 if self.window is None else max(0, t - self.window)
            window_data = series[start:t + 1]
            if len(window_data) < 8:
                continue
            q1, q3 = np.percentile(window_data, [25, 75])
            iqr = q3 - q1
            if iqr < 1e-9:
                continue
            lo, hi = q1 - self.iqr_k * iqr, q3 + self.iqr_k * iqr
            val = series[t]
            if (val < lo or val > hi) and t not in seen:
                seen.add(t)
                deviation = max(abs(val - lo), abs(val - hi)) / iqr
                conf = min(1.0, deviation / 3.0)
                report.add(AnomalyFinding(
                    timestep=t,
                    timestamp=float(timestamps[t]),
                    sensor=sensor,
                    severity=_severity_from_score(deviation),
                    anomaly_type="point",
                    confidence=round(conf, 3),
                    detector="iqr",
                    score=round(float(deviation), 4),
                    value=round(float(val), 4),
                    description=f"IQR fence [{lo:.2f}, {hi:.2f}], val={val:.2f}",
                ))
        return report

    def _ma_deviation(
        self, series: np.ndarray, sensor: str, timestamps: np.ndarray
    ) -> AnomalyReport:
        """Flag points where |x - moving_avg| > threshold * rolling_std."""
        report = AnomalyReport()
        n = len(series)
        win = self.window or n
        seen: set = set()

        for t in range(win, n):
            window_data = series[t - win: t]
            ma = np.mean(window_data)
            std = np.std(window_data)
            if std < 1e-9:
                continue
            dev = abs(series[t] - ma) / std
            if dev >= self.ma_threshold and t not in seen:
                seen.add(t)
                conf = min(1.0, (dev - self.ma_threshold) / self.ma_threshold + 0.5)
                report.add(AnomalyFinding(
                    timestep=t,
                    timestamp=float(timestamps[t]),
                    sensor=sensor,
                    severity=_severity_from_score(dev),
                    anomaly_type="contextual",
                    confidence=round(conf, 3),
                    detector="ma_deviation",
                    score=round(float(dev), 4),
                    value=round(float(series[t]), 4),
                    description=f"MA deviation {dev:.2f}σ (MA={ma:.2f})",
                ))
        return report
