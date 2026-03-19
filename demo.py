#!/usr/bin/env python3
"""
IoT Anomaly Detection Demo
==========================
Generates 500 timesteps of synthetic IoT sensor data (temperature, pressure,
vibration) with 10 injected anomalies, then compares StatisticalDetector vs
LSTMDetector accuracy.

Usage
-----
    ~/anaconda3/bin/python3 demo.py
"""

import numpy as np
from iot_anomaly_detector import (
    TimeSeriesGenerator,
    StatisticalDetector,
    LSTMDetector,
    AnomalyReport,
)
from iot_anomaly_detector.generator import SensorConfig


def evaluate(
    report: AnomalyReport,
    injected_timesteps: set,
    tolerance: int = 2,
    total_steps: int = 500,
) -> dict:
    """Compute precision/recall/F1 with a timestep tolerance window."""
    detected = set(report.detected_timesteps())

    true_positives = 0
    for t_gt in injected_timesteps:
        if any(abs(t_gt - t_det) <= tolerance for t_det in detected):
            true_positives += 1

    false_positives = 0
    for t_det in detected:
        if not any(abs(t_det - t_gt) <= tolerance for t_gt in injected_timesteps):
            false_positives += 1

    false_negatives = len(injected_timesteps) - true_positives

    precision = true_positives / (true_positives + false_positives + 1e-9)
    recall = true_positives / (len(injected_timesteps) + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    return {
        "tp": true_positives,
        "fp": false_positives,
        "fn": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def print_metrics(name: str, metrics: dict) -> None:
    print(f"\n{'─'*45}")
    print(f"  {name}")
    print(f"{'─'*45}")
    print(f"  TP={metrics['tp']}  FP={metrics['fp']}  FN={metrics['fn']}")
    print(f"  Precision : {metrics['precision']:.3f}")
    print(f"  Recall    : {metrics['recall']:.3f}")
    print(f"  F1        : {metrics['f1']:.3f}")


def main():
    print("=" * 50)
    print(" IoT Anomaly Detection — Demo")
    print("=" * 50)

    # ── 1. Generate data ──────────────────────────────────────
    print("\n[1/4] Generating 500-timestep IoT sensor stream (10 anomalies)...")
    gen = TimeSeriesGenerator(seed=42)
    readings, injected = gen.generate(n_steps=500, n_anomalies=10)
    data, sensor_names = gen.to_numpy(readings)
    timestamps = np.array([r.timestamp for r in readings])

    print(f"  Sensors     : {sensor_names}")
    print(f"  Data shape  : {data.shape}")
    print(f"\n  Injected anomalies:")
    injected_timesteps = set()
    for anom in injected:
        injected_timesteps.add(anom.timestep)
        print(f"    t={anom.timestep:3d}  sensor={anom.sensor:<12s}  "
              f"type={anom.anomaly_type.value:<12s}  magnitude={anom.magnitude:+.2f}")

    # ── 2. Statistical detector ───────────────────────────────
    print("\n[2/4] Running StatisticalDetector (z-score + IQR + MA deviation)...")
    stat_det = StatisticalDetector(window=50, zscore_threshold=3.0, iqr_k=1.5, ma_threshold=3.0)
    stat_report = stat_det.detect(data, sensor_names, timestamps)
    print(f"  Flagged {stat_report.n_findings} findings at "
          f"{len(stat_report.detected_timesteps())} unique timesteps")
    print(stat_report.summary())

    stat_metrics = evaluate(stat_report, injected_timesteps, tolerance=2, total_steps=500)
    print_metrics("StatisticalDetector", stat_metrics)

    # ── 3. LSTM detector ──────────────────────────────────────
    print("\n[3/4] Training LSTMDetector on 500-step clean baseline (no anomalies)...")
    # Train on the clean version of the same stream (same seed, no anomalies).
    # The LSTM learns temporal autocorrelation of the IoT signals;
    # anomalies appear as high reconstruction error spikes.
    clean_gen = TimeSeriesGenerator(seed=42)
    clean_readings, _ = clean_gen.generate(n_steps=500, n_anomalies=0)
    clean_data, _ = clean_gen.to_numpy(clean_readings)
    clean_data = clean_data.astype(np.float32)

    lstm_det = LSTMDetector(window=20, hidden_size=64, epochs=100, threshold_sigma=2.5)
    lstm_det.fit(clean_data, verbose=True)

    print("\n       Running LSTM detection on full 500-step stream...")
    lstm_report = lstm_det.detect(data, sensor_names, timestamps)
    print(f"  Flagged {lstm_report.n_findings} findings at "
          f"{len(lstm_report.detected_timesteps())} unique timesteps")
    print(lstm_report.summary())

    # LSTM uses a window of 20 steps — anomaly "contamination" spreads ±window steps.
    # Evaluate with a tolerance matching the window size.
    lstm_metrics = evaluate(lstm_report, injected_timesteps, tolerance=lstm_det.window, total_steps=500)
    print_metrics(f"LSTMDetector (window={lstm_det.window})", lstm_metrics)

    # ── 4. Combined ───────────────────────────────────────────
    print("\n[4/4] Combined report (union of both detectors)...")
    combined = stat_report.merge(lstm_report)
    combined_metrics = evaluate(combined, injected_timesteps, tolerance=lstm_det.window, total_steps=500)
    print_metrics("Combined (Statistical + LSTM)", combined_metrics)

    print("\n" + "=" * 50)
    print(" Summary")
    print("=" * 50)
    for name, m in [
        ("Statistical", stat_metrics),
        ("LSTM        ", lstm_metrics),
        ("Combined    ", combined_metrics),
    ]:
        print(f"  {name}  F1={m['f1']:.3f}  P={m['precision']:.3f}  R={m['recall']:.3f}")
    print()


if __name__ == "__main__":
    main()
