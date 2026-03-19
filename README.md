# IoT Anomaly Detector — Time Series

Real-time anomaly detection for IoT sensor streams. No cloud required, no data science degree needed.

Detects point spikes, contextual outliers, and collective pattern shifts in temperature, pressure, and vibration data — using both fast statistical methods and a trained LSTM autoencoder.

---

## What's inside

| Component | Description |
|---|---|
| `TimeSeriesGenerator` | Synthetic IoT sensor data with configurable anomaly injection |
| `StatisticalDetector` | Z-score, IQR, and moving-average deviation — fast, zero ML |
| `LSTMDetector` | LSTM autoencoder trained on normal data, flags high reconstruction error |
| `AnomalyReport` | Structured findings: timestamp, sensor, severity, type, confidence |

### Anomaly types

- **Point** — sudden spike or dip far outside normal range (sensor failure, physical event)
- **Contextual** — value that's globally plausible but wrong for the current operating state (e.g., temperature spiking during a cool cycle)
- **Collective** — a subsequence that's collectively anomalous (gradual drift, pattern shift)

---

## Quick start

```bash
# Install dependencies (conda recommended)
~/anaconda3/bin/pip install -e .

# Run the demo: 500 timesteps, 10 injected anomalies, compare both detectors
~/anaconda3/bin/python3 demo.py

# Run tests
~/anaconda3/bin/python3 -m pytest tests/ -v
```

---

## Usage

### Generate synthetic sensor data

```python
from iot_anomaly_detector import TimeSeriesGenerator

gen = TimeSeriesGenerator(seed=42)
readings, injected_anomalies = gen.generate(n_steps=500, n_anomalies=10)

# Convert to numpy for detector input
data, sensor_names = gen.to_numpy(readings)
# data.shape → (500, 3)   sensors: temperature, pressure, vibration
```

### Statistical detection (no training required)

```python
from iot_anomaly_detector import StatisticalDetector

det = StatisticalDetector(window=50, zscore_threshold=3.0)
report = det.detect(data, sensor_names)

print(report.summary())
# AnomalyReport: 12 findings
#   Critical: 2  High: 5
#   pressure: 4 anomalies
#   temperature: 5 anomalies
#   vibration: 3 anomalies
```

### LSTM autoencoder detection

```python
from iot_anomaly_detector import LSTMDetector

# Train on normal data (first half of stream, before anomalies start)
det = LSTMDetector(window=20, hidden_size=64, epochs=30)
det.fit(data[:250], verbose=True)

# Detect on full stream
report = det.detect(data, sensor_names)
```

### Work with findings

```python
from iot_anomaly_detector import AnomalyReport

# Merge reports from multiple detectors
combined = stat_report.merge(lstm_report)

# Inspect findings
for finding in combined.findings:
    print(f"t={finding.timestep:3d} | {finding.sensor:<12s} | "
          f"{finding.severity.value:<8s} | conf={finding.confidence:.2f} | "
          f"{finding.description}")

# Group by sensor
for sensor, findings in combined.by_sensor().items():
    print(f"{sensor}: {len(findings)} anomalies")
```

---

## Architecture

```
TimeSeriesGenerator
    └─ generate() → List[SensorReading], List[InjectedAnomaly]
    └─ to_numpy()  → ndarray(n_steps, n_sensors)

StatisticalDetector
    └─ detect(data, sensor_names) → AnomalyReport
        ├─ z-score (rolling window)
        ├─ IQR fences (Tukey)
        └─ moving-average deviation

LSTMDetector
    ├─ fit(normal_data)          ← train autoencoder
    └─ detect(data, sensor_names) → AnomalyReport
        └─ per-timestep reconstruction error > threshold → anomaly

AnomalyReport
    ├─ findings: List[AnomalyFinding]
    ├─ merge(other) → AnomalyReport
    ├─ detected_timesteps() → List[int]
    └─ summary() → str
```

---

## Design decisions

**Why two detectors?**

- Statistical methods are fast, interpretable, and work immediately with no training data. Good for obvious spikes.
- LSTM autoencoder catches subtle pattern deviations that rule-based methods miss. Requires a window of normal data for calibration.
- Combined, they cover more anomaly types with better recall.

**Why no sklearn?**

The statistical detector is implemented directly with numpy/scipy — no sklearn needed. Fewer dependencies, easier deployment to edge devices.

**Why PyTorch for LSTM?**

PyTorch's dynamic graph is more flexible for sequence models and runs well both on CPU (edge) and GPU (cloud).

---

## Requirements

- Python 3.8+
- numpy ≥ 1.21
- scipy ≥ 1.7
- torch ≥ 1.12

---

## License

MIT
