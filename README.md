# IoT Time Series Anomaly Detector

This project aims to develop an anomaly detection system for multivariate time series data, simulating readings from IoT sensors. We will use an LSTM-based autoencoder to learn normal operating patterns and identify anomalies based on reconstruction error.

## Project Goals
- Implement robust preprocessing for multivariate time series data, including normalization and windowing.
- Design and train an LSTM autoencoder model.
- Develop a mechanism to calculate reconstruction error and set a threshold for anomaly detection.
- Visualize normal vs. anomalous data points/sequences.
- Evaluate the anomaly detection performance (if using a dataset with labeled anomalies, otherwise qualitative).

## Tech Stack (Planned)
- Python
- TensorFlow / Keras or PyTorch
- Pandas, NumPy
- Scikit-learn (for preprocessing/scaling)
- Matplotlib / Seaborn (for visualization)

## Initial File Structure
iot-anomaly-detector-timeseries/
├── data/
│   └── raw/
│       └── sensor_data.csv # Simulated multivariate sensor data
│   └── processed/
├── notebooks/
│   └── timeseries_eda.ipynb
├── src/
│   ├── __init__.py
│   ├── data_preprocessor.py # Time series specific preprocessing
│   ├── autoencoder_model.py # LSTM autoencoder definition
│   ├── train_autoencoder.py # Training script
│   └── anomaly_detector.py  # Anomaly detection logic & visualization
├── tests/
│   ├── __init__.py
│   └── test_data_preprocessor.py
├── saved_models/
├── requirements.txt
├── .gitignore
└── README.md

## How to Contribute (and test Jules)
This project leverages Jules for building out the anomaly detection pipeline. Please create detailed issues for features and bug fixes.

## Quickstart

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Generate example data:
   ```bash
  python -m src.generate_data --num-samples 1000 --num-features 3 \
      --output-path data/raw/sensor_data.csv \
      --labels-path data/raw/sensor_labels.csv \
      --seed 42 \
      --anomaly-start 200 \
      --anomaly-length 20 \
      --anomaly-magnitude 3.0
   ```
   The command-line options let you control the number of samples, how many
   sensor streams to simulate, optionally generate a labels file with the
   anomaly locations, set a random seed for reproducibility, adjust the
   inserted anomaly window, and specify where the CSV is written.
3. Train the autoencoder:
   ```bash
   python -m src.train_autoencoder \
       --epochs 5 \
       --window-size 30 \
       --step 1 \
       --latent-dim 16 \
       --scaler standard \
       --model-path saved_models/autoencoder.h5 \
       --scaler-path saved_models/scaler.pkl
   ```
   The `--model-path` option controls where the trained model is written.
4. Detect anomalies:
   ```python
  from src.anomaly_detector import AnomalyDetector
  detector = AnomalyDetector('saved_models/autoencoder.h5', 'saved_models/scaler.pkl')
  anomalies = detector.predict('data/raw/sensor_data.csv')
   print(anomalies)
   ```
   You can also invoke anomaly detection from the command line:
   ```bash
  python -m src.anomaly_detector --model-path saved_models/autoencoder.h5 \
      --scaler-path saved_models/scaler.pkl \
      --csv-path data/raw/sensor_data.csv \
      --step 1 \
      --threshold 0.5 \
      --output predictions.csv
  ```
  Or derive the threshold from a reconstruction error quantile instead:
  ```bash
  python -m src.anomaly_detector --model-path saved_models/autoencoder.h5 \
      --scaler-path saved_models/scaler.pkl \
      --csv-path data/raw/sensor_data.csv \
      --step 1 \
      --quantile 0.95 \
      --output predictions.csv
  ```
  This writes a CSV file containing ``1`` for anomalous windows and ``0`` otherwise.
  Supply either ``--threshold`` for a manual value or ``--quantile`` to derive
  the threshold from the reconstruction error. These options are mutually
 exclusive and ``--quantile`` must be between 0 and 1 (exclusive). Invalid
 quantile values are rejected before the model is loaded.
5. Evaluate reconstruction error statistics:
   ```bash
   python -m src.evaluate_model \
       --window-size 30 \
       --step 1 \
       --threshold-factor 3 \
       --model-path saved_models/autoencoder.h5 \
       --scaler-path saved_models/scaler.pkl \
       --labels-path data/raw/sensor_labels.csv \
       --train-epochs 1 \
       --output eval.json
   ```
   Alternatively derive the threshold from a reconstruction error quantile:
   ```bash
   python -m src.evaluate_model \
       --window-size 30 \
       --step 1 \
       --quantile 0.95 \
       --model-path saved_models/autoencoder.h5 \
       --scaler-path saved_models/scaler.pkl \
       --labels-path data/raw/sensor_labels.csv \
       --train-epochs 1 \
       --output eval.json
   ```
   The ``--threshold-factor`` and ``--quantile`` options are mutually
   exclusive. ``--quantile`` must be between 0 and 1 (exclusive).
   If the model file does not exist, the script performs a training run before
   evaluating. Providing ``--labels-path`` enables computing precision, recall
   and F1 score. The number of epochs can be customized with ``--train-epochs``.
6. Visualize sequences and anomaly flags:
   ```bash
   python -m src.visualize --csv-path data/raw/sensor_data.csv \
       --anomalies predictions.csv --output plot.png
   ```
   This generates ``plot.png`` with red regions indicating detected anomalies.

## Data Validation

The project includes comprehensive data validation to ensure data quality and prevent pipeline failures. The data validator checks for schema compliance, data quality issues, and time series properties.

### Basic Data Validation

Validate a CSV file with default settings:

```bash
python -m src.data_validator data/raw/sensor_data.csv
```

### Advanced Validation Options

Set validation strictness level:
```bash
# Strict validation (strict error checking)
python -m src.data_validator data/raw/sensor_data.csv --validation-level strict

# Permissive validation (more lenient)
python -m src.data_validator data/raw/sensor_data.csv --validation-level permissive
```

Validate with expected columns:
```bash
python -m src.data_validator data/raw/sensor_data.csv \
    --expected-columns temperature,humidity,pressure
```

Enable auto-fix for common issues:
```bash
python -m src.data_validator data/raw/sensor_data.csv \
    --auto-fix \
    --output data/processed/cleaned_sensor_data.csv
```

Time series validation with timestamp column:
```bash
python -m src.data_validator data/raw/sensor_data.csv \
    --time-column timestamp \
    --report validation_report.md
```

Generate comprehensive reports:
```bash
python -m src.data_validator data/raw/sensor_data.csv \
    --report validation_report.md \
    --json-summary validation_summary.json \
    --verbose
```

### Validation Features

The data validator checks for:

- **File Format**: File existence, readability, and format compliance
- **Schema Validation**: Column presence, data types, and structure
- **Data Quality**: Missing values, duplicates, constant columns, and outliers
- **Time Series Properties**: Sequence length, time monotonicity, and intervals
- **Auto-fix Capabilities**: Automatic correction of common data issues

### Integration with Data Preprocessing

Data validation is automatically integrated into the data preprocessing pipeline. You can control validation behavior:

```python
from src.data_preprocessor import DataPreprocessor
from src.data_validator import ValidationLevel

# Enable validation with moderate strictness (default)
preprocessor = DataPreprocessor(enable_validation=True, validation_level=ValidationLevel.MODERATE)

# Disable validation for performance-critical applications
preprocessor = DataPreprocessor(enable_validation=False)

# Enable auto-fix during preprocessing
scaled_data = preprocessor.fit_transform(df, validate=True, auto_fix=True)
```

## Running Integration Tests

Integration tests verify the full data generation, training, and detection pipeline. They are marked with the `integration` marker. Execute them separately with:

```bash
pytest -m integration
```

You can also run them via Make:

```bash
make integration
```


## Development Setup

Install all runtime and development dependencies using the helper script:

```bash
./scripts/setup.sh
# or via Make
make setup
```

Run the full test suite including style and security scans via:

```bash
./scripts/test.sh
# or simply
make test
```

## Continuous Integration

Every push and pull request triggers our GitHub Actions workflow which installs dependencies, lints, runs security scans, and executes the test suite.
