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
       --output predictions.csv
   ```
   This writes a CSV file containing ``1`` for anomalous windows and ``0`` otherwise.
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
   If the model file does not exist, the script performs a training run before
   evaluating. Providing ``--labels-path`` enables computing precision, recall
   and F1 score. The number of epochs can be customized with ``--train-epochs``.
6. Visualize sequences and anomaly flags:
   ```bash
   python -m src.visualize --csv-path data/raw/sensor_data.csv \
       --anomalies predictions.csv --output plot.png
   ```
   This generates ``plot.png`` with red regions indicating detected anomalies.
