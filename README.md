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
   python -m src.generate_data
   ```
3. Train the autoencoder:
   ```bash
   python -m src.train_autoencoder --epochs 5 --window-size 30 --latent-dim 16 --scaler standard
   ```
4. Detect anomalies:
   ```python
   from src.anomaly_detector import AnomalyDetector
   detector = AnomalyDetector('saved_models/autoencoder.h5')
   anomalies = detector.predict('data/raw/sensor_data.csv')
   print(anomalies)
   ```
5. Evaluate reconstruction error statistics:
   ```bash
   python -m src.evaluate_model --window-size 30 --threshold-factor 3 -\
       --output eval.json
   ```
