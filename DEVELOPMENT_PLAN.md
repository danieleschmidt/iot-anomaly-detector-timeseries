# Development Plan

This checklist breaks down the main tasks for building the IoT Time Series Anomaly Detector.

## 1. Project Setup
- [x] Create the initial directory structure (data/, notebooks/, src/, tests/, saved_models/).
- [x] Add a `requirements.txt` with key dependencies.
- [x] Set up version control with an initial commit.

## 2. Data Generation and Preprocessing
- [x] Simulate multivariate sensor data and place it in `data/raw/`.
- [x] Implement `data_preprocessor.py` for normalization and windowing of sequences.
- [x] Write unit tests in `tests/test_data_preprocessor.py`.

## 3. Model Implementation
- [x] Create `autoencoder_model.py` defining the LSTM autoencoder.
- [x] Build a script `train_autoencoder.py` for training and saving the model.

## 4. Anomaly Detection Logic
- [x] Implement `anomaly_detector.py` to compute reconstruction error and flag anomalies.
- [x] Visualize normal vs. anomalous sequences.

## 5. Evaluation & Tuning
- [x] Define metrics and evaluate model performance.
- [x] Adjust network and preprocessing parameters based on results.

## 6. Documentation & Clean Up
- [x] Update `README.md` with usage instructions.
- [x] Provide example notebooks showcasing the workflow.
- [x] Ensure all tests pass and code is linted.


