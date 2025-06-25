# Development Plan

## Phase 1: Open Tasks
_No outstanding feature gaps from README_

## Phase 2: Testing & Hardening
- [ ] Add integration tests covering the end-to-end pipeline
- [ ] Run security (`bandit`) and style (`ruff`) scans and address findings

## Completed Tasks
- [x] Create the initial directory structure (data/, notebooks/, src/, tests/, saved_models/).
- [x] Add a `requirements.txt` with key dependencies.
- [x] Set up version control with an initial commit.
- [x] Simulate multivariate sensor data and place it in `data/raw/`.
- [x] Implement `data_preprocessor.py` for normalization and windowing of sequences.
- [x] Write unit tests in `tests/test_data_preprocessor.py`.
- [x] Create `autoencoder_model.py` defining the LSTM autoencoder.
- [x] Build a script `train_autoencoder.py` for training and saving the model.
- [x] Implement `anomaly_detector.py` to compute reconstruction error and flag anomalies.
- [x] Visualize normal vs. anomalous sequences.
- [x] Define metrics and evaluate model performance.
- [x] Adjust network and preprocessing parameters based on results.
- [x] Update `README.md` with usage instructions.
- [x] Provide example notebooks showcasing the workflow.
- [x] Ensure all tests pass and code is linted.
