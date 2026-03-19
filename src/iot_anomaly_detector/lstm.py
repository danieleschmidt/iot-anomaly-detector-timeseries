"""
LSTM Autoencoder anomaly detector.

Train on normal data, then flag timesteps whose reconstruction error
exceeds a threshold as anomalies (unsupervised).

Requires PyTorch (available in the conda environment).
"""

from __future__ import annotations

from typing import List, Optional, Tuple
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .report import AnomalyFinding, AnomalyReport, Severity


# -----------------------------------------------------------------------
# Model definition
# -----------------------------------------------------------------------

class _LSTMAutoencoder(nn.Module):
    """Sequence-to-sequence LSTM autoencoder."""

    def __init__(self, n_features: int, hidden_size: int = 64, n_layers: int = 1):
        super().__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size

        self.encoder = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
        )
        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
        )
        self.output_layer = nn.Linear(hidden_size, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (batch, seq_len, n_features)
        returns : (batch, seq_len, n_features)  — reconstruction
        """
        seq_len = x.size(1)
        enc_out, (h, c) = self.encoder(x)

        # Use last encoder hidden state as context, repeat for each decode step
        context = h[-1].unsqueeze(1).repeat(1, seq_len, 1)  # (batch, seq_len, hidden)
        dec_out, _ = self.decoder(context)
        reconstruction = self.output_layer(dec_out)
        return reconstruction


# -----------------------------------------------------------------------
# Sliding-window helpers
# -----------------------------------------------------------------------

def _sliding_windows(data: np.ndarray, window: int) -> np.ndarray:
    """Create (n_windows, window, n_features) array via sliding window."""
    n = len(data)
    if n < window:
        raise ValueError(f"Data length {n} < window size {window}")
    windows = np.stack([data[i: i + window] for i in range(n - window + 1)])
    return windows.astype(np.float32)


# -----------------------------------------------------------------------
# Public detector class
# -----------------------------------------------------------------------

class LSTMDetector:
    """
    LSTM Autoencoder–based anomaly detector.

    Workflow
    --------
    1. ``fit(normal_data)``  — train on data that contains no (or few) anomalies.
       Data is internally z-score normalised per sensor before training.
    2. ``detect(data, ...)`` — score each timestep by reconstruction error,
       return an ``AnomalyReport`` for points above the threshold.

    The threshold is set automatically from the training reconstruction errors
    (mean + ``threshold_sigma`` × std).

    Parameters
    ----------
    window : int
        Length of the sliding window fed to the LSTM (default 20).
    hidden_size : int
        LSTM hidden dimension (default 64).
    n_layers : int
        Number of LSTM layers in encoder/decoder (default 1).
    threshold_sigma : float
        How many std devs above train-set mean error to set the alert threshold.
    lr : float
        Adam learning rate.
    epochs : int
        Training epochs.
    batch_size : int
        Mini-batch size.
    device : str or None
        "cpu" / "cuda" / None (auto-detect).

    Example
    -------
    >>> det = LSTMDetector(window=20, epochs=20)
    >>> det.fit(normal_data)
    >>> report = det.detect(full_data, sensor_names, timestamps)
    """

    def __init__(
        self,
        window: int = 20,
        hidden_size: int = 64,
        n_layers: int = 1,
        threshold_sigma: float = 3.0,
        lr: float = 1e-3,
        epochs: int = 30,
        batch_size: int = 32,
        device: Optional[str] = None,
    ):
        self.window = window
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.threshold_sigma = threshold_sigma
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        # Percentile-based threshold (used if > 0; overrides sigma-based)
        self._threshold_percentile: float = 0.0

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model: Optional[_LSTMAutoencoder] = None
        self.threshold_: Optional[float] = None
        self._train_mean: Optional[float] = None
        self._train_std: Optional[float] = None
        self._fitted = False

        # Per-feature normalization stats (fit on training data)
        self._norm_mean: Optional[np.ndarray] = None
        self._norm_std: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, data: np.ndarray, verbose: bool = True) -> "LSTMDetector":
        """
        Train autoencoder on ``data``.

        Parameters
        ----------
        data : ndarray shape (n_steps, n_features)
        verbose : print loss every 10 epochs

        Returns self for chaining.
        """
        n_features = data.shape[1]
        self.model = _LSTMAutoencoder(n_features, self.hidden_size, self.n_layers)
        self.model.to(self.device)

        # Fit z-score normalisation on training data
        self._norm_mean = data.mean(axis=0)
        self._norm_std = data.std(axis=0)
        self._norm_std[self._norm_std < 1e-9] = 1.0  # avoid div-by-zero
        data = self._normalize(data)

        windows = _sliding_windows(data, self.window)  # (N, window, n_features)
        tensor = torch.from_numpy(windows).to(self.device)
        dataset = TensorDataset(tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for (batch,) in loader:
                optimizer.zero_grad()
                recon = self.model(batch)
                loss = criterion(recon, batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(batch)
            epoch_loss /= len(windows)
            if verbose and (epoch + 1) % 10 == 0:
                print(f"  [LSTM] epoch {epoch+1}/{self.epochs}  loss={epoch_loss:.6f}")

        # Compute per-window reconstruction errors on training set
        errors = self._reconstruction_errors(windows)
        self._train_mean = float(np.mean(errors))
        self._train_std = float(np.std(errors))
        # Threshold: mean + threshold_sigma * std of training errors
        self.threshold_ = self._train_mean + self.threshold_sigma * self._train_std

        if verbose:
            print(f"  [LSTM] threshold set to {self.threshold_:.6f} "
                  f"(mean={self._train_mean:.6f} + {self.threshold_sigma}σ={self._train_std:.6f})")

        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Detect
    # ------------------------------------------------------------------

    def detect(
        self,
        data: np.ndarray,
        sensor_names: List[str],
        timestamps: Optional[np.ndarray] = None,
    ) -> AnomalyReport:
        """
        Score data and return anomaly findings above threshold.

        The reconstruction error is averaged across all sensors for each
        timestep window. A timestep is flagged if its error exceeds the
        trained threshold.

        Parameters
        ----------
        data : ndarray (n_steps, n_features)
        sensor_names : list of str
        timestamps : optional (n_steps,) array

        Returns
        -------
        AnomalyReport
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before detect().")

        n_steps = data.shape[0]
        if timestamps is None:
            timestamps = np.arange(n_steps, dtype=float)

        data = self._normalize(data)

        windows = _sliding_windows(data, self.window)  # (N, window, n_features)
        errors = self._reconstruction_errors(windows)   # (N,)

        # Map window errors back to timesteps — each timestep t is covered by
        # windows starting at max(0, t-window+1) … t.
        # Use mean of covering windows as the timestep score.
        timestep_scores = np.zeros(n_steps)
        counts = np.zeros(n_steps, dtype=int)
        for w_idx, err in enumerate(errors):
            for t in range(w_idx, w_idx + self.window):
                if t < n_steps:
                    timestep_scores[t] += err
                    counts[t] += 1
        counts = np.maximum(counts, 1)
        timestep_scores /= counts

        # Compare against training error threshold.
        # Score = (error - train_mean) / train_std (how many sigma above training mean)
        z_scores = (timestep_scores - self._train_mean) / (self._train_std + 1e-9)

        report = AnomalyReport()
        for t in range(n_steps):
            z = z_scores[t]
            if z > self.threshold_sigma:
                severity = self._severity(z)
                conf = min(1.0, (z - self.threshold_sigma) / (self.threshold_sigma + 1e-9) + 0.5)
                dominant_sensor = self._dominant_sensor(data, t, sensor_names)
                report.add(AnomalyFinding(
                    timestep=t,
                    timestamp=float(timestamps[t]),
                    sensor=dominant_sensor,
                    severity=severity,
                    anomaly_type="unknown",
                    confidence=round(float(conf), 3),
                    detector="lstm_autoencoder",
                    score=round(float(z), 4),
                    value=round(float(data[t, sensor_names.index(dominant_sensor)]), 4),
                    description=f"Reconstruction z-score {z:.2f}σ > threshold {self.threshold_sigma}σ "
                                f"(raw error={timestep_scores[t]:.4f})",
                ))

        return report

    def reconstruction_error_series(self, data: np.ndarray) -> np.ndarray:
        """Return per-timestep reconstruction error array (useful for plotting)."""
        if not self._fitted:
            raise RuntimeError("Call fit() before using.")
        n_steps = data.shape[0]
        data = self._normalize(data)
        windows = _sliding_windows(data, self.window)
        errors = self._reconstruction_errors(windows)
        timestep_scores = np.zeros(n_steps)
        counts = np.zeros(n_steps, dtype=int)
        for w_idx, err in enumerate(errors):
            for t in range(w_idx, w_idx + self.window):
                if t < n_steps:
                    timestep_scores[t] += err
                    counts[t] += 1
        counts = np.maximum(counts, 1)
        return timestep_scores / counts

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _normalize(self, data: np.ndarray) -> np.ndarray:
        """Z-score normalise using training stats."""
        return (data - self._norm_mean) / self._norm_std

    def _reconstruction_errors(self, windows: np.ndarray) -> np.ndarray:
        """Compute mean-squared reconstruction error per window."""
        self.model.eval()
        tensor = torch.from_numpy(windows).to(self.device)
        with torch.no_grad():
            recon = self.model(tensor)
            errors = ((tensor - recon) ** 2).mean(dim=(1, 2))  # per-window scalar
        return errors.cpu().numpy()

    def _dominant_sensor(
        self, data: np.ndarray, t: int, sensor_names: List[str]
    ) -> str:
        """Heuristic: which sensor has the highest normalised deviation at t."""
        if self.model is None or t < self.window:
            return sensor_names[0]
        start = max(0, t - self.window + 1)
        window_data = self._normalize(data[start: start + self.window]).astype(np.float32)
        tensor = torch.from_numpy(window_data[np.newaxis]).to(self.device)
        self.model.eval()
        with torch.no_grad():
            recon = self.model(tensor)[0].cpu().numpy()
        # Error per sensor at position (t - start) within the window
        pos = t - start
        errors = np.abs(window_data[pos] - recon[pos])
        return sensor_names[int(np.argmax(errors))]

    @staticmethod
    def _severity(norm_score: float) -> Severity:
        if norm_score >= 6:
            return Severity.CRITICAL
        elif norm_score >= 4:
            return Severity.HIGH
        elif norm_score >= 2.5:
            return Severity.MEDIUM
        return Severity.LOW
