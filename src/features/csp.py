"""
csp.py — Common Spatial Patterns (CSP) feature extraction
The gold-standard spatial filter for motor imagery BCI.
"""

from __future__ import annotations
import logging
from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from mne.decoding import CSP as MNE_CSP

logger = logging.getLogger(__name__)


class CSPFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    CSP-based feature extractor for motor imagery EEG.

    Common Spatial Patterns finds linear combinations of EEG channels
    (spatial filters) that MAXIMIZE variance for one class while
    MINIMIZING it for the other. For L/R hand MI:
      - Left-hand filters activate left motor cortex (C3 area)
      - Right-hand filters activate right motor cortex (C4 area)

    The log-variance of filtered signals is used as feature vector,
    which is approximately Gaussian distributed → ideal for LDA/SVM.

    Reference:
        Müller-Gerking et al. (1999). Designing optimal spatial filters
        for single-trial EEG classification. *Electroencephalogr. Clin. Neurophysiol.*

    Args:
        n_components: Number of CSP components (typically 4–8)
        reg: Regularization for covariance estimation
        log: Apply log-variance transform (recommended True)
    """

    def __init__(
        self,
        n_components: int = 6,
        reg: Optional[str] = "ledoit_wolf",
        log: bool = True,
        norm_trace: bool = False,
    ):
        self.n_components = n_components
        self.reg = reg
        self.log = log
        self.norm_trace = norm_trace
        self._csp = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CSPFeatureExtractor":
        """
        Fit CSP spatial filters.

        Args:
            X: EEG epochs shape (n_epochs, n_channels, n_times)
            y: Labels shape (n_epochs,)

        Returns:
            self
        """
        logger.info(f"Fitting CSP: {self.n_components} components, reg={self.reg}")
        self._csp = MNE_CSP(
            n_components=self.n_components,
            reg=self.reg,
            log=self.log,
            norm_trace=self.norm_trace,
        )
        self._csp.fit(X, y)
        logger.info(f"  CSP fitted on {X.shape[0]} epochs, {X.shape[1]} channels")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply CSP filters and extract log-variance features.

        Args:
            X: EEG epochs shape (n_epochs, n_channels, n_times)

        Returns:
            Features shape (n_epochs, n_components)
        """
        if self._csp is None:
            raise RuntimeError("CSP not fitted. Call fit() first.")
        features = self._csp.transform(X)
        return features

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.fit(X, y).transform(X)

    def get_patterns(self) -> np.ndarray:
        """Return CSP patterns (activation maps) for visualization."""
        if self._csp is None:
            raise RuntimeError("CSP not fitted.")
        return self._csp.patterns_

    def get_filters(self) -> np.ndarray:
        """Return CSP filters (spatial filters)."""
        if self._csp is None:
            raise RuntimeError("CSP not fitted.")
        return self._csp.filters_


class FilterBankCSP(BaseEstimator, TransformerMixin):
    """
    Filter Bank CSP (FBCSP) — applies CSP to multiple frequency bands.

    Extracts CSP features from sub-bands independently, then
    concatenates. Captures band-specific motor information beyond
    a single broadband filter.

    Sub-bands: delta, theta, mu, beta-low, beta-high.

    Reference:
        Ang et al. (2012). Filter Bank Common Spatial Pattern (FBCSP).
    """

    BANDS = {
        "mu": (8, 12),
        "beta_low": (13, 20),
        "beta_high": (20, 30),
        "theta": (4, 8),
    }

    def __init__(
        self,
        n_components: int = 4,
        sfreq: float = 160.0,
        bands: Optional[dict] = None,
    ):
        self.n_components = n_components
        self.sfreq = sfreq
        self.bands = bands or self.BANDS
        self._csp_filters: dict = {}

    def _bandpass_epoch(
        self, X: np.ndarray, low: float, high: float
    ) -> np.ndarray:
        """Apply zero-phase bandpass filter to epoch array."""
        from scipy.signal import butter, sosfiltfilt

        nyq = self.sfreq / 2
        low_norm = max(low / nyq, 0.001)
        high_norm = min(high / nyq, 0.999)
        sos = butter(5, [low_norm, high_norm], btype="bandpass", output="sos")

        X_filtered = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                X_filtered[i, j] = sosfiltfilt(sos, X[i, j])
        return X_filtered

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FilterBankCSP":
        self._csp_filters = {}
        for band_name, (low, high) in self.bands.items():
            logger.info(f"  FBCSP fitting band: {band_name} ({low}–{high} Hz)")
            X_band = self._bandpass_epoch(X, low, high)
            csp = CSPFeatureExtractor(n_components=self.n_components)
            csp.fit(X_band, y)
            self._csp_filters[band_name] = (csp, low, high)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        features_all = []
        for band_name, (csp, low, high) in self._csp_filters.items():
            X_band = self._bandpass_epoch(X, low, high)
            feat = csp.transform(X_band)
            features_all.append(feat)
        return np.concatenate(features_all, axis=1)

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.fit(X, y).transform(X)
