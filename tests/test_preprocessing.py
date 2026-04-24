"""
test_preprocessing.py — Unit tests for EEG preprocessing pipeline
Run with: pytest tests/ -v
"""

import sys
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture
def synthetic_raw():
    """Create a small synthetic Raw for testing."""
    from src.preprocessing.loader import load_synthetic
    return load_synthetic(
        n_channels=22,
        sfreq=250.0,
        duration_s=60.0,
        n_trials_per_class=10,
        seed=42,
    )


class TestLoader:
    def test_synthetic_shape(self, synthetic_raw):
        raw = synthetic_raw
        assert raw.n_times > 0
        assert len(raw.ch_names) == 22
        assert raw.info["sfreq"] == 250.0

    def test_synthetic_has_events(self, synthetic_raw):
        import mne
        events, event_id = mne.events_from_annotations(synthetic_raw, verbose=False)
        assert len(events) > 0
        assert len(event_id) >= 2


class TestFiltering:
    def test_bandpass(self, synthetic_raw):
        from src.preprocessing.filter import apply_bandpass
        raw = synthetic_raw.copy()
        raw = apply_bandpass(raw, l_freq=1.0, h_freq=40.0)
        assert raw.info["sfreq"] == 250.0  # Sfreq unchanged

    def test_notch(self, synthetic_raw):
        from src.preprocessing.filter import apply_notch
        raw = synthetic_raw.copy()
        raw = apply_notch(raw, freqs=50.0)
        assert raw is not None

    def test_resample(self, synthetic_raw):
        from src.preprocessing.filter import resample
        raw = synthetic_raw.copy()
        raw = resample(raw, sfreq=160.0)
        assert raw.info["sfreq"] == 160.0

    def test_full_preprocess(self, synthetic_raw):
        from src.preprocessing.filter import preprocess_raw
        raw = preprocess_raw(synthetic_raw.copy())
        assert raw is not None


class TestEpocher:
    def test_create_epochs(self, synthetic_raw):
        import mne
        from src.preprocessing.filter import preprocess_raw
        from src.preprocessing.epocher import create_epochs, get_motor_imagery_data
        from src.preprocessing.loader import get_events

        raw = preprocess_raw(synthetic_raw.copy())
        events, event_id = get_events(raw, dataset="synthetic")

        if len(events) < 4:
            pytest.skip("Not enough events for epoching in short synthetic data")

        epochs = create_epochs(raw, events, event_id, tmin=-0.5, tmax=2.0,
                               baseline=(-0.5, 0), reject_peak_to_peak=None)
        assert len(epochs) > 0

        X, y = get_motor_imagery_data(epochs, tmin_mi=0.0, tmax_mi=1.5)
        assert X.ndim == 3
        assert X.shape[0] == len(y)
        assert X.shape[1] == 22  # n_channels
        assert set(np.unique(y)).issubset({0, 1})

    def test_balance_classes(self):
        from src.preprocessing.epocher import balance_classes
        X = np.random.randn(70, 22, 100)
        y = np.array([0] * 40 + [1] * 30)
        X_bal, y_bal = balance_classes(X, y, strategy="undersample")
        counts = np.bincount(y_bal)
        assert counts[0] == counts[1], "Classes should be balanced"


class TestFeatures:
    def test_csp_fit_transform(self):
        from src.features.csp import CSPFeatureExtractor
        X = np.random.randn(40, 22, 200)
        y = np.array([0] * 20 + [1] * 20)
        csp = CSPFeatureExtractor(n_components=4)
        features = csp.fit_transform(X, y)
        assert features.shape == (40, 4)

    def test_band_power(self):
        from src.features.erd_ers import extract_band_power_features
        X = np.random.randn(20, 22, 400)
        features = extract_band_power_features(X, sfreq=250.0)
        n_bands = 3  # mu, beta_low, beta_high
        assert features.shape == (20, 22 * n_bands)
