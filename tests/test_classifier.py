"""
test_classifier.py — Unit tests for BCI classifiers
"""

import sys
import numpy as np
import pytest
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture
def mock_eeg_data():
    """Generate mock epoched EEG data."""
    rng = np.random.default_rng(42)
    n_epochs = 80
    n_channels = 22
    n_times = 375  # 1.5s @ 250Hz

    X = rng.standard_normal((n_epochs, n_channels, n_times)) * 1e-6
    y = np.array([0] * 40 + [1] * 40)
    # Inject class-separable signal
    X[:40, 10:12, :] *= 0.3   # Class 0: suppress channels 10-11
    X[40:, 5:7, :] *= 0.3     # Class 1: suppress channels 5-6
    return X, y


class TestBuildPipeline:
    def test_lda_pipeline(self, mock_eeg_data):
        from src.models.classifier import build_pipeline
        X, y = mock_eeg_data
        pipe = build_pipeline("lda", n_csp_components=4)
        pipe.fit(X, y)
        preds = pipe.predict(X)
        assert len(preds) == len(y)

    def test_svm_pipeline(self, mock_eeg_data):
        from src.models.classifier import build_pipeline
        X, y = mock_eeg_data
        pipe = build_pipeline("svm", n_csp_components=4)
        pipe.fit(X, y)
        preds = pipe.predict(X)
        assert len(preds) == len(y)

    def test_rf_pipeline(self, mock_eeg_data):
        from src.models.classifier import build_pipeline
        X, y = mock_eeg_data
        pipe = build_pipeline("rf", n_csp_components=4)
        pipe.fit(X, y)
        preds = pipe.predict(X)
        assert len(preds) == len(y)

    def test_invalid_classifier_raises(self):
        from src.models.classifier import build_pipeline
        with pytest.raises(ValueError):
            build_pipeline("nonexistent_clf")


class TestCrossValidation:
    def test_cv_returns_dict(self, mock_eeg_data):
        from src.models.classifier import build_pipeline
        from src.models.cross_validate import run_cross_validation
        X, y = mock_eeg_data
        pipe = build_pipeline("lda", n_csp_components=4)
        results = run_cross_validation(pipe, X, y, strategy="kfold", n_splits=5)

        assert "accuracy_mean" in results
        assert "kappa_mean" in results
        assert "roc_auc_mean" in results
        assert 0.0 <= results["accuracy_mean"] <= 1.0

    def test_above_chance(self, mock_eeg_data):
        from src.models.classifier import build_pipeline
        from src.models.cross_validate import run_cross_validation
        X, y = mock_eeg_data
        pipe = build_pipeline("lda", n_csp_components=4)
        results = run_cross_validation(pipe, X, y, strategy="kfold", n_splits=5)
        # With injected signal, should beat chance (50%)
        assert results["accuracy_mean"] > 0.50, \
            f"Expected above-chance accuracy, got {results['accuracy_mean']:.3f}"


class TestModelIO:
    def test_save_load(self, tmp_path, mock_eeg_data):
        from src.models.classifier import build_pipeline, save_model, load_model
        X, y = mock_eeg_data
        pipe = build_pipeline("lda", n_csp_components=4)
        pipe.fit(X, y)
        orig_preds = pipe.predict(X)

        model_path = str(tmp_path / "test_model.pkl")
        save_model(pipe, model_path)
        loaded_pipe = load_model(model_path)
        loaded_preds = loaded_pipe.predict(X)

        np.testing.assert_array_equal(orig_preds, loaded_preds)
