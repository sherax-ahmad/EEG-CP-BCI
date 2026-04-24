"""
classifier.py — BCI Classifier Pipeline
LDA, SVM, Random Forest pipelines with CSP feature extraction.
"""

from __future__ import annotations
import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from mne.decoding import CSP

logger = logging.getLogger(__name__)


def build_pipeline(
    classifier: str = "lda",
    n_csp_components: int = 6,
    csp_reg: str = "ledoit_wolf",
    **clf_kwargs,
) -> Pipeline:
    """
    Build a scikit-learn Pipeline: CSP → (Scaler) → Classifier.

    The pipeline encapsulates all processing steps for clean
    cross-validation without data leakage. CSP is fitted only
    on training folds.

    Args:
        classifier: 'lda' | 'svm' | 'rf' | 'ensemble'
        n_csp_components: Number of CSP spatial filters
        csp_reg: CSP covariance regularization
        **clf_kwargs: Extra args passed to classifier

    Returns:
        sklearn Pipeline ready for fit/predict/cross_val_score
    """
    csp = CSP(
        n_components=n_csp_components,
        reg=csp_reg,
        log=True,
        norm_trace=False,
    )

    if classifier == "lda":
        clf = LinearDiscriminantAnalysis(
            solver=clf_kwargs.get("solver", "svd"),
            shrinkage=clf_kwargs.get("shrinkage", None),
        )
        steps = [("csp", csp), ("lda", clf)]

    elif classifier == "svm":
        clf = SVC(
            kernel=clf_kwargs.get("kernel", "rbf"),
            C=clf_kwargs.get("C", 1.0),
            gamma=clf_kwargs.get("gamma", "scale"),
            probability=True,
        )
        steps = [("csp", csp), ("scaler", StandardScaler()), ("svm", clf)]

    elif classifier == "rf":
        clf = RandomForestClassifier(
            n_estimators=clf_kwargs.get("n_estimators", 200),
            max_depth=clf_kwargs.get("max_depth", None),
            random_state=clf_kwargs.get("random_state", 42),
            n_jobs=-1,
        )
        steps = [("csp", csp), ("rf", clf)]

    elif classifier == "ensemble":
        lda = LinearDiscriminantAnalysis()
        svm = SVC(kernel="rbf", C=1.0, probability=True)
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        voting = VotingClassifier(
            estimators=[("lda", lda), ("svm", svm), ("rf", rf)],
            voting="soft",
        )
        steps = [("csp", csp), ("scaler", StandardScaler()), ("ensemble", voting)]

    else:
        raise ValueError(f"Unknown classifier: {classifier}. Choose: lda, svm, rf, ensemble")

    pipeline = Pipeline(steps)
    logger.info(f"Built pipeline: CSP({n_csp_components}) → {classifier.upper()}")
    return pipeline


def save_model(pipeline: Pipeline, path: str = "outputs/model.pkl") -> None:
    """Save trained pipeline to disk."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(pipeline, f)
    logger.info(f"Model saved: {path}")


def load_model(path: str = "outputs/model.pkl") -> Pipeline:
    """Load trained pipeline from disk."""
    if not Path(path).exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    with open(path, "rb") as f:
        pipeline = pickle.load(f)
    logger.info(f"Model loaded: {path}")
    return pipeline


def predict_proba(
    pipeline: Pipeline,
    X: np.ndarray,
) -> np.ndarray:
    """
    Get class probabilities for each epoch.

    Args:
        pipeline: Fitted sklearn Pipeline
        X: EEG epochs (n_epochs, n_channels, n_times)

    Returns:
        Probabilities (n_epochs, n_classes)
    """
    if hasattr(pipeline, "predict_proba"):
        return pipeline.predict_proba(X)
    else:
        # LDA doesn't always have predict_proba in all sklearn versions
        return pipeline.decision_function(X)
