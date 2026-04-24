"""
cross_validate.py — Cross-validation strategies for BCI evaluation
evaluate.py — Metrics: accuracy, kappa, ROC-AUC, confusion matrix
"""

from __future__ import annotations
import logging
from typing import Optional

import numpy as np
from sklearn.model_selection import (
    StratifiedKFold,
    LeaveOneOut,
    cross_val_score,
    cross_validate,
)
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


def run_cross_validation(
    pipeline: Pipeline,
    X: np.ndarray,
    y: np.ndarray,
    strategy: str = "kfold",
    n_splits: int = 10,
    random_state: int = 42,
) -> dict:
    """
    Run cross-validation and return comprehensive metrics.

    For BCI research, stratified k-fold (10-fold) is standard.
    Leave-one-out (LOO) is used for small trial counts.

    Args:
        pipeline: Fitted-able sklearn Pipeline (CSP + classifier)
        X: EEG epochs (n_epochs, n_channels, n_times)
        y: Labels (n_epochs,)
        strategy: 'kfold' | 'stratified' | 'loo'
        n_splits: Number of folds for k-fold
        random_state: Reproducibility

    Returns:
        Dict with mean/std accuracy, kappa, per-fold scores
    """
    logger.info(f"Cross-validation: {strategy}, {n_splits}-fold on {len(y)} epochs")

    if strategy in ("kfold", "stratified"):
        cv = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state
        )
    elif strategy == "loo":
        cv = LeaveOneOut()
    else:
        raise ValueError(f"Unknown CV strategy: {strategy}")

    # Run CV with multiple scoring metrics
    cv_results = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=["accuracy", "roc_auc"],
        return_train_score=True,
        n_jobs=-1,
        verbose=0,
    )

    # Compute kappa from per-fold predictions
    kappa_scores = []
    y_pred_all = np.zeros_like(y)
    y_true_all = np.zeros_like(y)
    idx = 0

    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        kappa_scores.append(cohen_kappa_score(y_test, y_pred))
        y_pred_all[test_idx] = y_pred
        y_true_all[test_idx] = y_test

    results = {
        "accuracy_mean": cv_results["test_accuracy"].mean(),
        "accuracy_std": cv_results["test_accuracy"].std(),
        "accuracy_per_fold": cv_results["test_accuracy"].tolist(),
        "kappa_mean": np.mean(kappa_scores),
        "kappa_std": np.std(kappa_scores),
        "kappa_per_fold": kappa_scores,
        "roc_auc_mean": cv_results["test_roc_auc"].mean(),
        "roc_auc_std": cv_results["test_roc_auc"].std(),
        "y_pred_cv": y_pred_all,
        "y_true_cv": y_true_all,
        "confusion_matrix": confusion_matrix(y_true_all, y_pred_all).tolist(),
        "n_epochs": len(y),
        "n_folds": n_splits,
        "strategy": strategy,
    }

    logger.info(
        f"Results: Accuracy={results['accuracy_mean']:.3f}±{results['accuracy_std']:.3f}, "
        f"Kappa={results['kappa_mean']:.3f}±{results['kappa_std']:.3f}, "
        f"ROC-AUC={results['roc_auc_mean']:.3f}±{results['roc_auc_std']:.3f}"
    )

    return results


def print_report(results: dict, class_names: Optional[list[str]] = None) -> None:
    """Print a formatted evaluation report."""
    print("\n" + "="*60)
    print("  EEG-BCI CLASSIFICATION REPORT")
    print("="*60)
    print(f"  Dataset size    : {results['n_epochs']} epochs")
    print(f"  CV strategy     : {results['strategy']} ({results['n_folds']} folds)")
    print(f"  Accuracy        : {results['accuracy_mean']*100:.2f}% ± {results['accuracy_std']*100:.2f}%")
    print(f"  Cohen's Kappa   : {results['kappa_mean']:.4f} ± {results['kappa_std']:.4f}")
    print(f"  ROC-AUC         : {results['roc_auc_mean']:.4f} ± {results['roc_auc_std']:.4f}")
    print("-"*60)

    cm = np.array(results["confusion_matrix"])
    print("  Confusion Matrix:")
    if class_names:
        header = "  " + "  ".join(f"{c:>12}" for c in class_names)
        print(header)
    for i, row in enumerate(cm):
        label = class_names[i] if class_names else str(i)
        print(f"  {label:>10} | " + "  ".join(f"{v:>12}" for v in row))

    print("-"*60)
    acc_threshold = 0.70
    if results["accuracy_mean"] >= acc_threshold:
        print(f"  ✅ Exceeds {acc_threshold*100:.0f}% accuracy threshold for BCI use")
    else:
        print(f"  ⚠️  Below {acc_threshold*100:.0f}% threshold. Consider more data or feature tuning.")
    print("="*60 + "\n")


def evaluate_final(
    pipeline: Pipeline,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """
    Train on full training set, evaluate on held-out test set.

    Args:
        pipeline: Unfitted Pipeline
        X_train, y_train: Training data
        X_test, y_test: Test data

    Returns:
        Dict with full evaluation metrics
    """
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    logger.info(f"Final evaluation: Accuracy={acc:.3f}, Kappa={kappa:.3f}")
    print(report)

    return {
        "accuracy": acc,
        "kappa": kappa,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "y_pred": y_pred,
        "y_true": y_test,
    }
