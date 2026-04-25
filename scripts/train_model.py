#!/usr/bin/env python3
"""
train_model.py — Train & evaluate BCI classifier for a single subject.

Loads EEG data, runs the full preprocessing + feature extraction +
cross-validated classification, saves the model, and prints the report.

Usage:
    python scripts/train_model.py --subject 1 --classifier lda --cv 10
    python scripts/train_model.py --subject 1 --dataset physionet --classifier svm --cv 10
    python scripts/train_model.py --subject 2 --dataset synthetic --classifier rf --cv 5
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.preprocessing.loader import load_physionet, load_synthetic, get_events
from src.preprocessing.filter import preprocess_raw
from src.preprocessing.epocher import create_epochs, get_motor_imagery_data, balance_classes
from src.models.classifier import build_pipeline, save_model
from src.models.cross_validate import run_cross_validation, print_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Train EEG-BCI classifier for a single subject"
    )
    parser.add_argument("--subject", type=int, default=1,
                        help="Subject ID (default: 1)")
    parser.add_argument("--dataset", default="physionet",
                        choices=["physionet", "synthetic"],
                        help="Dataset (default: physionet)")
    parser.add_argument("--classifier", default="lda",
                        choices=["lda", "svm", "rf", "ensemble"],
                        help="Classifier type (default: lda)")
    parser.add_argument("--cv", type=int, default=10,
                        help="Number of cross-validation folds (default: 10)")
    parser.add_argument("--n_csp", type=int, default=6,
                        help="Number of CSP components (default: 6)")
    parser.add_argument("--output_dir", default="outputs",
                        help="Output directory (default: outputs)")
    parser.add_argument("--tmin_mi", type=float, default=0.5,
                        help="MI window start in seconds (default: 0.5)")
    parser.add_argument("--tmax_mi", type=float, default=3.5,
                        help="MI window end in seconds (default: 3.5)")
    args = parser.parse_args()

    logger.info(f"Training subject {args.subject:03d} | {args.dataset} | {args.classifier} | {args.cv}-fold CV")

    # ── Load ──────────────────────────────────────────────────────────
    if args.dataset == "physionet":
        logger.info("Loading PhysioNet data (runs 4, 8, 12 = L/R hand MI)...")
        raw = load_physionet(subject=args.subject, runs=[4, 8, 12])
    else:
        logger.info("Generating synthetic EEG...")
        raw = load_synthetic(n_channels=22, sfreq=250, duration_s=360,
                             n_trials_per_class=60, seed=args.subject)

    # ── Preprocess ────────────────────────────────────────────────────
    logger.info("Preprocessing...")
    raw = preprocess_raw(raw, l_freq=1.0, h_freq=40.0, notch_freq=50.0,
                         reference="average")

    # ── Epoch ─────────────────────────────────────────────────────────
    logger.info("Epoching...")
    events, event_id = get_events(raw, dataset=args.dataset)
    epochs = create_epochs(raw, events, event_id,
                           tmin=-1.0, tmax=4.0,
                           baseline=(-1.0, 0.0),
                           reject_peak_to_peak=150e-6)

    X, y = get_motor_imagery_data(epochs, tmin_mi=args.tmin_mi, tmax_mi=args.tmax_mi)
    X, y = balance_classes(X, y)

    logger.info(f"Dataset ready: X={X.shape}, classes={np.bincount(y)}")

    if len(X) < 10:
        logger.error("Too few epochs after rejection. Cannot train reliably.")
        sys.exit(1)

    # ── Train & Evaluate ──────────────────────────────────────────────
    pipeline = build_pipeline(args.classifier, n_csp_components=args.n_csp)
    results = run_cross_validation(pipeline, X, y,
                                   strategy="stratified",
                                   n_splits=args.cv)

    class_names = list(event_id.keys())
    print_report(results, class_names=class_names)

    # ── Save model ────────────────────────────────────────────────────
    out_dir = Path(args.output_dir) / f"subject_{args.subject:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    pipeline.fit(X, y)
    model_path = str(out_dir / "model.pkl")
    save_model(pipeline, model_path)

    results_clean = {k: v for k, v in results.items() if not isinstance(v, np.ndarray)}
    results_clean.update({
        "subject_id": args.subject,
        "dataset": args.dataset,
        "classifier": args.classifier,
        "n_csp": args.n_csp,
        "cv_folds": args.cv,
    })
    with open(out_dir / "results.json", "w") as f:
        json.dump(results_clean, f, indent=2)

    logger.info(f"Model saved → {model_path}")
    logger.info(f"Results saved → {out_dir}/results.json")
    logger.info("Done. ✓")


if __name__ == "__main__":
    main()
