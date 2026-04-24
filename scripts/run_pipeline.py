#!/usr/bin/env python3
"""
run_pipeline.py — Full EEG-BCI Pipeline Runner

Executes the complete pipeline:
  1. Load EEG data (PhysioNet / BCI Comp IV / Synthetic)
  2. Preprocess (filter, ICA, epoch)
  3. Extract features (CSP, ERD/ERS)
  4. Train & evaluate classifier (LDA/SVM/RF)
  5. Generate visualizations and reports

Usage:
    # Quick test with synthetic data:
    python scripts/run_pipeline.py --dataset synthetic

    # PhysioNet subject 1:
    python scripts/run_pipeline.py --dataset physionet --subject 1

    # Multiple subjects with SVM:
    python scripts/run_pipeline.py --dataset physionet --subject 1 2 3 --classifier svm

    # Custom config:
    python scripts/run_pipeline.py --config configs/default_config.yaml --subject 1
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import yaml

# Add project root to path
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
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_single_subject(
    subject_id: int,
    dataset: str = "physionet",
    classifier: str = "lda",
    n_csp: int = 6,
    output_dir: str = "outputs",
    config: dict = None,
) -> dict:
    """Run full BCI pipeline for a single subject."""
    t_start = time.time()
    logger.info(f"\n{'='*60}")
    logger.info(f"  SUBJECT {subject_id:03d} | dataset={dataset} | clf={classifier}")
    logger.info(f"{'='*60}")

    # ── Step 1: Load Data ──────────────────────────────────────────────
    logger.info("[1/5] Loading EEG data...")
    if dataset == "physionet":
        raw = load_physionet(subject=subject_id, runs=[4, 8, 12])
    elif dataset == "synthetic":
        raw = load_synthetic(n_channels=22, sfreq=250, duration_s=360,
                             n_trials_per_class=60)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    # ── Step 2: Preprocess ─────────────────────────────────────────────
    logger.info("[2/5] Preprocessing (filter, reference)...")
    cfg = config or {}
    prep_cfg = cfg.get("preprocessing", {})

    raw = preprocess_raw(
        raw,
        l_freq=prep_cfg.get("bandpass", {}).get("l_freq", 1.0),
        h_freq=prep_cfg.get("bandpass", {}).get("h_freq", 40.0),
        notch_freq=prep_cfg.get("notch_freq", 50.0),
        sfreq_resample=prep_cfg.get("sfreq_resample", None),
        reference=prep_cfg.get("reference", "average"),
    )

    # ── Step 3: Epoch ──────────────────────────────────────────────────
    logger.info("[3/5] Creating epochs...")
    events, event_id = get_events(raw, dataset=dataset)

    ep_cfg = prep_cfg.get("epoch", {})
    epochs = create_epochs(
        raw=raw,
        events=events,
        event_id=event_id,
        tmin=ep_cfg.get("tmin", -1.0),
        tmax=ep_cfg.get("tmax", 4.0),
        baseline=tuple(ep_cfg.get("baseline", [-1.0, 0.0])),
        reject_peak_to_peak=ep_cfg.get("reject_peak_to_peak", 150e-6),
    )

    X, y = get_motor_imagery_data(epochs, tmin_mi=0.5, tmax_mi=3.5)
    X, y = balance_classes(X, y, strategy="undersample")
    logger.info(f"  Final dataset: X={X.shape}, y={np.bincount(y)}")

    if len(X) < 20:
        logger.warning("  ⚠️  Very few epochs! Results may be unreliable.")

    # ── Step 4: Classify ───────────────────────────────────────────────
    logger.info(f"[4/5] Training {classifier.upper()} classifier with CSP...")
    pipeline = build_pipeline(
        classifier=classifier,
        n_csp_components=n_csp,
        csp_reg="ledoit_wolf",
    )

    cv_cfg = cfg.get("classification", {}).get("cross_validation", {})
    results = run_cross_validation(
        pipeline=pipeline,
        X=X,
        y=y,
        strategy=cv_cfg.get("strategy", "stratified"),
        n_splits=cv_cfg.get("n_splits", 10),
        random_state=cv_cfg.get("random_state", 42),
    )

    class_names = [k for k in event_id.keys()]
    print_report(results, class_names=class_names)

    # ── Step 5: Save ───────────────────────────────────────────────────
    logger.info("[5/5] Saving model and results...")
    out_dir = Path(output_dir) / f"subject_{subject_id:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Fit on all data and save model
    pipeline.fit(X, y)
    save_model(pipeline, str(out_dir / "model.pkl"))

    # Save results JSON
    results_to_save = {k: v for k, v in results.items()
                       if not isinstance(v, np.ndarray)}
    results_to_save["subject_id"] = subject_id
    results_to_save["dataset"] = dataset
    results_to_save["classifier"] = classifier
    results_to_save["n_epochs"] = len(y)
    results_to_save["elapsed_seconds"] = round(time.time() - t_start, 2)

    with open(out_dir / "results.json", "w") as f:
        json.dump(results_to_save, f, indent=2)
    logger.info(f"  Results saved to: {out_dir}/")

    return results_to_save


def main():
    parser = argparse.ArgumentParser(
        description="EEG-BCI Pipeline for CP Motor Intention Decoding"
    )
    parser.add_argument("--dataset", default="synthetic",
                        choices=["physionet", "synthetic", "bciciv2a"],
                        help="EEG dataset (default: synthetic)")
    parser.add_argument("--subject", nargs="+", type=int, default=[1],
                        help="Subject IDs (default: 1)")
    parser.add_argument("--classifier", default="lda",
                        choices=["lda", "svm", "rf", "ensemble"],
                        help="Classifier (default: lda)")
    parser.add_argument("--n_csp", type=int, default=6,
                        help="CSP components (default: 6)")
    parser.add_argument("--output_dir", default="outputs",
                        help="Output directory (default: outputs)")
    parser.add_argument("--config", default=None,
                        help="YAML config file path")
    args = parser.parse_args()

    config = {}
    if args.config and Path(args.config).exists():
        config = load_config(args.config)
        logger.info(f"Loaded config: {args.config}")

    all_results = []
    for subj in args.subject:
        try:
            result = run_single_subject(
                subject_id=subj,
                dataset=args.dataset,
                classifier=args.classifier,
                n_csp=args.n_csp,
                output_dir=args.output_dir,
                config=config,
            )
            all_results.append(result)
        except Exception as e:
            logger.error(f"Subject {subj} failed: {e}", exc_info=True)

    if len(all_results) > 1:
        mean_acc = np.mean([r["accuracy_mean"] for r in all_results])
        mean_kappa = np.mean([r["kappa_mean"] for r in all_results])
        logger.info(f"\n{'='*60}")
        logger.info(f"  MULTI-SUBJECT SUMMARY ({len(all_results)} subjects)")
        logger.info(f"  Mean Accuracy: {mean_acc*100:.2f}%")
        logger.info(f"  Mean Kappa:    {mean_kappa:.4f}")
        logger.info(f"{'='*60}\n")

    logger.info("Pipeline complete. 🎉")


if __name__ == "__main__":
    main()
