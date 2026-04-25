"""
epocher.py — Trial segmentation (epoching)
Converts continuous EEG into fixed-length motor imagery trials.
"""

from __future__ import annotations
import logging
from typing import Optional

import mne
import numpy as np

logger = logging.getLogger(__name__)


def create_epochs(
    raw: mne.io.Raw,
    events: np.ndarray,
    event_id: dict[str, int],
    tmin: float = -1.0,
    tmax: float = 4.0,
    baseline: tuple[float, float] = (-1.0, 0.0),
    picks: str = "eeg",
    reject_peak_to_peak: Optional[float] = 150e-6,
    verbose: bool = False,
) -> mne.Epochs:
    """
    Segment continuous EEG into motor imagery epochs.

    Standard MI epoch: -1s to +4s around event marker.
    Baseline correction uses the pre-stimulus period (-1 to 0s).

    The MI signal of interest is typically 0.5–3.5s post-cue,
    capturing the mu/beta ERD that builds after movement onset.

    Args:
        raw: Preprocessed MNE Raw
        events: Event array shape (n_events, 3)
        event_id: Dict {label: event_code}
        tmin: Epoch start (s relative to event)
        tmax: Epoch end (s relative to event)
        baseline: Baseline correction window
        picks: Channel selection
        reject_peak_to_peak: Artifact rejection threshold (Volts)
        verbose: MNE verbosity

    Returns:
        mne.Epochs object
    """
    reject_criteria = None
    if reject_peak_to_peak:
        reject_criteria = {"eeg": reject_peak_to_peak}

    logger.info(
        f"Creating epochs: {tmin}–{tmax}s, baseline {baseline}, "
        f"reject={reject_peak_to_peak}"
    )

    epochs = mne.Epochs(
        raw,
        events=events,
        event_id=event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=baseline,
        picks=picks,
        reject=reject_criteria,
        preload=True,
        verbose=verbose,
    )

    n_original = sum(e[2] in event_id.values() for e in events)
    n_kept = len(epochs)
    n_rejected = n_original - n_kept

    logger.info(
        f"  Epochs: {n_kept} kept, {n_rejected} rejected "
        f"({100*n_rejected/max(n_original,1):.1f}% rejected)"
    )

    for label in event_id:
        count = len(epochs[label]) if label in epochs.event_id else 0
        logger.info(f"    {label}: {count} trials")

    return epochs


def get_motor_imagery_data(
    epochs: mne.Epochs,
    tmin_mi: float = 0.5,
    tmax_mi: float = 3.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract the motor imagery window from epochs.

    Crops epochs to the active MI period (0.5–3.5s post-cue),
    avoiding the early visual/cognitive response and the
    post-movement ERS rebound.

    Args:
        epochs: Epoched EEG
        tmin_mi: Start of MI window (s)
        tmax_mi: End of MI window (s)

    Returns:
        X: np.ndarray shape (n_epochs, n_channels, n_times)
        y: np.ndarray shape (n_epochs,) with class labels (0, 1)
    """
    epochs_mi = epochs.copy().crop(tmin=tmin_mi, tmax=tmax_mi)

    X = epochs_mi.get_data()  # (n_epochs, n_channels, n_times)
    labels = epochs_mi.events[:, 2]

    # Map event codes to 0-indexed labels
    event_id = epochs_mi.event_id
    label_names = list(event_id.keys())
    label_map = {code: i for i, (name, code) in enumerate(event_id.items())}
    y = np.array([label_map[code] for code in labels])

    logger.info(f"MI data shape: {X.shape}, labels: {np.bincount(y)}")
    return X, y


def balance_classes(
    X: np.ndarray,
    y: np.ndarray,
    strategy: str = "undersample",
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Balance class distribution for fair classifier evaluation.

    Args:
        X: Feature array
        y: Label array
        strategy: 'undersample' | 'oversample'
        seed: Random seed

    Returns:
        Balanced X, y arrays
    """
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    min_count = counts.min()

    if strategy == "undersample":
        indices = []
        for cls in classes:
            cls_idx = np.where(y == cls)[0]
            sampled = rng.choice(cls_idx, size=min_count, replace=False)
            indices.append(sampled)
        idx = np.concatenate(indices)
        idx = rng.permutation(idx)
        logger.info(f"Undersampling to {min_count} trials/class")
        return X[idx], y[idx]

    return X, y
