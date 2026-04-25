"""
loader.py — Multi-format EEG data loader
Supports: PhysioNet EEG MI DB, BCI Competition IV 2a/2b, synthetic data
"""

from __future__ import annotations
import os
import logging
from pathlib import Path
from typing import Optional

import mne
import numpy as np

logger = logging.getLogger(__name__)


def load_physionet(
    subject: int,
    runs: list[int] = [4, 8, 12],
    data_dir: Optional[str] = None,
    preload: bool = True,
) -> mne.io.Raw:
    """
    Load PhysioNet EEG Motor Movement/Imagery dataset.

    Subjects perform LEFT and RIGHT hand motor imagery.
    Runs 4, 8, 12 → left/right hand motor imagery.

    Args:
        subject: Subject ID (1–109)
        runs: PhysioNet run numbers [4, 8, 12] = left/right hand MI
        data_dir: Optional local path. If None, downloads via MNE.
        preload: Whether to load data into RAM immediately.

    Returns:
        mne.io.Raw concatenated across runs.
    """
    logger.info(f"Loading PhysioNet subject {subject:03d}, runs {runs}")

    if data_dir:
        # Load from local MNE data directory
        raw_fnames = mne.datasets.eegbci.load_data(
            subjects=subject,
            runs=runs,
            path=data_dir,
            update_path=False,  # Don't update path since files should already be there
            verbose=False,
        )
        raws = [mne.io.read_raw_edf(f, preload=preload, verbose=False) for f in raw_fnames]
        raw = mne.concatenate_raws(raws)
    else:
        # Auto-download via MNE's built-in downloader
        raw_fnames = mne.datasets.eegbci.load_data(
            subjects=subject,
            runs=runs,
            path=None,
            update_path=True,
            verbose=False,
        )
        raws = [mne.io.read_raw_edf(f, preload=preload, verbose=False) for f in raw_fnames]
        raw = mne.concatenate_raws(raws)

    # Standardize channel names for PhysioNet
    mne.datasets.eegbci.standardize(raw)
    raw.set_montage("standard_1005")

    logger.info(f"  Loaded: {raw.n_times} samples @ {raw.info['sfreq']} Hz, "
                f"{len(raw.ch_names)} channels")
    return raw


def load_bciciv2a(
    subject: int,
    session: str = "T",
    data_dir: str = "data/raw",
) -> mne.io.Raw:
    """
    Load BCI Competition IV Dataset 2a (GDF format).

    Classes: Left hand (769), Right hand (770), Feet (771), Tongue (772).
    Requires GDF files downloaded from: https://www.bbci.de/competition/iv/

    Args:
        subject: Subject ID (1–9)
        session: 'T' for training, 'E' for evaluation
        data_dir: Path to directory containing .gdf files

    Returns:
        mne.io.Raw object
    """
    fname = Path(data_dir) / f"A0{subject}{session}.gdf"
    if not fname.exists():
        raise FileNotFoundError(
            f"File not found: {fname}\n"
            "Download BCI Competition IV 2a from: https://www.bbci.de/competition/iv/\n"
            "Or run: python scripts/download_data.py --dataset bciciv2a"
        )

    logger.info(f"Loading BCI Competition IV 2a: Subject {subject}, Session {session}")
    raw = mne.io.read_raw_gdf(str(fname), preload=True, verbose=False)

    # Drop non-EEG channels (EOG)
    eog_chs = [ch for ch in raw.ch_names if "EOG" in ch.upper()]
    raw.drop_channels(eog_chs)

    montage = mne.channels.make_standard_montage("standard_1020")
    try:
        raw.set_montage(montage, on_missing="warn")
    except Exception as e:
        logger.warning(f"Montage setting failed: {e}")

    return raw


def load_bciciv2b(
    subject: int,
    session: int = 1,
    data_dir: str = "data/raw",
) -> mne.io.Raw:
    """
    Load BCI Competition IV Dataset 2b (GDF format).

    Classes: Left hand (769), Right hand (770). 3-electrode setup.

    Args:
        subject: Subject ID (1–9)
        session: Session number (1–5)
        data_dir: Path to GDF files
    """
    fname = Path(data_dir) / f"B0{subject}0{session}T.gdf"
    if not fname.exists():
        raise FileNotFoundError(f"File not found: {fname}")

    logger.info(f"Loading BCI Competition IV 2b: Subject {subject}, Session {session}")
    raw = mne.io.read_raw_gdf(str(fname), preload=True, verbose=False)
    return raw


def load_synthetic(
    n_channels: int = 22,
    sfreq: float = 250.0,
    duration_s: float = 300.0,
    n_trials_per_class: int = 50,
    seed: int = 42,
) -> mne.io.Raw:
    """
    Generate synthetic EEG with realistic motor imagery patterns.

    Simulates mu/beta ERD for left and right hand motor imagery
    to allow pipeline testing without real data.

    Args:
        n_channels: Number of EEG channels
        sfreq: Sampling frequency (Hz)
        duration_s: Total recording duration (seconds)
        n_trials_per_class: Number of trials per class
        seed: Random seed for reproducibility

    Returns:
        mne.io.Raw with synthetic MI data
    """
    rng = np.random.default_rng(seed)
    n_samples = int(sfreq * duration_s)

    logger.info(f"Generating synthetic EEG: {n_channels}ch, {sfreq}Hz, {duration_s}s")

    # Background EEG: 1/f noise + alpha oscillation
    times = np.arange(n_samples) / sfreq
    data = rng.standard_normal((n_channels, n_samples)) * 10e-6

    # Add alpha rhythm (10 Hz)
    alpha = 5e-6 * np.sin(2 * np.pi * 10 * times)
    data += alpha

    # Add beta rhythm (20 Hz, lower amplitude)
    beta = 2e-6 * np.sin(2 * np.pi * 20 * times)
    data += beta

    # Create channel info
    ch_names = [f"EEG{i+1:03d}" for i in range(n_channels)]
    ch_types = ["eeg"] * n_channels
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

    # Create event markers (every 6s: 2s rest, 4s MI)
    event_interval = int(sfreq * 6)
    events_left = []
    events_right = []

    offset = int(sfreq * 2)  # 2s rest before each trial
    for i in range(n_trials_per_class):
        t_left = offset + i * event_interval * 2
        t_right = t_left + event_interval
        if t_right + int(sfreq * 4) < n_samples:
            events_left.append(t_left)
            events_right.append(t_right)

    # Inject ERD patterns for motor imagery
    # Left hand MI → C4 (right hemisphere) ERD
    c4_idx = n_channels // 2 + 2
    # Right hand MI → C3 (left hemisphere) ERD
    c3_idx = n_channels // 2 - 2

    erd_dur = int(sfreq * 3)
    for t in events_left:
        erd_window = slice(t + int(sfreq * 0.5), t + int(sfreq * 0.5) + erd_dur)
        data[c4_idx, erd_window] *= 0.4  # Suppress mu rhythm
        data[c4_idx - 1, erd_window] *= 0.5

    for t in events_right:
        erd_window = slice(t + int(sfreq * 0.5), t + int(sfreq * 0.5) + erd_dur)
        data[c3_idx, erd_window] *= 0.4
        data[c3_idx + 1, erd_window] *= 0.5

    raw = mne.io.RawArray(data, info, verbose=False)

    # Build event array [sample, 0, event_id]
    events = []
    for t in events_left:
        if t < n_samples:
            events.append([t, 0, 1])  # 1 = left hand
    for t in events_right:
        if t < n_samples:
            events.append([t, 0, 2])  # 2 = right hand
    events = np.array(sorted(events, key=lambda x: x[0]))

    # Annotate raw with events
    if len(events) > 0:
        annotations = mne.annotations_from_events(
            events, sfreq, event_desc={1: "left_hand", 2: "right_hand"}
        )
        raw.set_annotations(annotations)

    logger.info(f"  Synthetic EEG created: {len(events_left)} left + {len(events_right)} right trials")
    return raw


def get_events(raw: mne.io.Raw, dataset: str = "physionet") -> tuple[np.ndarray, dict]:
    """
    Extract events from Raw object based on dataset type.

    Returns:
        events: np.ndarray shape (n_events, 3)
        event_id: dict mapping label → event code
    """
    if dataset == "physionet":
        events, event_id = mne.events_from_annotations(raw, verbose=False)
        # PhysioNet: T1=left hand, T2=right hand in runs 4,8,12
        event_id = {"left_hand": event_id.get("T1", 1),
                    "right_hand": event_id.get("T2", 2)}
    elif dataset in ("bciciv2a", "bciciv2b"):
        events, event_id = mne.events_from_annotations(raw, verbose=False)
        # GDF annotation codes
        event_id = {"left_hand": 769, "right_hand": 770}
        mask = np.isin(events[:, 2], [769, 770])
        events = events[mask]
    elif dataset == "synthetic":
        events, event_id = mne.events_from_annotations(raw, verbose=False)
        event_id = {"left_hand": 1, "right_hand": 2}
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    logger.info(f"  Events: {len(events)} total, classes: {list(event_id.keys())}")
    return events, event_id
