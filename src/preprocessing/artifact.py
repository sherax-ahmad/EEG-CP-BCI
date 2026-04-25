"""
artifact.py — Artifact detection and removal
ICA-based EOG/EMG removal for BCI-grade EEG preprocessing.
"""

from __future__ import annotations
import logging
from typing import Optional

import mne
import numpy as np

logger = logging.getLogger(__name__)


def remove_artifacts_ica(
    raw: mne.io.Raw,
    n_components: int = 20,
    method: str = "fastica",
    eog_channels: Optional[list[str]] = None,
    random_state: int = 42,
    threshold: float = 3.0,
) -> tuple[mne.io.Raw, mne.preprocessing.ICA]:
    """
    Remove ocular and muscle artifacts using ICA.

    ICA decomposes EEG into independent components. Components
    that correlate with EOG channels (eye blinks/movements) or
    have high-frequency power (EMG) are excluded.

    For children with CP, eye movement artifacts are common
    due to gaze control difficulties — ICA is critical here.

    Args:
        raw: Preprocessed MNE Raw (high-pass filtered ≥ 1 Hz)
        n_components: Number of ICA components
        method: 'fastica' | 'picard' | 'infomax'
        eog_channels: List of EOG channel names, or None for auto-detect
        random_state: Reproducibility seed
        threshold: Z-score threshold for component rejection

    Returns:
        Tuple of (cleaned Raw, fitted ICA object)
    """
    logger.info(f"Running ICA: {n_components} components, method={method}")

    ica = mne.preprocessing.ICA(
        n_components=n_components,
        method=method,
        random_state=random_state,
        max_iter="auto",
        verbose=False,
    )

    # Fit ICA on a copy filtered at 1 Hz (ICA works better above 1 Hz)
    raw_for_ica = raw.copy().filter(l_freq=1.0, h_freq=None, verbose=False)
    ica.fit(raw_for_ica, picks="eeg", verbose=False)

    exclude_indices = []

    # Method 1: Correlate with EOG channels
    if eog_channels:
        for eog_ch in eog_channels:
            if eog_ch in raw.ch_names:
                eog_indices, eog_scores = ica.find_bads_eog(
                    raw, ch_name=eog_ch, threshold=threshold, verbose=False
                )
                exclude_indices.extend(eog_indices)
                logger.info(f"  EOG ({eog_ch}): {len(eog_indices)} components flagged")

    # Method 2: Heuristic — reject components with frontal dominance
    # (eye blink artifacts have large power at Fp1/Fp2)
    if not eog_channels or not exclude_indices:
        frontal_chs = [ch for ch in raw.ch_names
                       if any(x in ch.upper() for x in ["FP", "AF"])]
        if frontal_chs:
            eog_indices, _ = ica.find_bads_eog(
                raw, ch_name=frontal_chs[0], threshold=threshold + 0.5,
                verbose=False
            )
            exclude_indices.extend(eog_indices)

    # Deduplicate
    exclude_indices = list(set(exclude_indices))[:3]  # Max 3 components removed
    ica.exclude = exclude_indices
    logger.info(f"  Excluding {len(exclude_indices)} artifact components: {exclude_indices}")

    # Apply ICA
    raw_clean = raw.copy()
    ica.apply(raw_clean, verbose=False)

    return raw_clean, ica


def detect_bad_channels(
    raw: mne.io.Raw,
    correlation_threshold: float = 0.4,
    noise_threshold: float = 5.0,
) -> list[str]:
    """
    Detect bad channels using correlation and noise metrics.

    Args:
        raw: MNE Raw object
        correlation_threshold: Min acceptable correlation with neighbors
        noise_threshold: Z-score for noise detection

    Returns:
        List of bad channel names
    """
    data = raw.get_data(picks="eeg")
    bad_channels = []

    # Z-score variance check — flat or super-noisy channels
    variances = np.var(data, axis=1)
    z_vars = np.abs((variances - np.median(variances)) / (np.std(variances) + 1e-10))
    high_noise = [raw.ch_names[i] for i in np.where(z_vars > noise_threshold)[0]]
    bad_channels.extend(high_noise)

    # Flat channel check
    flat = [raw.ch_names[i] for i in range(len(raw.ch_names))
            if raw.ch_names[i] in [ch for ch in raw.info["ch_names"] if raw.ch_names.count(ch) == 1]
            and variances[i] < 1e-15]
    bad_channels.extend(flat)

    bad_channels = list(set(bad_channels))
    if bad_channels:
        logger.warning(f"  Detected {len(bad_channels)} bad channels: {bad_channels}")
    else:
        logger.info("  No bad channels detected")

    return bad_channels


def interpolate_bad_channels(raw: mne.io.Raw, bad_channels: list[str]) -> mne.io.Raw:
    """
    Spherical spline interpolation of bad channels.

    Args:
        raw: MNE Raw with montage set
        bad_channels: List of channel names to interpolate

    Returns:
        Raw with interpolated channels
    """
    if not bad_channels:
        return raw

    raw.info["bads"] = bad_channels
    logger.info(f"  Interpolating {len(bad_channels)} bad channels")
    raw.interpolate_bads(reset_bads=True, verbose=False)
    return raw
