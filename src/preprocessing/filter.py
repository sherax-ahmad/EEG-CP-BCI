"""
filter.py — EEG signal filtering utilities
Bandpass, notch, and re-referencing for BCI pipelines.
"""

from __future__ import annotations
import logging
from typing import Optional

import mne
import numpy as np

logger = logging.getLogger(__name__)


def apply_bandpass(
    raw: mne.io.Raw,
    l_freq: float = 1.0,
    h_freq: float = 40.0,
    method: str = "iir",
) -> mne.io.Raw:
    """
    Apply bandpass filter to EEG data.

    For MI-BCI, we typically filter 1–40 Hz to capture
    mu (8–12 Hz) and beta (13–30 Hz) rhythms while removing
    slow drifts and high-frequency noise.

    Args:
        raw: MNE Raw object
        l_freq: Lower cutoff (Hz). Set to None for lowpass only.
        h_freq: Upper cutoff (Hz). Set to None for highpass only.
        method: 'iir' or 'fir'

    Returns:
        Filtered Raw (in-place modification)
    """
    logger.info(f"Bandpass filter: {l_freq}–{h_freq} Hz ({method})")
    raw.filter(
        l_freq=l_freq,
        h_freq=h_freq,
        method=method,
        picks="eeg",
        verbose=False,
    )
    return raw


def apply_notch(
    raw: mne.io.Raw,
    freqs: float | list[float] = 50.0,
) -> mne.io.Raw:
    """
    Apply notch filter to remove power line noise.

    Use 50 Hz for Europe/Asia, 60 Hz for North America.
    Also filters harmonics automatically.

    Args:
        raw: MNE Raw object
        freqs: Power line frequency. Can be list for multiple.

    Returns:
        Filtered Raw (in-place)
    """
    sfreq = raw.info["sfreq"]
    if isinstance(freqs, (int, float)):
        freqs = [freqs]

    # Add harmonics up to Nyquist
    notch_freqs = []
    for f in freqs:
        harmonic = f
        while harmonic < sfreq / 2:
            notch_freqs.append(harmonic)
            harmonic += f

    logger.info(f"Notch filter at: {notch_freqs} Hz")
    raw.notch_filter(freqs=notch_freqs, picks="eeg", verbose=False)
    return raw


def resample(raw: mne.io.Raw, sfreq: float = 160.0) -> mne.io.Raw:
    """
    Resample EEG to target frequency.

    For motor imagery BCI, 160–250 Hz is typically sufficient.

    Args:
        raw: MNE Raw object
        sfreq: Target sampling frequency

    Returns:
        Resampled Raw
    """
    orig_sfreq = raw.info["sfreq"]
    if orig_sfreq == sfreq:
        logger.info(f"Already at {sfreq} Hz, skipping resample")
        return raw

    logger.info(f"Resampling {orig_sfreq} → {sfreq} Hz")
    raw.resample(sfreq, npad="auto", verbose=False)
    return raw


def set_reference(
    raw: mne.io.Raw,
    ref: str = "average",
    ref_channels: Optional[list[str]] = None,
) -> mne.io.Raw:
    """
    Set EEG reference.

    Args:
        raw: MNE Raw
        ref: 'average' | 'mastoid' | 'REST' | channel name
        ref_channels: Specific channels for 'mastoid' reference

    Returns:
        Re-referenced Raw
    """
    if ref == "average":
        logger.info("Setting average reference")
        raw.set_eeg_reference("average", projection=False, verbose=False)
    elif ref == "mastoid" and ref_channels:
        logger.info(f"Setting mastoid reference: {ref_channels}")
        raw.set_eeg_reference(ref_channels=ref_channels, verbose=False)
    else:
        logger.info(f"Using reference: {ref}")
        raw.set_eeg_reference(ref_channels=ref, verbose=False)
    return raw


def preprocess_raw(
    raw: mne.io.Raw,
    l_freq: float = 1.0,
    h_freq: float = 40.0,
    notch_freq: float = 50.0,
    sfreq_resample: Optional[float] = None,
    reference: str = "average",
) -> mne.io.Raw:
    """
    Full preprocessing chain: resample → notch → bandpass → reference.

    Args:
        raw: MNE Raw object (loaded with preload=True)
        l_freq: Bandpass low cutoff
        h_freq: Bandpass high cutoff
        notch_freq: Notch filter frequency
        sfreq_resample: Target sample rate, or None to skip
        reference: EEG reference scheme

    Returns:
        Preprocessed MNE Raw
    """
    logger.info("=== Preprocessing Pipeline ===")

    raw.load_data(verbose=False)

    if sfreq_resample and sfreq_resample != raw.info["sfreq"]:
        raw = resample(raw, sfreq_resample)

    raw = apply_notch(raw, freqs=notch_freq)
    raw = apply_bandpass(raw, l_freq=l_freq, h_freq=h_freq)
    raw = set_reference(raw, ref=reference)

    logger.info("=== Preprocessing Complete ===")
    return raw
