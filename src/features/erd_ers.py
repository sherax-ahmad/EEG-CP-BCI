"""
erd_ers.py — Event-Related Desynchronization / Synchronization Analysis

ERD/ERS quantifies power changes in oscillatory EEG relative to baseline.
During motor imagery:
  - ERD (negative): Power DECREASE in mu/beta → active motor processing
  - ERS (positive): Power INCREASE (rebound) after movement cessation

Formula: ERD/ERS (%) = (A - R) / R × 100
  where A = epoch power, R = reference (baseline) power

Reference:
    Pfurtscheller & Lopes da Silva (1999). Event-related EEG/MEG synchronization
    and desynchronization. Clin Neurophysiol, 110(11):1842-1857.
"""

from __future__ import annotations
import logging
from typing import Optional

import numpy as np
import mne

logger = logging.getLogger(__name__)


def compute_erd_ers(
    epochs: mne.Epochs,
    freq_bands: dict[str, tuple[float, float]] = None,
    tmin_baseline: float = -1.0,
    tmax_baseline: float = 0.0,
    tmin_active: float = 0.5,
    tmax_active: float = 3.5,
    method: str = "morlet",
) -> dict[str, np.ndarray]:
    """
    Compute ERD/ERS for each frequency band and condition.

    Args:
        epochs: Epoched EEG with event_id labels
        freq_bands: Dict {band_name: (f_low, f_high)}
        tmin_baseline: Baseline start (s)
        tmax_baseline: Baseline end (s)
        tmin_active: Active MI window start (s)
        tmax_active: Active MI window end (s)
        method: TFR method: 'morlet' | 'multitaper'

    Returns:
        Dict mapping condition → erd_ers array (n_channels, n_freqs, n_times)
    """
    if freq_bands is None:
        freq_bands = {"mu": (8, 12), "beta": (13, 30)}

    # Collect all frequencies
    all_freqs = []
    for low, high in freq_bands.values():
        all_freqs.extend(np.arange(low, high + 1, 1.0))
    freqs = np.array(sorted(set(all_freqs)))

    results = {}
    conditions = list(epochs.event_id.keys())

    for cond in conditions:
        if cond not in epochs.event_id:
            continue

        epochs_cond = epochs[cond]
        logger.info(f"Computing ERD/ERS for '{cond}': {len(epochs_cond)} epochs")

        # Time-frequency analysis (Morlet wavelets)
        n_cycles = freqs / 2.0
        n_cycles = np.clip(n_cycles, 3, 7)

        if method == "morlet":
            power = mne.time_frequency.tfr_morlet(
                epochs_cond,
                freqs=freqs,
                n_cycles=n_cycles,
                use_fft=True,
                return_itc=False,
                average=False,
                verbose=False,
            )
        else:
            power = mne.time_frequency.tfr_multitaper(
                epochs_cond,
                freqs=freqs,
                n_cycles=n_cycles,
                return_itc=False,
                average=False,
                verbose=False,
            )

        # power.data shape: (n_epochs, n_channels, n_freqs, n_times)
        power_data = power.data  # µV² / Hz

        times = power.times
        baseline_mask = (times >= tmin_baseline) & (times <= tmax_baseline)
        active_mask = (times >= tmin_active) & (times <= tmax_active)

        # Baseline mean (R)
        R = power_data[:, :, :, baseline_mask].mean(axis=(0, 3), keepdims=True)
        R = R.mean(axis=3, keepdims=True)

        # Average power across epochs
        A_mean = power_data.mean(axis=0)  # (n_ch, n_freq, n_times)

        # ERD/ERS formula
        R_mean = power_data[:, :, :, baseline_mask].mean(axis=(0, 3), keepdims=False)
        R_mean = R_mean.mean(axis=2)  # (n_ch, n_freq)

        erd_ers = np.zeros_like(A_mean)
        for t_idx in range(A_mean.shape[2]):
            erd_ers[:, :, t_idx] = (
                (A_mean[:, :, t_idx] - R_mean) / (R_mean + 1e-10) * 100
            )

        results[cond] = {
            "erd_ers": erd_ers,
            "times": times,
            "freqs": freqs,
            "power_mean": A_mean,
            "baseline_power": R_mean,
        }
        logger.info(f"  ERD range: [{erd_ers[:, :, active_mask].min():.1f}%, "
                    f"{erd_ers[:, :, active_mask].max():.1f}%]")

    return results


def extract_band_power_features(
    X: np.ndarray,
    sfreq: float,
    freq_bands: dict[str, tuple[float, float]] = None,
) -> np.ndarray:
    """
    Extract bandpower features from EEG epochs using Welch PSD.

    Args:
        X: EEG array shape (n_epochs, n_channels, n_times)
        sfreq: Sampling frequency
        freq_bands: Dict {band: (f_low, f_high)}

    Returns:
        Features shape (n_epochs, n_channels × n_bands)
    """
    from scipy.signal import welch

    if freq_bands is None:
        freq_bands = {
            "mu": (8, 12),
            "beta_low": (13, 20),
            "beta_high": (20, 30),
        }

    n_epochs, n_channels, n_times = X.shape
    n_bands = len(freq_bands)
    features = np.zeros((n_epochs, n_channels * n_bands))

    nperseg = min(n_times, int(sfreq))  # 1s segments

    for ep_idx in range(n_epochs):
        band_feats = []
        for ch_idx in range(n_channels):
            ch_data = X[ep_idx, ch_idx]
            freqs_psd, psd = welch(ch_data, fs=sfreq, nperseg=nperseg)
            ch_bands = []
            for band_name, (f_low, f_high) in freq_bands.items():
                mask = (freqs_psd >= f_low) & (freqs_psd <= f_high)
                band_power = np.trapz(psd[mask], freqs_psd[mask])
                ch_bands.append(np.log(band_power + 1e-10))  # log for Gaussianity
            band_feats.extend(ch_bands)
        features[ep_idx] = band_feats

    return features


def compute_erp(
    epochs: mne.Epochs,
    picks: str = "eeg",
) -> dict[str, np.ndarray]:
    """
    Compute Event-Related Potentials (ERP) for each condition.

    Returns time-averaged ERP for visualization.

    Args:
        epochs: Epoched EEG
        picks: Channel selection

    Returns:
        Dict {condition: evoked_array shape (n_channels, n_times)}
    """
    erp_dict = {}
    for cond in epochs.event_id:
        if cond in epochs.event_id:
            evoked = epochs[cond].average(picks=picks)
            erp_dict[cond] = {
                "data": evoked.data,
                "times": evoked.times,
                "ch_names": evoked.ch_names,
            }
            logger.info(f"ERP '{cond}': peak {evoked.data.max()*1e6:.2f} µV")
    return erp_dict
