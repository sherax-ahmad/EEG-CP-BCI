"""
visualization.py — Topomap, ERD/ERS, and result visualization
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server use
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import mne

logger = logging.getLogger(__name__)


def plot_erd_ers_comparison(
    erd_results: dict,
    freq_bands: dict = None,
    output_path: Optional[str] = None,
    figsize: tuple = (16, 8),
) -> plt.Figure:
    """
    Plot ERD/ERS time courses for each condition and frequency band.

    Shows mu and beta ERD averaged over motor cortex channels (C3/C4),
    demonstrating the event-related power decrease during MI.

    Args:
        erd_results: Output from compute_erd_ers()
        freq_bands: Dict {band: (f_low, f_high)}
        output_path: Save path or None to show
        figsize: Figure dimensions

    Returns:
        Matplotlib Figure
    """
    if freq_bands is None:
        freq_bands = {"mu (8–12 Hz)": (8, 12), "beta (13–30 Hz)": (13, 30)}

    conditions = list(erd_results.keys())
    n_bands = len(freq_bands)
    n_conds = len(conditions)

    fig, axes = plt.subplots(
        n_bands, n_conds,
        figsize=figsize,
        sharey=True,
        sharex=True,
    )
    if n_bands == 1:
        axes = axes[np.newaxis, :]
    if n_conds == 1:
        axes = axes[:, np.newaxis]

    colors = {"left_hand": "#E63946", "right_hand": "#457B9D"}
    condition_labels = {"left_hand": "Left Hand MI", "right_hand": "Right Hand MI"}

    for b_idx, (band_name, (f_low, f_high)) in enumerate(freq_bands.items()):
        for c_idx, cond in enumerate(conditions):
            ax = axes[b_idx, c_idx]
            data = erd_results[cond]
            times = data["times"]
            erd_ers = data["erd_ers"]
            freqs = data["freqs"]

            # Average over band frequencies
            freq_mask = (freqs >= f_low) & (freqs <= f_high)
            erd_band = erd_ers[:, freq_mask, :].mean(axis=(0, 1))

            color = colors.get(cond, "#2D6A4F")
            ax.plot(times, erd_band, color=color, linewidth=2.0, label=cond)
            ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
            ax.axvspan(0.5, 3.5, alpha=0.1, color=color, label="MI window")
            ax.axvline(0, color="gray", linewidth=1.0, linestyle=":")

            ax.set_xlabel("Time (s)")
            ax.set_ylabel("ERD/ERS (%)")
            ax.set_title(f"{band_name}\n{condition_labels.get(cond, cond)}")
            ax.grid(True, alpha=0.3)
            ax.fill_between(times, erd_band, 0,
                            where=(erd_band < 0), alpha=0.3, color=color)
            ax.set_xlim(times[0], times[-1])

    fig.suptitle(
        "ERD/ERS Analysis — Motor Imagery (left vs. right hand)\n"
        "Negative values = desynchronization (ERD) → active motor planning",
        fontsize=13, fontweight="bold", y=1.02
    )
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"ERD/ERS plot saved: {output_path}")

    return fig


def plot_csp_patterns(
    csp_filters,
    info: mne.Info,
    n_components: int = 6,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot CSP activation patterns as EEG topomaps.

    The first patterns show RIGHT-class-dominant components,
    the last patterns show LEFT-class-dominant components.

    Args:
        csp_filters: Fitted MNE CSP object
        info: MNE Info with channel locations
        n_components: Number of components to plot
        output_path: Save path

    Returns:
        Matplotlib Figure
    """
    patterns = csp_filters.patterns_[:n_components]

    n_cols = min(6, n_components)
    n_rows = (n_components + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3.5))
    axes = np.array(axes).flatten()

    for idx in range(n_components):
        ax = axes[idx]
        pattern = patterns[idx]

        try:
            mne.viz.plot_topomap(
                pattern,
                info,
                axes=ax,
                show=False,
                contours=6,
                cmap="RdBu_r",
            )
            label = "Right-class" if idx < n_components // 2 else "Left-class"
            ax.set_title(f"CSP {idx+1}\n({label})", fontsize=9)
        except Exception as e:
            ax.text(0.5, 0.5, f"Pattern {idx+1}", ha="center", va="center")
            logger.warning(f"Topomap error for pattern {idx}: {e}")

    for idx in range(n_components, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("CSP Activation Patterns — Motor Cortex Lateralization",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"CSP patterns saved: {output_path}")

    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    accuracy: float,
    kappa: float,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot confusion matrix with metrics summary.

    Args:
        cm: Confusion matrix array (n_classes, n_classes)
        class_names: List of class label strings
        accuracy: Classification accuracy
        kappa: Cohen's kappa score
        output_path: Save path

    Returns:
        Matplotlib Figure
    """
    fig, (ax_cm, ax_metrics) = plt.subplots(1, 2, figsize=(10, 5),
                                              gridspec_kw={"width_ratios": [2, 1]})

    # Normalize confusion matrix
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    im = ax_cm.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax_cm, label="Proportion")

    tick_marks = np.arange(len(class_names))
    ax_cm.set_xticks(tick_marks)
    ax_cm.set_xticklabels(class_names, rotation=45, ha="right")
    ax_cm.set_yticks(tick_marks)
    ax_cm.set_yticklabels(class_names)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text_color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax_cm.text(j, i,
                       f"{cm[i, j]}\n({cm_norm[i, j]*100:.1f}%)",
                       ha="center", va="center", color=text_color, fontsize=10)

    ax_cm.set_xlabel("Predicted Label")
    ax_cm.set_ylabel("True Label")
    ax_cm.set_title("Confusion Matrix")

    # Metrics panel
    ax_metrics.axis("off")
    metrics_text = [
        ("Accuracy", f"{accuracy*100:.2f}%"),
        ("Cohen's κ", f"{kappa:.4f}"),
        ("Chance level", "50.00%"),
        ("Above chance", f"+{(accuracy-0.5)*100:.2f}%"),
    ]
    y_pos = 0.85
    for label, value in metrics_text:
        ax_metrics.text(0.1, y_pos, f"{label}:", fontsize=11, fontweight="bold",
                        transform=ax_metrics.transAxes)
        ax_metrics.text(0.6, y_pos, value, fontsize=11,
                        transform=ax_metrics.transAxes)
        y_pos -= 0.15

    kappa_interp = "Excellent" if kappa > 0.6 else "Moderate" if kappa > 0.4 else "Fair"
    ax_metrics.text(0.1, 0.2, f"BCI grade: {kappa_interp}", fontsize=10,
                    color="#2D6A4F" if kappa > 0.6 else "#E63946",
                    transform=ax_metrics.transAxes, fontweight="bold")

    fig.suptitle("BCI Classification Evaluation", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Confusion matrix saved: {output_path}")

    return fig


def plot_accuracy_per_subject(
    results_per_subject: dict[int, float],
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    Bar chart of per-subject classification accuracy.

    Args:
        results_per_subject: {subject_id: accuracy}
        output_path: Save path

    Returns:
        Matplotlib Figure
    """
    subjects = list(results_per_subject.keys())
    accuracies = [results_per_subject[s] * 100 for s in subjects]
    mean_acc = np.mean(accuracies)

    fig, ax = plt.subplots(figsize=(max(8, len(subjects) * 0.8), 5))

    colors = ["#2D6A4F" if a >= 70 else "#E63946" if a < 60 else "#F4A261"
              for a in accuracies]
    bars = ax.bar([f"S{s:02d}" for s in subjects], accuracies, color=colors,
                  edgecolor="white", linewidth=1.5)

    ax.axhline(y=50, color="gray", linestyle="--", linewidth=1.5, label="Chance (50%)")
    ax.axhline(y=mean_acc, color="navy", linestyle="-.", linewidth=2.0,
               label=f"Mean ({mean_acc:.1f}%)")
    ax.axhline(y=70, color="#2D6A4F", linestyle=":", linewidth=1.5,
               label="BCI threshold (70%)", alpha=0.7)

    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{acc:.1f}%", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Subject")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Per-Subject BCI Classification Accuracy\n(CSP + LDA, 10-fold CV)")
    ax.set_ylim(40, 105)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Per-subject accuracy plot saved: {output_path}")

    return fig
