#!/usr/bin/env python3
"""
visualize.py — Generate all EEG-BCI visualization plots.

Produces and saves the following plots:
  1. ERD/ERS time-frequency map (mu + beta bands, per condition)
  2. CSP activation topomaps (spatial filters)
  3. Confusion matrix + metrics panel
  4. Per-subject accuracy bar chart (if multiple subjects trained)
  5. Raw EEG + PSD overview

All plots are saved to: outputs/plots/

Usage:
    # Using synthetic data (no download needed):
    python scripts/visualize.py --dataset synthetic

    # Using PhysioNet subject 1 (must be downloaded first):
    python scripts/visualize.py --dataset physionet --subject 1

    # Load a previously saved model for the confusion matrix:
    python scripts/visualize.py --dataset physionet --subject 1 --model_path outputs/subject_001/model.pkl

    # Multiple subjects for the per-subject accuracy plot:
    python scripts/visualize.py --dataset physionet --subject 1 2 3 --multi_subject
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.preprocessing.loader import load_physionet, load_synthetic, get_events
from src.preprocessing.filter import preprocess_raw
from src.preprocessing.epocher import create_epochs, get_motor_imagery_data, balance_classes
from src.models.classifier import build_pipeline, load_model
from src.models.cross_validate import run_cross_validation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def load_and_preprocess(subject: int, dataset: str):
    """Load, preprocess, and epoch EEG. Returns (epochs, raw, event_id)."""
    logger.info(f"Loading {dataset} subject {subject}...")
    if dataset == "physionet":
        raw = load_physionet(subject=subject, runs=[4, 8, 12], data_dir="data/raw")
    else:
        raw = load_synthetic(n_channels=22, sfreq=250, duration_s=360,
                             n_trials_per_class=60, seed=subject)

    raw = preprocess_raw(raw, l_freq=1.0, h_freq=40.0, notch_freq=50.0,
                         reference="average")
    events, event_id = get_events(raw, dataset=dataset)
    epochs = create_epochs(raw, events, event_id,
                           tmin=-1.0, tmax=4.0,
                           baseline=(-1.0, 0.0),
                           reject_peak_to_peak=150e-6 if dataset == "synthetic" else None)
    return epochs, raw, event_id


def plot_raw_overview(raw, out_dir: Path, subject: int) -> None:
    """Plot raw EEG signal snippet and PSD."""
    logger.info("Plotting raw EEG overview + PSD...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7))

    # 10-second snippet
    sfreq = raw.info["sfreq"]
    data, times = raw[:8, int(10 * sfreq):int(20 * sfreq)]  # First 8 channels, 10–20s

    offset = 0
    for ch_idx in range(min(8, data.shape[0])):
        ax1.plot(times, data[ch_idx] * 1e6 + offset,
                 linewidth=0.7, alpha=0.85)
        ax1.text(times[0] - 0.1, offset,
                 raw.ch_names[ch_idx], fontsize=7, ha="right", va="center")
        offset += 40

    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude (µV + offset)")
    ax1.set_title(f"Subject {subject:03d} — Raw EEG (first 8 channels, 10–20s)")
    ax1.set_xlim(times[0], times[-1])
    ax1.grid(True, alpha=0.2)

    # PSD
    spectrum = raw.compute_psd(fmax=50, picks="eeg", verbose=False)
    psds = spectrum.get_data()
    freqs = spectrum.freqs
    mean_psd = psds.mean(axis=0) * 1e12  # Convert to µV²/Hz

    ax2.semilogy(freqs, mean_psd, color="#2D6A4F", linewidth=1.5)
    ax2.axvspan(8, 12, alpha=0.15, color="orange", label="Mu (8–12 Hz)")
    ax2.axvspan(13, 30, alpha=0.15, color="royalblue", label="Beta (13–30 Hz)")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Power (µV²/Hz)")
    ax2.set_title("Mean Power Spectral Density (all EEG channels)")
    ax2.legend()
    ax2.grid(True, which="both", alpha=0.2)

    plt.tight_layout()
    out_path = out_dir / f"01_raw_overview_S{subject:03d}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {out_path}")


def plot_erd_ers(epochs, out_dir: Path, subject: int) -> None:
    """Plot ERD/ERS time courses per band and condition."""
    logger.info("Computing ERD/ERS (this may take 1–2 min)...")

    conditions = list(epochs.event_id.keys())[:2]
    freq_bands = {"Mu (8–12 Hz)": (8, 12), "Beta (13–30 Hz)": (13, 30)}

    fig, axes = plt.subplots(len(freq_bands), len(conditions),
                             figsize=(14, 8), sharey=False, sharex=True)

    colors_map = {cond: c for cond, c in
                  zip(conditions, ["#E63946", "#457B9D"])}

    for b_idx, (band_name, (f_low, f_high)) in enumerate(freq_bands.items()):
        for c_idx, cond in enumerate(conditions):
            ax = axes[b_idx][c_idx]

            try:
                import mne
                freqs = np.arange(f_low, f_high + 0.5, 0.5)
                n_cycles = freqs / 2.0

                power = mne.time_frequency.tfr_morlet(
                    epochs[cond],
                    freqs=freqs,
                    n_cycles=n_cycles,
                    use_fft=True,
                    return_itc=False,
                    average=True,
                    verbose=False,
                )

                times = power.times
                power_data = power.data  # (n_ch, n_freq, n_times)

                baseline_mask = (times >= -1.0) & (times <= 0.0)
                R = power_data[:, :, baseline_mask].mean(axis=2, keepdims=True)

                erd_ers = (power_data - R) / (R + 1e-12) * 100
                erd_mean = erd_ers.mean(axis=(0, 1))  # avg over channels & freqs

                color = colors_map.get(cond, "#333333")
                ax.plot(times, erd_mean, color=color, linewidth=2.0)
                ax.fill_between(times, erd_mean, 0,
                                where=(erd_mean < 0), alpha=0.3, color=color)
                ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
                ax.axvline(0, color="gray", linewidth=1.0, linestyle=":", label="Cue onset")
                ax.axvspan(0.5, 3.5, alpha=0.08, color=color, label="MI window")

                label = cond.replace("_", " ").title()
                ax.set_title(f"{band_name}\n{label}", fontsize=10)
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("ERD/ERS (%)")
                ax.grid(True, alpha=0.25)
                ax.set_xlim(times[0], times[-1])

            except Exception as e:
                ax.text(0.5, 0.5, f"Error:\n{str(e)[:60]}",
                        ha="center", va="center", transform=ax.transAxes, fontsize=8)
                logger.warning(f"ERD/ERS failed for {cond}/{band_name}: {e}")

    fig.suptitle(
        f"Subject {subject:03d} — ERD/ERS Analysis (Motor Imagery)\n"
        "Negative = desynchronization (ERD) = active motor planning",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    out_path = out_dir / f"02_erd_ers_S{subject:03d}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {out_path}")


def plot_csp_topomaps(epochs, event_id, out_dir: Path, subject: int) -> None:
    """Fit CSP and plot spatial filter topomaps."""
    logger.info("Fitting CSP and plotting topomaps...")
    import mne
    from mne.decoding import CSP

    X, y = get_motor_imagery_data(epochs, tmin_mi=0.5, tmax_mi=3.5)
    if len(X) < 6:
        logger.warning("  Too few epochs for CSP topomap — skipping")
        return

    csp = CSP(n_components=6, reg="ledoit_wolf", log=True)
    csp.fit(X, y)

    info = epochs.info
    patterns = csp.patterns_[:6]
    n_patterns = len(patterns)

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    axes = axes.flatten()

    for idx in range(n_patterns):
        ax = axes[idx]
        try:
            mne.viz.plot_topomap(
                patterns[idx],
                info,
                axes=ax,
                show=False,
                contours=6,
                cmap="RdBu_r",
                vlim=(None, None),
            )
            side = "Right-dominant" if idx < 3 else "Left-dominant"
            ax.set_title(f"CSP Filter {idx + 1}\n({side})", fontsize=9)
        except Exception as e:
            ax.text(0.5, 0.5, f"Pattern {idx+1}", ha="center", va="center",
                    transform=ax.transAxes)
            logger.warning(f"  Topomap {idx} failed: {e}")

    cond_labels = " vs ".join(k.replace("_", " ").title() for k in event_id.keys())
    fig.suptitle(
        f"Subject {subject:03d} — CSP Spatial Filters\n"
        f"{cond_labels} Motor Imagery",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    out_path = out_dir / f"03_csp_topomaps_S{subject:03d}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {out_path}")


def plot_classification_results(epochs, event_id, out_dir: Path,
                                 subject: int, classifier: str = "lda") -> None:
    """Train classifier, run CV, plot confusion matrix and accuracy."""
    logger.info(f"Running {classifier.upper()} cross-validation for confusion matrix...")

    X, y = get_motor_imagery_data(epochs, tmin_mi=0.5, tmax_mi=3.5)
    X, y = balance_classes(X, y)

    if len(X) < 10:
        logger.warning("  Too few epochs for classification — skipping")
        return

    pipeline = build_pipeline(classifier, n_csp_components=6)
    results = run_cross_validation(pipeline, X, y, strategy="stratified", n_splits=10)

    acc = results["accuracy_mean"]
    kappa = results["kappa_mean"]
    cm = np.array(results["confusion_matrix"])
    class_names = list(event_id.keys())

    fig, (ax_cm, ax_metrics) = plt.subplots(1, 2, figsize=(11, 5),
                                             gridspec_kw={"width_ratios": [2, 1]})

    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)
    im = ax_cm.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax_cm, label="Proportion")

    ticks = np.arange(len(class_names))
    ax_cm.set_xticks(ticks)
    ax_cm.set_xticklabels([c.replace("_", " ").title() for c in class_names], rotation=30, ha="right")
    ax_cm.set_yticks(ticks)
    ax_cm.set_yticklabels([c.replace("_", " ").title() for c in class_names])

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            tc = "white" if cm_norm[i, j] > 0.5 else "black"
            ax_cm.text(j, i, f"{cm[i, j]}\n({cm_norm[i,j]*100:.1f}%)",
                       ha="center", va="center", color=tc, fontsize=11)

    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("True")
    ax_cm.set_title(f"Confusion Matrix — {classifier.upper()} + CSP\n10-fold Stratified CV")

    ax_metrics.axis("off")
    rows = [
        ("Accuracy",    f"{acc*100:.2f}%"),
        ("Cohen's κ",   f"{kappa:.4f}"),
        ("ROC-AUC",     f"{results['roc_auc_mean']:.4f}"),
        ("Chance level","50.00%"),
        ("Δ Chance",    f"+{(acc-0.5)*100:.2f}%"),
        ("n epochs",    str(len(y))),
    ]
    grade = ("Excellent ✅" if kappa > 0.6 else
             "Moderate ⚠️" if kappa > 0.4 else
             "Fair ❌")
    rows.append(("BCI Grade", grade))

    y_pos = 0.90
    for label, val in rows:
        ax_metrics.text(0.05, y_pos, f"{label}:", fontsize=11, fontweight="bold",
                        transform=ax_metrics.transAxes)
        ax_metrics.text(0.62, y_pos, val, fontsize=11,
                        transform=ax_metrics.transAxes)
        y_pos -= 0.12

    fig.suptitle(f"Subject {subject:03d} — Classification Results", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_path = out_dir / f"04_classification_S{subject:03d}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {out_path}")
    logger.info(f"  Accuracy={acc*100:.2f}%  Kappa={kappa:.4f}  AUC={results['roc_auc_mean']:.4f}")


def plot_multi_subject_accuracy(subjects: list[int], dataset: str,
                                out_dir: Path) -> None:
    """Load saved results JSONs and plot per-subject accuracy bar chart."""
    logger.info("Plotting multi-subject accuracy...")

    accuracies = {}
    for s in subjects:
        json_path = Path("outputs") / f"subject_{s:03d}" / "results.json"
        if json_path.exists():
            with open(json_path) as f:
                data = json.load(f)
            accuracies[s] = data.get("accuracy_mean", 0.0)
        else:
            logger.warning(f"  No results.json for subject {s} — train first.")

    if not accuracies:
        logger.warning("  No saved results found. Run train_model.py for each subject first.")
        return

    fig, ax = plt.subplots(figsize=(max(8, len(accuracies) * 1.1), 5))
    subj_ids = list(accuracies.keys())
    accs = [accuracies[s] * 100 for s in subj_ids]
    mean_acc = np.mean(accs)

    colors = ["#2D6A4F" if a >= 70 else "#F4A261" if a >= 60 else "#E63946"
              for a in accs]
    bars = ax.bar([f"S{s:02d}" for s in subj_ids], accs, color=colors,
                  edgecolor="white", linewidth=1.5, width=0.6)

    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{acc:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.axhline(50, color="gray", linestyle="--", linewidth=1.5, label="Chance (50%)", alpha=0.8)
    ax.axhline(mean_acc, color="navy", linestyle="-.", linewidth=2.0,
               label=f"Mean ({mean_acc:.1f}%)")
    ax.axhline(70, color="#2D6A4F", linestyle=":", linewidth=1.5,
               label="BCI threshold (70%)", alpha=0.7)

    ax.set_xlabel("Subject")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"Per-Subject BCI Accuracy — CSP + LDA, 10-fold CV\n{dataset.title()} dataset")
    ax.set_ylim(40, 108)
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    out_path = out_dir / "05_multi_subject_accuracy.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate EEG-BCI visualization plots")
    parser.add_argument("--subject", nargs="+", type=int, default=[1],
                        help="Subject ID(s) (default: 1)")
    parser.add_argument("--dataset", default="synthetic",
                        choices=["physionet", "synthetic"],
                        help="Dataset (default: synthetic)")
    parser.add_argument("--classifier", default="lda",
                        choices=["lda", "svm", "rf"],
                        help="Classifier for confusion matrix (default: lda)")
    parser.add_argument("--output_dir", default="outputs/plots",
                        help="Directory to save plots (default: outputs/plots)")
    parser.add_argument("--multi_subject", action="store_true",
                        help="Generate multi-subject accuracy plot from saved results")
    parser.add_argument("--skip_erd", action="store_true",
                        help="Skip ERD/ERS plot (slower, requires more RAM)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.multi_subject and len(args.subject) > 1:
        plot_multi_subject_accuracy(args.subject, args.dataset, out_dir)
        return

    # Generate plots for the first subject listed
    subject = args.subject[0]
    epochs, raw, event_id = load_and_preprocess(subject, args.dataset)

    logger.info(f"\nGenerating plots → {out_dir.absolute()}\n")

    # 1 — Raw EEG overview
    plot_raw_overview(raw, out_dir, subject)

    # 2 — ERD/ERS
    if not args.skip_erd:
        plot_erd_ers(epochs, out_dir, subject)
    else:
        logger.info("Skipping ERD/ERS plot (--skip_erd)")

    # 3 — CSP topomaps
    plot_csp_topomaps(epochs, event_id, out_dir, subject)

    # 4 — Confusion matrix + metrics
    plot_classification_results(epochs, event_id, out_dir, subject, args.classifier)

    # 5 — Multi-subject (if multiple given)
    if len(args.subject) > 1:
        plot_multi_subject_accuracy(args.subject, args.dataset, out_dir)

    logger.info(f"\n✅ All plots saved to: {out_dir.absolute()}")
    logger.info("Files generated:")
    for p in sorted(out_dir.glob("*.png")):
        logger.info(f"  📊 {p.name}")


if __name__ == "__main__":
    main()
