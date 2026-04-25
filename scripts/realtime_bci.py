#!/usr/bin/env python3
"""
realtime_bci.py — Mock Real-Time BCI Loop

Simulates a real-time BCI system by streaming synthetic EEG through
a sliding window and making live classifications.

This demonstrates the architecture for a deployable system that would
receive actual EEG from an amplifier (e.g., OpenBCI, g.tec, BrainProducts).

For real hardware integration:
    - Replace SyntheticEEGStream with pylsl.StreamInlet
    - Add feedback (visual/auditory/device control)

Usage:
    python scripts/realtime_bci.py --model_path outputs/subject_001/model.pkl
    python scripts/realtime_bci.py --demo  # Demo mode with synthetic stream
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from collections import deque

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class SyntheticEEGStream:
    """
    Simulates a real-time EEG stream with motor imagery patterns.

    In a real system, this is replaced by an LSL (Lab Streaming Layer)
    inlet from an EEG amplifier.
    """

    def __init__(
        self,
        n_channels: int = 22,
        sfreq: float = 250.0,
        paradigm_interval_s: float = 6.0,
        seed: int = 42,
    ):
        self.n_channels = n_channels
        self.sfreq = sfreq
        self.paradigm_interval = paradigm_interval_s
        self.rng = np.random.default_rng(seed)
        self._sample_idx = 0
        self._current_class = 0  # 0=left, 1=right
        self._class_start = 0
        self._times = []

    def get_chunk(self, n_samples: int = 10) -> np.ndarray:
        """Get next chunk of EEG samples. Returns (n_channels, n_samples)."""
        chunk = self.rng.standard_normal((self.n_channels, n_samples)) * 8e-6

        # Simulate background alpha (10 Hz)
        t = np.arange(self._sample_idx, self._sample_idx + n_samples) / self.sfreq
        alpha = 5e-6 * np.sin(2 * np.pi * 10 * t)
        chunk += alpha

        # Check if in MI phase (after 2s from class start)
        trial_t = self._sample_idx / self.sfreq % self.paradigm_interval
        if trial_t > 2.0:  # MI phase
            # Suppress mu/beta in contralateral channels
            if self._current_class == 0:  # Left hand → C4 (right hemisphere)
                chunk[self.n_channels // 2 + 2] *= 0.3
            else:  # Right hand → C3 (left hemisphere)
                chunk[self.n_channels // 2 - 2] *= 0.3

        self._sample_idx += n_samples
        return chunk

    def get_current_label(self) -> int:
        """Get current true label (for evaluation in simulation)."""
        trial_t = self._sample_idx / self.sfreq % self.paradigm_interval
        if trial_t > 2.0:
            return self._current_class
        return -1  # Rest period

    def next_trial(self):
        """Advance to next trial."""
        self._current_class = 1 - self._current_class  # Alternate L/R


class RealtimeBCISystem:
    """
    Sliding-window real-time BCI classifier.

    Architecture:
        EEG Stream → Ring Buffer → Preprocessing → CSP Features → Classifier → Output
    """

    def __init__(
        self,
        model_path: str,
        sfreq: float = 250.0,
        buffer_size_s: float = 4.0,
        step_size_s: float = 0.25,
        n_channels: int = 22,
    ):
        self.sfreq = sfreq
        self.buffer_size = int(sfreq * buffer_size_s)
        self.step_size = int(sfreq * step_size_s)
        self.n_channels = n_channels

        # Ring buffer
        self.buffer = deque(maxlen=self.buffer_size)

        # Load model
        if model_path and Path(model_path).exists():
            from src.models.classifier import load_model
            self.pipeline = load_model(model_path)
            logger.info(f"Loaded model: {model_path}")
        else:
            logger.warning("No model found. Using demo classifier.")
            self._build_demo_classifier()

        self.class_names = ["LEFT HAND", "RIGHT HAND"]
        self.predictions = []
        self.confidences = []

    def _build_demo_classifier(self):
        """Build a simple demo classifier for testing without saved model."""
        from src.models.classifier import build_pipeline
        from src.preprocessing.loader import load_synthetic
        from src.preprocessing.epocher import get_motor_imagery_data

        logger.info("Building and fitting demo classifier...")
        raw = load_synthetic(n_channels=self.n_channels, sfreq=self.sfreq,
                             duration_s=300, n_trials_per_class=50)
        from src.preprocessing.filter import preprocess_raw
        raw = preprocess_raw(raw, l_freq=1.0, h_freq=40.0)

        import mne
        events, event_id = mne.events_from_annotations(raw, verbose=False)
        epochs = mne.Epochs(raw, events, event_id, tmin=-1.0, tmax=4.0,
                            baseline=(-1.0, 0.0), preload=True, verbose=False)
        X, y = get_motor_imagery_data(epochs)

        if len(X) >= 10:
            self.pipeline = build_pipeline("lda", n_csp_components=4)
            self.pipeline.fit(X, y)
            logger.info(f"  Demo classifier trained on {len(X)} epochs")
        else:
            self.pipeline = None
            logger.warning("  Insufficient data for demo classifier")

    def add_samples(self, chunk: np.ndarray) -> None:
        """Add new EEG samples to buffer. chunk: (n_channels, n_samples)."""
        for sample_idx in range(chunk.shape[1]):
            self.buffer.append(chunk[:, sample_idx])

    def is_ready(self) -> bool:
        """Check if buffer has enough data for prediction."""
        return len(self.buffer) >= self.buffer_size

    def predict(self) -> tuple[int, float, str]:
        """
        Run prediction on current buffer.

        Returns:
            (class_idx, confidence, class_name)
        """
        if not self.is_ready() or self.pipeline is None:
            return -1, 0.0, "BUFFERING..."

        # Get buffer as array (n_channels, n_times)
        buffer_array = np.array(list(self.buffer)).T  # (n_ch, n_times)

        # Apply basic bandpass (simplified for real-time)
        from scipy.signal import butter, sosfiltfilt
        sos = butter(4, [8 / (self.sfreq / 2), 30 / (self.sfreq / 2)],
                     btype="bandpass", output="sos")
        buffer_filtered = np.zeros_like(buffer_array)
        for ch in range(buffer_array.shape[0]):
            buffer_filtered[ch] = sosfiltfilt(sos, buffer_array[ch])

        # Shape: (1, n_channels, n_times) for sklearn pipeline
        X = buffer_filtered[np.newaxis, :, :]

        try:
            pred = self.pipeline.predict(X)[0]
            if hasattr(self.pipeline, "predict_proba"):
                proba = self.pipeline.predict_proba(X)[0]
                confidence = proba.max()
            else:
                confidence = 0.75  # Fallback

            class_name = self.class_names[int(pred)] if int(pred) < len(self.class_names) else "UNKNOWN"
            self.predictions.append(pred)
            self.confidences.append(confidence)
            return int(pred), float(confidence), class_name

        except Exception as e:
            logger.debug(f"Prediction error: {e}")
            return -1, 0.0, "ERROR"

    def display_prediction(self, pred: int, confidence: float, class_name: str) -> None:
        """Display real-time prediction in terminal."""
        if pred < 0:
            bar = "⏳ " + class_name
        else:
            filled = int(confidence * 20)
            bar = "█" * filled + "░" * (20 - filled)
            arrow = "←" if pred == 0 else "→"
            bar = f"{arrow} [{bar}] {confidence*100:.1f}%  {class_name}"

        print(f"\r  {bar:60s}", end="", flush=True)


def run_realtime_demo(model_path: str = None, duration_s: float = 30.0):
    """
    Run real-time BCI simulation for specified duration.

    Args:
        model_path: Path to saved model .pkl file
        duration_s: Simulation duration in seconds
    """
    logger.info("="*60)
    logger.info("  EEG-BCI REAL-TIME SIMULATION")
    logger.info("  Cerebral Palsy Motor Intention Decoder")
    logger.info("="*60)
    logger.info(f"Duration: {duration_s}s | Simulating real EEG stream...")

    sfreq = 250.0
    n_channels = 22
    stream = SyntheticEEGStream(n_channels=n_channels, sfreq=sfreq)
    bci = RealtimeBCISystem(
        model_path=model_path or "",
        sfreq=sfreq,
        n_channels=n_channels,
    )

    chunk_size = 25  # 100ms chunks
    total_chunks = int(duration_s * sfreq / chunk_size)
    trial_duration = 6.0
    samples_per_trial = int(trial_duration * sfreq)

    logger.info("\nStarting stream... (Press Ctrl+C to stop)")
    print("\n  REAL-TIME OUTPUT:")
    print("  " + "-"*58)

    correct = 0
    total = 0

    try:
        for chunk_idx in range(total_chunks):
            # Check for new trial
            if chunk_idx * chunk_size % samples_per_trial == 0 and chunk_idx > 0:
                stream.next_trial()

            chunk = stream.get_chunk(chunk_size)
            bci.add_samples(chunk)

            # Predict every step_size samples
            if chunk_idx % (bci.step_size // chunk_size + 1) == 0:
                pred, conf, class_name = bci.predict()
                true_label = stream.get_current_label()
                bci.display_prediction(pred, conf, class_name)

                if pred >= 0 and true_label >= 0:
                    if pred == true_label:
                        correct += 1
                    total += 1

            time.sleep(chunk_size / sfreq)  # Real-time pacing

    except KeyboardInterrupt:
        print("\n\n  [Stopped by user]")

    print("\n\n" + "="*60)
    if total > 0:
        online_acc = correct / total * 100
        print(f"  SIMULATION RESULTS:")
        print(f"  Online Accuracy: {online_acc:.1f}% ({correct}/{total} correct)")
        print(f"  Total predictions: {len(bci.predictions)}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Real-time EEG-BCI Simulation")
    parser.add_argument("--model_path", default=None,
                        help="Path to trained model .pkl file")
    parser.add_argument("--duration", type=float, default=30.0,
                        help="Simulation duration in seconds (default: 30)")
    parser.add_argument("--demo", action="store_true",
                        help="Run demo with built-in classifier")
    args = parser.parse_args()

    run_realtime_demo(
        model_path=args.model_path if not args.demo else None,
        duration_s=args.duration,
    )


if __name__ == "__main__":
    main()
