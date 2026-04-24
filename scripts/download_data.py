#!/usr/bin/env python3
"""
download_data.py — One-click public EEG dataset downloader

Supports:
  - PhysioNet EEG Motor Movement/Imagery Database (109 subjects, free)
  - BCI Competition IV 2a (instructions, requires manual download)
  - BCI Competition IV 2b (instructions)

Usage:
    python scripts/download_data.py --dataset physionet --subjects 1 5
    python scripts/download_data.py --dataset physionet --subjects all
    python scripts/download_data.py --dataset info
"""

import argparse
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def download_physionet(subjects: list[int], data_dir: str = "data/raw") -> None:
    """
    Download PhysioNet EEG MI Database via MNE.

    Dataset: https://physionet.org/content/eegmmidb/1.0.0/
    - 109 subjects
    - 64 EEG channels at 160 Hz
    - Tasks: left/right hand, feet, imagery versions
    - Runs 4,8,12 = left/right hand motor imagery
    - License: PhysioNet Credentialed Health Data License 1.5.0
    - No account required for this specific dataset!
    """
    try:
        import mne
    except ImportError:
        logger.error("MNE-Python not installed. Run: pip install mne")
        sys.exit(1)

    runs = [4, 8, 12]  # Left/right hand motor imagery runs
    Path(data_dir).mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading PhysioNet EEG MI Database")
    logger.info(f"Subjects: {subjects}")
    logger.info(f"Runs: {runs} (left/right hand motor imagery)")
    logger.info(f"Target directory: {data_dir}/")

    for subj in subjects:
        logger.info(f"  Downloading subject {subj:03d}...")
        try:
            fnames = mne.datasets.eegbci.load_data(
                subject=subj,
                runs=runs,
                path=data_dir,
                update_path=True,
                verbose=False,
            )
            logger.info(f"    ✓ Subject {subj:03d}: {len(fnames)} files")
        except Exception as e:
            logger.error(f"    ✗ Subject {subj:03d} failed: {e}")

    logger.info("Download complete!")
    logger.info(f"Files saved to: {Path(data_dir).absolute()}")


def print_bciciv_instructions() -> None:
    """Print instructions for BCI Competition IV datasets."""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║         BCI Competition IV Dataset Download Instructions         ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  Dataset 2a (4-class motor imagery, 22 channels)                 ║
║  ─────────────────────────────────────────────                   ║
║  URL: https://www.bbci.de/competition/iv/                        ║
║  1. Go to the URL above                                          ║
║  2. Click "Data sets" → "Dataset 2a"                             ║
║  3. Register (free) to get download link                         ║
║  4. Download all A0xT.gdf and A0xE.gdf files                     ║
║  5. Place in: data/raw/bciciv2a/                                 ║
║                                                                  ║
║  Dataset 2b (2-class left/right, 3 channels)                     ║
║  ─────────────────────────────────────────────                   ║
║  URL: https://www.bbci.de/competition/iv/                        ║
║  Same steps as above, choose Dataset 2b                          ║
║  Place in: data/raw/bciciv2b/                                    ║
║                                                                  ║
║  MOABB (Multiple Benchmark Datasets) — RECOMMENDED               ║
║  ─────────────────────────────────────────────────               ║
║  pip install moabb                                               ║
║  python -c "from moabb.datasets import BNCI2014_001; BNCI2014_001().download()"  ║
║  This downloads BCI Comp IV 2a automatically!                    ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
""")


def print_dataset_info() -> None:
    """Print overview of all available datasets."""
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║           Public EEG Motor Imagery Datasets for CP-BCI              ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  1. PhysioNet EEG MI DB (RECOMMENDED — No account required)          ║
║     • 109 healthy subjects, 64 ch, 160 Hz                           ║
║     • Left/right hand + feet motor imagery                           ║
║     • Download: python scripts/download_data.py --dataset physionet  ║
║     • URL: https://physionet.org/content/eegmmidb/1.0.0/            ║
║                                                                      ║
║  2. BCI Competition IV 2a (Gold standard benchmark)                  ║
║     • 9 subjects, 22 ch, 250 Hz                                     ║
║     • 4 classes: L-hand, R-hand, feet, tongue                       ║
║     • Manual download: https://www.bbci.de/competition/iv/          ║
║     • Also via MOABB: pip install moabb                              ║
║                                                                      ║
║  3. BCI Competition IV 2b                                            ║
║     • 9 subjects, 3 ch (C3, Cz, C4), 250 Hz                        ║
║     • 2 classes: L-hand, R-hand, 5 sessions/subject                 ║
║     • Manual download: same as above                                 ║
║                                                                      ║
║  4. Large MI Dataset (Stieger et al., 2018)                          ║
║     • 13 subjects, 38 ch, 256 Hz, 60h total, 60k+ trials            ║
║     • URL: https://doi.org/10.1038/sdata2018211                     ║
║     • Figshare download (free)                                       ║
║                                                                      ║
║  ⚠️  CP-Specific Note:                                               ║
║  No large open EEG dataset exists specifically for CP children.      ║
║  These healthy-subject MI datasets are the standard surrogate.       ║
║  See docs/CP_BCI_BACKGROUND.md for clinical context.                ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
""")


def main():
    parser = argparse.ArgumentParser(
        description="Download public EEG datasets for CP-BCI research"
    )
    parser.add_argument(
        "--dataset",
        choices=["physionet", "bciciv", "info"],
        default="physionet",
        help="Dataset to download (default: physionet)",
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=["1", "2", "3"],
        help="Subject IDs or 'all' (default: 1 2 3)",
    )
    parser.add_argument(
        "--data_dir",
        default="data/raw",
        help="Output directory (default: data/raw)",
    )
    args = parser.parse_args()

    if args.dataset == "info":
        print_dataset_info()
        return

    if args.dataset == "bciciv":
        print_bciciv_instructions()
        return

    # Parse subjects
    if "all" in args.subjects:
        subjects = list(range(1, 110))
        logger.warning("Downloading ALL 109 subjects (~4 GB). This may take a while.")
    else:
        subjects = [int(s) for s in args.subjects]
        invalid = [s for s in subjects if not 1 <= s <= 109]
        if invalid:
            logger.error(f"Invalid subject IDs: {invalid}. Must be 1–109.")
            sys.exit(1)

    if args.dataset == "physionet":
        download_physionet(subjects, args.data_dir)


if __name__ == "__main__":
    main()
