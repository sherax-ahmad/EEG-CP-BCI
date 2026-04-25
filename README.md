# 🧠 EEG-Based Motor Intention Decoder for Children with Cerebral Palsy

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![MNE](https://img.shields.io/badge/MNE--Python-1.6%2B-green)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)
![License](https://img.shields.io/badge/License-MIT-purple)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

**A Brain-Computer Interface (BCI) pipeline that decodes motor imagery signals from EEG — enabling assistive device control for children with Cerebral Palsy.**

</div>

---

## 🎯 Overview

Children with Cerebral Palsy (CP) often have **intact motor intentions** but **impaired execution** due to upper motor neuron lesions. This project builds a complete BCI pipeline that:

- Detects **motor imagery (MI) signals** (e.g., imagined hand movement) from EEG
- Applies **Event-Related Desynchronization (ERD)** analysis on mu (8–12 Hz) and beta (13–30 Hz) rhythms
- Uses **Common Spatial Patterns (CSP)** for spatial filtering
- Trains **ML classifiers** to distinguish left vs. right hand imagery
- Enables potential **communication or assistive device control**

---

## 🗺️ Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    EEG-BCI Pipeline for CP Children                      │
└─────────────────────────────────────────────────────────────────────────┘

  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
  │   EEG DATA   │────▶│  INGESTION   │────▶│  FILTERING   │
  │              │     │              │     │              │
  │ • PhysioNet  │     │ • MNE load   │     │ • Bandpass   │
  │ • BCI Comp.  │     │ • GDF/EDF    │     │   1–40 Hz    │
  │ • Synthetic  │     │ • Format     │     │ • Notch 50Hz │
  └──────────────┘     │   parsing    │     └──────┬───────┘
                       └──────────────┘            │
                                                   ▼
  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
  │  CLASSIFIER  │◀────│   FEATURES   │◀────│  EPOCHING    │
  │              │     │              │     │              │
  │ • LDA        │     │ • CSP        │     │ • Event      │
  │ • SVM        │     │ • ERD/ERS    │     │   markers    │
  │ • Random     │     │ • Band power │     │ • -1s to +4s │
  │   Forest     │     │ • Covariance │     │ • Baseline   │
  └──────┬───────┘     └──────────────┘     └──────────────┘
         │
         ▼
  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
  │  EVALUATION  │────▶│    OUTPUT    │────▶│  REAL-TIME   │
  │              │     │              │     │  INTERFACE   │
  │ • Accuracy   │     │ • Reports    │     │              │
  │ • Kappa      │     │ • Plots      │     │ • LSL stream │
  │ • ROC-AUC    │     │ • Model save │     │ • Mock BCI   │
  │ • Cross-val  │     │ • JSON logs  │     │ • Feedback   │
  └──────────────┘     └──────────────┘     └──────────────┘
```

---

## 📂 Repository Structure

```
eeg-cp-bci/
│
├── 📁 data/
│   ├── raw/                    # Raw EEG files (.gdf, .edf, .mat)
│   ├── processed/              # Cleaned & epoched data (.fif)
│   └── synthetic/              # Simulated EEG for testing
│
├── 📁 src/
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── loader.py           # Multi-format EEG loader
│   │   ├── filter.py           # Bandpass, notch filtering
│   │   ├── artifact.py         # ICA, EOG/EMG artifact removal
│   │   └── epocher.py          # Trial segmentation
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   ├── csp.py              # Common Spatial Patterns
│   │   ├── erd_ers.py          # ERD/ERS analysis
│   │   ├── band_power.py       # Frequency band power features
│   │   └── erp.py              # Event-Related Potentials
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── classifier.py       # LDA, SVM, Random Forest pipeline
│   │   ├── cross_validate.py   # Leave-one-out / k-fold CV
│   │   └── evaluate.py         # Metrics, confusion matrix
│   │
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── topomap.py          # EEG topomap plots
│   │   ├── erd_plot.py         # ERD/ERS time-frequency plots
│   │   └── report.py           # HTML report generator
│   │
│   └── utils/
│       ├── __init__.py
│       ├── data_download.py    # Auto-download public datasets
│       └── logger.py           # Logging utility
│
├── 📁 notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing_pipeline.ipynb
│   ├── 03_feature_extraction.ipynb
│   ├── 04_classification.ipynb
│   └── 05_realtime_simulation.ipynb
│
├── 📁 scripts/
│   ├── download_data.py        # One-click dataset download
│   ├── run_pipeline.py         # Full pipeline runner
│   ├── train_model.py          # Model training entry point
│   └── realtime_bci.py         # Mock real-time BCI loop
│
├── 📁 configs/
│   └── default_config.yaml     # All tunable hyperparameters
│
├── 📁 tests/
│   ├── test_preprocessing.py
│   ├── test_features.py
│   └── test_classifier.py
│
├── 📁 docs/
│   └── CP_BCI_BACKGROUND.md    # Clinical background & references
│
├── 📁 .github/workflows/
│   └── ci.yml                  # GitHub Actions CI
│
├── requirements.txt
├── setup.py
├── .gitignore
└── README.md
```

---

## 🗃️ Public Datasets

This project is designed to work with the following **freely available** EEG datasets:

| Dataset | Classes | Subjects | Channels | Fs | Download |
|---------|---------|----------|----------|----|----------|
| **BCI Competition IV 2a** | 4 (L/R hand, feet, tongue) | 9 | 22 EEG | 250 Hz | [PhysioNet](https://www.bbci.de/competition/iv/) |
| **BCI Competition IV 2b** | 2 (L/R hand) | 9 | 3 EEG | 250 Hz | [BCI Comp IV](https://www.bbci.de/competition/iv/) |
| **PhysioNet EEG MI DB** | 4 | 109 | 64 EEG | 160 Hz | [PhysioNet](https://physionet.org/content/eegmmidb/1.0.0/) |
| **Large MI Dataset** | 4 paradigms | 13 | 38 EEG | 256 Hz | [Figshare](https://doi.org/10.1038/sdata2018211) |

> **Note for CP Research:** No large-scale open EEG dataset exists specifically for CP children. The BCI Competition datasets (healthy subjects doing motor imagery) are used as the gold standard surrogate — the MI paradigm is neurologically comparable. See `docs/CP_BCI_BACKGROUND.md` for clinical context.

### Quick Download (PhysioNet — No Account Required)
```bash
python scripts/download_data.py --dataset physionet --subjects 1 5
```

---

## ⚡ Quick Start

### 1. Install Dependencies
```bash
git clone https://github.com/YOUR_USERNAME/eeg-cp-bci.git
cd eeg-cp-bci
pip install -r requirements.txt
```

### 2. Download Data
```bash
python scripts/download_data.py --dataset physionet --subjects 1 3
```

### 3. Run Full Pipeline
```bash
python scripts/run_pipeline.py --config configs/default_config.yaml
```

### 4. Train & Evaluate
```bash
python scripts/train_model.py --subject 1 --classifier lda --cv 10
```

### 5. Real-time Simulation
```bash
python scripts/realtime_bci.py --model_path outputs/model.pkl
```

---

## 🧪 Key Methods

### Common Spatial Patterns (CSP)
CSP finds spatial filters that **maximize variance** for one class while **minimizing** it for another — isolating motor-cortex activity.

### ERD/ERS Analysis
- **Event-Related Desynchronization (ERD):** Decrease in mu/beta power during motor imagery → active movement preparation
- **Event-Related Synchronization (ERS):** Post-movement power rebound

### Classification Pipeline
```
EEG Epochs → CSP (n_components=6) → log(variance) → LDA/SVM/RF → Class Label
```

---

## 📊 Expected Results

| Classifier | Dataset | Accuracy | Cohen's Kappa |
|------------|---------|----------|---------------|
| LDA + CSP | BCI IV 2a | ~70–75% | ~0.60 |
| SVM + CSP | BCI IV 2a | ~72–78% | ~0.63 |
| RF + CSP | BCI IV 2a | ~68–73% | ~0.57 |

---

## 🩺 Clinical Background

See [`docs/CP_BCI_BACKGROUND.md`](docs/CP_BCI_BACKGROUND.md) for:
- Neurophysiology of CP and motor imagery
- EEG findings in CP children
- Ethical considerations
- Key references

---

## 📜 License

MIT License. See [LICENSE](LICENSE) for details.

---

## 🙏 References

1. Ang, K.K. et al. (2012). Filter Bank Common Spatial Pattern for Motor Imagery BCI. *IEEE TNSRE*
2. Steenbergen et al. (2009). Motor imagery in CP. *Dev Med Child Neurol*
3. Daly et al. (2013). BCI control by users with cerebral palsy. *Clin Neurophysiol*
4. BCI Competition IV datasets — Graz University of Technology
5. PhysioNet EEG Motor Movement/Imagery Dataset
