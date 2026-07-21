# AI-Aided ENF Timestamping

Learned latent matching for **Electric Network Frequency (ENF)** timestamping in multimedia forensics.

This repository contains the official source code for the paper *"AI-Aided ENF Timestamping in Multimedia Forensics"* by **Raz Barak, Lital Dabush, Nir Shlezinger, and Tirza Routtenberg** (School of ECE, Ben-Gurion University of the Negev).

---

## Overview

ENF fluctuations of the power grid are passively embedded in audio and video recordings, acting as an environmental signature that can be used to verify *when* a recording was captured. Conventional timestamping matches the ENF trace extracted from a target recording against a reference timeline using correlation-based similarity measures (Correlation Coefficient, CC, or Normalized Misalignment, NM). These measures degrade sharply for short recordings (under two minutes), low SNR, and heterogeneous sensing conditions.

This project reframes the similarity function itself as the key bottleneck and *learns* it from data. The core is a **learned latent matching** module that:

1. Embeds ENF segments into a compact latent space with a **Temporal Convolutional Network (TCN)** encoder, trained with a supervised contrastive objective (InfoNCE + triplet loss).
2. **Fuses** the learned cosine similarity with classical correlation-based scores through a lightweight linear layer, preserving the strengths of raw-signal matching where it is reliable while adding robustness where it fails.

Timestamping is posed as a retrieval task: for a target segment `x` and reference timeline `r`, the estimated timestamp is the offset `τ` that maximizes the (learned) similarity `s(x, r_τ)`. An estimate is counted correct when it lands within a ±15 s tolerance of the ground truth.

The framework is trained and evaluated in a **cross-grid** protocol (trained on Chinese recordings, tested on Israeli recordings), demonstrating that the learned representation transfers across distinct electrical grids.

### Headline results

Success accuracy (% of targets whose estimated timestamp falls within ±15 s of ground truth):

| Recording length | Baseline (CC) | Baseline (NM) | Encoder | Encoder + Fusion |
|---|---|---|---|---|
| ≤ 2 min | 46.53% | 55.14% | **74.04%** | 73.14% |
| All lengths | 74.28% | 77.63% | 81.85% | **85.63%** |

The learned matcher yields the largest gains exactly where classical methods struggle: short recordings and low-SNR conditions.

---

## Repository structure

```
AI-Aided-ENF-Timestamping/
│
├── src/                          # Core library (importable package)
│   ├── config.py                 # argparse-based configuration and path resolution
│   ├── constants.py              # Dataset names and directory-stage constants
│   ├── models/
│   │   ├── encoder.py            # TwoStageTCNEncoder (TCN blocks, stats pooling, projection head)
│   │   └── fusion.py             # LinearFusion (learned weighted sum of similarity scores)
│   ├── data/
│   │   ├── processing.py         # Resampling, mono conversion, segmentation, AWGN injection
│   │   ├── enf_extraction.py     # STFT + band-pass ENF extraction (per harmonic)
│   │   ├── filtering.py          # Sliding-window CC/NM, ground-truth alignment
│   │   ├── triplet_mine.py       # Positive / soft-negative / hard-negative mining
│   │   └── dataloader.py         # Length-bucketed batch loaders (padding-free)
│   ├── training/
│   │   ├── losses.py             # InfoNCE, in-batch triplet, listwise cross-entropy
│   │   └── trainer.py            # Encoder and fusion-layer training loops
│   └── evaluation/
│       └── metrics.py            # Batched CC/NM scores, moving average, top-k retrieval
│
├── scripts/                      # Runnable entry points
│   ├── download_enf_whu.py       # Fetch the ENF-WHU (H1) training set
│   ├── download_enf_bgu.py       # Fetch the ENF-BGU / Wavemark evaluation set
│   ├── 01_generate_dataset.py    # End-to-end dataset construction pipeline
│   └── train_encoder.py          # Train the TCN encoder
│
├── pipelines/                    # Legacy / notebook-derived reference implementations
│
├── data/                         # Datasets and saved weights (see below)
│   ├── datasets/
│   │   ├── ENF_WHU/              # Training grid (China)
│   │   └── ENF_wave_mark/        # Evaluation grid (Israel), incl. SNR-perturbed test sets
│   └── models/
│       ├── encoder.pth           # Pretrained encoder weights
│       └── fusion.pth            # Pretrained fusion-layer weights
│
├── AI-Aided-ENF-Timestamping.ipynb   # Full end-to-end notebook (Colab-oriented)
└── README.md
```

> **`src/` vs `pipelines/`** — `src/` is the clean, modular library that the scripts import. The `pipelines/` directory holds earlier, notebook-derived versions of the same routines and is kept only for reference; new work should build on `src/`.

---

## Installation

```bash
git clone https://github.com/r-barak/AI-Aided-ENF-Timestamping.git
cd AI-Aided-ENF-Timestamping

# (recommended) create an environment first
python -m venv .venv && source .venv/bin/activate

pip install torch numpy scipy pandas tqdm requests
```

The notebook additionally uses `matplotlib` for the result plots. A CUDA-capable GPU is highly recommended for training but not required — the code falls back to CPU automatically (pass `--no_cuda` to force it).

---

## Data

The pipeline uses two real-world datasets in a cross-grid setup:

- **Training — ENF-WHU (H1 subset):** ~20 hours of recordings from China. Fetched from the public [ENF-WHU-Dataset](https://github.com/ghua-ac/ENF-WHU-Dataset).
- **Evaluation — ENF-Wavemark / ENF-BGU:** ~5 hours of recordings from Israel, used strictly for testing (including SNR-perturbed variants).

Download the raw data with the provided scripts (run from the repository root):

```bash
python -m scripts.download_enf_whu     # → data/datasets/ENF_WHU/01_raw/...
python -m scripts.download_enf_bgu     # → data/datasets/ENF_BGU/01_raw/...
```

### Directory convention

The dataset pipeline organizes each grid's data into numbered stages, with target ("query") and reference recordings kept separate:

```
data/datasets/<DATASET_NAME>/
├── 01_raw/            # Raw .wav recordings
│   ├── queries/
│   └── references/
├── 02_preprocessed/   # Resampled + segmented (.npz)
└── 03_enf/            # Extracted ENF traces per harmonic (50hz/, 100hz/, 150hz/)
```

Pretrained encoder and fusion weights ship under `data/models/`, so you can run evaluation without retraining.

---

## Usage

### 1. Build the training dataset

`scripts/01_generate_dataset.py` runs the four-stage preparation pipeline: signal preprocessing (resample + segment), multi-harmonic ENF extraction, ground-truth alignment verification, and contrastive triplet mining. It writes `train_pairs.pt` and `val_pairs.pt`.

```bash
python -m scripts.01_generate_dataset
```

Each stage is modular — the later stages read the artifacts produced by earlier ones, so you can comment individual stages in/out when re-running.

### 2. Train the encoder

Trains the `TwoStageTCNEncoder` with the combined InfoNCE + triplet contrastive loss, keeping the best-validation checkpoint.

```bash
python -m scripts.train_encoder
```

### 3. Train the fusion layer

The linear fusion layer learns to combine the encoder's latent cosine similarity with classical correlation scores. The full flow for building the fusion dataset (embedding the training set, mining top-k candidates) and training the layer is provided in the notebook (see *Train Fusion Layer*), driven by `train_fusion_layer` in `src/training/trainer.py`.

### 4. Evaluate

The notebook's *Results* section reproduces the paper's figures and tables, both for clean signals and across SNR levels (−20 dB to +10 dB via injected AWGN), comparing the encoder-only and encoder+fusion variants against the CC and NM baselines.

### End-to-end notebook

`AI-Aided-ENF-Timestamping.ipynb` walks through the complete workflow — environment setup, dataset generation, encoder and fusion training, and evaluation — and is the most convenient way to reproduce the paper's results (it is oriented toward Google Colab / Drive).

---

## Method details

**Encoder (`src/models/encoder.py`).** A two-stage TCN operating on a `[C, T]` input, where the `C` harmonic channels (the first three ENF harmonics: 50, 100, 150 Hz) are treated as equally weighted channels. Stacked dilated residual TCN blocks capture short- and long-range temporal structure; a temporal statistics-pooling layer (mean + std) produces a fixed-size representation regardless of segment length `T`, which is mapped through a projection head into the contrastive embedding space.

**Fusion (`src/models/fusion.py`).** `LinearFusion` learns a weighted sum (plus bias) over a vector of similarity scores — the latent cosine similarity together with correlation-based scores — producing a single fused similarity used for retrieval. The shipped weights fuse four scores.

**Losses (`src/training/losses.py`).** The encoder is trained with a weighted sum of an in-batch InfoNCE loss and a hardest-negative triplet loss. Positives are reference segments aligned to the target within the tolerance window; negatives are everything else in the batch. The fusion layer is trained with a listwise cross-entropy over candidate scores.

**Contrastive sampling (`src/data/triplet_mine.py`).** For each target segment the miner builds one positive, per-harmonic *soft negatives* (high-scoring segments from the same reference but outside the tolerance window), and per-harmonic *hard negatives* (high-scoring segments from different reference recordings — the "impostors").

**Length bucketing (`src/data/dataloader.py`).** Samples are grouped by segment length so each batch is uniform, eliminating padding and preserving the temporal integrity of ENF traces while still allowing efficient batched encoding.

---

## Configuration

Configuration is handled by `src/config.py` via command-line flags (all optional; sensible defaults match the paper). Key options:

| Flag | Default | Description |
|---|---|---|
| `--network_freq` | `50` | Nominal grid frequency (Hz) |
| `--harmonics` | `1 2 3` | Harmonics to extract (× `network_freq`) |
| `--fs` | `400` | Target sampling rate after downsampling |
| `--f_range` | `0.2` | Frequency band around each harmonic |
| `--window_size_sec` | `5` | STFT window length (s) |
| `--overlap_ratio` | `0.9` | STFT overlap |
| `--band_pass_order` | `1501` | FIR band-pass filter order |
| `--segment_dur` | `1.0 … 5.0` | Target segment durations (min) |
| `--allowed_lag_sec` | `15` | Timestamp tolerance window (s) |
| `--batch_size` | `128` | Training batch size |
| `--num_epochs` | `20` | Training epochs |
| `--lr` | `1e-3` | Adam/AdamW learning rate |
| `--snr_levels` | `-20 … 20` | SNR levels (dB) for AWGN robustness tests |
| `--no_cuda` | `False` | Force CPU training |

Run any script with `-h` to see the full list.

---

## Citation

If you use this code, please cite the paper:

```bibtex

```

---

## Acknowledgements

This work was supported by the Israeli Ministry of Science and Technology. The training data is drawn from the [ENF-WHU dataset](https://github.com/ghua-ac/ENF-WHU-Dataset). This repository builds on the observation, established in prior work, that the *similarity function* — rather than the ENF extraction algorithm — is the decisive factor for timestamping accuracy under challenging conditions.
