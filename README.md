# Feature-Space Drift Detection for Image Classification

This repository contains the capstone experiment pipeline for evaluating whether label-free feature drift metrics can act as practical proxies for model degradation in image classification systems.

The project compares deep feature distributions from pretrained `ResNet-50` and `EfficientNet-B0`, measures drift with `KS` and `MMD`, measures performance drop with a linear probe, and calibrates alert thresholds from clean or reference windows.

## Research Goal

In simple terms: if the input data distribution changes, can we detect that change from model features before or while classification accuracy starts to degrade?

The repository evaluates that question across:

- `CIFAR-10` as a controlled synthetic-corruption benchmark
- `BloodMNIST` as a second controlled benchmark in a medical-image domain
- `Waterbirds` as a realistic natural-shift benchmark with spurious correlations

## What Each Benchmark Demonstrates

- `CIFAR-10`
  Controlled corruption benchmark for testing whether feature drift tracks accuracy drop under synthetic shift.
- `BloodMNIST`
  A second controlled benchmark showing the same methodology on a different image domain.
- `Waterbirds`
  A natural-shift benchmark showing how the method behaves under subgroup and environment shift rather than synthetic severity levels.

## Main Methods

- Backbone feature extractors: `ResNet-50`, `EfficientNet-B0`
- Drift metrics: Kolmogorov-Smirnov (`KS`), Maximum Mean Discrepancy (`MMD`)
- Performance proxy: linear probe trained on clean/reference features
- Thresholding: calibration from clean or matched-reference windows

## Repository Structure

```text
.
├── cap_experiments.py      # Main experiment runner
├── CAP.ipynb               # Lightweight notebook entry point / narrative companion
├── RUN_EXPERIMENTS.md      # Exact run commands and dataset setup notes
├── REPORT_RESULTS.md       # Thesis-safe interpretation of the completed runs
├── requirements.txt        # Python dependencies
├── LICENSE                 # Repository license
└── Levon Gevorgyan_Capstone Project Proposal.pdf
```

Generated data and result artifacts are intentionally not committed.

## Installation

1. Create and activate a Python environment.
2. Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

3. Optional but recommended: verify the script imports cleanly.

```bash
python3 -m py_compile cap_experiments.py
```

## Datasets

Datasets are not committed to this repository.

- `CIFAR-10` can be downloaded automatically by `torchvision`
- `BloodMNIST` can be downloaded automatically through `medmnist`
- `Waterbirds` must be downloaded separately and extracted locally

Expected local dataset locations:

- `./data/` for CIFAR-10 and BloodMNIST cache files
- `./data/waterbirds/` for the Waterbirds dataset root containing `metadata.csv`

## Exact Commands

### CIFAR-10

Recommended run:

```bash
python3 cap_experiments.py --datasets cifar10 --train-size 300 --test-size 150 --probe-epochs 8 --trials 2 --max-feature-dims 128 --download --output-dir results_cifar10
```

### BloodMNIST

```bash
python3 cap_experiments.py --datasets bloodmnist --train-size 300 --test-size 150 --probe-epochs 8 --trials 2 --max-feature-dims 128 --download --output-dir results_bloodmnist
```

### Waterbirds

Download and extract the official Waterbirds dataset locally, then run:

```bash
python3 cap_experiments.py --datasets waterbirds --waterbirds-root ./data/waterbirds --train-size 300 --test-size 150 --probe-epochs 8 --trials 2 --max-feature-dims 128 --output-dir results_waterbirds
```

### Smoke Test

```bash
python3 cap_experiments.py --datasets fake --train-size 32 --test-size 24 --probe-epochs 1 --trials 1 --random-weights
```

## Expected Outputs

Each run writes a result directory containing:

- `experiment_results_raw.csv`
- `experiment_results_summary.csv`
- `correlation_summary.csv`
- `calibration_summary.csv`
- `summary.json`

## Reproducibility Notes

- CIFAR-10 and BloodMNIST share the same controlled synthetic drift families:
  `gaussian_noise`, `motion_blur`, `brightness_shift`
- Waterbirds uses a different evaluation design because it is a natural-shift benchmark
- The pipeline uses fixed seeds and repeated trials for more stable estimates
- `--max-feature-dims` provides a faster CPU-friendly approximation for KS and MMD on high-dimensional feature maps

## Limitations

- Controlled corruption results do not automatically imply real-world deployment robustness
- Waterbirds evaluates natural subgroup shift, which is conceptually different from synthetic severity experiments
- Threshold calibration is useful for monitoring experiments, but not by itself a full production alerting policy

## License

This repository is released under the MIT License.
