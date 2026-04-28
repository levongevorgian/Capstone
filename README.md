# Feature-Space Drift Detection for Image Classification

This repository contains the code, saved result snapshots, generated figures, and analysis artifacts for a capstone project on label-free feature-space drift detection in image classification.

The project evaluates whether feature-space drift statistics, especially K-S and MMD over pretrained feature embeddings, can indicate downstream model performance degradation across:

- `CIFAR-10` with controlled synthetic corruptions
- `BloodMNIST` with controlled synthetic corruptions
- `Waterbirds` with natural subgroup and spurious-correlation shift

## Repository Contents

```text
.
├── code/
│   ├── main_codes/                    # experiment runners
│   ├── utils/                         # shared plotting/style utilities
│   └── visualizations/
│       ├── code/                      # figure and analysis scripts
│       └── figures/                   # generated final and diagnostic figures
├── data/
│   ├── raw_data/                      # local dataset cache placeholder
│   └── processed_data/
│       └── waterbirds_analysis/       # generated Waterbirds tables/report
├── docs/
│   └── run_guide.md                   # concise command reference
├── outputs/
│   ├── figures/                       # reserved figure export directory
│   ├── reports/                       # reserved report export directory
│   └── results/                       # saved experiment outputs
├── paper/
│   └── figures/                       # paper figure placeholder/assets
├── LICENSE
├── README.md
└── requirements.txt
```

## Main Files

- [code/main_codes/experiment_runner.py](code/main_codes/experiment_runner.py) runs the main controlled-benchmark and Waterbirds experiments.
- [code/main_codes/waterbirds_reference_runner.py](code/main_codes/waterbirds_reference_runner.py) evaluates an alternative class-conditional Waterbirds reference design.
- [code/main_codes/waterbirds_mitigation_runner.py](code/main_codes/waterbirds_mitigation_runner.py) compares Waterbirds subgroup-aware probe-training strategies.
- [code/visualizations/code/analysis_visualizations.py](code/visualizations/code/analysis_visualizations.py) generates the final and diagnostic figures from saved outputs.
- [code/visualizations/code/waterbirds_detailed_analysis.py](code/visualizations/code/waterbirds_detailed_analysis.py) produces Waterbirds subgroup summaries and a Markdown report.
- [docs/run_guide.md](docs/run_guide.md) provides a compact command reference.

## Setup

From the repository root, install the required Python packages:

```bash
python3 -m pip install -r requirements.txt
```

The project uses `numpy`, `pandas`, `Pillow`, `scipy`, `torch`, `torchvision`, and `medmnist`.

Optional syntax check:

```bash
python3 -m py_compile \
  code/main_codes/experiment_runner.py \
  code/main_codes/waterbirds_reference_runner.py \
  code/main_codes/waterbirds_mitigation_runner.py \
  code/visualizations/code/analysis_visualizations.py \
  code/visualizations/code/waterbirds_detailed_analysis.py
```

## Data Requirements

Large raw datasets are intentionally not versioned in this repository.

Expected local dataset locations:

- `data/raw_data/` for CIFAR-10 and BloodMNIST downloads/caches
- `data/raw_data/waterbirds/` for Waterbirds, including `metadata.csv`
- an external ImageFolder-style directory when using `imagenet_local`

CIFAR-10 and BloodMNIST can be downloaded automatically by adding `--download` to the experiment command. Waterbirds must be placed locally because it is not downloaded by the scripts.

## Quick Reproduction Run

For final submission verification, the following smoke-test workflow was run locally. If your clone is in a different location, replace the first command with `cd <repository-root>`.

```bash
cd /Users/levongevorgyan/Desktop/Experience
python3 -m pip install -r requirements.txt
python3 code/main_codes/experiment_runner.py --datasets fake --train-size 32 --test-size 24 --probe-epochs 1 --trials 1 --random-weights
python3 code/visualizations/code/analysis_visualizations.py
python3 code/visualizations/code/waterbirds_detailed_analysis.py
```

The smoke-test experiment uses the synthetic `fake` dataset and random model weights, so it is intended to validate that the pipeline executes end to end without requiring external datasets or pretrained-weight downloads. Because no `--output-dir` is passed, this run saves its experiment outputs to:

- [outputs/results/controlled/default_run/](outputs/results/controlled/default_run/)

Each experiment run writes:

- `experiment_results_raw.csv`
- `experiment_results_summary.csv`
- `correlation_summary.csv`
- `calibration_summary.csv`
- `summary.json`

The visualization script writes figures to:

- [code/visualizations/figures/final/](code/visualizations/figures/final/)
- [code/visualizations/figures/diagnostics/](code/visualizations/figures/diagnostics/)

The Waterbirds analysis script writes tables and the Markdown report to:

- [data/processed_data/waterbirds_analysis/](data/processed_data/waterbirds_analysis/)

## Full Experiment Commands

### CIFAR-10

```bash
python3 code/main_codes/experiment_runner.py \
  --datasets cifar10 \
  --train-size 300 \
  --test-size 150 \
  --probe-epochs 8 \
  --trials 2 \
  --max-feature-dims 128 \
  --download \
  --output-dir outputs/results/controlled/results_final
```

Saved outputs:

- [outputs/results/controlled/results_final/](outputs/results/controlled/results_final/)

### BloodMNIST

```bash
python3 code/main_codes/experiment_runner.py \
  --datasets bloodmnist \
  --train-size 300 \
  --test-size 150 \
  --probe-epochs 8 \
  --trials 2 \
  --max-feature-dims 128 \
  --download \
  --output-dir outputs/results/controlled/results_bloodmnist
```

Saved outputs:

- [outputs/results/controlled/results_bloodmnist/](outputs/results/controlled/results_bloodmnist/)

### Waterbirds

```bash
python3 code/main_codes/experiment_runner.py \
  --datasets waterbirds \
  --waterbirds-root data/raw_data/waterbirds \
  --train-size 300 \
  --test-size 150 \
  --probe-epochs 8 \
  --trials 2 \
  --max-feature-dims 128 \
  --output-dir outputs/results/waterbirds/results_waterbirds
```

Saved outputs:

- [outputs/results/waterbirds/results_waterbirds/](outputs/results/waterbirds/results_waterbirds/)

## Additional Commands

Generate final and diagnostic figures from the saved result folders:

```bash
python3 code/visualizations/code/analysis_visualizations.py
```

Generate the Waterbirds subgroup analysis:

```bash
python3 code/visualizations/code/waterbirds_detailed_analysis.py
```

Run the Waterbirds reference evaluation:

```bash
python3 code/main_codes/waterbirds_reference_runner.py \
  --waterbirds-root data/raw_data/waterbirds \
  --output-dir outputs/results/improved/results_waterbirds_improved
```

Run the Waterbirds bias-mitigation comparison:

```bash
python3 code/main_codes/waterbirds_mitigation_runner.py \
  --waterbirds-root data/raw_data/waterbirds \
  --output-dir outputs/results/bias_mitigation/results_waterbirds_mitigation
```

## Included Results and Figures

Representative saved result snapshots are included under:

- [outputs/results/controlled/results_final/](outputs/results/controlled/results_final/)
- [outputs/results/controlled/results_bloodmnist/](outputs/results/controlled/results_bloodmnist/)
- [outputs/results/waterbirds/results_waterbirds/](outputs/results/waterbirds/results_waterbirds/)

Generated final figures are included under:

- [code/visualizations/figures/final/clean_accuracy_overview.png](code/visualizations/figures/final/clean_accuracy_overview.png)
- [code/visualizations/figures/final/controlled_drift_profiles.png](code/visualizations/figures/final/controlled_drift_profiles.png)
- [code/visualizations/figures/final/correlation_overview.png](code/visualizations/figures/final/correlation_overview.png)
- [code/visualizations/figures/final/waterbirds_drift_vs_drop.png](code/visualizations/figures/final/waterbirds_drift_vs_drop.png)
- [code/visualizations/figures/final/waterbirds_subgroup_accuracy.png](code/visualizations/figures/final/waterbirds_subgroup_accuracy.png)

Diagnostic figures and interpretation notes are included under:

- [code/visualizations/figures/diagnostics/](code/visualizations/figures/diagnostics/)

Waterbirds processed analysis files are included under:

- [data/processed_data/waterbirds_analysis/alignment_summary.csv](data/processed_data/waterbirds_analysis/alignment_summary.csv)
- [data/processed_data/waterbirds_analysis/environment_gap_summary.csv](data/processed_data/waterbirds_analysis/environment_gap_summary.csv)
- [data/processed_data/waterbirds_analysis/subgroup_summary.csv](data/processed_data/waterbirds_analysis/subgroup_summary.csv)
- [data/processed_data/waterbirds_analysis/waterbirds_detailed_analysis.md](data/processed_data/waterbirds_analysis/waterbirds_detailed_analysis.md)
- [data/processed_data/waterbirds_analysis/worst_group_summary.csv](data/processed_data/waterbirds_analysis/worst_group_summary.csv)

## Notes for Reviewers

- The smoke-test command is the fastest way to verify the execution path without external datasets.
- Full CIFAR-10 and BloodMNIST runs may download data when `--download` is supplied.
- Full Waterbirds runs require a local Waterbirds dataset at `data/raw_data/waterbirds/`.
- The saved figures and processed tables can be regenerated from the included result snapshots using the two visualization/analysis commands above.
