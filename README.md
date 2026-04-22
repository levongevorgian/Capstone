# Feature-Space Drift Detection for Image Classification

This repository contains the cleaned capstone project for evaluating whether label-free feature drift metrics can serve as practical proxies for model degradation in image classification systems.

The project studies that question across three benchmark settings:

- `CIFAR-10` for controlled synthetic corruptions
- `BloodMNIST` for a second controlled benchmark in a medical-image domain
- `Waterbirds` for natural subgroup and spurious-correlation shift

## Project Layout

```text
CAP/
├── paper/                      # Paper PDF 
├── code/
│   ├── main_codes/             # experiment runners
│   ├── visualizations/         # figure-generation code and figure assets
│   └── utils/                  # shared helpers
├── data/
│   ├── raw_data/               # local datasets (gitignored)
│   └── processed_data/         # processed analysis tables kept with the repo
├── notebooks/                  # narrative notebook
├── outputs/                    # result snapshots and local run outputs
├── docs/                       # internal notes and file guide
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE
```

## Main Files

- [code/main_codes/experiment_runner.py](/Users/levongevorgyan/Desktop/CAP/code/main_codes/experiment_runner.py) runs the main controlled-benchmark and Waterbirds experiments.
- [code/main_codes/waterbirds_reference_runner.py](/Users/levongevorgyan/Desktop/CAP/code/main_codes/waterbirds_reference_runner.py) evaluates the class-conditional Waterbirds reference design.
- [code/main_codes/waterbirds_mitigation_runner.py](/Users/levongevorgyan/Desktop/CAP/code/main_codes/waterbirds_mitigation_runner.py) compares subgroup-aware probe-training strategies.
- [code/visualizations/code/analysis_visualizations.py](/Users/levongevorgyan/Desktop/CAP/code/visualizations/code/analysis_visualizations.py) generates the diagnostic figures.
- [code/visualizations/code/waterbirds_detailed_analysis.py](/Users/levongevorgyan/Desktop/CAP/code/visualizations/code/waterbirds_detailed_analysis.py) produces the Waterbirds subgroup analysis tables and report.

## Setup

Create a Python environment and install the requirements:

```bash
python3 -m pip install -r requirements.txt
```

Optional import check:

```bash
python3 -m py_compile \
  code/main_codes/experiment_runner.py \
  code/main_codes/waterbirds_reference_runner.py \
  code/main_codes/waterbirds_mitigation_runner.py \
  code/visualizations/code/analysis_visualizations.py \
  code/visualizations/code/waterbirds_detailed_analysis.py
```

## Data Locations

Expected local dataset paths:

- `data/raw_data/` for CIFAR-10 cache files and BloodMNIST downloads
- `data/raw_data/waterbirds/` for the Waterbirds dataset root containing `metadata.csv`

Raw datasets are ignored by Git and should remain local.

## Running The Core Experiments

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
  --output-dir outputs/results/controlled/results_cifar10
```

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

### Smoke Test

```bash
python3 code/main_codes/experiment_runner.py \
  --datasets fake \
  --train-size 32 \
  --test-size 24 \
  --probe-epochs 1 \
  --trials 1 \
  --random-weights
```

## Generating Figures And Reports

Generate the diagnostic figures:

```bash
python3 code/visualizations/code/analysis_visualizations.py
```

Generate the Waterbirds subgroup report:

```bash
python3 code/visualizations/code/waterbirds_detailed_analysis.py
```

By default, figures are written to:

- `code/visualizations/figures/final/`
- `code/visualizations/figures/diagnostics/`

The Waterbirds analysis tables and report are written to:

- `data/processed_data/waterbirds_analysis/`

## Result Files

Each experiment run writes:

- `experiment_results_raw.csv`
- `experiment_results_summary.csv`
- `correlation_summary.csv`
- `calibration_summary.csv`
- `summary.json`

The repository currently keeps representative result snapshots under `outputs/results/` and uses `code/visualizations/figures/` for the curated figures included with the project.

## Notes On Interpretation

- Controlled corruption results support strong drift-versus-degradation claims within the tested settings.
- Waterbirds should be interpreted separately as a natural subgroup-shift benchmark.
- The strongest Waterbirds takeaway is subgroup disparity and worst-group collapse, not a single pooled drift-performance correlation.
