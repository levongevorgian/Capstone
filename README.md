# Feature-Space Drift Detection for Image Classification

This repository contains the code, result snapshots, figures, and supporting materials for experiments on label-free feature drift detection in image classification.

The project evaluates whether feature-space drift statistics can serve as practical indicators of model performance degradation across:

- `CIFAR-10` under controlled synthetic corruptions
- `BloodMNIST` under controlled synthetic corruptions
- `Waterbirds` under natural subgroup and spurious-correlation shift

## Repository Structure

```text
├── paper/                      # Paper PDF
├── code/
│   ├── main_codes/                    # experiment runners
│   ├── utils/                         # shared plotting utilities
│   └── visualizations/
│       ├── code/                      # figure and analysis scripts
│       └── figures/                   # generated final and diagnostic figures
├── data/
│   ├── raw_data/                      # local datasets and caches
│   └── processed_data/
│       └── waterbirds_analysis/       # generated Waterbirds analysis tables and report
├── docs/
│   └── run_guide.md                   # command reference
├── outputs/
│   ├── figures/                       # reserved export directory for additional figure copies
│   ├── reports/                       # reserved export directory for additional report copies
│   └── results/                       # experiment outputs and archived runs
├── paper/
│   ├── figures/                       # paper-specific figure assets
│   └── proposal/                      # paper PDF
├── LICENSE
├── README.md
└── requirements.txt
```

## Main Components

- [code/main_codes/experiment_runner.py](/Users/levongevorgyan/Desktop/CAP/code/main_codes/experiment_runner.py) runs the main controlled-benchmark and Waterbirds experiments.
<<<<<<< HEAD
- [code/main_codes/waterbirds_reference_runner.py](/Users/levongevorgyan/Desktop/CAP/code/main_codes/waterbirds_reference_runner.py) evaluates the class-conditional Waterbirds reference design.
- [code/main_codes/waterbirds_mitigation_runner.py](/Users/levongevorgyan/Desktop/CAP/code/main_codes/waterbirds_mitigation_runner.py) compares subgroup-aware probe-training strategies.
- [code/visualizations/code/analysis_visualizations.py](/Users/levongevorgyan/Desktop/CAP/code/visualizations/code/analysis_visualizations.py) generates the diagnostic figures.
- [code/visualizations/code/waterbirds_detailed_analysis.py](/Users/levongevorgyan/Desktop/CAP/code/visualizations/code/waterbirds_detailed_analysis.py) produces the Waterbirds subgroup analysis tables and report.
=======
- [code/main_codes/waterbirds_reference_runner.py](/Users/levongevorgyan/Desktop/CAP/code/main_codes/waterbirds_reference_runner.py) evaluates Waterbirds with class-conditional reference groups.
- [code/main_codes/waterbirds_mitigation_runner.py](/Users/levongevorgyan/Desktop/CAP/code/main_codes/waterbirds_mitigation_runner.py) compares probe-training strategies for Waterbirds subgroup robustness.
- [code/visualizations/code/analysis_visualizations.py](/Users/levongevorgyan/Desktop/CAP/code/visualizations/code/analysis_visualizations.py) generates final and diagnostic figures from saved experiment outputs.
- [code/visualizations/code/waterbirds_detailed_analysis.py](/Users/levongevorgyan/Desktop/CAP/code/visualizations/code/waterbirds_detailed_analysis.py) produces Waterbirds subgroup summaries and a Markdown report.
- [docs/run_guide.md](/Users/levongevorgyan/Desktop/CAP/docs/run_guide.md) provides a concise command reference.
>>>>>>> 88338b3 (Add improved versions of files)

## Setup

Install the Python dependencies:

```bash
python3 -m pip install -r requirements.txt
```

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

Expected dataset locations:

- `data/raw_data/` for CIFAR-10 cache files and BloodMNIST downloads
- `data/raw_data/waterbirds/` for the Waterbirds dataset root containing `metadata.csv`
- a separate local ImageFolder-style directory if `imagenet_local` is used

Large raw datasets are kept local and are not intended to be versioned with the repository.

## Running Experiments

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

### Waterbirds Reference Evaluation

<<<<<<< HEAD
Generate the diagnostic figures:
=======
```bash
python3 code/main_codes/waterbirds_reference_runner.py \
  --waterbirds-root data/raw_data/waterbirds \
  --output-dir outputs/results/improved/results_waterbirds_improved
```

### Waterbirds Bias-Mitigation Comparison

```bash
python3 code/main_codes/waterbirds_mitigation_runner.py \
  --waterbirds-root data/raw_data/waterbirds \
  --output-dir outputs/results/bias_mitigation/results_waterbirds_mitigation_resnet
```

## Generating Figures and Analysis Outputs

Generate the final and diagnostic figures:
>>>>>>> 88338b3 (Add improved versions of files)

```bash
python3 code/visualizations/code/analysis_visualizations.py
```

This script writes figures to:

- `code/visualizations/figures/final/`
- `code/visualizations/figures/diagnostics/`

Generate the Waterbirds subgroup analysis:

```bash
python3 code/visualizations/code/waterbirds_detailed_analysis.py
```

This script writes tables and the Markdown report to:

- `data/processed_data/waterbirds_analysis/`

## Output Locations

- `outputs/results/` stores experiment run outputs such as CSV summaries and JSON summaries.
- `code/visualizations/figures/` stores the project's generated figure assets currently used by the repository.
- `data/processed_data/waterbirds_analysis/` stores derived Waterbirds analysis tables and the generated Markdown report.
- `outputs/figures/` and `outputs/reports/` currently function as reserved export directories for additional deliverables, but they are not the primary destinations used by the current scripts.

Each standard experiment run produces:

- `experiment_results_raw.csv`
- `experiment_results_summary.csv`
- `correlation_summary.csv`
- `calibration_summary.csv`
- `summary.json`

## Included Results

The repository includes representative output snapshots under:

- `outputs/results/controlled/`
- `outputs/results/waterbirds/`
- `outputs/results/improved/`
- `outputs/results/bias_mitigation/`
- `outputs/results/archive/`

These directories contain saved runs for analysis, comparison, and figure generation.
