# Running the Experiments

The main workflow lives in `cap_experiments.py`.

This repository supports three primary datasets:

- `cifar10`
- `bloodmnist`
- `waterbirds`

It also supports:

- `cifar100` as an extra controlled benchmark
- `imagenet_local` for an ImageFolder-style local dataset
- `fake` for smoke testing

## Common Outputs

Each run writes the following files into the selected output directory:

- `experiment_results_raw.csv`
- `experiment_results_summary.csv`
- `correlation_summary.csv`
- `calibration_summary.csv`
- `summary.json`

## Recommended Commands

### CIFAR-10

```bash
python3 cap_experiments.py --datasets cifar10 --train-size 300 --test-size 150 --probe-epochs 8 --trials 2 --max-feature-dims 128 --download --output-dir results_cifar10
```

### BloodMNIST

```bash
python3 cap_experiments.py --datasets bloodmnist --train-size 300 --test-size 150 --probe-epochs 8 --trials 2 --max-feature-dims 128 --download --output-dir results_bloodmnist
```

### Waterbirds

```bash
python3 cap_experiments.py --datasets waterbirds --waterbirds-root ./data/waterbirds --train-size 300 --test-size 150 --probe-epochs 8 --trials 2 --max-feature-dims 128 --output-dir results_waterbirds
```

## Other Useful Commands

### Smoke Test

```bash
python3 cap_experiments.py --datasets fake --train-size 32 --test-size 24 --probe-epochs 1 --trials 1 --random-weights
```

### CIFAR-10 With Default Settings

```bash
python3 cap_experiments.py --datasets cifar10 --download
```

### Local ImageNet-Style Folder

```bash
python3 cap_experiments.py --datasets imagenet_local --imagenet-root /path/to/imagenet_root --train-size 300 --test-size 150 --probe-epochs 8 --trials 2 --max-feature-dims 128 --output-dir results_imagenet_local
```

## Dataset Notes

### CIFAR-10

- Downloaded automatically by `torchvision` when `--download` is used.
- Uses the controlled corruption families:
  - `gaussian_noise`
  - `motion_blur`
  - `brightness_shift`

### BloodMNIST

- Requires the `medmnist` package.
- Can be downloaded automatically when `--download` is used.
- Uses the same controlled corruption protocol as CIFAR-10.

### Waterbirds

- Requires a local dataset root with `metadata.csv`.
- The official Stanford DRO dataset variant used in this project is:
  `waterbird_complete95_forest2water2`
- The loader accepts image-path columns named:
  `img_filename`, `filepath`, or `path`
- Images may live either directly under the dataset root or under `images/`.
- Waterbirds is evaluated separately because it measures natural subgroup/environment shift rather than synthetic severity levels.

## Interpretation Notes

- `accuracy_drop` measures how much the linear probe degrades under shift.
- `pearson_accuracy_vs_ks` and `pearson_accuracy_vs_mmd` summarize whether feature drift aligns with worse performance.
- `ks_threshold` and `mmd_threshold` are calibrated from clean or matched-reference windows rather than hand-picked.
- `--max-feature-dims` reduces runtime by subsampling feature dimensions for KS and MMD.
