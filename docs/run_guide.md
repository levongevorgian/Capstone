# Running The Experiments

The main workflow lives in `code/main_codes/experiment_runner.py`.

Supported benchmark modes:

- `cifar10`
- `bloodmnist`
- `waterbirds`
- `cifar100`
- `imagenet_local`
- `fake` for smoke tests

## Common Outputs

Each run writes:

- `experiment_results_raw.csv`
- `experiment_results_summary.csv`
- `correlation_summary.csv`
- `calibration_summary.csv`
- `summary.json`

## Recommended Commands

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

## Other Useful Commands

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

### Improved Waterbirds Reference Design

```bash
python3 code/main_codes/waterbirds_reference_runner.py \
  --waterbirds-root data/raw_data/waterbirds \
  --output-dir outputs/results/improved/results_waterbirds_improved
```

### Waterbirds Bias-Mitigation Comparison

```bash
python3 code/main_codes/waterbirds_mitigation_runner.py \
  --waterbirds-root data/raw_data/waterbirds \
  --output-dir outputs/results/bias_mitigation/results_waterbirds_mitigation
```

## Dataset Notes

- `CIFAR-10` downloads into `data/raw_data/` through `torchvision`.
- `BloodMNIST` downloads into `data/raw_data/` through `medmnist`.
- `Waterbirds` must already exist at `data/raw_data/waterbirds/` with `metadata.csv`.
- Waterbirds is intentionally run separately because its subgroup-shift evaluation differs from the controlled corruption protocol.
