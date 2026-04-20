# Capstone Results Notes

This file summarizes the corrected methodology and the completed experiments in thesis-safe language.

## Corrected Methodology

The revised experiment pipeline fixes the main issues from the original notebook:

- drift types are evaluated separately instead of being mixed together
- classification degradation is measured explicitly through `accuracy_drop`
- feature drift is measured with both `KS` and `MMD`
- drift thresholds are calibrated from clean or reference windows
- multiple architectures are evaluated under the same protocol

The controlled corruption families are:

- Gaussian noise
- Motion blur
- Brightness shift

The backbone models are:

- ResNet-50
- EfficientNet-B0

## Dataset Roles

- `CIFAR-10`
  Controlled benchmark for synthetic corruption analysis.
- `BloodMNIST`
  Second controlled benchmark in a medical-image domain.
- `Waterbirds`
  Natural-shift benchmark for subgroup and spurious-correlation analysis.

## MMD Calibration Note

Clean-window MMD calibration can become numerically unstable on small reference windows. For threshold calibration, the pipeline therefore uses a biased MMD estimator so the monitoring statistic stays non-negative and produces more stable alert thresholds.

This affects threshold estimation only. The interpretation remains the same: larger MMD indicates stronger feature-space shift.

## CIFAR-10 Main Findings

These findings come from the stronger two-trial CIFAR-10 configuration.

Correlation between drift metrics and accuracy drop:

- ResNet-50
  - `accuracy_drop` vs `ks_mean`: `0.8956`
  - `accuracy_drop` vs `mmd`: `0.8606`
- EfficientNet-B0
  - `accuracy_drop` vs `ks_mean`: `0.9764`
  - `accuracy_drop` vs `mmd`: `0.8839`

Clean accuracy:

- ResNet-50: `0.7867`
- EfficientNet-B0: `0.8467`

Most damaging drift family:

- both models: `gaussian_noise / high`

Interpretation:

- on CIFAR-10, feature-space drift metrics are strongly associated with classification degradation under synthetic shift
- Gaussian noise is the most harmful of the tested corruption families

## BloodMNIST Main Findings

BloodMNIST extends the controlled methodology beyond natural object images.

Clean accuracy:

- ResNet-50: `0.6900`
- EfficientNet-B0: `0.7633`

Correlation between drift metrics and accuracy drop:

- ResNet-50
  - `accuracy_drop` vs `ks_mean`: `0.8661`
  - `accuracy_drop` vs `mmd`: `0.8739`
- EfficientNet-B0
  - `accuracy_drop` vs `ks_mean`: `0.9287`
  - `accuracy_drop` vs `mmd`: `0.9224`

Most damaging drift family:

- both models: `gaussian_noise / high`

Interpretation:

- the controlled pipeline generalizes beyond CIFAR-10 to a second benchmark
- feature drift remains strongly associated with accuracy degradation in a different image domain

## Waterbirds Main Findings

Waterbirds is intentionally evaluated differently from CIFAR-10 and BloodMNIST:

- the probe is trained on the train split
- matched validation examples define the reference distribution
- matched and mismatched validation/test groups are compared against that reference
- label-background subgroup environments are evaluated directly

Correlation between drift metrics and accuracy drop:

- ResNet-50
  - `accuracy_drop` vs `ks_mean`: `0.3346`
  - `accuracy_drop` vs `mmd`: `0.3314`
- EfficientNet-B0
  - `accuracy_drop` vs `ks_mean`: `0.3454`
  - `accuracy_drop` vs `mmd`: `0.3288`

Largest observed degradation:

- ResNet-50: `val_waterbird_land`, accuracy drop `0.5540`
- EfficientNet-B0: `test_waterbird_land`, accuracy drop `0.6267`

Interpretation:

- under natural subgroup/environment shift, drift metrics still move in the correct direction, but the relationship to performance degradation is weaker than in the controlled benchmarks
- Waterbirds strengthens the capstone by testing the method on realistic spurious-correlation shift rather than only synthetic perturbations

## Safe Claims by Dataset

### CIFAR-10

Supports:

- strong evidence that KS and MMD track degradation under synthetic corruption
- ranking corruption families by harmfulness
- controlled architecture comparison

Does not support:

- broad real-world generalization by itself

### BloodMNIST

Supports:

- the same controlled methodology in a second image domain
- strong evidence that feature drift still tracks degradation outside the CIFAR setting

Does not support:

- natural-shift claims by itself

### Waterbirds

Supports:

- a realistic subgroup/environment-shift evaluation
- evidence that drift metrics have a positive but weaker relationship with subgroup accuracy degradation

Does not support:

- claiming the same strong drift-performance relationship seen in controlled corruption benchmarks
- broad claims beyond the tested Waterbirds subgroup structure

## Final Thesis Caution

Use precise wording in the final report:

- say `on CIFAR-10` or `on BloodMNIST` for synthetic corruption findings
- discuss Waterbirds separately as a natural-shift benchmark
- avoid claiming full production readiness unless you also evaluate alert stability, false positives, and operational policy
