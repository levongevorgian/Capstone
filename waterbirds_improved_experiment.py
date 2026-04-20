#!/usr/bin/env python3
"""Improved Waterbirds evaluation with class-conditional references.

The original Waterbirds path in ``cap_experiments.py`` compares every
target group to one pooled matched-validation reference. That mixes two
effects:

1. harmful environment shift
2. harmless label-composition differences

This standalone script keeps the main pipeline untouched and evaluates
Waterbirds with a class-conditional matched-validation reference that
matches each target group's label mix. That makes the benchmark better
aligned with what Waterbirds is intended to test.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from cap_experiments import (
    MAX_FEATURE_DIMS,
    SEED,
    aggregate_rows,
    base_transform,
    batched_images,
    calibrate_clean_thresholds,
    compute_ks_statistic_fast,
    compute_mmd,
    correlation_summary,
    dataset_to_batch,
    evaluate_accuracy,
    load_models,
    load_waterbirds_spec,
    sample_fixed_indices,
    set_seed,
    train_linear_probe,
    write_csv,
)


def log(message: str) -> None:
    print(message, flush=True)


def build_target_specs(metadata) -> list[tuple[str, object]]:
    target_specs = [
        ("val_matched", (metadata["split_name"] == "val") & (metadata["aligned"] == 1)),
        ("val_mismatched", (metadata["split_name"] == "val") & (metadata["aligned"] == 0)),
        ("test_matched", (metadata["split_name"] == "test") & (metadata["aligned"] == 1)),
        ("test_mismatched", (metadata["split_name"] == "test") & (metadata["aligned"] == 0)),
    ]
    for split_name in ["val", "test"]:
        for group_name in sorted(metadata["group_name"].unique()):
            target_specs.append(
                (
                    f"{split_name}_{group_name}",
                    (metadata["split_name"] == split_name) & (metadata["group_name"] == group_name),
                )
            )
    return target_specs


def extract_index_feature_cache(
    model,
    probe_layer: str,
    dataset,
    indices: Sequence[int],
    batch_size: int,
) -> Dict[int, np.ndarray]:
    from cap_experiments import FeatureExtractor

    unique_indices = [int(idx) for idx in dict.fromkeys(int(idx) for idx in indices)]
    batch = dataset_to_batch(dataset, unique_indices)
    extractor = FeatureExtractor(model, [probe_layer])
    model.eval()
    try:
        import torch

        with torch.no_grad():
            for image_batch in batched_images(batch.images, base_transform(), batch_size):
                model(image_batch.to(next(model.parameters()).device))
        features = extractor.activations[probe_layer]
        if not features:
            raise RuntimeError(f"No activations captured for layer {probe_layer}.")
        matrix = np.concatenate([chunk.numpy() for chunk in features], axis=0)
    finally:
        extractor.close()
    return {idx: matrix[pos] for pos, idx in enumerate(unique_indices)}


def gather_feature_matrix(index_feature_cache: Dict[int, np.ndarray], indices: Sequence[int]) -> np.ndarray:
    return np.stack([index_feature_cache[int(idx)] for idx in indices], axis=0)


def make_class_conditional_reference_indices(
    metadata,
    reference_pools_by_label: Dict[int, Sequence[int]],
    target_indices: Sequence[int],
    seed: int,
) -> list[int]:
    labels = metadata.loc[list(target_indices), "y"].astype(int).to_numpy()
    reference_indices: list[int] = []
    for label in sorted(np.unique(labels)):
        pool = list(reference_pools_by_label[int(label)])
        needed = int((labels == label).sum())
        if not pool:
            raise RuntimeError(f"Missing matched-validation reference pool for label {label}.")
        reference_indices.extend(sample_fixed_indices(pool, needed, seed + int(label) * 1000))
    return reference_indices


def summarize_findings(summary_rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for row in summary_rows:
        grouped.setdefault(row["model"], []).append(row)

    findings = []
    for model_name, rows in grouped.items():
        best_drop = max(rows, key=lambda row: row.get("accuracy_drop", float("-inf")))
        best_mmd = max(rows, key=lambda row: row.get("mmd", float("-inf")))
        findings.append(
            {
                "dataset": "waterbirds",
                "model": model_name,
                "largest_accuracy_drop": {
                    "target_group": best_drop["target_group"],
                    "accuracy_drop": best_drop["accuracy_drop"],
                },
                "largest_mmd": {
                    "target_group": best_mmd["target_group"],
                    "mmd": best_mmd["mmd"],
                },
            }
        )
    return {"high_level_findings": findings}


def run_waterbirds_improved_experiment(
    waterbirds_root: str,
    train_size: int,
    test_size: int,
    batch_size: int,
    probe_epochs: int,
    trials: int,
    output_dir: Path,
    random_weights: bool = False,
) -> Dict[str, object]:
    set_seed(SEED)
    output_dir.mkdir(parents=True, exist_ok=True)
    log(f"[start] output_dir={output_dir}")

    spec = load_waterbirds_spec(waterbirds_root)
    metadata = spec.metadata
    dataset = spec.dataset
    log(f"[data] loaded Waterbirds metadata rows={len(metadata)}")
    models_cfg = load_models(random_weights=random_weights)
    log(f"[models] loaded {list(models_cfg.keys())}")

    train_indices_all = metadata.index[metadata["split_name"] == "train"].tolist()
    reference_pools_by_label = {
        int(label): metadata.index[
            (metadata["split_name"] == "val") & (metadata["aligned"] == 1) & (metadata["y"] == int(label))
        ].tolist()
        for label in sorted(metadata["y"].unique())
    }
    if not train_indices_all or any(len(indices) == 0 for indices in reference_pools_by_label.values()):
        raise RuntimeError("Waterbirds requires train data and matched validation reference pools for both labels.")

    target_specs = build_target_specs(metadata)

    raw_rows: List[Dict[str, object]] = []
    correlations = []
    calibration_rows = []

    for trial_idx in range(trials):
        trial_seed = SEED + trial_idx * 101
        log(f"[trial {trial_idx}] sampling groups")
        train_indices = sample_fixed_indices(train_indices_all, train_size, trial_seed)

        sampled_targets: Dict[str, list[int]] = {}
        sampled_references: Dict[str, list[int]] = {}
        for target_group, mask in target_specs:
            target_indices_all = metadata.index[mask].tolist()
            if not target_indices_all:
                continue
            target_indices = sample_fixed_indices(target_indices_all, test_size, trial_seed + len(target_group))
            sampled_targets[target_group] = target_indices
            sampled_references[target_group] = make_class_conditional_reference_indices(
                metadata,
                reference_pools_by_label,
                target_indices,
                seed=trial_seed + len(target_group) * 17,
            )

        train_batch = dataset_to_batch(dataset, train_indices)

        for model_name, cfg in models_cfg.items():
            log(f"[trial {trial_idx}] [model {model_name}] extracting cached features")
            model = cfg["model"]
            probe_layer = cfg["probe_layer"]
            all_needed_indices = list(train_indices)
            for indices in sampled_targets.values():
                all_needed_indices.extend(indices)
            for indices in sampled_references.values():
                all_needed_indices.extend(indices)
            index_feature_cache = extract_index_feature_cache(
                model,
                probe_layer,
                dataset,
                all_needed_indices,
                batch_size=batch_size,
            )
            train_features = gather_feature_matrix(index_feature_cache, train_indices)

            probe = train_linear_probe(
                train_features,
                train_batch.labels,
                num_classes=spec.num_classes,
                epochs=probe_epochs,
                batch_size=batch_size,
            )
            log(f"[trial {trial_idx}] [model {model_name}] trained probe")

            reference_for_calibration = gather_feature_matrix(index_feature_cache, sampled_references["val_matched"])
            thresholds = calibrate_clean_thresholds(
                reference_for_calibration,
                window_size=min(32, max(8, reference_for_calibration.shape[0] // 2)),
                num_windows=12,
                seed=trial_seed,
            )
            log(f"[trial {trial_idx}] [model {model_name}] calibrated thresholds")
            calibration_rows.append(
                {
                    "dataset": "waterbirds",
                    "model": model_name,
                    "trial": trial_idx,
                    "reference_group": "class_conditional_val_matched",
                    "ks_threshold": thresholds["ks_threshold"],
                    "mmd_threshold": thresholds["mmd_threshold"],
                    "reference_accuracy": evaluate_accuracy(
                        probe,
                        reference_for_calibration,
                        metadata.loc[sampled_references["val_matched"], "y"].astype(int).tolist(),
                    ),
                }
            )

            for target_group, target_indices in sampled_targets.items():
                reference_indices = sampled_references[target_group]
                reference_features = gather_feature_matrix(index_feature_cache, reference_indices)
                target_features = gather_feature_matrix(index_feature_cache, target_indices)

                reference_labels = metadata.loc[reference_indices, "y"].astype(int).tolist()
                target_labels = metadata.loc[target_indices, "y"].astype(int).tolist()
                reference_accuracy = evaluate_accuracy(probe, reference_features, reference_labels)
                target_accuracy = evaluate_accuracy(probe, target_features, target_labels)

                split_name = metadata.loc[target_indices[0], "split_name"]
                alignment_name = (
                    metadata.loc[target_indices, "alignment_name"].mode().iloc[0]
                    if "alignment_name" in metadata.columns
                    else "mixed"
                )

                ks_stats = compute_ks_statistic_fast(
                    reference_features,
                    target_features,
                    max_feature_dims=MAX_FEATURE_DIMS,
                )
                mmd_value = compute_mmd(
                    reference_features,
                    target_features,
                    max_feature_dims=MAX_FEATURE_DIMS,
                )

                raw_rows.append(
                    {
                        "dataset": "waterbirds",
                        "model": model_name,
                        "trial": trial_idx,
                        "reference_group": "class_conditional_val_matched",
                        "reference_strategy": "label_matched_validation_mix",
                        "target_group": target_group,
                        "target_split": split_name,
                        "alignment": alignment_name,
                        "clean_accuracy": float(reference_accuracy),
                        "drift_accuracy": float(target_accuracy),
                        "accuracy_drop": float(reference_accuracy - target_accuracy),
                        "probe_layer": probe_layer,
                        "ks_threshold": thresholds["ks_threshold"],
                        "mmd_threshold": thresholds["mmd_threshold"],
                        "group_size": len(target_indices),
                        "reference_size": len(reference_indices),
                        "ks_mean": ks_stats["ks_mean"],
                        "ks_max": ks_stats["ks_max"],
                        "pvalue_mean": ks_stats["pvalue_mean"],
                        "mmd": mmd_value,
                        "ks_alert": bool(
                            not np.isnan(thresholds["ks_threshold"]) and ks_stats["ks_mean"] > thresholds["ks_threshold"]
                        ),
                        "mmd_alert": bool(
                            not np.isnan(thresholds["mmd_threshold"]) and mmd_value > thresholds["mmd_threshold"]
                        ),
                    }
                )
            log(f"[trial {trial_idx}] [model {model_name}] finished target groups")

    for model_name in models_cfg.keys():
        model_rows = [row for row in raw_rows if row["model"] == model_name]
        correlations.append(
            {
                "dataset": "waterbirds",
                "model": model_name,
                "reference_strategy": "label_matched_validation_mix",
                **correlation_summary(model_rows),
            }
        )

    summary_rows = aggregate_rows(
        raw_rows,
        [
            "dataset",
            "model",
            "reference_group",
            "reference_strategy",
            "target_group",
            "target_split",
            "alignment",
            "probe_layer",
        ],
    )
    findings = summarize_findings(summary_rows)

    raw_csv = output_dir / "experiment_results_raw.csv"
    summary_csv = output_dir / "experiment_results_summary.csv"
    corr_csv = output_dir / "correlation_summary.csv"
    calib_csv = output_dir / "calibration_summary.csv"
    write_csv(raw_csv, raw_rows)
    write_csv(summary_csv, summary_rows)
    write_csv(corr_csv, correlations)
    write_csv(calib_csv, calibration_rows)
    log("[done] wrote csv outputs")

    summary = {
        "seed": SEED,
        "datasets": ["waterbirds"],
        "dataset_notes": {"waterbirds": spec.note},
        "train_size": train_size,
        "test_size": test_size,
        "trials": trials,
        "random_weights": random_weights,
        "waterbirds_design": {
            "reference_group": "class-conditional matched validation subset",
            "reference_strategy": "per-target label-matched reference mix",
            "targets": "validation/test alignment groups and class-background environments",
            "claim_scope": "natural distribution shift with reduced label-composition confounding",
        },
        "output_files": {
            "raw_results": str(raw_csv.resolve()),
            "summary_results": str(summary_csv.resolve()),
            "correlations": str(corr_csv.resolve()),
            "calibration": str(calib_csv.resolve()),
        },
        "correlations": correlations,
        **findings,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    log("[done] wrote summary.json")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Improved Waterbirds experiment with class-conditional references.")
    parser.add_argument("--waterbirds-root", type=str, default="./data/waterbirds")
    parser.add_argument("--train-size", type=int, default=300)
    parser.add_argument("--test-size", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--probe-epochs", type=int, default=8)
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=Path("results_waterbirds_improved"))
    parser.add_argument("--random-weights", action="store_true")
    args = parser.parse_args()

    summary = run_waterbirds_improved_experiment(
        waterbirds_root=args.waterbirds_root,
        train_size=args.train_size,
        test_size=args.test_size,
        batch_size=args.batch_size,
        probe_epochs=args.probe_epochs,
        trials=args.trials,
        output_dir=args.output_dir,
        random_weights=args.random_weights,
    )
    print(json.dumps(summary["correlations"], indent=2))


if __name__ == "__main__":
    main()
