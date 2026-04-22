#!/usr/bin/env python3
"""Waterbirds bias-mitigation experiments with frozen backbones.

This keeps the architecture fixed and changes only how the linear probe is
trained on Waterbirds features:

- baseline: standard shuffled minibatches
- balanced_alignment_sampler: equalize matched vs mismatched sampling
- subgroup_loss_reweight: inverse-frequency loss weighting by subgroup
- subgroup_oversample: oversample 4-way subgroups to the max group count
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from experiment_runner import (
    DEFAULT_WATERBIRDS_ROOT,
    DEVICE,
    FeatureExtractor,
    LinearProbe,
    aggregate_rows,
    base_transform,
    batched_images,
    dataset_to_batch,
    load_models,
    load_waterbirds_spec,
    sample_fixed_indices,
    set_seed,
    write_csv,
)


def log(message: str) -> None:
    print(message, flush=True)


def subgroup_name(row) -> str:
    return f"{row['label_name']}_{row['place_name']}"


def build_target_specs(metadata) -> list[tuple[str, object]]:
    target_specs = []
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
    unique_indices = [int(idx) for idx in dict.fromkeys(int(idx) for idx in indices)]
    batch = dataset_to_batch(dataset, unique_indices)
    extractor = FeatureExtractor(model, [probe_layer])
    model.eval()
    with torch.no_grad():
        for image_batch in batched_images(batch.images, base_transform(), batch_size):
            model(image_batch.to(DEVICE))
    try:
        chunks = extractor.activations[probe_layer]
        if not chunks:
            raise RuntimeError(f"No features captured for probe layer {probe_layer}")
        matrix = torch.cat(chunks).numpy()
    finally:
        extractor.close()
    return {idx: matrix[pos] for pos, idx in enumerate(unique_indices)}


def gather_feature_matrix(index_feature_cache: Dict[int, np.ndarray], indices: Sequence[int]) -> np.ndarray:
    return np.stack([index_feature_cache[int(idx)] for idx in indices], axis=0)


def make_balanced_oversample_indices(group_ids: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    groups = sorted(np.unique(group_ids))
    max_count = max(int(np.sum(group_ids == group)) for group in groups)
    chosen = []
    for group in groups:
        pool = np.flatnonzero(group_ids == group)
        sampled = rng.choice(pool, size=max_count, replace=True)
        chosen.extend(sampled.tolist())
    rng.shuffle(chosen)
    return np.asarray(chosen, dtype=int)


def train_probe_with_strategy(
    train_features: np.ndarray,
    train_labels: Sequence[int],
    subgroup_ids: Sequence[int],
    alignment_ids: Sequence[int],
    num_classes: int,
    strategy: str,
    epochs: int,
    batch_size: int,
    lr: float = 1e-2,
    weight_decay: float = 1e-4,
    seed: int = 42,
) -> LinearProbe:
    x = torch.from_numpy(train_features).float()
    y = torch.tensor(train_labels, dtype=torch.long)
    subgroup_t = torch.tensor(subgroup_ids, dtype=torch.long)
    alignment_t = torch.tensor(alignment_ids, dtype=torch.long)
    base_dataset = torch.utils.data.TensorDataset(x, y, subgroup_t, alignment_t)

    sampler = None
    dataset = base_dataset

    subgroup_count_tensor = torch.bincount(subgroup_t).float()

    if strategy == "baseline":
        pass
    elif strategy == "balanced_alignment_sampler":
        counts = torch.bincount(alignment_t)
        sample_weights = (1.0 / counts[alignment_t].float()).double()
        sampler = torch.utils.data.WeightedRandomSampler(
            sample_weights,
            num_samples=len(alignment_t),
            replacement=True,
        )
    elif strategy == "subgroup_loss_reweight":
        counts = torch.bincount(subgroup_t)
        loss_weights = 1.0 / counts[subgroup_t].float()
        loss_weights = loss_weights / loss_weights.mean()
    elif strategy == "subgroup_oversample":
        oversampled_idx = make_balanced_oversample_indices(np.asarray(subgroup_ids), seed=seed)
        dataset = torch.utils.data.TensorDataset(
            x[oversampled_idx],
            y[oversampled_idx],
            subgroup_t[oversampled_idx],
            alignment_t[oversampled_idx],
        )
    else:
        raise ValueError(f"Unsupported strategy: {strategy}")

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
    )

    probe = LinearProbe(train_features.shape[1], num_classes=num_classes).to(DEVICE)
    optimizer = optim.Adam(probe.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(reduction="none")

    probe.train()
    for _ in range(epochs):
        for batch in loader:
            batch_x, batch_y, batch_subgroup, _batch_alignment = batch
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)
            batch_subgroup = batch_subgroup.to(DEVICE)
            optimizer.zero_grad()
            logits = probe(batch_x)
            losses = criterion(logits, batch_y)
            if strategy == "subgroup_loss_reweight":
                batch_weights = (1.0 / subgroup_count_tensor.to(DEVICE)[batch_subgroup]).detach()
                batch_weights = batch_weights / batch_weights.mean()
                loss = (losses * batch_weights).mean()
            else:
                loss = losses.mean()
            loss.backward()
            optimizer.step()

    return probe.eval()


def evaluate_probe_by_groups(probe: LinearProbe, features: np.ndarray, labels: Sequence[int], batch_size: int = 256) -> np.ndarray:
    x = torch.from_numpy(features).float()
    dataset = torch.utils.data.TensorDataset(x)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    preds = []
    with torch.no_grad():
        for (batch_x,) in loader:
            logits = probe(batch_x.to(DEVICE))
            preds.append(torch.argmax(logits, dim=1).cpu().numpy())
    return np.concatenate(preds, axis=0)


def compute_group_accuracy(preds: np.ndarray, labels: Sequence[int]) -> float:
    labels_arr = np.asarray(labels, dtype=int)
    return float(np.mean(preds == labels_arr)) if len(labels_arr) else float("nan")


def run_bias_mitigation(
    waterbirds_root: str,
    train_size: int,
    eval_size: int,
    batch_size: int,
    probe_epochs: int,
    output_dir: Path,
    model_names: Sequence[str] | None = None,
) -> Dict[str, object]:
    set_seed(42)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    output_dir.mkdir(parents=True, exist_ok=True)

    spec = load_waterbirds_spec(waterbirds_root)
    metadata = spec.metadata.copy()
    dataset = spec.dataset
    target_specs = build_target_specs(metadata)
    models_cfg = load_models(random_weights=False)
    if model_names:
        models_cfg = {name: models_cfg[name] for name in model_names}

    train_indices_all = metadata.index[metadata["split_name"] == "train"].tolist()
    trial_seed = 42
    train_indices = sample_fixed_indices(train_indices_all, train_size, trial_seed)
    train_meta = metadata.loc[train_indices].copy()
    train_meta["subgroup_name"] = train_meta.apply(subgroup_name, axis=1)
    subgroup_order = {name: idx for idx, name in enumerate(sorted(train_meta["subgroup_name"].unique()))}
    train_subgroup_ids = train_meta["subgroup_name"].map(subgroup_order).astype(int).to_numpy()
    train_alignment_ids = train_meta["aligned"].astype(int).to_numpy()
    train_labels = train_meta["y"].astype(int).tolist()

    sampled_targets: Dict[str, list[int]] = {}
    for target_group, mask in target_specs:
        indices_all = metadata.index[mask].tolist()
        sampled_targets[target_group] = sample_fixed_indices(indices_all, eval_size, trial_seed + len(target_group))

    strategies = [
        "baseline",
        "balanced_alignment_sampler",
        "subgroup_loss_reweight",
        "subgroup_oversample",
    ]
    raw_rows: List[Dict[str, object]] = []

    for model_name, cfg in models_cfg.items():
        log(f"[model {model_name}] extracting features")
        probe_layer = cfg["probe_layer"]
        model = cfg["model"]
        all_needed = list(train_indices)
        for indices in sampled_targets.values():
            all_needed.extend(indices)
        index_feature_cache = extract_index_feature_cache(
            model=model,
            probe_layer=probe_layer,
            dataset=dataset,
            indices=all_needed,
            batch_size=batch_size,
        )
        train_features = gather_feature_matrix(index_feature_cache, train_indices)

        for strategy in strategies:
            log(f"[model {model_name}] training {strategy}")
            probe = train_probe_with_strategy(
                train_features=train_features,
                train_labels=train_labels,
                subgroup_ids=train_subgroup_ids,
                alignment_ids=train_alignment_ids,
                num_classes=spec.num_classes,
                strategy=strategy,
                epochs=probe_epochs,
                batch_size=batch_size,
                seed=trial_seed,
            )

            for target_group, indices in sampled_targets.items():
                features = gather_feature_matrix(index_feature_cache, indices)
                labels = metadata.loc[indices, "y"].astype(int).tolist()
                preds = evaluate_probe_by_groups(probe, features, labels, batch_size=batch_size)
                raw_rows.append(
                    {
                        "model": model_name,
                        "strategy": strategy,
                        "target_group": target_group,
                        "target_split": metadata.loc[indices[0], "split_name"],
                        "alignment": metadata.loc[indices, "alignment_name"].mode().iloc[0],
                        "accuracy": compute_group_accuracy(preds, labels),
                        "group_size": len(indices),
                    }
                )

    summary_rows = aggregate_rows(raw_rows, ["model", "strategy", "target_group", "target_split", "alignment"])
    write_csv(output_dir / "mitigation_raw.csv", raw_rows)
    write_csv(output_dir / "mitigation_summary.csv", summary_rows)

    summary = {
        "models": list(models_cfg.keys()),
        "strategies": strategies,
        "train_size": train_size,
        "eval_size": eval_size,
        "probe_epochs": probe_epochs,
        "output_files": {
            "raw": str((output_dir / "mitigation_raw.csv").resolve()),
            "summary": str((output_dir / "mitigation_summary.csv").resolve()),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Waterbirds bias-mitigation experiments with frozen backbones.")
    parser.add_argument("--waterbirds-root", type=str, default=DEFAULT_WATERBIRDS_ROOT)
    parser.add_argument("--train-size", type=int, default=300)
    parser.add_argument("--eval-size", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--probe-epochs", type=int, default=12)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/results/bias_mitigation/results_waterbirds_mitigation"),
    )
    parser.add_argument("--models", nargs="*", default=None)
    args = parser.parse_args()

    summary = run_bias_mitigation(
        waterbirds_root=args.waterbirds_root,
        train_size=args.train_size,
        eval_size=args.eval_size,
        batch_size=args.batch_size,
        probe_epochs=args.probe_epochs,
        output_dir=args.output_dir,
        model_names=args.models,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
