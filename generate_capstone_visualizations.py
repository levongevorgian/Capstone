#!/usr/bin/env python3
"""Generate thesis-ready figures from existing experiment outputs."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str((Path.cwd() / ".mpl-cache").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str((Path.cwd() / ".cache").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


SEVERITY_ORDER = ["low", "medium", "high"]
DRIFT_COLORS = {
    "gaussian_noise": "#d1495b",
    "motion_blur": "#2e86ab",
    "brightness_shift": "#edae49",
}
MODEL_COLORS = {
    "ResNet-50": "#1b4332",
    "EfficientNet-B0": "#9c6644",
}
ALIGNMENT_COLORS = {
    "matched": "#2a9d8f",
    "mismatched": "#e76f51",
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_controlled_results(result_dirs: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_frames = []
    corr_frames = []
    for result_dir in result_dirs:
        summary_path = result_dir / "experiment_results_summary.csv"
        corr_path = result_dir / "correlation_summary.csv"
        if summary_path.exists():
            df = pd.read_csv(summary_path)
            if "drift_type" in df.columns:
                summary_frames.append(df)
        if corr_path.exists():
            corr_frames.append(pd.read_csv(corr_path))
    return pd.concat(summary_frames, ignore_index=True), pd.concat(corr_frames, ignore_index=True)


def load_waterbirds_results(result_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    return (
        pd.read_csv(result_dir / "experiment_results_summary.csv"),
        pd.read_csv(result_dir / "correlation_summary.csv"),
    )


def plot_correlation_overview(controlled_corr: pd.DataFrame, waterbirds_corr: pd.DataFrame, output_dir: Path) -> None:
    corr_df = pd.concat([controlled_corr, waterbirds_corr], ignore_index=True)
    corr_df = corr_df[["dataset", "model", "pearson_accuracy_vs_ks", "pearson_accuracy_vs_mmd"]].copy()
    corr_df["label"] = corr_df["dataset"] + "\n" + corr_df["model"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for ax, metric, title in [
        (axes[0], "pearson_accuracy_vs_ks", "Accuracy Drop Vs KS"),
        (axes[1], "pearson_accuracy_vs_mmd", "Accuracy Drop Vs MMD"),
    ]:
        values = corr_df[metric]
        colors = [MODEL_COLORS.get(model, "#666666") for model in corr_df["model"]]
        ax.bar(range(len(corr_df)), values, color=colors)
        ax.set_xticks(range(len(corr_df)))
        ax.set_xticklabels(corr_df["label"], rotation=35, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Pearson correlation")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Drift Metrics Track Performance Strongly On Controlled Benchmarks, Weakly On Waterbirds")
    fig.savefig(output_dir / "correlation_overview.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_controlled_drift_profiles(controlled_summary: pd.DataFrame, output_dir: Path) -> None:
    datasets = list(dict.fromkeys(controlled_summary["dataset"]))
    models = list(dict.fromkeys(controlled_summary["model"]))
    fig, axes = plt.subplots(len(datasets), len(models), figsize=(12, 7), sharex=True, sharey=True, constrained_layout=True)

    if len(datasets) == 1 and len(models) == 1:
        axes = [[axes]]
    elif len(datasets) == 1:
        axes = [axes]
    elif len(models) == 1:
        axes = [[ax] for ax in axes]

    for row_idx, dataset_name in enumerate(datasets):
        for col_idx, model_name in enumerate(models):
            ax = axes[row_idx][col_idx]
            subset = controlled_summary[
                (controlled_summary["dataset"] == dataset_name) & (controlled_summary["model"] == model_name)
            ].copy()
            subset["severity"] = pd.Categorical(subset["severity"], categories=SEVERITY_ORDER, ordered=True)
            subset = subset.sort_values("severity")
            for drift_type, drift_rows in subset.groupby("drift_type"):
                ax.plot(
                    drift_rows["severity"].astype(str),
                    drift_rows["accuracy_drop"],
                    marker="o",
                    linewidth=2,
                    color=DRIFT_COLORS.get(drift_type, "#666666"),
                    label=drift_type.replace("_", " "),
                )
            ax.set_title(f"{dataset_name} | {model_name}")
            ax.set_ylabel("Accuracy drop")
            ax.grid(alpha=0.25)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle("Controlled Benchmarks: Accuracy Drop By Drift Family And Severity")
    fig.savefig(output_dir / "controlled_drift_profiles.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_clean_accuracy_overview(controlled_summary: pd.DataFrame, waterbirds_summary: pd.DataFrame, output_dir: Path) -> None:
    controlled = (
        controlled_summary.groupby(["dataset", "model"], as_index=False)["clean_accuracy"]
        .mean()
        .rename(columns={"clean_accuracy": "reference_accuracy"})
    )
    waterbirds = (
        waterbirds_summary.groupby(["dataset", "model"], as_index=False)["clean_accuracy"]
        .mean()
        .rename(columns={"clean_accuracy": "reference_accuracy"})
    )
    merged = pd.concat([controlled, waterbirds], ignore_index=True)
    merged["label"] = merged["dataset"] + "\n" + merged["model"]

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.bar(
        range(len(merged)),
        merged["reference_accuracy"],
        color=[MODEL_COLORS.get(model, "#666666") for model in merged["model"]],
    )
    ax.set_xticks(range(len(merged)))
    ax.set_xticklabels(merged["label"], rotation=35, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Reference accuracy")
    ax.set_title("Reference/Clean Accuracy Across Benchmarks")
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(output_dir / "clean_accuracy_overview.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_waterbirds_subgroup_accuracy(waterbirds_summary: pd.DataFrame, output_dir: Path) -> None:
    subgroup_rows = waterbirds_summary[
        waterbirds_summary["target_group"].str.contains("landbird_|waterbird_", regex=True)
    ].copy()
    subgroup_rows["plot_label"] = subgroup_rows["target_group"].str.replace("_", "\n", n=1)
    models = list(dict.fromkeys(subgroup_rows["model"]))

    fig, axes = plt.subplots(1, len(models), figsize=(14, 6), sharey=True, constrained_layout=True)
    if len(models) == 1:
        axes = [axes]

    for ax, model_name in zip(axes, models):
        subset = subgroup_rows[subgroup_rows["model"] == model_name].sort_values("drift_accuracy")
        colors = [ALIGNMENT_COLORS.get(value, "#666666") for value in subset["alignment"]]
        ax.barh(subset["target_group"], subset["drift_accuracy"], color=colors)
        ax.set_xlim(0, 1.05)
        ax.set_title(model_name)
        ax.set_xlabel("Subgroup accuracy")
        ax.grid(axis="x", alpha=0.25)

    fig.suptitle("Waterbirds: Subgroup Accuracy Reveals Large Matched-Vs-Mismatched Gaps")
    fig.savefig(output_dir / "waterbirds_subgroup_accuracy.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_waterbirds_drift_vs_drop(waterbirds_summary: pd.DataFrame, output_dir: Path) -> None:
    subgroup_rows = waterbirds_summary[
        waterbirds_summary["target_group"].str.contains("landbird_|waterbird_", regex=True)
    ].copy()
    models = list(dict.fromkeys(subgroup_rows["model"]))

    fig, axes = plt.subplots(1, len(models), figsize=(14, 5), sharey=True, constrained_layout=True)
    if len(models) == 1:
        axes = [axes]

    for ax, model_name in zip(axes, models):
        subset = subgroup_rows[subgroup_rows["model"] == model_name]
        for alignment_name, group in subset.groupby("alignment"):
            ax.scatter(
                group["mmd"],
                group["accuracy_drop"],
                s=80,
                color=ALIGNMENT_COLORS.get(alignment_name, "#666666"),
                label=alignment_name,
                alpha=0.9,
            )
            for _, row in group.iterrows():
                ax.annotate(row["target_group"].replace("test_", "t:").replace("val_", "v:"), (row["mmd"], row["accuracy_drop"]), fontsize=7, alpha=0.8)
        ax.set_title(model_name)
        ax.set_xlabel("MMD")
        ax.set_ylabel("Accuracy drop")
        ax.grid(alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle("Waterbirds: Drift Alone Does Not Fully Explain Worst-Group Collapse")
    fig.savefig(output_dir / "waterbirds_drift_vs_drop.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate visualization figures from existing CAP result folders.")
    parser.add_argument(
        "--controlled-results",
        nargs="+",
        default=["results_final", "results_bloodmnist"],
        help="Controlled benchmark result directories.",
    )
    parser.add_argument(
        "--waterbirds-results",
        default="results_waterbirds",
        help="Waterbirds result directory.",
    )
    parser.add_argument(
        "--output-dir",
        default="figures",
        help="Directory for generated figures.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)
    ensure_dir(Path(os.environ["MPLCONFIGDIR"]))

    controlled_summary, controlled_corr = load_controlled_results([Path(path) for path in args.controlled_results])
    waterbirds_summary, waterbirds_corr = load_waterbirds_results(Path(args.waterbirds_results))

    plot_correlation_overview(controlled_corr, waterbirds_corr, output_dir)
    plot_controlled_drift_profiles(controlled_summary, output_dir)
    plot_clean_accuracy_overview(controlled_summary, waterbirds_summary, output_dir)
    plot_waterbirds_subgroup_accuracy(waterbirds_summary, output_dir)
    plot_waterbirds_drift_vs_drop(waterbirds_summary, output_dir)

    print(f"Wrote figures to {output_dir}")


if __name__ == "__main__":
    main()
