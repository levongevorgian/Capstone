#!/usr/bin/env python3
"""Generate presentation-quality project visualizations from existing results."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str((Path.cwd() / ".mpl-cache").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str((Path.cwd() / ".cache").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_RESULTS = {
    "cifar10": Path("results_final"),
    "bloodmnist": Path("results_bloodmnist"),
    "waterbirds": Path("results_waterbirds"),
}
OUTPUT_DIR = Path("results/figures")

DATASET_COLORS = {
    "cifar10": "#d1495b",
    "bloodmnist": "#2e86ab",
    "waterbirds": "#6a994e",
}
MODEL_COLORS = {
    "ResNet-50": "#1b4332",
    "EfficientNet-B0": "#9c6644",
}
DRIFT_COLORS = {
    "gaussian_noise": "#d1495b",
    "motion_blur": "#2e86ab",
    "brightness_shift": "#edae49",
}
ALIGNMENT_COLORS = {
    "matched": "#2a9d8f",
    "mismatched": "#e76f51",
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 220,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "axes.titlesize": 13,
            "figure.titlesize": 16,
            "legend.frameon": False,
            "font.size": 10,
            "grid.alpha": 0.22,
        }
    )


def load_results() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    controlled_frames = []
    corr_frames = []
    for dataset_name in ["cifar10", "bloodmnist"]:
        summary = pd.read_csv(BASE_RESULTS[dataset_name] / "experiment_results_summary.csv")
        corr = pd.read_csv(BASE_RESULTS[dataset_name] / "correlation_summary.csv")
        controlled_frames.append(summary)
        corr_frames.append(corr)

    waterbirds_summary = pd.read_csv(BASE_RESULTS["waterbirds"] / "experiment_results_summary.csv")
    waterbirds_corr = pd.read_csv(BASE_RESULTS["waterbirds"] / "correlation_summary.csv")
    corr_frames.append(waterbirds_corr)
    return pd.concat(controlled_frames, ignore_index=True), waterbirds_summary, pd.concat(corr_frames, ignore_index=True)


def add_trendline(ax, x: np.ndarray, y: np.ndarray, color: str) -> None:
    if len(x) < 2 or np.allclose(x, x[0]):
        return
    slope, intercept = np.polyfit(x, y, 1)
    x_line = np.linspace(float(np.min(x)), float(np.max(x)), 100)
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, color=color, linewidth=1.8, alpha=0.9)


def plot_drift_vs_accuracy(
    controlled_df: pd.DataFrame,
    waterbirds_df: pd.DataFrame,
    metric: str,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(13, 13), constrained_layout=True)
    datasets = [
        ("cifar10", controlled_df[controlled_df["dataset"] == "cifar10"]),
        ("bloodmnist", controlled_df[controlled_df["dataset"] == "bloodmnist"]),
        ("waterbirds", waterbirds_df),
    ]
    models = ["ResNet-50", "EfficientNet-B0"]

    for row_idx, (dataset_name, dataset_df) in enumerate(datasets):
        for col_idx, model_name in enumerate(models):
            ax = axes[row_idx, col_idx]
            subset = dataset_df[dataset_df["model"] == model_name].copy()
            if dataset_name == "waterbirds":
                subgroup_mask = subset["target_group"].str.contains("landbird_|waterbird_", regex=True)
                subset = subset[subgroup_mask]
                colors = [ALIGNMENT_COLORS[val] for val in subset["alignment"]]
                labels = subset["target_group"].tolist()
            else:
                colors = [DRIFT_COLORS[val] for val in subset["drift_type"]]
                labels = (subset["drift_type"] + " / " + subset["severity"]).tolist()
            x = subset[metric].to_numpy(dtype=float)
            y = subset["accuracy_drop"].to_numpy(dtype=float)
            ax.scatter(x, y, s=65, c=colors, edgecolor="white", linewidth=0.6, alpha=0.9)
            add_trendline(ax, x, y, DATASET_COLORS[dataset_name])
            top_idx = np.argsort(y)[-2:]
            for idx in top_idx:
                ax.annotate(labels[idx], (x[idx], y[idx]), fontsize=7, alpha=0.8)
            ax.set_title(f"{dataset_name.upper()} | {model_name}")
            ax.set_xlabel(metric.upper())
            ax.set_ylabel("Accuracy drop")
            ax.grid(True)

    fig.suptitle(f"{metric.upper()} Drift Vs Accuracy Drop Across Benchmarks")
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_correlation_bars(corr_df: pd.DataFrame, output_path: Path) -> None:
    corr_df = corr_df.copy()
    corr_df["label"] = corr_df["dataset"].str.upper() + "\n" + corr_df["model"].str.replace("EfficientNet-B0", "EffNet-B0")
    metrics = [
        ("pearson_accuracy_vs_ks", "Pearson: KS"),
        ("pearson_accuracy_vs_mmd", "Pearson: MMD"),
        ("spearman_accuracy_vs_ks", "Spearman: KS"),
        ("spearman_accuracy_vs_mmd", "Spearman: MMD"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    for ax, (metric, title) in zip(axes.flat, metrics):
        colors = [DATASET_COLORS[row["dataset"]] for _, row in corr_df.iterrows()]
        ax.bar(np.arange(len(corr_df)), corr_df[metric], color=colors)
        ax.set_xticks(np.arange(len(corr_df)))
        ax.set_xticklabels(corr_df["label"], rotation=30, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_title(title)
        ax.set_ylabel("Correlation")
        ax.grid(axis="y")
    fig.suptitle("Correlation Summary: Controlled Benchmarks Strong, Waterbirds Weaker")
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_matched_mismatched(waterbirds_df: pd.DataFrame, output_path: Path) -> None:
    subset = waterbirds_df[
        waterbirds_df["target_group"].isin(["val_matched", "val_mismatched", "test_matched", "test_mismatched"])
    ].copy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True, constrained_layout=True)
    for ax, model_name in zip(axes, ["ResNet-50", "EfficientNet-B0"]):
        model_df = subset[subset["model"] == model_name].copy()
        order = ["val_matched", "val_mismatched", "test_matched", "test_mismatched"]
        model_df["target_group"] = pd.Categorical(model_df["target_group"], categories=order, ordered=True)
        model_df = model_df.sort_values("target_group")
        ax.bar(
            model_df["target_group"].astype(str),
            model_df["drift_accuracy"],
            color=[ALIGNMENT_COLORS[val] for val in model_df["alignment"]],
        )
        ax.set_title(model_name)
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1.05)
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y")
    fig.suptitle("Waterbirds: Matched Backgrounds Preserve Accuracy, Mismatched Backgrounds Collapse It")
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def waterbirds_subgroups(waterbirds_df: pd.DataFrame) -> pd.DataFrame:
    return waterbirds_df[waterbirds_df["target_group"].str.contains("landbird_|waterbird_", regex=True)].copy()


def plot_subgroup_accuracy_bars(waterbirds_df: pd.DataFrame, output_path: Path) -> None:
    subset = waterbirds_subgroups(waterbirds_df)
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharex=True, sharey=True, constrained_layout=True)
    for ax, model_name in zip(axes, ["ResNet-50", "EfficientNet-B0"]):
        model_df = subset[subset["model"] == model_name].sort_values("drift_accuracy")
        ax.barh(
            model_df["target_group"],
            model_df["drift_accuracy"],
            color=[ALIGNMENT_COLORS[val] for val in model_df["alignment"]],
        )
        ax.set_xlim(0, 1.05)
        ax.set_xlabel("Accuracy")
        ax.set_title(model_name)
        ax.grid(axis="x")
    fig.suptitle("Waterbirds Subgroup Accuracy by Label-Background Environment")
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_worst_group_comparison(waterbirds_df: pd.DataFrame, output_path: Path) -> None:
    subset = waterbirds_subgroups(waterbirds_df)
    rows = []
    for model_name, group in subset.groupby("model"):
        worst = float(group["drift_accuracy"].min())
        average = float(group["drift_accuracy"].mean())
        best = float(group["drift_accuracy"].max())
        rows.append({"model": model_name, "Worst-group": worst, "Average-group": average, "Best-group": best})
    plot_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    x = np.arange(len(plot_df))
    width = 0.22
    for offset, column, color in [
        (-width, "Worst-group", "#c1121f"),
        (0, "Average-group", "#457b9d"),
        (width, "Best-group", "#2a9d8f"),
    ]:
        ax.bar(x + offset, plot_df[column], width=width, label=column, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["model"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_title("Worst-Group Accuracy Comparison on Waterbirds")
    ax.grid(axis="y")
    ax.legend()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_subgroup_heatmap(waterbirds_df: pd.DataFrame, output_path: Path) -> None:
    subset = waterbirds_subgroups(waterbirds_df)
    heatmap_data = []
    row_labels = []
    col_labels = [
        "val_landbird_land",
        "val_landbird_water",
        "val_waterbird_land",
        "val_waterbird_water",
        "test_landbird_land",
        "test_landbird_water",
        "test_waterbird_land",
        "test_waterbird_water",
    ]
    for model_name in ["ResNet-50", "EfficientNet-B0"]:
        row_labels.append(model_name)
        row = []
        model_df = subset[subset["model"] == model_name].set_index("target_group")
        for label in col_labels:
            row.append(float(model_df.loc[label, "drift_accuracy"]))
        heatmap_data.append(row)

    fig, ax = plt.subplots(figsize=(14, 4.8), constrained_layout=True)
    im = ax.imshow(np.asarray(heatmap_data), cmap="YlGnBu", aspect="auto", vmin=0.3, vmax=1.0)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_title("Waterbirds Subgroup Accuracy Heatmap")
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            ax.text(j, i, f"{heatmap_data[i][j]:.2f}", ha="center", va="center", color="black", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.85, label="Accuracy")
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_drift_vs_worst_group_accuracy(waterbirds_df: pd.DataFrame, output_path: Path) -> None:
    subset = waterbirds_subgroups(waterbirds_df)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True, constrained_layout=True)
    for ax, metric in zip(axes, ["ks_mean", "mmd"]):
        for model_name in ["ResNet-50", "EfficientNet-B0"]:
            model_df = subset[subset["model"] == model_name]
            ax.scatter(
                model_df[metric],
                model_df["drift_accuracy"],
                s=75,
                color=MODEL_COLORS[model_name],
                alpha=0.85,
                label=model_name,
            )
            worst_row = model_df.sort_values("drift_accuracy").iloc[0]
            ax.scatter(
                [worst_row[metric]],
                [worst_row["drift_accuracy"]],
                s=140,
                marker="*",
                color="#c1121f",
                edgecolor="black",
                linewidth=0.6,
                zorder=5,
            )
            ax.annotate(worst_row["target_group"], (worst_row[metric], worst_row["drift_accuracy"]), fontsize=8)
        ax.set_xlabel(metric.upper())
        ax.set_ylabel("Subgroup accuracy")
        ax.set_title(f"Waterbirds: {metric.upper()} Vs Subgroup Accuracy")
        ax.grid(True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def write_interpretations(output_path: Path) -> None:
    text = """# Visualization Interpretations

## `drift_vs_accuracy_ks.png`

This figure directly addresses the core hypothesis from the proposal: whether feature-space drift tracks model degradation. On CIFAR-10 and BloodMNIST, the points follow a clear upward pattern, showing that larger KS drift is associated with larger accuracy drop under controlled corruptions. On Waterbirds, the relationship is much weaker, which supports the thesis-safe claim that natural subgroup shift is harder to summarize with a single drift-performance trend.

## `drift_vs_accuracy_mmd.png`

The MMD scatter provides the same research question through a second drift metric, which helps answer whether the effect is metric-specific or consistent across methods. The controlled datasets again show a strong positive relationship between drift and degradation, while Waterbirds shows only a partial association. This supports the proposal’s broader finding that label-free drift signals are useful proxies, but their strength depends on the shift type.

## `correlation_bars.png`

These bars summarize the strength of the drift-performance relationship in one place across datasets, models, metrics, and correlation definitions. The plot makes the capstone narrative clear: controlled synthetic shift produces high Pearson and Spearman correlations, while Waterbirds remains positive but notably weaker. That visual contrast directly answers the proposal question about generalization across benchmarks.

## `matched_vs_mismatched_accuracy.png`

This plot shows the practical effect of spurious correlation on Waterbirds by separating matched and mismatched background environments. Both models remain very strong on matched groups and drop sharply on mismatched groups, especially on validation mismatched and test mismatched splits. This supports the proposal’s claim that Waterbirds contributes a realistic environment-shift test that is qualitatively different from synthetic corruption.

## `subgroup_accuracy_bars.png`

The subgroup bars expose exactly where performance collapses instead of hiding it inside one average number. The lowest bars are the `waterbird_land` groups, while matched groups such as `landbird_land` and `waterbird_water` remain much stronger. This directly addresses the proposal’s concern about subgroup sensitivity and shows how background bias shapes model behavior.

## `worst_group_comparison.png`

Worst-group accuracy is the most important fairness-style summary for Waterbirds because it captures the part of the distribution that fails first. The figure shows a wide gap between worst-group and average-group performance for both backbones, with EfficientNet-B0 reaching the lowest worst-group value. This strengthens the finding that strong average behavior can coexist with severe subgroup failure.

## `subgroup_accuracy_heatmap.png`

The heatmap turns subgroup behavior into a compact benchmark overview that is easy to use in a presentation or thesis defense. Its pattern highlights a consistent structure: matched environments are dark and high-performing, while mismatched environments are lighter and lower-performing. This makes the Waterbirds spurious-correlation story visible at a glance and answers the proposal’s question about which subgroup conditions are most harmful.

## `drift_vs_worst_group_accuracy.png`

This plot asks a sharper Waterbirds-specific question: do the worst-performing groups also show elevated feature drift? The answer is broadly yes, but not perfectly, which is exactly why Waterbirds should be interpreted separately from the controlled benchmarks. The figure supports the final thesis framing that drift still moves in the correct direction under natural shift, but it is not as strong or clean a proxy for worst-group failure as it is under synthetic corruption.
"""
    output_path.write_text(text)


def main() -> None:
    ensure_dir(OUTPUT_DIR)
    ensure_dir(Path(os.environ["MPLCONFIGDIR"]))
    ensure_dir(Path(os.environ["XDG_CACHE_HOME"]))
    setup_style()

    controlled_df, waterbirds_df, corr_df = load_results()

    plot_drift_vs_accuracy(controlled_df, waterbirds_df, "ks_mean", OUTPUT_DIR / "drift_vs_accuracy_ks.png")
    plot_drift_vs_accuracy(controlled_df, waterbirds_df, "mmd", OUTPUT_DIR / "drift_vs_accuracy_mmd.png")
    plot_correlation_bars(corr_df, OUTPUT_DIR / "correlation_bars.png")
    plot_matched_mismatched(waterbirds_df, OUTPUT_DIR / "matched_vs_mismatched_accuracy.png")
    plot_subgroup_accuracy_bars(waterbirds_df, OUTPUT_DIR / "subgroup_accuracy_bars.png")
    plot_worst_group_comparison(waterbirds_df, OUTPUT_DIR / "worst_group_comparison.png")
    plot_subgroup_heatmap(waterbirds_df, OUTPUT_DIR / "subgroup_accuracy_heatmap.png")
    plot_drift_vs_worst_group_accuracy(waterbirds_df, OUTPUT_DIR / "drift_vs_worst_group_accuracy.png")
    write_interpretations(OUTPUT_DIR / "visualization_interpretations.md")
    print(f"Wrote visualizations to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
