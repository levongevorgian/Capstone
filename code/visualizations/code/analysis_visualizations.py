#!/usr/bin/env python3
"""Generate final and diagnostic figures for the Capstone repository."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from code.utils.visualization_style import (  # noqa: E402
    ALIGNMENT_COLORS,
    DATASET_COLORS,
    DRIFT_COLORS,
    MODEL_COLORS,
    SEVERITY_ORDER,
    apply_publication_style,
    configure_plot_environment,
    ensure_dir,
)

configure_plot_environment()

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


CONTROLLED_RESULTS_DIRS = [
    PROJECT_ROOT / "outputs/results/controlled/results_final",
    PROJECT_ROOT / "outputs/results/controlled/results_bloodmnist",
]
WATERBIRDS_RESULTS_DIR = PROJECT_ROOT / "outputs/results/waterbirds/results_waterbirds"
FINAL_FIGURES_DIR = PROJECT_ROOT / "code/visualizations/figures/final"
DIAGNOSTIC_FIGURES_DIR = PROJECT_ROOT / "code/visualizations/figures/diagnostics"


def load_results(
    controlled_dirs: list[Path],
    waterbirds_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the summary and correlation tables needed for plotting."""
    controlled_frames: list[pd.DataFrame] = []
    correlation_frames: list[pd.DataFrame] = []

    for result_dir in controlled_dirs:
        summary_path = result_dir / "experiment_results_summary.csv"
        corr_path = result_dir / "correlation_summary.csv"
        if not summary_path.exists() or not corr_path.exists():
            raise FileNotFoundError(f"Missing controlled benchmark outputs in {result_dir}")
        controlled_frames.append(pd.read_csv(summary_path))
        correlation_frames.append(pd.read_csv(corr_path))

    waterbirds_summary_path = waterbirds_dir / "experiment_results_summary.csv"
    waterbirds_corr_path = waterbirds_dir / "correlation_summary.csv"
    if not waterbirds_summary_path.exists() or not waterbirds_corr_path.exists():
        raise FileNotFoundError(f"Missing Waterbirds outputs in {waterbirds_dir}")

    waterbirds_summary = pd.read_csv(waterbirds_summary_path)
    correlation_frames.append(pd.read_csv(waterbirds_corr_path))
    return (
        pd.concat(controlled_frames, ignore_index=True),
        waterbirds_summary,
        pd.concat(correlation_frames, ignore_index=True),
    )


def save_figure(fig: plt.Figure, output_path: Path) -> None:
    """Save a figure at publication quality and close it cleanly."""
    ensure_dir(output_path.parent)
    fig.savefig(output_path)
    plt.close(fig)


def waterbirds_subgroups(waterbirds_df: pd.DataFrame) -> pd.DataFrame:
    """Return only class-background subgroup rows for Waterbirds."""
    return waterbirds_df[
        waterbirds_df["target_group"].str.contains("landbird_|waterbird_", regex=True)
    ].copy()


def add_trendline(ax: plt.Axes, x: np.ndarray, y: np.ndarray, color: str) -> None:
    """Overlay a simple least-squares trendline when the data support it."""
    if len(x) < 2 or np.allclose(x, x[0]):
        return
    slope, intercept = np.polyfit(x, y, 1)
    x_line = np.linspace(float(np.min(x)), float(np.max(x)), 100)
    ax.plot(x_line, slope * x_line + intercept, color=color, linewidth=1.8, alpha=0.85)


def plot_clean_accuracy_overview(
    controlled_summary: pd.DataFrame,
    waterbirds_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    """Compare reference accuracies across all benchmarks and backbones."""
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
    merged["label"] = merged["dataset"].str.upper() + "\n" + merged["model"].str.replace(
        "EfficientNet-B0",
        "EffNet-B0",
    )

    fig, ax = plt.subplots(figsize=(8.5, 5.5), constrained_layout=True)
    ax.bar(
        range(len(merged)),
        merged["reference_accuracy"],
        color=[MODEL_COLORS.get(model, "#666666") for model in merged["model"]],
    )
    ax.set_xticks(range(len(merged)))
    ax.set_xticklabels(merged["label"], rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Reference accuracy")
    ax.set_title("Reference Accuracy Across Benchmarks")
    ax.grid(axis="y")
    save_figure(fig, output_path)


def plot_controlled_drift_profiles(controlled_summary: pd.DataFrame, output_path: Path) -> None:
    """Show accuracy drop by severity for each controlled drift family."""
    datasets = list(dict.fromkeys(controlled_summary["dataset"]))
    models = list(dict.fromkeys(controlled_summary["model"]))
    fig, axes = plt.subplots(
        len(datasets),
        len(models),
        figsize=(12, 7),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

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
                (controlled_summary["dataset"] == dataset_name)
                & (controlled_summary["model"] == model_name)
            ].copy()
            subset["severity"] = pd.Categorical(
                subset["severity"],
                categories=SEVERITY_ORDER,
                ordered=True,
            )
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
            ax.set_title(f"{dataset_name.upper()} | {model_name}")
            ax.set_ylabel("Accuracy drop")
            ax.grid(True)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3)
    fig.suptitle("Controlled Benchmarks: Accuracy Drop by Drift Type and Severity")
    save_figure(fig, output_path)


def plot_correlation_overview(corr_df: pd.DataFrame, output_path: Path) -> None:
    """Summarize how strongly drift metrics track accuracy drop."""
    plot_df = corr_df[
        ["dataset", "model", "pearson_accuracy_vs_ks", "pearson_accuracy_vs_mmd"]
    ].copy()
    plot_df["label"] = plot_df["dataset"].str.upper() + "\n" + plot_df["model"].str.replace(
        "EfficientNet-B0",
        "EffNet-B0",
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for ax, metric, title in [
        (axes[0], "pearson_accuracy_vs_ks", "Accuracy Drop vs KS"),
        (axes[1], "pearson_accuracy_vs_mmd", "Accuracy Drop vs MMD"),
    ]:
        ax.bar(
            range(len(plot_df)),
            plot_df[metric],
            color=[DATASET_COLORS.get(dataset, "#666666") for dataset in plot_df["dataset"]],
        )
        ax.set_xticks(range(len(plot_df)))
        ax.set_xticklabels(plot_df["label"], rotation=30, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Pearson correlation")
        ax.set_title(title)
        ax.grid(axis="y")

    fig.suptitle("Controlled Shift Correlations Are Stronger Than Waterbirds")
    save_figure(fig, output_path)


def plot_waterbirds_subgroup_accuracy(waterbirds_df: pd.DataFrame, output_path: Path) -> None:
    """Compare Waterbirds subgroup accuracies across backbones."""
    subgroup_rows = waterbirds_subgroups(waterbirds_df)
    models = list(dict.fromkeys(subgroup_rows["model"]))
    fig, axes = plt.subplots(
        1,
        len(models),
        figsize=(14, 6),
        sharey=True,
        constrained_layout=True,
    )
    if len(models) == 1:
        axes = [axes]

    for ax, model_name in zip(axes, models):
        subset = subgroup_rows[subgroup_rows["model"] == model_name].sort_values("drift_accuracy")
        colors = [ALIGNMENT_COLORS.get(value, "#666666") for value in subset["alignment"]]
        ax.barh(subset["target_group"], subset["drift_accuracy"], color=colors)
        ax.set_xlim(0, 1.05)
        ax.set_xlabel("Subgroup accuracy")
        ax.set_title(model_name)
        ax.grid(axis="x")

    fig.suptitle("Waterbirds Subgroup Accuracy by Label-Background Environment")
    save_figure(fig, output_path)


def plot_waterbirds_drift_vs_drop(waterbirds_df: pd.DataFrame, output_path: Path) -> None:
    """Compare Waterbirds MMD with downstream subgroup accuracy drop."""
    subgroup_rows = waterbirds_subgroups(waterbirds_df)
    models = list(dict.fromkeys(subgroup_rows["model"]))
    fig, axes = plt.subplots(
        1,
        len(models),
        figsize=(14, 5),
        sharey=True,
        constrained_layout=True,
    )
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
        ax.set_title(model_name)
        ax.set_xlabel("MMD")
        ax.set_ylabel("Accuracy drop")
        ax.grid(True)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.suptitle("Waterbirds: Drift Signals Rise, but the Relationship Is Weaker")
    save_figure(fig, output_path)


def plot_drift_vs_accuracy(
    controlled_df: pd.DataFrame,
    waterbirds_df: pd.DataFrame,
    metric: str,
    output_path: Path,
) -> None:
    """Create a benchmark-by-model scatter plot for drift versus accuracy drop."""
    fig, axes = plt.subplots(3, 2, figsize=(13, 12), constrained_layout=True)
    datasets = [
        ("cifar10", controlled_df[controlled_df["dataset"] == "cifar10"]),
        ("bloodmnist", controlled_df[controlled_df["dataset"] == "bloodmnist"]),
        ("waterbirds", waterbirds_subgroups(waterbirds_df)),
    ]
    models = ["ResNet-50", "EfficientNet-B0"]

    for row_idx, (dataset_name, dataset_df) in enumerate(datasets):
        for col_idx, model_name in enumerate(models):
            ax = axes[row_idx, col_idx]
            subset = dataset_df[dataset_df["model"] == model_name].copy()
            if dataset_name == "waterbirds":
                point_colors = [ALIGNMENT_COLORS[val] for val in subset["alignment"]]
            else:
                point_colors = [DRIFT_COLORS[val] for val in subset["drift_type"]]
            x = subset[metric].to_numpy(dtype=float)
            y = subset["accuracy_drop"].to_numpy(dtype=float)
            ax.scatter(x, y, s=65, c=point_colors, edgecolor="white", linewidth=0.6, alpha=0.9)
            add_trendline(ax, x, y, DATASET_COLORS[dataset_name])
            ax.set_title(f"{dataset_name.upper()} | {model_name}")
            ax.set_xlabel(metric.upper())
            ax.set_ylabel("Accuracy drop")
            ax.grid(True)

    fig.suptitle(f"{metric.upper()} vs Accuracy Drop Across Benchmarks")
    save_figure(fig, output_path)


def plot_matched_mismatched(waterbirds_df: pd.DataFrame, output_path: Path) -> None:
    """Highlight the matched-versus-mismatched accuracy gap on Waterbirds."""
    subset = waterbirds_df[
        waterbirds_df["target_group"].isin(
            ["val_matched", "val_mismatched", "test_matched", "test_mismatched"]
        )
    ].copy()
    order = ["val_matched", "val_mismatched", "test_matched", "test_mismatched"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True, constrained_layout=True)
    for ax, model_name in zip(axes, ["ResNet-50", "EfficientNet-B0"]):
        model_df = subset[subset["model"] == model_name].copy()
        model_df["target_group"] = pd.Categorical(
            model_df["target_group"],
            categories=order,
            ordered=True,
        )
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

    fig.suptitle("Matched Backgrounds Preserve Accuracy; Mismatched Backgrounds Collapse It")
    save_figure(fig, output_path)


def plot_worst_group_comparison(waterbirds_df: pd.DataFrame, output_path: Path) -> None:
    """Compare worst-group, average-group, and best-group accuracy."""
    subgroup_rows = waterbirds_subgroups(waterbirds_df)
    rows = []
    for model_name, group in subgroup_rows.groupby("model"):
        rows.append(
            {
                "model": model_name,
                "Worst-group": float(group["drift_accuracy"].min()),
                "Average-group": float(group["drift_accuracy"].mean()),
                "Best-group": float(group["drift_accuracy"].max()),
            }
        )
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
    save_figure(fig, output_path)


def plot_subgroup_heatmap(waterbirds_df: pd.DataFrame, output_path: Path) -> None:
    """Render a compact subgroup-accuracy heatmap for presentations."""
    subgroup_rows = waterbirds_subgroups(waterbirds_df)
    column_labels = [
        "val_landbird_land",
        "val_landbird_water",
        "val_waterbird_land",
        "val_waterbird_water",
        "test_landbird_land",
        "test_landbird_water",
        "test_waterbird_land",
        "test_waterbird_water",
    ]
    row_labels = ["ResNet-50", "EfficientNet-B0"]
    heatmap_rows = []

    for model_name in row_labels:
        model_df = subgroup_rows[subgroup_rows["model"] == model_name].set_index("target_group")
        heatmap_rows.append([float(model_df.loc[label, "drift_accuracy"]) for label in column_labels])

    fig, ax = plt.subplots(figsize=(14, 4.8), constrained_layout=True)
    image = ax.imshow(np.asarray(heatmap_rows), cmap="YlGnBu", aspect="auto", vmin=0.3, vmax=1.0)
    ax.set_xticks(np.arange(len(column_labels)))
    ax.set_xticklabels(column_labels, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_title("Waterbirds Subgroup Accuracy Heatmap")
    for row_idx in range(len(row_labels)):
        for col_idx in range(len(column_labels)):
            ax.text(
                col_idx,
                row_idx,
                f"{heatmap_rows[row_idx][col_idx]:.2f}",
                ha="center",
                va="center",
                color="black",
                fontsize=8,
            )
    fig.colorbar(image, ax=ax, shrink=0.85, label="Accuracy")
    save_figure(fig, output_path)


def plot_drift_vs_worst_group_accuracy(waterbirds_df: pd.DataFrame, output_path: Path) -> None:
    """Plot subgroup accuracy against KS and MMD for quick diagnosis."""
    subgroup_rows = waterbirds_subgroups(waterbirds_df)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True, constrained_layout=True)

    for ax, metric in zip(axes, ["ks_mean", "mmd"]):
        for model_name in ["ResNet-50", "EfficientNet-B0"]:
            model_df = subgroup_rows[subgroup_rows["model"] == model_name]
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
        ax.set_title(f"{metric.upper()} vs Subgroup Accuracy")
        ax.grid(True)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    save_figure(fig, output_path)


def write_interpretations(output_path: Path) -> None:
    """Write a simple interpretation guide for the diagnostic figures."""
    text = """# Visualization Interpretations

## Final figures

- `correlation_overview.png`: summarizes how strongly drift metrics align with accuracy drop across all datasets.
- `controlled_drift_profiles.png`: shows which corruption families are most damaging as severity increases.
- `clean_accuracy_overview.png`: compares baseline/reference accuracy across datasets and backbones.
- `waterbirds_subgroup_accuracy.png`: highlights the large subgroup disparities on Waterbirds.
- `waterbirds_drift_vs_drop.png`: shows that Waterbirds drift metrics move in the right direction, but less cleanly than in controlled settings.

## Diagnostic figures

- `drift_vs_accuracy_ks.png` and `drift_vs_accuracy_mmd.png`: benchmark-by-benchmark scatter plots for the two drift metrics.
- `matched_vs_mismatched_accuracy.png`: direct comparison of Waterbirds matched and mismatched environments.
- `worst_group_comparison.png`: best/average/worst subgroup accuracy for each backbone.
- `subgroup_accuracy_heatmap.png`: compact overview of the eight Waterbirds subgroup environments.
- `drift_vs_worst_group_accuracy.png`: diagnostic view of whether the worst groups also carry the strongest drift signal.
"""
    output_path.write_text(text)


def build_all_figures(
    controlled_dirs: list[Path],
    waterbirds_dir: Path,
    final_dir: Path,
    diagnostic_dir: Path,
) -> None:
    """Generate the full publication and diagnostic figure set."""
    controlled_summary, waterbirds_summary, corr_df = load_results(controlled_dirs, waterbirds_dir)

    ensure_dir(final_dir)
    ensure_dir(diagnostic_dir)
    apply_publication_style(plt)

    plot_correlation_overview(corr_df, final_dir / "correlation_overview.png")
    plot_controlled_drift_profiles(controlled_summary, final_dir / "controlled_drift_profiles.png")
    plot_clean_accuracy_overview(controlled_summary, waterbirds_summary, final_dir / "clean_accuracy_overview.png")
    plot_waterbirds_subgroup_accuracy(waterbirds_summary, final_dir / "waterbirds_subgroup_accuracy.png")
    plot_waterbirds_drift_vs_drop(waterbirds_summary, final_dir / "waterbirds_drift_vs_drop.png")

    plot_drift_vs_accuracy(controlled_summary, waterbirds_summary, "ks_mean", diagnostic_dir / "drift_vs_accuracy_ks.png")
    plot_drift_vs_accuracy(controlled_summary, waterbirds_summary, "mmd", diagnostic_dir / "drift_vs_accuracy_mmd.png")
    plot_matched_mismatched(waterbirds_summary, diagnostic_dir / "matched_vs_mismatched_accuracy.png")
    plot_worst_group_comparison(waterbirds_summary, diagnostic_dir / "worst_group_comparison.png")
    plot_subgroup_heatmap(waterbirds_summary, diagnostic_dir / "subgroup_accuracy_heatmap.png")
    plot_drift_vs_worst_group_accuracy(
        waterbirds_summary,
        diagnostic_dir / "drift_vs_worst_group_accuracy.png",
    )
    write_interpretations(diagnostic_dir / "visualization_interpretations.md")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for figure generation."""
    parser = argparse.ArgumentParser(description="Generate publication-ready CAP figures from existing result folders.")
    parser.add_argument(
        "--controlled-results",
        nargs="+",
        type=Path,
        default=CONTROLLED_RESULTS_DIRS,
        help="Controlled benchmark result directories.",
    )
    parser.add_argument(
        "--waterbirds-results",
        type=Path,
        default=WATERBIRDS_RESULTS_DIR,
        help="Waterbirds result directory.",
    )
    parser.add_argument(
        "--final-dir",
        type=Path,
        default=FINAL_FIGURES_DIR,
        help="Directory for presentation-ready figures.",
    )
    parser.add_argument(
        "--diagnostic-dir",
        type=Path,
        default=DIAGNOSTIC_FIGURES_DIR,
        help="Directory for diagnostic figures and notes.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for figure generation."""
    args = parse_args()
    build_all_figures(
        controlled_dirs=[Path(path) for path in args.controlled_results],
        waterbirds_dir=Path(args.waterbirds_results),
        final_dir=Path(args.final_dir),
        diagnostic_dir=Path(args.diagnostic_dir),
    )
    print(f"Wrote final figures to {args.final_dir}")
    print(f"Wrote diagnostic figures to {args.diagnostic_dir}")


if __name__ == "__main__":
    main()
