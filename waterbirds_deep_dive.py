#!/usr/bin/env python3
"""Waterbirds-specific analysis on top of existing experiment outputs.

This script does not rerun the benchmark. It reframes the existing
Waterbirds results around subgroup behavior, which is more faithful to
the benchmark than a single pooled drift-vs-accuracy correlation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_group_components(target_group: str) -> tuple[str | None, str | None]:
    if "landbird_" in target_group:
        return "landbird", target_group.rsplit("_", 1)[-1]
    if "waterbird_" in target_group:
        return "waterbird", target_group.rsplit("_", 1)[-1]
    return None, None


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    rows = [[str(value) for value in row] for row in df.itertuples(index=False, name=None)]
    widths = [len(header) for header in headers]
    for row in rows:
        for idx, value in enumerate(row):
            widths[idx] = max(widths[idx], len(value))

    def render_row(values: list[str]) -> str:
        return "| " + " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(values)) + " |"

    separator = "| " + " | ".join("-" * widths[idx] for idx in range(len(headers))) + " |"
    lines = [render_row(headers), separator]
    lines.extend(render_row(row) for row in rows)
    return "\n".join(lines)


def build_alignment_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = df[df["target_group"].isin(["val_matched", "val_mismatched", "test_matched", "test_mismatched"])].copy()
    return rows[
        ["model", "target_split", "alignment", "drift_accuracy", "accuracy_drop", "ks_mean", "mmd", "group_size"]
    ].sort_values(["model", "target_split", "alignment"])


def build_subgroup_summary(df: pd.DataFrame) -> pd.DataFrame:
    subgroup_rows = df[
        df["target_group"].str.contains("landbird_|waterbird_", regex=True)
    ].copy()
    subgroup_rows["label_name"], subgroup_rows["background"] = zip(
        *subgroup_rows["target_group"].map(parse_group_components)
    )
    return subgroup_rows


def build_environment_gap_summary(subgroup_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, split_name, label_name), group in subgroup_df.groupby(["model", "target_split", "label_name"]):
        by_background = group.set_index("background")
        if "land" not in by_background.index or "water" not in by_background.index:
            continue
        land_row = by_background.loc["land"]
        water_row = by_background.loc["water"]
        rows.append(
            {
                "model": model,
                "target_split": split_name,
                "label_name": label_name,
                "land_background_accuracy": float(land_row["drift_accuracy"]),
                "water_background_accuracy": float(water_row["drift_accuracy"]),
                "environment_accuracy_gap": float(abs(land_row["drift_accuracy"] - water_row["drift_accuracy"])),
                "land_background_ks": float(land_row["ks_mean"]),
                "water_background_ks": float(water_row["ks_mean"]),
                "land_background_mmd": float(land_row["mmd"]),
                "water_background_mmd": float(water_row["mmd"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["model", "target_split", "label_name"])


def build_worst_group_summary(subgroup_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, split_name), group in subgroup_df.groupby(["model", "target_split"]):
        worst = group.sort_values(["drift_accuracy", "accuracy_drop"], ascending=[True, False]).iloc[0]
        best = group.sort_values(["drift_accuracy", "accuracy_drop"], ascending=[False, True]).iloc[0]
        rows.append(
            {
                "model": model,
                "target_split": split_name,
                "worst_group": worst["target_group"],
                "worst_group_accuracy": float(worst["drift_accuracy"]),
                "best_group": best["target_group"],
                "best_group_accuracy": float(best["drift_accuracy"]),
                "worst_to_best_accuracy_gap": float(best["drift_accuracy"] - worst["drift_accuracy"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["model", "target_split"])


def write_report(
    output_path: Path,
    correlations: pd.DataFrame,
    alignment_summary: pd.DataFrame,
    environment_gaps: pd.DataFrame,
    worst_groups: pd.DataFrame,
) -> None:
    lines: list[str] = []
    lines.append("# Waterbirds Deep Dive")
    lines.append("")
    lines.append("This report improves the Waterbirds presentation by focusing on subgroup behavior rather than only a pooled drift-performance correlation.")
    lines.append("")
    lines.append("## Why This Framing Is Better")
    lines.append("")
    lines.append("- Waterbirds is a spurious-correlation benchmark, so worst-group behavior and environment gaps are the most informative outcomes.")
    lines.append("- A single pooled reference correlation mixes class composition shift with harmful background shift.")
    lines.append("- The benchmark is strongest when reported as matched-vs-mismatched degradation and land-vs-water subgroup gaps within each bird class.")
    lines.append("")
    lines.append("## Correlation Summary")
    lines.append("")
    lines.append(dataframe_to_markdown(correlations))
    lines.append("")
    lines.append("## Matched Vs Mismatched Split Summary")
    lines.append("")
    lines.append(dataframe_to_markdown(alignment_summary))
    lines.append("")
    lines.append("## Environment Sensitivity Within Each Label")
    lines.append("")
    lines.append(dataframe_to_markdown(environment_gaps))
    lines.append("")
    lines.append("## Worst-Group Summary")
    lines.append("")
    lines.append(dataframe_to_markdown(worst_groups))
    lines.append("")

    key_findings = []
    for _, row in worst_groups.iterrows():
        key_findings.append(
            f"- {row['model']} on {row['target_split']} has worst group `{row['worst_group']}` at accuracy {row['worst_group_accuracy']:.3f} with a worst-to-best gap of {row['worst_to_best_accuracy_gap']:.3f}."
        )
    for _, row in environment_gaps.iterrows():
        key_findings.append(
            f"- {row['model']} on {row['target_split']} shows a {row['environment_accuracy_gap']:.3f} environment gap for `{row['label_name']}`."
        )

    lines.append("## Thesis-Safe Takeaways")
    lines.append("")
    lines.extend(key_findings)
    lines.append("- The Waterbirds result is strongest as evidence that subgroup shift can cause major worst-group accuracy collapse even when average feature drift only partially tracks that collapse.")
    lines.append("- This complements the controlled CIFAR-10 and BloodMNIST findings instead of competing with them.")
    lines.append("")

    output_path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a Waterbirds-specific analysis report from existing results.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results_waterbirds"),
        help="Directory containing Waterbirds CSV outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("waterbirds_analysis"),
        help="Directory for generated analysis artifacts.",
    )
    args = parser.parse_args()

    ensure_dir(args.output_dir)

    summary_path = args.results_dir / "experiment_results_summary.csv"
    corr_path = args.results_dir / "correlation_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing Waterbirds summary file: {summary_path}")
    if not corr_path.exists():
        raise FileNotFoundError(f"Missing Waterbirds correlation file: {corr_path}")

    summary_df = pd.read_csv(summary_path)
    corr_df = pd.read_csv(corr_path)

    alignment_summary = build_alignment_summary(summary_df)
    subgroup_summary = build_subgroup_summary(summary_df)
    environment_gaps = build_environment_gap_summary(subgroup_summary)
    worst_groups = build_worst_group_summary(subgroup_summary)

    alignment_summary.to_csv(args.output_dir / "alignment_summary.csv", index=False)
    subgroup_summary.to_csv(args.output_dir / "subgroup_summary.csv", index=False)
    environment_gaps.to_csv(args.output_dir / "environment_gap_summary.csv", index=False)
    worst_groups.to_csv(args.output_dir / "worst_group_summary.csv", index=False)

    write_report(
        args.output_dir / "waterbirds_deep_dive.md",
        corr_df,
        alignment_summary,
        environment_gaps,
        worst_groups,
    )

    print(f"Wrote Waterbirds analysis to {args.output_dir}")


if __name__ == "__main__":
    main()
