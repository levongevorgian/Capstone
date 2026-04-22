# Visualization Interpretations

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
