# Waterbirds Deep Dive

This report improves the Waterbirds presentation by focusing on subgroup behavior rather than only a pooled drift-performance correlation.

## Why This Framing Is Better

- Waterbirds is a spurious-correlation benchmark, so worst-group behavior and environment gaps are the most informative outcomes.
- A single pooled reference correlation mixes class composition shift with harmful background shift.
- The benchmark is strongest when reported as matched-vs-mismatched degradation and land-vs-water subgroup gaps within each bird class.

## Correlation Summary

| dataset    | model           | pearson_accuracy_vs_ks | pearson_accuracy_vs_mmd | spearman_accuracy_vs_ks | spearman_accuracy_vs_mmd |
| ---------- | --------------- | ---------------------- | ----------------------- | ----------------------- | ------------------------ |
| waterbirds | ResNet-50       | 0.3345757282969682     | 0.3314219557980495      | 0.428695652173913       | 0.3817391304347826       |
| waterbirds | EfficientNet-B0 | 0.3453804186900272     | 0.3288118946414866      | 0.4843478260869564      | 0.477391304347826        |

## Matched Vs Mismatched Split Summary

| model           | target_split | alignment  | drift_accuracy     | accuracy_drop       | ks_mean            | mmd                | group_size |
| --------------- | ------------ | ---------- | ------------------ | ------------------- | ------------------ | ------------------ | ---------- |
| EfficientNet-B0 | test         | matched    | 0.95               | 0.0166666666666666  | 0.096796875        | 0.074928481131792  | 150.0      |
| EfficientNet-B0 | test         | mismatched | 0.55               | 0.4166666666666666  | 0.130859375        | 0.1315942928194999 | 150.0      |
| EfficientNet-B0 | val          | matched    | 0.9833333333333334 | -0.0166666666666666 | 0.081484375        | 0.0587197989225387 | 150.0      |
| EfficientNet-B0 | val          | mismatched | 0.4833333333333333 | 0.4833333333333333  | 0.136328125        | 0.1316278837621212 | 150.0      |
| ResNet-50       | test         | matched    | 0.9633333333333334 | -0.0033333333333332 | 0.094375           | 0.0695191211998462 | 150.0      |
| ResNet-50       | test         | mismatched | 0.5433333333333333 | 0.4166666666666667  | 0.1301822916666666 | 0.1492703482508659 | 150.0      |
| ResNet-50       | val          | matched    | 0.98               | -0.0199999999999999 | 0.0830208333333333 | 0.0620838515460491 | 150.0      |
| ResNet-50       | val          | mismatched | 0.5433333333333333 | 0.4166666666666667  | 0.1348177083333333 | 0.1678484380245208 | 150.0      |

## Environment Sensitivity Within Each Label

| model           | target_split | label_name | land_background_accuracy | water_background_accuracy | environment_accuracy_gap | land_background_ks | water_background_ks | land_background_mmd | water_background_mmd |
| --------------- | ------------ | ---------- | ------------------------ | ------------------------- | ------------------------ | ------------------ | ------------------- | ------------------- | -------------------- |
| EfficientNet-B0 | test         | landbird   | 0.98                     | 0.5533333333333333        | 0.42666666666666664      | 0.10953125         | 0.1530208333333333  | 0.0917903371155262  | 0.1616091877222061   |
| EfficientNet-B0 | test         | waterbird  | 0.3399999999999999       | 0.89                      | 0.55                     | 0.139921875        | 0.1826302083333333  | 0.1539737209677696  | 0.2125840187072754   |
| EfficientNet-B0 | val          | landbird   | 0.9833333333333334       | 0.5766666666666667        | 0.40666666666666673      | 0.0908854166666666 | 0.1495052083333333  | 0.0787941440939903  | 0.1621114015579223   |
| EfficientNet-B0 | val          | waterbird  | 0.3759398496240601       | 0.913533834586466         | 0.5375939849624058       | 0.1423645050125313 | 0.1824579025689223  | 0.14974045753479    | 0.2181305587291717   |
| ResNet-50       | test         | landbird   | 0.99                     | 0.56                      | 0.42999999999999994      | 0.1065625          | 0.1511979166666666  | 0.0942493341863155  | 0.2039560675621032   |
| ResNet-50       | test         | waterbird  | 0.42                     | 0.9066666666666668        | 0.48666666666666686      | 0.1566666666666666 | 0.1944791666666666  | 0.1970386803150177  | 0.265118196606636    |
| ResNet-50       | val          | landbird   | 0.99                     | 0.5966666666666667        | 0.3933333333333333       | 0.0903645833333333 | 0.1493489583333333  | 0.0842261984944343  | 0.1895192563533783   |
| ResNet-50       | val          | waterbird  | 0.4060150375939849       | 0.913533834586466         | 0.507518796992481        | 0.1539738016917293 | 0.1939947525062656  | 0.1871050968766212  | 0.2754631787538528   |

## Worst-Group Summary

| model           | target_split | worst_group         | worst_group_accuracy | best_group         | best_group_accuracy | worst_to_best_accuracy_gap |
| --------------- | ------------ | ------------------- | -------------------- | ------------------ | ------------------- | -------------------------- |
| EfficientNet-B0 | test         | test_waterbird_land | 0.3399999999999999   | test_landbird_land | 0.98                | 0.6400000000000001         |
| EfficientNet-B0 | val          | val_waterbird_land  | 0.3759398496240601   | val_landbird_land  | 0.9833333333333334  | 0.6073934837092734         |
| ResNet-50       | test         | test_waterbird_land | 0.42                 | test_landbird_land | 0.99                | 0.5700000000000001         |
| ResNet-50       | val          | val_waterbird_land  | 0.4060150375939849   | val_landbird_land  | 0.99                | 0.5839849624060152         |

## Thesis-Safe Takeaways

- EfficientNet-B0 on test has worst group `test_waterbird_land` at accuracy 0.340 with a worst-to-best gap of 0.640.
- EfficientNet-B0 on val has worst group `val_waterbird_land` at accuracy 0.376 with a worst-to-best gap of 0.607.
- ResNet-50 on test has worst group `test_waterbird_land` at accuracy 0.420 with a worst-to-best gap of 0.570.
- ResNet-50 on val has worst group `val_waterbird_land` at accuracy 0.406 with a worst-to-best gap of 0.584.
- EfficientNet-B0 on test shows a 0.427 environment gap for `landbird`.
- EfficientNet-B0 on test shows a 0.550 environment gap for `waterbird`.
- EfficientNet-B0 on val shows a 0.407 environment gap for `landbird`.
- EfficientNet-B0 on val shows a 0.538 environment gap for `waterbird`.
- ResNet-50 on test shows a 0.430 environment gap for `landbird`.
- ResNet-50 on test shows a 0.487 environment gap for `waterbird`.
- ResNet-50 on val shows a 0.393 environment gap for `landbird`.
- ResNet-50 on val shows a 0.508 environment gap for `waterbird`.
- The Waterbirds result is strongest as evidence that subgroup shift can cause major worst-group accuracy collapse even when average feature drift only partially tracks that collapse.
- This complements the controlled CIFAR-10 and BloodMNIST findings instead of competing with them.
