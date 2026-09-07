# TSFM-BENCHMARK-GAP-DISCOVERY-v1 — status

Generated 2026-09-07T03:32:50.189879+00:00

## 1. Executive verdict

**NO_STRONG_GAP_FOUND** — no descriptor bucket met the registered failure gate on the discovery split

## 2. What was actually evaluated

- 6 estimators × 18 fev-bench tasks × 3 information condition(s) = 174 completed cells
- tracks run: ['C', 'M', 'U']
- splits present: ['confirmation', 'discovery']
- total evaluated origins: 363,180
- total measured inference time: 0.85 GPU-hours

| track   | split        | model                |   tasks |   seconds |
|:--------|:-------------|:---------------------|--------:|----------:|
| C       | confirmation | chronos-2            |       2 |    80.587 |
| C       | confirmation | chronos-2-synth      |       2 |    83.025 |
| C       | confirmation | linear-ar-specialist |       2 |     5.918 |
| C       | confirmation | seasonal-naive       |       2 |     0.992 |
| C       | confirmation | timesfm-3.0          |       2 |   404.034 |
| C       | confirmation | tirex-2              |       2 |   178.2   |
| C       | discovery    | chronos-2            |       3 |    20.425 |
| C       | discovery    | chronos-2-synth      |       3 |    31.006 |
| C       | discovery    | linear-ar-specialist |       3 |    13.094 |
| C       | discovery    | seasonal-naive       |       3 |     0.655 |
| C       | discovery    | timesfm-3.0          |       3 |   125.05  |
| C       | discovery    | tirex-2              |       3 |   144.407 |
| M       | confirmation | chronos-2            |       2 |     0.98  |
| M       | confirmation | chronos-2-synth      |       2 |     2.253 |
| M       | confirmation | linear-ar-specialist |       2 |     0.052 |
| M       | confirmation | seasonal-naive       |       2 |     0.02  |
| M       | confirmation | timesfm-3.0          |       2 |     3.623 |
| M       | confirmation | tirex-2              |       2 |     6.329 |
| M       | discovery    | chronos-2            |       4 |     9.412 |
| M       | discovery    | chronos-2-synth      |       4 |     9.748 |
| M       | discovery    | linear-ar-specialist |       4 |    49.322 |
| M       | discovery    | seasonal-naive       |       4 |     0.076 |
| M       | discovery    | timesfm-3.0          |       4 |    94.923 |
| M       | discovery    | tirex-2              |       4 |    18.932 |
| U       | confirmation | chronos-2            |       6 |    48.208 |
| U       | confirmation | chronos-2-synth      |       6 |    48.847 |
| U       | confirmation | linear-ar-specialist |       6 |     6.71  |
| U       | confirmation | seasonal-naive       |       6 |     1.11  |
| U       | confirmation | timesfm-3.0          |       6 |   351.251 |
| U       | confirmation | tirex-2              |       6 |    81.38  |
| U       | discovery    | chronos-2            |      12 |    47.513 |
| U       | discovery    | chronos-2-synth      |      12 |    49.528 |
| U       | discovery    | linear-ar-specialist |      12 |   306.893 |
| U       | discovery    | seasonal-naive       |      12 |     1.383 |
| U       | discovery    | timesfm-3.0          |      12 |   745.398 |
| U       | discovery    | tirex-2              |      12 |    92.62  |

## 3. Model audit

| model_id        | role                             | hf_repo                    | resolved_revision                        | architecture_family       |   parameter_count_m | license                             | status   |
|:----------------|:---------------------------------|:---------------------------|:-----------------------------------------|:--------------------------|--------------------:|:------------------------------------|:---------|
| chronos-2       | primary                          | amazon/chronos-2           | 29ec3766d36d6f73f0696f85560a422f50e8498c | transformer_encoder       |               120   | apache-2.0                          | READY    |
| tirex-2         | primary                          | NX-AI/TiRex-2              | 05e5b26db52bfb256f1ae1bdf785589850482de3 | recurrent_xlstm           |                82.5 | apache-2.0                          | READY    |
| timesfm-3.0     | primary                          | google/timesfm-3.0-pytorch | 43046b85ec22d584a13f8098c2ed39c889e129c2 | transformer_patch_variate |               330   | timesfm-non-commercial-license-v1.0 | READY    |
| chronos-2-synth | diagnostic_contamination_control | autogluon/chronos-2-synth  | 3607918a9fd027d5c465d8213e46b98e2c041cea | transformer_encoder       |               120   | apache-2.0                          | READY    |

### Contamination status on fev-bench

| model_id        | claimed_fev_bench_exclusion   | contamination_status_fev_bench   |
|:----------------|:------------------------------|:---------------------------------|
| chronos-2       | False                         | OVERLAP_RISK_UNKNOWN             |
| tirex-2         | False                         | OVERLAP_RISK_UNKNOWN             |
| timesfm-3.0     | True                          | CLEAN_BY_OFFICIAL_EXCLUSION      |
| chronos-2-synth | True                          | SYNTH_ONLY_DIAGNOSTIC            |

One substitution was made; see `MODEL_SUBSTITUTION.md`.

## 4. Benchmark audit

- **name**: `fev-bench`
- **pinned_commit**: `eadb28ed3a3f8fc2db8dd4d3d6850894efcbc4d1`
- **task_definition_sha256**: `c7160f61a5e1ded66a3954ef1c514d55d13be18534b34fca817356312a6520a9`
- **evaluator**: `fev (native)`
- **evaluator_version**: `0.10.0`
- **n_tasks**: `100`
- **eval_metric**: `['SQL']`

- split semantics: Each task defines num_windows rolling evaluation windows. A window exposes past data up to its cutoff plus known-future columns over the horizon; the horizon values of the target are never visible to the model.
- leakage policy: fev-bench does not ship a model-specific exclusion list. Pretraining overlap is a property of each checkpoint and is tracked in contamination_matrix.csv.

## 5. Information-condition fairness

| model                | track   | adaptation                | future_covariates_used   | multivariate   | context                           | contamination_status        | comparability_status   |
|:---------------------|:--------|:--------------------------|:-------------------------|:---------------|:----------------------------------|:----------------------------|:-----------------------|
| chronos-2            | C       | zero-shot, no fine-tuning | True                     | False          | native_default; median 7593 steps | OVERLAP_RISK_UNKNOWN        | DIRECT                 |
| chronos-2-synth      | C       | zero-shot, no fine-tuning | True                     | False          | native_default; median 7593 steps | SYNTH_ONLY_DIAGNOSTIC       | PARTIAL                |
| linear-ar-specialist | C       | task-fitted               | True                     | False          | native_default; median 7593 steps | NOT_APPLICABLE              | PARTIAL                |
| seasonal-naive       | C       | zero-shot, no fine-tuning | True                     | False          | native_default; median 7593 steps | NOT_APPLICABLE              | DIRECT                 |
| timesfm-3.0          | C       | zero-shot, no fine-tuning | True                     | False          | native_default; median 7593 steps | CLEAN_BY_OFFICIAL_EXCLUSION | DIRECT                 |
| tirex-2              | C       | zero-shot, no fine-tuning | True                     | False          | native_default; median 7593 steps | OVERLAP_RISK_UNKNOWN        | DIRECT                 |
| chronos-2            | M       | zero-shot, no fine-tuning | False                    | True           | native_default; median 1542 steps | OVERLAP_RISK_UNKNOWN        | DIRECT                 |
| chronos-2-synth      | M       | zero-shot, no fine-tuning | False                    | True           | native_default; median 1542 steps | SYNTH_ONLY_DIAGNOSTIC       | PARTIAL                |
| linear-ar-specialist | M       | task-fitted               | False                    | True           | native_default; median 1542 steps | NOT_APPLICABLE              | PARTIAL                |
| seasonal-naive       | M       | zero-shot, no fine-tuning | False                    | True           | native_default; median 1542 steps | NOT_APPLICABLE              | DIRECT                 |
| timesfm-3.0          | M       | zero-shot, no fine-tuning | False                    | True           | native_default; median 1542 steps | CLEAN_BY_OFFICIAL_EXCLUSION | DIRECT                 |
| tirex-2              | M       | zero-shot, no fine-tuning | False                    | True           | native_default; median 1542 steps | OVERLAP_RISK_UNKNOWN        | DIRECT                 |
| chronos-2            | U       | zero-shot, no fine-tuning | False                    | False          | native_default; median 2220 steps | OVERLAP_RISK_UNKNOWN        | DIRECT                 |
| chronos-2-synth      | U       | zero-shot, no fine-tuning | False                    | False          | native_default; median 2220 steps | SYNTH_ONLY_DIAGNOSTIC       | PARTIAL                |
| linear-ar-specialist | U       | task-fitted               | False                    | False          | native_default; median 2220 steps | NOT_APPLICABLE              | PARTIAL                |
| seasonal-naive       | U       | zero-shot, no fine-tuning | False                    | False          | native_default; median 2220 steps | NOT_APPLICABLE              | DIRECT                 |
| timesfm-3.0          | U       | zero-shot, no fine-tuning | False                    | False          | native_default; median 2220 steps | CLEAN_BY_OFFICIAL_EXCLUSION | DIRECT                 |
| tirex-2              | U       | zero-shot, no fine-tuning | False                    | False          | native_default; median 2220 steps | OVERLAP_RISK_UNKNOWN        | DIRECT                 |

- `chronos-2-synth` (C): PARTIAL — synthetic-only pretraining makes it a contamination anchor rather than a leaderboard entry; kept out of the primary gap map
- `linear-ar-specialist` (C): PARTIAL — fitted on each task's visible history, so it is a supervised specialist and not zero-shot; compared as a headroom anchor, never ranked against zero-shot models as if the information condition were identical
- `chronos-2-synth` (M): PARTIAL — synthetic-only pretraining makes it a contamination anchor rather than a leaderboard entry; kept out of the primary gap map
- `linear-ar-specialist` (M): PARTIAL — fitted on each task's visible history, so it is a supervised specialist and not zero-shot; compared as a headroom anchor, never ranked against zero-shot models as if the information condition were identical
- `chronos-2-synth` (U): PARTIAL — synthetic-only pretraining makes it a contamination anchor rather than a leaderboard entry; kept out of the primary gap map
- `linear-ar-specialist` (U): PARTIAL — fitted on each task's visible history, so it is a supervised specialist and not zero-shot; compared as a headroom anchor, never ranked against zero-shot models as if the information condition were identical

## 6. Discovery task selection

- rule: Round-robin over (domain, frequency bucket) strata with tasks sorted lexically by task_uid inside a stratum, deferring any task whose dataset family has already been drawn; the first 12 tasks are discovery, the next 6 confirmation. Documented repair swaps then guarantee the registered coverage minimums.
- frozen at: 2026-09-07T01:31:19.108087+00:00
- sha256: `f10d6a205a3394a9bbb0bde50d0deaa7a50142ed377462cd7f706e31c7c30234`

**discovery** — 12 tasks, 12 dataset families, domains {'cloud': 2, 'econ': 1, 'energy': 3, 'healthcare': 1, 'mobility': 3, 'nature': 1, 'retail': 1}, horizon buckets {'long': 4, 'medium': 4, 'short': 4}, 4 multivariate, 3 with known covariates

**confirmation** — 6 tasks, 6 dataset families, domains {'cloud': 1, 'econ': 1, 'energy': 2, 'nature': 1, 'retail': 1}, horizon buckets {'long': 1, 'medium': 3, 'short': 2}, 2 multivariate, 2 with known covariates

Descriptor values (train-visible data only):

| task_uid                                         | split        | D1_horizon_ratio   | D2_frequency     | D3_dimensionality   | D4_future_covariates   | D5_past_covariates   | D6_zero_fraction   | D7_missingness   | D8_shift   | D9_seasonal_strength   |
|:-------------------------------------------------|:-------------|:-------------------|:-----------------|:--------------------|:-----------------------|:---------------------|:-------------------|:-----------------|:-----------|:-----------------------|
| fevbench::ETT_1D::ETT_1D                         | discovery    | long               | daily_or_coarser | 2_16                | none                   | none                 | lt0.1              | low              | high       | weak                   |
| fevbench::LOOP_SEATTLE_1D::LOOP_SEATTLE_1D       | discovery    | long               | daily_or_coarser | 1                   | none                   | none                 | lt0.1              | low              | mid        | moderate               |
| fevbench::M_DENSE_1H::M_DENSE_1H                 | discovery    | short              | hourly           | 1                   | none                   | none                 | lt0.1              | low              | low        | moderate               |
| fevbench::SZ_TAXI_15T::SZ_TAXI_15T               | discovery    | medium             | sub_hourly       | 1                   | none                   | none                 | lt0.1              | low              | high       | weak                   |
| fevbench::australian_tourism::australian_tourism | discovery    | long               | daily_or_coarser | 1                   | none                   | none                 | lt0.1              | low              | mid        | weak                   |
| fevbench::bizitobs_l2c_1H::bizitobs_l2c_1H       | discovery    | short              | hourly           | 2_16                | none                   | none                 | 0.1_0.5            | low              | low        | weak                   |
| fevbench::boomlet_1062::boomlet_1062             | discovery    | medium             | sub_hourly       | gt16                | none                   | none                 | lt0.1              | low              | low        | weak                   |
| fevbench::ecdc_ili::ecdc_ili                     | discovery    | medium             | daily_or_coarser | 1                   | none                   | none                 | 0.1_0.5            | low              | high       | nan                    |
| fevbench::entsoe_1H::entsoe_1H                   | discovery    | short              | hourly           | 1                   | available              | none                 | lt0.1              | low              | mid        | moderate               |
| fevbench::hermes::hermes                         | discovery    | long               | daily_or_coarser | 1                   | available              | none                 | lt0.1              | low              | high       | nan                    |
| fevbench::kdd_cup_2022_10T::kdd_cup_2022_10T     | discovery    | short              | sub_hourly       | 1                   | none                   | available            | lt0.1              | high             | low        | weak                   |
| fevbench::uci_air_quality_1H::uci_air_quality_1H | discovery    | medium             | hourly           | 2_16                | available              | none                 | lt0.1              | high             | mid        | weak                   |
| fevbench::epf_be::epf_be                         | confirmation | short              | hourly           | 1                   | available              | none                 | lt0.1              | low              | high       | weak                   |
| fevbench::ercot_1D::ercot_1D                     | confirmation | short              | daily_or_coarser | 1                   | none                   | none                 | lt0.1              | low              | high       | moderate               |
| fevbench::favorita_stores_1D::favorita_stores_1D | confirmation | short              | daily_or_coarser | 1                   | available              | available            | 0.1_0.5            | low              | high       | weak                   |
| fevbench::fred_md_2025::fred_md_2025/cee         | confirmation | short              | daily_or_coarser | 2_16                | none                   | available            | lt0.1              | low              | high       | nan                    |
| fevbench::jena_weather_1D::jena_weather_1D       | confirmation | long               | daily_or_coarser | gt16                | none                   | none                 | lt0.1              | low              | high       | nan                    |
| fevbench::redset_1H::redset_1H                   | confirmation | short              | hourly           | 1                   | none                   | none                 | 0.1_0.5            | low              | mid        | weak                   |

## 7. Raw benchmark results

### TRACK C — confirmation: native SQL / SeasonalNaive SQL (lower is better)

| task_uid                                         |   chronos-2 |   chronos-2-synth |   linear-ar-specialist |   seasonal-naive |   timesfm-3.0 |   tirex-2 |
|:-------------------------------------------------|------------:|------------------:|-----------------------:|-----------------:|--------------:|----------:|
| fevbench::epf_be::epf_be                         |       0.516 |             0.557 |                  1.114 |                1 |         0.514 |     0.546 |
| fevbench::favorita_stores_1D::favorita_stores_1D |       0.572 |             0.567 |                  0.783 |                1 |         0.55  |     0.583 |

Median across tasks:

| model                |   median relative_to_naive |
|:---------------------|---------------------------:|
| chronos-2            |                      0.544 |
| chronos-2-synth      |                      0.562 |
| linear-ar-specialist |                      0.949 |
| seasonal-naive       |                      1     |
| timesfm-3.0          |                      0.532 |
| tirex-2              |                      0.564 |

### TRACK C — discovery: native SQL / SeasonalNaive SQL (lower is better)

| task_uid                                         |   chronos-2 |   chronos-2-synth |   linear-ar-specialist |   seasonal-naive |   timesfm-3.0 |   tirex-2 |
|:-------------------------------------------------|------------:|------------------:|-----------------------:|-----------------:|--------------:|----------:|
| fevbench::entsoe_1H::entsoe_1H                   |       0.388 |             0.404 |                  0.912 |                1 |         0.314 |     0.449 |
| fevbench::hermes::hermes                         |       0.281 |             0.281 |                  0.454 |                1 |         0.279 |     0.309 |
| fevbench::uci_air_quality_1H::uci_air_quality_1H |       0.606 |             0.665 |                  0.915 |                1 |         0.56  |     0.632 |

Median across tasks:

| model                |   median relative_to_naive |
|:---------------------|---------------------------:|
| chronos-2            |                      0.388 |
| chronos-2-synth      |                      0.404 |
| linear-ar-specialist |                      0.912 |
| seasonal-naive       |                      1     |
| timesfm-3.0          |                      0.314 |
| tirex-2              |                      0.449 |

### TRACK M — confirmation: native SQL / SeasonalNaive SQL (lower is better)

| task_uid                                   |   chronos-2 |   chronos-2-synth |   linear-ar-specialist |   seasonal-naive |   timesfm-3.0 |   tirex-2 |
|:-------------------------------------------|------------:|------------------:|-----------------------:|-----------------:|--------------:|----------:|
| fevbench::fred_md_2025::fred_md_2025/cee   |       0.639 |             0.714 |                  0.778 |                1 |         0.563 |     0.556 |
| fevbench::jena_weather_1D::jena_weather_1D |       0.517 |             0.525 |                  0.664 |                1 |         0.532 |     0.515 |

Median across tasks:

| model                |   median relative_to_naive |
|:---------------------|---------------------------:|
| chronos-2            |                      0.578 |
| chronos-2-synth      |                      0.619 |
| linear-ar-specialist |                      0.721 |
| seasonal-naive       |                      1     |
| timesfm-3.0          |                      0.547 |
| tirex-2              |                      0.535 |

### TRACK M — discovery: native SQL / SeasonalNaive SQL (lower is better)

| task_uid                                         |   chronos-2 |   chronos-2-synth |   linear-ar-specialist |   seasonal-naive |   timesfm-3.0 |   tirex-2 |
|:-------------------------------------------------|------------:|------------------:|-----------------------:|-----------------:|--------------:|----------:|
| fevbench::ETT_1D::ETT_1D                         |       0.825 |             0.838 |                  0.991 |                1 |         0.821 |     0.817 |
| fevbench::bizitobs_l2c_1H::bizitobs_l2c_1H       |       0.342 |             0.351 |                  0.616 |                1 |         0.32  |     0.39  |
| fevbench::boomlet_1062::boomlet_1062             |       0.56  |             0.566 |                  0.706 |                1 |         0.553 |     0.563 |
| fevbench::uci_air_quality_1H::uci_air_quality_1H |       0.677 |             0.729 |                  0.915 |                1 |         0.638 |     0.647 |

Median across tasks:

| model                |   median relative_to_naive |
|:---------------------|---------------------------:|
| chronos-2            |                      0.618 |
| chronos-2-synth      |                      0.647 |
| linear-ar-specialist |                      0.81  |
| seasonal-naive       |                      1     |
| timesfm-3.0          |                      0.596 |
| tirex-2              |                      0.605 |

### TRACK U — confirmation: native SQL / SeasonalNaive SQL (lower is better)

| task_uid                                         |   chronos-2 |   chronos-2-synth |   linear-ar-specialist |   seasonal-naive |   timesfm-3.0 |   tirex-2 |
|:-------------------------------------------------|------------:|------------------:|-----------------------:|-----------------:|--------------:|----------:|
| fevbench::epf_be::epf_be                         |       0.556 |             0.629 |                  1.114 |                1 |         0.525 |     0.594 |
| fevbench::ercot_1D::ercot_1D                     |       0.664 |             0.667 |                  0.833 |                1 |         0.588 |     0.609 |
| fevbench::favorita_stores_1D::favorita_stores_1D |       0.573 |             0.582 |                  0.783 |                1 |         0.567 |     0.601 |
| fevbench::fred_md_2025::fred_md_2025/cee         |       0.654 |             0.722 |                  0.778 |                1 |         0.582 |     0.559 |
| fevbench::jena_weather_1D::jena_weather_1D       |       0.501 |             0.555 |                  0.664 |                1 |         0.508 |     0.513 |
| fevbench::redset_1H::redset_1H                   |       0.613 |             0.658 |                  1.068 |                1 |         0.629 |     0.686 |

Median across tasks:

| model                |   median relative_to_naive |
|:---------------------|---------------------------:|
| chronos-2            |                      0.593 |
| chronos-2-synth      |                      0.643 |
| linear-ar-specialist |                      0.808 |
| seasonal-naive       |                      1     |
| timesfm-3.0          |                      0.575 |
| tirex-2              |                      0.597 |

### TRACK U — discovery: native SQL / SeasonalNaive SQL (lower is better)

| task_uid                                         |   chronos-2 |   chronos-2-synth |   linear-ar-specialist |   seasonal-naive |   timesfm-3.0 |   tirex-2 |
|:-------------------------------------------------|------------:|------------------:|-----------------------:|-----------------:|--------------:|----------:|
| fevbench::ETT_1D::ETT_1D                         |       0.808 |             0.837 |                  0.991 |                1 |         0.817 |     0.818 |
| fevbench::LOOP_SEATTLE_1D::LOOP_SEATTLE_1D       |       0.741 |             0.753 |                  0.951 |                1 |         0.725 |     0.743 |
| fevbench::M_DENSE_1H::M_DENSE_1H                 |       0.457 |             0.471 |                  0.911 |                1 |         0.396 |     0.466 |
| fevbench::SZ_TAXI_15T::SZ_TAXI_15T               |       0.631 |             0.641 |                  0.832 |                1 |         0.628 |     0.635 |
| fevbench::australian_tourism::australian_tourism |       0.827 |             0.818 |                  1     |                1 |         0.884 |     0.831 |
| fevbench::bizitobs_l2c_1H::bizitobs_l2c_1H       |       0.372 |             0.371 |                  0.616 |                1 |         0.349 |     0.385 |
| fevbench::boomlet_1062::boomlet_1062             |       0.565 |             0.573 |                  0.706 |                1 |         0.558 |     0.564 |
| fevbench::ecdc_ili::ecdc_ili                     |       0.653 |             0.678 |                  0.802 |                1 |         0.568 |     0.582 |
| fevbench::entsoe_1H::entsoe_1H                   |       0.405 |             0.448 |                  0.912 |                1 |         0.33  |     0.451 |
| fevbench::hermes::hermes                         |       0.29  |             0.289 |                  0.454 |                1 |         0.284 |     0.302 |
| fevbench::kdd_cup_2022_10T::kdd_cup_2022_10T     |       0.595 |             0.606 |                  0.705 |                1 |         0.547 |     0.561 |
| fevbench::uci_air_quality_1H::uci_air_quality_1H |       0.695 |             0.742 |                  0.915 |                1 |         0.644 |     0.649 |

Median across tasks:

| model                |   median relative_to_naive |
|:---------------------|---------------------------:|
| chronos-2            |                      0.613 |
| chronos-2-synth      |                      0.623 |
| linear-ar-specialist |                      0.872 |
| seasonal-naive       |                      1     |
| timesfm-3.0          |                      0.563 |
| tirex-2              |                      0.573 |

Every score above is fev's own `evaluation_summary` output. A per-origin decomposition of the same metric is kept alongside it so the oracle probes and any task-internal resampling operate on the same quantity.

- it reproduces the official aggregate to below 1e-6 relative on 16/18 tasks; the floor of 1.6e-08 is float32 storage of the per-origin array, not a modelling difference
- it reaches 3.7e-03 on kdd_cup_2022_10T, uci_air_quality_1H. fev's SQL averages per-target-dimension means, and on these tasks ground truth is missing at different rates across the target dimensions, which no mean of per-item quantities can match. Baseline, oracle and simple fixes are aggregated identically inside the per-origin space, so the residual cancels out of every headroom ratio.

## 8. Failure map

Full table in `failure_map.md`. Gate outcome:

| descriptor   | descriptor_label              | bucket           |   n_tasks_in_bucket |   n_affected_families |   median_condition_gap_pct |   median_regret_pct | passes_failure_gate   | verdict         |
|:-------------|:------------------------------|:-----------------|--------------------:|----------------------:|---------------------------:|--------------------:|:----------------------|:----------------|
| D1           | horizon ratio                 | long             |                   4 |                     3 |                     38.812 |               1.586 | False                 | NOT_A_CANDIDATE |
| D1           | horizon ratio                 | medium           |                   4 |                     3 |                     22.018 |               0.997 | False                 | NOT_A_CANDIDATE |
| D1           | horizon ratio                 | short            |                   4 |                     0 |                    nan     |              12.127 | False                 | NOT_A_CANDIDATE |
| D2           | frequency                     | daily_or_coarser |                   5 |                     3 |                     32.492 |               2.088 | False                 | NOT_A_CANDIDATE |
| D2           | frequency                     | hourly           |                   4 |                     0 |                    nan     |              11.705 | False                 | NOT_A_CANDIDATE |
| D2           | frequency                     | sub_hourly       |                   3 |                     0 |                    nan     |               1.034 | False                 | NOT_A_CANDIDATE |
| D3           | target dimensionality         | 1                |                   8 |                     0 |                    nan     |               2.5   | False                 | NOT_A_CANDIDATE |
| D3           | target dimensionality         | 2_16             |                   3 |                     3 |                     15.3   |               1.24  | False                 | NOT_A_CANDIDATE |
| D3           | target dimensionality         | gt16             |                   1 |                     0 |                    nan     |               0.96  | False                 | CASE_ONLY       |
| D4           | future covariates             | available        |                   3 |                     0 |                    nan     |               6.573 | False                 | NOT_A_CANDIDATE |
| D4           | future covariates             | none             |                   9 |                     3 |                     55.683 |               2.204 | False                 | NOT_A_CANDIDATE |
| D5           | past covariates               | available        |                   1 |                     0 |                    nan     |               2.514 | False                 | CASE_ONLY       |
| D5           | past covariates               | none             |                  11 |                     1 |                      5.958 |               2.204 | False                 | NOT_A_CANDIDATE |
| D6           | zero fraction / intermittency | 0.1_0.5          |                   2 |                     0 |                    nan     |               6.366 | False                 | CASE_ONLY       |
| D6           | zero fraction / intermittency | lt0.1            |                  10 |                     3 |                     23.996 |               2.025 | False                 | NOT_A_CANDIDATE |
| D7           | missingness                   | high             |                   2 |                     3 |                      5.712 |               1.658 | False                 | CASE_ONLY       |
| D7           | missingness                   | low              |                  10 |                     0 |                    nan     |               2.146 | False                 | NOT_A_CANDIDATE |
| D8           | train-only distribution shift | high             |                   4 |                     3 |                      8.253 |               1.243 | False                 | NOT_A_CANDIDATE |
| D8           | train-only distribution shift | low              |                   4 |                     0 |                    nan     |               6.38  | False                 | NOT_A_CANDIDATE |
| D8           | train-only distribution shift | mid              |                   4 |                     3 |                     23.814 |               2.025 | False                 | NOT_A_CANDIDATE |
| D9           | seasonal strength             | moderate         |                   3 |                     0 |                    nan     |              15.368 | False                 | NOT_A_CANDIDATE |
| D9           | seasonal strength             | weak             |                   7 |                     3 |                     38.073 |               1.191 | False                 | NOT_A_CANDIDATE |

## 9. Candidate failures

No condition passed the failure gate, so no candidate was generated.

## 10. Oracle and headroom probes

| task_uid                                         |   BASELINE_strongest_foundation_model |   P0_cross_model_origin_oracle |   P1_validation_selected_model |   P3_affine_calibration |   P4_supervised_specialist |   P5_covariate_ablation |   P6_quantile_average_ensemble |
|:-------------------------------------------------|--------------------------------------:|-------------------------------:|-------------------------------:|------------------------:|---------------------------:|------------------------:|-------------------------------:|
| fevbench::ETT_1D::ETT_1D                         |                                1.1369 |                         1.0833 |                         1.1304 |                  1.2214 |                     1.3724 |                nan      |                         1.1231 |
| fevbench::LOOP_SEATTLE_1D::LOOP_SEATTLE_1D       |                                0.7704 |                         0.7422 |                         0.776  |                  0.7739 |                     1.0099 |                nan      |                         0.7706 |
| fevbench::M_DENSE_1H::M_DENSE_1H                 |                                0.5169 |                         0.5136 |                         0.5169 |                  0.5566 |                     1.1683 |                nan      |                         0.5582 |
| fevbench::SZ_TAXI_15T::SZ_TAXI_15T               |                                0.3963 |                         0.391  |                         0.3963 |                  0.4019 |                     0.5222 |                nan      |                         0.3962 |
| fevbench::australian_tourism::australian_tourism |                                0.6831 |                         0.6146 |                         0.6757 |                  0.6742 |                     0.847  |                nan      |                         0.6534 |
| fevbench::bizitobs_l2c_1H::bizitobs_l2c_1H       |                                0.3229 |                         0.3138 |                         0.3285 |                  0.3429 |                     0.5428 |                nan      |                         0.3311 |
| fevbench::boomlet_1062::boomlet_1062             |                                0.5408 |                         0.5384 |                         0.5429 |                  0.9009 |                     0.6878 |                nan      |                         0.5395 |
| fevbench::ecdc_ili::ecdc_ili                     |                                1.9163 |                         1.6465 |                         1.9163 |                  3.8514 |                     2.7687 |                nan      |                         1.9595 |
| fevbench::entsoe_1H::entsoe_1H                   |                                0.3478 |                         0.3309 |                         0.3478 |                  0.3672 |                     0.9821 |                  0.3305 |                         0.393  |
| fevbench::hermes::hermes                         |                                0.6088 |                         0.577  |                       nan      |                nan      |                     0.9746 |                  0.5987 |                         0.6119 |
| fevbench::kdd_cup_2022_10T::kdd_cup_2022_10T     |                                0.3798 |                         0.3258 |                         0.398  |                  0.5973 |                     0.5271 |                nan      |                         0.3726 |
| fevbench::uci_air_quality_1H::uci_air_quality_1H |                                0.8149 |                         0.7884 |                         0.8179 |                  2.5186 |                     1.1631 |                  0.6981 |                         0.8166 |

_no rows_

`P0_cross_model_origin_oracle` reads evaluation labels. It is a diagnostic upper bound, not an achievable score.

## 10b. Is the oracle headroom reachable?

Over all 12 discovery tasks treated as one set, the per-origin oracle sits 6.75% below the strongest single model, and the best simple deployable fix (`P6_quantile_average_ensemble`) recovers -16% of that - it is worse than doing nothing.

Bootstrapping origins within windows, paired across probes: oracle headroom 6.95% [5.76, 8.58]. The interval is conditional on these tasks and says nothing about which tasks the benchmark contains.

Taken alone that number would look like a research opportunity. It is not, and two tests say so. Picking the lowest of three losses at every origin beats any single model even when the three are statistically indistinguishable, so the question is whether the winner is a property of the series or is redrawn each window.

| split        |   tasks |   oracle headroom % |   per-series router % |   headroom captured |   winner agreement |   chance agreement |   excess |   pairs |
|:-------------|--------:|--------------------:|----------------------:|--------------------:|-------------------:|-------------------:|---------:|--------:|
| discovery    |      11 |                5.6  |                 -1.85 |              -0.33  |             0.3716 |             0.3657 |   0.0059 |    6300 |
| confirmation |       6 |                6.48 |                 -1.46 |              -0.225 |             0.3847 |             0.3577 |   0.027  |   15644 |

The winner agrees with the previous window barely above the rate the per-window marginals already predict, and a router that picks each series' past leader loses to simply committing to the strongest single model. Both splits agree. The oracle gap is selection noise, not structure a method could capture.

One exception is worth naming rather than averaging away: redset_1H, fred_md_2025 shows winner persistence clearly above chance and a positive router gain. It is a single task and a lead, not a result.

## 11. Simple baselines

| track   | task_uid                                         |   linear-ar-specialist |   seasonal-naive |
|:--------|:-------------------------------------------------|-----------------------:|-----------------:|
| C       | fevbench::entsoe_1H::entsoe_1H                   |                 0.9794 |           1.074  |
| C       | fevbench::epf_be::epf_be                         |                 1.0869 |           0.9756 |
| C       | fevbench::favorita_stores_1D::favorita_stores_1D |                 1.293  |           1.6509 |
| C       | fevbench::hermes::hermes                         |                 0.9746 |           2.1461 |
| C       | fevbench::uci_air_quality_1H::uci_air_quality_1H |                 1.2049 |           1.3167 |
| M       | fevbench::ETT_1D::ETT_1D                         |                 1.3603 |           1.3728 |
| M       | fevbench::bizitobs_l2c_1H::bizitobs_l2c_1H       |                 0.5426 |           0.8804 |
| M       | fevbench::boomlet_1062::boomlet_1062             |                 0.6962 |           0.9865 |
| M       | fevbench::fred_md_2025::fred_md_2025/cee         |                 4.6073 |           5.9196 |
| M       | fevbench::jena_weather_1D::jena_weather_1D       |                 1.428  |           2.1502 |
| M       | fevbench::uci_air_quality_1H::uci_air_quality_1H |                 1.2049 |           1.3167 |
| U       | fevbench::ETT_1D::ETT_1D                         |                 1.3603 |           1.3728 |
| U       | fevbench::LOOP_SEATTLE_1D::LOOP_SEATTLE_1D       |                 0.9954 |           1.0462 |
| U       | fevbench::M_DENSE_1H::M_DENSE_1H                 |                 1.1686 |           1.2823 |
| U       | fevbench::SZ_TAXI_15T::SZ_TAXI_15T               |                 0.5193 |           0.6244 |
| U       | fevbench::australian_tourism::australian_tourism |                 0.8947 |           0.8947 |
| U       | fevbench::bizitobs_l2c_1H::bizitobs_l2c_1H       |                 0.5426 |           0.8804 |
| U       | fevbench::boomlet_1062::boomlet_1062             |                 0.6962 |           0.9865 |
| U       | fevbench::ecdc_ili::ecdc_ili                     |                 3.0289 |           3.7764 |
| U       | fevbench::entsoe_1H::entsoe_1H                   |                 0.9794 |           1.074  |
| U       | fevbench::epf_be::epf_be                         |                 1.0869 |           0.9756 |
| U       | fevbench::ercot_1D::ercot_1D                     |                 1.1041 |           1.3252 |
| U       | fevbench::favorita_stores_1D::favorita_stores_1D |                 1.293  |           1.6509 |
| U       | fevbench::fred_md_2025::fred_md_2025/cee         |                 4.6073 |           5.9196 |
| U       | fevbench::hermes::hermes                         |                 0.9746 |           2.1461 |
| U       | fevbench::jena_weather_1D::jena_weather_1D       |                 1.428  |           2.1502 |
| U       | fevbench::kdd_cup_2022_10T::kdd_cup_2022_10T     |                 0.5636 |           0.8    |
| U       | fevbench::redset_1H::redset_1H                   |                 2.2641 |           2.1202 |
| U       | fevbench::uci_air_quality_1H::uci_air_quality_1H |                 1.2049 |           1.3167 |

## 12. Confirmation

The confirmation split was not opened: no candidate reached the confirmation stage.
The 6 holdout tasks remain unused and are listed in `selected_tasks.json`.

## 13. Literature ownership

No candidate reached the literature audit stage.

## 14. Candidate ranking

No candidate to rank.

## 15. Integrity and deviations

20/20 checks pass. Detail in `SELF_AUDIT.md`.

| check   | result   | description                                                       |
|:--------|:---------|:------------------------------------------------------------------|
| A01     | PASS     | selected_tasks hash frozen before model scores                    |
| A02     | PASS     | discovery and confirmation are disjoint                           |
| A03     | PASS     | same target timestamps within a comparable track                  |
| A04     | PASS     | covariate information conditions match inside a track             |
| A18     | PASS     | simple baseline uses the same information condition               |
| A05     | PASS     | no future target leakage                                          |
| A06     | PASS     | benchmark official metric reproduced on a reference decomposition |
| A19     | PASS     | per-model failures recomputable from the raw result matrix        |
| A07     | PASS     | model revision pinned                                             |
| A08     | PASS     | exact context length recorded                                     |
| A09     | PASS     | SeasonalNaive uses only legal past                                |
| A10     | PASS     | supervised specialist uses visible history only                   |
| A11     | PASS     | confirmation untouched before candidate freeze                    |
| A17     | PASS     | all candidate thresholds frozen before confirmation               |
| A12     | PASS     | diagnostic oracle clearly tagged non-deployable                   |
| A13     | PASS     | no oracle value mixed into a deployable leaderboard               |
| A14     | PASS     | task-level normalisation direction correct                        |
| A15     | PASS     | result exclusion rule applied symmetrically                       |
| A16     | PASS     | contamination status present for every model                      |
| A20     | PASS     | candidate scores recomputable from artifacts                      |

### Deviations from the study contract

- The recurrent-family slot uses `NX-AI/TiRex-2` rather than the advertised fev-bench decontaminated checkpoint, because that repository publishes byte-identical weights. Recorded in `MODEL_SUBSTITUTION.md`.
- TiRex-2 runs its pure-PyTorch kernels on CUDA: this host has neither an MSVC toolchain nor nvcc, so the fused FlashRNN/Triton kernels cannot be built. CPU and CUDA outputs were measured to agree to a relative 1.0e-06 (see `smoke_report.json`).
- B0 SeasonalNaive and B1 the linear autoregressive specialist are implemented in-repo rather than through statsforecast/autogluon, whose install downgrades pandas underneath three already-verified foundation models. The estimator contracts are stated in `baselines.py` and covered by `tests/test_study_contract.py`.
- GIFT-Eval and TIME were not run. Both are secondary or optional in the contract and would each need a second harness and data download; the budget went to the primary benchmark plus its holdout instead.

## 16. What was NOT tested

- GIFT-Eval, TIME, and every fev-bench task outside the frozen 18.
- Fine-tuning of any foundation model: the contract confines this study to zero-shot inference, simple baselines and probes.
- Whether the observed behaviour generalises beyond fev-bench. One benchmark cannot support a claim about time-series forecasting in general.
- Absolute model ranking under a clean contamination contract: two of the three primary models carry unresolved fev-bench overlap risk.
- The confirmation split, which stays sealed unless a candidate reaches it.

## 17. Recommended next action

Do not manufacture a method topic from this run. The next search should widen the space rather than dig here. What this run's own numbers point to, in order:

1. The largest condition effects are real but shared: `D4`/`none` (future covariates, 56% over 9 tasks), `D1`/`long` (horizon ratio, 39% over 4 tasks), `D9`/`weak` (seasonal strength, 38% over 7 tasks). Every one of them failed the gate on regret alone. These conditions are genuinely harder for all three families at once, and on them no evaluated estimator - not the specialist, not the naive anchor, not another foundation model - does better. What is missing is a comparator that actually wins there, not more tasks. The productive next step is to find or build one strong task-specific model for a condition like these; if it beats the foundation models by a wide margin, the gap becomes measurable, and if it does not, the condition is simply hard and should be dropped.
2. Conditions that could not be tested at all for want of tasks: D3/gt16, D5/available, D6/0.1_0.5, D7/high. A selection drawn to fill these buckets specifically, rather than to span the benchmark evenly, would test them without needing a new benchmark.
3. A second benchmark. Everything here is fev-bench; GIFT-Eval and TIME were left unrun, and a condition that fails to reach the gate on one benchmark's 18 tasks may simply be under-sampled rather than absent.
4. A different task family. This study only looked at forecasting. Classification, anomaly detection and imputation are where these same backbones have had the least public benchmarking, so the prior for an open gap there is higher than for another pass over forecasting accuracy.
