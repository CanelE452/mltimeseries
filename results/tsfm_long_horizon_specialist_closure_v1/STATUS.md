# TSFM-LONG-HORIZON-SPECIALIST-CLOSURE-v1 — status

Generated 2026-09-07T08:13:57.665280+00:00

## 1. Executive verdict

**SHARED_TASK_DIFFICULTY_CLOSE** — a strong supervised specialist did not expose the registered 5-8% recoverable gap on either split; the long-horizon deficit looks like task difficulty every model shares

## 2. The unresolved question this closes

The source study found that on long-horizon tasks all three foundation model families lost ground together: a median condition gap of 38.8% across 4 tasks and 3 architecture families. It failed the registered gap gate on regret alone: 1.59% against the best evaluated deployable estimator, far short of the 8% threshold.

Those comparators were other foundation models, SeasonalNaive and a linear ridge autoregression. So the evidence said the models struggle together, but it could not say whether a genuinely strong task-trained model would do better. That is what this study tests, and nothing else: it does not ask why.

## 3. Source study provenance

- **source_branch**: `tsfm-benchmark-gap-discovery-v1`
- **source_remote_sha**: `dbdc8b82feb9fcec6448e647a2eef7141d0136fa`
- **source_final_token**: `NO_STRONG_GAP_FOUND`
- **source_benchmark**: `fev-bench`
- **source_benchmark_commit**: `eadb28ed3a3f8fc2db8dd4d3d6850894efcbc4d1`
- **selected_tasks_sha256**: `f10d6a205a3394a9bbb0bde50d0deaa7a50142ed377462cd7f706e31c7c30234`

Source artifacts are read only. `verify` check A20 diffs them against the base commit.

## 4. Development tasks — the source D1-long set

Re-derived from the source `task_metadata.csv`, not copied from a list:

- `fevbench::ETT_1D::ETT_1D`
- `fevbench::LOOP_SEATTLE_1D::LOOP_SEATTLE_1D`
- `fevbench::australian_tourism::australian_tourism`
- `fevbench::hermes::hermes`

## 5. Fresh holdout selection

- rule: Exclude the source study's 18 tasks and every task sharing a dataset family with them. Bound horizon/visible-context from metadata, keep whatever the bound cannot rule out, measure the exact ratio on window 0's visible past, keep the ones above the frozen long cut. Then round-robin over (domain, frequency bucket) strata with tasks in lexical order inside a stratum, taking at most one task per dataset family, and stop at 6.
- long definition: `horizon_to_context_ratio > 0.1652439024390244` (threshold reused from the source study)
- frozen at: 2026-09-07T05:49:15.265321+00:00
- sha256: `38890e6b71573aa9ccfc18e123893c69642d2998385df50d68c1faa7d27bf5e2`
- funnel: 36 candidates after excluding the previous 18 tasks and their families -> 19 past the metadata bound -> 13 measured long -> 6 selected

- coverage: {'dataset_families': ['hospital', 'm5', 'restaurant', 'solar', 'us', 'world'], 'domains': {'econ': 2, 'energy': 1, 'healthcare': 1, 'retail': 2}, 'freq_buckets': {'daily_or_coarser': 6}, 'n': 6, 'total_track_u_forecasts': 43493}

| task                   | domain     | freq_bucket      | dataset_family   |   horizon |   n_series |   horizon_to_context_ratio |   n_forecasts_track_u | selected   |
|:-----------------------|:-----------|:-----------------|:-----------------|----------:|-----------:|---------------------------:|----------------------:|:-----------|
| hospital               | healthcare | daily_or_coarser | hospital         |        12 |        767 |                     0.3333 |                  3068 | True       |
| hospital_admissions_1W | healthcare | daily_or_coarser | hospital         |        13 |          8 |                     0.3421 |                   128 | False      |
| m5_1M                  | retail     | daily_or_coarser | m5               |        12 |      30490 |                     0.25   |                 30490 | True       |
| restaurant             | retail     | daily_or_coarser | restaurant       |        28 |        817 |                     0.3889 |                  6536 | True       |
| rossmann_1W            | retail     | daily_or_coarser | rossmann         |        13 |       1115 |                     0.4483 |                  8920 | False      |
| solar_1D               | energy     | daily_or_coarser | solar            |        28 |        137 |                     0.3294 |                  1370 | True       |
| solar_1W               | energy     | daily_or_coarser | solar            |        13 |        137 |                     0.3333 |                   137 | False      |
| uk_covid_utla_1W       | healthcare | daily_or_coarser | uk               |        13 |        214 |                     0.3333 |                  1070 | False      |
| us_consumption_1Y      | econ       | daily_or_coarser | us               |         5 |         31 |                     0.3571 |                   310 | True       |
| walmart                | retail     | daily_or_coarser | walmart          |        39 |       2936 |                     0.375  |                  2936 | False      |
| world_co2_emissions    | econ       | daily_or_coarser | world            |         5 |        191 |                     0.3333 |                  1719 | True       |
| world_life_expectancy  | econ       | daily_or_coarser | world            |         5 |        237 |                     0.2083 |                  2370 | False      |
| world_tourism          | econ       | daily_or_coarser | world            |         5 |        178 |                     0.4545 |                   356 | False      |

Every selected task is `daily_or_coarser`. That is a property of the condition rather than a choice: a high horizon-to-context ratio means a short visible history, and among the eligible tasks none of the sub-hourly or hourly ones reach the long cut. The stratification tried for frequency spread and the data did not offer it.

## 6. TSFM revisions and contamination

| model | revision | fev-bench contamination |
|---|---|---|
| `chronos-2` | `29ec3766d36d6f73f0696f85560a422f50e8498c` | OVERLAP_RISK_UNKNOWN |
| `tirex-2` | `05e5b26db52bfb256f1ae1bdf785589850482de3` | OVERLAP_RISK_UNKNOWN |
| `timesfm-3.0` | `43046b85ec22d584a13f8098c2ed39c889e129c2` | CLEAN_BY_OFFICIAL_EXCLUSION |

These are the source study's checkpoints, asserted against the live hub before the holdout ran. Two of the three carry unresolved overlap risk, which matters for reading the TSFM side of any comparison here.

## 7. Specialist suite

- environment: `.venv-tsfm-specialist`, AutoGluon 1.6.1, torch 2.11.0+cu128, CUDA True
- models requested and available: 5/5

| requested     | actual_class       | family               | available   |
|:--------------|:-------------------|:---------------------|:------------|
| PatchTST      | PatchTSTModel      | patch transformer    | True        |
| TiDE          | TiDEModel          | dense encoder        | True        |
| DLinear       | DLinearModel       | linear decomposition | True        |
| DeepAR        | DeepARModel        | autoregressive RNN   | True        |
| DirectTabular | DirectTabularModel | tabular regression   | True        |

AutoGluon ships Chronos, Chronos-2 and Toto wrappers and installing it pulled `chronos-forecasting` in as a dependency. None is in the suite: the pool is given as an explicit hyperparameter dict, and `verify` check A18 reads the fitted model list back.

## 8. Training chronology and the no-refit contract

```
first evaluation cutoff
        |
  training history  ->  fit once per (task, seed)  ->  parameters frozen
        |
  window 0 forecast
  window 1  newer observed history is input context, no refit
  window k  ...
```

`fit` is called once per (task, seed) and every later forecast goes through `predict(context)`. `refit_full` is never called. Checks A07 to A10 test the timestamps and the call path rather than taking the diagram's word for it.

| task                |       seed |   num_val_windows |   fit_seconds | model_best    | ensemble_model   |
|:--------------------|-----------:|------------------:|--------------:|:--------------|:-----------------|
| ETT_1D              | 2026090701 |                 2 |         290.3 | TiDE          | WeightedEnsemble |
| ETT_1D              | 2026090702 |                 2 |         219.4 | DeepAR        | WeightedEnsemble |
| LOOP_SEATTLE_1D     | 2026090701 |                 1 |         138.9 | DeepAR        | WeightedEnsemble |
| LOOP_SEATTLE_1D     | 2026090702 |                 1 |         115.9 | DirectTabular | WeightedEnsemble |
| australian_tourism  | 2026090701 |                 1 |          87.2 | DeepAR        | WeightedEnsemble |
| australian_tourism  | 2026090702 |                 1 |          90   | DirectTabular | WeightedEnsemble |
| hermes              | 2026090701 |                 2 |         872.5 | DirectTabular | WeightedEnsemble |
| hermes              | 2026090702 |                 2 |         865   | DirectTabular | WeightedEnsemble |
| hospital            | 2026090701 |               nan |         169.4 | DeepAR        | WeightedEnsemble |
| hospital            | 2026090702 |               nan |         115.4 | PatchTST      | WeightedEnsemble |
| m5_1M               | 2026090701 |               nan |         289.3 | PatchTST      | WeightedEnsemble |
| m5_1M               | 2026090702 |               nan |         262   | PatchTST      | WeightedEnsemble |
| restaurant          | 2026090701 |                 1 |         371.9 | TiDE          | WeightedEnsemble |
| restaurant          | 2026090702 |                 1 |         386.9 | TiDE          | WeightedEnsemble |
| solar_1D            | 2026090701 |               nan |          97.5 | DeepAR        | WeightedEnsemble |
| solar_1D            | 2026090702 |               nan |         102.2 | TiDE          | WeightedEnsemble |
| us_consumption_1Y   | 2026090701 |               nan |         113.6 | PatchTST      | WeightedEnsemble |
| us_consumption_1Y   | 2026090702 |               nan |          88.7 | DeepAR        | WeightedEnsemble |
| world_co2_emissions | 2026090701 |               nan |         123.6 | PatchTST      | WeightedEnsemble |
| world_co2_emissions | 2026090702 |               nan |         132.6 | PatchTST      | WeightedEnsemble |

Fit failures: 0 of 20 (task, seed) runs.

## 9. Specialist strength sanity

- median RI over the source study's linear autoregression: 0.54% (5 wins of 10)
- median RI over SeasonalNaive: 4.97% (7 wins of 10)
- **gate: PASS**

## 10. Development raw results

| task               |   timesfm-3.0 |   chronos-2 |   tirex-2 |   F_FAMILY_ENVELOPE |   PatchTST |   S_BEST |   S_ENSEMBLE |   LinearAR |   SeasonalNaive |
|:-------------------|--------------:|------------:|----------:|--------------------:|-----------:|---------:|-------------:|-----------:|----------------:|
| ETT_1D             |        1.1217 |      1.109  |    1.1227 |              1.109  |     1.8294 |   2.7006 |       2.4823 |     1.3603 |          1.3728 |
| LOOP_SEATTLE_1D    |        0.7587 |      0.7754 |    0.777  |              0.7587 |     1.2893 |   1.1382 |       1.0028 |     0.9954 |          1.0462 |
| australian_tourism |        0.7906 |      0.7396 |    0.7437 |              0.7396 |     1.1169 |   0.922  |       0.8785 |     0.8947 |          0.8947 |
| hermes             |        0.6088 |      0.6216 |    0.6489 |              0.6088 |     0.6621 |   0.6895 |       0.6649 |     0.9746 |          2.1461 |

## 11. Fresh holdout raw results

| task                |   timesfm-3.0 |   chronos-2 |   tirex-2 |   F_FAMILY_ENVELOPE |   PatchTST |   S_BEST |   S_ENSEMBLE |   LinearAR |   SeasonalNaive |
|:--------------------|--------------:|------------:|----------:|--------------------:|-----------:|---------:|-------------:|-----------:|----------------:|
| hospital            |        0.6825 |      0.7014 |    0.6841 |              0.6825 |     0.9441 |   0.8558 |       0.7608 |     0.8642 |          0.8075 |
| m5_1M               |        0.9636 |      0.9802 |    0.9752 |              0.9636 |     0.9923 |   0.9923 |       0.9995 |     1.1444 |          1.1399 |
| restaurant          |        0.7001 |      0.6858 |    0.6796 |              0.6796 |     0.7092 |   0.7038 |       0.6996 |     0.7784 |          0.9838 |
| solar_1D            |        0.6037 |      0.5975 |    0.6043 |              0.5975 |     1.0493 |   1.1762 |       1.1081 |     0.7192 |          1.4068 |
| us_consumption_1Y   |        3.6859 |      3.9206 |    3.4402 |              3.4402 |    16.4908 |  11.9416 |       7.7524 |     5.8877 |          6.5846 |
| world_co2_emissions |        2.7799 |      2.7523 |    2.6396 |              2.6396 |     5.3718 |   5.3718 |       5.5335 |     3.1013 |          3.155  |

## 12. Effect tables

### Table C — per-task relative improvement, positive means the specialist wins

| split       | task                |   RI_vs_envelope_pct |   RI_vs_fixed_pct |   RI_vs_linear_ar_pct |   RI_vs_naive_pct |
|:------------|:--------------------|---------------------:|------------------:|----------------------:|------------------:|
| development | ETT_1D              |              -123.83 |           -121.3  |                -82.48 |            -80.81 |
| development | LOOP_SEATTLE_1D     |               -32.17 |            -32.17 |                 -0.74 |              4.15 |
| development | australian_tourism  |               -18.77 |            -11.11 |                  1.82 |              1.82 |
| development | hermes              |                -9.21 |             -9.21 |                 31.78 |             69.02 |
| holdout     | hospital            |               -11.47 |            -11.47 |                 11.97 |              5.79 |
| holdout     | m5_1M               |                -3.73 |             -3.73 |                 12.66 |             12.31 |
| holdout     | restaurant          |                -2.95 |              0.07 |                 10.12 |             28.89 |
| holdout     | solar_1D            |               -85.47 |            -83.54 |                -54.08 |             21.23 |
| holdout     | us_consumption_1Y   |              -125.34 |           -110.33 |                -31.67 |            -17.73 |
| holdout     | world_co2_emissions |              -109.63 |            -99.05 |                -78.42 |            -75.39 |

### Table D — aggregates

| split       | label      |   n_usable |   median_RI_vs_envelope_pct |   mean_RI_vs_envelope_pct |   median_RI_vs_fixed_pct |   wins_vs_envelope |
|:------------|:-----------|-----------:|----------------------------:|--------------------------:|-------------------------:|-------------------:|
| development | S_BEST     |          4 |                      -37.34 |                    -57.86 |                   -33.32 |                  0 |
| development | S_ENSEMBLE |          4 |                      -25.47 |                    -46    |                   -21.64 |                  0 |
| holdout     | S_BEST     |          6 |                      -61.13 |                    -79.9  |                   -59.31 |                  0 |
| holdout     | S_ENSEMBLE |          6 |                      -48.47 |                    -56.43 |                   -47.51 |                  0 |

### Table E — specialist strength

| split       | label      |   median_RI_vs_linear_ar_pct |   wins_vs_linear_ar |   median_RI_vs_naive_pct |   wins_vs_naive |
|:------------|:-----------|-----------------------------:|--------------------:|-------------------------:|----------------:|
| development | S_BEST     |                        -8.7  |                   1 |                    -5.92 |               1 |
| development | S_ENSEMBLE |                         0.54 |                   2 |                     2.98 |               3 |
| holdout     | S_BEST     |                       -31.28 |                   3 |                     3.48 |               3 |
| holdout     | S_ENSEMBLE |                       -10.78 |                   3 |                     9.05 |               4 |

## 13. Seed stability

| split       | label      |   median_RI_vs_envelope_seed_2026090701_pct |   median_RI_vs_envelope_seed_2026090702_pct |
|:------------|:-----------|--------------------------------------------:|--------------------------------------------:|
| development | S_BEST     |                                      -37.43 |                                      -37.25 |
| development | S_ENSEMBLE |                                      -23.02 |                                      -27.92 |
| holdout     | S_BEST     |                                      -55.62 |                                      -64.63 |
| holdout     | S_ENSEMBLE |                                      -33.88 |                                      -49.69 |

A conclusion that flips with the seed is not a conclusion. Both seeds are reported and the gates require them to agree in sign.

## 14. Final gate

**development**

- FAIL — A_median_vs_envelope_ge_8
- FAIL — B_median_vs_fixed_ge_8
- FAIL — C_positive_on_at_least_3_of_4
- FAIL — D_both_seed_medians_positive
- PASS — E_specialist_strength_plausible

signal: **False**

**holdout**

- FAIL — A_median_vs_envelope_ge_5
- FAIL — B_median_vs_fixed_ge_5
- FAIL — C_positive_on_at_least_4
- FAIL — D_both_seed_medians_positive
- PASS — E_usable_at_least_5

signal: **False**

### A tension inside the contract, stated rather than resolved quietly

Section 17 passes the strength gate on either the linear-AR criterion or the SeasonalNaive one, and only the SeasonalNaive one is met: the suite wins 7/10 against the naive anchor but only 5/10 against the source study's linear ridge autoregression, at a median of 0.54%. Section 22.3 lists exactly that - a suite that does not stably beat linear AR - as an example of INCONCLUSIVE_SPECIALIST_POWER.

Section 17 is the operational gate and names its own consequence, so it governs the token. Section 22.3 is a list of examples.

Does it change what to do next? No. Both readings say the same thing about what to do next: do not start method work on long-horizon now. They differ only in whether the question is closed or left open, and the safer of the two is stated in section 17 of STATUS.md.

## 15. Integrity and deviations

21/22 checks pass. Detail in `SELF_AUDIT.md`.

| check   | result   | description                                                               |
|:--------|:---------|:--------------------------------------------------------------------------|
| A01     | PASS     | source branch SHA pinned                                                  |
| A02     | PASS     | source D1-long tasks exactly reproduced                                   |
| A03     | PASS     | fresh holdout excludes the previous 18 tasks                              |
| A04     | PASS     | fresh holdout excludes the previous dataset families                      |
| A05     | PASS     | holdout hash frozen before specialist training                            |
| A06     | PASS     | Track U only                                                              |
| A11     | PASS     | no future covariates in Track U                                           |
| A07     | PASS     | specialist training data ends at or before the first evaluation cutoff    |
| A08     | PASS     | no evaluation label used in fitting                                       |
| A09     | PASS     | predictor parameters not refit after the first cutoff                     |
| A10     | PASS     | later observed target used only as inference context                      |
| A12     | PASS     | quantile levels match fev                                                 |
| A13     | PASS     | native SQL evaluator used                                                 |
| A14     | PASS     | development TSFM rows match the source benchmark_results exactly          |
| A15     | PASS     | holdout TSFM revisions match the source model_audit                       |
| A16     | PASS     | F_FAMILY_ENVELOPE is the taskwise minimum of the three primary TSFMs only |
| A17     | PASS     | S_BEST selected from training-only validation                             |
| A18     | PASS     | S_ENSEMBLE contains no pretrained foundation model                        |
| A19     | PASS     | same evaluation windows for every comparator                              |
| A20     | PASS     | old study files unchanged                                                 |
| A21     | PASS     | origin/main unchanged                                                     |
| A22     | FAIL     | final remote branch tip == local HEAD                                     |

### Deviations

- Dataset arrow files are cached under `runs/dscache` rather than beside the model weights. `datasets` names its lock files after the full cache path, and the source study's deep cache directory pushed them past Windows' 260-character limit, which made one holdout candidate impossible to load.
- The specialist environment is a second venv. Installing AutoGluon into the TSFM environment would have moved pandas, numpy and torch under three models this closure must keep byte-identical to the source study.

## 16. What was NOT tested

- Why the specialist and the foundation models land where they do. This study measures existence, not cause.
- Any track other than U. Covariates and native multivariate are out of scope by Section 5.
- Fine-tuning a foundation model, or any new method. Section 35 forbids it here.
- Frequencies other than daily-or-coarser, because no eligible sub-hourly or hourly task meets the frozen long-horizon definition.
- A larger training budget for the specialist: the cap is fixed per task and seed by Section 14 and was applied identically everywhere.

## 17. Next action

Do not make long-horizon a method topic. A task-trained specialist suite, given each task's own history and a fixed training budget, did not expose the registered 5-8% recoverable gap on either split — it lost to the foundation models on all ten tasks.

Two cautions on how far that carries. First, Section 23: this is an investment screen, not an equivalence claim. The exact statement the evidence supports is that a stronger supervised specialist did not expose the registered gap, not that none exists. Second, the label undersells what was measured. 'Shared difficulty' suggests every model struggles equally; here the foundation models were clearly better, by 25% on the development split and 48% on the holdout. The long-horizon condition is hard for a model that must learn it from 14 to 209 observations, and the foundation models' pretraining is exactly what covers that.

If the question is reopened later, the thing to change is the comparator's training signal, not its architecture: every task here gives a specialist a very short history, and a suite that only ties a linear ridge autoregression under that constraint cannot bound what task specialisation could achieve with more.
