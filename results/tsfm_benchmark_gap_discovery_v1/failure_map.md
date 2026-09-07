# Failure map - TRACK U, discovery split

Numbers first. `relative_to_naive` is the task's native SQL divided by the
SeasonalNaive SQL on the same task, so 1.0 is the naive anchor and lower is better.
`condition_gap_pct` compares a model's median inside the bucket with its median on
every discovery task outside the bucket; positive means worse inside.

## Candidate gate

A condition is promoted only with at least 3 discovery tasks, at least 2 affected architecture families, a median condition gap of at least 5.0%, and a median regret against the best deployable model of at least 8.0%.

| descriptor   | descriptor_label              | bucket           |   n_tasks_in_bucket |   n_affected_models |   n_affected_families |   median_condition_gap_pct |   median_regret_pct | gate_min_tasks   | gate_min_families   | gate_min_gap   | gate_min_regret   | passes_failure_gate   | verdict         |
|:-------------|:------------------------------|:-----------------|--------------------:|--------------------:|----------------------:|---------------------------:|--------------------:|:-----------------|:--------------------|:---------------|:------------------|:----------------------|:----------------|
| D4           | future covariates             | none             |                   9 |                   3 |                     3 |                   55.6831  |            2.204    | True             | True                | True           | False             | False                 | NOT_A_CANDIDATE |
| D1           | horizon ratio                 | long             |                   4 |                   3 |                     3 |                   38.8115  |            1.58619  | True             | True                | True           | False             | False                 | NOT_A_CANDIDATE |
| D9           | seasonal strength             | weak             |                   7 |                   3 |                     3 |                   38.0734  |            1.19063  | True             | True                | True           | False             | False                 | NOT_A_CANDIDATE |
| D2           | frequency                     | daily_or_coarser |                   5 |                   3 |                     3 |                   32.4923  |            2.08847  | True             | True                | True           | False             | False                 | NOT_A_CANDIDATE |
| D6           | zero fraction / intermittency | lt0.1            |                  10 |                   3 |                     3 |                   23.9961  |            2.02489  | True             | True                | True           | False             | False                 | NOT_A_CANDIDATE |
| D8           | train-only distribution shift | mid              |                   4 |                   3 |                     3 |                   23.8143  |            2.02489  | True             | True                | True           | False             | False                 | NOT_A_CANDIDATE |
| D1           | horizon ratio                 | medium           |                   4 |                   3 |                     3 |                   22.0177  |            0.99696  | True             | True                | True           | False             | False                 | NOT_A_CANDIDATE |
| D3           | target dimensionality         | 2_16             |                   3 |                   3 |                     3 |                   15.2996  |            1.23959  | True             | True                | True           | False             | False                 | NOT_A_CANDIDATE |
| D8           | train-only distribution shift | high             |                   4 |                   3 |                     3 |                    8.25329 |            1.24262  | True             | True                | True           | False             | False                 | NOT_A_CANDIDATE |
| D5           | past covariates               | none             |                  11 |                   1 |                     1 |                    5.95795 |            2.204    | True             | False               | True           | False             | False                 | NOT_A_CANDIDATE |
| D7           | missingness                   | high             |                   2 |                   3 |                     3 |                    5.7116  |            1.65845  | False            | True                | True           | False             | False                 | CASE_ONLY       |
| D1           | horizon ratio                 | short            |                   4 |                   0 |                     0 |                  nan       |           12.1275   | True             | False               | False          | True              | False                 | NOT_A_CANDIDATE |
| D2           | frequency                     | hourly           |                   4 |                   0 |                     0 |                  nan       |           11.7051   | True             | False               | False          | True              | False                 | NOT_A_CANDIDATE |
| D2           | frequency                     | sub_hourly       |                   3 |                   0 |                     0 |                  nan       |            1.03371  | True             | False               | False          | False             | False                 | NOT_A_CANDIDATE |
| D3           | target dimensionality         | 1                |                   8 |                   0 |                     0 |                  nan       |            2.50047  | True             | False               | False          | False             | False                 | NOT_A_CANDIDATE |
| D3           | target dimensionality         | gt16             |                   1 |                   0 |                     0 |                  nan       |            0.960206 | False            | False               | False          | False             | False                 | CASE_ONLY       |
| D4           | future covariates             | available        |                   3 |                   0 |                     0 |                  nan       |            6.57261  | True             | False               | False          | False             | False                 | NOT_A_CANDIDATE |
| D5           | past covariates               | available        |                   1 |                   0 |                     0 |                  nan       |            2.51444  | False            | False               | False          | False             | False                 | CASE_ONLY       |
| D6           | zero fraction / intermittency | 0.1_0.5          |                   2 |                   0 |                     0 |                  nan       |            6.36598  | False            | False               | False          | False             | False                 | CASE_ONLY       |
| D7           | missingness                   | low              |                  10 |                   0 |                     0 |                  nan       |            2.14624  | True             | False               | False          | False             | False                 | NOT_A_CANDIDATE |
| D8           | train-only distribution shift | low              |                   4 |                   0 |                     0 |                  nan       |            6.37995  | True             | False               | False          | False             | False                 | NOT_A_CANDIDATE |
| D9           | seasonal strength             | moderate         |                   3 |                   0 |                     0 |                  nan       |           15.3678   | True             | False               | False          | True              | False                 | NOT_A_CANDIDATE |

## Per-bucket detail

Bucket sizes here are three to eight tasks. Section 26 asks for the raw task effects at
that size rather than an interval built from them, so every per-task number is listed
below the bucket medians.

### D1 - horizon ratio

|               |   chronos-2 |   chronos-2-synth |   timesfm-3.0 |   tirex-2 |
|:--------------|------------:|------------------:|--------------:|----------:|
| ('long', 4)   |       0.775 |             0.786 |         0.771 |     0.78  |
| ('medium', 4) |       0.642 |             0.659 |         0.598 |     0.608 |
| ('short', 4)  |       0.431 |             0.46  |         0.372 |     0.458 |

### D2 - frequency

|                         |   chronos-2 |   chronos-2-synth |   timesfm-3.0 |   tirex-2 |
|:------------------------|------------:|------------------:|--------------:|----------:|
| ('daily_or_coarser', 5) |       0.741 |             0.753 |         0.725 |     0.743 |
| ('hourly', 4)           |       0.431 |             0.46  |         0.372 |     0.458 |
| ('sub_hourly', 3)       |       0.595 |             0.606 |         0.558 |     0.564 |

### D3 - target dimensionality

|             |   chronos-2 |   chronos-2-synth |   timesfm-3.0 |   tirex-2 |
|:------------|------------:|------------------:|--------------:|----------:|
| ('1', 8)    |       0.613 |             0.623 |         0.557 |     0.571 |
| ('2_16', 3) |       0.695 |             0.742 |         0.644 |     0.649 |
| ('gt16', 1) |       0.565 |             0.573 |         0.558 |     0.564 |

### D4 - future covariates

|                  |   chronos-2 |   chronos-2-synth |   timesfm-3.0 |   tirex-2 |
|:-----------------|------------:|------------------:|--------------:|----------:|
| ('available', 3) |       0.405 |             0.448 |         0.33  |     0.451 |
| ('none', 9)      |       0.631 |             0.641 |         0.568 |     0.582 |

### D5 - past covariates

|                  |   chronos-2 |   chronos-2-synth |   timesfm-3.0 |   tirex-2 |
|:-----------------|------------:|------------------:|--------------:|----------:|
| ('available', 1) |       0.595 |             0.606 |         0.547 |     0.561 |
| ('none', 11)     |       0.631 |             0.641 |         0.568 |     0.582 |

### D6 - zero fraction / intermittency

|                |   chronos-2 |   chronos-2-synth |   timesfm-3.0 |   tirex-2 |
|:---------------|------------:|------------------:|--------------:|----------:|
| ('0.1_0.5', 2) |       0.513 |             0.524 |         0.458 |     0.483 |
| ('lt0.1', 10)  |       0.613 |             0.623 |         0.593 |     0.599 |

### D7 - missingness

|             |   chronos-2 |   chronos-2-synth |   timesfm-3.0 |   tirex-2 |
|:------------|------------:|------------------:|--------------:|----------:|
| ('high', 2) |       0.645 |             0.674 |         0.595 |     0.605 |
| ('low', 10) |       0.598 |             0.607 |         0.563 |     0.573 |

### D8 - train-only distribution shift

|             |   chronos-2 |   chronos-2-synth |   timesfm-3.0 |   tirex-2 |
|:------------|------------:|------------------:|--------------:|----------:|
| ('high', 4) |       0.642 |             0.659 |         0.598 |     0.608 |
| ('low', 4)  |       0.511 |             0.522 |         0.471 |     0.513 |
| ('mid', 4)  |       0.718 |             0.748 |         0.684 |     0.696 |

### D9 - seasonal strength

|                 |   chronos-2 |   chronos-2-synth |   timesfm-3.0 |   tirex-2 |
|:----------------|------------:|------------------:|--------------:|----------:|
| ('moderate', 3) |       0.457 |             0.471 |         0.396 |     0.466 |
| ('weak', 7)     |       0.631 |             0.641 |         0.628 |     0.635 |
