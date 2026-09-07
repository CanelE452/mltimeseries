# TSFM-LONG-HORIZON-SPECIALIST-CLOSURE-v1

A closure, not a method. The previous study found that on long-horizon fev-bench
tasks all three foundation model families lose ground together — a median
condition gap of 38.8% over four tasks — but the effect failed its registered
gate on regret alone: no other evaluated estimator did better there. That leaves
one question open, and this branch answers only that question.

> On long-horizon tasks, does a genuinely strong task-trained non-foundation
> forecasting system beat the same-information zero-shot TSFMs, repeatedly?

If it does, long-horizon is a real recoverable gap and worth a method. If it does
not, the deficit is task difficulty every model shares and the topic is dropped.
If the specialist cannot even be shown to be strong, neither conclusion is
available and the honest answer is `INCONCLUSIVE_SPECIALIST_POWER`.

## What is compared

| | |
|---|---|
| Development | the source study's four D1-long discovery tasks, re-derived from its `task_metadata.csv` |
| Fresh holdout | six long-horizon tasks that share no task and no dataset family with the previous eighteen |
| TSFMs | `chronos-2`, `tirex-2`, `timesfm-3.0` at the source study's pinned revisions |
| Specialists | AutoGluon PatchTST, TiDE, DLinear, DeepAR, DirectTabular, fitted per task |
| Track | U only — one univariate series at a time, no covariates |
| Metric | fev native SQL, computed by `fev.Task.evaluation_summary` |

Three TSFM comparators per task: `F_FIXED` (TimesFM 3.0), `F_FAMILY_ENVELOPE`
(the taskwise minimum of the three, chosen with the scores in view and therefore
deliberately hard to beat), and `F_MEAN` (descriptive only).

## The contract that makes this fair

```
first evaluation cutoff
        |
  training history  ->  fit once per (task, seed)  ->  parameters frozen
        |
  window 0 forecast
  window 1  newer observed history is input context, no refit
  window k  ...
```

The specialist is fitted once, on data strictly before the first evaluation
cutoff, and then frozen. Later windows' newly observed values reach it as
inference context and never as training signal. `tests/` asserts the timestamps
and parses the training module to confirm `fit` is called exactly once and
`refit_full` never.

The two training regimes are deliberately different — zero-shot versus
task-trained — because that asymmetry *is* the question. `fairness_matrix.csv`
states it row by row so the two cannot be confused.

## Two environments

`.venv-tsfm` holds the three foundation models on the source study's pinned
versions and is not modified. `.venv-tsfm-specialist` holds AutoGluon, which pins
pandas, numpy and torch ranges of its own. Specialists produce forecasts there;
the metric is always computed in the first environment, so the two never share a
scoring path.

## Running it

```bash
V=.venv-tsfm/Scripts/python.exe
S=.venv-tsfm-specialist/Scripts/python.exe
M=experiments.tsfm_long_horizon_specialist_closure_v1.src.run

$V -m $M audit-source
$V -m $M select-holdout          # freezes and hashes the task list
$V -m $M convert
$S -m $M audit-specialists
$S -m $M train-development
$V -m $M run-holdout-tsfms
$S -m $M train-holdout
$V -m $M evaluate --pattern "S_*.npz" --out specialist_scores.csv
$V -m $M compare
$V -m $M fairness
$V -m $M verdict
$V -m $M verify
$V -m $M report
$V -m $M publication-scan
# commit, push, then
$V -m $M publication-verify
```

Each stage refuses to run under the wrong interpreter rather than failing halfway
through an import.

## Environment notes

Dataset arrow files are cached in `runs/dscache`. `datasets` names its lock files
after the full cache path, and the source study's deeper cache directory pushed
them past Windows' 260-character limit, which made one holdout candidate
impossible to load at all.

## Tests

```bash
.venv-tsfm/Scripts/python.exe -m pytest experiments/tsfm_long_horizon_specialist_closure_v1/tests -q
```

They cover the parts whose failure would be silent: that the development set is
re-derived rather than copied (and that the assertion fires when it should), that
the holdout excludes the previous tasks and families, that the frozen D1
threshold is reused rather than recomputed, that no evaluation row reaches the
training frame, that the row order maps forecasts back to the right target
column, and that the training module cannot refit.
