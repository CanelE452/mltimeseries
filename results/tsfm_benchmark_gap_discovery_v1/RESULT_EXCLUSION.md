# Result exclusions

Section 7 allows a result to be dropped only for a reason that applies equally to
every model — an out-of-memory failure, a bug, corrupt data — and never because
of what the number turned out to be.

## Exclusions applied

None. Every (model, track, task) cell that was attempted completed, and no row
was removed from `benchmark_results.csv` after the fact. `verify` check A15 reads
the same file and fails if any row carries a non-OK status that is not listed
here.

## Cells never attempted, and why

These are absences by contract rather than exclusions: a track only opens for the
tasks whose metadata admits it.

- TRACK M runs only on tasks with more than one target column. Running it on a
  single-target task would be TRACK U under a different name.
- TRACK C runs only on tasks that declare `known_dynamic_columns`. Supplying a
  covariate that the benchmark does not mark as known-in-advance would break the
  information condition.
- Within TRACK C, categorical known-future columns are passed over rather than
  encoded. An encoding chosen here would be a modelling decision made by the
  study rather than by the model, and Section 10 forbids per-model tuning. The
  columns skipped for this reason are recorded in `fairness_matrix.csv`.

The set of tasks each track opens on is fixed by `task_pool.csv` and was known
before any model ran.
