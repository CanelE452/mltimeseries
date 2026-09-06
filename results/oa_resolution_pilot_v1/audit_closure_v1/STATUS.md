OA-RESOLUTION-PILOT-v1-AUDIT-CLOSURE-v1

ORIGINAL SCIENTIFIC DECISION:
INCONCLUSIVE

CURRENT IMPLEMENTATION RECOMMENDATION:
STOP_SCALING_CURRENT_INTERVAL_INTEGRATED_FOURIER_O

BROADER OBSERVATION-AWARE TOPIC:
OPEN_NOT_DIRECTLY_TESTED

---

## 1. Audit scope

Re-check the finished pilot: recompute the headline arithmetic independently,
reproduce the confidence interval from the raw per-key errors, establish what is
actually in the repository, and correct two semantic errors and three overstated
interpretations. The question this feeds is narrow — should the current
interval-integrated Fourier representation be carried into a pretrained model?

## 2. What was not rerun

Zero model fits. No optimizer was constructed, no backward pass ran, no checkpoint
was reselected, no lambda was rescreened, no seed was added, no dataset or operator
or architecture was introduced, no threshold was created or moved, and FlowState was
not re-invoked. `audit_verdict.json` records `core_model_fits_in_audit: 0`, and a
test asserts the audit script contains no training entry point and does not import
torch.

The audit also does not import `experiments/oa_resolution_pilot_v1/report.py`. Every
number below was recomputed by code written a second time in
`scripts/audit_oa_resolution_pilot_v1.py`.

## 3. Git and artifact provenance

| item | value |
|---|---|
| audited base | `a4956c502795125ab6173a706fab28ef6126bfd9` (`origin/oa-resolution-pilot-v1`) |
| audit branch | `oa-resolution-pilot-v1-audit-closure-v1` |
| `origin/main` | `36b01d2b84df16f03447a8e214723fc61ae147fd`, untouched |
| original artifacts | 21 files hashed before and after; none changed |

The historical results under `results/oa_resolution_pilot_v1/` are left exactly as
the run produced them. Everything corrected here lives in `audit_closure_v1/`.

## 4. Original scientific decision

`INCONCLUSIVE`, unchanged. This audit issues an engineering recommendation, not a
new scientific token. `NO_GO`, `GO`, `SCREEN_NEGATIVE` and `PROMISING` do not appear
as verdicts anywhere in it.

## 5. Arithmetic reproduction

Every macro was recomputed from `metrics.csv` and compared against
`primary_contrasts.json`.

| | |
|---|---|
| status | `ARITHMETIC_REPRODUCTION_OK` |
| largest disagreement | 3.06e-14 percentage points |
| tolerance | 1e-10 percentage points |

## 6. Raw-error reproducibility status

`LOCAL_12_RAW_ERRORS_PRESENT_GIT_UNTRACKED`.

Twelve `errors_<dataset>_<arm>_<seed>.npy` arrays exist locally under
`runs/oa_resolution_pilot_v1/errors/`. `runs/` is gitignored (`.gitignore:18`), so
none of them is in any commit. An earlier report said they were "present in the
commit"; that was wrong and is corrected here and in the history.

Their SHA256s are recorded in `raw_error_inventory.json`. Rather than force-adding
tens of megabytes of arrays, the audit commits
`bootstrap_block_sufficient_stats.csv` — per (dataset, 7-day block, operation, r,
arm) the sum of the seed-averaged primary loss and the number of logical keys. That
file alone regenerates the intervals below.

## 7. Bootstrap revalidation

Recomputed from the raw errors through the block sufficient statistics, with an
independent implementation.

| contrast, unseen interpolation | recomputed | stored |
|---|---|---|
| O vs M | -0.0544 % [-0.2793, +0.1746] | -0.0544 % [-0.2793, +0.1746] |
| O vs R | +0.5318 % [-0.7882, +1.7357] | +0.5318 % [-0.7882, +1.7357] |
| M vs R | +0.5877 % [-0.6210, +1.7197] | +0.5877 % [-0.6210, +1.7197] |

Largest disagreement 1.25e-14 percentage points; `BOOTSTRAP_REPRODUCED`.

**Naming correction.** The original docstring called this a moving-block bootstrap.
The implementation partitions origins into fixed, non-overlapping seven-day blocks
via `origin // (7 * 144)` and draws whole blocks with replacement. The accurate name
is a **paired 7-day time-block (cluster) bootstrap**. The historical file keeps its
wording; this is the correction of record.

## 8. Primary O vs M

| | |
|---|---|
| unseen interpolation macro (r = 3, 6) | -0.049 % |
| interval | [-0.279, +0.175], contains zero |
| per dataset | jena -0.139 %, uci +0.042 % |
| per seed | 2026090601 -0.300 %, 2026090602 +0.202 % |
| seen macro (r = 2, 4, 8) | -0.182 % |

Both the dataset split and the seed split change sign. The pre-registered method GO
was not met.

## 9. M/O fairness

| check | result |
|---|---|
| trainable parameters | 411,585 = 411,585 |
| initial state under one seed | byte-identical across every tensor |
| training schedule SHA per (dataset, seed) | one value across R, M and O |
| the differing input | `time_basis`, a numpy array with no trainable parameter |
| width, operation id, support bounds | identical arrays for M and O at every (operation, r) |
| checkpoint chosen | identical `best_update` for M and O in all four dataset x seed pairs |

The last row matters for reading the null: M and O did not stop at different points,
so the flat result is not a checkpoint-selection artifact.

## 10. Loss components

The near-zero primary contrast is not a cancellation. At unseen interpolation:

| component | O vs M | O vs R | M vs R |
|---|---|---|---|
| primary | -0.049 % | +0.574 % | +0.624 % |
| MSE 10-minute | -0.050 % | +0.379 % | +0.431 % |
| MSE 60-minute | -0.045 % | +0.913 % | +0.960 % |

Both horizons are independently near zero for O vs M.

## 11. Training and checkpoint state

Every core fit selected an interior checkpoint — never the first and never the last
validation point. Final validation loss sits 0.27 % to 1.73 % above the selected
best. That is an observation about the stored curves. It is not a claim that the
fits converged, that a global optimum was reached, or that 5,000 updates were
enough.

## 12. R comparison and its limits

R is the best arm on Jena at every resolution role and the worst on UCI at every
role. Pooled M vs R spans zero.

M and R differ in more than metadata. R reconstructs the coarse observations to 288
base bins and forecasts from that grid; M reads the coarse tokens directly. The
metadata contribution and the tokenisation-path contribution are not separated by
this pilot, so `M > R` on UCI is not evidence that "metadata alone is sufficient".

The R contrasts are also unstable across seeds, not only across datasets. At unseen
interpolation the per-seed macros are:

| seed | O vs M | O vs R | M vs R |
|---|---|---|---|
| 2026090601 | -0.300 % | -0.225 % | +0.088 % |
| 2026090602 | +0.202 % | +1.338 % | +1.135 % |

Every one of the three contrasts changes sign or size substantially between two
seeds. With two seeds this is not a sample to build an interval from, and no
standard error is computed from it; it is reported because it bounds how much weight
any single pooled R comparison can carry. Full per (dataset, operation, r, seed)
values are in `paired_seed_effects.csv`.

## 13. Reconstruction diagnostic and causal limits

`reconstruction_quality.json` covers **one channel per dataset and 200 test windows**
at the selected lambda — `temperature` for Jena, `global_active_power` for UCI.

Supported: the dataset-level reversal is consistent with that diagnostic; the tested
Jena channel was reconstructed far more accurately (RMSE 0.029-0.052 against a signal
std of 0.910) than the tested UCI channel (0.384-0.586 against 0.864).

Withdrawn: that reconstruction quality *caused* the reversal. No intervention on
reconstruction quality was run, and the arms differ in more than that one property.

## 14. r = 12 exploratory result

| dataset | operation | O vs M |
|---|---|---|
| jena | INTERVAL_MEAN | +7.44 % |
| jena | END_BIN | -0.48 % |
| uci | END_BIN | +0.05 % |
| uci | INTERVAL_MEAN | +0.03 % |
| macro over all four | +1.76 % |
| macro without the Jena INTERVAL_MEAN cell | **-0.13 %** |

Removing the single dominant cell flips the sign and leaves nothing. This is an
exploratory unseen-extrapolation observation in one dataset and one operation. It is
not evidence that the representation works at wide support in general, and R still
beats both M and O on that Jena cell.

## 15. Corrected r = 12 seed table

The r = 12 seed table printed in the original `STATUS.md` took its R column from the
**END_BIN** row while M and O came from INTERVAL_MEAN. Parsed back out of that file
and matched at the five decimals it printed:

| seed | printed R | correct INTERVAL_MEAN R | END_BIN R at same r and seed |
|---|---|---|---|
| 2026090601 | 0.14457 | 0.16108 | 0.14457 |
| 2026090602 | 0.14936 | 0.16376 | 0.14936 |

Corrected table, every value joined on the exact `(dataset, operation, r, seed, arm)`
key, Jena INTERVAL_MEAN r = 12:

| seed | M | O | R | O vs M | O vs R |
|---|---|---|---|---|---|
| 2026090601 | 0.17568 | 0.16782 | 0.16108 | +4.47 % | -4.19 % |
| 2026090602 | 0.18705 | 0.16794 | 0.16376 | +10.22 % | -2.55 % |

**Scope: `REPORTING_CELL_MAPPING_BUG_ONLY`.** The M and O columns were right, the O
vs M figures quoted in the narrative were right, and `metrics.csv` reproduces exactly
from the per-seed cells (largest disagreement 1.1e-16). The aggregate results, the
contrasts, the bootstrap and the verdict are unaffected. What the error did change is
the apparent size of R's lead in that table: the true gap is -4.19 % and -2.55 %, not
the -16.08 % and -12.44 % the mismatched values imply. R still wins that cell.

The corrected table is generated through `select_cell(dataset, operation, r, arm,
seed)`, which raises unless the key matches exactly one row, and a test re-derives
every value from the raw error arrays.

## 16. WRONG_SUPPORT diagnostic

What it shows: a trained O model's predictions change when its inputs are re-encoded
with the centre basis at inference, in some conditions — sensitivity to a
train/inference representation mismatch.

What it does not show: that the integrated basis is superior. The comparison is
between a model and a corrupted version of itself, not between two models each
trained on its own representation.

The direct counterexample is in the artifacts: Jena INTERVAL_MEAN at r = 8 has
+7.93 % wrong-support sensitivity and O vs M of **-0.56 %**. Representation
dependence and predictive superiority come apart in the same cell.

## 17. Width diagnostic

Supported: the explicit scalar width channel showed little sensitivity under the
WIDTH_MISMATCH diagnostic — at most 0.088 % across the individual INTERVAL_MEAN
cells (M 0.088 %, O 0.076 %).

The pilot's own STATUS.md put this bound at 0.016 %. That figure was the maximum
of the dataset- and seed-averaged cells, not of the individual ones, so it read as
a tighter bound than the diagnostic supports. The averaging is what shrank it; the
qualitative reading is unchanged, since 0.088 % is still negligible beside the
contrasts under test.

Withdrawn: "the model does not use interval width". Resolution is available to the
arms through the token count, the token spacing, the time basis and the operation
embedding, so a flat width channel does not establish that interval width is unused.
The END_BIN rows of that variant are uninformative by construction: the support is
one base bin at every r, so the injected width equals the true width.

## 18. FlowState semantic audit

**Native rate, END_BIN: `INVALID_FOR_CORE_HOURLY_MEAN_METRIC`.** Averaging END_BIN
forecasts cannot equal the hourly mean at any r, because only the last base bin of
each report interval is predicted and the other r-1 never enter the average. On base
bins [1, 2, 3, 4, 5, 6] whose hourly mean is 3.5, a perfect END_BIN forecaster gives
4.0 at r = 2, 4.5 at r = 3 and 6.0 at r = 6. The original run scored those rows
against the 60-minute target, which measured the operator mismatch rather than the
model. The stored numbers are kept and marked `HISTORICAL_NUMBER_NOT_COMPARABLE`.

**Native rate, INTERVAL_MEAN:** semantically valid where the coarse grid tiles the
hour, r in {2, 3, 6}. Unchanged.

**Resampled variant: `VALID_REFERENCE_WITH_LIMITS`**, with its information condition
stated more precisely than before —
`MODEL_WEIGHTS_ZERO_SHOT + TARGET_VALIDATION_TUNED_PREPROCESSOR`. The FlowState
weights never saw these series, but the reconstruction lambda feeding them was chosen
on target validation data, so the pipeline is not target-data-free and must not be
described as one.

`references.py` now carries `can_form_hourly_mean_from_native(operation, r)`, and
conditions that fail it are stored as `NOT_EVALUATED_SEMANTIC_TARGET_MISMATCH` rather
than scored. No new metric was invented for END_BIN.

## 19. Claims that remain valid

- M and O use the same architecture, the same trainable parameter count, the same
  initial state under one seed, and the same training schedule.
- r = 3 and r = 6 were absent from training and from checkpoint selection.
- The primary O vs M unseen-interpolation effect is approximately zero, with an
  interval containing zero.
- The sign is not consistent across the two datasets or the two seeds.
- O did not satisfy the pre-registered method GO.
- The R / M / O ordering differs between Jena and UCI.
- The r = 12 effect is concentrated in one Jena INTERVAL_MEAN cell.
- The current interval-integrated Fourier representation does not justify Phase-2
  scaling.
- The SUM = r x MEAN unit control holds across all twelve models: forming the
  discrete total by an independent sum over base bins and dividing by the report
  width reproduces the INTERVAL_MEAN observation exactly (difference 0.0) and the
  arms forecast identically from it (difference 0.0). SUM is a unit control here and
  was never scored as a performance arm.
- The committed `bootstrap_block_sufficient_stats.csv` regenerates every interval in
  section 7 on its own, with no raw error arrays present
  (`python scripts/audit_oa_resolution_pilot_v1.py --from-stats-only`).

## 20. Claims weakened or withdrawn

| # | earlier wording | corrected |
|---|---|---|
| A | "the mechanism works" | The representation is used, and inference is sensitive to it in some wide-support conditions. That establishes neither superiority nor general operator understanding, and there is a counterexample at Jena INTERVAL_MEAN r = 8. |
| B | reconstructability explains the R reversal | The reversal is consistent with a limited reconstruction diagnostic (one channel, 200 windows, per dataset). Causality was not established. |
| C | "M and O do not use width" | The explicit scalar width channel had negligible sensitivity under the mismatch diagnostic. Resolution may still reach the models through other inputs. |
| D | the FlowState native reference is valid for both operations | Native-rate hourly-mean evaluation is valid only where the predicted quantity tiles the hour. END_BIN native rows cannot be read as hourly-mean forecasts at any r. |
| E | "12 raw error arrays are present in the commit" | They are present locally under gitignored `runs/` and were never committed. Their hashes are recorded and block-level sufficient statistics are committed instead. |
| F | r = 12 seed table | The R column came from END_BIN. Corrected table in section 15; scope is reporting only. |
| G | "moving-block bootstrap" | Paired 7-day time-block (cluster) bootstrap. |
| I | width sensitivity "at most 0.016 %" | 0.088 % across individual cells. The smaller figure was a maximum over dataset- and seed-averaged cells and understated the bound. |
| H | FlowState "zero-shot" pipeline | Zero-shot weights with a target-validation-tuned reconstruction preprocessor. |

## 21. Current implementation recommendation

`STOP_SCALING_CURRENT_INTERVAL_INTEGRATED_FOURIER_O`. No new threshold was
introduced; this rests on observations already in the artifacts:

- primary O vs M at unseen interpolation is approximately zero
- no consistent sign across datasets
- no consistent sign across seeds
- the interval contains zero
- the seen-resolution contrast shows no improvement either
- the r = 12 gain is one cell, and vanishes without it
- O does not beat the simple R baseline on that same Jena r = 12 cell
- the pre-registered method GO failed

That is an engineering call about where to spend the next effort. The scientific
verdict remains `INCONCLUSIVE`.

## 22. What the broader topic still contains

This pilot tested one representation — an interval-averaged Fourier positional
basis — inside one small target-trained architecture. Untested by v1:

- a learned latent continuous-time state with an explicit operator likelihood
- a model handling point, interval mean, integral and extrema together
- irregular intervals, and heterogeneous operators or widths inside one sequence
- operator-conditioned decoder queries and multi-resolution output requests
- probabilistic uncertainty arising from coarse observations
- foundation-model adaptation, more domains, wider resolution ranges
- genuine source-to-source measurement-semantics transfer

`OPEN_NOT_DIRECTLY_TESTED`. Nothing here shows that observation-aware modelling
fails as a class.

## 23. Next research choice

Move to topic 03 (uncertain future covariates) or another candidate. Do not start a
Phase-2 adapter on this representation. No new experiment is launched by this audit.
