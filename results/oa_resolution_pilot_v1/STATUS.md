OA-RESOLUTION-PILOT-v1 RESULT

EXECUTION STATUS:
COMPLETE

SCIENTIFIC DECISION:
INCONCLUSIVE

PRIMARY O vs M UNSEEN-INTERPOLATION:
-0.049 % macro (bootstrap mean -0.054 %, 95 % CI [-0.279, +0.175])

O vs R:
+0.574 % macro (bootstrap mean +0.532 %, 95 % CI [-0.788, +1.736])

FLOWSTATE REFERENCE:
OK — ibm-research/flowstate revision r1.1, zero-shot, both variants run

---

## What the pilot asked

Given the value, the support interval, the operation type and the report width —
the same information for every arm — does folding the observation interval into the
positional representation (O) beat a metadata-conditioned generic model (M) and a
correct operator-aware resampling baseline (R) at time resolutions never trained on?

## Answer

**No, at the pre-registered primary contrast.** At the unseen interpolation
resolutions (30 and 60 minutes) O and M are indistinguishable: the macro relative
improvement is -0.049 %, the paired time-block bootstrap interval spans zero, the
sign flips between the two model seeds and between the two datasets. Four of the
six Go/No-Go conditions fail.

`OPERATOR_AWARE_REPRESENTATION_PROMISING` is **not** met.

The decision token is INCONCLUSIVE rather than a cleaner "resampling is enough" or
"metadata is enough" because the R comparison does not point one way: R is the best
arm on Jena at every resolution role and the worst on UCI at every resolution role.
Pooled, M vs R is +0.588 % with a 95 % interval of [-0.621, +1.720] — not the
significant advantage that `METADATA_MODEL_SUFFICIENT` requires.

## Primary table — unseen interpolation (r = 3, 6)

Values are the primary loss `0.5·MSE_10min + 0.5·MSE_60min` on the normalised scale,
averaged over origins, channels and both model seeds on the frozen test keys.

| dataset | operation | r | R | M | O | O vs M | O vs R |
|---|---|---|---|---|---|---|---|
| jena | END_BIN | 3 | 0.14777 | 0.14914 | 0.14979 | -0.44% | -1.37% |
| jena | END_BIN | 6 | 0.14774 | 0.14869 | 0.14964 | -0.64% | -1.29% |
| jena | INTERVAL_MEAN | 3 | 0.15013 | 0.15154 | 0.15012 | +0.94% | +0.00% |
| jena | INTERVAL_MEAN | 6 | 0.15339 | 0.15608 | 0.15672 | -0.42% | -2.17% |
| uci | END_BIN | 3 | 0.45178 | 0.44056 | 0.44001 | +0.12% | +2.60% |
| uci | END_BIN | 6 | 0.46773 | 0.45407 | 0.45363 | +0.10% | +3.01% |
| uci | INTERVAL_MEAN | 3 | 0.45447 | 0.44415 | 0.44411 | +0.01% | +2.28% |
| uci | INTERVAL_MEAN | 6 | 0.46106 | 0.45380 | 0.45409 | -0.06% | +1.51% |

The eight cells are weighted equally. Seen (r = 2, 4, 8) and unseen extrapolation
(r = 12) are in `tables.md` and are never pooled into this macro.

## The pre-registered gate

| # | condition | threshold | value | pass |
|---|---|---|---|---|
| 1 | O vs M, unseen interpolation macro | ≥ +1.0 % | -0.049 % | no |
| 2 | both datasets positive | > 0 | jena -0.139 %, uci +0.042 % | no |
| 3 | both seeds positive | > 0 | seed0 -0.300 %, seed1 +0.202 % | no |
| 4 | paired bootstrap lower 95 % | > 0 | -0.279 % | no |
| 5 | O vs R, unseen interpolation macro | ≥ +0.5 % | +0.574 % | yes |
| 6 | O vs M, seen macro | ≥ -0.5 % | -0.182 % | yes |

## Secondary observations

**The R comparison flips sign by dataset.** The gap tracks how reconstructible each
signal is: R recovers the Jena base grid almost exactly from coarse observations
(RMSE 0.03-0.05 against a signal std of 0.91) but only roughly on the spiky UCI
household load (0.38-0.59 against 0.86), and that is where the token models win.
 On Jena the operator-aware
reconstruction plus a base-grid forecaster is the best arm everywhere; on UCI it is
the worst everywhere. Pooled numbers hide this, so the per-dataset intervals are
reported alongside.

Bootstrap mean and 95 % interval (the point macro from `primary_contrasts.json` is
+0.624 % for M vs R and +0.574 % for O vs R; the bootstrap mean differs slightly
because it averages over resampled block compositions).

| contrast, unseen interpolation | pooled | jena | uci |
|---|---|---|---|
| O vs M | -0.054 % [-0.279, +0.175] | -0.141 % [-0.529, +0.257] | +0.040 % [-0.149, +0.203] |
| M vs R | +0.588 % [-0.621, +1.720] | -1.137 % [-3.447, +0.858] | +2.339 % [+1.159, +3.506] |
| O vs R | +0.532 % [-0.788, +1.736] | -1.280 % [-3.668, +0.836] | +2.378 % [+1.201, +3.530] |

**At the widest unseen resolution the picture is different.** At r = 12 (120 min,
unseen extrapolation) O beats M by +1.760 % macro, bootstrap [+1.193, +2.430]. That
number is carried almost entirely by one dataset and one operation: Jena with
INTERVAL_MEAN, +7.44 %. On UCI the same contrast is +0.049 % [-0.318, +0.381]. And
R still beats both arms on Jena at r = 12. This is reported as an observation on a
single unseen extrapolation resolution, not as evidence for the mechanism, and it is
kept out of the primary macro.

That cell reproduces in both seeds, but not at the same size, and the movement is on
M's side rather than O's:

| jena INTERVAL_MEAN r = 12 | M | O | R | O vs M |
|---|---|---|---|---|
| seed 2026090601 | 0.17568 | 0.16782 | 0.14457 | +4.47 % |
| seed 2026090602 | 0.18705 | 0.16794 | 0.14936 | +10.22 % |

O lands on the same loss under both seeds while M does not, so what the number
records is that the centre-point basis becomes unreliable at 120-minute interval
means, not that the integrated basis found extra signal. At r = 2 through 8 on the
same dataset and operation the O vs M sign flips between seeds and stays near zero.

**O really does read the support, and exactly where the support is wide.**
Replacing the integrated basis with the centre basis at inference (O-WRONG-SUPPORT,
no retraining) barely moves END_BIN, whose support is one base bin at every
resolution, and costs a lot on wide interval means:

| O, loss increase when the support is described as a point | r=2 | r=3 | r=4 | r=6 | r=8 | r=12 |
|---|---|---|---|---|---|---|
| jena END_BIN | -0.55 % | -0.10 % | -0.36 % | +0.32 % | -0.15 % | +0.53 % |
| jena INTERVAL_MEAN | +0.42 % | +1.56 % | +0.57 % | -0.09 % | +7.93 % | +7.81 % |
| uci END_BIN | +0.05 % | +0.10 % | +0.06 % | +0.07 % | +0.03 % | -0.03 % |
| uci INTERVAL_MEAN | +0.04 % | +0.18 % | +0.03 % | +0.14 % | +0.65 % | +1.00 % |

That is coherent with the r = 12 result: the one cell where O beat M is Jena
INTERVAL_MEAN, and it is the same cell where taking O's integrated support away hurts
most. The mechanism is doing what it was designed to do. It is simply not worth
anything at the 30- and 60-minute resolutions the pilot pre-registered as primary.

**Neither arm uses the scalar width channel.** Feeding the width of a different
report interval (WIDTH_MISMATCH) changes the loss by at most 0.016 % for either M or
O. For END_BIN this variant is a no-op by construction — the support is one base bin
whatever r is — so only the INTERVAL_MEAN rows carry information, and they are flat
too. O gets its width knowledge from the integrated basis; M appears not to use
width at all.

The spec asks instead for a within-batch permutation of the width metadata. That is
provably the identity here: one batch holds one report interval, so all of its tokens
share one width. The evidence is recorded in `mechanism_diagnostics.json` under
`within_batch_width_permutation`, and the mismatch variant was run in its place.

A diagnostic that moves is not by itself a result: the primary contrast has to clear
M and R first, and it does not.

## FlowState reference

`ibm-research/flowstate`, revision `r1.1`, Apache-2.0, 18.5 M parameters, run through
the official `tsfm_public` API. `scale_factor` is not guessed — the model card
defines it as base seasonality 24 divided by the steps in the data's daily cycle, so
the 10-minute grid takes 24/144 and a report interval of r base bins takes r/6.

| primary loss, unseen interpolation | R | M | O | FlowState (resampled) |
|---|---|---|---|---|
| jena | 0.14976 | 0.15136 | 0.15157 | 0.17703 |
| uci | 0.45876 | 0.44815 | 0.44796 | 0.63547 |

FlowState is **zero-shot pretrained**; R, M and O are trained on the target data.
The numbers sit in one table but the information conditions are not the same, and
this is not a matched training-budget contest. Nothing here says the pilot improved
on, or beat, FlowState.

The native-rate variant was run where the official contract allows it: the coarse
output grid tiles the hour only for r ∈ {2, 3, 6}, so the 60-minute metric is formed
there with the same block mean the core arms use. The other six conditions are
recorded as not evaluated rather than filled in with a locally invented upsampler.

Chronos-2: `CHRONOS_REFERENCE_SKIPPED_TIMEBOX` — not importable in this interpreter,
and section 39 allows no dependency work for it. TiRex-2: not a required reference,
no environment time spent.

## Integrity checks

| check | result |
|---|---|
| measurement semantics from official documentation | PASS, both datasets |
| operator contract tests T01-T10, A01-A04, F02-F04 | 130 tests pass, gating the fits |
| analytical representation vs dense quadrature | max abs error 2.7e-10, tolerance 1e-05 |
| leakage L01-L08 and fairness F01-F06 | 15 tests pass (added after the fits; recorded separately) |
| M and O trainable parameter counts | 411,585 = 411,585 |
| R forecaster capacity | 407,041 trainable parameters, 1.1 % below M and O; R has no width projection because a base bin has one width, which a linear layer could absorb anyway |
| M and O initial state under one seed | byte-identical |
| training schedule SHA per (dataset, seed) | one value across all three arms |
| SUM = r · MEAN unit contract | representation and prediction differences both exactly 0 |
| evaluation windows | identical set across arms, operations and resolutions |
| learning anchors | every arm beats persistence and the one-day seasonal naive by a wide margin |
| resource caps | peak RSS 1.41 GiB of 10 GiB, peak GPU 0.35 GiB of 9.59 GiB |
| best checkpoint position | never the first or the last validation point in any of the 12 fits |
| diagnostic BASE path vs frozen metrics | agree to 4.4e-08 across all 144 (model, operation, r) cells |
| R baseline not crippled | at the selected lambda the reconstruction RMSE against the true base grid is 0.029-0.052 on Jena and 0.384-0.586 on UCI, against signal std 0.910 and 0.864 |

## The ten questions that gate a method GO

| # | question | answer | evidence |
|---|---|---|---|
| 1 | Did R use the observation semantics correctly? | yes | its operator matrix puts 1/r on the support bins for INTERVAL_MEAN and 1 on the last base bin for END_BIN; T05 checks the rows, and R recovers a linear signal to 1e-6 |
| 2 | Did M and O receive the same metadata? | yes | `token_features` returns identical width, operation id and support for both; asserted for every (operation, r) |
| 3 | Did O alone see a hidden high-resolution signal? | no | both arms consume the same `observe(history, r, op)` tensor from one shared code path; O's extra input is a fixed analytic function of bounds M also holds |
| 4 | Is the M/O difference really the integrated interval representation? | yes | same module class, 411,585 parameters each, byte-identical initial state under one seed, one schedule SHA; only the time basis array differs |
| 5 | Were 30 and 60 minutes genuinely unseen? | yes | training cycled over r = 2, 4, 8 only, and checkpoint selection scored validation on those same three |
| 6 | Was the 120-minute extrapolation kept out of the interpolation macro? | yes | the role field separates them, the macro is computed per role, and the three tables are printed separately |
| 7 | Was a SUM/MEAN unit difference counted as a model gain? | no | SUM was never a performance arm; it appears only in the invariance control, which returned exactly zero difference |
| 8 | Was FlowState called a matched training condition? | no | it is labelled zero-shot pretrained against target-trained arms everywhere it appears |
| 9 | Do the point estimate and the bootstrap read the same keys? | yes | both derive from the one seed-averaged frame built from the frozen per-key error arrays, filtered identically |
| 10 | Were the period, variables or seeds changed to improve the result? | no | the period came from a pre-registered coverage rule run before any model existed, the channels from the semantics audit, and both seeds were run from the start and both are reported |

All ten hold, so nothing here blocks a method GO. The GO fails on the numbers.

## What was NOT established

- Nothing about resolutions outside 20–120 minutes, horizons other than 12 hours, or
  histories other than 48 hours.
- Nothing about observation operators other than END_BIN and INTERVAL_MEAN. The
  discrete SUM was used only as a unit-invariance control, never as a performance arm.
- Nothing about whether the r = 12 gain would survive as a pre-registered primary
  contrast, on more datasets, or at more extrapolation resolutions.
- Nothing about large pretrained models carrying this representation. FlowState was
  used unmodified as a reference; no adapter or input layer was attached to it.
- Nothing about probabilistic forecasts, calibration or coherence across aggregation
  levels. Only point error was measured.
- The 5,000-update budget is a pilot budget, identical for every arm. The arms are
  small target-trained models, not foundation models.

## Known limitation in the source contract

The UCI documentation states the values are minute-averaged but never says whether
the printed timestamp labels the start or the end of that minute. The convention
`[t, t+1min)` was adopted and recorded rather than inferred from the numbers. The
two possible anchors differ by a uniform one-minute shift of the whole labelled
grid, which moves history, origin and target together and is identical for all three
arms, so it cannot change any contrast reported here — only the wall-clock reading
of the results.

## Correction applied after the numbers were in

The first gate implementation read the section 35 clause "M is significantly better
than R" as a point-estimate macro threshold and dropped the significance
requirement, which produced `METADATA_MODEL_SUFFICIENT`. The paired bootstrap puts
that interval across zero and the sign flips by dataset, so the clause is not
satisfied and the outcome falls through to `INCONCLUSIVE`. No threshold was lowered,
no arm was added, and the correction makes the verdict weaker rather than stronger.
Only the scoring stage was rerun; the fits, the frozen keys and the per-key error
arrays are untouched. Both tokens are recorded in `verdict.json`.

## Next decision

The pre-registered primary contrast failed, so the Phase-2 extension — carrying this
representation into a pretrained model as an adapter or input layer — is **not**
proposed. The recommendation is to stop expanding this method and move to another
candidate topic.

If the topic is ever revisited, the one thread worth pulling is the wide-support
regime: both the r = 12 contrast and the O-WRONG-SUPPORT diagnostic point the same
way, and both are currently single-resolution, largely single-dataset observations.
That would be a new pre-registration, not a continuation of this one.
