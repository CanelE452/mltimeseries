HQ-TOKEN-PILOT-v1 AUDIT RESULT

ORIGINAL SCIENTIFIC DECISION: INCONCLUSIVE

CURRENT IMPLEMENTATION RECOMMENDATION: STOP_SCALING_CURRENT_FIXED_NEIGHBORHOOD_C

BROADER TOPIC: OPEN_NOT_DIRECTLY_TESTED

These are three separate statements. The first is the pre-registered decision and was not revised. The second is an engineering recommendation about this exact implementation. The third says the wider question was never directly tested.

## 1. What was audited

Commit `36b01d2b84df16f03447a8e214723fc61ae147fd`. Every number below was recomputed from the stored artifacts by a script that imports nothing from the experiment package, so a disagreement would be a real disagreement and not a shared bug.

- the six contrasts, recomputed from `metrics.csv`
- the paired bootstrap, re-derived from the raw keyed error arrays
- validation curves and checkpoint selection for the six paired I and C fits
- the training-schedule hashes across arms
- the synthetic capacity control, the seed study, the compression gap, the query diagnostics, the efficiency table and the reference scope

## 2. What was NOT rerun

No model was fitted. The 27 core fits, the 24 seed-noise fits, the synthetic capacity training and the two foundation references were all left as they were. No learning rate, budget, seed, token budget or threshold was changed, and the pre-registered decision was not revised. Existing result files were read-only: 19 were hashed before and after this audit, result **ORIGINAL_ARTIFACTS_UNCHANGED**.

## 3. Arithmetic reproduction

Status **ARITHMETIC_REPRODUCTION_OK**. Largest absolute disagreement over every cell, macro and per-seed value: 2.22e-14 percentage points against a tolerance of 1e-10.

| contrast | macro RI | seeds used |
| --- | --- | --- |
| C vs I | -0.037% | 2 |
| C vs H STATIC | +1.891% | 2 |
| C vs R | -0.008% | 1 |
| C vs U | +2.839% | 1 |
| C vs DENSE | -0.973% | 1 |
| DENSE vs U | +3.719% | 1 |

| dataset | C vs I at H=96 | at H=336 |
| --- | --- | --- |
| ETTm2 | -0.019% | -0.234% |
| weather | +0.030% | -0.040% |
| electricity | +0.032% | +0.007% |

Bootstrap re-derived independently under the same contract: **BOOTSTRAP_REPRODUCTION_OK**. Recomputed macro -0.039% with interval [-0.128, +0.033], against the stored -0.039% [-0.128, +0.033]. Effective time blocks: ETTm2 8, weather 8, electricity 4.

Per-seed macro, computed within each seed:

- seed 2026090601: -0.072%
- seed 2026090602: +0.006%

Two seeds are not a sample from which a standard error or an interval is derived here, and none is reported.

## 4. Training curves

| dataset | seed | arm | best update | position | last interval | final / best |
| --- | --- | --- | --- | --- | --- | --- |
| ETTm2 | ...01 | I | 600 | EARLY_BEST | LAST_INTERVAL_WORSENING | 1.127 |
| ETTm2 | ...01 | C | 600 | EARLY_BEST | LAST_INTERVAL_WORSENING | 1.133 |
| ETTm2 | ...02 | I | 600 | EARLY_BEST | LAST_INTERVAL_WORSENING | 1.162 |
| ETTm2 | ...02 | C | 600 | EARLY_BEST | LAST_INTERVAL_WORSENING | 1.170 |
| weather | ...01 | I | 3000 | LATE_BEST | LAST_INTERVAL_IMPROVING | 1.000 |
| weather | ...01 | C | 3000 | LATE_BEST | LAST_INTERVAL_IMPROVING | 1.000 |
| weather | ...02 | I | 1200 | INTERIOR_BEST | LAST_INTERVAL_WORSENING | 1.032 |
| weather | ...02 | C | 1200 | INTERIOR_BEST | LAST_INTERVAL_WORSENING | 1.032 |
| electricity | ...01 | I | 3000 | LATE_BEST | LAST_INTERVAL_IMPROVING | 1.000 |
| electricity | ...01 | C | 3000 | LATE_BEST | LAST_INTERVAL_IMPROVING | 1.000 |
| electricity | ...02 | I | 3000 | LATE_BEST | LAST_INTERVAL_IMPROVING | 1.000 |
| electricity | ...02 | C | 3000 | LATE_BEST | LAST_INTERVAL_IMPROVING | 1.000 |

Three different optimization states, described rather than diagnosed:

- **ETTm2** picked its best checkpoint at the very first validation point, update 600 of 3000, after which validation loss rose by 13 to 17 percent to the end of the budget. Both arms, both seeds. The reported ETTm2 numbers therefore come from models trained for 600 updates, and the rest of the budget made them worse.
- **weather** is mixed: one seed was still improving at the last checkpoint, the other peaked in the middle and then worsened.
- **electricity** was still improving at the final checkpoint in both seeds, so the budget was the binding constraint there.

These labels describe the curves. They are not a diagnosis of overfitting or of non-convergence, and no checkpoint was re-selected.

Sampling exposure, read next to those labels:

| dataset | eligible channel-origin pairs | windows drawn | exposure ratio | curve position |
| --- | --- | --- | --- | --- |
| ETTm2 | 232,407 | 192,000 | 0.826 | EARLY_BEST |
| weather | 746,088 | 192,000 | 0.257 | INTERIOR_BEST, LATE_BEST |
| electricity | 5,474,013 | 192,000 | 0.035 | LATE_BEST |

Called a sampling exposure ratio and not an epoch: training windows are sliding and overlap heavily, so a ratio below one does not mean a correspondingly small amount of distinct signal was seen.

A small exposure ratio is only evidence of an insufficient budget when the validation curve was still improving at the last checkpoint. The labels are given next to the ratio for that reason.

## 5. Checkpoint selection

6 of 6 paired fits selected the same update for C and for I. Where the two arms selected different updates, the reported test contrast compares two independently validation-selected checkpoints. That is the pre-registered procedure, but it means the pairing is on the seed and the data schedule, not on the optimizer step.

No checkpoint was re-chosen. Test results were not consulted for selection at any point in this audit.

The two arms also track each other throughout training and not only at the selected point: across the 30 shared validation checkpoints the median absolute difference in validation MSE is 0.037 percent and the largest is 1.400 percent. The 3 checkpoints above 0.5 percent are all on ETTm2 and change sign between them, so the closeness at the selected checkpoint is not manufactured by the selection.

## 6. Query mechanism

| dataset | H | C vs I | C vs R | query swap | mean weight shift | mean centre shift |
| --- | --- | --- | --- | --- | --- | --- |
| ETTm2 | 96 | -0.019% | -0.019% | +0.006% | 0.00051 | 0.008 samples |
| ETTm2 | 336 | -0.234% | +0.050% | -0.001% | 0.00051 | 0.008 samples |
| weather | 96 | +0.030% | -0.001% | -0.001% | 0.00122 | 0.020 samples |
| weather | 336 | -0.040% | -0.083% | +0.003% | 0.00122 | 0.020 samples |
| electricity | 96 | +0.032% | +0.003% | -0.005% | 0.00135 | 0.022 samples |
| electricity | 336 | +0.007% | +0.001% | +0.008% | 0.00135 | 0.022 samples |

This is the most direct mechanism evidence in the closure. Under this implementation the true horizon query bought no measurable accuracy over an input-only query, a random horizon query performed the same, and changing only the pooling query on fixed inputs barely moved the weights or the prediction.

## 7. Synthetic capacity control, reinterpreted

| arm | MSE H=96 | MSE H=336 | joint equal-weight mean |
| --- | --- | --- | --- |
| I | 0.01063 | 0.02826 | 0.01945 |
| C | 0.00525 | 0.29112 | 0.14819 |

- QUERY_PATH_RESPONSIVE = True (weight separation 0.813)
- JOINT_HORIZON_TASK_SOLVED = False (C is 7.6 times worse than I on the joint objective)

This scorer path was able to learn a function that changes the pooling weight substantially with the horizon on the synthetic task.

On that same synthetic task the horizon-conditioned arm did not beat the input-only arm on the joint multi-horizon objective, which is the objective the real runs optimize. The control therefore establishes query-path responsiveness only; it does not exclude architecture or optimization limitations as an explanation for the real-data null.

The task was built on the argument that one token per group can carry only one mixture of its two patches. That argument is not airtight: when the mixture coefficient itself varies with content, the coefficient can act as a side channel, so a content-only pooler is not strictly limited to a single linear combination. No claim is made about whether the trained I arm exploits this.

## 8. Seed uncertainty, reinterpreted

| dataset | mean validation MSE | seed sd | sd as percent of mean | seeds |
| --- | --- | --- | --- | --- |
| ETTm2 | 0.19924 | 0.01006 | 5.05% | 8 |
| weather | 0.43137 | 0.00413 | 0.96% | 8 |
| electricity | 0.15291 | 0.00119 | 0.78% | 8 |

The absolute validation MSE of arm I varies noticeably across seeds, most on ETTm2.

The seed-to-seed variability of the paired C minus I difference was not measured. C and I share their initial state and their data schedule within a seed, so the paired difference could be far more stable than either arm's absolute level, or less so. The 8-seed study was run on I alone and cannot distinguish these.

`paired_CI_effect_seed_variability_estimated` = False, because extra seeds exist only for I. The core pair has 2 seeds.

## 9. Compression gap, reinterpreted

OBSERVED_DENSE_VS_UNIFORM_COMPRESSION_GAP = +3.719%, at model seed 2026090601 only.

The trained dense arm is one trained model, not an oracle. A compressed model can beat it, for instance through the regularizing effect of the bottleneck, so the observed gap does not cap what a pooling rule could win.

`DENSE_IS_NOT_EMPIRICAL_UPPER_BOUND` = True: 2 of six cells already have the compressed C arm ahead of the uncompressed arm (ETTm2 H=96, ETTm2 H=336).

## 10. Efficiency

| config | tokens | end-to-end b64 (ms) | vs dense | peak (MiB) |
| --- | --- | --- | --- | --- |
| U_B32 | 32 | 2.124 | +7.8% | 39.7 |
| I_B32 | 32 | 2.488 | +26.3% | 39.8 |
| C_B32 | 32 | 2.733 | +38.8% | 39.8 |
| H_STATIC_B32 | 32 | 2.187 | +11.0% | 39.7 |
| R_B32 | 32 | 2.627 | +33.4% | 39.8 |
| DENSE_B64 | 64 | 1.970 | +0.0% | 45.7 |

Per-stage micro-profile, inference only, diagnostic:

| stage | U (32) | C (32) | DENSE (64) |
| --- | --- | --- | --- |
| embed | 0.046 | 0.043 | 0.051 |
| pool | 0.035 | 0.288 | 0.004 |
| encoder | 0.622 | 0.619 | 0.662 |
| unmerge readout head | 0.101 | 0.101 | 0.081 |
| total forward | 0.968 | 1.196 | 0.981 |

Halving the token count saves 0.040 ms in the encoder, while the content scorer alone adds 0.253 ms and the unmerge path adds a further 0.020 ms.

A disagreement worth stating rather than smoothing over: the two measurements disagree on whether uniform pooling is slower than dense. The stored table puts uniform pooling +7.8 percent against dense, while the longer micro-profile puts it at 0.968 ms against 0.981 ms. The sign of the U versus DENSE gap flips between the stored measurement and the longer micro-profile, and both differences are a few percent. That ordering is therefore not established. Only the ordering that survives both measurements is reported as a finding: C is the slowest configuration, and the scorer is the reason.

Measured on a 1.5M parameter model at batch 64 on one RTX 4070. It says nothing about whether token compression pays off in a larger backbone, where attention is a much larger share of the cost.

## 11. Foundation reference scope

- Chronos-2: COMPLETE, pretrained, evaluated zero-shot, scored at the 0.5 quantile, while the core arms were trained on each target for squared error.
- TiRex-2: PARTIAL_HORIZON_UNSUPPORTED, H=336 is not available from the current public checkpoint, which caps prediction length at 320, so only H=96 has a reference row; run on CPU.

These are native reference rows. No performance claim in either direction is drawn from them and neither model was rerun in this audit.

## 12. Strong-baseline gap

No official implementation of Local Merging, byte pair encoding for time series, TimeSqueeze or PATK was located within the search timebox, so no strong dynamic-tokenization baseline was run under the same contract.

That is a statement about this search, not about whether such implementations exist, and it is not a claim of superiority over recent dynamic tokenization methods.

## 13. Claims that remain valid

- C and I were evaluated on exactly the same frozen keys, from the same initial state and the same training sample schedule within a seed; the schedule hashes match across arms for every dataset and seed.
- The primary C-versus-I effect under this pilot is -0.037 percent, with a re-derived interval of [-0.128, +0.033] percent.
- A random horizon query performed essentially the same as the true one: C versus R is -0.008 percent.
- Swapping only the pooling query on fixed inputs changed test MSE by at most 0.0082 percent, and the mean pooling weight moved by at most 0.00135 between the two horizons.
- Learned pooling did outperform uniform pooling in this pilot: C versus U is +2.839 percent.
- The phase-shifted evaluation grid gives -0.144 percent, so the grid phase does not reverse the null.
- On this small architecture the 32-token C model was not faster end to end than the 64-token dense model, and the per-stage profile attributes that to the scorer.

## 14. Claims that are withdrawn or weakened

| earlier claim | corrected statement |
| --- | --- |
| CAPACITY_CONFIRMED rules out architecture and optimization failure | Only query-path responsiveness was shown. On that same synthetic task the horizon-conditioned arm was worse than the input-only arm on the joint multi-horizon objective, so architecture and optimization limits are not excluded. |
| The 5.05 percent seed spread puts a 1 percent effect outside the resolution of the experiment | 5.05 percent describes the absolute variability of arm I across eight seeds. The seed variability of the paired C minus I difference was never measured and could be far smaller. |
| DENSE versus U bounds what any pooling rule could gain | It is one observed comparison between two trained models, not an upper bound. Two of six cells already have the compressed C arm ahead of the uncompressed arm. |
| The optimizer found no reason to use the horizon | Under this architecture and training budget, horizon-conditioned pooling produced no measurable advantage and the learned weights were nearly horizon-invariant. Why is not established. |
| This topic should be stopped | Stop scaling this fixed-neighbourhood implementation. The wider question of forecast-query-conditioned budget allocation remains open and was not directly tested. |

## 15. Current implementation recommendation

**STOP_SCALING_CURRENT_FIXED_NEIGHBORHOOD_C** — There is not enough empirical support to carry this exact C implementation to a larger backbone, more datasets, or a foundation model.

The observations behind it, listed rather than combined into a new threshold:

- `C_vs_I_macro_pct`: -0.03738746955939676
- `C_vs_I_per_seed_macro_pct`: {'2026090601': -0.07202014917790563, '2026090602': 0.005908315563426155}
- `C_vs_I_bootstrap`: {'macro_mean': -0.03938479848322728, 'lower95': -0.1275647237878516, 'upper95': 0.03276452743664624}
- `C_vs_R_macro_pct`: -0.008254918871619665
- `max_abs_query_swap_effect_pct`: 0.008205534256378044
- `max_mean_horizon_weight_shift`: 0.0013487989781424403
- `phase_shifted_macro_pct`: -0.14413693645100775
- `C_vs_DENSE_end_to_end_latency_pct`: 38.75634497898177
- `training_curve_position_labels`: {'ETTm2': ['EARLY_BEST'], 'weather': ['INTERIOR_BEST', 'LATE_BEST'], 'electricity': ['LATE_BEST']}
- `paired_fits_with_same_best_update`: 6
- `synthetic_joint_task_solved_by_C`: False

These are the observations already in the record. They were not combined into a new threshold, and the pre-registered scientific decision was not revised.

## 16. What remains untested

This pilot never had the freedom to:

- assign more tokens to one stretch of history and fewer to another
- move a group boundary
- drop an unimportant group entirely
- represent a long quiet stretch with one token and a busy stretch with several
- select non-local or non-contiguous regions of the history
- vary the total token budget with the horizon

Also untested: query-conditioned tokenizer adaptation inside a pretrained foundation model, and any regime where the token saving outweighs the pooling overhead, which at 1.5M parameters it does not.

Results apply to fixed adjacent grouping at r=2, one token per group, a 1.5M parameter model, context 1024, horizons 96 and 336, token budget 32, and the training budget actually used.

The horizon embedding module is shared between the pooling query and the decoder, so C and I differ by slightly more than the presence of a query. That does not flatter the current null, but it would need separating before any future positive result from this design were believed.

## 17. Exact next research choice

Two options, and no new training starts until one is chosen.

**A.** Close topic 1 and move to the second topic note.

**B.** Keep topic 1, but not by adjusting the r=2 fixed-neighbourhood mixture. Design a separate pre-registered v2 around horizon-dependent variable token allocation with movable boundaries, which is the part of the original idea this pilot never had the freedom to express.

