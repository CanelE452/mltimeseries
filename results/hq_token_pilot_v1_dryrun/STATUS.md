# HQ-TOKEN-PILOT-v1 — STATUS

**execution_status = COMPLETE**, **scientific_decision = INCONCLUSIVE**, evidence_level = DEVELOPMENT_SCREEN.

27 of 27 pre-registered fits completed. Total wall time this invocation: 0.9 min.

## 1. What this run does and does not claim

This is a development screen of horizon-conditioned local pooling at a fixed token budget. It is not a foundation-model result, not a benchmark submission, and not a statement that the topic as a whole works or fails. The model is a small pilot that borrows patching and channel independence from PatchTST; it is not official PatchTST and has no instance normalization, so its absolute numbers are not comparable to published tables.

## 2. Primary contrast — C versus I

| dataset | H=96 | H=336 | dataset mean |
| --- | --- | --- | --- |
| ETTm2 | +0.01% | +0.00% | +0.01% |
| weather | +0.00% | -0.00% | +0.00% |
| electricity | -0.03% | -0.02% | -0.02% |
| **macro (6 cells, equal weight)** | | | **-0.01%** |

Paired circular moving block bootstrap, 50 draws, block = 15 origins: macro mean -0.01%, 95% interval [-0.01%, -0.00%]. Model seeds are not resampled, so this is a time-sample interval conditional on the two observed seeds.

Bootstrap flags: LOW_EFFECTIVE_TIME_BLOCKS:electricity:4.

Effective time blocks per dataset: ETTm2 8, weather 8, electricity 4.

Per-seed macro RI (sign condition of section 14):

- seed 2026090601: +0.00%
- seed 2026090602: -0.01%

## 3. Pre-registered decision conditions

| condition | met |
| --- | --- |
| core complete for I H STATIC C both seeds | yes |
| macro RI C vs I at least 1pct | no |
| at least 2 datasets positive | yes |
| no dataset worse than -1pct | yes |
| both seed macros positive | no |
| bootstrap lower95 positive | no |
| macro RI C vs H STATIC at least 0.3pct | yes |
| seed0 C vs R positive | no |

Reading notes:

- effect present but at least one pre-registered condition failed
- compression headroom (DENSE over U) is 0.77 percent, below the 1.0 percent gate. Under this budget no pooling rule could have reached the threshold, so a negative reading here is a statement about the setting, not about the hypothesis.
- seed-to-seed sd of validation MSE reaches 9.90 percent, at or above the 1.0 percent gate; the two-seed interval does not carry that source of variation

## 4. Scope and explanation contrasts

| contrast | macro RI | role |
| --- | --- | --- |
| C vs I | -0.01% | primary |
| C vs H_STATIC | +5.31% | separates content-conditioned pooling from a fixed horizon prior |
| C vs R | -0.00% | seed0 only; random query control |
| C vs U | +4.32% | seed0 only; uniform pooling |
| C vs DENSE | +3.35% | seed0 only; no compression |
| DENSE vs U | +0.77% | precondition: how much accuracy the 64 to 32 compression costs at all |

The DENSE versus U row bounds what any pooling rule could win at this budget. If it is smaller than the 1.0 percent gate, the gate was unreachable by construction and a null must not be read as evidence against the hypothesis.

Robustness: on a phase-shifted evaluation grid the C versus I macro is -0.00% against -0.01% on the pre-registered grid. The pre-registered grid uses stride 96, which is exactly 24 h on ETTm2 and exactly 4 days on electricity, so all of its origins share one time of day.

## 5. Can this architecture express the mechanism at all

Verdict: **CAPACITY_NOT_DEMONSTRATED**.

The task gives every group one a-patch and one b-patch at independent levels; the short future is the mean of the a-levels and the long future the mean of the b-levels. Because the a-levels are mutually independent, no fixed allocation of the 32 groups serves both futures.

| arm | weight on the a-patch at H=96 | at H=336 | MSE H=96 | MSE H=336 |
| --- | --- | --- | --- | --- |
| I | 0.505 | 0.505 | 0.6860 | 0.7888 |
| C | 0.504 | 0.504 | 0.8051 | 0.8733 |

C moves its pooling weight by 0.000 between the two horizons and improves the H=96 cell by -17.36%. The verdict reads the H=96 cell and the weight separation. The H=336 cell is expected to favour I on this task and is not evidence against capacity: the H=336 loss mask spans steps 0..335, so it still contains the 96 a-steps, and a single weight per horizon cannot serve both halves of that window. What matters is whether the scorer moves the weight when the horizon changes.

This is synthetic and is not a benchmark result. Its only job is to keep a null on real data from being ambiguous between a wrong hypothesis and an architecture that cannot learn the function. Two rejected earlier versions of this check are themselves informative and are recorded in the run notes: with a single shared level on the a-patches, the input-only pooler solved the task by giving different groups different roles, so horizon conditioning bought nothing. Spatial allocation across 32 groups is a real alternative to horizon conditioning, and H_STATIC is the arm that probes it.

## 6. Resolution available to the 1.0 percent gate

| dataset | mean validation MSE | seed sd | sd as percent of mean |
| --- | --- | --- | --- |
| ETTm2 | 1.03215 | 0.10218 | 9.90% |
| weather | 0.73250 | 0.00281 | 0.38% |
| electricity | 0.72164 | 0.03194 | 4.43% |

Arm I trained at eight extra seeds, validation split only, before the core. Worst per-seed spread 9.90 percent against a 1.0 percent gate. This is a measurement of resolution, not an input to the verdict; the core uses exactly the two pre-registered seeds.

## 7. Accuracy table

| dataset | H | B | seed | arm | MSE | MAE |
| --- | --- | --- | --- | --- | --- | --- |
| ETTm2 | 96 | 32 | 2026090601 | C | 1.57612 | 0.87247 |
| ETTm2 | 96 | 32 | 2026090602 | C | 1.56595 | 0.87617 |
| ETTm2 | 96 | 64 | 2026090601 | DENSE | 1.83707 | 0.95608 |
| ETTm2 | 96 | 32 | 2026090601 | H_STATIC | 1.75559 | 0.93560 |
| ETTm2 | 96 | 32 | 2026090602 | H_STATIC | 1.76121 | 0.93983 |
| ETTm2 | 96 | 32 | 2026090601 | I | 1.57639 | 0.87259 |
| ETTm2 | 96 | 32 | 2026090602 | I | 1.56607 | 0.87622 |
| ETTm2 | 96 | 32 | 2026090601 | R | 1.57615 | 0.87247 |
| ETTm2 | 96 | 32 | 2026090601 | U | 1.75563 | 0.93562 |
| ETTm2 | 336 | 32 | 2026090601 | C | 2.43764 | 1.09666 |
| ETTm2 | 336 | 32 | 2026090602 | C | 2.33298 | 1.07230 |
| ETTm2 | 336 | 64 | 2026090601 | DENSE | 2.53148 | 1.13547 |
| ETTm2 | 336 | 32 | 2026090601 | H_STATIC | 2.49725 | 1.12293 |
| ETTm2 | 336 | 32 | 2026090602 | H_STATIC | 2.53547 | 1.13314 |
| ETTm2 | 336 | 32 | 2026090601 | I | 2.43779 | 1.09671 |
| ETTm2 | 336 | 32 | 2026090602 | I | 2.33300 | 1.07231 |
| ETTm2 | 336 | 32 | 2026090601 | R | 2.43764 | 1.09665 |
| ETTm2 | 336 | 32 | 2026090601 | U | 2.49726 | 1.12293 |
| electricity | 96 | 32 | 2026090601 | C | 0.73999 | 0.69040 |
| electricity | 96 | 32 | 2026090602 | C | 0.67018 | 0.64916 |
| electricity | 96 | 64 | 2026090601 | DENSE | 0.68745 | 0.65914 |
| electricity | 96 | 32 | 2026090601 | H_STATIC | 0.74238 | 0.68958 |
| electricity | 96 | 32 | 2026090602 | H_STATIC | 0.73587 | 0.68981 |
| electricity | 96 | 32 | 2026090601 | I | 0.73991 | 0.69040 |
| electricity | 96 | 32 | 2026090602 | I | 0.66985 | 0.64896 |
| electricity | 96 | 32 | 2026090601 | R | 0.73998 | 0.69039 |
| electricity | 96 | 32 | 2026090601 | U | 0.74239 | 0.68959 |
| electricity | 336 | 32 | 2026090601 | C | 0.92532 | 0.77476 |
| electricity | 336 | 32 | 2026090602 | C | 0.87415 | 0.74446 |
| electricity | 336 | 64 | 2026090601 | DENSE | 0.89697 | 0.75926 |
| electricity | 336 | 32 | 2026090601 | H_STATIC | 0.91749 | 0.77253 |
| electricity | 336 | 32 | 2026090602 | H_STATIC | 0.89022 | 0.75581 |
| electricity | 336 | 32 | 2026090601 | I | 0.92518 | 0.77473 |
| electricity | 336 | 32 | 2026090602 | I | 0.87395 | 0.74434 |
| electricity | 336 | 32 | 2026090601 | R | 0.92534 | 0.77476 |
| electricity | 336 | 32 | 2026090601 | U | 0.91750 | 0.77253 |
| weather | 96 | 32 | 2026090601 | C | 0.41344 | 0.44608 |
| weather | 96 | 32 | 2026090602 | C | 0.42048 | 0.45884 |
| weather | 96 | 64 | 2026090601 | DENSE | 0.46632 | 0.48609 |
| weather | 96 | 32 | 2026090601 | H_STATIC | 0.47286 | 0.49112 |
| weather | 96 | 32 | 2026090602 | H_STATIC | 0.44359 | 0.47052 |
| weather | 96 | 32 | 2026090601 | I | 0.41345 | 0.44611 |
| weather | 96 | 32 | 2026090602 | I | 0.42048 | 0.45885 |
| weather | 96 | 32 | 2026090601 | R | 0.41342 | 0.44607 |
| weather | 96 | 32 | 2026090601 | U | 0.47287 | 0.49113 |
| weather | 336 | 32 | 2026090601 | C | 0.57564 | 0.54746 |
| weather | 336 | 32 | 2026090602 | C | 0.55979 | 0.54008 |
| weather | 336 | 64 | 2026090601 | DENSE | 0.58548 | 0.55751 |
| weather | 336 | 32 | 2026090601 | H_STATIC | 0.58308 | 0.55649 |
| weather | 336 | 32 | 2026090602 | H_STATIC | 0.57502 | 0.54973 |
| weather | 336 | 32 | 2026090601 | I | 0.57564 | 0.54746 |
| weather | 336 | 32 | 2026090602 | I | 0.55978 | 0.54006 |
| weather | 336 | 32 | 2026090601 | R | 0.57564 | 0.54746 |
| weather | 336 | 32 | 2026090601 | U | 0.58308 | 0.55649 |

Seasonal-naive anchor on the same grid, same standardized space:

| dataset | H=96 MSE | H=336 MSE | period (samples) |
| --- | --- | --- | --- |
| ETTm2 | 0.26602 | 0.36898 | 96 |
| weather | 0.32035 | 0.38159 | 144 |
| electricity | 0.29910 | 0.32461 | 24 |

## 8. Efficiency

| config | tokens | params | tokenizer params | model call b64 (ms) | end-to-end b64 (ms) | p90 (ms) | peak alloc (MiB) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| U_B32 | 32 | 1,538,992 | 0 | 1.974 | 2.081 | 2.435 | 39.7 |
| I_B32 | 32 | 1,563,697 | 24,705 | 2.328 | 2.511 | 2.681 | 39.8 |
| C_B32 | 32 | 1,563,697 | 24,705 | 2.563 | 2.702 | 2.810 | 39.8 |
| H_STATIC_B32 | 32 | 1,539,120 | 128 | 2.111 | 2.300 | 2.427 | 39.7 |
| R_B32 | 32 | 1,563,697 | 24,705 | 2.655 | 2.748 | 2.868 | 39.8 |
| DENSE_B64 | 64 | 1,538,992 | 0 | 1.813 | 1.958 | 2.184 | 45.7 |

Fewer attention FLOPs is not the same claim as a faster forecast. The end-to-end column includes the host-to-device copy; the model-call column does not. Token counts from BPE-style or foundation tokenizers are not comparable to these numbers, because a token does not mean the same thing.

## 9. Query contribution diagnostics (C only)

| dataset | H | MSE with true query | MSE with swapped query | change |
| --- | --- | --- | --- | --- |
| ETTm2 | 96 | 1.57612 | 1.57613 | +0.00% |
| ETTm2 | 336 | 2.43764 | 2.43764 | +0.00% |
| weather | 96 | 0.41344 | 0.41344 | +0.00% |
| weather | 336 | 0.57564 | 0.57564 | +0.00% |
| electricity | 96 | 0.73999 | 0.74001 | +0.00% |
| electricity | 336 | 0.92532 | 0.92531 | -0.00% |

| dataset | mean absolute weight difference (96 vs 336) | mean token centre shift (samples) |
| --- | --- | --- |
| ETTm2 | 0.0003 | 0.01 |
| weather | 0.0004 | 0.01 |
| electricity | 0.0007 | 0.01 |

A swapped query hurting is not by itself evidence for the method: the model never saw that mismatch during training. The separately trained I, H_STATIC and R arms carry the argument. Weight pictures are explanatory and cannot stand in for an accuracy effect.

## 10. Fits, blocks and evaluation scope

- completed fits / planned fits: 27 / 27
- no blocked or failed fits
- evaluation scope: sparse origin grid at stride 96, both horizons, all channels. This is a development screen and is not the official stride-1 benchmark.
- training budget as a fraction of one pass over eligible windows: ETTm2 0.006, weather 0.002, electricity 0.000. Every arm gets the same budget, so the contrast is fair, but what is measured is early-optimization quality rather than converged quality.
- peak process-tree RSS 1.49 GiB, peak GPU allocated 0.07 GiB, minimum available RAM 16.04 GiB. This guard bounds this job only and is not a guarantee about the machine.

## 11. Stated limitations

- **epoch fraction** — The fixed 3000-update budget is far below one pass over the eligible windows, most severely on electricity. What is measured is early-optimization quality, not converged quality.
- **horizon reaches the encoder through the tokens** — In C the pooling weights depend on H, so the merged tokens themselves encode H and the encoder can read it; in I they cannot. H_STATIC controls this path because its weights also depend on H while ignoring content, which is why the C versus H_STATIC contrast carries the identification weight here.
- **shared horizon embedding** — Section 6 defines a single e_H and section 7 reuses that same e_H as the C/R pooling query, so the module is shared. In C its parameters therefore receive gradient from both the decoder and the scorer path, which I does not. Implemented as written rather than split, because splitting it would create an arm the plan does not contain.
- **checkpoint selection noise** — One checkpoint out of five is chosen on the validation grid, which is only 24 origins on electricity. That selection noise is not carried into the interval.
- **no instance normalization** — The pilot model has no RevIN-style instance normalization. Section 6 does not include one. Absolute accuracy is therefore not comparable to published PatchTST numbers.

## 12. Artifacts

- `results/hq_token_pilot_v1/bootstrap.json`
- `results/hq_token_pilot_v1/contrasts.json`
- `results/hq_token_pilot_v1/data_manifest.json`
- `results/hq_token_pilot_v1/diagnostics.json`
- `results/hq_token_pilot_v1/efficiency.csv`
- `results/hq_token_pilot_v1/environment.json`
- `results/hq_token_pilot_v1/execution_spec.json`
- `results/hq_token_pilot_v1/execution_spec.sha256`
- `results/hq_token_pilot_v1/fit_manifest.csv`
- `results/hq_token_pilot_v1/mechanism_capacity_check.json`
- `results/hq_token_pilot_v1/metrics.csv`
- `results/hq_token_pilot_v1/seed_noise_floor.json`
- `results/hq_token_pilot_v1/verdict.json`

