# UCP-PATH-PILOT-v1 — STATUS

## 1. Executive verdict

**NO_INCREMENTAL_PATH_VALUE**

P over D: RI -0.11% (95% CI -0.19 to -0.05), 0 of 3 seeds positive.
P over S: RI -2.13% (95% CI -2.85 to -1.49), 0 of 3 seeds positive.

Screens: {"PATH_VALUE_GO": false, "ACCURACY_SCREEN_GO": false, "EFFICIENCY_SCREEN_GO": false, "NO_INCREMENTAL_PATH_VALUE": true}

This is a decision about whether the next step is worth funding, not an
acceptance bar, and not a novelty claim.

## 2. Exact scientific question

Given a weather tensor C in R^(K x T x D) of ensemble forecasts issued at the
origin, does preserving each member's temporal trajectory before pooling over
members (P) beat pooling members at each lead and reading the pooled path
afterwards (D), and beat per-lead summary statistics (S), in probabilistic wind
power forecasting on top of a frozen foundation model?

D loses cross-time member identity by construction. P keeps it. Everything else
about the two arms is identical.

## 3. Data contract

See `DATA_CONTRACT.md` for the full audit. K = 50 perturbed ENS members,
T = 12 leads at 6 to 72 h in 6 h steps, D = 4 weather channels, origins at 00 and
12 UTC. Target is BMRA metered power in MW, joined on exact timestamps.

Panel sizes: {"train": 191, "val": 206, "test": 181}.

Metered data is treated as available at the origin; publication latency was not
verified, so this is a base-time-aligned retrospective operational forecast
experiment, not a deployable operational result.

## 4. Source provenance

ECMWF/ENS Great Britain 2019-2020 (Zenodo 13255991) and 2021 (13256007), BMRA
wind power (13256014), author forecasting software (13309948) and BMRA
preprocessing code (13309890). Every archive md5 was verified against the Zenodo
record before use. Backbone `autogluon/chronos-2-synth` at revision
`3607918a9fd027d5c465d8213e46b98e2c041cea`, frozen, chronos-forecasting 2.3.1,
transformers 5.16.1, torch 2.11.0+cu128.

## 5. Farm selection

Eight wind farm sites, not eight datasets. Ranked on 2019 information only.

| farm | name | type | 2019 origins | 2020 origins | 2021 origins |
|---|---|---|---|---|---|
| E_BURBO | Burbo Bank Offshore Windfarm  | Offshore wind | 724 | 700 | 260 |
| T_BEATO-1 | Beatrice Offshore Wind Unit 1  | Offshore wind | 706 | 720 | 224 |
| T_GANW-13 | Galloper Offshore Windfarm 13  | Offshore wind | 718 | 698 | 421 |
| T_RMPNO-1 | Rampion Offshore Windfarm 1  | Offshore wind | 718 | 715 | 350 |
| E_GFLDW-1 | Goole Fields 1 Windfarm  | Onshore wind | 724 | 726 | 305 |
| E_BNWKW-1 | Burn of Whilk Windfarm  | Onshore wind | 653 | 520 | 267 |
| T_COUWW-1 | Cour Wind Farm  | Onshore wind | 717 | 674 | 320 |
| T_AKGLW-2 | Aikengall 2 Wind Farm Generation  | Onshore wind | 696 | 389 | 291 |

Pairwise separation 33.4 to 879.8 km.

## 6. Model and backbone revisions

Frozen autogluon/chronos-2-synth at `3607918a9fd027d5c465d8213e46b98e2c041cea`; no fine-tuning, no LoRA.
Adapter: base MLP to 64 d, weather encoder to 64 d per lead, shared fusion MLP,
32 empirical particles per lead from a learned particle embedding. Seeds
[2026090601, 2026090602, 2026090603], 21 fits, 0.00 GPU-hours in total.

## 7. Main results (2021 test, 3-seed mean)

| arm | scaled CRPS | CRPS MW | mean RMSE MW | median MAE MW | 90% coverage | 90% width MW |
|---|---|---|---|---|---|---|
| H | 0.8712 | 26.6243 | 39.2056 | 26.9131 | 0.0021 | 0.2223 |
| M | 0.8279 | 25.4525 | 37.8329 | 25.7414 | 0.0015 | 0.2152 |
| S | 0.8114 | 25.0624 | 37.5253 | 25.3511 | 0.0015 | 0.2111 |
| D | 0.8278 | 25.4483 | 37.8248 | 25.7369 | 0.0014 | 0.2146 |
| P | 0.8287 | 25.4722 | 37.8472 | 25.7604 | 0.0021 | 0.2158 |
| P_BROKEN | 0.8286 | 25.4704 | 37.8453 | 25.7586 | 0.0021 | 0.2158 |
| MC | 0.8198 | 25.2250 | 37.8594 | 25.7559 | 0.0141 | 1.7382 |

Mean and median estimands are reported separately and are not mixed.

## 8. Primary paired effects

| kind | contrast | RI % | CI lower | CI upper |
|---|---|---|---|---|
| primary | P over D | -0.109 | -0.186 | -0.047 |
| primary | P over S | -2.128 | -2.854 | -1.490 |
| secondary | P over P_BROKEN | -0.008 | -0.014 | -0.004 |
| secondary | S over M | 1.991 | 1.366 | 2.672 |
| secondary | M over H | 4.975 | 4.366 | 5.683 |
| secondary | P over MC | -1.076 | -1.304 | -0.865 |

Paired 7-day time-block cluster bootstrap, 2000 replicates over
13 blocks. These are 2000 resamples of one test
year, not 2000 independent samples.

## 9. Seed stability

| contrast | seed | RI % |
|---|---|---|
| P over D | 2026090601 | -0.203 |
| P over D | 2026090602 | -0.037 |
| P over D | 2026090603 | -0.087 |
| P over S | 2026090601 | -0.898 |
| P over S | 2026090602 | -1.919 |
| P over S | 2026090603 | -3.609 |

Three seeds are reported as they are. No confidence interval is built from them.

## 10. Per-farm and per-lead scope

| arm | lead group | scaled CRPS |
|---|---|---|
| D | 30-48h | 0.8370 |
| D | 54-72h | 0.7905 |
| D | 6-24h | 0.8558 |
| H | 30-48h | 0.8884 |
| H | 54-72h | 0.8259 |
| H | 6-24h | 0.8994 |
| M | 30-48h | 0.8373 |
| M | 54-72h | 0.7905 |
| M | 6-24h | 0.8559 |
| MC | 30-48h | 0.8299 |
| MC | 54-72h | 0.7783 |
| MC | 6-24h | 0.8513 |
| P | 30-48h | 0.8385 |
| P | 54-72h | 0.7916 |
| P | 6-24h | 0.8559 |
| P_BROKEN | 30-48h | 0.8384 |
| P_BROKEN | 54-72h | 0.7915 |
| P_BROKEN | 6-24h | 0.8559 |
| S | 30-48h | 0.8206 |
| S | 54-72h | 0.7743 |
| S | 6-24h | 0.8393 |

| farm | arm | scaled CRPS |
|---|---|---|
| E_BNWKW-1 | D | 0.7484 |
| E_BNWKW-1 | P | 0.7498 |
| E_BNWKW-1 | S | 0.7310 |
| E_BURBO | D | 0.7756 |
| E_BURBO | P | 0.7769 |
| E_BURBO | S | 0.7579 |
| E_GFLDW-1 | D | 0.6616 |
| E_GFLDW-1 | P | 0.6623 |
| E_GFLDW-1 | S | 0.6492 |
| T_AKGLW-2 | D | 0.8026 |
| T_AKGLW-2 | P | 0.8042 |
| T_AKGLW-2 | S | 0.7923 |
| T_BEATO-1 | D | 0.9216 |
| T_BEATO-1 | P | 0.9231 |
| T_BEATO-1 | S | 0.9055 |
| T_COUWW-1 | D | 0.7939 |
| T_COUWW-1 | P | 0.7931 |
| T_COUWW-1 | S | 0.7558 |
| T_GANW-13 | D | 0.8645 |
| T_GANW-13 | P | 0.8663 |
| T_GANW-13 | S | 0.8574 |
| T_RMPNO-1 | D | 1.0539 |
| T_RMPNO-1 | P | 1.0537 |
| T_RMPNO-1 | S | 1.0420 |


## 11. Latency and memory

| arm | batch | adapter-only ms | end-to-end ms | adapter peak MB |
|---|---|---|---|---|
| H | 1 | 0.777 | 39.563 | 498.700 |
| H | 64 | 0.788 | 157.989 | 519.400 |
| M | 1 | 1.452 | 41.065 | 498.800 |
| M | 64 | 1.520 | 158.533 | 519.500 |
| S | 1 | 1.879 | 40.599 | 498.800 |
| S | 64 | 2.022 | 159.514 | 519.500 |
| D | 1 | 1.399 | 38.829 | 498.800 |
| D | 64 | 1.438 | 158.938 | 519.500 |
| P | 1 | 1.471 | 41.221 | 499.400 |
| P | 64 | 1.735 | 159.960 | 577.400 |
| P_BROKEN | 1 | 1.466 | 40.719 | 499.400 |
| P_BROKEN | 64 | 1.501 | 159.724 | 577.400 |
| MC | 1 | 73.273 | 115.859 | 498.900 |
| MC | 64 | 78.474 | 236.024 | 524.300 |

Adapter-only and end-to-end are separate costs and are never added together. MC
pays for every scenario call and for building the mixture.

## 12. Path-breaking control

P_BROKEN uses the same architecture as P, trained and evaluated on data whose
member index is permuted independently at every lead from a hash of
(example_id, lead, fixed seed). Every per-lead marginal is preserved bit for bit:
the sorted member values differ by 0.0e+00,
while the linkage itself moves by 5.315.

A gap between P and P_BROKEN says the cross-lead linkage carried signal. It says
nothing causal about the physics.

## 13. Integrity tests

T01 to T13 all pass; see `integrity_tests.json`. Whole-path permutation
invariance of P re-checked on the trained checkpoint:
5.96e-08 against a 1e-6 tolerance.
D and P carry 65,921 parameters each.

## 14. Known deviations from the pre-registration

- Repository. The instruction named `covariate-trust-pilot`; the prior state it
  refers to and this topic's note live in `mltimeseries`, and the user confirmed
  the move before any work started.
- Farm eligibility is stated in usable origins rather than 30-minute coverage, and
  adds a 2020/2021 availability floor. Reason and numbers in
  `farm_selection_rule.json`; without it, farms whose metering stops on
  2020-12-31 are selected and contribute nothing to the test year.
- One source metadata row (`E_BURBO`) was relabelled where the file contradicts
  itself.
- The weather channel set adds `ws10`, a deterministic transform of two channels
  the author already uses, identical across all arms.
- MC sees one member per example per epoch from a fixed subset of 8, cycling, so
  its gradient budget matches the other arms exactly.

## 15. Novelty boundary

See `novelty_audit.md`. Both halves of the P architecture already exist in the
ensemble postprocessing literature: permutation-invariant pooling over members
(Höhlein et al. 2024; PoET 2024) and a temporal encoder applied per member
(Roberts, arXiv 2026-02). Their composition for conditioning a downstream
forecaster was not found, but the gap is small. Nothing here licenses
NOVEL_METHOD_CONFIRMED.

## 16. Interpretation

The contrast measures whether a representation that keeps member trajectories
scores better than one that does not, under matched capacity. It does not show
that any model understands uncertainty, and it supports no causal claim.
Efficiency results are efficiency results and are not restated as accuracy.
A confidence interval that contains zero is not evidence of equivalence.

## 17. What was not tested

Eight sites in Great Britain, two years of training and validation, one test year.
No spatial graph, no attention between farms, no fine-tuning of the backbone, no
2022-2023 data, no other region, no other foundation model, no deployment latency
audit, no alternative particle counts, no hyperparameter search.

## 18. Next action

Determined by the verdict token above and by `verdict.json`. If the token is not
an expand token, the instruction is explicit: do not rescue the topic by bolting
on a heavier path encoder.

---
Branch `uncertain-covariate-path-pilot-v1`, base `36b01d2b84df`,
origin/main `36b01d2b84df`. Integrity all_passed =
True.
