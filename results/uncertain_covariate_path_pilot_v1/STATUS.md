# UCP-PATH-PILOT-v1 — STATUS

## 1. Executive verdict

**NO_INCREMENTAL_PATH_VALUE**

P over D: RI -1.19% (95% CI -2.25 to -0.27), 0 of 3 seeds positive.
P over S: RI -2.05% (95% CI -3.73 to -0.56), 0 of 3 seeds positive.

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

Panel sizes: {"train": 5547, "val": 4899, "test": 1362}.

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
[2026090601, 2026090602, 2026090603], 21 fits, 0.10 GPU-hours in total.

## 7. Main results (2021 test, 3-seed mean)

| arm | scaled CRPS | CRPS MW | mean RMSE MW | median MAE MW | 90% coverage | 90% width MW |
|---|---|---|---|---|---|---|
| H | 0.5511 | 14.6958 | 32.1050 | 23.6299 | 0.8923 | 82.2220 |
| M | 0.2995 | 7.8221 | 18.3108 | 10.8817 | 0.8433 | 37.4774 |
| S | 0.3010 | 7.7771 | 18.2098 | 11.1339 | 0.8331 | 36.5683 |
| D | 0.3035 | 7.8838 | 18.4936 | 11.0633 | 0.8389 | 37.7319 |
| P | 0.3072 | 7.9518 | 18.7062 | 11.0906 | 0.8195 | 36.4393 |
| P_BROKEN | 0.3070 | 7.9497 | 18.6526 | 11.0754 | 0.8186 | 36.3983 |
| MC | 0.3066 | 7.9211 | 18.5903 | 11.1839 | 0.9355 | 51.9894 |

Mean and median estimands are reported separately and are not mixed.

## 8. Primary paired effects

| kind | contrast | RI % | CI lower | CI upper |
|---|---|---|---|---|
| primary | P over D | -1.194 | -2.252 | -0.270 |
| primary | P over S | -2.046 | -3.726 | -0.555 |
| secondary | P over P_BROKEN | -0.052 | -0.257 | 0.173 |
| secondary | S over M | -0.518 | -1.246 | 0.191 |
| secondary | M over H | 45.663 | 41.702 | 48.805 |
| secondary | P over MC | -0.187 | -1.481 | 0.895 |

Paired 7-day time-block cluster bootstrap, 2000 replicates over
46 blocks. These are 2000 resamples of one test
year, not 2000 independent samples.

## 9. Seed stability

| contrast | seed | RI % |
|---|---|---|
| P over D | 2026090601 | -0.492 |
| P over D | 2026090602 | -1.699 |
| P over D | 2026090603 | -1.398 |
| P over S | 2026090601 | -1.496 |
| P over S | 2026090602 | -3.580 |
| P over S | 2026090603 | -1.108 |

Three seeds are reported as they are. No confidence interval is built from them.

## 10. Per-farm and per-lead scope

| arm | lead group | scaled CRPS |
|---|---|---|
| D | 30-48h | 0.3003 |
| D | 54-72h | 0.3375 |
| D | 6-24h | 0.2728 |
| H | 30-48h | 0.5634 |
| H | 54-72h | 0.5761 |
| H | 6-24h | 0.5139 |
| M | 30-48h | 0.2959 |
| M | 54-72h | 0.3337 |
| M | 6-24h | 0.2688 |
| MC | 30-48h | 0.3031 |
| MC | 54-72h | 0.3412 |
| MC | 6-24h | 0.2755 |
| P | 30-48h | 0.3040 |
| P | 54-72h | 0.3415 |
| P | 6-24h | 0.2760 |
| P_BROKEN | 30-48h | 0.3039 |
| P_BROKEN | 54-72h | 0.3409 |
| P_BROKEN | 6-24h | 0.2762 |
| S | 30-48h | 0.2974 |
| S | 54-72h | 0.3362 |
| S | 6-24h | 0.2694 |

| farm | arm | scaled CRPS |
|---|---|---|
| E_BNWKW-1 | D | 0.3124 |
| E_BNWKW-1 | P | 0.3211 |
| E_BNWKW-1 | S | 0.3041 |
| E_BURBO | D | 0.3142 |
| E_BURBO | P | 0.3181 |
| E_BURBO | S | 0.3130 |
| E_GFLDW-1 | D | 0.3593 |
| E_GFLDW-1 | P | 0.3617 |
| E_GFLDW-1 | S | 0.3609 |
| T_AKGLW-2 | D | 0.2404 |
| T_AKGLW-2 | P | 0.2443 |
| T_AKGLW-2 | S | 0.2313 |
| T_BEATO-1 | D | 0.3042 |
| T_BEATO-1 | P | 0.3001 |
| T_BEATO-1 | S | 0.2992 |
| T_COUWW-1 | D | 0.3976 |
| T_COUWW-1 | P | 0.4025 |
| T_COUWW-1 | S | 0.4046 |
| T_GANW-13 | D | 0.2330 |
| T_GANW-13 | P | 0.2386 |
| T_GANW-13 | S | 0.2328 |
| T_RMPNO-1 | D | 0.2675 |
| T_RMPNO-1 | P | 0.2709 |
| T_RMPNO-1 | S | 0.2622 |


## 11. Latency and memory

| arm | batch | adapter-only ms | end-to-end ms | adapter peak MB |
|---|---|---|---|---|
| H | 1 | 0.272 | 44.928 | 724.400 |
| H | 64 | 0.707 | 148.585 | 745.600 |
| M | 1 | 1.355 | 36.006 | 724.500 |
| M | 64 | 1.332 | 149.357 | 745.200 |
| S | 1 | 1.774 | 37.901 | 724.500 |
| S | 64 | 1.805 | 150.010 | 745.200 |
| D | 1 | 1.308 | 38.566 | 724.500 |
| D | 64 | 1.375 | 149.955 | 745.200 |
| P | 1 | 1.332 | 35.805 | 725.100 |
| P | 64 | 1.387 | 149.904 | 803.100 |
| P_BROKEN | 1 | 1.303 | 36.296 | 725.100 |
| P_BROKEN | 64 | 1.359 | 149.988 | 803.100 |
| MC | 1 | 66.874 | 107.775 | 724.600 |
| MC | 64 | 70.203 | 217.734 | 750.000 |

Adapter-only and end-to-end are separate costs and are never added together. MC
pays for every scenario call and for building the mixture.

## 12. Path-breaking control

P_BROKEN uses the same architecture as P, trained and evaluated on data whose
member index is permuted independently at every lead from a hash of
(example_id, lead, fixed seed). Every per-lead marginal is preserved bit for bit:
the sorted member values differ by 0.0e+00,
while the linkage itself moves by 6.594.

A gap between P and P_BROKEN says the cross-lead linkage carried signal. It says
nothing causal about the physics.

## 13. Integrity tests

T01 to T13 all pass; see `integrity_tests.json`. Whole-path permutation
invariance of P re-checked on the trained checkpoint:
4.77e-07 against a 1e-6 tolerance.
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
