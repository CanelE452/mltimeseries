# Study34: initial adaptation headroom, before another controller

2026-09-11. User approved the next direction after Study33 closure. This is a development-only diagnostic, not a new PEFT method or a confirmatory test. No Study33 E retuning and no new final-test labels are permitted.

## Purpose and alternatives

Paper objective -> identify a reproducible limitation of frozen features -> first establish that tuned initial adaptations actually differ in their selected outputs. Audience: researcher deciding whether to invest in causal interventions and a new method.

Candidate A (chosen first): tune the existing head and LoRA fairly, compare actual V-selected outputs on a new development diagnostic period D. This addresses missing headroom directly. Candidate B: move the old probe earlier. It could help because Study33 selected checkpoints preceded its fork, but this still presumes useful alternatives and is deferred. Candidate C: transplant an independently learned head onto an adapted backbone, analogous to a component swap in a factorial experiment. This can expose coadaptation, but off-manifold combinations are not a proof of representational necessity; defer until A passes.

## Fixed design

- Existing cached Chronos-2 checkpoint, L336/H48, 21-quantile pinball, two targets and two context covariates. These keep the question comparable to Study33 without adding backbone/token/loss confounds. Raw periods and availability/exposure limitations are recorded in PURPOSE_DATA.md and prepared summary.
- Two source panels, FULL90 and SPREAD30 optimizer-origin pools, seeds 29000 and 29001. These are eight paired cells but only two source periods; seeds and conditions share target labels. SPREAD30 is not a strict 30-label regime because past context and train statistics are shared.
- HEAD: frozen native backbone + residual MLP533 (589301 trainable). WIDE: frozen native backbone + width1601 partially biased residual MLP (1768949 trainable). JOINT: same narrow head + rank8/alpha16/dropout0 attention LoRA (1768949 trainable). Equal counts do not imply equal function classes.
- Four trials each. HEAD/WIDE LR = [1e-5,3e-5,1e-4,3e-4]. JOINT (head,LoRA) LR = [(1e-5,1e-5),(3e-5,3e-5),(1e-4,3e-5),(1e-4,1e-4)]. Same number of trials, not a comprehensive optimization claim. JOINT recipes 2/3 isolate LoRA LR with fixed head LR. Compare HEAD recipes 0/1/2 to JOINT with matching head LR as additional diagnostics.
- AdamW weight_decay=0, foreach=False; global norm clip1; batch8/micro4; same sampled origins for paired methods at each seed. Frozen original backbone must hash identically before/after. Small-head initial weights must match across HEAD and JOINT for the same seed.
- FULL90 max180 updates, validation at [0,1,2,4,8,15,30,60,120,180]. SPREAD30 max60, validation at [0,1,2,4,5,10,20,40,60]. All families get identical checkpoint opportunities. Final returned model is the lowest-V checkpoint including step0; ties prefer earlier step then recipe ID. Full trajectories are diagnostics, not an efficient early-stop implementation.
- Additional baseline CORRECTION: fit positive affine location/scale on F0 train forecasts by least squares, plus mean-bias alternative. Slope clipped to [0.25,4], ridge=1e-6, and shrinkages [0,.25,.5,1]. V chooses among seven distinct candidates including identity. Affine transformation acts on every quantile equally and preserves order. This baseline has a different, very small hypothesis class and does not claim pinball-optimal fitting. Train forecasts use the same origin pool as the paired neural methods; no V or D labels enter coefficient fitting.
- Fit all 96 trajectories, seal per-cell/per-family best recipe+checkpoint and correction coefficients before reading D predictions/losses. D uses every fourth prespecified eval origin (20 origins); the archive's eval split is explicitly renamed D for this study. The cal split is unused.
- F0, HEAD, WIDE, JOINT, CORRECTION predictions on D are paired. Report D/F0 and improvements as %F0; save per-origin pinball numerators/counts so scores can be independently recomputed.

## Prespecified decisions

Correction coefficients use centered least squares in train-standard-deviation units, with 1e-6 added to predictor variance. G2 chooses its hindsight global and source-fixed comparators separately within each seed. Descriptive 90% intervals use 2000 paired circular block resamples of two D origins, seed/condition rows kept together, and recomputed F0 denominators; these are conditional on the two periods.

G1, a useful initial-LoRA gap: JOINT beats min(F0,HEAD,WIDE,CORRECTION) by >0.25%F0 in both seeds of at least one of four source/condition cells. This hindsight strong comparator is diagnostic only, not a deployable selector. If G1 fails, do not build a controller or claim head-inaccessible structure from these data. Close this screen and document the remaining uncertainty.

G2, controller headroom: hindsight per-cell oracle over the five V-selected families improves over BOTH hindsight best global family and hindsight best source-fixed family by >0.25%F0 in each seed. These hindsight comparisons are deliberately stronger than a deployable baseline; oracle itself is not a method. If G1 passes but G2 fails, study a fixed adaptation/mechanism, not a controller. If both pass, only then consider transferable signals.

G3, broader limitation check: disclose LR response, step0 choices, exact prediction equality, source sign reversals and actual measured trajectory time. This is descriptive, not an additional pass criterion. Any confidence intervals resample origin blocks with seeds/conditions kept together and are conditional on these two periods.

One fixed grid only. No expanding LR, changing the .25 threshold, changing periods/channels after outcomes, or tuning another policy on D in this run. A positive screen warrants a separate, preregistered mechanism intervention; it does not itself authorize a paper claim. A negative screen is a completed outcome, not a reason to silently search until positive.

## Prior work and distinction

- Kumar et al., ICLR 2022, [Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution](https://openreview.net/pdf?id=UYneFzXSJWh): motivates separating fixed readout from feature adaptation. Its image/classification findings are not established time-series mechanisms.
- Hayou et al., ICML 2024, [LoRA+](https://proceedings.mlr.press/v235/hayou24a.html): adapter optimization matters, including distinct A/B rates. Our head/LoRA grid is an optimization control, not an implementation or improvement of LoRA+; A/B-ratio tuning remains untested.
- [Official Chronos repository](https://github.com/amazon-science/chronos-forecasting): native time-series model and existing LoRA adaptation support. Adding LoRA alone is not novel.

## Execution contract

Single GPU guard, admission available commit >=13GiB and RAM>=5GiB, existing emergency limits unchanged. Preserve failed attempts. S0 uses train-only pseudo-validation and tests all three families at small steps. Freeze plan/source/input hashes before S0; seal all model-selection results before D. No OS/security edits, unrelated process termination, deletion, commit or push. Record completed checks, graphs, limitations and next decision in note34 and existing history.
