# Study35: bounded head optimization, then conditional component interventions

Approved sequence: user said '그렇게해줘' after Study34's convergence -> frozen-backbone head refit -> measured mechanism proposal. Purpose: determine whether initial LoRA gains survive a substantially larger head optimization budget, before claiming a time-series representation limitation or designing a new adapter.

## Choices and alternatives

Keep Chronos-2, native 21-quantile pinball, L336/H48, 2 targets/2 covariates, AdamW wd0/foreachFalse, batch8/micro4/clip1/threads2 and identical initial head/sample streams. This isolates budget and component ownership rather than changing model/loss. HEAD 589301 trainable; WIDE and JOINT 1768949 each. Same four LR recipes as Study34. Equal counts are not equal function classes or exhaustive optimizer tuning.

Focus FULL90 on both sources, because initial LoRA beat strong comparators in both in Study34 and one Jena head selected the 180-step cap. This is a targeted diagnostic; SPREAD30 and data-volume effects are outside this run. Final data-only decision: Jena2018 May4 and BMRA2017 May4 242-day blocks. Jena2020 was rejected for earlier weather-pilot target exposure; BMRA2017 Jan4 and May4 with the original target pair failed the unchanged70% target-window coverage gate. BMRA keeps the same four series but uses E_BRYBW-1/E_BURBO as targets and E_DALSW-1/E_BNWKW-1 as context-only covariates. This changes the target pair and season, so it is not a same-unit replication. Only raw availability/history were used for these choices before GPU or model outcomes. Same existing sources, retrospectively earlier periods, not new sources/future generalization. FM pretraining overlap unknown; inherited four-series pool was selected using later-period availability, and target roles now condition on the proposed-period availability. No replacing these final periods/roles after model outcomes.

Alternatives: increase only head capacity (already controlled and would mix capacity with optimization); create a controller (Study34 oracle margin too small); transplant/reset a head on learned backbone (useful intervention below, analogous to disentangling coupled components, but still does not prove intrinsic representational necessity). No fabricated novel complexity metric.

## Stage A: bounded convergence screen

2 sources x 2 seeds (30000/30001) x HEAD/WIDE/JOINT x 4 LR =48 trajectories. Each 720 steps, checkpoints [0,1,2,4,8,15,30,60,120,180,240,360,540,720]. Save V predictions and train diagnostic scores at each checkpoint. Simultaneously retain V-best S180 (checkpoints <=180) and L720; these share the identical first180 updates. Select each budget's best recipe on V, ties earlier step then recipe. Preserve step0. Train diagnostic scores do not influence selection; no claim mathematical convergence from a plateau.

HEAD/WIDE rates [1e-5,3e-5,1e-4,3e-4]. JOINT (head,LoRA) [(1e-5,1e-5),(3e-5,3e-5),(1e-4,3e-5),(1e-4,1e-4)]. Same train-only bias/positive-affine correction candidates as Study34, V-selected. Seal all48 fits and both budget selections before D forecasts. D20 every fourth prepared eval origin, cal unused. Forecast S180/L720 per selected family (24 forecasts, even identical selections are explicitly verified). F0 from HEAD forecast. Compare L720 JOINT to min(F0,L720 HEAD,L720 WIDE,CORRECTION), normalized by F0.

Entry G1: >0.25%F0 in both seeds of at least one source. Also report S180 vs L720, all cap choices, V and train curves, final and selected model differences, costs, adverse sources. This is a pragmatic development gate, not statistical significance or a paper threshold. If G1 fails, stop the intervention branch and report whether the gap vanished on the new period or head caught up; these explanations must be separated using the within-period S180 comparison.

## Stage B: conditional, prespecified interventions

Only if G1 passes, execute all four source/seed cells, not just winners. Dependency checkpoint selection uses sealed L720 V choices; no D tuning. All branch recipes, seed streams, budgets and decisions frozen here before Stage A.

REFIT: reconstruct JOINT, load only its selected L720 LoRA weights, freeze them and the native model; use exactly the original HEAD initialization, identical samples and four HEAD LRs for720 steps. Original Stage-A HEAD trajectories are the matched F0-backbone refit control (no need to train them twice). Preserve initial/selected output and immutable adapter hash. This measures the value of a fixed adapted backbone plus its native output and a newly trained residual head; it does not isolate hidden representation from the native output or prove optimality over all heads.

WARM_HEAD/WARM_JOINT: start from the identical selected L720 HEAD; WARM_JOINT adds zero-output LoRA, WARM_HEAD keeps the original backbone. Both reset AdamW state, use same seed+200000 origin stream and the same four rates as HEAD/JOINT above, for720 additional steps. Step0 remains selectable. Initial V prediction must equal source HEAD selected V exactly. Report preprocessing/prefix, search and branch costs; this is not a free improvement. All four branches within a family get equal budget; no A/B LoRA-rate tuning. Joint global clipping couples head/adapter gradients, so any benefit can include optimization geometry.

16 REFIT +16 WARM_HEAD +16 WARM_JOINT =48 additional fits, each one V-best L720; seal all before12 D forecasts. D is already a development diagnostic used for entry, so these are adaptive mechanistic diagnostics, not an untouched test. Keep all source/seed cells in reporting. Stage-A smoke uses train-only pseudo-validation. Stage-B three-step smoke uses the already-used full V to verify equality to its sealed prefix; it cannot change selection rules and does not access D.

G2 refit: REFIT beats min(F0,HEAD) >0.25%F0 in both seeds of at least one source. G3 warm: WARM_JOINT beats min(F0,WARM_HEAD) >0.25%F0 in both seeds of at least one source. Report whether the SAME source passes G1/G2/G3. If either fails, do not silently tune another adapter; report which cause remains. Even simultaneous passes only warrant separately planned measurable-signal research. No automatic new method creation or final test in this run.

## Statistics, verification, operations

Data QC clarification before freeze: the inherited helper named min_target_window_finite_fraction actually takes the minimum of aggregated split metrics. In addition to that unchanged aggregate gate, Study35 independently requires >=70% finite scored cells in EVERY individual48h origin window, and reports per-target minima. The original BMRA May2017 pair failed this stronger per-window preflight check, even though its split-average coverage was high. This is a pre-outcome stricter data check, not a relaxed gate.

Save per-origin/target/quantile loss numerators, valid counts and scaling for all reported models. Conditional90% intervals:2000 paired circular block resamples length2 origins, same weights across seeds/methods, F0 resampled, fixed source periods. Four cells are not four independent datasets. Exact source/head/sample and frozen-state assertions, independent score/selection audits, preserve failed attempts, no outcome-based source edits. All timestamps retain fractional seconds.

Admission unchanged: available commit>=13GiB,RAM>=5GiB; runtime emergency commit>=6/RAM>=5, guard unchanged. Single GPU, per-job timeout900sec, no threshold relaxation, no OS/security modifications/unrelated kills/deletion/commit/push. Stage results and self-monitoring progress persisted, parent continues through bounded branch and final analysis.

## Prior work

- [Kumar et al., ICLR2022](https://openreview.net/pdf?id=UYneFzXSJWh): probing before fine-tuning motivates warm-start control; image classification findings are not established here.
- [Hayou et al., ICML2024, LoRA+](https://proceedings.mlr.press/v235/hayou24a.html): optimization can explain LoRA differences; our control is not LoRA+.
- [Chronos official implementation](https://github.com/amazon-science/chronos-forecasting): existing LoRA support; applying it is not novelty.
