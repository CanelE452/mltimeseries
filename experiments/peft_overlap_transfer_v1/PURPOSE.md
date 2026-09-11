# Past-only LoRA decision: a frozen future-period pilot

Authorized by the user's 2026-09-10 request to complete steps 1–3. Conditional on the frozen optimization-control gate passing. This is a simple decision-rule pilot, not a learned complexity estimator, a new adapter, or causal identification.

## Hypothesis and development boundary

Use the six existing source/condition cells only as development observations. Seeds are optimization repetitions, not independent datasets. FULL90 and SPREAD30 have identical deduplicated context-time unions, so a deterministic descriptor of that union alone cannot distinguish them. A training-design descriptor can: overlap ratio = selected origin count × 48 / unique target timestamp count, before missing-label masks. It equals approximately 1.978/1/1.935 for FULL90/SPREAD30/RECENT30. Repeated target times at different origins have different leads/contexts and are not identical examples.

Freeze OVERLAP: use ALL (rank-8 LoRA + the same MLP) if ratio >= 1.5, otherwise MLP. This threshold is a development hypothesis, chosen between the observed designs; it is not statistically fitted or independently validated. It may simply encode the sampling condition. Controls: ALWAYS_ALL, ALWAYS_MLP, and COUNT (ALL iff origins >= 60). No feature/threshold changes after future evaluation. No calibration labels or residual correction.

## Future data and adaptation

- Reuse immutable study20 preparation implementation with new panel starts: Bike 2012-04-06; Household 2008-08-30. Same 242-day template, context336h, horizon48h, daily origins, 90/30/20/80 train/V/cal/E origins, 48h train/V and V/cal embargoes.
- Bike E: [2012-09-14, 2012-12-04); Household E: [2009-02-07, 2009-04-29). E target times must be disjoint from the previous study20 E. Bike's past fit/V/context reuse some former development E observations; the new E targets alone are locally unseen in the bounded audit. Foundation pretraining overlap is unknown. Both sources are familiar, not independent new sources.
- Causal forward fill and precontext-only fallback; train-only scaling; preserved target missing masks. Fixed >=70% target observation QC. Preparation may check missingness, not choose periods by future forecast outcomes.
- Two sources × three conditions × two arms × seeds26000/26001 = 24 fits. Reuse the prior EXPOSURE V-selected learning rate separately for source/condition/arm. No future LR search.
- Reuse immutable optimization-control fit.py. Only EXPOSURE schedules: FULL90 [0,15,30,60,120,180], subsets [0,5,10,20,40,60]. Train exactly180 or60 updates respectively, batch8/micro4. Select checkpoint strictly on V, including step0. Six opportunities each, origin exposure matched; unique supervision, context, total parameters and optimization paths remain unequal.
- Seal all24 selected checkpoint hashes before any future E forecast. Then24 selected E forecasts plus2 F0 forecasts. MLP589301 vs ALL1768949 parameters. All initial heads/sample streams matched within condition/seed. Frozen backbone and exact checkpoint replay verified.

## Prespecified practical decision criteria

Regret = 100 × (rule score − ALWAYS_ALL score) / F0 score; positive is worse. For each seed separately, OVERLAP must have macro regret over six cells <=0.25%F0, each source's mean over three conditions <=0.5%F0, and measured counterfactual fit-time savings >5% vs ALWAYS_ALL. These are practical pilot thresholds, not statistical noninferiority tests. Show every cell and seed, including losses. If COUNT or ALWAYS_MLP meets these accuracy limits at lower measured cost, prefer the simpler/cheaper control; OVERLAP has no unique decision benefit. Report the OVERLAP-versus-COUNT comparison regardless.

Cost is complete serial fit subprocess seconds, including model load, validation and checkpoint replay, excluding queue/admission and E evaluation. It is a retrospective counterfactual deployment estimate from fitting both arms offline, not savings actually realized by the research campaign. Report campaign elapsed/GPU job counts separately. No uncertainty claim from two seeds. Quantile-loss score independently recomputed from saved predictions.

## Execution and stopping

Use the existing serial guard unchanged: admission RAM>=5GiB, commit>=13GiB twice; running RAM>=5GiB/commit>=6GiB, GPU<=10500MiB, temperature<=85C, child900s. CPU preparation uses the same guard without requiring GPU. Preserve failed attempts; no OS/driver/security changes, threshold relaxation, or broad process cleanup. Estimated additional serial work 30–45 minutes plus data preparation/admission, to be updated from measured fit times.

Original files/results remain immutable. Store new code/run/results separately. If the step1 gate fails, do not run this dependent pilot. If the pilot fails, reject this simple selection rule for the tested future blocks; do not conclude all PEFT fails. Even success requires independent sources, more time blocks and capacity-matched controls before a method or paper claim.
