# Controlled PEFT failure-condition screen

[Plan](../../_docs/notes/tsfm_topics/08_peft_shift_mechanism_plan_20260908.md) fixes four Gaussian lagged-triangular conditions, three independent training corpora, 64/128/512 train/validation/evaluation episodes, Y-only supervision, and eight FM configurations. This is a development screen on one backbone, not a new method or a claim about the model's pretraining distribution.

Roles: `data.py` creates paired-condition independent episodes and QC; `train.py` runs a single trial; `run_study.py` selects two learning rates on corpus0 validation and repeats the chosen rate on corpora1/2; `raw.py` provides episode-OOF ridge and F0 residual baselines; `analyse.py` recomputes scores, verifies selection/input/source contracts, and reports conditional paired uncertainty.

Run from the repository root in the existing `.venv-peft` environment:

```powershell
& '.\.venv-peft\Scripts\python.exe' -m experiments.peft_shift_mechanism_v1.data --output runs/peft_shift_mechanism_v1/data
& '.\.venv-peft\Scripts\python.exe' -m experiments.peft_shift_mechanism_v1.run_study --smoke
& '.\.venv-peft\Scripts\python.exe' -m experiments.peft_shift_mechanism_v1.run_study
& '.\.venv-peft\Scripts\python.exe' -m experiments.peft_adaptation_scope_v1.guard --output runs/peft_shift_mechanism_v1/analysis_guard --cpu-only --timeout-seconds 180 -- .venv-peft/Scripts/python.exe -m experiments.peft_shift_mechanism_v1.analyse
```

Do not regenerate data while trials run. The runner locks its own study, and every GPU trial shares the original S1 guard lock. It resumes completed trials only under matching source/plan/data contracts. A contract mismatch stops; preserve old artifacts and resolve the cause before starting a new run. Existing completed trial files are not overwritten. The training contract hashes the trainer, data generator, runner and imported S1 sources. Analysis source hashes are recorded separately so a reporting fix does not alter the training contract.

S0 completed eight guarded methods successfully. The zero-update difference between cache and direct four-group paths was 0, joint/LP initialization and sampler hashes matched, LP performed both stages, and frozen parameters/checkpoint restoration were verified. These checks do not establish forecasting quality. The BF16 U/V swap diagnostic recorded a nonzero rounding-path difference; exact channel permutation invariance follows from the architecture and is distinct from bitwise numerical invariance.

The full design contains 112 adaptation fits plus 12 F0/cache jobs. The final selection has 96 FM entries, plus 24 RAW/F0_RAW entries and 12 analytic-oracle entries. Future drivers are never observed. RAW has a known candidate-lag dictionary and named channels; the FM does not have fixed channel-ID embeddings. Both differences are explicit interpretation limits.

All model calls use 4 complete groups, 12 encoder rows and 4 Y probe rows. Effective training batch is 8 groups via two microbatches. BF16 autocast weight caching is disabled. CPU threads are 2/interop1; CUDA allocation cap is .67. No loader workers, model deepcopy or compilation. The guard monitors RAM/commit/Git/GPU resources and owns only its child job. This reduces observed resource risks without guaranteeing that OS/driver failures cannot occur.

The primary score is raw mean **2-pinball**, not standard WQL or the earlier real-data fit-std-scaled score. Each family uses its stated Bonferroni confidence level; this is not a single joint error guarantee across all exploratory outputs. Evaluation bootstrap intervals are conditional on the three fitted corpus/optimizer repetitions. See the plan for practical-effect and stopping rules.

The separate [diagnostic addendum](../../_docs/notes/tsfm_topics/08_peft_shift_mechanism_diagnostic_addendum_20260908.md) fixes a no-training FP32/BF16 permutation check, causal lag-aligned known-future-covariate inference, and descriptive projection of forecasts on the true mean components. These are exploratory follow-ups; the original training contract stays unchanged. In the alignment diagnostic, all driver values come from the original past, but retiming, missing early context, normalization and use of the future-covariate path change together. The candidate lag dictionary contains the true lag.

After the main runner has exited and released its lock, run these commands **sequentially**. Never start another guard while the main runner is active. The alignment script saves validation choices before its first evaluation call and refuses to overwrite a previous diagnostic directory.

```powershell
& '.\.venv-peft\Scripts\python.exe' -m experiments.peft_adaptation_scope_v1.guard --output runs/peft_shift_mechanism_v1/precision_guard --timeout-seconds 180 -- .venv-peft/Scripts/python.exe -m experiments.peft_shift_mechanism_v1.diagnose_precision --output results/peft_shift_mechanism_v1/precision_diagnostic.json
& '.\.venv-peft\Scripts\python.exe' -m experiments.peft_adaptation_scope_v1.guard --output runs/peft_shift_mechanism_v1/alignment_guard --timeout-seconds 300 -- .venv-peft/Scripts/python.exe -m experiments.peft_shift_mechanism_v1.diagnose_alignment --output results/peft_shift_mechanism_v1/alignment_diagnostic
& '.\.venv-peft\Scripts\python.exe' -m experiments.peft_adaptation_scope_v1.guard --output runs/peft_shift_mechanism_v1/analysis_guard --cpu-only --timeout-seconds 180 -- .venv-peft/Scripts/python.exe -m experiments.peft_shift_mechanism_v1.analyse
& '.\.venv-peft\Scripts\python.exe' -m experiments.peft_shift_mechanism_v1.plot
```

`analyse.py` verifies the full trial/selection/source/data contracts, recomputes scores from saved predictions, and checks that selected step-zero predictions equal F0. It writes conditional effects, costs and sampled resource summaries. `plot.py` requires successful numerical verification and exports PNG/PDF score comparisons and validation trajectories. Execution completion and scientific conclusions are recorded only after the corresponding result files exist.
