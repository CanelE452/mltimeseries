# PEFT adaptation scope: S0 and S1

This implements the first two stages of `_docs/notes/tsfm_topics/04_peft_adaptation_scope_experiment_plan.md`.
The question is whether updating Chronos-2 internally provides value beyond a strong output/readout adaptation. S1 does not identify temporal versus inter-variable shift causes.

## Data and comparison

- ETTm2: 7 channels, context 384, horizon 96.
- Jena 2024: 21 channels, context 576, horizon 144.
- Each panel uses 116 consecutive days: 4 days context, 64 fit, 16 validation, 32 development evaluation.
- Fit origins have stride 16; validation and evaluation labels do not overlap.
- Input features contain only values observed before each origin. Each origin/sample instance has its own native Chronos group.
- Primary score is fit-standard-deviation-scaled 2-pinball, averaging valid cells within each target, then targets equally. This is not conventional WQL.
- F0, AFF, H_LIN, H_MLP, H_FULL, OFF_LORA, FULL use the same quantile grid and native training normalization. RAW ridge is a separate supervised estimator.
- Seed 0 selects among three learning rates per method using validation only. Seeds 1 and 2 reuse the selected rate. Each training trial has at most 500 updates and includes the unadapted step 0 checkpoint.
- Effective batch is eight complete panel groups. Microbatching never splits channels of one group.
- Deterministic backbone dropout 0, adapter dropout 0, AdamW weight decay 0, gradient clipping 1.0, float32 weights and inverse normalization, BF16 autocast.
- Checkpoint: locally cached `amazon/chronos-2`, revision `29ec3766d36d6f73f0696f85560a422f50e8498c`.

## Safety and execution

Use `.venv-peft/Scripts/python.exe`. This environment adds `peft==0.20.0` while sharing the pre-existing ML packages. Do not invoke `train.py` without the guard.

The study runner executes one trial at a time. A study-wide lock prevents duplicate runners. Every trial gets a separate subprocess; checkpoints, cache and logs go beneath the Git-ignored `runs/peft_adaptation_scope_v1/` directory.

The guard stops its own child tree if available RAM drops below 5 GiB, available commit below 6 GiB, the child tree RSS exceeds 8 GiB, Git count exceeds 32, GPU memory use exceeds 10,500 MiB, or temperature reaches 85 C. GPU observations are sampled every ten seconds and the other resources every two seconds; they cannot prevent every sudden driver or OS failure. A Windows Job Object releases the owned children if the guard exits. No unrelated application, driver or security setting is changed.

```powershell
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$env:OPENBLAS_NUM_THREADS = '2'
$env:PYTHONIOENCODING = 'utf-8'
& .\.venv-peft\Scripts\python.exe -m experiments.peft_adaptation_scope_v1.data
& .\.venv-peft\Scripts\python.exe -m experiments.peft_adaptation_scope_v1.run_study --stage smoke --micro-groups 4
& .\.venv-peft\Scripts\python.exe -m experiments.peft_adaptation_scope_v1.run_study --stage s1 --micro-groups 4
& .\.venv-peft\Scripts\python.exe -m experiments.peft_adaptation_scope_v1.analyse
& .\.venv-peft\Scripts\python.exe -m experiments.peft_adaptation_scope_v1.verify_results
```

These are sequential commands, not a command to launch a second copy while the study is active. The recorded S1 contract uses micro-groups 4 and effective-groups 8. A resume must retain that contract. Analysis also requires the separate RAW results under `runs/peft_adaptation_scope_v1/raw/`; the preserved `raw_rescue/` directory records the single additional regularization candidate and its validation-only selection. Run analysis before verification so the verifier can reconcile the final CSV with the selected trial artifacts.

The BF16 autocast weight cache is disabled consistently in frozen feature generation and training. A preflight diagnostic found that enabling it changed the zero-update output when parameter `requires_grad` changed. The corrected path reproduced the frozen prediction without widening the identity tolerance. The successful preflight artifacts have the suffix `_ampnocache_micro4`.

S0 runs five updates per internal method and panel, keeping the real L/H and all channels. It audits module counts, frozen tensors, initial identity, group isolation and future masks. It is not a performance result. The runner requires a successful S0 summary before S1 and preserves the source/configuration contract on resume.

Monitor `runs/peft_adaptation_scope_v1/progress.json`, each trial's `progress.jsonl`, and its `guard/status.json` / `guard/resource_log.jsonl`. A nonzero child exit, safety stop or missing metrics is a failed trial, not a completed result. Inspect a failed trial before retrying; do not overwrite its artifacts.

## Interpretation

ETT/Jena are development data already used in this project. Three optimizer seeds do not create three independent domains. The analysis selects the output/head comparator on validation and reports paired differences to OFF_LORA/FULL. Circular day-block bootstrap lengths 1, 3 and 7 are conditional on the observed trained seeds and 32 evaluation days. A small or uncertain effect is not proof that every internal adaptation is unnecessary. Boundary learning rates or late-selected checkpoints indicate possible optimization limits.

The time/group interaction study (S2), new source-family confirmation and independent architecture (S3) remain separate stages. Do not infer their outcomes from S1.

## Completed run: 2026-09-08

All 62 S1 trials completed in 68 minutes 30 seconds. Analysis produced 40 selected comparison rows and four paired effects; `verify_results.py` passed. The A1 conclusion is inconclusive: ETT LoRA showed a positive mean difference versus the validation-selected readout, while Jena showed a negative mean difference, and both LoRA comparisons included zero under every registered day-block length.

See the [Korean results report](../../_docs/notes/tsfm_topics/04_peft_adaptation_scope_s1_results_20260908.md) and `results/peft_adaptation_scope_v1/verification.json`. A separate inference-only diagnostic reproduced the small difference between cached F0 and a selected FULL step0 checkpoint: cache generation used one group per encoder call, whereas direct evaluation used four. The S0 identity audit covers one matching group, not bitwise identity across all batch shapes. Preserve these original results; match cache/direct evaluation batch contracts in a future confirmation run.
