# Native LoRA module ablation on Q00

Completed 2026-09-08: 12/12 new fits plus 3 S0 trials, numeric verification passed. At fixed LR3e-5, attention deletion costs +6.4615% of F0 [97.5% CI +5.3183%, +7.5690%]; output-projection deletion costs -0.0416% [-0.0517%, -0.0309%], within the prespecified ±1% practical-equivalence band. These are conditional Q00 retraining results, not a new-method contribution. See the [result report](../../_docs/notes/tsfm_topics/09_peft_module_ablation_results_20260908.md) and [numeric artifacts](../../results/peft_module_ablation_v1/verification.json).

The first CPU analysis invocation failed on string episode IDs and its logs remain in `runs/peft_module_ablation_v1/analysis_guard`. The corrected analysis completed under `analysis_guard_verified`; no GPU fit was repeated. Use a fresh guard output path for any future analysis invocation so existing logs are retained. Plot verified outputs with `python -m experiments.peft_module_ablation_v1.plot`.

The [execution plan](../../_docs/notes/tsfm_topics/09_peft_module_ablation_plan_20260908.md) fixes a retraining ablation of the previously completed Q00 OFF_LORA model. OUT_ONLY retains rank-8 LoRA on the final native output projection (27,264 parameters). ATTN_ONLY retains rank-8 LoRA on 96 time/group attention projections (1,179,648 parameters). Neither adds a residual head. F0 and BOTH reuse the verified parent study's original results.

Primary comparisons fix LR=3e-5, the previously selected BOTH rate. Secondary comparisons choose between 3e-5 and 1e-4 on corpus0 validation, then use that rate for all three corpora. All 12 new fits (2 subsets × 2 rates × 3 corpora) are counted, including the unused rate in later repetitions. This is a development follow-up on previously examined evaluation data and a module deletion comparison, not an equal-parameter ranking or an identified neural mechanism.

Use the existing `.venv-peft`; no packages or system settings need changing. Run from the repository root, sequentially:

```powershell
& '.\.venv-peft\Scripts\python.exe' -m experiments.peft_module_ablation_v1.run_study --smoke
& '.\.venv-peft\Scripts\python.exe' -m experiments.peft_module_ablation_v1.run_study
& '.\.venv-peft\Scripts\python.exe' -m experiments.peft_adaptation_scope_v1.guard --output runs/peft_module_ablation_v1/analysis_guard --cpu-only --timeout-seconds 180 -- .venv-peft/Scripts/python.exe -m experiments.peft_module_ablation_v1.analyse
```

The original trainer is reused through a process-local registration wrapper. Its source files remain unchanged. The wrapper must finish its additional audits before setting both `completed` and `wrapped_completed` to true. The runner freezes source/plan/reference hashes, including original data, baseline predictions and cache arrays. Production trials reuse the original cache. S0 uses a separate cache and verifies that wrapped BOTH reproduces the original BOTH smoke predictions and initialization.

Every CUDA child runs under the shared Windows resource guard, one at a time. Never launch a second guard while the runner is active. Limits: available RAM5GiB/commit6GiB, child RSS8GiB, Git32, GPU10500MiB/85°C. A failure stops progression and retains artifacts. Existing completed trials resume only under the same contract; failed or partial outputs require diagnosis and preservation before a new run.

The original 200 updates, micro4/effective8 groups, native Y-only quantile loss, BF16 without autocast weight caching, TF32 off, CPU2 threads and allocation cap .67 are retained. Quantile scores are raw mean 2-pinball. Paired bootstrap intervals are conditional on the three fitted data/optimizer repetitions. Module counts, shared initialization and sampler, frozen parameters, checkpoint reload, validation selection and independently recomputed scores are verified before scientific conclusions are written.
