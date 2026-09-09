# Train-only lag estimation and native TSFM adaptation

This fixed follow-up asks whether a simple procedure can recover the lagged signal without the previous true-lag dictionary. It uses fresh independent draws from the same Q00 Gaussian family, not a new real-world source. The [execution contract](../../_docs/notes/tsfm_topics/10_peft_trainlag_plan_20260908.md) was written before generating the new data.

Each of Y/U/V selects one lag from the prespecified integer range16…128 using only the same64×16 future-Y training labels available to the FM. No past Y is added as extra supervised targets. The native FM retains Y history and aligns U/V separately; retimed future values come entirely from the original observed past. RAW uses the identical estimated Y/U/V lags, an intercept plus three OLS coefficients, and empirical in-sample training residual quantiles. Its residual calibration can be optimistic and is reported with coverage and oracle-mean error.

Four FM procedures are compared: F0, ATTN, ALIGN_F0 and ALIGN_ATTN. The aligned frozen FM has a trained preprocessing step and is not zero-shot. ATTN uses rank8 attention LoRA on96 time/group qkvo projections,1,179,648 trainable parameters, with the native output projection frozen. There is no new residual head. All18 GPU trials are new:12 fits across two input modes, two learning rates and three training corpora, plus6 frozen-model/cache runs. Corpus0 validation selects LR separately for each mode; all unused-rate repetitions remain in the cost and all-results CSV.

Use the existing `.venv-peft` from the repository root. One GPU child runs at a time under the existing shared guard. Do not run another guard while the study runner is active. Failed or partial artifacts are preserved and require diagnosis before retrying. Source/plan/data/lag contracts must match before resuming a completed trial.

```powershell
& '.\.venv-peft\Scripts\python.exe' -m experiments.peft_trainlag_v1.run_study --smoke
& '.\.venv-peft\Scripts\python.exe' -m experiments.peft_trainlag_v1.run_study
```

Generate the fixed data and lag JSON files before S0. RAW fitting and the final analysis are CPU operations performed only after the full FM study finishes. Give every analysis invocation a new guard output directory to retain earlier logs. The original studies, system settings and unrelated processes are preserved. No package installation, commit or push is required.

For a clean checkout with an empty data output directory, generate once with `python -m experiments.peft_trainlag_v1.data --output runs/peft_trainlag_v1/data`. Keep an existing data directory for contract-preserving resumes. After all18 GPU trials complete, the following CPU analysis fits the three RAW controls if absent and verifies existing ones before reuse:

```powershell
& '.\.venv-peft\Scripts\python.exe' -m experiments.peft_adaptation_scope_v1.guard --output runs/peft_trainlag_v1/analysis_guard --cpu-only --timeout-seconds 180 -- .venv-peft/Scripts/python.exe -m experiments.peft_trainlag_v1.analyse
& '.\.venv-peft\Scripts\python.exe' -m experiments.peft_trainlag_v1.plot
```

The two primary contrasts are alignment versus raw attention adaptation and the additional benefit of attention adaptation after alignment. They use paired97.5% intervals as one family of two contrasts. RAW-to-oracle proximity is a separate exploratory95% diagnostic with a1% F0-score operational margin. Intervals condition on the three fitted repetitions and do not include the full uncertainty of training and validation selection. This is a test of the need for a new PEFT method on one synthetic family; a positive FM comparison alone is not a method contribution.

Completed on2026-09-08: all18 production GPU trials, four S0 trials, three CPU RAW fits and31 CPU tests passed. [Results and research decision](../../_docs/notes/tsfm_topics/10_peft_trainlag_results_20260908.md) report score means F0 .536967, ATTN .497214, ALIGN_F0 .460572, ALIGN_ATTN .368454, RAW .323352 and ORACLE .322664. All three train corpora selected Y/U/V lags32/48/48. Alignment and subsequent LoRA each improved the score, but RAW met the prespecified near-oracle criterion, so new adapter search on this Gaussian condition stops. ALIGN_ATTN coverage of its80% interval was only69.97%; score improvement does not establish calibration. This is fresh sampling of the same synthetic family, not external confirmation or a new PEFT method.

Production wall time was700.25s; CPU analysis guard4.078s. All production guards exited0 with no safety stop. The76 production resource samples recorded at least14.99GiB available RAM and at most2295MiB GPU memory/53C. Windows queries from S0 start through13:26:34KST found no matching Application1000/1002 or selected System events. These observations do not guarantee future stability. Prediction recalculation, complete selection keys, lag scores, OLS refits and frozen contracts passed in `results/peft_trainlag_v1/verification.json`.
