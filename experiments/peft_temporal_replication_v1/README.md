# Temporal replication of the real-source PEFT procedure

This study repeats the complete fixed selection and calibration procedure from study12 on each source's next nonoverlapping190-day block. It starts from the original Chronos-2 checkpoint and fits new adapters, heads, scaling statistics and calibration offsets. See the [plan and stopping rule](../../_docs/notes/tsfm_topics/13_peft_temporal_replication_plan_20260908.md).

Thin wrappers reuse protected study12 implementation in separate namespaces. Source, data, selected models and analysis artifacts from the original study remain protected. This is another season of the same sources, not an independent dataset or backbone confirmation.

After data preparation and CPU verification, root alone runs the shared GPU guard sequentially:

```powershell
& '.\.venv-peft\Scripts\python.exe' -m experiments.peft_temporal_replication_v1.run_study --smoke
& '.\.venv-peft\Scripts\python.exe' -m experiments.peft_temporal_replication_v1.run_study
```

Completed 2026-09-08: 28 fits, 14 selected forecasts, two separately guarded RAW baselines, verified analysis and four PNG/PDF figures. All 29 study CPU checks and 21 original-analysis regression checks passed. See the [results and decision](../../_docs/notes/tsfm_topics/13_peft_temporal_replication_results_20260908.md).

Bike's head-relative QCAL effect was +7.416% of F0 (97.5% CI 4.550–11.263%); Household's was +1.776% (0.531–3.147%), below the predeclared lower-bound >1% gate. Bike LoRA did not improve over F0. The current A branch is closed within this deployment and budget; this is not evidence that PEFT is universally unnecessary.

The last forecast was interrupted once when 56 transient Git processes exceeded the unchanged limit of 32. Its files were archived and only that forecast was retried. **There were 43 GPU attempts for 42 logical jobs.** Successful-only cost/resource files omit the archived attempt: use `results/peft_temporal_replication_v1/recovery_audit.json` for total cost and resource maxima. The recorded runner invocation wall time is the retry only. All-attempt guard time was 942.205 seconds; first guard to final completion including inspection and waiting was 1092.898 seconds. S0 is separate at 113.400 seconds. No training rerun or system-setting change was needed.
