# Native PEFT on new real-data forecasting panels

This development screen compares output adaptation, standard LoRA and a direct ridge control on hourly bicycle demand and household power. The [execution plan](../../_docs/notes/tsfm_topics/12_peft_external_gap_plan_20260908.md) fixes source columns, missing-label handling, temporal boundaries, selection and decision criteria. A positive result would motivate further research; it is not a new PEFT algorithm.

Only past observations enter every forecast. Context336/horizon48 use daily origins. The chronological sequence is14 days of context,64 fitting days,14 selection days,14 calibration days and84 evaluation days. Missing target measurements remain masked. Context forward filling and leading train-median substitution are recorded separately from the original target observations.

Fitting subprocesses receive prepared files containing only training and selection data. The runner completes26 adaptation fits and two F0/cache jobs before freezing all fourteen selected procedures. Only then does it dispatch fourteen held-out forecast jobs. H_MLP/H_FULL have six combined development candidates versus three LoRA candidates, so this is not equal HPO cost and selection noise may affect the comparison. Shared QCAL is fitted on separate calibration data after selection.

Use the existing `.venv-peft`, prepared data, and the repository root. S0 uses training data only. Root orchestration runs one GPU child at a time under the shared resource guard.

```powershell
& '.\.venv-peft\Scripts\python.exe' -m experiments.peft_external_gap_v1.run_study --smoke
& '.\.venv-peft\Scripts\python.exe' -m experiments.peft_external_gap_v1.run_study
```

`--fit-only` stops after freezing the selected fit results. A later invocation verifies and reuses those exact completed fits before forecasting. Existing incomplete outputs are preserved and rejected. Source/plan/data contracts must remain identical to S0. Logs and large predictions/checkpoints remain under ignored `runs/peft_external_gap_v1`; small results belong in `results/peft_external_gap_v1`.

No old study source, package environment, driver or system setting is changed. Resource observations constrain this run but do not guarantee future system stability. Do not run another GPU guard concurrently.

Status: all 28 fits, 14 forecasts, two RAW CPU fits and eight S0 trials completed on 2026-09-08. Audited analysis passed; Bike's QCAL effect exceeds the practical threshold, while Household remains inconclusive. The overall two-source entry gate was not met. See the [results and self-evaluation](../../_docs/notes/tsfm_topics/12_peft_external_gap_results_20260908.md).

The first analysis stopped while summarizing CPU records with null GPU fields. Its logs remain in `analysis_guard`; `analysis_guard_verified` completed after the reporting fix. Final figures are in `results/peft_external_gap_v1/figures_verified`; initial plots are preserved separately. No GPU fits were repeated to fix either reporting issue.
