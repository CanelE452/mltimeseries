# Observation operator entry audit

Known linear Gaussian conditioning with mixed END_BASE/MEAN/SUM reports, overlapping sensor noise, and delayed arrivals. This is a numerical entry audit, not a trained PEFT method or a forecasting benchmark.

The fixed plan is [16_observation_operator_entry_plan_20260908.md](../../_docs/notes/tsfm_topics/16_observation_operator_entry_plan_20260908.md). `prepare.py` freezes the plan, four numerical source files, and 1,079 parent artifacts. `run.py` refuses to replace existing numerical results. The existing shared guard runs the numerical audit on CPU only.

`data_qc.py` has its own contract, written before downloading the two fixed USCRN Tucson 2024 products. It compares timestamps and temperature support semantics; it does not train or score a forecast. Data, response provenance and QC are stored separately under `data_external/uscrn_operator_entry_v1/`. Revised annual archives do not establish historical availability at forecast time.

```powershell
.\.venv-peft\Scripts\python.exe -m experiments.peft_observation_operator_entry_v1.prepare
.\.venv-peft\Scripts\python.exe -m experiments.peft_adaptation_scope_v1.guard --cpu-only --timeout-seconds 120 --output runs/peft_observation_operator_entry_v1/numerical_guard -- .\.venv-peft\Scripts\python.exe -m experiments.peft_observation_operator_entry_v1.run
.\.venv-tsfm\Scripts\python.exe -m pytest experiments/peft_observation_operator_entry_v1/tests -q
```

Completed artifacts must be inspected and preserved before considering another invocation. Do not delete an existing result, contract or partial download to make a rerun pass.
