# C calibration closure

CPU-only diagnostic on the already selected twelve FM trials from `peft_trainlag_v1`.
The frozen execution contract is `_docs/notes/tsfm_topics/11_peft_calibration_closure_plan_20260908.md`.

Run once through the root agent's CPU resource guard:

```text
.venv-peft/Scripts/python.exe -m experiments.peft_calibration_closure_v1.analyse
```

`SORT` orders each forecast's quantile axis. `QCAL` fits one offset per quantile from
128 validation episodes × 16 leads, using NumPy's linear empirical quantile, adds
the offsets to sorted forecasts, and sorts again. All twelve offsets are saved
before evaluation arrays are decoded. No checkpoint, LR, adapter strength or
additional calibration candidates are selected here.

The output directory `results/peft_calibration_closure_v1` contains all 24 rows,
eight procedure means, four descriptive paired QCAL-minus-SORT comparisons,
validation offsets and corrected predictions. F0 denominators are the original
native F0 forecasts, before either correction, and are recomputed for every
paired bootstrap draw. Coverage changes are fractions, not percentage points.
Three fitted corpora share the same 512 ordered evaluation episodes; bootstrap
sampling preserves this pairing and does not resample the prior fitting/selection.

The primary ALIGN_ATTN criterion requires every corpus's QCAL score worsening
versus SORT to be at most 1% of its native F0 score, and mean QCAL coverage80 to be
between 0.78 and 0.82, inclusively. Confidence intervals are descriptive. The same
validation was already used for model selection, and evaluation was previously
observed; this is neither a fresh confirmation nor a conformal coverage guarantee.

Prior source, data, model, cache, selected artifacts, output and execution-contract
hashes are verified before and after analysis. Existing output is never overwritten.
`verification.json` is written last and is the completion marker. Tests use synthetic
arrays and do not execute the production analysis or access a GPU.
