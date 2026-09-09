# Selection regret among saved PEFT candidates

This diagnostic follows the closed A branch. It uses the saved seed-12000 candidate grids from studies 12 and 13: four source/time-block cells, ten candidates per cell. It does not perform new training or recover missing intermediate checkpoints. The [fixed plan](../../_docs/notes/tsfm_topics/14_peft_selection_regret_plan_20260908.md) defines seven validation-only selection rules and the stopping criteria.

Twelve original F0/selected-head/selected-LoRA forecasts are reused; 28 previously unselected candidate forecasts are added. Before production, four S0 runs must reproduce each cell's original selected LoRA forecasts. All original sources, selection files, checkpoints, predictions, analyses and recovery records remain protected.

SORT is primary for this new selection question; the original studies' QCAL primary conclusions remain unchanged. Hindsight oracle regret is descriptive and cannot serve as a deployable selector. Full HPO grids exist for only one optimizer seed. These partially observed development sources do not provide independent confirmation of a new method.

Root runs one GPU child at a time through the shared guard with the existing limits, including at most 32 Git processes. Interrupted attempts are retained and included in cost/resource reporting. Invocation records distinguish actual new attempts from reuse on restart.

Completed on 2026-09-08: 47 CPU tests and four subtests passed; all 40 saved validation forecasts reproduced their scores and frozen choices. Four S0 runs had zero difference from the original C/E forecasts. All 28 new forecasts completed without failures in 457.341 seconds; no training was added. Independent NumPy recomputation matched all 80 scores and four primary intervals exactly. The fixed B gate failed, including the fixed-low-LR simple-rule veto. See the [results and limitations](../../_docs/notes/tsfm_topics/14_peft_selection_regret_results_20260908.md). This closes the current B branch, not all PEFT selection research. The 557 protected parent artifacts remain unchanged.

Run with `.venv-peft/Scripts/python.exe -m experiments.peft_selection_regret_v1.prepare`, then the `run_study` module with `--smoke`, then without that flag. After every forecast completes, execute the `analyse` module through a CPU guard and then `plot`. Changed contracts and incomplete outputs are rejected; preserve failed attempts before any recovery.
