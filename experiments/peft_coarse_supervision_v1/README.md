# Monthly-only supervision and internal adaptation

This is study 17, a necessity screen for standard attention LoRA, not a claim of a new PEFT method.

Completed on 2026-09-08: `CLOSE_CURRENT_COARSE_SUPERVISION_SCREEN`. The fitted head already damaged fine patterns; fixed-head LoRA did not recover simple-baseline performance. See the [results and interpretation](../../_docs/notes/tsfm_topics/17_coarse_supervision_results_20260908.md). All nine guarded stages exited successfully; the method criterion failed, not execution.

The fixed BDG2 Eagle/Lamb panel has eight 2016-complete donors and eight targets per site. Only donors supply fine training profiles. Targets supply monthly totals. Train origins are February–December 2016; coarse validation is January–March 2017; sealed fine evaluation is April–June 2017. July–December 2017 is reserved. The checkpoint's exact pretraining overlap is unknown.

See the [data entry plan](../../_docs/notes/tsfm_topics/17_coarse_supervision_data_entry_plan_20260908.md) and [learning plan](../../_docs/notes/tsfm_topics/17_coarse_supervision_learning_plan_20260908.md) for the frozen rules. Actual completion and decisions must be read from the stage JSON files rather than inferred from this implementation.

The five point forecasts are PROFILE, F0, COARSE_LIFT, FROZEN_HEAD and ATTN_LORA. The two ridge baselines choose from three fixed regularization strengths using monthly validation labels. LoRA starts from the selected frozen head and selects among six checkpoints using those same coarse labels. All choices precede evaluation forecasts; every forecast is saved before fine scoring.

The existing resource guard runs cache, ridge, three-update smoke, 200-update training, forecast and analysis serially. It keeps the previous RAM, commit, process-count, GPU-memory and temperature limits. Partial runs are preserved and require an explicit recovery analysis; no automatic retry or overwrite is performed.

Artifacts are in `runs/peft_coarse_supervision_v1/`, source acquisition in `data_external/bdg2_coarse_supervision_v1/`, and final metrics in `results/peft_coarse_supervision_v1/`.
