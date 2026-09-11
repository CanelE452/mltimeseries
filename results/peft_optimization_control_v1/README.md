# Optimization controls, completed 2026-09-10

Read the Korean [report28](../../_docs/notes/tsfm_topics/07_research_direction/28_optimization_control_20260910.md) and frozen [protocol](../../experiments/peft_optimization_control_v1/PURPOSE.md).

48 trajectories, 48 selected evaluations, 2 smoke fits completed. Practical continuation gate passed under UPDATE and EXPOSURE. This reuses development E, not a new confirmation test. Same MLP architecture/initialization; final head weights are trained separately and total trainable capacity differs.

- `summary.json`: source/condition/seed gains, gate, independent score and resource audit.
- `metrics.csv`: each selected model's score, LR, checkpoint and fit key.
- `matched_head_gain.png`: previous result versus the two optimization controls.
- `validation_trajectories.png`: early-checkpoint/LR behavior.
- `exposure_units.png`, `exposure_unit_audit.json`: origin versus overlapping target-hour repetition; not effective sample size.
- `evidence/`: small contracts, selections and input/code audits. Heavy predictions/checkpoints remain under ignored `runs/peft_optimization_control_v1/`.

Independent reanalysis, with the local run present:

```powershell
.venv-peft/Scripts/python.exe -m experiments.peft_optimization_control_v1.analyse
```

The new-period decision pilot is documented separately in [report29](../../_docs/notes/tsfm_topics/07_research_direction/29_overlap_transfer_20260910.md). Existing results are preserved; the new pilot does not turn these development observations into independent validation.
