# Frozen overlap decision pilot: completed 2026-09-10

Read Korean [report29](../../_docs/notes/tsfm_topics/07_research_direction/29_overlap_transfer_20260910.md), frozen [PURPOSE](../../experiments/peft_overlap_transfer_v1/PURPOSE.md), and the pre-E [cost-clock reconciliation](../../experiments/peft_overlap_transfer_v1/COST_CLOCK.md).

24 fits, 26 forecasts and 1 CPU preparation completed successfully. OVERLAP failed the prespecified accuracy gate: mean regret0.503/0.674%F0 versus allowed0.25, while saving11.54/12.63% of counterfactual fit time. All12 matched-arm comparisons favored LoRA on the new E periods. Static sampling geometry did not preserve the SPREAD30/RECENT30 utility ordering.

- `summary.json`, `metrics.csv`, `rule_metrics.csv`: primary decisions, individual scores and complete guarded fit times. Internal routine times are also retained.
- `future_gain.png`: development versus new E utility.
- `rule_tradeoff.png`: fixed decision rules' accuracy/cost.
- `validation_transfer.json`, `validation_transfer.png`: descriptive selected V versus E utility; no new rule fitted.
- `interpretation_audit.json`: absolute F0 comparisons and rank reversal checks.
- `evidence/`: frozen plan/contracts/selections, prepared manifest, clock and diagnostic seals, read-only data/analysis reviews, independent numerical verification and system safety audit.

Two familiar sources, one new E period each, two optimization seeds. Bike reuses past development observations for fit/V/context; new E targets are time-disjoint from prior E in the bounded local audit. Pretraining overlap unknown. Same head architecture and initialization, not identical final head weights or equal total capacity. Cost savings are retrospective deployment counterfactuals, not savings realized by this research campaign.

With the complete local run present, final analysis modules are:

```powershell
.venv-peft/Scripts/python.exe -m experiments.peft_overlap_transfer_v1.analyse_guard_cost
.venv-peft/Scripts/python.exe -m experiments.peft_overlap_transfer_v1.validation_transfer_diagnostic
```

The primary analysis intentionally refuses to overwrite this existing result directory. Preserve it when reproducing elsewhere. The initially frozen `analyse.py` remains as provenance; it uses the internal clock and is not the final cost report. Heavy checkpoints/predictions stay in ignored `runs/`; no commit/push was performed in this turn.
