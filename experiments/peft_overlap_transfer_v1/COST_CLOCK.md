# Cost clock reconciliation before future E

The frozen PURPOSE defines complete serial fit subprocess time, not just the training routine. Inspection during the first future fit found that legacy fit.py starts its `seconds` clock after module imports. The initially frozen analyse.py used that field and would exclude Python/Torch imports. No future E has opened and no decision threshold or selected arm changes.

Keep the frozen fit/controller/analysis sources intact. Use the separate `analyse_guard_cost.py` copy for final analysis: its primary fit_seconds is guard status elapsed_seconds, which includes child imports, model loading, training, V, replay, guard preflight/launch and cleanup, but excludes the external admission queue. Also retain internal_fit_seconds for transparency. Guard overhead is included for both arms and is not claimed to be pure GPU compute. This implements the complete-work cost intent, with the explicitly disclosed small guard overhead.

Seal the copy hash and this note hash in runs/peft_overlap_transfer_v1/analysis_clock_contract.json before selection.json/new E exists. The final analysis verifies both hashes and that the clock contract predates the first future forecast. Preserve the original analysis; do not run it as the final cost assessment. No model/data/accuracy rule or cost threshold changes.
