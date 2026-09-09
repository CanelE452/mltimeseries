# R1 revision entry diagnostic

This namespace implements the CPU entry check approved in study19. It does not implement a new PEFT method or load a foundation model.

- Protocol: `_docs/notes/tsfm_topics/19_peft_revision_entry_plan_20260908.md`
- Transport diagnostic addendum: `_docs/notes/tsfm_topics/19a_revision_download_transport_20260908.md`
- `fetch.py`: official ALFRED two-series real-time-period archive request, bounded transfer and preserved receipts.
- `model.py`: standard ridge, release-bias helper and exact fixed-feature revision sufficient statistics.
- `run.py`: chronological data construction, validation selection, evaluation, paired time-block bootstrap and previous-artifact verification.
- `tests/`: algebra, arrival/window rules and future-vintage perturbation tests.

Completed: official browser archives imported after two urllib POST timeouts; raw and causal coverage passed. The 17 tests and 4 subtests passed. CPU fitting/evaluation and external CSV-based metric audit completed. Main revised-label screen passed, but posthoc matched-regularization effects were adverse on PAYEMS and tiny on INDPRO; zero-growth had the lowest full-evaluation MSE on both. New PEFT GPU work on these two series is deferred. No FM or GPU training ran. See `_docs/notes/tsfm_topics/19_peft_revision_entry_results_20260908.md`.

The completed source/input contract is `runs/peft_revision_entry_v1/cpu_contract.json`; final raw inputs are under `data_external/alfred_revision_entry_v1/complete`. The shared CPU-only guard ran `python -m experiments.peft_revision_entry_v1.run freeze` and then `run`. Existing outputs are not overwritten. `diagnose.py` is explicitly post-evaluation and has its own contract; `independent_audit.py` independently parses raw intervals without importing the main experiment.

The six-month-vintage target is not final economic truth. INDPRO and PAYEMS are related macroeconomic series. The fixed two-event forecast horizon and six growth lags must not silently become a one-step nowcast. A successful exact ridge identity demonstrates an existing linear solution, not a new algorithm. A correction arm selected after comparing evaluation scores remains exploratory.
