# Study36: dependence PEFT preparation audit

User authorization: 2026-09-11, execute steps 1–4 under the proposed conditional gates.

Original plan: `_docs/notes/tsfm_topics/07_research_direction/36_dependence_peft_paper_plan_20260911.md`.
SHA256 before execution: `71ba9837ace86bb54e4d3792d7e7a2c4f2c9d10c25d4938f4fec9a6c92a20f04`.

Scope: CPU-only validation of the proposed DGP and simple controls before preparing the 72-fit GPU experiment. No model fitting on a TSFM, no adapter development, no final holdout access. The independent population audit is an executed design check, not a preregistered positive result.

Ownership: `independent_gate_audit.py` by primary agent; `data_gate.py` by independent data-engineer. Neither changes previous experiments. Each writes to its own results subdirectory.

Stop reason: the proposed forward/reverse pair is exactly related by observation sign relabelling; the observed last state is sufficient for Bayes prediction, and a three-coefficient quadratic represents the one-step optimum. These findings invalidate the intended interpretation of a directional score gap as evidence for temporal-interaction PEFT. No claim about actual LoRA performance follows.

Do not resume GPU training from this folder merely because the earlier plan listed 72 fits. A revised research design must establish a distinct, relevant conditional-history effect and its controls first. Preserve the original plan and all failed-design evidence.
