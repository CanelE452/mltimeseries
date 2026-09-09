# Head-complement candidate: CPU entry gate

This prototype implements existing fixed-support output residualization. It is not a novel PEFT method or a forecasting training run. The candidate was stopped after direct overlap with SDDR/ONO/PHO and the actual small-support rank failure were checked.

- [Plan](../../_docs/notes/tsfm_topics/07_head_complement_method_gate_plan_20260908.md)
- [Results and scope](../../_docs/notes/tsfm_topics/07_head_complement_method_gate_results_20260908.md)
- [Verified numeric artifact](../../results/peft_head_complement_v1/entry_gate_verified.json)

`projection.py` implements SVD-based fit-support residualization, coefficient extraction, and extension to new inputs using fit coefficients. `audit.py` compares it against synthetic joint-head reparameterization, prediction-preserving PHO, and an independent MSE profiled-head solve. It also checks rank and forward/VJP annihilation on three ETTm2 fit-feature minibatches and the full fit support. Random probes are not actual LoRA updates or forecasts. No held-out feature rows or labels are used.

Run from the repository root in PowerShell using the existing environment:

```powershell
& '.\.venv-peft\Scripts\python.exe' -m experiments.peft_adaptation_scope_v1.guard --output runs/peft_head_complement_v1/reproduce/guard --cpu-only --timeout-seconds 180 -- .venv-peft/Scripts/python.exe -m experiments.peft_head_complement_v1.audit --output runs/peft_head_complement_v1/reproduce/entry_gate.json
```

Requires the existing S1 ETTm2 prepared NPZ and feature cache at the paths recorded by `audit.py`. The guard shares the S1 study lock and checks RAM/commit/Git counts. CPU linear algebra uses two threads. There is no GPU allocation, model forward, optimizer step, or forecasting score. Source and input hashes are saved. Reproductions write to ignored runs and leave the original numeric artifacts intact.

The first `entry_gate.json` passed numerically but emitted a metric conversion warning. The verified run uses explicit detach when converting metrics; its stderr is empty. Both run artifacts are retained. Synthetic MSE equivalence does not establish equivalence for quantile pinball loss, regularized or early-stopped heads, or finite-step optimizer trajectories.
