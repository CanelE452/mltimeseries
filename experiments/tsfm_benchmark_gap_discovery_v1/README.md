# TSFM-BENCHMARK-GAP-DISCOVERY-v1

An upstream study, not a method. It asks where recent time-series foundation
models repeatedly lose ground on a public benchmark, whether that loss is
recoverable, and whether a simple deployable fix or existing 2025–2026 work
already closes it. `NO_STRONG_GAP_FOUND` is a legitimate outcome and no method is
implemented here.

## What it compares

| | |
|---|---|
| Benchmark | fev-bench, task definition pinned to `autogluon/fev@eadb28e`, evaluated by the `fev` library itself |
| Primary models | `amazon/chronos-2` (T5-style encoder), `NX-AI/TiRex-2` (xLSTM recurrent), `google/timesfm-3.0-pytorch` (patch/variate transformer) |
| Diagnostic model | `autogluon/chronos-2-synth`, synthetic-only pretraining, used as a contamination anchor |
| Baselines | SeasonalNaive (B0) and a direct multi-horizon ridge autoregression fitted per task (B1) |
| Metric | SQL, the metric every fev-bench task declares, computed by the native evaluator |

Information conditions are kept apart. TRACK U is the common univariate condition
every model can meet; TRACK M is native multivariate; TRACK C supplies known-future
covariates. Scores from different tracks are never pooled.

## Order of operations

The freezes matter more than the code. Each stage refuses to start if the freeze
it depends on is missing.

```
task_pool          benchmark metadata only, no model touched
select_tasks       deterministic 12/6 split, hashed        <- frozen before any score
audit_models       official cards + live HF revisions
smoke              every estimator on one real task
run_models         TRACK U discovery
evaluate           raw matrix, then normalisation
descriptors        D1-D9 from train-visible data
failure_map        registered gate: >=3 tasks, >=2 families, >=5% gap, >=8% regret
make_candidates    at most 4, each with >=2 competing mechanisms, hashed
run_probes         oracle + simple fixes, headroom arithmetic
confirm_candidates opens the holdout only now
literature_audit   only for candidates that cleared the headroom screen
rank_candidates    7 components, 33 points, hard fails are not outscorable
verify             A01-A20 against the artifacts on disk
report / status    verdict.json and STATUS.md
```

## Running it

```bash
python -m experiments.tsfm_benchmark_gap_discovery_v1.src.run setup
python -m experiments.tsfm_benchmark_gap_discovery_v1.src.run smoke
python -m experiments.tsfm_benchmark_gap_discovery_v1.src.run models --split discovery --track U
python -m experiments.tsfm_benchmark_gap_discovery_v1.src.run analyse-discovery
python -m experiments.tsfm_benchmark_gap_discovery_v1.src.run models --split confirmation --track U
python -m experiments.tsfm_benchmark_gap_discovery_v1.src.run confirmation
python -m experiments.tsfm_benchmark_gap_discovery_v1.src.run finish
```

Results land in `results/tsfm_benchmark_gap_discovery_v1/`; caches, forecasts and
model weights stay in `runs/` and `data_external/`, neither of which is tracked.

## Environment notes

This host is Windows with no MSVC toolchain and no CUDA toolkit, which shapes two
choices recorded in `STATUS.md`:

- TiRex-2 runs the pure-PyTorch kernels its package ships for CPU and Metal,
  because the fused FlashRNN and Triton kernels need a compiler that is not
  present. CPU and CUDA agree to a relative 1.0e-06 on a reference series.
- Everything that touches `datasets` passes `num_proc=1`. Multiprocess dataset
  maps on Windows re-import the entry point, and a script without an
  `if __name__ == "__main__"` guard forks until it hangs.

## Tests

```bash
python -m pytest experiments/tsfm_benchmark_gap_discovery_v1/tests -q
```

They cover the baseline contracts, the TRACK U row-to-target mapping that the
whole multi-target evaluation depends on, and the determinism of the task split.
