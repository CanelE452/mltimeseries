"""Record the benchmark contract this study evaluates under."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd
import yaml

from . import paths
from .task_pool import TASKS_YAML, canonical_yaml_hash

FEV_COMMIT = "eadb28ed3a3f8fc2db8dd4d3d6850894efcbc4d1"


def build() -> dict:
    import fev
    import importlib.metadata as md

    raw = yaml.safe_load(TASKS_YAML.read_text(encoding="utf-8"))["tasks"]
    pool = pd.read_csv(paths.RESULTS / "task_pool.csv")
    quantiles = {tuple(task["quantile_levels"]) for task in raw}
    extra = {json.dumps(task["extra_metrics"], sort_keys=True) for task in raw}

    return {
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "primary_benchmark": {
            "name": "fev-bench",
            "paper": "fev-bench: A Realistic Benchmark for Time Series Forecasting (2025, arXiv:2509.26468)",
            "official_repo": "https://github.com/autogluon/fev",
            "pinned_commit": FEV_COMMIT,
            "task_definition_url": (
                f"https://raw.githubusercontent.com/autogluon/fev/{FEV_COMMIT}/"
                "benchmarks/fev_bench/tasks.yaml"
            ),
            "task_definition_sha256": canonical_yaml_hash(),
            "evaluator": "fev (native)",
            "evaluator_version": md.version("fev"),
            "evaluator_module": fev.__file__,
            "n_tasks": len(raw),
            "dataset_hub_path": sorted({task["dataset_path"] for task in raw}),
            "eval_metric": sorted({task["eval_metric"] for task in raw}),
            "extra_metrics": sorted(extra),
            "quantile_levels": sorted(list(q) for q in quantiles),
            "split_semantics": (
                "Each task defines num_windows rolling evaluation windows. A window exposes past "
                "data up to its cutoff plus known-future columns over the horizon; the horizon "
                "values of the target are never visible to the model."
            ),
            "covariate_semantics": {
                "known_dynamic_columns": "available over the past AND the forecast horizon",
                "past_dynamic_columns": "available over the past only",
                "static_columns": "time invariant per series",
            },
            "missing_handling": (
                "Series with fewer than min_context_length observations before the cutoff or fewer "
                "than horizon observations after it are dropped by fev, identically for every model."
            ),
            "leakage_policy": (
                "fev-bench does not ship a model-specific exclusion list. Pretraining overlap is a "
                "property of each checkpoint and is tracked in contamination_matrix.csv."
            ),
            "license": "Apache-2.0 for the fev library; per-dataset licences upstream in autogluon/fev_datasets",
            "expected_data_volume": (
                f"{int(pool.n_forecasts_track_u.sum()):,} TRACK U forecasts over the full 100-task "
                "pool; the frozen 18-task selection is a small fraction of it"
            ),
        },
        "secondary_benchmarks_considered": [
            {
                "name": "GIFT-Eval",
                "official_repo": "https://github.com/SalesforceAIResearch/gift-eval",
                "status": "NOT_RUN",
                "reason": (
                    "Declared SECONDARY by the study contract. Running it needs a second harness "
                    "and a second full data download; under the 24 GPU-hour budget the primary "
                    "discovery plus its holdout confirmation on fev-bench was funded instead. "
                    "Recorded in 'what was not tested' rather than partially attempted."
                ),
            },
            {
                "name": "TIME",
                "status": "NOT_RUN",
                "reason": "Optional in the study contract; not required once fev-bench is primary.",
            },
        ],
        "metric_contract": {
            "native": "SQL (scaled quantile loss), the metric every fev-bench task declares",
            "computed_by": "fev native evaluator only; no metric is reimplemented in this study",
            "normalisation": (
                "relative_to_naive(m,t) = score(m,t) / score(seasonal_naive,t); "
                "regret(m,t) = 100 * (score(m,t) - best_deployable(t)) / best_deployable(t)"
            ),
        },
    }


def main() -> None:
    audit = build()
    out = paths.RESULTS / "benchmark_audit.json"
    out.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    print(json.dumps(audit["primary_benchmark"], indent=2)[:1400])


if __name__ == "__main__":
    main()
