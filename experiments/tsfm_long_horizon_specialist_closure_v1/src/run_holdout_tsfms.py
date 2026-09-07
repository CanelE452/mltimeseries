"""Score the three primary TSFMs on the fresh holdout, with the source study's code.

Section 6.2 forbids re-deriving the TSFM side: the adapters, the track builder,
the quantile contract and the evaluator are imported from the source study rather
than reimplemented, and the checkpoint revisions are asserted against the source
`model_audit.csv` so a silent upgrade cannot slip in.

Development TSFM scores are not recomputed at all - they are copied from the
source `benchmark_results.csv`, with the branch and SHA they came from recorded
on every row.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yaml

from . import paths

paths.configure_environment()

from experiments.tsfm_benchmark_gap_discovery_v1.src import models as source_models  # noqa: E402
from experiments.tsfm_benchmark_gap_discovery_v1.src import scoring, tracks  # noqa: E402

PRIMARY = ["chronos-2", "tirex-2", "timesfm-3.0"]
DIAGNOSTIC = ["chronos-2-synth"]
BASELINES = ["seasonal-naive", "linear-ar-specialist"]
TRACK = "U"


def assert_revisions() -> dict[str, str]:
    """The closure must keep the source study's checkpoints, not the latest ones."""
    from huggingface_hub import HfApi

    source = json.loads((paths.RESULTS / "SOURCE_STUDY.json").read_text(encoding="utf-8"))
    pinned = source["primary_model_revisions"]
    api = HfApi()
    live = {}
    for model_id in PRIMARY:
        repo = source_models.REGISTRY[model_id]["hf_repo"]
        live[model_id] = api.model_info(repo).sha
    drifted = {m: (pinned[m], live[m]) for m in PRIMARY if pinned[m] != live[m]}
    if drifted:
        raise SystemExit(
            "MODEL_REVISION_MISMATCH: the upstream checkpoints moved since the source study "
            f"({drifted}). Pin the revision explicitly before continuing."
        )
    return pinned


def load_task(task_uid: str):
    import fev

    raw = yaml.safe_load(
        (paths.SOURCE_DATA / "fev_bench_tasks.yaml").read_text(encoding="utf-8")
    )["tasks"]
    pool = pd.read_csv(paths.SOURCE_RESULTS / "task_pool.csv").set_index("task_uid")
    task = fev.Task(**raw[int(pool.loc[task_uid, "yaml_index"])])
    task.load_full_dataset(num_proc=1)
    return task


def run_model(model_id: str, task_uid: str, adapter=None) -> dict:
    from experiments.tsfm_benchmark_gap_discovery_v1.src.baselines import BASELINES as BASELINE_FN

    task = load_task(task_uid)
    quantile_levels = list(task.quantile_levels)
    seasonality = int(task.seasonality)
    safe = task_uid.replace("::", "__").replace("/", "-")

    predictions_per_window, losses, keys = [], [], []
    seconds, peak, contexts = 0.0, 0.0, []
    stored_q, stored_p, stored_w = [], [], []
    for window_idx in range(task.num_windows):
        window = task.get_window(window_idx, num_proc=1)
        inputs = tracks.build_inputs(window, TRACK, task)
        contexts.append(int(np.median(inputs.context_lengths)))
        if model_id in BASELINE_FN:
            result = BASELINE_FN[model_id](inputs, quantile_levels, seasonality)
        else:
            result = adapter.forecast(inputs, quantile_levels)
        seconds += result.seconds
        if np.isfinite(result.peak_memory_mb):
            peak = max(peak, result.peak_memory_mb)
        if not np.isfinite(result.quantiles).all():
            raise SystemExit(f"non-finite forecast from {model_id} on {task_uid} w{window_idx}")
        predictions = tracks.to_predictions(result.quantiles, result.point, inputs, quantile_levels)
        predictions_per_window.append(predictions)
        stored_q.append(result.quantiles.astype(np.float32))
        stored_p.append(result.point.astype(np.float32))
        stored_w.append(np.full(result.quantiles.shape[0], window_idx, dtype=np.int16))
        decomposition = scoring.per_origin_sql(window, predictions, quantile_levels, seasonality)
        losses.append(decomposition["origin_loss"])
        keys.extend([f"w{window_idx}::{i}" for i in decomposition["item_ids"]])

    summary = task.evaluation_summary(
        predictions_per_window,
        model_name=model_id,
        inference_time_s=seconds,
        trained_on_this_dataset=model_id == "linear-ar-specialist",
    )
    loss = np.concatenate(losses)
    windows = np.concatenate([np.full(p.size, i) for i, p in enumerate(losses)])
    np.savez_compressed(
        paths.RUNS / f"origins__TSFM_{model_id}__{safe}.npz",
        origin_loss=loss.astype(np.float32),
        origin_window=windows.astype(np.int16),
        origin_key=np.array(keys, dtype=object),
    )
    np.savez_compressed(
        paths.FORECASTS / f"TSFM_{model_id}__{safe}__seed0.npz",
        **{f"w{i:02d}": q for i, q in enumerate(stored_q)},
        quantile_levels=np.asarray(quantile_levels, dtype=np.float64),
        model_name=np.array([model_id], dtype=object),
    )
    return {
        "task_uid": task_uid,
        "model": model_id,
        "track": TRACK,
        "native_metric": task.eval_metric,
        "native_score": float(summary["test_error"]),
        "origin_aggregate": scoring.task_aggregate(loss, windows),
        "reconciliation_gap": scoring.reconcile(
            scoring.task_aggregate(loss, windows), float(summary["test_error"])
        ),
        "n_origins": int(loss.size),
        "n_windows": int(task.num_windows),
        "actual_context_median": int(np.median(contexts)),
        "inference_seconds": seconds,
        "peak_memory_mb": peak if peak else float("nan"),
        "source": "computed_here",
        "status": "OK",
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def copy_development_rows() -> pd.DataFrame:
    """Development TSFM scores come from the source study, unchanged."""
    source = json.loads((paths.RESULTS / "SOURCE_STUDY.json").read_text(encoding="utf-8"))
    results = pd.read_csv(paths.SOURCE_RESULTS / "benchmark_results.csv")
    wanted = results[
        (results.track == TRACK)
        & (results.split == "discovery")
        & (results.task_uid.isin(source["development_tasks"]))
        & (results.model.isin(PRIMARY + DIAGNOSTIC + BASELINES))
        & (results.status == "OK")
    ]
    expected = len(source["development_tasks"]) * len(PRIMARY + DIAGNOSTIC + BASELINES)
    if len(wanted) != expected:
        raise SystemExit(
            f"SOURCE_CONTRACT_MISMATCH: expected {expected} development rows, found {len(wanted)}"
        )
    return wanted.assign(
        source="copied_from_source_study",
        source_branch=paths.SOURCE_BRANCH,
        source_sha=source["source_remote_sha"],
        original_score=wanted.native_score,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", default=None)
    args = parser.parse_args()

    pinned = assert_revisions()
    print(f"checkpoint revisions match the source study: {pinned}", flush=True)

    spec = json.loads((paths.RESULTS / "long_horizon_tasks.json").read_text(encoding="utf-8"))
    holdout = spec["fresh_holdout_tasks"]
    wanted = args.models or (PRIMARY + DIAGNOSTIC + BASELINES)

    cache_dir = paths.RUNS / "tsfm_rows"
    cache_dir.mkdir(exist_ok=True)
    for model_id in wanted:
        adapter, loaded = None, False
        for task_uid in holdout:
            safe = task_uid.replace("::", "__").replace("/", "-")
            cache = cache_dir / f"{model_id}__{safe}.json"
            if cache.exists():
                print(f"cached  {model_id:22s} {task_uid}", flush=True)
                continue
            if model_id in source_models.REGISTRY and not loaded:
                adapter = source_models.build(model_id)
                start = time.perf_counter()
                adapter.load()
                print(f"loaded  {model_id} in {time.perf_counter() - start:.1f}s", flush=True)
                loaded = True
            print(f"running {model_id:22s} {task_uid}", flush=True)
            row = run_model(model_id, task_uid, adapter)
            cache.write_text(json.dumps(row, indent=2), encoding="utf-8")
            print(f"  SQL={row['native_score']:.4f} in {row['inference_seconds']:.1f}s", flush=True)
        if adapter is not None:
            adapter.unload()

    rows = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(cache_dir.glob("*.json"))]
    holdout_frame = pd.DataFrame(rows).assign(split="holdout")
    holdout_frame.to_csv(paths.RESULTS / "holdout_tsfm_results.csv", index=False)
    development = copy_development_rows().assign(split="development")
    development.to_csv(paths.RESULTS / "development_tsfm_results.csv", index=False)
    print(f"\nholdout rows {len(holdout_frame)}, development rows copied {len(development)}")


if __name__ == "__main__":
    main()
