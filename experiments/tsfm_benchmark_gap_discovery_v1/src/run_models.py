"""Run models and baselines over the frozen task selection.

One (model, track, task) pair at a time. Results are cached under
runs/tsfm_benchmark_gap_discovery_v1/predictions so an interrupted run resumes
without recomputing, and so the confirmation split can stay untouched until the
candidates are frozen.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
import traceback
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
import yaml

from . import paths, tracks
from .baselines import BASELINES
from .models import REGISTRY, build, configure_environment

configure_environment()

CONTEXT_MODE = "native_default"


def _task_configs() -> dict[str, dict]:
    raw = yaml.safe_load(
        (paths.DATA_EXTERNAL / "fev_bench_tasks.yaml").read_text(encoding="utf-8")
    )["tasks"]
    pool = pd.read_csv(paths.RESULTS / "task_pool.csv")
    return {row.task_uid: raw[int(row.yaml_index)] for _, row in pool.iterrows()}


def _selected() -> dict:
    payload = (paths.RESULTS / "selected_tasks.json").read_text(encoding="utf-8")
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    recorded = (paths.RESULTS / "selected_tasks.sha256").read_text(encoding="utf-8").split()[0]
    if digest != recorded:
        raise SystemExit("HARD STOP: TASK_SELECTION_NOT_FROZEN (selected_tasks.json hash changed)")
    return json.loads(payload)


def _cache_path(model_id: str, track: str, task_uid: str) -> "paths.pathlib.Path":
    safe = task_uid.replace("::", "__").replace("/", "-")
    return paths.PRED_CACHE / f"{model_id}__track{track}__{safe}.json"


def _origin_path(model_id: str, track: str, task_uid: str) -> "paths.pathlib.Path":
    safe = task_uid.replace("::", "__").replace("/", "-")
    return paths.PRED_CACHE / f"{model_id}__track{track}__{safe}__origins.npz"


def forecast_path(model_id: str, track: str, task_uid: str) -> "paths.pathlib.Path":
    """Raw quantile forecasts, kept so the probes can recombine and recalibrate them."""
    safe = task_uid.replace("::", "__").replace("/", "-")
    return paths.PRED_CACHE / f"{model_id}__track{track}__{safe}__forecasts.npz"


def run_one(
    model_id: str,
    track: str,
    task_uid: str,
    task_config: dict,
    pool_row: pd.Series,
    adapter=None,
) -> dict:
    import fev

    from . import scoring

    task = fev.Task(**task_config)
    task.load_full_dataset(num_proc=1)
    quantile_levels = list(task.quantile_levels)
    seasonality = int(task.seasonality)

    predictions_per_window = []
    origin_losses, origin_keys = [], []
    stored_quantiles, stored_point, stored_window = [], [], []
    total_seconds = 0.0
    peak_memory = 0.0
    context_used = []
    notes: dict = {}
    known_used: list[str] = []

    for window_idx in range(task.num_windows):
        window = task.get_window(window_idx, num_proc=1)
        inputs = tracks.build_inputs(window, track, task)
        known_used = list(inputs.known_columns)
        context_used.append(
            {
                "window": window_idx,
                "min": int(inputs.context_lengths.min()),
                "median": int(np.median(inputs.context_lengths)),
                "max": int(inputs.context_lengths.max()),
            }
        )
        if model_id in BASELINES:
            result = BASELINES[model_id](inputs, quantile_levels, seasonality)
        else:
            result = adapter.forecast(inputs, quantile_levels)
        notes = result.notes
        total_seconds += result.seconds
        if np.isfinite(result.peak_memory_mb):
            peak_memory = max(peak_memory, result.peak_memory_mb)
        if not np.isfinite(result.quantiles).all():
            raise SystemExit(
                f"non-finite forecast from {model_id} on {task_uid} window {window_idx}"
            )
        predictions = tracks.to_predictions(result.quantiles, result.point, inputs, quantile_levels)
        predictions_per_window.append(predictions)
        stored_quantiles.append(result.quantiles.astype(np.float32))
        stored_point.append(result.point.astype(np.float32))
        stored_window.append(np.full(result.quantiles.shape[0], window_idx, dtype=np.int16))
        decomposition = scoring.per_origin_sql(window, predictions, quantile_levels, seasonality)
        origin_losses.append(decomposition["origin_loss"])
        origin_keys.extend(
            [f"w{window_idx}::{item}" for item in decomposition["item_ids"]]
        )

    summary = task.evaluation_summary(
        predictions_per_window,
        model_name=model_id,
        inference_time_s=total_seconds,
        trained_on_this_dataset=model_id == "linear-ar-specialist",
    )
    origin_loss = np.concatenate(origin_losses)
    origin_windows = np.concatenate(
        [np.full(part.size, idx) for idx, part in enumerate(origin_losses)]
    )
    official = float(summary["test_error"])
    aggregate_from_origins = scoring.task_aggregate(origin_loss, origin_windows)

    np.savez_compressed(
        forecast_path(model_id, track, task_uid),
        quantiles=np.concatenate(stored_quantiles, axis=0),
        point=np.concatenate(stored_point, axis=0),
        window_index=np.concatenate(stored_window, axis=0),
        quantile_levels=np.asarray(quantile_levels, dtype=np.float64),
    )
    np.savez_compressed(
        _origin_path(model_id, track, task_uid),
        origin_loss=origin_loss.astype(np.float32),
        origin_key=np.array(origin_keys, dtype=object),
        origin_window=origin_windows.astype(np.int16),
    )

    return {
        "benchmark": "fev-bench",
        "track": track,
        "task_uid": task_uid,
        "model": model_id,
        "model_role": REGISTRY.get(model_id, {}).get("role", "baseline"),
        "native_metric": task.eval_metric,
        "native_score": official,
        "origin_aggregate": aggregate_from_origins,
        "origin_reconciliation_gap": scoring.reconcile(aggregate_from_origins, official),
        "n_origins": int(origin_loss.size),
        "context_mode": CONTEXT_MODE,
        "actual_context_min": int(min(c["min"] for c in context_used)),
        "actual_context_median": int(np.median([c["median"] for c in context_used])),
        "actual_context_max": int(max(c["max"] for c in context_used)),
        "future_covariates_used": track == "C",
        "known_columns_used": json.dumps(known_used),
        "known_columns_skipped": json.dumps(
            [c for c in task.known_dynamic_columns if c not in known_used] if track == "C" else []
        ),
        "multivariate_mode": "native" if track == "M" else ("none" if track == "U" else "targets_only"),
        "n_windows": int(task.num_windows),
        "n_targets": len(task.target_columns),
        "inference_seconds": total_seconds,
        "peak_memory_mb": peak_memory if peak_memory else float("nan"),
        "status": "OK",
        "notes": json.dumps(notes),
        "extra_metrics": json.dumps(
            {k: v for k, v in summary.items() if k.startswith("test_error[") or k in ("MASE", "WAPE", "WQL")}
        ),
        "summary": json.dumps({k: v for k, v in summary.items() if isinstance(v, (int, float, str))}),
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["discovery", "confirmation"], default="discovery")
    parser.add_argument("--track", choices=["U", "M", "C"], default="U")
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--only-task", default=None)
    args = parser.parse_args()

    selected = _selected()
    configs = _task_configs()
    pool = pd.read_csv(paths.RESULTS / "task_pool.csv").set_index("task_uid")
    task_uids = selected[f"{args.split}_tasks"]
    if args.only_task:
        task_uids = [t for t in task_uids if t == args.only_task]

    model_ids = args.models or (list(REGISTRY) + list(BASELINES))
    for model_id in model_ids:
        adapter = None
        loaded = False
        for task_uid in task_uids:
            pool_row = pool.loc[task_uid]
            if args.track not in tracks.applicable_tracks(pool_row):
                continue
            cache = _cache_path(model_id, args.track, task_uid)
            if cache.exists():
                print(f"cached  {model_id:20s} track{args.track} {task_uid}", flush=True)
                continue
            if adapter is None and model_id not in BASELINES:
                adapter = build(model_id)
            if not loaded and model_id not in BASELINES:
                start = time.perf_counter()
                adapter.load()
                print(f"loaded  {model_id} in {time.perf_counter() - start:.1f}s", flush=True)
                loaded = True
            print(f"running {model_id:20s} track{args.track} {task_uid}", flush=True)
            started = time.perf_counter()
            try:
                row = run_one(
                    model_id, args.track, task_uid, configs[task_uid], pool_row, adapter=adapter
                )
            except Exception as exc:  # noqa: BLE001 - recorded, not silently dropped
                row = {
                    "benchmark": "fev-bench",
                    "track": args.track,
                    "task_uid": task_uid,
                    "model": model_id,
                    "status": f"FAILED: {type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc()[-2000:],
                    "run_at_utc": datetime.now(timezone.utc).isoformat(),
                }
                print(f"  FAILED {type(exc).__name__}: {exc}", flush=True)
            cache.write_text(tracks.dumps(row), encoding="utf-8")
            # Long-context multivariate calls leave large intermediates behind; drop
            # them between tasks so resident memory does not climb across a sweep.
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(
                f"  -> {row.get('status')} score={row.get('native_score')} "
                f"wall={time.perf_counter() - started:.1f}s",
                flush=True,
            )
        if adapter is not None:
            adapter.unload()


if __name__ == "__main__":
    main()
