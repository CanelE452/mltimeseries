"""Smoke test: every model on one small real task, before any full run.

Checks what Section 12 asks for - the model loads on the expected device, output
shapes and quantile levels match the task contract, forecasts are finite, the
native evaluator accepts them, and the per-origin decomposition reconciles with
the official aggregate. Also measures latency so the runtime budget can be
projected before committing to the full sweep.
"""

from __future__ import annotations

import json
import time
import traceback
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
import yaml

from . import paths, scoring, tracks
from .baselines import BASELINES
from .models import DEVICE, REGISTRY, build, configure_environment

configure_environment()

# One target, one covariate-free window set, 20 windows of horizon 168 over a
# single series: small enough to be fast, real enough to exercise the contract.
SMOKE_TASK = "fevbench::bizitobs_l2c_1H::bizitobs_l2c_1H"
SMOKE_WINDOWS = 2


def _task(task_uid: str):
    import fev

    raw = yaml.safe_load(
        (paths.DATA_EXTERNAL / "fev_bench_tasks.yaml").read_text(encoding="utf-8")
    )["tasks"]
    pool = pd.read_csv(paths.RESULTS / "task_pool.csv").set_index("task_uid")
    config = raw[int(pool.loc[task_uid, "yaml_index"])]
    task = fev.Task(**config)
    task.load_full_dataset(num_proc=1)
    return task, pool.loc[task_uid]


def _check_one(model_id: str, task, track: str) -> dict:
    import fev.metrics

    quantile_levels = list(task.quantile_levels)
    seasonality = int(task.seasonality)
    record: dict = {"model": model_id, "track": track}
    adapter = None
    try:
        if model_id not in BASELINES:
            adapter = build(model_id)
            start = time.perf_counter()
            adapter.load()
            record["load_seconds"] = round(time.perf_counter() - start, 2)
            record["device"] = DEVICE
        gaps, seconds, shapes = [], [], []
        for window_idx in range(min(SMOKE_WINDOWS, task.num_windows)):
            window = task.get_window(window_idx, num_proc=1)
            inputs = tracks.build_inputs(window, track, task)
            if model_id in BASELINES:
                result = BASELINES[model_id](inputs, quantile_levels, seasonality)
            else:
                result = adapter.forecast(inputs, quantile_levels)
            shapes.append(list(result.quantiles.shape))
            seconds.append(result.seconds)
            expected = (
                inputs.n_items,
                inputs.targets[0].shape[0],
                inputs.horizon,
                len(quantile_levels),
            )
            if tuple(result.quantiles.shape) != expected:
                raise AssertionError(f"quantile shape {result.quantiles.shape} != {expected}")
            if not np.isfinite(result.quantiles).all():
                raise AssertionError("non-finite quantile forecast")
            predictions = tracks.to_predictions(
                result.quantiles, result.point, inputs, quantile_levels
            )
            metrics = window.compute_metrics(
                predictions,
                metrics=[fev.metrics.get_metric(task.eval_metric)],
                seasonality=seasonality,
                quantile_levels=quantile_levels,
            )
            decomposition = scoring.per_origin_sql(
                window, predictions, quantile_levels, seasonality
            )
            gaps.append(
                scoring.reconcile(
                    decomposition["aggregate_from_origins"], float(metrics[task.eval_metric])
                )
            )
            record.setdefault("window_scores", []).append(float(metrics[task.eval_metric]))
            record["n_items"] = inputs.n_items
            record["horizon"] = inputs.horizon
            record["context_median"] = int(np.median(inputs.context_lengths))
        record.update(
            {
                "quantile_shapes": shapes,
                "seconds_per_window": [round(s, 3) for s in seconds],
                "forecasts_per_second": round(
                    record["n_items"] / max(np.mean(seconds), 1e-9), 1
                ),
                "origin_reconciliation_gap_max": float(np.nanmax(gaps)),
                "peak_memory_mb": round(result.peak_memory_mb, 1)
                if np.isfinite(result.peak_memory_mb)
                else None,
                "notes": result.notes,
                "status": "OK",
            }
        )
    except Exception as exc:  # noqa: BLE001 - a blocked model is a recorded outcome
        record["status"] = f"BLOCKED_ENV: {type(exc).__name__}: {exc}"
        record["traceback"] = traceback.format_exc()[-1500:]
    finally:
        if adapter is not None:
            adapter.unload()
    return record


def kernel_agreement() -> dict:
    """TiRex-2 runs unfused kernels here; measure CPU vs CUDA agreement."""
    from tirex2 import TimeseriesType, load_model

    from .models import TiRex2Adapter

    probe = TiRex2Adapter("tirex-2-general", "NX-AI/TiRex-2")
    probe.load()  # installs the native-kernel patch process-wide
    context = np.sin(np.arange(1024) / 11.0).astype("float32")
    outputs = {}
    for device in ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",):
        model = load_model("NX-AI/TiRex-2", device=device)
        series = TimeseriesType(
            target=torch.tensor(context).unsqueeze(0), past_covariates=None, future_covariates=None
        )
        outputs[device] = np.asarray(
            model.forecast([series], prediction_length=96, output_type="numpy")[0]
        )
        del model
    if "cuda" not in outputs:
        return {"available": False}
    difference = np.abs(outputs["cpu"] - outputs["cuda"])
    return {
        "available": True,
        "max_abs_difference": float(difference.max()),
        "max_relative_difference": float(
            difference.max() / (np.abs(outputs["cpu"]).max() + 1e-12)
        ),
    }


def main() -> None:
    task, pool_row = _task(SMOKE_TASK)
    report = {
        "smoke_task": SMOKE_TASK,
        "ran_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": DEVICE,
        "task_contract": {
            "eval_metric": task.eval_metric,
            "quantile_levels": list(task.quantile_levels),
            "horizon": task.horizon,
            "num_windows": task.num_windows,
            "seasonality": task.seasonality,
            "target_columns": list(task.target_columns),
            "freq": task.freq,
        },
        "windows_checked": SMOKE_WINDOWS,
        "results": [],
    }
    for model_id in list(REGISTRY) + list(BASELINES):
        print(f"smoke {model_id} ...", flush=True)
        record = _check_one(model_id, task, track="U")
        print(f"  {record['status']}", flush=True)
        report["results"].append(record)
    print("kernel agreement probe ...", flush=True)
    try:
        report["tirex2_kernel_agreement"] = kernel_agreement()
    except Exception as exc:  # noqa: BLE001
        report["tirex2_kernel_agreement"] = {"error": f"{type(exc).__name__}: {exc}"}

    ok = [r for r in report["results"] if r["status"] == "OK"]
    report["n_ok"] = len(ok)
    report["n_blocked"] = len(report["results"]) - len(ok)
    out = paths.RESULTS / "smoke_report.json"
    out.write_text(json.dumps(report, indent=2, default=tracks.json_default), encoding="utf-8")
    print(f"\nwrote {out}")
    for record in report["results"]:
        print(
            f"  {record['model']:22s} {record['status'][:60]:60s} "
            f"fps={record.get('forecasts_per_second')} "
            f"gap={record.get('origin_reconciliation_gap_max')}"
        )
    print("kernel agreement:", report["tirex2_kernel_agreement"])


if __name__ == "__main__":
    main()
