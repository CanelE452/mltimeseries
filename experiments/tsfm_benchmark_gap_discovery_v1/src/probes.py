"""The headroom probe bank (Section 18).

Every probe is either a *diagnostic oracle* - allowed to see the evaluation
labels, never reported as achievable - or a *simple deployable fix* that uses
only information available before the origin it forecasts. The distinction is
carried on every row as `deployable`, and `verify` checks that no oracle value
leaks into a deployable comparison.

The rolling-origin structure of fev-bench gives the deployable probes their
validation signal for free: to forecast window k they may use windows 0..k-1 and
nothing later. Window 0 has no such history, so deployable probes are scored on
windows 1..K-1 and every quantity they are compared against is restricted to the
same windows.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import yaml

from . import paths, scoring, tracks
from .models import PRIMARY_MODELS
from .run_models import forecast_path

ORACLE_PROBES = {"P0_cross_model_origin_oracle"}


def _load_origins(model: str, track: str, task_uid: str):
    safe = task_uid.replace("::", "__").replace("/", "-")
    path = paths.PRED_CACHE / f"{model}__track{track}__{safe}__origins.npz"
    if not path.exists():
        return None
    payload = np.load(path, allow_pickle=True)
    keys = np.array([str(k) for k in payload["origin_key"]])
    windows = np.array([int(k.split("::")[0][1:]) for k in keys])
    return payload["origin_loss"].astype(np.float64), keys, windows


def _load_forecasts(model: str, track: str, task_uid: str):
    path = forecast_path(model, track, task_uid)
    if not path.exists():
        return None
    payload = np.load(path)
    return payload["quantiles"], payload["point"], payload["window_index"]


def _task(task_uid: str):
    import fev

    raw = yaml.safe_load(
        (paths.DATA_EXTERNAL / "fev_bench_tasks.yaml").read_text(encoding="utf-8")
    )["tasks"]
    pool = pd.read_csv(paths.RESULTS / "task_pool.csv").set_index("task_uid")
    task = fev.Task(**raw[int(pool.loc[task_uid, "yaml_index"])])
    task.load_full_dataset(num_proc=1)
    return task


def _score_arrays(task, track, quantiles, point, window_index) -> tuple[np.ndarray, np.ndarray]:
    """Recompute per-origin losses for arbitrary forecast arrays."""
    quantile_levels = list(task.quantile_levels)
    seasonality = int(task.seasonality)
    losses, windows = [], []
    for window_idx in sorted(set(window_index.tolist())):
        window = task.get_window(int(window_idx), num_proc=1)
        inputs = tracks.build_inputs(window, track, task)
        mask = window_index == window_idx
        predictions = tracks.to_predictions(
            quantiles[mask].astype(np.float64), point[mask].astype(np.float64), inputs, quantile_levels
        )
        decomposition = scoring.per_origin_sql(window, predictions, quantile_levels, seasonality)
        losses.append(decomposition["origin_loss"])
        windows.append(np.full(decomposition["origin_loss"].size, window_idx))
    return np.concatenate(losses), np.concatenate(windows)


def p0_cross_model_origin_oracle(track: str, task_uid: str, models: list[str]) -> dict | None:
    """Lowest evaluated loss per origin. Uses test labels; never deployable."""
    stack, windows = [], None
    for model in models:
        loaded = _load_origins(model, track, task_uid)
        if loaded is None:
            return None
        loss, _, window = loaded
        stack.append(loss)
        windows = window
    matrix = np.vstack(stack)
    return {
        "probe": "P0_cross_model_origin_oracle",
        "deployable": False,
        "origin_loss": np.nanmin(matrix, axis=0),
        "windows": windows,
        "detail": {"models": models},
    }


def p1_validation_selected_model(track: str, task_uid: str, models: list[str]) -> dict | None:
    """Pick the model that led on every earlier window; apply it to this one."""
    loaded = {}
    for model in models:
        entry = _load_origins(model, track, task_uid)
        if entry is None:
            return None
        loaded[model] = entry
    windows = loaded[models[0]][2]
    unique = sorted(set(windows.tolist()))
    if len(unique) < 2:
        return None
    selected_loss = np.full(windows.size, np.nan)
    choices = {}
    for window_idx in unique[1:]:
        earlier = windows < window_idx
        means = {m: float(np.nanmean(loaded[m][0][earlier])) for m in models}
        winner = min(means, key=means.get)
        choices[int(window_idx)] = winner
        mask = windows == window_idx
        selected_loss[mask] = loaded[winner][0][mask]
    return {
        "probe": "P1_validation_selected_model",
        "deployable": True,
        "origin_loss": selected_loss,
        "windows": windows,
        "detail": {"per_window_choice": choices},
    }


def p3_affine_calibration(track: str, task_uid: str, model: str) -> dict | None:
    """Fit one multiplicative and one additive correction on earlier windows.

    The correction is a single scalar pair per task, estimated by least squares
    against the observed targets of windows 0..k-1 and applied unchanged to every
    quantile of window k.
    """
    import fev

    forecasts = _load_forecasts(model, track, task_uid)
    if forecasts is None:
        return None
    quantiles, point, window_index = forecasts
    task = _task(task_uid)
    unique = sorted(set(window_index.tolist()))
    if len(unique) < 2:
        return None

    truths, medians = {}, {}
    median_index = list(task.quantile_levels).index(0.5)
    for window_idx in unique:
        window = task.get_window(int(window_idx), num_proc=1)
        _, _, test_data = window._get_past_future_test_data()
        if track == "U":
            values = np.concatenate(
                [
                    np.asarray(test_data[column], dtype=np.float64)
                    for column in task.target_columns
                ],
                axis=0,
            )
            order = np.argsort(
                np.tile(np.arange(len(test_data)), len(task.target_columns)), kind="stable"
            )
            truths[window_idx] = values[order]
        else:
            truths[window_idx] = np.stack(
                [np.asarray(test_data[column], dtype=np.float64) for column in task.target_columns],
                axis=1,
            )
        mask = window_index == window_idx
        medians[window_idx] = quantiles[mask][..., median_index]

    corrected_q = quantiles.astype(np.float64).copy()
    corrected_p = point.astype(np.float64).copy()
    coefficients = {}
    for window_idx in unique[1:]:
        x_parts, y_parts = [], []
        for earlier in unique:
            if earlier >= window_idx:
                break
            prediction = medians[earlier].reshape(-1)
            truth = np.asarray(truths[earlier]).reshape(-1)
            if prediction.size != truth.size:
                return None
            finite = np.isfinite(prediction) & np.isfinite(truth)
            x_parts.append(prediction[finite])
            y_parts.append(truth[finite])
        x = np.concatenate(x_parts)
        y = np.concatenate(y_parts)
        if x.size < 32 or np.std(x) <= 0:
            scale, offset = 1.0, 0.0
        else:
            design = np.stack([x, np.ones_like(x)], axis=1)
            scale, offset = np.linalg.lstsq(design, y, rcond=None)[0]
            if not np.isfinite(scale) or not np.isfinite(offset) or scale <= 0:
                scale, offset = 1.0, 0.0
        coefficients[int(window_idx)] = {"scale": float(scale), "offset": float(offset)}
        mask = window_index == window_idx
        corrected_q[mask] = corrected_q[mask] * scale + offset
        corrected_p[mask] = corrected_p[mask] * scale + offset

    loss, windows = _score_arrays(task, track, corrected_q, corrected_p, window_index)
    loss = np.where(windows == unique[0], np.nan, loss)
    return {
        "probe": "P3_affine_calibration",
        "deployable": True,
        "origin_loss": loss,
        "windows": windows,
        "detail": {"model": model, "per_window_coefficients": coefficients},
    }


def p6_quantile_average_ensemble(track: str, task_uid: str, models: list[str]) -> dict | None:
    """Average the models' quantile forecasts level by level. No fitting at all."""
    stacks_q, stacks_p, window_index = [], [], None
    for model in models:
        forecasts = _load_forecasts(model, track, task_uid)
        if forecasts is None:
            return None
        quantiles, point, windows = forecasts
        stacks_q.append(quantiles.astype(np.float64))
        stacks_p.append(point.astype(np.float64))
        window_index = windows
    mean_q = np.mean(np.stack(stacks_q, axis=0), axis=0)
    mean_q = np.sort(mean_q, axis=-1)  # keep quantiles monotone after averaging
    mean_p = np.mean(np.stack(stacks_p, axis=0), axis=0)
    task = _task(task_uid)
    loss, windows = _score_arrays(task, track, mean_q, mean_p, window_index)
    return {
        "probe": "P6_quantile_average_ensemble",
        "deployable": True,
        "origin_loss": loss,
        "windows": windows,
        "detail": {"models": models},
    }


def p5_covariate_ablation(track: str, task_uid: str, model: str) -> dict | None:
    """The same model, same task, with the officially supported known-future covariates.

    This crosses information conditions on purpose: it is the ablation that says
    whether a deficit measured without covariates is closed by supplying them.
    Only deployable information is used - the covariates fev marks as known ahead
    of time - and it is only defined for TRACK U rows whose task carries them.
    """
    if track != "U":
        return None
    loaded = _load_origins(model, "C", task_uid)
    if loaded is None:
        return None
    loss, _, windows = loaded
    reference = _load_origins(model, "U", task_uid)
    if reference is None or reference[0].shape != loss.shape:
        return None
    return {
        "probe": "P5_covariate_ablation",
        "deployable": True,
        "origin_loss": loss,
        "windows": windows,
        "detail": {
            "model": model,
            "from_track": "C",
            "note": "same model and task, given the task's known-future covariates",
        },
    }


def p4_specialist(track: str, task_uid: str) -> dict | None:
    """The trained specialist as-is: how much of the gap is adaptation, not architecture."""
    loaded = _load_origins("linear-ar-specialist", track, task_uid)
    if loaded is None:
        return None
    loss, _, windows = loaded
    return {
        "probe": "P4_supervised_specialist",
        "deployable": True,
        "origin_loss": loss,
        "windows": windows,
        "detail": {"model": "linear-ar-specialist"},
    }


def strongest_model(track: str, tasks: list[str], models: list[str]) -> tuple[str, dict]:
    """One foundation model for the whole task set, not a different one per task.

    Picking the winner separately on each task would make the comparator itself a
    per-task oracle and shrink the measured headroom for the wrong reason. The
    choice here is still made with the scores in view, which if anything makes the
    baseline stronger and the reported oracle headroom smaller.
    """
    aggregates = {}
    for model in models:
        per_task = []
        for task_uid in tasks:
            loaded = _load_origins(model, track, task_uid)
            if loaded is not None:
                per_task.append(scoring.task_aggregate(loaded[0], loaded[2]))
        if per_task:
            aggregates[model] = float(np.mean(per_task))
    best = min(aggregates, key=aggregates.get)
    return best, aggregates


def baseline_reference(track: str, task_uid: str, model: str, aggregates: dict) -> dict:
    """The chosen single foundation model, evaluated on this task."""
    loss, _, windows = _load_origins(model, track, task_uid)
    return {
        "probe": "BASELINE_strongest_foundation_model",
        "deployable": True,
        "origin_loss": loss,
        "windows": windows,
        "detail": {
            "model": model,
            "chosen": "single model for the whole task set",
            "per_model_mean_aggregate": aggregates,
        },
    }


def run_bank(
    track: str,
    task_uid: str,
    models: list[str] | None = None,
    baseline_model: str | None = None,
    aggregates: dict | None = None,
) -> list[dict]:
    models = models or PRIMARY_MODELS
    if baseline_model is None:
        baseline_model, aggregates = strongest_model(track, [task_uid], models)
    probes = [
        baseline_reference(track, task_uid, baseline_model, aggregates or {}),
        p0_cross_model_origin_oracle(track, task_uid, models),
        p1_validation_selected_model(track, task_uid, models),
        p6_quantile_average_ensemble(track, task_uid, models),
        p4_specialist(track, task_uid),
        p3_affine_calibration(track, task_uid, baseline_model),
        p5_covariate_ablation(track, task_uid, baseline_model),
    ]
    return [probe for probe in probes if probe is not None]
