"""Per-origin decomposition of the benchmark's own metric.

The task-level number always comes from fev's `Task.evaluation_summary`. This
module additionally reproduces the loss one origin at a time, reusing fev's own
`_quantile_loss` and `_abs_seasonal_error_per_item`, so that oracle probes and
task-internal resampling operate on the same quantity. Every call records how far
the mean of the per-origin losses lands from the official aggregate; the runner
stores that gap and `verify` checks it.
"""

from __future__ import annotations

import numpy as np
import pyarrow.compute as pc


def per_origin_sql(window, predictions, quantile_levels: list[float], seasonality: int) -> dict:
    """Return per-origin scaled quantile losses for one evaluation window.

    Shapes follow fev: N items, H horizon steps, D targets, Q quantile levels.
    """
    from fev.metrics import _abs_seasonal_error_per_item, _quantile_loss

    past_data, _, test_data = window._get_past_future_test_data()
    target_columns = list(window.target_columns)
    n_items = len(test_data)
    horizon = window.horizon
    n_targets = len(target_columns)
    n_quantiles = len(quantile_levels)

    test_table = test_data.data.table
    y_true = np.stack(
        [
            pc.list_flatten(test_table.column(column)).to_numpy(zero_copy_only=False)
            for column in target_columns
        ],
        axis=1,
        dtype=np.float64,
    ).reshape(n_items, horizon, n_targets)

    quantile_arrays = []
    for column in target_columns:
        table = predictions[column].data.table
        for q in quantile_levels:
            quantile_arrays.append(
                pc.list_flatten(table.column(str(q))).to_numpy(zero_copy_only=False)
            )
    q_pred = np.stack(quantile_arrays, axis=1, dtype=np.float64).reshape(
        n_items, horizon, n_targets, n_quantiles
    )

    past_table = past_data.data.table
    y_past = np.stack(
        [
            pc.list_flatten(past_table.column(column)).to_numpy(zero_copy_only=False)
            for column in target_columns
        ],
        axis=1,
        dtype=np.float64,
    )
    y_past_lengths = pc.list_value_length(past_table.column(target_columns[0])).to_numpy()

    quantile_loss = _quantile_loss(y_true=y_true, q_pred=q_pred, quantile_levels=quantile_levels)
    seasonal_error = _abs_seasonal_error_per_item(
        y_past=y_past, y_past_lengths=y_past_lengths, seasonality=seasonality
    )
    scaled = quantile_loss / seasonal_error[:, None, :, None]
    scaled = np.where(np.isfinite(scaled), scaled, np.nan)

    with np.errstate(invalid="ignore"):
        origin_loss = np.nanmean(scaled, axis=(1, 2, 3))
    item_ids = list(test_data[window.id_column])
    return {
        "origin_loss": origin_loss,
        "item_ids": item_ids,
        "n_items": n_items,
        "aggregate_from_origins": float(np.nanmean(origin_loss)),
    }


def task_aggregate(origin_loss: np.ndarray, windows: np.ndarray) -> float:
    """Aggregate per-origin losses the way fev aggregates a task.

    fev's `test_error` is the eval metric "averaged over all evaluation windows",
    an unweighted mean of the per-window scores. Pooling every origin instead
    would weight a window by how many series survived its cutoff, which differs
    whenever short series drop out of some windows and not others.
    """
    per_window = []
    for window_idx in np.unique(windows):
        values = origin_loss[windows == window_idx]
        if values.size and not np.all(np.isnan(values)):
            per_window.append(np.nanmean(values))
    if not per_window:
        return float("nan")
    return float(np.mean(per_window))


def reconcile(origin_aggregate: float, official: float) -> float:
    """Relative gap between the origin-level mean and the official task score."""
    if official == 0 or not np.isfinite(official):
        return float("nan")
    return abs(origin_aggregate - official) / abs(official)
