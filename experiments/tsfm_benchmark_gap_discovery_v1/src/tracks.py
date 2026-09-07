"""Information conditions (tracks) and the shared input contract.

A track fixes exactly what every model may look at. Section 6 of the study
contract forbids pooling different information conditions into one table, so the
track is carried on every result row.

TRACK U  common univariate, no covariates. Multi-target tasks are expanded with
         fev's own `as_univariate` mode, so every model sees the same rows.
TRACK M  native multivariate targets, no covariates. Only for tasks with >1 target.
TRACK C  targets plus officially supported known-future covariates. Only for tasks
         that declare known_dynamic_columns.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

TRACKS = ("U", "M", "C")


@dataclass(frozen=True)
class WindowInputs:
    """What a model receives for one evaluation window, identical across models."""

    track: str
    # One entry per forecast unit. In TRACK U a unit is a single series/target
    # pair; in TRACK M and C a unit is one series with all of its targets.
    targets: list[np.ndarray]  # each (n_variates, context_length)
    past_covariates: list[np.ndarray | None]  # each (n_past_cov, context_length)
    future_covariates: list[np.ndarray | None]  # each (n_known_cov, horizon)
    past_covariate_history: list[np.ndarray | None]  # known covariates over the context
    target_columns: list[str]
    known_columns: list[str]
    horizon: int
    n_items: int  # number of series rows in the window
    context_lengths: np.ndarray


def _stack(rows: list, columns: list[str]) -> np.ndarray | None:
    if not columns:
        return None
    return np.stack([np.asarray(rows[column], dtype=np.float64) for column in columns], axis=0)


def _numeric_known_columns(window, past_data) -> list[str]:
    """Known-future columns fev can hand every model as plain numbers.

    Categorical known covariates are excluded rather than encoded, because an
    encoding chosen here would be a per-model design decision and Section 10
    forbids per-model tuning.
    """
    numeric = []
    for column in window.known_dynamic_columns:
        sample = past_data[0][column]
        if len(sample) and isinstance(sample[0], (int, float, np.integer, np.floating)):
            numeric.append(column)
    return numeric


def build_inputs(window, track: str, task) -> WindowInputs:
    import fev

    horizon = window.horizon
    if track == "U":
        past, future = fev.convert_input_data(window, adapter="datasets", as_univariate=True)
        targets = [np.asarray(row, dtype=np.float64)[None, :] for row in past["target"]]
        n_items = len(past)
        lengths = np.array([t.shape[1] for t in targets])
        return WindowInputs(
            track="U",
            targets=targets,
            past_covariates=[None] * n_items,
            future_covariates=[None] * n_items,
            past_covariate_history=[None] * n_items,
            target_columns=list(window.target_columns),
            known_columns=[],
            horizon=horizon,
            n_items=n_items,
            context_lengths=lengths,
        )

    past, future = window.get_input_data()
    target_columns = list(window.target_columns)
    known = _numeric_known_columns(window, past) if track == "C" else []
    n_items = len(past)
    targets, future_cov, past_cov_hist = [], [], []
    for i in range(n_items):
        past_row = past[i]
        targets.append(_stack(past_row, target_columns))
        if known:
            future_row = future[i]
            future_cov.append(_stack(future_row, known)[:, :horizon])
            past_cov_hist.append(_stack(past_row, known))
        else:
            future_cov.append(None)
            past_cov_hist.append(None)
    lengths = np.array([t.shape[1] for t in targets])
    return WindowInputs(
        track=track,
        targets=targets,
        past_covariates=[None] * n_items,
        future_covariates=future_cov,
        past_covariate_history=past_cov_hist,
        target_columns=target_columns,
        known_columns=known,
        horizon=horizon,
        n_items=n_items,
        context_lengths=lengths,
    )


def applicable_tracks(pool_row: pd.Series) -> list[str]:
    tracks = ["U"]
    if bool(pool_row.is_multivariate):
        tracks.append("M")
    if bool(pool_row.has_known_cov):
        tracks.append("C")
    return tracks


def to_predictions(
    quantiles: np.ndarray,
    point: np.ndarray,
    inputs: WindowInputs,
    quantile_levels: list[float],
):
    """Assemble fev's DatasetDict from raw arrays.

    quantiles: (n_units, n_variates, horizon, n_quantiles)
    point:     (n_units, n_variates, horizon)

    In TRACK U the units cycle through target columns exactly as
    `fev.convert_input_data(..., as_univariate=True)` emits them, which is the
    ordering `fev.utils.convert_forecast_df_to_predictions` documents.
    """
    import datasets

    columns = inputs.target_columns
    n_targets = len(columns)
    prediction_dict = {}
    if inputs.track == "U":
        for i, column in enumerate(columns):
            rows = slice(i, None, n_targets)
            data = {"predictions": point[rows, 0, :]}
            for qi, q in enumerate(quantile_levels):
                data[str(q)] = quantiles[rows, 0, :, qi]
            prediction_dict[column] = datasets.Dataset.from_dict(
                {k: v.astype(np.float64) for k, v in data.items()}
            )
    else:
        for i, column in enumerate(columns):
            data = {"predictions": point[:, i, :]}
            for qi, q in enumerate(quantile_levels):
                data[str(q)] = quantiles[:, i, :, qi]
            prediction_dict[column] = datasets.Dataset.from_dict(
                {k: v.astype(np.float64) for k, v in data.items()}
            )
    return datasets.DatasetDict(prediction_dict)


def track_description(track: str) -> str:
    return {
        "U": "common univariate, no covariates (fev as_univariate mode)",
        "M": "native multivariate targets, no covariates",
        "C": "targets plus numeric known-future covariates",
    }[track]


def json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON serialisable: {type(value)}")


def dumps(payload) -> str:
    return json.dumps(payload, indent=2, default=json_default)
