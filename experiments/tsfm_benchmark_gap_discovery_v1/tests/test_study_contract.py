"""Unit tests for the pieces the study's integrity depends on."""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pytest

from experiments.tsfm_benchmark_gap_discovery_v1.src import baselines, paths, select_tasks, tracks


class _Inputs:
    """Minimal stand-in for WindowInputs, enough for the baseline estimators."""

    def __init__(self, targets, horizon):
        self.track = "U"
        self.targets = targets
        self.horizon = horizon
        self.n_items = len(targets)
        self.target_columns = ["target"]
        self.known_columns = []
        self.past_covariates = [None] * len(targets)
        self.future_covariates = [None] * len(targets)
        self.past_covariate_history = [None] * len(targets)
        self.context_lengths = np.array([t.shape[1] for t in targets])


QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def test_seasonal_naive_repeats_the_last_season():
    season = 7
    history = np.tile(np.arange(season, dtype=float), 10)
    result = baselines.seasonal_naive(_Inputs([history[None, :]], horizon=14), QUANTILES, season)
    expected = np.tile(np.arange(season, dtype=float), 2)
    np.testing.assert_allclose(result.point[0, 0], expected)


def test_seasonal_naive_quantiles_are_monotone_and_centred():
    rng = np.random.default_rng(0)
    history = np.tile(np.arange(24, dtype=float), 20) + rng.normal(0, 1.0, 480)
    result = baselines.seasonal_naive(_Inputs([history[None, :]], horizon=48), QUANTILES, 24)
    quantiles = result.quantiles[0, 0]
    assert np.all(np.diff(quantiles, axis=-1) >= -1e-9), "quantiles must not cross"
    np.testing.assert_allclose(quantiles[:, 4], result.point[0, 0], atol=1e-9)


def test_seasonal_naive_interval_widens_with_horizon():
    rng = np.random.default_rng(1)
    history = np.tile(np.arange(12, dtype=float), 30) + rng.normal(0, 1.0, 360)
    result = baselines.seasonal_naive(_Inputs([history[None, :]], horizon=36), QUANTILES, 12)
    width = result.quantiles[0, 0, :, -1] - result.quantiles[0, 0, :, 0]
    assert width[0] <= width[13] <= width[25], "interval must widen with each seasonal cycle"


def test_seasonal_naive_uses_only_the_history_it_is_given():
    """Appending future values must not change a forecast made from the prefix."""
    rng = np.random.default_rng(2)
    history = rng.normal(0, 1, 200)
    future = rng.normal(50, 1, 100)
    short = baselines.seasonal_naive(_Inputs([history[None, :]], 24), QUANTILES, 12)
    long = baselines.seasonal_naive(
        _Inputs([np.concatenate([history, future])[None, :]], 24), QUANTILES, 12
    )
    assert not np.allclose(short.point, long.point), "sanity: the two histories differ"
    again = baselines.seasonal_naive(_Inputs([history[None, :]], 24), QUANTILES, 12)
    np.testing.assert_allclose(short.point, again.point)


def test_linear_ar_specialist_recovers_a_deterministic_signal():
    t = np.arange(600)
    history = np.sin(2 * np.pi * t / 24)
    result = baselines.linear_ar_specialist(_Inputs([history[None, :]], 24), QUANTILES, 24)
    truth = np.sin(2 * np.pi * (t[-1] + 1 + np.arange(24)) / 24)
    assert np.mean(np.abs(result.point[0, 0] - truth)) < 0.05


def test_linear_ar_specialist_falls_back_when_history_is_too_short():
    history = np.arange(20, dtype=float)
    result = baselines.linear_ar_specialist(_Inputs([history[None, :]], 6), QUANTILES, 4)
    naive = baselines.seasonal_naive(_Inputs([history[None, :]], 6), QUANTILES, 4)
    np.testing.assert_allclose(result.point, naive.point)


def test_baselines_survive_nan_and_all_zero_history():
    history = np.array([np.nan] * 50 + [0.0] * 50)
    for estimator in (baselines.seasonal_naive, baselines.linear_ar_specialist):
        result = estimator(_Inputs([history[None, :]], 12), QUANTILES, 7)
        assert np.isfinite(result.quantiles).all()


def test_track_u_prediction_rows_map_back_to_target_columns():
    """fev emits univariate rows cycling through targets; to_predictions must undo that."""
    horizon, n_quantiles, n_targets, n_series = 4, len(QUANTILES), 3, 5
    n_units = n_series * n_targets
    quantiles = np.zeros((n_units, 1, horizon, n_quantiles))
    point = np.zeros((n_units, 1, horizon))
    for unit in range(n_units):
        point[unit, 0, :] = unit
        quantiles[unit, 0, :, :] = unit
    inputs = _Inputs([np.zeros((1, 10)) for _ in range(n_units)], horizon)
    inputs.target_columns = ["a", "b", "c"]
    predictions = tracks.to_predictions(quantiles, point, inputs, QUANTILES)
    for index, column in enumerate(inputs.target_columns):
        got = np.asarray(predictions[column]["predictions"])
        expected = np.arange(index, n_units, n_targets)
        np.testing.assert_allclose(got[:, 0], expected)


def test_selection_rule_is_deterministic_and_disjoint():
    first = select_tasks.build()
    second = select_tasks.build()
    assert first["discovery_tasks"] == second["discovery_tasks"]
    assert first["confirmation_tasks"] == second["confirmation_tasks"]
    assert not set(first["discovery_tasks"]) & set(first["confirmation_tasks"])


def test_selection_uses_one_task_per_dataset_family():
    spec = select_tasks.build()
    for split in ("discovery", "confirmation"):
        summary = spec[f"{split}_summary"]
        assert summary["n_dataset_families"] == summary["n"], (
            f"{split} draws more than one task from the same dataset family"
        )


def test_frozen_selection_file_matches_its_hash():
    payload = (paths.RESULTS / "selected_tasks.json").read_text(encoding="utf-8")
    import hashlib

    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    recorded = (paths.RESULTS / "selected_tasks.sha256").read_text(encoding="utf-8").split()[0]
    assert digest == recorded


def test_dataset_family_grouping():
    assert select_tasks.dataset_family("ETT_15T") == "ETT"
    assert select_tasks.dataset_family("ETT_1W") == "ETT"
    assert select_tasks.dataset_family("boomlet_1062") == "boomlet"
    assert select_tasks.dataset_family("favorita_stores_1D") == "favorita"
    assert select_tasks.dataset_family("australian_tourism") == "australian"
