"""Baselines: the naive anchor and one simple supervised specialist.

B0 SeasonalNaive is the denominator for every normalised score, so its contract
is spelled out here rather than inherited from a package:

    point(h)      = y[T - m + ((h - 1) mod m)]
    residual(t)   = y[t] - y[t - m]        over the visible history only
    sigma         = sqrt(mean(residual^2))
    sigma(h)      = sigma * sqrt(floor((h - 1) / m) + 1)
    quantile(h,q) = point(h) + z_q * sigma(h)

This is the standard seasonal random walk with Gaussian interval widening (the
same formula statsforecast's SeasonalNaive uses). The official fev wrapper calls
statsforecast, which on this host would downgrade pandas underneath three
already-verified foundation models, so the estimator is implemented directly and
covered by tests/test_baselines.py.

B1 is a direct multi-horizon ridge autoregression on lags and calendar-free
seasonal dummies: one linear map per horizon step, fitted per task on visible
history only. Section 9 prefers the benchmark's official supervised baseline,
which needs autogluon.timeseries; the same dependency conflict applies, and the
fallback the contract names next is a linear autoregressive model.
"""

from __future__ import annotations

import time

import numpy as np
from scipy.stats import norm

from .models import ForecastResult
from .tracks import WindowInputs

MIN_TRAIN_ROWS = 24
RIDGE_ALPHA = 1.0


def _seasonal_naive_unit(
    history: np.ndarray, horizon: int, seasonality: int, quantile_levels: list[float]
) -> tuple[np.ndarray, np.ndarray]:
    """history (T,) -> quantiles (horizon, Q), point (horizon,)."""
    clean = history[np.isfinite(history)]
    if clean.size == 0:
        clean = np.zeros(1)
    season = max(1, min(seasonality, clean.size))
    pattern = clean[-season:]
    steps = np.arange(horizon)
    point = pattern[steps % season]

    if clean.size > season:
        residuals = clean[season:] - clean[:-season]
        sigma = float(np.sqrt(np.mean(np.square(residuals)))) if residuals.size else 0.0
    else:
        sigma = float(np.std(clean)) if clean.size > 1 else 0.0
    widening = np.sqrt(np.floor(steps / season) + 1.0)
    scale = sigma * widening
    z = norm.ppf(np.asarray(quantile_levels))
    quantiles = point[:, None] + scale[:, None] * z[None, :]
    return quantiles, point


def seasonal_naive(
    inputs: WindowInputs, quantile_levels: list[float], seasonality: int
) -> ForecastResult:
    start = time.perf_counter()
    quantiles, point = [], []
    for target in inputs.targets:
        unit_q, unit_p = [], []
        for variate in target:
            q, p = _seasonal_naive_unit(variate, inputs.horizon, seasonality, quantile_levels)
            unit_q.append(q)
            unit_p.append(p)
        quantiles.append(np.stack(unit_q, axis=0))
        point.append(np.stack(unit_p, axis=0))
    return ForecastResult(
        quantiles=np.stack(quantiles, axis=0),
        point=np.stack(point, axis=0),
        seconds=time.perf_counter() - start,
        peak_memory_mb=float("nan"),
        notes={"estimator": "seasonal_naive", "seasonality": seasonality},
    )


def _lag_set(seasonality: int, max_lag_budget: int) -> list[int]:
    lags = {1, 2, 3}
    if seasonality > 1:
        lags.update({seasonality, 2 * seasonality, seasonality - 1, seasonality + 1})
    lags = sorted(lag for lag in lags if 1 <= lag <= max_lag_budget)
    return lags or [1]


def _fit_predict_unit(
    history: np.ndarray,
    horizon: int,
    seasonality: int,
    quantile_levels: list[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Direct multi-horizon ridge autoregression, fitted on this series' history."""
    clean = np.where(np.isfinite(history), history, np.nan)
    if np.isnan(clean).any():
        index = np.arange(clean.size)
        valid = ~np.isnan(clean)
        if valid.sum() == 0:
            clean = np.zeros_like(clean)
        else:
            clean = np.interp(index, index[valid], clean[valid])
    length = clean.size
    lags = _lag_set(seasonality, max_lag_budget=max(1, length // 3))
    max_lag = max(lags)
    n_rows = length - max_lag - horizon + 1
    if n_rows < MIN_TRAIN_ROWS:
        return _seasonal_naive_unit(history, horizon, seasonality, quantile_levels)

    centre = float(np.mean(clean))
    spread = float(np.std(clean))
    if spread <= 0:
        spread = 1.0
    scaled = (clean - centre) / spread

    origins = np.arange(max_lag, max_lag + n_rows)
    design = np.stack([scaled[origins - lag] for lag in lags], axis=1)
    design = np.concatenate([design, np.ones((n_rows, 1))], axis=1)
    outcomes = np.stack([scaled[origins + h] for h in range(horizon)], axis=1)

    gram = design.T @ design + RIDGE_ALPHA * np.eye(design.shape[1])
    coefficients = np.linalg.solve(gram, design.T @ outcomes)

    fitted = design @ coefficients
    residual_sigma = np.sqrt(np.mean(np.square(outcomes - fitted), axis=0))

    query = np.concatenate([[scaled[length - lag] for lag in lags], [1.0]])
    point_scaled = query @ coefficients
    point = point_scaled * spread + centre
    z = norm.ppf(np.asarray(quantile_levels))
    quantiles = point[:, None] + (residual_sigma * spread)[:, None] * z[None, :]
    return quantiles, point


def linear_ar_specialist(
    inputs: WindowInputs, quantile_levels: list[float], seasonality: int
) -> ForecastResult:
    start = time.perf_counter()
    quantiles, point = [], []
    for target in inputs.targets:
        unit_q, unit_p = [], []
        for variate in target:
            q, p = _fit_predict_unit(variate, inputs.horizon, seasonality, quantile_levels)
            unit_q.append(q)
            unit_p.append(p)
        quantiles.append(np.stack(unit_q, axis=0))
        point.append(np.stack(unit_p, axis=0))
    return ForecastResult(
        quantiles=np.stack(quantiles, axis=0),
        point=np.stack(point, axis=0),
        seconds=time.perf_counter() - start,
        peak_memory_mb=float("nan"),
        notes={
            "estimator": "linear_ar_specialist",
            "seasonality": seasonality,
            "ridge_alpha": RIDGE_ALPHA,
            "fitted_on": "visible history of each series only",
        },
    )


BASELINES = {
    "seasonal-naive": seasonal_naive,
    "linear-ar-specialist": linear_ar_specialist,
}
