"""Float64 linear controls for the frozen study19 revision protocol."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Hashable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class RidgeConfig:
    lam: float
    window: int | None = None


def candidate_configs() -> tuple[RidgeConfig, ...]:
    return tuple(RidgeConfig(lam, window) for lam in (0.01, 0.1, 1.0, 10.0, 100.0) for window in (None, 120))


def select_config(scores: Sequence[tuple[RidgeConfig, float]]) -> RidgeConfig:
    if not scores or not all(np.isfinite(score) for _, score in scores):
        raise ValueError('Configuration scores must be nonempty and finite.')
    return min(scores, key=lambda item: (item[1], item[0].lam, item[0].window is not None))[0]


def _validate_xy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 1 or x.shape[0] != y.size or x.shape[0] == 0 or x.shape[1] == 0:
        raise ValueError('Expected a nonempty feature matrix and one scalar label per row.')
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('All training features and labels must be finite; rows are not dropped.')
    if not np.equal(x[:, 0], 1.0).all():
        raise ValueError('The first feature must be the intercept, exactly one.')
    return x, y


def _penalty(n_features: int, lam: float) -> np.ndarray:
    if not np.isfinite(lam) or lam <= 0:
        raise ValueError('Ridge lambda must be positive and finite.')
    penalty = np.eye(n_features, dtype=np.float64) * lam
    penalty[0, 0] = 0.0
    return penalty


def fit_ridge(x: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    x, y = _validate_xy(x, y)
    return np.linalg.solve(x.T @ x + _penalty(x.shape[1], lam), x.T @ y)


def ridge(
    x: np.ndarray,
    y: np.ndarray,
    x_prediction: np.ndarray,
    lam: float,
    window: int | None,
    origin: int,
    event_months: np.ndarray,
    min_train: int = 100,
) -> dict:
    """Fit historical rows in [origin-window, origin); arrival checks belong to the caller."""
    x, y = _validate_xy(x, y)
    events = np.asarray(event_months)
    if events.shape != y.shape or not np.isfinite(events).all():
        raise ValueError('Each training row needs a finite event month ordinal.')
    if np.unique(events).size != events.size:
        raise ValueError('Duplicate training event months would double-count samples.')
    if np.any(events >= origin):
        raise ValueError('Training events must be strictly historical to the prediction origin.')
    if window not in (None, 120):
        raise ValueError('Study19 supports only expanding or 120-month windows.')
    keep = np.ones(y.size, dtype=bool) if window is None else events >= origin - window
    n_train = int(keep.sum())
    if n_train < min_train:
        raise ValueError(f'Only {n_train} eligible rows; require at least {min_train}.')
    point = np.asarray(x_prediction, dtype=np.float64)
    if point.shape != (x.shape[1],) or not np.isfinite(point).all() or point[0] != 1.0:
        raise ValueError('Prediction features must be finite, aligned and include the intercept.')
    coef = fit_ridge(x[keep], y[keep], lam)
    return {
        'prediction': float(point @ coef),
        'coefficients': coef.tolist(),
        'n_train': n_train,
        'lam': float(lam),
        'window': window,
    }


def predict_arms(
    arm_data: Mapping[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    x_prediction: np.ndarray,
    configs: Mapping[str, RidgeConfig],
    origin_month: int,
    first_bias: float = 0.0,
    last_known_growth: float | None = None,
    min_train: int = 100,
) -> dict[str, dict]:
    """Use caller-built causal labels/features; FIRST_BIAS has its own selected config."""
    if last_known_growth is None or not np.isfinite(last_known_growth) or not np.isfinite(first_bias):
        raise ValueError('Provide finite first_bias and the actually last known growth.')
    results = {
        'ZERO_GROWTH': {'prediction': 0.0, 'n_train': 0, 'lam': None, 'window': None},
        'LAST_KNOWN_GROWTH': {'prediction': float(last_known_growth), 'n_train': 0, 'lam': None, 'window': None},
    }
    for arm, config in configs.items():
        source_arm = 'FIRST_FIXED_X' if arm == 'FIRST_BIAS' else arm
        x, y, events = arm_data[source_arm]
        result = ridge(x, y, x_prediction, config.lam, config.window, origin_month, events, min_train)
        if arm == 'FIRST_BIAS':
            result['prediction'] += float(first_bias)
            result['bias'] = float(first_bias)
        results[arm] = result
    return results


def revision_bias(
    provisional: np.ndarray,
    mature: np.ndarray,
    maturity_dates: Sequence[date],
    cutoff: date,
    *,
    ages: Sequence[int] | None = None,
    age: int | None = None,
) -> tuple[float, int]:
    """Mean revision from rows matured by the inclusive cutoff, optionally at one age."""
    provisional = np.asarray(provisional, dtype=np.float64)
    mature = np.asarray(mature, dtype=np.float64)
    dates = np.asarray(maturity_dates, dtype='datetime64[D]')
    if provisional.ndim != 1 or mature.shape != provisional.shape or dates.shape != provisional.shape:
        raise ValueError('Revision samples, mature values and dates must align.')
    if np.isnat(dates).any():
        raise ValueError('Maturity dates must be known.')
    keep = dates <= np.datetime64(cutoff, 'D')
    if (ages is None) != (age is None):
        raise ValueError('Specify both ages and the requested age, or neither.')
    if ages is not None:
        sample_ages = np.asarray(ages)
        if sample_ages.shape != provisional.shape:
            raise ValueError('Age coordinates must align with revision samples.')
        keep &= sample_ages == age
    if not np.isfinite(provisional[keep]).all() or not np.isfinite(mature[keep]).all():
        raise ValueError('Already mature correction samples must be finite.')
    n = int(keep.sum())
    return (float(np.mean(mature[keep] - provisional[keep])), n) if n else (0.0, 0)


class RevisionRidge:
    """Exact expanding ridge sufficient statistics for fixed-feature label corrections."""

    def __init__(self, n_features: int = 7, lam: float = 1.0):
        if n_features < 1:
            raise ValueError('At least an intercept feature is required.')
        self.lam = float(lam)
        self.A = _penalty(n_features, self.lam)
        self.b = np.zeros(n_features, dtype=np.float64)
        self._rows: dict[Hashable, tuple[np.ndarray, float]] = {}

    @property
    def n_rows(self) -> int:
        return len(self._rows)

    def update(self, row_id: Hashable, x: np.ndarray, y: float) -> None:
        point = np.asarray(x, dtype=np.float64)
        if point.shape != self.b.shape or not np.isfinite(point).all() or point[0] != 1.0 or not np.isfinite(y):
            raise ValueError('Revision row must have finite, aligned fixed features and label.')
        if row_id in self._rows:
            old_x, old_y = self._rows[row_id]
            if not np.array_equal(point, old_x):
                raise ValueError('This identity tracker requires fixed features for each row.')
            self.b += point * (float(y) - old_y)
        else:
            self.A += np.outer(point, point)
            self.b += point * float(y)
        self._rows[row_id] = (point.copy(), float(y))

    def coefficients(self) -> np.ndarray:
        if not self._rows:
            raise ValueError('No rows have arrived.')
        return np.linalg.solve(self.A, self.b)

    def predict(self, x_prediction: np.ndarray) -> float:
        point = np.asarray(x_prediction, dtype=np.float64)
        if point.shape != self.b.shape or not np.isfinite(point).all() or point[0] != 1.0:
            raise ValueError('Prediction features must be finite and aligned.')
        return float(point @ self.coefficients())

    def compare_batch(self, x_prediction: np.ndarray, tolerance: float = 1e-8) -> dict:
        if not self._rows:
            raise ValueError('No rows have arrived.')
        x = np.stack([row[0] for row in self._rows.values()])
        y = np.asarray([row[1] for row in self._rows.values()], dtype=np.float64)
        batch_a = x.T @ x + _penalty(self.b.size, self.lam)
        batch_b = x.T @ y
        batch_coef = np.linalg.solve(batch_a, batch_b)
        coef_error = float(np.max(np.abs(self.coefficients() - batch_coef)))
        prediction_error = abs(self.predict(x_prediction) - float(np.asarray(x_prediction) @ batch_coef))
        return {
            'n_rows': self.n_rows,
            'a_max_abs': float(np.max(np.abs(self.A - batch_a))),
            'b_max_abs': float(np.max(np.abs(self.b - batch_b))),
            'coef_max_abs': coef_error,
            'prediction_abs': prediction_error,
            'tolerance': float(tolerance),
            'passed': bool(coef_error <= tolerance and prediction_error <= tolerance),
        }
