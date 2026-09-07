"""Uncertainty, kept at the level the design actually supports (Section 26).

Two levels, never mixed:

*Origin level.* Inside a fixed set of tasks the origins can be resampled, paired
across probes so that baseline, oracle and every simple fix are recomputed on the
same draw. Resampling happens inside each evaluation window, because the task
aggregate is a mean over windows and a resample that crossed windows would change
their weights. This gives an interval on the headroom arithmetic conditional on
the tasks, and says nothing about which tasks were drawn.

*Task level.* Twelve discovery tasks is too few for a task-level interval to mean
much, so the failure map reports the raw per-task effects and their count instead
of a confidence interval.
"""

from __future__ import annotations

import numpy as np

N_REPLICATES = 1000
SEED = 20260907


def _resample_within_windows(
    windows: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Indices of a bootstrap draw that keeps each window's origin count fixed."""
    picks = []
    for window_idx in np.unique(windows):
        positions = np.flatnonzero(windows == window_idx)
        picks.append(rng.choice(positions, size=positions.size, replace=True))
    return np.concatenate(picks)


def _aggregate(loss: np.ndarray, windows: np.ndarray, index: np.ndarray) -> float:
    values, keys = loss[index], windows[index]
    per_window = []
    for window_idx in np.unique(keys):
        window_values = values[keys == window_idx]
        if window_values.size and not np.all(np.isnan(window_values)):
            per_window.append(np.nanmean(window_values))
    return float(np.mean(per_window)) if per_window else float("nan")


def headroom_interval(
    per_task: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]],
    simple_probe: str,
    n_replicates: int = N_REPLICATES,
) -> dict:
    """Bootstrap H_oracle, R_simple and H_residual over origins, tasks held fixed.

    `per_task[task][probe] = (origin_loss, window_index)`, with the probes aligned
    origin by origin so a single resample applies to all of them.
    """
    rng = np.random.default_rng(SEED)
    tasks = sorted(per_task)
    draws = {"H_oracle_pct": [], "R_simple": [], "H_residual_pct": []}
    for _ in range(n_replicates):
        baseline_values, oracle_values, simple_values = [], [], []
        for task in tasks:
            probes = per_task[task]
            windows = probes["BASELINE_strongest_foundation_model"][1]
            index = _resample_within_windows(windows, rng)
            baseline_values.append(
                _aggregate(*probes["BASELINE_strongest_foundation_model"], index)
            )
            oracle_values.append(_aggregate(*probes["P0_cross_model_origin_oracle"], index))
            simple_values.append(_aggregate(*probes[simple_probe], index))
        baseline = float(np.mean(baseline_values))
        oracle = float(np.mean(oracle_values))
        simple = float(np.mean(simple_values))
        if baseline <= 0 or not np.isfinite(baseline):
            continue
        denominator = baseline - oracle
        draws["H_oracle_pct"].append(100.0 * (baseline - oracle) / baseline)
        draws["R_simple"].append(
            (baseline - simple) / denominator if denominator > 0 else np.nan
        )
        draws["H_residual_pct"].append(100.0 * (simple - oracle) / baseline)

    summary = {
        "n_replicates": len(draws["H_oracle_pct"]),
        "n_tasks": len(tasks),
        "resampling": "origins within evaluation windows, paired across probes",
        "interpretation": (
            "conditional on this set of tasks; it does not express uncertainty about which "
            "tasks the benchmark contains"
        ),
    }
    for name, values in draws.items():
        array = np.asarray(values, dtype=float)
        array = array[np.isfinite(array)]
        if array.size:
            summary[name] = {
                "median": float(np.median(array)),
                "ci_lower": float(np.percentile(array, 2.5)),
                "ci_upper": float(np.percentile(array, 97.5)),
            }
    return summary
