"""Exact finite-window Gaussian conditioning; no learned model or online claim."""

import numpy as np


HISTORY, HORIZON, ORIGIN = 24, 4, 23
A, INNOVATION_VARIANCE = 0.8, 0.36
BASE_SENSOR_VARIANCE, CANONICAL_REPORT_VARIANCE = 0.04, 0.0025
SEED, TOLERANCE = 2026090816, 1e-10


def schedule():
    specifications = [
        ("END_BASE", 0, 0, 0), ("MEAN", 0, 2, 2), ("SUM", 1, 4, 5),
        ("END_BASE", 4, 4, 4), ("MEAN", 3, 6, 6), ("SUM", 6, 7, 9),
        ("END_BASE", 10, 10, 10), ("MEAN", 8, 12, 12), ("SUM", 11, 15, 16),
        ("END_BASE", 17, 17, 18), ("MEAN", 16, 20, 20), ("SUM", 19, 23, 23),
        ("MEAN", 18, 22, 25),
    ]
    return [{"index": i, "operator": op, "start": start, "end": end, "arrival": arrival,
             "support_count": end - start + 1,
             "report_variance": CANONICAL_REPORT_VARIANCE * ((end - start + 1) if op == "SUM" else 1)**2,
             "available": arrival <= ORIGIN}
            for i, (op, start, end, arrival) in enumerate(specifications)]


def prior_covariance(size):
    indices = np.arange(size)
    return A ** np.abs(indices[:, None] - indices[None, :])


def observation_system(values=None):
    rows = schedule()
    all_h = np.zeros((len(rows), HISTORY), dtype=np.float64)
    for row in rows:
        if not 0 <= row["start"] <= row["end"] < HISTORY or row["arrival"] < row["end"]:
            raise ValueError("An observation has invalid support or arrival")
        weight = 1 / row["support_count"] if row["operator"] == "MEAN" else 1.0
        all_h[row["index"], row["start"]:row["end"] + 1] = weight
    all_variance = np.array([row["report_variance"] for row in rows], dtype=np.float64)
    if values is None:
        covariance = all_h @ (prior_covariance(HISTORY) + BASE_SENSOR_VARIANCE * np.eye(HISTORY)) @ all_h.T
        covariance += np.diag(all_variance)
        values = np.linalg.cholesky(covariance) @ np.random.default_rng(SEED).standard_normal(len(rows))
    values = np.array(values, dtype=np.float64, copy=True)
    if values.shape != (len(rows),) or not np.isfinite(values).all():
        raise ValueError("One finite value is required per scheduled observation")
    available = np.array(sorted((r["index"] for r in rows if r["available"]),
                                key=lambda i: (rows[i]["arrival"], i)), dtype=np.int64)
    excluded = np.array([r["index"] for r in rows if not r["available"]], dtype=np.int64)
    return {"rows": rows, "H": all_h[available], "report_variance": all_variance[available],
            "y": values[available], "available_indices": available, "excluded_indices": excluded,
            "all_H": all_h, "all_report_variance": all_variance, "all_y": values}


def _inputs(h, report_variance, y):
    h, report_variance, y = (np.array(v, dtype=np.float64, copy=True) for v in (h, report_variance, y))
    if h.ndim != 2 or h.shape[1] != HISTORY or y.shape != (len(h),):
        raise ValueError("H must have 24 columns and one row per observation value")
    if report_variance.shape != (len(h),) or np.any(report_variance <= 0) or not np.isfinite(report_variance).all():
        raise ValueError("Report variance must be finite and positive for each observation")
    if not np.isfinite(h).all() or not np.isfinite(y).all():
        raise ValueError("Observation values and operators must be finite")
    return h, report_variance, y


def direct_condition(h, report_variance, y, *, diagonal_noise=False):
    """Condition the full Gaussian prior, retaining shared-noise cross covariance."""
    h, report_variance, y = _inputs(h, report_variance, y)
    history_prior = prior_covariance(HISTORY)
    noise = BASE_SENSOR_VARIANCE * (h @ h.T) + np.diag(report_variance)
    if diagonal_noise:
        noise = np.diag(np.diag(noise))
    observation_covariance = h @ history_prior @ h.T + noise
    future = np.arange(HISTORY, HISTORY + HORIZON)
    future_history = A ** (future[:, None] - np.arange(HISTORY)[None, :])
    cross = np.vstack((history_prior @ h.T, future_history @ h.T))
    all_gain = np.linalg.solve(observation_covariance, cross.T).T
    history_gain, gain = all_gain[:HISTORY], all_gain[HISTORY:]
    return {"gain": gain, "mean": gain @ y,
            "covariance": prior_covariance(HORIZON) - gain @ h @ future_history.T,
            "history_gain": history_gain, "history_mean": history_gain @ y,
            "history_covariance": history_prior - history_gain @ h @ history_prior,
            "noise_covariance": noise}


def sequential_condition(h, report_variance, y):
    """Scalar Kalman updates of the static [x0..23, epsilon0..23] joint state."""
    h, report_variance, y = _inputs(h, report_variance, y)
    dimension, count = 2 * HISTORY, len(h)
    covariance = np.zeros((dimension, dimension), dtype=np.float64)
    covariance[:HISTORY, :HISTORY] = prior_covariance(HISTORY)
    covariance[HISTORY:, HISTORY:] = BASE_SENSOR_VARIANCE * np.eye(HISTORY)
    state_gain = np.zeros((dimension, count), dtype=np.float64)
    identity = np.eye(dimension)
    for i, row in enumerate(np.concatenate((h, h), axis=1)):
        kalman_gain = covariance @ row / (row @ covariance @ row + report_variance[i])
        innovation_gain = -row @ state_gain
        innovation_gain[i] += 1.0
        state_gain += np.outer(kalman_gain, innovation_gain)
        residual = identity - np.outer(kalman_gain, row)
        covariance = residual @ covariance @ residual.T + report_variance[i] * np.outer(kalman_gain, kalman_gain)
    leads = np.arange(1, HORIZON + 1)
    powers = A ** leads
    gain = np.outer(powers, state_gain[HISTORY - 1])
    future_noise = np.array([[INNOVATION_VARIANCE * sum(A ** (i + j - 2 * k) for k in range(1, min(i, j) + 1))
                              for j in leads] for i in leads], dtype=np.float64)
    return {"gain": gain, "mean": gain @ y,
            "covariance": np.outer(powers, powers) * covariance[HISTORY - 1, HISTORY - 1] + future_noise,
            "history_gain": state_gain[:HISTORY], "history_mean": state_gain[:HISTORY] @ y,
            "history_covariance": covariance[:HISTORY, :HISTORY]}


def operator_counterexample():
    """Same value/arrival, distinct operators; analytic metadata heads suffice."""
    prior = np.array([[1.0, A], [A, 1.0]])
    cross = np.array([A**2, A])
    independent_noise = 0.04
    result = {"origin": 1, "target_time": 2, "value": 1.0, "arrival": 1,
              "independent_noise_variance": independent_noise, "same_operator_blind_input": True}
    head_errors = []
    for op, row in (("END_BASE", np.array([1.0, 0.0])), ("MEAN", np.array([0.5, 0.5]))):
        variance_y = row @ prior @ row + independent_noise
        slope = float(cross @ row / variance_y)
        variance = float(1 - (cross @ row)**2 / variance_y)
        average_variance = (1 + A) / 2
        head_slope = A**2 / (1 + independent_noise) if op == "END_BASE" else A * average_variance / (average_variance + independent_noise)
        head_variance = 1 - A**4 / (1 + independent_noise) if op == "END_BASE" else 1 - (A * average_variance)**2 / (average_variance + independent_noise)
        head_errors.extend(abs((slope - head_slope) * value) for value in (-2.0, 0.0, 1.0, 3.0))
        head_errors.append(abs(variance - head_variance))
        result[op] = {"H": row.tolist(), "mean": slope, "variance": variance,
                      "metadata_head": {"slope": float(head_slope), "intercept": 0.0, "variance": float(head_variance)}}
    result.update(mean_difference=abs(result["END_BASE"]["mean"] - result["MEAN"]["mean"]),
                  variance_difference=abs(result["END_BASE"]["variance"] - result["MEAN"]["variance"]),
                  metadata_head_max_abs_error=float(max(head_errors)))
    return result


def identifiability(h, functional):
    h, functional = np.atleast_2d(np.asarray(h, dtype=np.float64)), np.atleast_2d(np.asarray(functional, dtype=np.float64))
    _, singular_values, right = np.linalg.svd(h, full_matrices=True)
    rank = int(np.count_nonzero(singular_values > 1e-12))
    nullspace = right[rank:].T
    residual = float(np.max(np.abs(functional @ nullspace), initial=0.0))
    return {"H": h.tolist(), "L": functional.tolist(), "rank": rank, "nullity": int(h.shape[1] - rank),
            "singular_values": singular_values.tolist(), "nullspace": nullspace.tolist(),
            "functional_nullspace_max_abs": residual, "identifiable": residual <= TOLERANCE,
            "scope": "Noiseless deterministic functional identifiability, not future process-noise recovery"}


def _errors(first, second):
    return {key: float(np.max(np.abs(first[key] - second[key]), initial=0.0))
            for key in ("gain", "mean", "covariance")}


def _psd(covariance):
    symmetry = float(np.max(np.abs(covariance - covariance.T)))
    minimum = float(np.linalg.eigvalsh((covariance + covariance.T) / 2).min())
    return {"symmetry_max_abs": symmetry, "minimum_eigenvalue": minimum,
            "passed": symmetry <= TOLERANCE and minimum >= -TOLERANCE}


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def diagnostics():
    """Return the fixed audit as JSON data; this function performs no file I/O."""
    system = observation_system()
    h, variance, y = (system[key] for key in ("H", "report_variance", "y"))
    direct, sequential = direct_condition(h, variance, y), sequential_condition(h, variance, y)
    equality = _errors(direct, sequential)
    history_errors = {key: float(np.max(np.abs(direct[key] - sequential[key])))
                      for key in ("history_gain", "history_mean", "history_covariance")}
    psd = {f"{name}_{key}": _psd(result[key]) for name, result in (("direct", direct), ("sequential", sequential))
           for key in ("covariance", "history_covariance")}

    rows = [system["rows"][i] for i in system["available_indices"]]
    conversion = np.array([r["support_count"] if r["operator"] == "MEAN" else
                           1 / r["support_count"] if r["operator"] == "SUM" else 1.0 for r in rows])
    converted = direct_condition(h * conversion[:, None], variance * conversion**2, y * conversion)
    converted_original_units = {"gain": converted["gain"] * conversion, "mean": converted["mean"], "covariance": converted["covariance"]}
    conversion_errors = _errors(direct, converted_original_units)
    conversion_noise_error = float(np.max(np.abs(converted["noise_covariance"] - conversion[:, None] * direct["noise_covariance"] * conversion)))

    changed_values = system["all_y"].copy()
    changed_values[system["excluded_indices"]] = 1e12
    late_system = observation_system(changed_values)
    late = direct_condition(late_system["H"], late_system["report_variance"], late_system["y"])
    late_errors = _errors(direct, late)
    wrong_noise = direct_condition(h, variance, y, diagonal_noise=True)
    wrong_noise_errors = _errors(direct, wrong_noise)
    off_diagonal_noise = direct["noise_covariance"] - np.diag(np.diag(direct["noise_covariance"]))

    counterexample = operator_counterexample()
    average, point = np.array([[0.5, 0.5]]), np.array([[1.0, 0.0]])
    kernel = {"average_from_average": identifiability(average, average),
              "point_from_average": identifiability(average, point),
              "point_from_both_points": identifiability(np.eye(2), point),
              "witness": {"delta": [1.0, -1.0], "H_delta": (average @ np.array([1.0, -1.0])).tolist(),
                          "L_delta": (point @ np.array([1.0, -1.0])).tolist()}}
    functionals = np.vstack((np.eye(HORIZON), np.ones((1, HORIZON)) / HORIZON, np.ones((1, HORIZON))))
    functional_results = {name: {"gain": functionals @ value["gain"], "mean": functionals @ value["mean"],
                                 "covariance": functionals @ value["covariance"] @ functionals.T}
                          for name, value in (("direct", direct), ("sequential", sequential))}
    functional_errors = _errors(functional_results["direct"], functional_results["sequential"])
    psd["direct_functionals"] = _psd(functional_results["direct"]["covariance"])
    psd["sequential_functionals"] = _psd(functional_results["sequential"]["covariance"])
    checks = {
        "schedule_valid": len(system["available_indices"]) == 12 and system["excluded_indices"].tolist() == [12],
        "stationary_variance_one": abs(INNOVATION_VARIANCE / (1 - A**2) - 1) <= TOLERANCE,
        "exact_conditioning": max(equality.values()) <= TOLERANCE and max(history_errors.values()) <= TOLERANCE,
        "posterior_psd": all(value["passed"] for value in psd.values()),
        "future_functionals_equal": max(functional_errors.values()) <= TOLERANCE,
        "mean_sum_equivalence": max(conversion_errors.values()) <= TOLERANCE and conversion_noise_error <= TOLERANCE,
        "late_observation_excluded": max(late_errors.values()) == 0.0 and np.array_equal(y, late_system["y"]),
        "shared_noise_matters": np.max(np.abs(off_diagonal_noise)) > 1e-8 and max(wrong_noise_errors.values()) > 1e-8,
        "operator_blind_collision": counterexample["same_operator_blind_input"] and counterexample["mean_difference"] > TOLERANCE and counterexample["variance_difference"] > TOLERANCE,
        "metadata_head_exact": counterexample["metadata_head_max_abs_error"] <= TOLERANCE,
        "nullspace_condition": kernel["average_from_average"]["identifiable"] and not kernel["point_from_average"]["identifiable"] and kernel["point_from_both_points"]["identifiable"],
    }
    passed = all(checks.values())
    return _jsonable({
        "passed": passed, "verdict": "EXISTING_LINEAR_CONDITIONING_SUFFICIENT" if passed else "NUMERICAL_AUDIT_FAILED",
        "settings": {"history": HISTORY, "horizon": HORIZON, "origin": ORIGIN, "a": A,
                     "innovation_variance": INNOVATION_VARIANCE, "base_sensor_variance": BASE_SENSOR_VARIANCE,
                     "canonical_report_variance": CANONICAL_REPORT_VARIANCE, "seed": SEED,
                     "max_error": TOLERANCE, "device": "cpu", "new_training_updates": 0},
        "scope": "Known linear Gaussian calculation only; no new method, forecasting performance ranking, or proof of PEFT necessity",
        "sequential_state": "Static 48D joint [x0..23, shared_sensor_noise0..23]; independent report noise in scalar Joseph updates",
        "checks": checks, "observations": system, "direct": direct, "sequential": sequential,
        "equality_max_abs": equality, "history_equality_max_abs": history_errors, "posterior_psd": psd,
        "functionals": {"rows": functionals, "labels": ["x24", "x25", "x26", "x27", "future_mean", "future_sum"],
                        **functional_results, "equality_max_abs": functional_errors},
        "mean_sum_conversion": {"row_multiplier": conversion, "converted": converted,
                                "gain_in_original_units": converted_original_units["gain"],
                                "equality_max_abs": conversion_errors, "noise_covariance_max_abs": conversion_noise_error},
        "late_exclusion": {"replacement": 1e12, "excluded_indices": system["excluded_indices"], "equality_max_abs": late_errors},
        "diagonal_noise_ablation": {"result": wrong_noise, "difference_max_abs": wrong_noise_errors,
                                    "omitted_noise_max_abs": float(np.max(np.abs(off_diagonal_noise))),
                                    "interpretation": "Misspecified sensor-noise covariance, not evidence for PEFT"},
        "operator_counterexample": counterexample, "identifiability": kernel})
