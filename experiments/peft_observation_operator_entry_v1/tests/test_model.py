import importlib
import importlib.util
import json

import numpy as np
import pytest


def model():
    name = "experiments.peft_observation_operator_entry_v1.model"
    assert importlib.util.find_spec(name) is not None, "The observation model is not implemented"
    return importlib.import_module(name)


def test_schedule_and_shared_noise_contract():
    m = model()
    system = m.observation_system()
    assert len(m.schedule()) == 13
    assert system["H"].shape == (12, 24)
    assert system["excluded_indices"].tolist() == [12]
    assert system["available_indices"].tolist() == [0, 1, 3, 2, 4, 5, 6, 7, 8, 9, 10, 11]
    assert all(row["end"] <= row["arrival"] for row in m.schedule())
    for row, h, variance in zip(m.schedule(), system["all_H"], system["all_report_variance"]):
        count = row["end"] - row["start"] + 1
        multiplier = count if row["operator"] == "SUM" else 1
        assert h.sum() == pytest.approx(multiplier)
        assert variance == pytest.approx(0.0025 * multiplier**2)
    assert system["H"].dtype == np.float64
    assert system["all_H"][0] @ system["all_H"][1] > 0


@pytest.mark.parametrize("method", ["direct_condition", "sequential_condition"])
def test_single_observation_matches_independent_scalar_formula(method):
    m = model()
    h = np.zeros((1, 24), dtype=np.float64)
    h[0, 23] = 1
    result = getattr(m, method)(h, np.array([0.0025]), np.array([2.0]))
    cross = 0.8 ** np.arange(1, 5)
    expected_gain = cross[:, None] / 1.0425
    expected_covariance = 0.8 ** np.abs(np.arange(4)[:, None] - np.arange(4))
    expected_covariance -= np.outer(cross, cross) / 1.0425
    np.testing.assert_allclose(result["gain"], expected_gain, rtol=0, atol=1e-12)
    np.testing.assert_allclose(result["mean"], expected_gain[:, 0] * 2, rtol=0, atol=1e-12)
    np.testing.assert_allclose(result["covariance"], expected_covariance, rtol=0, atol=1e-12)


def test_full_gain_matches_and_sequential_order_is_irrelevant():
    m = model()
    h = np.zeros((4, 24))
    h[0, 20:24] = 0.25
    h[1, 22:24] = 1
    h[2, 19] = 1
    h[3, 19:22] = 1 / 3
    variance = np.array([0.0025, 0.01, 0.0025, 0.0025])
    y = np.array([0.3, -0.1, 1.1, -0.2])
    direct = m.direct_condition(h, variance, y)
    sequential = m.sequential_condition(h, variance, y)
    order = np.array([3, 1, 0, 2])
    reversed_result = m.sequential_condition(h[order], variance[order], y[order])
    for key in ("gain", "mean", "covariance", "history_gain", "history_mean", "history_covariance"):
        np.testing.assert_allclose(sequential[key], direct[key], rtol=0, atol=1e-10)
        restored = reversed_result[key][:, np.argsort(order)] if key.endswith("gain") else reversed_result[key]
        np.testing.assert_allclose(restored, direct[key], rtol=0, atol=1e-10)


def test_mean_sum_transform_includes_gain_and_noise_units():
    m = model()
    h = np.zeros((2, 24))
    h[0, 19:23] = 0.25
    h[1, 21:24] = 1
    rv, y, multiplier = np.array([0.0025, 0.0225]), np.array([1.2, -0.5]), np.array([4.0, 1 / 3])
    a = m.direct_condition(h, rv, y)
    b = m.direct_condition(h * multiplier[:, None], rv * multiplier**2, y * multiplier)
    np.testing.assert_allclose(a["gain"], b["gain"] * multiplier, rtol=0, atol=1e-12)
    for key in ("mean", "covariance"):
        np.testing.assert_allclose(a[key], b[key], rtol=0, atol=1e-12)
    np.testing.assert_allclose(b["noise_covariance"], multiplier[:, None] * a["noise_covariance"] * multiplier, rtol=0, atol=1e-12)


def test_late_values_are_excluded_without_mutating_input():
    m = model()
    values = np.arange(13, dtype=np.float64)
    untouched = values.copy()
    first = m.observation_system(values)
    np.testing.assert_array_equal(values, untouched)
    values[12] = 1e12
    later = m.observation_system(values)
    for key in ("H", "report_variance", "y", "available_indices"):
        np.testing.assert_array_equal(first[key], later[key])
    before = [first[key].copy() for key in ("H", "report_variance", "y")]
    for method in (m.direct_condition, m.sequential_condition):
        method(first["H"], first["report_variance"], first["y"])
    for key, expected in zip(("H", "report_variance", "y"), before):
        np.testing.assert_array_equal(first[key], expected)


def test_shared_noise_cannot_be_replaced_by_independent_report_noise():
    m = model()
    h = np.zeros((2, 24))
    h[0, 22:24] = 0.5
    h[1, 23] = 1
    rv, y = np.full(2, 0.0025), np.array([1.0, 0.5])
    correct = m.direct_condition(h, rv, y)
    wrong = m.direct_condition(h, rv, y, diagonal_noise=True)
    assert np.max(np.abs(correct["covariance"] - wrong["covariance"])) > 1e-6


def test_counterexample_and_metadata_head_have_known_analytic_answer():
    m = model()
    result = m.operator_counterexample()
    assert result["END_BASE"]["mean"] == pytest.approx(0.64 / 1.04)
    assert result["END_BASE"]["variance"] == pytest.approx(1 - 0.4096 / 1.04)
    assert result["MEAN"]["mean"] == pytest.approx(0.72 / 0.94)
    assert result["MEAN"]["variance"] == pytest.approx(1 - 0.5184 / 0.94)
    assert result["same_operator_blind_input"]
    assert result["mean_difference"] > 0.1
    assert result["metadata_head_max_abs_error"] < 1e-14


def test_nullspace_distinguishes_functional_from_latent_identifiability():
    m = model()
    average = np.array([[0.5, 0.5]])
    point = np.array([[1.0, 0.0]])
    yes = m.identifiability(average, average)
    no = m.identifiability(average, point)
    complete = m.identifiability(np.eye(2), point)
    assert yes["identifiable"] and complete["identifiable"]
    assert not no["identifiable"]
    assert no["nullity"] == 1
    assert no["functional_nullspace_max_abs"] == pytest.approx(2**-0.5)


@pytest.mark.parametrize("method", ["direct_condition", "sequential_condition"])
def test_invalid_variance_is_rejected(method):
    m = model()
    with pytest.raises(ValueError, match="variance"):
        getattr(m, method)(np.zeros((1, 24)), np.array([-0.1]), np.array([0.0]))


def test_diagnostics_interface_is_callable_without_running_production_audit():
    m = model()
    assert callable(m.diagnostics)
    json.dumps(m.schedule(), allow_nan=False)
