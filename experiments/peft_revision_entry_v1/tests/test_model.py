from datetime import date

import numpy as np
import pytest


def test_ridge_matches_augmented_least_squares_with_unpenalized_intercept():
    from experiments.peft_revision_entry_v1.model import fit_ridge

    rng = np.random.default_rng(19)
    x = np.column_stack([np.ones(115), rng.normal(size=(115, 6))])
    y = 4 + rng.normal(size=115)
    penalty = np.diag([0.0] + [np.sqrt(10.0)] * 6)
    expected = np.linalg.lstsq(
        np.vstack([x, penalty]), np.concatenate([y, np.zeros(7)]), rcond=None
    )[0]
    np.testing.assert_allclose(fit_ridge(x, y, 10.0), expected, atol=1e-12)
    np.testing.assert_allclose(fit_ridge(x, np.full(115, 8.0), 1e6), [8] + [0] * 6, atol=1e-12)


def test_calendar_window_keeps_exact_boundary_and_requires_historical_events():
    from experiments.peft_revision_entry_v1.model import ridge

    events = np.array([879, 880, 998, 999])
    x = np.column_stack([np.ones(4), np.arange(4)])
    y = np.array([99999, 2, 3, 4])
    result = ridge(x, y, np.array([1, 4]), 1.0, 120, 1000, events, min_train=3)
    assert result['n_train'] == 3
    with pytest.raises(ValueError, match='historical'):
        ridge(x, y, np.array([1, 4]), 1.0, None, 999, events, min_train=3)
    with pytest.raises(ValueError, match='100'):
        ridge(x, y, np.array([1, 4]), 1.0, None, 1000, events)


def test_nonfinite_rows_fail_instead_of_being_dropped():
    from experiments.peft_revision_entry_v1.model import fit_ridge

    with pytest.raises(ValueError, match='finite'):
        fit_ridge(np.array([[1, np.nan], [1, 2]]), np.ones(2), 1)
    with pytest.raises(ValueError, match='finite'):
        fit_ridge(np.array([[1, 1], [1, 2]]), np.array([1, np.nan]), 1)


def test_predict_arms_uses_each_arm_config_and_first_bias():
    from experiments.peft_revision_entry_v1.model import RidgeConfig, predict_arms, ridge

    events = np.arange(200, 320)
    x = np.column_stack([np.ones(120), np.linspace(-2, 2, 120)])
    y = 2 * x[:, 1]
    prediction_x = np.array([1, 4])
    data = {'FIRST_FIXED_X': (x, y, events), 'REVISED_FIXED_X': (x, y + 1, events)}
    configs = {name: RidgeConfig(lam, None) for name, lam in [
        ('FIRST_FIXED_X', 1), ('FIRST_BIAS', 100), ('REVISED_FIXED_X', 1)]}
    got = predict_arms(data, prediction_x, configs, 320, first_bias=0.25, last_known_growth=5)
    expected_bias = ridge(x, y, prediction_x, 100, None, 320, events)['prediction'] + 0.25
    assert got['FIRST_BIAS']['prediction'] == pytest.approx(expected_bias)
    assert got['REVISED_FIXED_X']['prediction'] - got['FIRST_FIXED_X']['prediction'] == pytest.approx(1)
    assert got['ZERO_GROWTH']['prediction'] == 0
    assert got['LAST_KNOWN_GROWTH']['prediction'] == 5


def test_config_ties_have_small_lambda_then_expanding():
    from experiments.peft_revision_entry_v1.model import candidate_configs, select_config

    configs = candidate_configs()
    assert len(configs) == 10
    best = select_config([(config, 2.0) for config in reversed(configs)])
    assert best.lam == 0.01
    assert best.window is None


def test_revision_updates_do_not_duplicate_rows_or_change_a():
    from experiments.peft_revision_entry_v1.model import RevisionRidge, fit_ridge

    rng = np.random.default_rng(1908)
    x = np.column_stack([np.ones(120), rng.normal(size=(120, 6))])
    y = rng.normal(size=120)
    state = RevisionRidge(n_features=7, lam=1)
    for row_id, (features, value) in enumerate(zip(x, y)):
        state.update(row_id, features, value)
    before = state.A.copy()
    for row_id in [0, 18, 42, 119]:
        y[row_id] += 0.5
        state.update(row_id, x[row_id], y[row_id])
        state.update(row_id, x[row_id], y[row_id])
    np.testing.assert_array_equal(state.A, before)
    assert state.n_rows == 120
    np.testing.assert_allclose(state.coefficients(), fit_ridge(x, y, 1), atol=1e-12)
    result = state.compare_batch(np.array([1, 1, 2, 3, 4, 5, 6]))
    assert result['passed']
    assert max(result[key] for key in ['a_max_abs', 'b_max_abs', 'coef_max_abs', 'prediction_abs']) < 1e-12


def test_revision_state_rejects_changed_fixed_features():
    from experiments.peft_revision_entry_v1.model import RevisionRidge

    state = RevisionRidge(n_features=2)
    state.update('row', np.array([1, 2]), 3)
    with pytest.raises(ValueError, match='fixed'):
        state.update('row', np.array([1, 3]), 3)


def test_revision_bias_uses_only_mature_same_age_examples():
    from experiments.peft_revision_entry_v1.model import revision_bias

    provisional = np.array([1, 2, 3, 4])
    mature = np.array([2, 4, 103, np.nan])
    maturity_dates = [date(2010, 1, 31), date(2010, 2, 28), date(2010, 3, 31), date(2011, 1, 31)]
    got = revision_bias(provisional, mature, maturity_dates, date(2010, 2, 28))
    assert got == (1.5, 2)
    age_got = revision_bias(provisional, mature, maturity_dates, date(2010, 2, 28), ages=[1, 2, 1, 1], age=1)
    assert age_got == (1.0, 1)
    assert revision_bias(provisional, mature, maturity_dates, date(2009, 12, 31)) == (0.0, 0)
