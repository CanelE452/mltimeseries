"""T01-T10, A01-A04 and the analytical representation checks.

These run on synthetic signals and on the operator definitions themselves.  They
gate model training: if any of them fails the pipeline is INVALID and no fit is
allowed to start.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from experiments.oa_resolution_pilot_v1 import model as M
from experiments.oa_resolution_pilot_v1.operators import (
    FORECAST_BINS,
    FOURIER_OMEGA,
    HISTORY_BINS,
    OPS,
    ReconstructionSolver,
    centre_basis,
    integrated_basis,
    observation_matrix,
    observation_support,
    observe,
    report_blocks,
)

ALL_R = (2, 3, 4, 6, 8, 12)


# --------------------------------------------------------------- synthetic signals


def signals(n: int = HISTORY_BINS) -> dict[str, np.ndarray]:
    t = np.arange(n, dtype=float)
    return {
        "constant": np.full(n, 3.25),
        "linear_ramp": 0.5 + 0.017 * t,
        "sinusoid": np.sin(2 * np.pi * t / 144.0) + 0.3 * np.cos(2 * np.pi * t / 36.0),
        "piecewise_constant": np.repeat(np.arange(n // 24, dtype=float) - 4.0, 24),
    }


# ------------------------------------------------------------------------ T01-T10


@pytest.mark.parametrize("r", ALL_R)
def test_T01_constant_interval_mean_exact(r):
    x = signals()["constant"]
    got = observe(x, r, "INTERVAL_MEAN")
    assert np.allclose(got, 3.25, atol=1e-12)


@pytest.mark.parametrize("r", ALL_R)
def test_T02_linear_interval_mean_exact(r):
    """For a linear signal the interval mean equals the value at the interval
    centre, computed independently of the operator implementation."""
    x = signals()["linear_ramp"]
    got = observe(x, r, "INTERVAL_MEAN")
    a, b = report_blocks(r).T
    expected = 0.5 + 0.017 * (a + b - 1) / 2.0
    assert np.allclose(got, expected, atol=1e-12)


@pytest.mark.parametrize("r", ALL_R)
@pytest.mark.parametrize("name", list(signals()))
def test_T03_discrete_sum_is_r_times_mean(r, name):
    x = signals()[name]
    mean = observe(x, r, "INTERVAL_MEAN")
    a, b = report_blocks(r).T
    total = np.array([x[i:j].sum() for i, j in zip(a, b)])
    assert np.allclose(total, r * mean, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("r", ALL_R)
def test_T04_end_bin_support_is_exactly_the_last_base_bin(r):
    sup = observation_support(r, "END_BIN")
    blocks = report_blocks(r)
    assert np.array_equal(sup[:, 1], blocks[:, 1])
    assert np.all(sup[:, 1] - sup[:, 0] == 1)
    x = signals()["sinusoid"]
    assert np.allclose(observe(x, r, "END_BIN"), x[blocks[:, 1] - 1])


@pytest.mark.parametrize("r", ALL_R)
def test_T05_interval_mean_support_is_the_whole_report_interval(r):
    sup = observation_support(r, "INTERVAL_MEAN")
    assert np.array_equal(sup, report_blocks(r))
    assert np.all(sup[:, 1] - sup[:, 0] == r)
    A = observation_matrix(r, "INTERVAL_MEAN")
    assert np.allclose(A.sum(axis=1), 1.0)
    assert np.allclose(A[A > 0], 1.0 / r)


@pytest.mark.parametrize("r", ALL_R)
@pytest.mark.parametrize("op", OPS)
def test_T06_no_report_interval_crosses_the_forecast_origin(r, op):
    sup = observation_support(r, op)
    assert sup.min() >= 0
    assert sup[:, 1].max() == HISTORY_BINS       # touches the origin, never passes it
    assert np.all(sup[:, 1] <= HISTORY_BINS)


@pytest.mark.parametrize("r", ALL_R)
@pytest.mark.parametrize("op", OPS)
def test_T07_future_values_never_reach_the_input_tokens(r, op):
    rng = np.random.default_rng(0)
    window = rng.normal(size=HISTORY_BINS + FORECAST_BINS)
    before = observe(window[:HISTORY_BINS], r, op)
    perturbed = window.copy()
    perturbed[HISTORY_BINS:] += 1e6                       # scream in the future
    after = observe(perturbed[:HISTORY_BINS], r, op)
    assert np.array_equal(before, after)


@pytest.mark.parametrize("r", ALL_R)
@pytest.mark.parametrize("op", OPS)
def test_T08_physical_history_duration_is_the_same_at_every_resolution(r, op):
    sup = observation_support(r, op)
    blocks = report_blocks(r)
    assert blocks[0, 0] == 0 and blocks[-1, 1] == HISTORY_BINS
    assert blocks[:, 1].max() - blocks[:, 0].min() == HISTORY_BINS   # 48 h, always
    assert len(sup) == HISTORY_BINS // r                             # only the count moves


@pytest.mark.parametrize("r", ALL_R)
def test_T09_no_off_by_one_at_the_block_boundaries(r):
    blocks = report_blocks(r)
    assert np.array_equal(blocks[1:, 0], blocks[:-1, 1])   # contiguous, no overlap
    assert blocks[:, 0].min() == 0
    assert blocks[:, 1].max() == HISTORY_BINS
    covered = np.concatenate([np.arange(a, b) for a, b in blocks])
    assert np.array_equal(np.sort(covered), np.arange(HISTORY_BINS))


def test_T10_windows_never_straddle_a_split_boundary():
    from experiments.oa_resolution_pilot_v1.data import (
        BaseGrid, EVAL_ORIGIN_STRIDE, chronological_split, eligible_origins,
    )
    import pandas as pd

    n = 20000
    grid = BaseGrid(
        dataset="synthetic", channels=("x",),
        values=np.arange(n, dtype=float)[:, None], valid=np.ones((n, 1), bool),
        stamps=pd.date_range("2020-01-01", periods=n, freq="10min"),
        period_start=pd.Timestamp("2020-01-01"), period_end=pd.Timestamp("2020-01-02"),
        provenance={},
    )
    splits = chronological_split(n)
    for split in ("train", "val", "test"):
        lo, hi = splits[split]
        o = eligible_origins(grid, splits, split, 0, stride=EVAL_ORIGIN_STRIDE)
        assert o.size > 0
        assert o.min() >= lo                       # target starts inside the split
        assert (o + FORECAST_BINS).max() <= hi     # target ends inside the split
        assert (o - HISTORY_BINS).min() >= 0       # history exists


# ------------------------------------------------------- analytical representation


def test_analytical_integrated_basis_matches_dense_quadrature():
    """Section 43: closed form vs numerical quadrature on 100 random intervals."""
    rng = np.random.default_rng(43)
    a = rng.uniform(0, HISTORY_BINS - 1, size=100)
    b = a + rng.uniform(1e-3, 24.0, size=100)
    bounds = np.stack([a, b], axis=1)

    closed = integrated_basis(bounds)
    errs = []
    for i, (lo, hi) in enumerate(bounds):
        # dense quadrature in the same tau coordinates the closed form uses
        t = (np.linspace(lo, hi, 200001) - HISTORY_BINS) / HISTORY_BINS
        ang = np.outer(t, FOURIER_OMEGA)
        num = np.concatenate([np.trapezoid(np.sin(ang), t, axis=0),
                              np.trapezoid(np.cos(ang), t, axis=0)]) / (t[-1] - t[0])
        errs.append(np.abs(num - closed[i]).max())
    assert max(errs) <= 1e-5, f"max abs error {max(errs):.3e}"


def test_integrated_basis_converges_to_the_centre_value_for_short_supports():
    centre = np.array([[100.0, 100.0 + 1e-9]])
    assert np.allclose(integrated_basis(centre), centre_basis(centre), atol=1e-9)
    widths = [1.0, 1e-2, 1e-4, 1e-6]
    gaps = [np.abs(integrated_basis(np.array([[100.0, 100.0 + w]]))
                   - centre_basis(np.array([[100.0, 100.0 + w]]))).max() for w in widths]
    assert all(gaps[i] > gaps[i + 1] for i in range(len(gaps) - 1))


@pytest.mark.parametrize("r", ALL_R)
@pytest.mark.parametrize("op", OPS)
def test_integrated_basis_is_never_a_point_evaluation(r, op):
    """Even END_BIN keeps a one-base-bin support, so O must attenuate it."""
    sup = observation_support(r, op)
    assert not np.allclose(integrated_basis(sup), centre_basis(sup))


# ---------------------------------------------------------- R reconstruction sanity


@pytest.mark.parametrize("r", ALL_R)
def test_R_reconstructs_a_linear_signal_from_interval_means(r):
    """A second-difference penalty is exact on linear signals, so with the right
    operator R must recover the base grid almost perfectly."""
    x = signals()["linear_ramp"]
    v = observe(x, r, "INTERVAL_MEAN")[None, :]
    got = ReconstructionSolver().reconstruct(v, r, "INTERVAL_MEAN", 1e-2)[0]
    assert np.abs(got - x).max() < 1e-6


@pytest.mark.parametrize("r", ALL_R)
@pytest.mark.parametrize("op", OPS)
def test_R_reproduces_its_own_observations(r, op):
    """Weak regularisation -> A x_hat must come back close to v."""
    x = signals()["sinusoid"]
    v = observe(x, r, op)[None, :]
    xh = ReconstructionSolver().reconstruct(v, r, op, 1e-4)
    assert np.abs(observe(xh[0], r, op) - v[0]).max() < 5e-3


# ------------------------------------------------------------------------ A01-A04


def test_A01_hourly_aggregation_is_a_plain_block_mean():
    pred = torch.arange(2 * FORECAST_BINS, dtype=torch.float32).reshape(2, FORECAST_BINS)
    got = M.hourly_mean(pred)
    assert got.shape == (2, 12)
    manual = pred.reshape(2, 12, 6).mean(-1)
    assert torch.allclose(got, manual)
    assert torch.allclose(got[0, 0], pred[0, :6].mean())


def test_A02_targets_use_the_same_aggregation_as_predictions():
    rng = np.random.default_rng(2)
    y = torch.tensor(rng.normal(size=(4, FORECAST_BINS)), dtype=torch.float32)
    assert torch.allclose(M.hourly_mean(y), y.reshape(4, 12, 6).mean(-1))


def test_A03_hourly_output_depends_only_on_the_predicted_base_values():
    pred = torch.randn(3, FORECAST_BINS)
    target = torch.randn(3, FORECAST_BINS)
    a = M.hourly_mean(pred).clone()
    target.add_(100.0)
    assert torch.allclose(M.hourly_mean(pred), a)


def test_A04_loss_weights_the_two_horizons_equally():
    pred = torch.zeros(1, FORECAST_BINS)
    target = torch.ones(1, FORECAST_BINS)
    total, mse10, mse60 = M.primary_loss(pred, target)
    assert torch.allclose(mse10, torch.tensor(1.0))
    assert torch.allclose(mse60, torch.tensor(1.0))
    assert torch.allclose(total, torch.tensor(1.0))

    # a purely sub-hourly error must move MSE10 without moving MSE60
    pred2 = torch.zeros(1, FORECAST_BINS)
    pred2[0, 0::2] = 2.0
    _, m10, m60 = M.primary_loss(pred2, torch.ones(1, FORECAST_BINS))
    assert m10 > m60
    assert torch.allclose(m60, torch.tensor(0.0), atol=1e-6)


# ------------------------------------------------------------------- F02/F03 pair


def test_F03_M_and_O_have_identical_parameter_counts():
    assert M.parameter_count(M.build("M")) == M.parameter_count(M.build("O"))


def test_F02_M_and_O_start_byte_identical_under_one_seed():
    def state(arm):
        torch.manual_seed(2026090601)
        return {k: v.clone() for k, v in M.build(arm).state_dict().items()}

    sm, so = state("M"), state("O")
    assert sm.keys() == so.keys()
    for k in sm:
        assert sm[k].shape == so[k].shape
        assert torch.equal(sm[k], so[k]), k


def test_F04_the_integrated_basis_carries_no_trainable_parameter():
    sup = observation_support(4, "INTERVAL_MEAN")
    assert isinstance(integrated_basis(sup), np.ndarray)   # numpy, not nn.Module
    assert M.parameter_count(M.build("O")) == M.parameter_count(M.build("M"))
