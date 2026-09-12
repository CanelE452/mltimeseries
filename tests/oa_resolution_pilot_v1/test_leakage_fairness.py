"""L01-L08 and F01-F06.

These assert the properties on constructed grids rather than reading them off the
source code, so a later edit that quietly breaks one of them fails here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from experiments.oa_resolution_pilot_v1 import data as D
from experiments.oa_resolution_pilot_v1 import model as MODEL
from experiments.oa_resolution_pilot_v1 import report as RPT
from experiments.oa_resolution_pilot_v1 import train as T
from experiments.oa_resolution_pilot_v1.data import (
    BaseGrid, chronological_split, eligible_origins, train_scaling,
)
from experiments.oa_resolution_pilot_v1.evaluate import EVAL_R, OPS, frozen_keys

N = 30000


def make_grid(values: np.ndarray | None = None, valid: np.ndarray | None = None) -> BaseGrid:
    rng = np.random.default_rng(7)
    v = rng.normal(size=(N, 2)) if values is None else values
    ok = np.ones((N, 2), bool) if valid is None else valid
    return BaseGrid(
        dataset="synthetic", channels=("a", "b"), values=v, valid=ok,
        stamps=pd.date_range("2020-01-01", periods=N, freq="10min"),
        period_start=pd.Timestamp("2020-01-01"), period_end=pd.Timestamp("2020-08-01"),
        provenance={},
    )


# ------------------------------------------------------------------------ L01-L08


def test_L01_scaler_uses_train_bins_only():
    g = make_grid()
    sp = chronological_split(N)
    mu, sd = train_scaling(g, sp)

    poisoned = g.values.copy()
    poisoned[sp["val"][0]:] += 1000.0            # wreck everything after train
    mu2, sd2 = train_scaling(make_grid(poisoned), sp)
    assert np.allclose(mu, mu2) and np.allclose(sd, sd2)

    moved = g.values.copy()
    moved[: sp["train"][1]] += 5.0
    mu3, _ = train_scaling(make_grid(moved), sp)
    assert not np.allclose(mu, mu3)              # and it does depend on the train bins


def test_L02_lambda_screening_never_reads_the_test_split():
    """The screen scores on validation windows; those windows are inside val."""
    g = make_grid()
    sp = chronological_split(N)
    ch, org = T.validation_windows(g, sp)
    lo, hi = sp["val"]
    assert org.min() >= lo
    assert (org + D.FORECAST_BINS).max() <= hi


def test_L03_target_timestamps_never_enter_the_observations():
    """Covered analytically by T07; repeated here on the batch path used in training."""
    g = make_grid()
    sp = chronological_split(N)
    mu, sd = train_scaling(g, sp)
    dev = torch.device("cpu")
    store = T.WindowStore(g, mu, sd, dev)
    c = np.zeros(8, dtype=int)
    o = np.arange(1000, 1008)
    hist, _ = store.batch(c, o)

    poisoned = g.values.copy()
    poisoned[1000:] += 1e6                       # every bin at or after the origin
    store2 = T.WindowStore(make_grid(poisoned), mu, sd, dev)
    hist2, _ = store2.batch(c, np.full(8, 1000))
    assert torch.allclose(hist[0], hist2[0])


@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_L04_no_aggregation_window_crosses_the_forecast_origin(split):
    g = make_grid()
    sp = chronological_split(N)
    o = eligible_origins(g, sp, split, 0, stride=D.EVAL_ORIGIN_STRIDE)
    assert o.size > 0
    for r in EVAL_R:
        for op in OPS:
            from experiments.oa_resolution_pilot_v1.operators import observation_support
            assert observation_support(r, op)[:, 1].max() <= D.HISTORY_BINS


def test_L05_training_origins_stay_inside_the_train_split():
    g = make_grid()
    sp = chronological_split(N)
    sch = T.build_schedule(g, sp, 2026090601, 50)
    lo, hi = sp["train"]
    assert sch.origins.min() >= lo
    assert (sch.origins + D.FORECAST_BINS).max() <= hi
    assert (sch.origins - D.HISTORY_BINS).min() >= 0


def test_L06_L07_the_key_set_cannot_depend_on_arm_operation_or_resolution():
    """eligible_origins takes neither an arm, an operation nor an r, and frozen_keys
    is built from the grid alone -- so all three arms score identical keys."""
    import inspect

    params = set(inspect.signature(eligible_origins).parameters)
    assert not params & {"arm", "op", "operation", "r"}
    assert not set(inspect.signature(frozen_keys).parameters) & {"arm", "op", "r"}

    g = make_grid()
    sp = chronological_split(N)
    k1, k2 = frozen_keys(g, sp), frozen_keys(g, sp)
    for split in ("val", "test"):
        assert np.array_equal(k1[split][0], k2[split][0])
        assert np.array_equal(k1[split][1], k2[split][1])


def test_L07_missing_bins_exclude_the_same_windows_for_every_channel_query():
    rng = np.random.default_rng(3)
    valid = np.ones((N, 2), bool)
    valid[rng.choice(N, 200, replace=False), 0] = False
    g = make_grid(valid=valid)
    sp = chronological_split(N)
    a = eligible_origins(g, sp, "test", 0, stride=72)
    b = eligible_origins(g, sp, "test", 1, stride=72)
    assert a.size < b.size                       # channel 0 really is the damaged one
    for o in a:                                  # and nothing eligible touches a hole
        assert valid[o - D.HISTORY_BINS : o + D.FORECAST_BINS, 0].all()


def test_L08_aggregation_joins_on_keys_not_row_order():
    rng = np.random.default_rng(11)
    rows = []
    for arm, base in (("R", 1.0), ("M", 0.9), ("O", 0.8)):
        for r in (3, 6):
            for op in OPS:
                for o in range(0, 2000, 72):
                    rows.append({"dataset": "jena", "split": "test", "origin": o, "channel": 0,
                                 "operation": op, "r": r, "role": "UNSEEN_INTERPOLATION",
                                 "arm": arm, "primary": base + rng.normal(0, 0.01)})
    df = pd.DataFrame(rows)
    a = RPT.macro(RPT.cell_table(df), (3, 6), "O", "M")["macro_relative_improvement_pct"]
    shuffled = df.sample(frac=1.0, random_state=5).reset_index(drop=True)
    b = RPT.macro(RPT.cell_table(shuffled), (3, 6), "O", "M")["macro_relative_improvement_pct"]
    assert a == pytest.approx(b)


# ------------------------------------------------------------------------ F01-F06


def test_F01_all_three_arms_share_one_schedule_per_dataset_and_seed():
    g = make_grid()
    sp = chronological_split(N)
    a = T.build_schedule(g, sp, 2026090601, 120)
    b = T.build_schedule(g, sp, 2026090601, 120)
    assert a.sha == b.sha                        # reproducible, so R/M/O get the same one
    c = T.build_schedule(g, sp, 2026090602, 120)
    assert a.sha != c.sha                        # and the seed actually moves it
    assert [a.condition(u) for u in range(12)] == [
        (2, "END_BIN"), (2, "INTERVAL_MEAN"), (4, "END_BIN"), (4, "INTERVAL_MEAN"),
        (8, "END_BIN"), (8, "INTERVAL_MEAN")] * 2


def test_F01_schedule_is_balanced_over_the_conditions():
    counts = {}
    for u in range(3000):
        counts[T.CONDITION_CYCLE[u % 6]] = counts.get(T.CONDITION_CYCLE[u % 6], 0) + 1
    assert len(set(counts.values())) == 1
    assert set(counts) == {(r, op) for r in T.TRAIN_R for op in T.TRAIN_OPS}


def test_F05_one_checkpoint_per_fit_not_one_per_condition():
    """The selection metric is a single scalar pooled over every training condition,
    so there is nothing that could be selected per (r, op)."""
    g = make_grid()
    sp = chronological_split(N)
    mu, sd = train_scaling(g, sp)
    dev = torch.device("cpu")
    net = MODEL.build("M")
    inputs = T.ArmInputs("M", dev)
    store = T.WindowStore(g, mu, sd, dev)
    ch, org = T.validation_windows(g, sp)
    sel = org[:64], ch[:64]
    v = T.validation_loss(net, inputs, store, sel[1], sel[0])
    assert set(v) == {"primary", "mse10", "mse60"}
    assert all(isinstance(x, float) for x in v.values())

    out = T.train_arm("M", g, sp, mu, sd, 2026090601, T.Tier("t", 6, 2, 3), dev,
                      log=lambda s: None)
    assert isinstance(out["best_update"], int)          # exactly one checkpoint
    assert out["best_val_primary"] == min(c["primary"] for c in out["curve"])


def test_F06_per_seed_rows_survive_aggregation():
    rows = []
    for seed in (2026090601, 2026090602):
        for arm in ("R", "M", "O"):
            rows.append({"dataset": "jena", "split": "test", "origin": 0, "channel": 0,
                         "operation": "END_BIN", "r": 3, "role": "UNSEEN_INTERPOLATION",
                         "arm": arm, "seed": seed, "primary": 1.0,
                         "mse10": 1.0, "mse60": 1.0, "mae10": 1.0, "mae60": 1.0})
    df = pd.DataFrame(rows)
    assert set(df["seed"].unique()) == {2026090601, 2026090602}
    sa = RPT.seed_averaged(df)
    assert len(sa) == 3                          # seeds collapse only where asked to
    assert len(df) == 6


def test_F_the_two_arms_differ_only_in_the_observation_time_basis():
    """M and O share every module shape; the divergence is the numpy basis fed in."""
    from experiments.oa_resolution_pilot_v1.operators import token_features

    torch.manual_seed(1)
    m = MODEL.build("M")
    torch.manual_seed(1)
    o = MODEL.build("O")
    assert [tuple(p.shape) for p in m.parameters()] == [tuple(p.shape) for p in o.parameters()]

    for r in EVAL_R:
        for op in OPS:
            fm, fo = token_features(r, op, "M"), token_features(r, op, "O")
            assert np.array_equal(fm["width"], fo["width"])
            assert np.array_equal(fm["op_id"], fo["op_id"])
            assert np.array_equal(fm["support"], fo["support"])
            assert not np.allclose(fm["time_basis"], fo["time_basis"])
