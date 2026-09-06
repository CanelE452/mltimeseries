"""Operational invariants that must pass before any accuracy number is looked at.

Contract source: 01_forecast_query_tokenization_CLI.txt sections 8 and 20.

These check wiring, leakage and joins. None of them assert that a trained C actually
produces horizon-dependent weights -- section 8 explicitly forbids that, because it is
an experimental outcome and not an implementation invariant.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from experiments.hq_token_pilot_v1 import data as D
from experiments.hq_token_pilot_v1 import evaluate as E
from experiments.hq_token_pilot_v1 import model as M

DEV = "cpu"
SEED = 2026090601


@pytest.fixture(scope="module")
def toy():
    rng = np.random.default_rng(0)
    t = np.arange(6000)[:, None]
    base = np.sin(t / 37.0) + 0.3 * np.sin(t / 211.0)
    values = base * np.array([1.0, 0.7, 1.3]) + 0.1 * rng.standard_normal((6000, 3))
    return D.Dataset.from_array("toy", values)


def _batch(ds, n=8, h=96):
    o = ds.train_origins()[:n]
    c = np.arange(n) % ds.n_channels
    x = torch.from_numpy(ds.inputs(o, c))
    y = torch.from_numpy(ds.targets(o, c))
    return x, y, torch.full((n,), float(h))


# --- section 8 --------------------------------------------------------------


@pytest.mark.parametrize("arm", ["U", "I", "C", "R", "H_STATIC"])
def test_weights_are_a_partition_of_unity(toy, arm):
    net = M.build(arm, 32, SEED).eval()
    x, _, h = _batch(toy)
    _, w = net(x, h, h, return_weights=True)
    assert w.shape == (x.shape[0], 32, 2)
    assert torch.isfinite(w).all()
    assert (w >= 0).all()
    assert torch.allclose(w.sum(-1), torch.ones_like(w.sum(-1)), atol=1e-6)


@pytest.mark.parametrize("arm,B,expect", [("C", 32, 32), ("C", 16, 16), ("DENSE", 64, 64)])
def test_encoder_really_receives_B_tokens(toy, arm, B, expect):
    """A mask over 64 positions would not be compression. Hook the real input length."""
    net = M.build(arm, B, SEED).eval()
    seen = []
    net.encoder.register_forward_pre_hook(lambda m, inp: seen.append(inp[0].shape[1]))
    x, _, h = _batch(toy)
    net(x, h, h)
    assert seen == [expect]


def test_output_shape_and_loss_mask(toy):
    net = M.build("C", 32, SEED).eval()
    x, y, _ = _batch(toy)
    for h in (96, 336):
        hh = torch.full((x.shape[0],), float(h))
        out = net(x, hh, hh)
        assert out.shape == (x.shape[0], 336)
        manual = ((out[:, :h] - y[:, :h]) ** 2).mean()
        assert torch.allclose(M.masked_mse(out, y, hh), manual, atol=1e-6)


def test_ICR_share_structure_and_initial_state(toy):
    nets = {a: M.build(a, 32, SEED) for a in ("I", "C", "R")}
    keys = {a: sorted(n.state_dict()) for a, n in nets.items()}
    assert keys["I"] == keys["C"] == keys["R"]
    for k in keys["I"]:
        assert torch.equal(nets["I"].state_dict()[k], nets["C"].state_dict()[k])
        assert torch.equal(nets["I"].state_dict()[k], nets["R"].state_dict()[k])
    counts = {a: n.parameter_report()["total_parameters"] for a, n in nets.items()}
    assert len(set(counts.values())) == 1


def test_identity_at_B64_makes_I_and_C_agree(toy):
    """Group size 1 -> softmax over one element -> the score cannot matter."""
    i_net = M.build("I", 64, SEED).eval()
    c_net = M.build("C", 64, SEED).eval()
    c_net.load_state_dict(i_net.state_dict())
    x, _, _ = _batch(toy)
    for h in (96, 336):
        hh = torch.full((x.shape[0],), float(h))
        assert torch.allclose(i_net(x, hh, hh), c_net(x, hh, hh), atol=1e-6)


def test_pool_query_has_no_effect_for_I(toy):
    net = M.build("I", 32, SEED).eval()
    x, _, _ = _batch(toy)
    h_true = torch.full((x.shape[0],), 96.0)
    a = net(x, h_true, torch.full_like(h_true, 96.0))
    b = net(x, h_true, torch.full_like(h_true, 336.0))
    assert torch.allclose(a, b, atol=1e-7)


def test_pool_query_does_change_C(toy):
    """Wiring check only: the query must reach the scorer. Says nothing about accuracy."""
    net = M.build("C", 32, SEED).eval()
    x, _, _ = _batch(toy)
    h_true = torch.full((x.shape[0],), 96.0)
    _, w96 = net(x, h_true, torch.full_like(h_true, 96.0), return_weights=True)
    _, w336 = net(x, h_true, torch.full_like(h_true, 336.0), return_weights=True)
    assert not torch.allclose(w96, w336, atol=1e-7)


def test_scorer_receives_gradient_in_C(toy):
    net = M.build("C", 32, SEED)
    net.train()
    x, y, h = _batch(toy)
    M.masked_mse(net(x, h, h), y, h).backward()
    grads = [p.grad for p in net.scorer.parameters()]
    assert all(g is not None for g in grads)
    assert max(float(g.abs().max()) for g in grads) > 0.0


def test_future_values_never_reach_the_input(toy):
    """Mutating y at timestamps >= o must not move the forecast or the weights."""
    net = M.build("C", 32, SEED).eval()
    o = toy.train_origins()[:8]
    c = np.zeros(8, dtype=np.int64)
    x = torch.from_numpy(toy.inputs(o, c))
    h = torch.full((8,), 96.0)
    y0, w0 = net(x, h, h, return_weights=True)

    poisoned = D.Dataset.from_array("toy2", toy.values.copy())
    poisoned.values[int(o.max()):] += 1000.0
    x2 = torch.from_numpy(
        (poisoned.values[o[:, None] + np.arange(-D.L, 0)[None, :], c[:, None]]
         - toy.mean[c][:, None]) / toy.std[c][:, None]
    )
    y1, w1 = net(x2, h, h, return_weights=True)
    assert torch.allclose(y0, y1, atol=1e-6)
    assert torch.allclose(w0, w1, atol=1e-6)


def test_tail_of_the_336_output_does_not_touch_the_H96_loss(toy):
    _, y, _ = _batch(toy)
    h = torch.full((y.shape[0],), 96.0)
    pred = torch.randn_like(y)
    a = M.masked_mse(pred, y, h)
    pred2 = pred.clone()
    pred2[:, 96:] += 50.0
    assert torch.allclose(a, M.masked_mse(pred2, y, h), atol=1e-7)


def test_key_join_survives_row_shuffling(toy):
    net = M.build("C", 32, SEED).eval()
    r = E.evaluate_split(net, toy, "test", 96, DEV, arm="C", seed=SEED, keep_keys=True)
    n = r["origin"].shape[0]
    perm = np.random.default_rng(1).permutation(n)
    key = np.stack([r["origin"][perm], r["channel"][perm]], axis=1)
    order = np.lexsort((key[:, 1], key[:, 0]))
    restored = r["se_sum"][perm][order]
    ref_order = np.lexsort((r["channel"], r["origin"]))
    assert np.allclose(restored, r["se_sum"][ref_order])
    assert np.isclose(r["se_sum"].sum() / r["count"].sum(), r["mse"])


# --- section 20 -------------------------------------------------------------


def test_uniform_pooling_matches_hand_computation():
    net = M.build("U", 32, SEED).eval()
    e = torch.arange(2 * 64 * 128, dtype=torch.float32).reshape(2, 64, 128)
    tokens, w = net.compress(e, torch.full((2,), 96.0))
    assert torch.allclose(w, torch.full_like(w, 0.5))
    expected = (e[:, 0::2, :] + e[:, 1::2, :]) / 2.0
    assert torch.allclose(tokens, expected, atol=1e-4)


def test_h_static_weights_ignore_content_but_follow_horizon(toy):
    net = M.build("H_STATIC", 32, SEED).eval()
    with torch.no_grad():
        net.static_logits.copy_(torch.tensor([[[2.0, -2.0]] * 32, [[-1.0, 1.0]] * 32]))
    x1, _, _ = _batch(toy)
    x2 = torch.randn_like(x1)
    h96 = torch.full((x1.shape[0],), 96.0)
    h336 = torch.full((x1.shape[0],), 336.0)
    _, wa = net(x1, h96, h96, return_weights=True)
    _, wb = net(x2, h96, h96, return_weights=True)
    assert torch.allclose(wa, wb, atol=1e-7)          # content independent
    _, wc = net(x1, h336, h336, return_weights=True)
    assert not torch.allclose(wa, wc, atol=1e-3)      # horizon dependent


def test_R_uses_a_reproducible_query_independent_of_the_true_horizon():
    o = np.array([2000, 2000, 2096, 2096], dtype=np.int64)
    c = np.array([0, 1, 0, 1], dtype=np.int64)
    h = np.array([96, 96, 96, 96], dtype=np.int64)
    a = D.fake_horizon_eval("toy", "test", o, c, h)
    b = D.fake_horizon_eval("toy", "test", o, c, h)
    assert np.array_equal(a, b)
    assert set(np.unique(a)).issubset({96, 336})
    shuffled = np.array([1, 3, 0, 2])
    assert np.array_equal(
        D.fake_horizon_eval("toy", "test", o[shuffled], c[shuffled], h[shuffled]), a[shuffled]
    )
    train_a = D.fake_horizon_train(SEED, "toy", 4, 8)
    train_b = D.fake_horizon_train(SEED, "toy", 4, 8)
    assert np.array_equal(train_a, train_b)


def test_training_schedule_is_shared_across_arms_and_reproducible(toy):
    a = D.make_schedule(toy, SEED, 8, 16)
    b = D.make_schedule(toy, SEED, 8, 16)
    assert a["schedule_sha256"] == b["schedule_sha256"]
    assert np.array_equal(a["channels"], b["channels"])
    assert (a["horizons"] == np.array([96, 336] * 4)).all()
    c = D.make_schedule(toy, SEED + 1, 8, 16)
    assert c["schedule_sha256"] != a["schedule_sha256"]


def test_update_checkpoint_reload_evaluate_roundtrip(toy, tmp_path):
    from experiments.hq_token_pilot_v1 import train as T

    cfg = {
        "max_updates": 2,
        "warmup": 1,
        "validate_every": 2,
        "batch": 8,
        "learning_rate": 2e-4,
        "weight_decay": 0.01,
        "grad_clip": 1.0,
    }
    rep = T.fit("C", toy, 32, SEED, cfg, str(tmp_path), device=DEV)
    assert rep["status"] == "COMPLETE"
    assert (tmp_path / "best.pt").exists() and (tmp_path / "initial.pt").exists()

    reloaded = M.build("C", 32, SEED)
    reloaded.load_state_dict(torch.load(tmp_path / "best.pt", map_location=DEV))
    for h in (96, 336):
        r = E.evaluate_split(reloaded, toy, "test", h, DEV, arm="C", seed=SEED, keep_keys=True)
        assert np.isfinite(r["mse"]) and r["n_keys"] > 0
        assert np.isclose(E.cell_error(r), r["mse"])


def test_bootstrap_keeps_all_channels_of_a_drawn_origin_together(toy):
    net = M.build("C", 32, SEED).eval()
    table = {}
    for arm in ("C", "I"):
        n2 = M.build(arm, 32, SEED).eval()
        for h in (96, 336):
            r = E.evaluate_split(n2, toy, "test", h, DEV, arm=arm, seed=SEED, keep_keys=True)
            table[(arm, "toy", h, SEED)] = r
    res = E.bootstrap_macro(table, "C", "I", ["toy"], [96, 336], [SEED], draws=25)
    assert np.isfinite(res["macro_mean"]) and res["lower95"] <= res["upper95"]
    assert res["block_origins"] == 15
    # every origin appears with the full channel set in the prepared arrays
    agg = E.seed_averaged(table, "C", "toy", 96, [SEED])
    counts = np.unique(agg["origin"], return_counts=True)[1]
    assert (counts == toy.n_channels).all()


def test_point_estimate_and_bootstrap_read_the_same_arrays(toy):
    net_c = M.build("C", 32, SEED).eval()
    net_i = M.build("I", 32, SEED).eval()
    table = {}
    for arm, net in (("C", net_c), ("I", net_i)):
        for h in (96, 336):
            table[(arm, "toy", h, SEED)] = E.evaluate_split(
                net, toy, "test", h, DEV, arm=arm, seed=SEED, keep_keys=True
            )
    ec = E.cell_error(E.seed_averaged(table, "C", "toy", 96, [SEED]))
    assert np.isclose(ec, table[("C", "toy", 96, SEED)]["mse"])
    ri, flag = E.relative_improvement(ec, E.cell_error(E.seed_averaged(table, "I", "toy", 96, [SEED])))
    assert flag is None and np.isfinite(ri)


def test_missing_cells_do_not_become_a_scientific_failure():
    """A cell that was never run must read as absent, never as a zero or a loss."""
    assert E.seed_averaged({}, "C", "nope", 96, [SEED]) is None
    assert E.macro_ri({("d", 96): None, ("d", 336): None}) is None
    ri, flag = E.relative_improvement(1.0, 0.0)
    assert ri is None and flag == "ZERO_BASELINE_LOSS"
    res = E.bootstrap_macro({}, "C", "I", ["d"], [96], [SEED], draws=5)
    assert res["macro"] is None and "NO_PAIRED_CELLS" in res["flags"]
