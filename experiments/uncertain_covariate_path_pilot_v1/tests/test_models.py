"""A07, A12, A13, A14: fairness and invariance contracts of the arms."""

import torch

from experiments.uncertain_covariate_path_pilot_v1.src.models import (
    ParticleForecaster,
    count_parameters,
)

SHAPE = dict(n_base=15, n_weather=3, n_farms=8)
B, K, T = 4, 11, 12


def build(arm: str, seed: int = 0) -> ParticleForecaster:
    torch.manual_seed(seed)
    return ParticleForecaster(arm=arm, **SHAPE)


def batch(seed: int = 3):
    g = torch.Generator().manual_seed(seed)
    base = torch.randn(B, T, SHAPE["n_base"], generator=g)
    weather = torch.randn(B, K, T, SHAPE["n_weather"], generator=g)
    farm = torch.randint(0, SHAPE["n_farms"], (B,), generator=g)
    return base, weather, farm


def test_a12_d_and_p_have_identical_parameter_counts():
    d, p = build("D"), build("P")
    assert count_parameters(d) == count_parameters(p)


def test_a13_d_and_p_share_the_same_initial_weights():
    d, p = build("D", seed=7), build("P", seed=7)
    dp, pp = dict(d.named_parameters()), dict(p.named_parameters())
    assert dp.keys() == pp.keys()
    for name in dp:
        assert torch.equal(dp[name], pp[name]), name


def test_a12_p_broken_matches_p_exactly():
    p, pb = build("P", seed=7), build("P_BROKEN", seed=7)
    assert count_parameters(p) == count_parameters(pb)
    for (n1, a), (n2, b) in zip(p.named_parameters(), pb.named_parameters()):
        assert n1 == n2 and torch.equal(a, b)


def test_a07_p_is_invariant_to_permuting_whole_member_paths():
    model = build("P").eval()
    base, weather, farm = batch()
    perm = torch.randperm(K)
    with torch.no_grad():
        out = model(base, weather, farm)
        out_perm = model(base, weather[:, perm], farm)
    assert (out - out_perm).abs().max().item() <= 1e-6


def test_d_is_blind_to_path_breaking_but_p_is_not():
    """D pools per lead, so a per-lead member shuffle cannot reach it. P can see it."""
    base, weather, farm = batch()
    broken = weather.clone()
    g = torch.Generator().manual_seed(11)
    for t in range(T):
        broken[:, :, t, :] = weather[:, torch.randperm(K, generator=g), t, :]

    d = build("D").eval()
    p = build("P").eval()
    with torch.no_grad():
        assert (d(base, weather, farm) - d(base, broken, farm)).abs().max().item() <= 1e-5
        assert (p(base, weather, farm) - p(base, broken, farm)).abs().max().item() > 1e-4


def test_a14_h_output_does_not_depend_on_weather():
    model = build("H").eval()
    base, weather, farm = batch()
    with torch.no_grad():
        a = model(base, weather, farm)
        b = model(base, torch.randn_like(weather) * 50.0, farm)
    assert torch.equal(a, b)


def test_output_shape_is_particles_per_lead():
    model = build("P").eval()
    base, weather, farm = batch()
    with torch.no_grad():
        assert model(base, weather, farm).shape == (B, T, model.n_particles)
