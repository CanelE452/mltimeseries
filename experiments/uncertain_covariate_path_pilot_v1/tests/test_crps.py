"""A11: the O(R^2) reference and the sorted form must agree."""

import torch

from experiments.uncertain_covariate_path_pilot_v1.src.crps import (
    crps_efficient,
    crps_reference,
)


def test_reference_matches_efficient_random():
    torch.manual_seed(0)
    for R in (2, 3, 8, 32, 64):
        particles = torch.randn(17, 12, R, dtype=torch.float64) * 3.0 + 1.0
        y = torch.randn(17, 12, dtype=torch.float64)
        a = crps_reference(particles, y)
        b = crps_efficient(particles, y)
        assert torch.allclose(a, b, atol=1e-10), (R, (a - b).abs().max().item())


def test_degenerate_particles_reduce_to_absolute_error():
    particles = torch.full((5, 32), 2.0, dtype=torch.float64)
    y = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
    assert torch.allclose(crps_efficient(particles, y), (y - 2.0).abs())


def test_crps_is_non_negative():
    torch.manual_seed(1)
    particles = torch.randn(64, 32, dtype=torch.float64)
    y = torch.randn(64, dtype=torch.float64)
    assert (crps_efficient(particles, y) >= -1e-12).all()
