"""Empirical CRPS for a particle predictive distribution.

CRPS(a, y) = mean_r |a_r - y| - 0.5 * mean_{r,s} |a_r - a_s|

Both the O(R^2) reference and the O(R log R) form are kept: the reference
defines the estimand, the sorted form is what training uses. Test A11 checks
that they agree.
"""

from __future__ import annotations

import torch


def crps_reference(particles: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """O(R^2) definition. particles: (..., R), y: (...) -> (...)."""
    term1 = (particles - y.unsqueeze(-1)).abs().mean(dim=-1)
    pairwise = (particles.unsqueeze(-1) - particles.unsqueeze(-2)).abs()
    term2 = pairwise.mean(dim=(-2, -1))
    return term1 - 0.5 * term2


def crps_efficient(particles: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Sorted form of the same quantity.

    sum_{r,s} |a_r - a_s| = 2 * sum_i (2i - R - 1) * a_(i)   (i = 1..R, sorted)
    so 0.5 * mean_{r,s} |a_r - a_s| = (1 / R^2) * sum_i (2i - R - 1) * a_(i).
    """
    R = particles.shape[-1]
    term1 = (particles - y.unsqueeze(-1)).abs().mean(dim=-1)
    ordered, _ = torch.sort(particles, dim=-1)
    i = torch.arange(1, R + 1, device=particles.device, dtype=particles.dtype)
    weights = 2.0 * i - R - 1.0
    term2 = (ordered * weights).sum(dim=-1) / (R * R)
    return term1 - term2


def crps_loss(particles: torch.Tensor, y: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Mean empirical CRPS over batch and lead. mask selects evaluated entries."""
    per_entry = crps_efficient(particles, y)
    if mask is None:
        return per_entry.mean()
    mask = mask.to(per_entry.dtype)
    return (per_entry * mask).sum() / mask.sum().clamp(min=1.0)
