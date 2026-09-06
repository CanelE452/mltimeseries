"""Deterministic path breaking for the P_BROKEN arm (instruction section 15).

For each example and each lead the member index is permuted independently, from
a hash of (example_id, lead, fixed seed). Every lead therefore keeps exactly the
same set of member values -- so its mean, standard deviation and quantiles are
untouched -- while the link that ties member k at 6 h to member k at 72 h is
destroyed. P_BROKEN trains on the broken data, it is not a test-time corruption.
"""

from __future__ import annotations

import hashlib

import numpy as np

FIXED_SHUFFLE_SEED = 20260906


def permutation_for(example_id: str, lead_index: int, n_members: int) -> np.ndarray:
    digest = hashlib.sha256(
        f"{example_id}|{lead_index}|{FIXED_SHUFFLE_SEED}".encode()
    ).digest()
    seed = int.from_bytes(digest[:8], "big") % (2**32)
    return np.random.default_rng(seed).permutation(n_members)


def break_paths(weather: np.ndarray, example_ids: np.ndarray) -> np.ndarray:
    """weather: [N, K, T, D] -> the same values, with member linkage across leads gone."""
    if len(weather) != len(example_ids):
        raise ValueError("weather and example_ids disagree in length")
    out = np.empty_like(weather)
    n_members, n_leads = weather.shape[1], weather.shape[2]
    for i, example_id in enumerate(example_ids):
        for t in range(n_leads):
            out[i, :, t, :] = weather[i, permutation_for(str(example_id), t, n_members), t, :]
    return out
