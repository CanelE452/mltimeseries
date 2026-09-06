"""Synthetic unit control (instruction section 17).

A positive control for the D/P implementations only. Two example types share
identical per-lead member marginals and differ only in the joint path:

  type A members: [0, 0] and [1, 1]
  type B members: [0, 1] and [1, 0]

At every lead both types show the same half-zero / half-one member set, so a
distribution-first encoder cannot separate them; a path-first encoder can. The
target depends on the type, so P should fit and D should sit at the intercept.

This checks the code, not wind power. It is never used as evidence about the
real experiment.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.uncertain_covariate_path_pilot_v1.src.crps import crps_loss
from experiments.uncertain_covariate_path_pilot_v1.src.models import ParticleForecaster

K, T, N_TRAIN, N_TEST = 8, 2, 2048, 512
TARGET = {"A": 1.0, "B": -1.0}
NOISE = 0.1


def make_split(n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    is_a = torch.randint(0, 2, (n,), generator=g).bool()
    weather = torch.zeros(n, K, T, 1)
    half = K // 2
    # type A: half the members stay low, half stay high
    weather[is_a, :half, 0, 0], weather[is_a, :half, 1, 0] = 0.0, 0.0
    weather[is_a, half:, 0, 0], weather[is_a, half:, 1, 0] = 1.0, 1.0
    # type B: the same per-lead sets, but every member crosses over
    weather[~is_a, :half, 0, 0], weather[~is_a, :half, 1, 0] = 0.0, 1.0
    weather[~is_a, half:, 0, 0], weather[~is_a, half:, 1, 0] = 1.0, 0.0
    # break the member ordering so the split is not readable from the index
    for i in range(n):
        weather[i] = weather[i][torch.randperm(K, generator=g)]
    y = torch.where(is_a, TARGET["A"], TARGET["B"]).unsqueeze(1).expand(n, T).clone()
    y = y + NOISE * torch.randn(n, T, generator=g)
    base = torch.zeros(n, T, 1)
    farm = torch.zeros(n, dtype=torch.long)
    return base, weather, farm, y


def marginals_match(weather: torch.Tensor, is_a: torch.Tensor) -> bool:
    """Both types must show the same sorted member values at every lead."""
    a = weather[is_a][:, :, :, 0].sort(dim=1).values.mean(0)
    b = weather[~is_a][:, :, :, 0].sort(dim=1).values.mean(0)
    return torch.allclose(a, b, atol=1e-6)


def train_arm(arm: str, data, epochs: int = 60, seed: int = 0) -> float:
    torch.manual_seed(seed)
    model = ParticleForecaster(arm=arm, n_base=1, n_weather=1, n_farms=1)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    (base, weather, farm, y), (tb, tw, tf, ty) = data
    n = base.shape[0]
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, 256):
            idx = perm[i : i + 256]
            loss = crps_loss(model(base[idx], weather[idx], farm[idx]), y[idx])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    model.eval()
    with torch.no_grad():
        return crps_loss(model(tb, tw, tf), ty).item()


def main() -> int:
    g = torch.Generator().manual_seed(0)
    train = make_split(N_TRAIN, seed=1)
    test = make_split(N_TEST, seed=2)

    is_a_train = train[3][:, 0] > 0
    assert marginals_match(train[1], is_a_train), "per-lead marginals differ; toy is invalid"

    results = {}
    for arm in ["H", "M", "S", "D", "P"]:
        results[arm] = train_arm(arm, (train, test))
        print(f"  {arm:2s} test CRPS {results[arm]:.4f}", flush=True)

    verdict = {
        "per_lead_marginals_identical": True,
        "test_crps": results,
        "p_better_than_d_percent": 100 * (results["D"] - results["P"]) / results["D"],
        "p_better_than_s_percent": 100 * (results["S"] - results["P"]) / results["S"],
        "d_close_to_h_percent": 100 * abs(results["D"] - results["H"]) / results["H"],
        "implementation_control_passed": bool(
            results["P"] < 0.5 * results["D"] and results["P"] < 0.5 * results["S"]
        ),
        "scope": (
            "Implementation check for the D and P encoders only. Not evidence about "
            "wind power or about any real forecasting gain."
        ),
    }
    out = (
        Path(__file__).resolve().parents[3]
        / "results/uncertain_covariate_path_pilot_v1/synthetic_unit_control.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(verdict, indent=2))
    print(json.dumps(verdict, indent=2))
    return 0 if verdict["implementation_control_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
