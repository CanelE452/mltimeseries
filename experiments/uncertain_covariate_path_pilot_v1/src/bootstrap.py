"""Paired 7-day time-block cluster bootstrap on the 2021 test split.

Drawing a block takes every origin, farm, lead and both arms inside that week, so
the pairing survives and the dependence between neighbouring origins is respected.
Independent row resampling would treat 12 leads of one origin as 12 independent
observations and would shrink the interval for free.

2000 replicates are 2000 resamples of the same year. They are not 2000
independent samples and are never described as such.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
from experiments.uncertain_covariate_path_pilot_v1.src.paths import out_dir

BLOCK_DAYS = 7
N_REPS = 2000
BOOTSTRAP_SEED = 20260906
PRIMARY = [("P", "D"), ("P", "S")]
SECONDARY = [("P", "P_BROKEN"), ("S", "M"), ("M", "H"), ("P", "MC")]


def relative_improvement(loss_a: float, loss_b: float) -> float:
    """RI(A over B) = 100 * (L_B - L_A) / L_B. Positive means A is better."""
    return 100.0 * (loss_b - loss_a) / loss_b


def farm_macro(loss: np.ndarray, farm_idx: np.ndarray, farms: np.ndarray) -> float:
    per_farm = [loss[farm_idx == f].mean() for f in farms if (farm_idx == f).any()]
    return float(np.mean(per_farm))


def run(tag: str = "full", split: str = "test") -> dict:
    store = np.load(out_dir(tag) / "per_example_losses.npz", allow_pickle=False)
    farm_idx = store[f"meta__{split}__farm_idx"]
    origins = pd.to_datetime(store[f"meta__{split}__origin"])
    farms = np.unique(farm_idx)

    start = origins.min().normalize()
    block = ((origins - start).days // BLOCK_DAYS).to_numpy()
    blocks = np.unique(block)
    members = {b: np.flatnonzero(block == b) for b in blocks}

    arms = sorted({k.split("__")[0] for k in store if not k.startswith("meta__")})
    per_lead = {a: store[f"{a}__{split}"] for a in arms if f"{a}__{split}" in store}
    per_example = {a: v.mean(axis=1) for a, v in per_lead.items()}

    point = {a: farm_macro(v, farm_idx, farms) for a, v in per_example.items()}

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(0, len(blocks), size=(N_REPS, len(blocks)))
    replicate = {a: np.empty(N_REPS) for a in per_example}
    for r in range(N_REPS):
        rows = np.concatenate([members[blocks[i]] for i in draws[r]])
        fi = farm_idx[rows]
        for a, v in per_example.items():
            replicate[a][r] = farm_macro(v[rows], fi, farms)

    contrasts = []
    for kind, pairs in (("primary", PRIMARY), ("secondary", SECONDARY)):
        for a, b in pairs:
            if a not in point or b not in point:
                continue
            ri = np.array([relative_improvement(replicate[a][r], replicate[b][r]) for r in range(N_REPS)])
            contrasts.append(
                {
                    "kind": kind,
                    "contrast": f"{a} over {b}",
                    "arm_a": a,
                    "arm_b": b,
                    "loss_a": point[a],
                    "loss_b": point[b],
                    "ri_percent": relative_improvement(point[a], point[b]),
                    "ci_lower": float(np.percentile(ri, 2.5)),
                    "ci_upper": float(np.percentile(ri, 97.5)),
                    "share_positive": float((ri > 0).mean()),
                }
            )

    result = {
        "split": split,
        "block_days": BLOCK_DAYS,
        "n_blocks": int(len(blocks)),
        "n_replicates": N_REPS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "resampling": (
            "clusters of 7 consecutive days; a drawn block contributes every origin, farm, "
            "lead and arm inside it, so contrasts stay paired"
        ),
        "estimand": "farm equal-weight macro of the per-example mean scaled CRPS, 3-seed mean",
        "reading": (
            f"{N_REPS} resamples of one test year, not {N_REPS} independent samples"
        ),
        "point_estimates": point,
        "contrasts": contrasts,
    }
    (out_dir(tag) / "bootstrap_primary.json").write_text(json.dumps(result, indent=2))
    pd.DataFrame(contrasts).to_csv(out_dir(tag) / "paired_effects.csv", index=False)
    print(pd.DataFrame(contrasts)[
        ["kind", "contrast", "ri_percent", "ci_lower", "ci_upper"]
    ].to_string(index=False))
    return result


def seed_table(tag: str = "full", split: str = "test") -> pd.DataFrame:
    """Per-seed RI, reported as it is. Three seeds do not make a confidence interval."""
    metrics = pd.read_csv(out_dir(tag) / "metrics.csv")
    metrics = metrics[metrics["split"] == split]
    wide = metrics.pivot_table(index="seed", columns="arm", values="macro_scaled_crps")
    rows = []
    for kind, pairs in (("primary", PRIMARY), ("secondary", SECONDARY)):
        for a, b in pairs:
            if a not in wide or b not in wide:
                continue
            for seed in wide.index:
                rows.append(
                    {
                        "kind": kind,
                        "contrast": f"{a} over {b}",
                        "seed": int(seed),
                        "loss_a": wide.loc[seed, a],
                        "loss_b": wide.loc[seed, b],
                        "ri_percent": relative_improvement(wide.loc[seed, a], wide.loc[seed, b]),
                    }
                )
    table = pd.DataFrame(rows)
    table.to_csv(out_dir(tag) / "seed_effects.csv", index=False)
    return table


def main() -> int:
    run()
    print()
    print(seed_table().to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
