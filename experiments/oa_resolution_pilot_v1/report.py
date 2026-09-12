"""Aggregation, contrasts, the paired time-block bootstrap and the verdict.

Point estimates and confidence intervals read the same frozen per-key error array,
so they can never disagree about which keys were scored.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DATASETS = ("jena", "uci")
OPS = ("END_BIN", "INTERVAL_MEAN")
INTERP_R = (3, 6)
SEEN_R = (2, 4, 8)
EXTRAP_R = (12,)

BOOTSTRAP_SEED = 2026090699
BOOTSTRAP_DRAWS = 1000
BLOCK_DAYS = 7
ORIGIN_STRIDE = 72            # 12 h
ORIGINS_PER_BLOCK = BLOCK_DAYS * 24 // 12     # 14

GO = {
    "interp_O_vs_M_macro_min": 1.0,
    "interp_O_vs_R_macro_min": 0.5,
    "seen_O_vs_M_macro_min": -0.5,
}


def to_frame(rows: np.ndarray, dataset: str) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df.insert(0, "dataset", dataset)
    df["mse10"] = df["SE10"] / df["count10"]
    df["mse60"] = df["SE60"] / df["count60"]
    df["mae10"] = df["AE10"] / df["count10"]
    df["mae60"] = df["AE60"] / df["count60"]
    df["primary"] = 0.5 * df["mse10"] + 0.5 * df["mse60"]
    return df


def seed_averaged(df: pd.DataFrame) -> pd.DataFrame:
    """One value per logical key with the two model seeds averaged.  This is the
    unit the bootstrap resamples; the per-seed numbers are reported separately."""
    keys = ["dataset", "split", "origin", "channel", "operation", "r", "role", "arm"]
    return df.groupby(keys, as_index=False)[["primary", "mse10", "mse60", "mae10", "mae60"]].mean()


def cell_table(df: pd.DataFrame, split: str = "test") -> pd.DataFrame:
    """Mean loss per (dataset, operation, r, arm) over origins, channels and seeds."""
    d = df[df["split"] == split]
    return d.groupby(["dataset", "operation", "r", "role", "arm"], as_index=False)["primary"].mean()


def macro(cells: pd.DataFrame, rs: tuple[int, ...], arm_a: str, arm_b: str) -> dict:
    """Equal-weight macro over the (dataset, operation, r) cells for the given
    resolutions.  Every cell counts once, so the dataset with more channels cannot
    dominate."""
    sub = cells[cells["r"].isin(rs)]
    piv = sub.pivot_table(index=["dataset", "operation", "r"], columns="arm", values="primary")
    if arm_a not in piv or arm_b not in piv:
        return {"macro_relative_improvement_pct": float("nan"), "n_cells": 0, "cells": []}
    ri = 100.0 * (1.0 - piv[arm_a] / piv[arm_b])
    return {
        "macro_relative_improvement_pct": float(ri.mean()),
        "n_cells": int(len(ri)),
        "cells": [{"dataset": i[0], "operation": i[1], "r": int(i[2]),
                   arm_a: float(piv.loc[i, arm_a]), arm_b: float(piv.loc[i, arm_b]),
                   "relative_improvement_pct": float(v)} for i, v in ri.items()],
    }


def per_dataset_macro(cells: pd.DataFrame, rs, arm_a: str, arm_b: str) -> dict:
    out = {}
    for ds in DATASETS:
        sub = cells[cells["dataset"] == ds]
        if len(sub):
            out[ds] = macro(sub, rs, arm_a, arm_b)["macro_relative_improvement_pct"]
    return out


# ------------------------------------------------------------------- bootstrap


BINS_PER_DAY = 144


def _block_id(origins: np.ndarray) -> np.ndarray:
    """7-day blocks of consecutive evaluation origins, in base-bin coordinates."""
    return origins // (BLOCK_DAYS * BINS_PER_DAY)


def paired_block_bootstrap(df: pd.DataFrame, rs: tuple[int, ...], arm_a: str, arm_b: str,
                           draws: int = BOOTSTRAP_DRAWS, seed: int = BOOTSTRAP_SEED) -> dict:
    """Moving-block bootstrap over evaluation origins.

    Resampling happens at the origin-block level, independently per dataset.  A
    block carries every channel, operation, resolution and arm belonging to its
    origins, so the arms stay paired and the macro is recomputed inside each draw
    from exactly the keys the point estimate uses.
    """
    d = df[(df["split"] == "test") & (df["r"].isin(rs))].copy()
    d["block"] = _block_id(d["origin"].to_numpy())

    per_ds = {}
    for ds in DATASETS:
        sub = d[d["dataset"] == ds]
        if not len(sub):
            continue
        blocks = np.sort(sub["block"].unique())
        by_block = {b: sub[sub["block"] == b] for b in blocks}
        per_ds[ds] = (blocks, by_block)

    if not per_ds:
        return {"status": "NO_DATA"}

    n_eff = min(len(b) for b, _ in per_ds.values())
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(draws):
        parts = []
        for ds, (blocks, by_block) in per_ds.items():
            pick = rng.choice(blocks, size=len(blocks), replace=True)
            parts.append(pd.concat([by_block[b] for b in pick], ignore_index=True))
        draw = pd.concat(parts, ignore_index=True)
        cells = draw.groupby(["dataset", "operation", "r", "arm"], as_index=False)["primary"].mean()
        piv = cells.pivot_table(index=["dataset", "operation", "r"], columns="arm", values="primary")
        if arm_a in piv and arm_b in piv:
            stats.append(float((100.0 * (1.0 - piv[arm_a] / piv[arm_b])).mean()))

    s = np.array(stats)
    return {
        "arm_a": arm_a, "arm_b": arm_b, "resolutions": list(rs), "draws": int(len(s)),
        "block_days": BLOCK_DAYS, "origins_per_block": ORIGINS_PER_BLOCK,
        "effective_blocks_min_over_datasets": int(n_eff),
        "low_effective_time_blocks": bool(n_eff < 5),
        "mean": float(s.mean()), "lower95": float(np.percentile(s, 2.5)),
        "upper95": float(np.percentile(s, 97.5)),
    }


# ---------------------------------------------------------------------- verdict


def decide(contrasts: dict, bootstrap: dict, per_seed: dict, per_dataset: dict,
           bootstrap_M_vs_R: dict | None = None) -> dict:
    """The pre-registered gate, evaluated in order.  No threshold moves after the
    numbers are in.

    The section 35 fallback tokens turn on whether M is *significantly* better than
    R, so that clause is read against a paired bootstrap interval and not against a
    point estimate alone.  Without an interval the clause cannot be satisfied and
    the outcome falls through to INCONCLUSIVE.
    """
    interp_om = contrasts["unseen_interpolation"]["O_vs_M"]["macro_relative_improvement_pct"]
    interp_or = contrasts["unseen_interpolation"]["O_vs_R"]["macro_relative_improvement_pct"]
    interp_mr = contrasts["unseen_interpolation"]["M_vs_R"]["macro_relative_improvement_pct"]
    seen_om = contrasts["seen"]["O_vs_M"]["macro_relative_improvement_pct"]

    checks = {
        "1_interp_O_vs_M_macro_ge_1pct": interp_om >= GO["interp_O_vs_M_macro_min"],
        "2_both_datasets_positive": all(v > 0 for v in per_dataset.values()),
        "3_both_seeds_positive": all(v > 0 for v in per_seed.values()),
        "4_bootstrap_lower95_gt_0": bootstrap.get("lower95", float("-inf")) > 0,
        "5_interp_O_vs_R_macro_ge_0p5pct": interp_or >= GO["interp_O_vs_R_macro_min"],
        "6_seen_O_vs_M_macro_ge_minus0p5pct": seen_om >= GO["seen_O_vs_M_macro_min"],
    }

    mr_lower = (bootstrap_M_vs_R or {}).get("lower95")
    mr_upper = (bootstrap_M_vs_R or {}).get("upper95")
    M_beats_R_significantly = mr_lower is not None and mr_lower > 0
    R_at_least_as_good = mr_upper is not None and mr_upper <= 0
    O_matches_M = abs(interp_om) < GO["interp_O_vs_M_macro_min"]

    if all(checks.values()):
        token = "OPERATOR_AWARE_REPRESENTATION_PROMISING"
    elif O_matches_M and R_at_least_as_good:
        token = "RESAMPLING_SUFFICIENT"
    elif O_matches_M and M_beats_R_significantly:
        token = "METADATA_MODEL_SUFFICIENT"
    elif interp_om >= GO["interp_O_vs_M_macro_min"] and interp_or < GO["interp_O_vs_R_macro_min"]:
        token = "OPERATOR_REPRESENTATION_NOT_NEEDED_OVER_RESAMPLING"
    else:
        token = "INCONCLUSIVE"

    return {
        "scientific_decision": token,
        "checks": {k: bool(v) for k, v in checks.items()},
        "fallback_clause_inputs": {
            "O_matches_M_within_1pct": bool(O_matches_M),
            "M_beats_R_significantly": bool(M_beats_R_significantly),
            "R_at_least_as_good_as_M": bool(R_at_least_as_good),
            "M_vs_R_bootstrap": bootstrap_M_vs_R,
        },
        "primary_O_vs_M_unseen_interpolation_macro_pct": interp_om,
        "O_vs_R_unseen_interpolation_macro_pct": interp_or,
        "M_vs_R_unseen_interpolation_macro_pct": interp_mr,
        "seen_O_vs_M_macro_pct": seen_om,
        "per_seed_macro_O_vs_M": per_seed,
        "per_dataset_macro_O_vs_M": per_dataset,
        "bootstrap": bootstrap,
        "thresholds": GO,
    }


def markdown_table(cells: pd.DataFrame, rs: tuple[int, ...]) -> str:
    sub = cells[cells["r"].isin(rs)]
    piv = sub.pivot_table(index=["dataset", "operation", "r"], columns="arm", values="primary")
    lines = ["| dataset | operation | r | R | M | O | O vs M | O vs R |",
             "|---|---|---|---|---|---|---|---|"]
    for idx, row in piv.iterrows():
        ds, op, r = idx
        rr, mm, oo = row.get("R", np.nan), row.get("M", np.nan), row.get("O", np.nan)
        lines.append(
            f"| {ds} | {op} | {int(r)} | {rr:.5f} | {mm:.5f} | {oo:.5f} | "
            f"{100*(1-oo/mm):+.2f}% | {100*(1-oo/rr):+.2f}% |"
        )
    return "\n".join(lines)
