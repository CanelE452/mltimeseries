"""Assemble the comparator table and the relative-improvement effects.

Three TSFM comparators per task (Section 18):

F_FIXED             TimesFM 3.0. One deployable model, chosen once, and the only
                    primary whose card documents a fev-bench exclusion.
F_FAMILY_ENVELOPE   the taskwise minimum over the three primaries. Chosen with
                    the test scores in view, so it is not a deployable router -
                    it is deliberately the hardest TSFM comparator to beat.
F_MEAN              their average. Descriptive only, never a gate.

RI(S over F) = 100 * (Loss_F - Loss_S) / Loss_F, positive when the specialist wins.
Two specialist seeds are averaged into the primary estimate and also reported
separately, because a conclusion that flips with the seed is not a conclusion.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import paths

PRIMARY = ["chronos-2", "tirex-2", "timesfm-3.0"]
F_FIXED_MODEL = "timesfm-3.0"


def _relative_improvement(specialist: float, comparator: float) -> float:
    if not np.isfinite(specialist) or not np.isfinite(comparator) or comparator <= 0:
        return float("nan")
    return 100.0 * (comparator - specialist) / comparator


def tsfm_table() -> pd.DataFrame:
    """One row per (split, task) with every TSFM comparator and the old baselines."""
    development = pd.read_csv(paths.RESULTS / "development_tsfm_results.csv")
    holdout = pd.read_csv(paths.RESULTS / "holdout_tsfm_results.csv")
    frames = []
    for frame, split in ((development, "development"), (holdout, "holdout")):
        pivot = frame.pivot_table(
            index="task_uid", columns="model", values="native_score", aggfunc="first"
        )
        pivot = pivot.assign(split=split)
        frames.append(pivot)
    table = pd.concat(frames).reset_index()
    table["F_FIXED"] = table[F_FIXED_MODEL]
    table["F_FAMILY_ENVELOPE"] = table[PRIMARY].min(axis=1)
    table["F_ENVELOPE_MODEL"] = table[PRIMARY].idxmin(axis=1)
    table["F_MEAN"] = table[PRIMARY].mean(axis=1)
    return table


def specialist_table() -> pd.DataFrame:
    """Specialist scores per task and seed, plus the two-seed mean."""
    scores = pd.read_csv(paths.RESULTS / "specialist_scores.csv")
    wide = scores.pivot_table(
        index=["task_uid", "label"], columns="seed", values="native_score", aggfunc="first"
    )
    wide.columns = [f"seed_{c}" for c in wide.columns]
    wide["mean_over_seeds"] = wide.mean(axis=1)
    wide["n_seeds"] = wide[[c for c in wide.columns if c.startswith("seed_")]].notna().sum(axis=1)
    return wide.reset_index()


def build() -> tuple[pd.DataFrame, pd.DataFrame]:
    tsfm = tsfm_table()
    specialist = specialist_table()
    seed_columns = [c for c in specialist.columns if c.startswith("seed_")]

    merged = specialist.merge(tsfm, on="task_uid", how="left")
    rows = []
    for _, row in merged.iterrows():
        entry = {
            "split": row.split,
            "task_uid": row.task_uid,
            "task": row.task_uid.split("::")[1],
            "label": row.label,
            "specialist_score": row.mean_over_seeds,
            "n_seeds": int(row.n_seeds),
            "F_FIXED": row.F_FIXED,
            "F_FAMILY_ENVELOPE": row.F_FAMILY_ENVELOPE,
            "F_ENVELOPE_MODEL": row.F_ENVELOPE_MODEL,
            "F_MEAN": row.F_MEAN,
            "linear_ar": row.get("linear-ar-specialist"),
            "seasonal_naive": row.get("seasonal-naive"),
            "RI_vs_envelope_pct": _relative_improvement(row.mean_over_seeds, row.F_FAMILY_ENVELOPE),
            "RI_vs_fixed_pct": _relative_improvement(row.mean_over_seeds, row.F_FIXED),
            "RI_vs_mean_pct": _relative_improvement(row.mean_over_seeds, row.F_MEAN),
            "RI_vs_linear_ar_pct": _relative_improvement(
                row.mean_over_seeds, row.get("linear-ar-specialist")
            ),
            "RI_vs_naive_pct": _relative_improvement(row.mean_over_seeds, row.get("seasonal-naive")),
        }
        for column in seed_columns:
            entry[f"RI_vs_envelope_{column}_pct"] = _relative_improvement(
                row[column], row.F_FAMILY_ENVELOPE
            )
        rows.append(entry)
    effects = pd.DataFrame(rows)

    aggregates = []
    for (split, label), group in effects.groupby(["split", "label"]):
        usable = group[np.isfinite(group.RI_vs_envelope_pct)]
        entry = {
            "split": split,
            "label": label,
            "n_tasks": int(len(group)),
            "n_usable": int(len(usable)),
            "median_RI_vs_envelope_pct": float(usable.RI_vs_envelope_pct.median()),
            "mean_RI_vs_envelope_pct": float(usable.RI_vs_envelope_pct.mean()),
            "median_RI_vs_fixed_pct": float(usable.RI_vs_fixed_pct.median()),
            "median_RI_vs_mean_pct": float(usable.RI_vs_mean_pct.median()),
            "median_RI_vs_linear_ar_pct": float(usable.RI_vs_linear_ar_pct.median()),
            "median_RI_vs_naive_pct": float(usable.RI_vs_naive_pct.median()),
            "wins_vs_envelope": int((usable.RI_vs_envelope_pct > 0).sum()),
            "wins_vs_fixed": int((usable.RI_vs_fixed_pct > 0).sum()),
            "wins_vs_linear_ar": int((usable.RI_vs_linear_ar_pct > 0).sum()),
            "wins_vs_naive": int((usable.RI_vs_naive_pct > 0).sum()),
        }
        for column in seed_columns:
            key = f"RI_vs_envelope_{column}_pct"
            entry[f"median_{key}"] = float(usable[key].median()) if key in usable else float("nan")
        aggregates.append(entry)
    return effects, pd.DataFrame(aggregates)


def main() -> None:
    effects, aggregates = build()
    effects.to_csv(paths.RESULTS / "comparison_effects.csv", index=False)
    aggregates.to_csv(paths.RESULTS / "aggregate_effects.csv", index=False)
    tsfm_table().to_csv(paths.RESULTS / "comparator_scores.csv", index=False)

    for split in ("development", "holdout"):
        subset = effects[(effects.split == split) & (effects.label == "S_ENSEMBLE")]
        if subset.empty:
            continue
        print(f"\n=== {split}: S_ENSEMBLE ===")
        print(
            subset[
                [
                    "task",
                    "specialist_score",
                    "F_FAMILY_ENVELOPE",
                    "F_FIXED",
                    "linear_ar",
                    "RI_vs_envelope_pct",
                    "RI_vs_fixed_pct",
                    "RI_vs_linear_ar_pct",
                ]
            ]
            .round(3)
            .to_string(index=False)
        )
    print("\n=== aggregates ===")
    print(
        aggregates[
            [
                "split",
                "label",
                "n_usable",
                "median_RI_vs_envelope_pct",
                "median_RI_vs_fixed_pct",
                "wins_vs_envelope",
                "median_RI_vs_linear_ar_pct",
                "wins_vs_linear_ar",
            ]
        ]
        .round(3)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
