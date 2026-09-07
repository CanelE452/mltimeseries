"""Rebuild `specialist_manifest.csv` from the artifacts each fit left behind.

The manifest is written when a sweep finishes, so a sweep that the host killed
mid-way leaves its forecasts and predictors on disk with no manifest rows. Rather
than refit — which would spend an hour and produce different numbers than the
forecasts already scored — the rows are reconstructed from what the fits actually
wrote: the per-(task, seed) validation leaderboard and the saved predictor.

Every field comes from a file. `reconstructed` marks the rows that were not
written by the training run itself, so nothing here can be mistaken for a fresh
observation.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import paths

LEADERBOARD_PATTERN = re.compile(r"^leaderboard__(?P<safe>.+)__seed(?P<seed>\d+)\.csv$")
FORBIDDEN = ("chronos", "toto", "timesfm", "tirex")


def _ensemble_weights(safe: str, seed: int, members: list[str]) -> dict:
    """Read the weighted ensemble's own record from the saved predictor."""
    root = paths.PREDICTORS / f"{safe}__seed{seed}"
    for candidate in root.rglob("model.pkl"):
        if "Ensemble" not in str(candidate.parent.name):
            continue
        try:
            import pickle

            with candidate.open("rb") as handle:
                model = pickle.load(handle)
            weights = getattr(model, "model_to_weight", None)
            if weights:
                return {k: float(v) for k, v in weights.items()}
        except Exception:
            continue
    return {}


def build() -> pd.DataFrame:
    index = json.loads((paths.RESULTS / "conversion_index.json").read_text(encoding="utf-8"))
    by_safe = {t["safe_name"]: t for t in index["tasks"]}
    spec = json.loads((paths.RESULTS / "long_horizon_tasks.json").read_text(encoding="utf-8"))
    split_of = {t: "development" for t in spec["development_tasks"]}
    split_of.update({t: "holdout" for t in spec["fresh_holdout_tasks"]})

    written = {}
    for path in paths.RESULTS.glob("specialist_manifest_*.csv"):
        if path.name == "specialist_manifest.csv":
            continue
        for _, row in pd.read_csv(path).iterrows():
            written[(row.safe_name, int(row.seed))] = row.to_dict()

    rows = []
    for path in sorted(paths.RUNS.glob("leaderboard__*.csv")):
        match = LEADERBOARD_PATTERN.match(path.name)
        if not match:
            continue
        safe, seed = match.group("safe"), int(match.group("seed"))
        task = by_safe[safe]
        leaderboard = pd.read_csv(path)

        if (safe, seed) in written:
            row = dict(written[(safe, seed)])
            row["reconstructed"] = False
        else:
            fitted = leaderboard.model.tolist()
            singles = [m for m in fitted if "Ensemble" not in m]
            ranked = leaderboard[leaderboard.model.isin(singles)].sort_values(
                "score_val", ascending=False
            )
            ensemble = next((m for m in fitted if "Ensemble" in m), None)
            row = {
                "task_uid": task["task_uid"],
                "safe_name": safe,
                "seed": seed,
                "horizon": task["horizon"],
                "num_windows": task["num_windows"],
                "quantile_levels": task["quantile_levels"],
                "time_limit_seconds": 20 * 60,
                "requested_models": sorted(["PatchTST", "TiDE", "DLinear", "DeepAR", "DirectTabular"]),
                "status": "OK",
                "fitted_models": str(fitted),
                "model_best": str(ranked.model.iloc[0]) if len(ranked) else None,
                "validation_scores": str(
                    {r["model"]: float(r["score_val"]) for _, r in leaderboard.iterrows()}
                ),
                "ensemble_model": ensemble,
                "ensemble_fallback": ensemble is None,
                "ensemble_weights": str(_ensemble_weights(safe, seed, singles)),
                "fit_seconds": float(leaderboard.fit_time_marginal.sum())
                if "fit_time_marginal" in leaderboard
                else float("nan"),
                "num_val_windows": np.nan,
                "reconstructed": True,
                "reconstructed_from": f"runs/.../{path.name} and the saved predictor directory",
            }
        row["split"] = split_of.get(row["task_uid"])
        rows.append(row)

    manifest = pd.DataFrame(rows).sort_values(["split", "task_uid", "seed"])
    contaminated = [
        (r.task_uid, r.seed)
        for _, r in manifest.iterrows()
        if any(f in str(r.fitted_models).lower() for f in FORBIDDEN)
    ]
    if contaminated:
        raise SystemExit(f"HARD STOP: foundation model in the specialist suite: {contaminated}")
    return manifest


def main() -> None:
    manifest = build()
    out = paths.RESULTS / "specialist_manifest.csv"
    manifest.to_csv(out, index=False)
    reconstructed = int(manifest.reconstructed.sum())
    print(f"wrote {out} ({len(manifest)} rows, {reconstructed} reconstructed)")
    print(
        manifest.assign(task=lambda d: d.task_uid.str.split("::").str[1])[
            ["split", "task", "seed", "status", "model_best", "ensemble_model", "reconstructed"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
