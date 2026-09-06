"""Join the ENS weather tensors with BMRA targets into per-split example panels.

One example is one (farm, origin) pair. It carries

  history   1344 half-hourly target values strictly before the origin
  weather   [K, T, D] with member identity intact across leads
  target    T future metered values at origin + 6..72 h
  calendar  known-at-origin time features, identical for every arm

Every arm sees exactly the same examples in the same order, so the evaluation
keys cannot differ between arms (test A03).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data_external/ucp_path_pilot_v1"
BMRA = DATA / "extracted/bmra/BMRA_Data/standardFormat"
PROCESSED = DATA / "processed"
RESULTS = ROOT / "results/uncertain_covariate_path_pilot_v1"

LEADS_H = tuple(range(6, 73, 6))
HISTORY_STEPS = 28 * 48  # 1344 half-hourly points
SPLITS = {"train": 2019, "val": 2020, "test": 2021}
ARCHIVE_FOR_YEAR = {2019: "ens_2019_2020", 2020: "ens_2019_2020", 2021: "ens_2021"}

# A dry run needs three non-overlapping windows before 2020 and 2021 exist on disk.
# It writes to the same panel files on purpose: the real build overwrites them.
DRYRUN_WINDOWS = {
    "train": ("2019-01-01", "2019-04-30"),
    "val": ("2019-05-01", "2019-08-31"),
    "test": ("2019-09-01", "2019-12-31"),
}
MAX_HISTORY_MISSING = 0.10


def load_weather():
    """origin -> [n_farms, K, T, D], plus the farm order the tensors use."""
    blocks, farms = {}, None
    for key in sorted(set(ARCHIVE_FOR_YEAR.values())):
        path = PROCESSED / f"weather_{key}.npz"
        if not path.exists():
            continue
        d = np.load(path, allow_pickle=False)
        order = [str(f) for f in d["farms"]]
        if farms is None:
            farms = order
        elif farms != order:
            raise RuntimeError("farm order differs between weather archives")
        for t, block in zip(pd.to_datetime(d["origins"]), d["weather"]):
            blocks[pd.Timestamp(t)] = block
    if farms is None:
        raise RuntimeError("no weather archive processed yet")
    return blocks, farms


def calendar_features(valid_times: pd.DatetimeIndex) -> np.ndarray:
    """(T, 4): hour-of-day and day-of-year on the unit circle."""
    hour = valid_times.hour + valid_times.minute / 60.0
    doy = valid_times.dayofyear.to_numpy(dtype=np.float64)
    return np.stack(
        [
            np.sin(2 * np.pi * hour / 24.0),
            np.cos(2 * np.pi * hour / 24.0),
            np.sin(2 * np.pi * doy / 365.25),
            np.cos(2 * np.pi * doy / 365.25),
        ],
        axis=-1,
    ).astype(np.float32)


def build(max_origins_per_split: int | None = None, windows: dict | None = None) -> dict:
    power = pd.read_parquet(BMRA / "BMRA_data.parquet")
    farm_table = pd.read_csv(RESULTS / "farm_selection.csv")
    weather_blocks, weather_farms = load_weather()
    if list(farm_table["farm_id"]) != weather_farms:
        raise RuntimeError("farm_selection.csv order differs from the extracted tensors")

    step = pd.Timedelta(minutes=30)
    lead_deltas = [pd.Timedelta(hours=h) for h in LEADS_H]
    index_pos = pd.Series(np.arange(len(power.index)), index=power.index)

    panels, dropped = {}, {}
    for split, year in SPLITS.items():
        if windows:
            lo, hi = (pd.Timestamp(x) for x in windows[split])
            origins = sorted(t for t in weather_blocks if lo <= t <= hi)
            split_end = hi.normalize() + pd.Timedelta(hours=23, minutes=30)
        else:
            origins = sorted(t for t in weather_blocks if t.year == year)
            split_end = pd.Timestamp(f"{year}-12-31 23:30:00")
        if max_origins_per_split:
            origins = origins[:max_origins_per_split]

        rows = {k: [] for k in ("history", "weather", "target", "calendar", "farm_idx",
                                "origin", "example_id", "lead_h")}
        reasons = {"future_crosses_split": 0, "target_missing": 0, "history_short": 0,
                   "history_missing": 0}

        for origin in origins:
            valid_times = pd.DatetimeIndex([origin + d for d in lead_deltas])
            if valid_times[-1] > split_end:
                reasons["future_crosses_split"] += len(farm_table)
                continue
            if origin not in index_pos.index:
                reasons["history_short"] += len(farm_table)
                continue
            end = int(index_pos.loc[origin])  # first index at or after the origin
            start = end - HISTORY_STEPS
            if start < 0:
                reasons["history_short"] += len(farm_table)
                continue
            hist_slice = power.iloc[start:end]
            if hist_slice.index[-1] >= origin:  # strict: nothing at or after the origin
                raise RuntimeError(f"history window touches the origin at {origin}")
            cal = calendar_features(valid_times)
            block = weather_blocks[origin]

            for f, farm in enumerate(farm_table["farm_id"]):
                target = power.loc[valid_times, farm].to_numpy(dtype=np.float32)
                if not np.isfinite(target).all():
                    reasons["target_missing"] += 1
                    continue
                history = hist_slice[farm].to_numpy(dtype=np.float32)
                if np.isnan(history).mean() > MAX_HISTORY_MISSING:
                    reasons["history_missing"] += 1
                    continue
                rows["history"].append(history)
                rows["weather"].append(block[f])
                rows["target"].append(target)
                rows["calendar"].append(cal)
                rows["farm_idx"].append(f)
                rows["origin"].append(str(origin))
                rows["example_id"].append(f"{farm}|{origin.isoformat()}")

        panels[split] = {
            "history": np.asarray(rows["history"], dtype=np.float32),
            "weather": np.asarray(rows["weather"], dtype=np.float32),
            "target": np.asarray(rows["target"], dtype=np.float32),
            "calendar": np.asarray(rows["calendar"], dtype=np.float32),
            "farm_idx": np.asarray(rows["farm_idx"], dtype=np.int64),
            "origin": np.asarray(rows["origin"]),
            "example_id": np.asarray(rows["example_id"]),
        }
        dropped[split] = reasons
        print(f"[{split}] {len(rows['example_id'])} examples from {len(origins)} origins "
              f"| dropped {reasons}", flush=True)

    return {"panels": panels, "dropped": dropped, "farms": list(farm_table["farm_id"]),
            "windows": windows}


def fit_normalisation(train: dict, n_farms: int) -> dict:
    """Target scaling is per farm on 2019; weather scaling is global on 2019."""
    target_mean = np.zeros(n_farms, np.float32)
    target_std = np.ones(n_farms, np.float32)
    for f in range(n_farms):
        sel = train["farm_idx"] == f
        if sel.sum() == 0:
            raise RuntimeError(f"farm index {f} has no training example")
        values = train["target"][sel].ravel()
        target_mean[f] = values.mean()
        target_std[f] = max(float(values.std()), 1e-6)
    weather_mean = train["weather"].reshape(-1, train["weather"].shape[-1]).mean(0)
    weather_std = train["weather"].reshape(-1, train["weather"].shape[-1]).std(0)
    return {
        "target_mean": target_mean,
        "target_std": target_std,
        "weather_mean": weather_mean.astype(np.float32),
        "weather_std": np.maximum(weather_std, 1e-6).astype(np.float32),
    }


def main(max_origins_per_split: int | None = None, windows: dict | None = None) -> int:
    built = build(max_origins_per_split, windows)
    panels, farms = built["panels"], built["farms"]
    norm = fit_normalisation(panels["train"], len(farms))

    PROCESSED.mkdir(parents=True, exist_ok=True)
    for split, panel in panels.items():
        np.savez_compressed(PROCESSED / f"panel_{split}.npz", **panel)
    np.savez(PROCESSED / "normalisation.npz", **norm)

    summary = {
        "leads_h": list(LEADS_H),
        "history_steps": HISTORY_STEPS,
        "history_rule": "strictly before the origin; max 10% missing",
        "target": "BMRA_data.parquet metered power in MW, exact timestamp join",
        "splits": {
            split: {
    "year": SPLITS[split] if not built.get("windows") else built["windows"][split],
                "n_examples": int(len(panel["example_id"])),
                "n_origins": int(len(set(panel["origin"].tolist()))),
                "per_farm": {
                    farms[f]: int((panel["farm_idx"] == f).sum()) for f in range(len(farms))
                },
                "dropped": built["dropped"][split],
            }
            for split, panel in panels.items()
        },
        "normalisation": {
            "target": "per farm, mean and std of the 2019 training targets",
            "weather": "global per feature, 2019 training only",
            "target_mean_mw": {farms[f]: float(norm["target_mean"][f]) for f in range(len(farms))},
            "target_std_mw": {farms[f]: float(norm["target_std"][f]) for f in range(len(farms))},
        },
    }
    (RESULTS / "panel_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["splits"], indent=2))
    return 0


if __name__ == "__main__":
    import sys

    dryrun = "--dryrun" in sys.argv
    raise SystemExit(main(None, DRYRUN_WINDOWS if dryrun else None))
