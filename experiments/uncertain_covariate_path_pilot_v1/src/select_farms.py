"""Farm selection for UCP-PATH-PILOT-v1 (instruction section 7).

Eight sites, four onshore and four offshore, ranked on 2019 information only,
before any 2020 or 2021 result is looked at.

Eligibility is stated in usable origins rather than raw 30-minute coverage. An
origin is usable for a farm when all twelve requested future targets (6..72 h)
are present; that is the rule the main evaluation uses, so it is the honest way
to ask whether a farm has enough examples. Raw coverage overstates this badly:
GB curtailment blanks the metered series in clustered runs, and in 2021 the
median farm keeps only 21% of its origins even though its 30-minute coverage
reads 75%.

The 2020 and 2021 minimums are an availability gate. They read whether data
exists, never how any model scores on it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
BMRA = ROOT / "data_external/ucp_path_pilot_v1/extracted/bmra/BMRA_Data/standardFormat"
RESULTS = ROOT / "results/uncertain_covariate_path_pilot_v1"

# ENS grid read from the archive on 2026-09-06: 0.5 deg, lat 50.0..59.5, lon -8.0..3.5
GRID = {"lat_min": 50.0, "lat_max": 59.5, "lon_min": -8.0, "lon_max": 3.5, "res": 0.5}
LEADS_H = tuple(range(6, 73, 6))
SPLIT_YEARS = {"train": 2019, "val": 2020, "test": 2021}
MIN_ORIGINS = {"train": 600, "val": 200, "test": 200}
N_PER_GROUP = 4
GROUPS = ("Onshore wind", "Offshore wind")


def great_circle_km(a, b):
    lat1, lon1, lat2, lon2 = map(np.radians, (a[0], a[1], b[0], b[1]))
    d = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return float(6371.0 * 2 * np.arcsin(np.sqrt(d)))


def inside_grid(lat, lon):
    return GRID["lat_min"] <= lat <= GRID["lat_max"] and GRID["lon_min"] <= lon <= GRID["lon_max"]


def origins_for(year):
    """NWP base times: 00 and 12 UTC, read from the ENS filenames."""
    return pd.date_range(f"{year}-01-01 00:00", f"{year}-12-31 12:00", freq="12h")


def usable_origin_counts(power):
    """Per farm and year, how many origins have all twelve future targets present."""
    counts = {}
    for year in SPLIT_YEARS.values():
        origins = origins_for(year)
        present = None
        for lead in LEADS_H:
            block = power.reindex(origins + pd.Timedelta(hours=lead)).notna().to_numpy()
            present = block if present is None else (present & block)
        counts[year] = pd.Series(present.sum(axis=0), index=power.columns)
    return counts


def relabel_self_contradicting_types(meta):
    """Fix rows whose own Name says Offshore while Type says Onshore wind.

    Only internal contradictions in the source are corrected, so the rule needs
    no outside knowledge and stays reproducible. On the 2026-09-06 snapshot this
    is one row, E_BURBO 'Burbo Bank Offshore Windfarm'.
    """
    contradiction = meta["Name"].str.contains("Offshore", case=False, na=False) & (
        meta["Type"] == "Onshore wind"
    )
    meta = meta.copy()
    meta.loc[contradiction, "Type"] = "Offshore wind"
    return meta, sorted(meta.index[contradiction])


def build_table():
    power = pd.read_parquet(BMRA / "BMRA_data.parquet")
    meta = pd.read_parquet(BMRA / "BMRA_metadata.parquet").set_index("BMU ID")
    meta, relabelled = relabel_self_contradicting_types(meta)
    build_table.relabelled = relabelled
    counts = usable_origin_counts(power)

    rows = []
    for farm in power.columns:
        if farm not in meta.index:
            continue
        m = meta.loc[farm]
        lat, lon = m["lat"], m["lon"]
        has_loc = bool(pd.notna(lat) and pd.notna(lon))
        rows.append(
            {
                "farm_id": farm,
                "name": m["Name"],
                "type": m["Type"],
                "lat": lat,
                "lon": lon,
                "usable_origins_2019": int(counts[2019][farm]),
                "usable_origins_2020": int(counts[2020][farm]),
                "usable_origins_2021": int(counts[2021][farm]),
                "coverage_2019_30min": float(
                    power.loc["2019-01-01":"2019-12-31 23:30:00", farm].notna().mean()
                ),
                "has_location": has_loc,
                "inside_ens_grid": bool(has_loc and inside_grid(lat, lon)),
                "history_before_2019": bool(
                    pd.notna(m["startDate"]) and m["startDate"] <= pd.Timestamp("2018-12-01")
                ),
            }
        )
    df = pd.DataFrame(rows)
    df["eligible"] = (
        (df["usable_origins_2019"] >= MIN_ORIGINS["train"])
        & (df["usable_origins_2020"] >= MIN_ORIGINS["val"])
        & (df["usable_origins_2021"] >= MIN_ORIGINS["test"])
        & df["has_location"]
        & df["inside_ens_grid"]
        & df["history_before_2019"]
        & df["type"].isin(GROUPS)
    )
    return df


def farthest_point_pick(pool, n):
    """Seed on the best 2019 record, then repeatedly take the farm farthest from the set."""
    pool = pool.sort_values(
        ["usable_origins_2019", "farm_id"], ascending=[False, True]
    ).reset_index(drop=True)
    coords = {r["farm_id"]: (r["lat"], r["lon"]) for _, r in pool.iterrows()}
    chosen = [pool.iloc[0]["farm_id"]]
    while len(chosen) < n:
        best, best_key = None, None
        for _, r in pool.iterrows():
            fid = r["farm_id"]
            if fid in chosen:
                continue
            min_dist = min(great_circle_km(coords[fid], coords[c]) for c in chosen)
            key = (round(min_dist, 3), r["usable_origins_2019"], [-ord(ch) for ch in fid])
            if best_key is None or key > best_key:
                best, best_key = fid, key
        chosen.append(best)
    return chosen


def main():
    table = build_table()
    RESULTS.mkdir(parents=True, exist_ok=True)
    table.to_csv(RESULTS / "farm_eligibility_table.csv", index=False)

    selection, shortfall = {}, {}
    for group in GROUPS:
        pool = table[table["eligible"] & (table["type"] == group)]
        print(f"{group}: {len(pool)} eligible of {(table['type'] == group).sum()}", flush=True)
        take = min(N_PER_GROUP, len(pool))
        if take < N_PER_GROUP:
            shortfall[group] = {"eligible": int(len(pool)), "wanted": N_PER_GROUP}
        selection[group] = farthest_point_pick(pool, take)

    picked = [f for g in GROUPS for f in selection[g]]
    out = table[table["farm_id"].isin(picked)].copy()
    out["group_rank"] = [selection[r["type"]].index(r["farm_id"]) for _, r in out.iterrows()]
    out = out.sort_values(["type", "group_rank"])
    out.to_csv(RESULTS / "farm_selection.csv", index=False)

    coords = {r["farm_id"]: (r["lat"], r["lon"]) for _, r in out.iterrows()}
    pairwise = [
        great_circle_km(coords[a], coords[b]) for i, a in enumerate(picked) for b in picked[i + 1 :]
    ]
    rule = {
        "ranking_information": "2019 usable-origin count only; no 2020 or 2021 result was inspected",
        "origin_definition": "NWP base times 00 and 12 UTC; leads 6..72 h in 6 h steps (T=12)",
        "usable_origin_definition": "all twelve future targets present in BMRA_data.parquet",
        "eligibility": {
            "min_usable_origins": MIN_ORIGINS,
            "location_metadata_required": True,
            "inside_ens_grid": GRID,
            "acquisition_start_before": "2018-12-01, so a 28-day history exists for early 2019 origins",
        },
        "why_origins_not_coverage": (
            "Curtailment blanks the metered series in clustered runs, so 30-minute coverage "
            "overstates usable examples. In 2021 the median farm reads 75% coverage but keeps "
            "only 21% of its origins under the all-twelve-leads rule."
        ),
        "deviation_from_prereg": (
            "Instruction section 7 states the eligibility threshold as 2019 target coverage >= 90%. "
            "It is applied here as 2019 usable origins >= 600 of 730, the same idea expressed in "
            "the unit the evaluation actually uses. The 2020 and 2021 minimums are added because "
            "the literal 2019-only reading selects farms whose metering stops on 2020-12-31 and "
            "which therefore contribute zero test examples. Both gates read data availability, "
            "never a score."
        ),
        "balance": f"{N_PER_GROUP} onshore + {N_PER_GROUP} offshore",
        "balance_shortfall": shortfall,
        "within_group_rule": (
            "seed = most 2019 usable origins (ties by farm ID ascending); then farthest-point "
            "selection maximising the minimum great-circle distance to the already chosen farms, "
            "ties by 2019 usable origins then farm ID ascending"
        ),
        "n_eligible": {g: int((table["eligible"] & (table["type"] == g)).sum()) for g in GROUPS},
        "selected": selection,
        "min_pairwise_km": round(min(pairwise), 1),
        "max_pairwise_km": round(max(pairwise), 1),
        "source_type_relabelled": {
            "farms": build_table.relabelled,
            "reason": (
                "the source generator_list.csv contradicts itself: the Name field reads "
                "'Offshore Windfarm' while the Type field reads 'Onshore wind'. Only rows with "
                "that internal contradiction are corrected, so no outside knowledge is used."
            ),
        },
        "naming": "eight wind farm sites, not eight datasets",
    }
    (RESULTS / "farm_selection_rule.json").write_text(json.dumps(rule, indent=2))

    cols = ["farm_id", "name", "type", "lat", "lon",
            "usable_origins_2019", "usable_origins_2020", "usable_origins_2021"]
    print(out[cols].to_string(index=False))
    print(f"pairwise distance {rule['min_pairwise_km']}..{rule['max_pairwise_km']} km")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
