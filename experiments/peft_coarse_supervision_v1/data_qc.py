from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_SITES = ("Eagle", "Lamb")
DEFAULT_EXPECTED_COUNTS = {"Eagle": 40, "Lamb": 17}
DEFAULT_EXPECTED_TIMEZONES = {"Eagle": "US/Eastern", "Lamb": "Europe/London"}
DEFAULT_START = pd.Timestamp("2016-01-01 00:00:00")
DEFAULT_END = pd.Timestamp("2017-12-31 23:00:00")
EXPECTED_2016_ROWS = 8784
TIMESTAMP_COLUMN = "timestamp"
MONTHS_2016 = [f"2016-{month:02d}" for month in range(1, 13)]


def compute_qc(
    metadata_path: str | Path,
    raw_path: str | Path,
    *,
    expected_counts: Mapping[str, int] | None = None,
    expected_timezones: Mapping[str, str] | None = None,
    chunksize: int = 1024,
) -> dict[str, Any]:
    """Compute train-only entry QC for the fixed BDG2 Eagle/Lamb office panel."""

    metadata_path = Path(metadata_path)
    raw_path = Path(raw_path)
    expected_counts = dict(DEFAULT_EXPECTED_COUNTS if expected_counts is None else expected_counts)
    expected_timezones = dict(
        DEFAULT_EXPECTED_TIMEZONES if expected_timezones is None else expected_timezones
    )

    metadata = pd.read_csv(metadata_path, dtype=str, keep_default_na=False)
    site_info = _metadata_site_info(metadata, expected_counts, expected_timezones)
    candidate_ids = [
        building_id
        for site in DEFAULT_SITES
        for building_id in site_info[site]["candidate_ids"]
    ]
    stats = _empty_candidate_stats(site_info)
    timestamp_chunks: list[pd.Series] = []

    usecols = [TIMESTAMP_COLUMN, *candidate_ids]
    for chunk in pd.read_csv(raw_path, usecols=usecols, chunksize=chunksize):
        timestamps = pd.to_datetime(
            chunk[TIMESTAMP_COLUMN],
            format="%Y-%m-%d %H:%M:%S",
            errors="coerce",
        )
        timestamp_chunks.append(timestamps)
        in_2016 = (timestamps >= DEFAULT_START) & (
            timestamps <= pd.Timestamp("2016-12-31 23:00:00")
        )
        if not bool(in_2016.any()):
            continue
        values = chunk.loc[in_2016, candidate_ids].apply(pd.to_numeric, errors="coerce")
        train_timestamps = timestamps.loc[in_2016].reset_index(drop=True)
        _accumulate_train_stats(stats, values.reset_index(drop=True), train_timestamps)

    if not timestamp_chunks:
        raise ValueError("timestamp grid mismatch: raw file has no rows")

    all_timestamps = pd.concat(timestamp_chunks, ignore_index=True)
    timestamp_qc = _timestamp_qc(all_timestamps)
    if not timestamp_qc["matches_expected_grid"]:
        raise ValueError(f"timestamp grid mismatch: {timestamp_qc}")

    _finalize_candidate_stats(stats, EXPECTED_2016_ROWS)
    selection = _select_sites(site_info, stats)
    selected_target_monthly = _selected_target_monthly(selection, stats)

    decision = (
        "PASS"
        if all(site["n"] >= 4 for site in selection["sites"].values())
        else "INSUFFICIENT_COMPLETE_TRAIN_METERS"
    )

    return {
        "completed": True,
        "decision": decision,
        "domain": "nonresidential hourly electricity meter panel",
        "downstream_decision": (
            "train-only completeness gate before coarse-supervision PEFT data entry"
        ),
        "leakage_contract": {
            "metadata_used_for_candidate_pool": True,
            "energy_values_used_for_selection": "2016 only",
            "energy_values_not_analyzed_for_selection": "2017",
            "cleaned_meter_file_used": False,
            "timestamp_interpretation": "local-naive hourly grid; UTC/DST support unresolved",
            "no_imputation": True,
            "no_threshold_lowering_after_qc": True,
        },
        "inputs": {
            "metadata_path": str(metadata_path),
            "raw_path": str(raw_path),
        },
        "metadata": {
            "sites": site_info,
            "expected_counts": expected_counts,
            "expected_timezones": expected_timezones,
        },
        "timestamp_qc": timestamp_qc,
        "candidate_train_2016_qc": stats,
        "selection": selection,
        "selected_target_monthly_2016": selected_target_monthly,
    }


def _metadata_site_info(
    metadata: pd.DataFrame,
    expected_counts: Mapping[str, int],
    expected_timezones: Mapping[str, str],
) -> dict[str, dict[str, Any]]:
    required = {
        "building_id",
        "site_id",
        "primaryspaceusage",
        "timezone",
        "electricity",
    }
    missing = sorted(required.difference(metadata.columns))
    if missing:
        raise ValueError(f"metadata missing required columns: {missing}")

    site_info: dict[str, dict[str, Any]] = {}
    for site in DEFAULT_SITES:
        filtered = metadata[
            (metadata["site_id"] == site)
            & (metadata["primaryspaceusage"] == "Office")
            & (metadata["electricity"] == "Yes")
        ].copy()
        candidate_ids = sorted(filtered["building_id"].tolist())
        actual_count = len(candidate_ids)
        if site in expected_counts and actual_count != expected_counts[site]:
            raise ValueError(
                f"metadata count mismatch for {site}: expected {expected_counts[site]}, "
                f"found {actual_count}"
            )
        timezones = sorted(set(filtered["timezone"].tolist()))
        if site in expected_timezones and timezones != [expected_timezones[site]]:
            raise ValueError(
                f"metadata timezone mismatch for {site}: expected {expected_timezones[site]}, "
                f"found {timezones}"
            )
        site_info[site] = {
            "site_id": site,
            "candidate_count": actual_count,
            "candidate_ids": candidate_ids,
            "timezones": timezones,
            "filter": {
                "primaryspaceusage": "Office",
                "electricity": "Yes",
            },
        }
    return site_info


def _empty_candidate_stats(site_info: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = {}
    for site, info in site_info.items():
        for building_id in info["candidate_ids"]:
            stats[building_id] = {
                "site_id": site,
                "finite_count": 0,
                "missing_count": None,
                "negative_count": 0,
                "zero_count": 0,
                "annual_sum": 0.0,
                "eligible": False,
                "monthly_sum": [0.0 for _ in MONTHS_2016],
                "monthly_count": [0 for _ in MONTHS_2016],
            }
    return stats


def _accumulate_train_stats(
    stats: dict[str, dict[str, Any]],
    values: pd.DataFrame,
    timestamps: pd.Series,
) -> None:
    month_indices = timestamps.dt.month.to_numpy() - 1
    for building_id in values.columns:
        arr = values[building_id].to_numpy(dtype=float)
        finite = np.isfinite(arr)
        finite_values = arr[finite]
        stats[building_id]["finite_count"] += int(finite.sum())
        stats[building_id]["negative_count"] += int((finite_values < 0).sum())
        stats[building_id]["zero_count"] += int((finite_values == 0).sum())
        stats[building_id]["annual_sum"] += float(finite_values.sum())
        for month_index in range(12):
            month_mask = finite & (month_indices == month_index)
            if not bool(month_mask.any()):
                continue
            month_values = arr[month_mask]
            stats[building_id]["monthly_count"][month_index] += int(month_mask.sum())
            stats[building_id]["monthly_sum"][month_index] += float(month_values.sum())


def _finalize_candidate_stats(
    stats: dict[str, dict[str, Any]],
    expected_train_rows: int,
) -> None:
    for candidate in stats.values():
        finite_count = int(candidate["finite_count"])
        missing_count = expected_train_rows - finite_count
        candidate["missing_count"] = int(missing_count)
        candidate["eligible"] = (
            finite_count == expected_train_rows
            and missing_count == 0
            and int(candidate["negative_count"]) == 0
            and float(candidate["annual_sum"]) > 0.0
        )


def _select_sites(
    site_info: Mapping[str, Mapping[str, Any]],
    stats: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    sites: dict[str, Any] = {}
    total_donors = 0
    total_targets = 0
    for site in DEFAULT_SITES:
        candidate_ids = site_info[site]["candidate_ids"]
        eligible_ids = [
            building_id for building_id in candidate_ids if bool(stats[building_id]["eligible"])
        ]
        n = min(8, len(eligible_ids) // 2)
        donor_ids = eligible_ids[:n]
        target_ids = eligible_ids[n : 2 * n]
        total_donors += len(donor_ids)
        total_targets += len(target_ids)
        sites[site] = {
            "site_id": site,
            "eligible_count": len(eligible_ids),
            "eligible_ids": eligible_ids,
            "n": n,
            "decision": "PASS" if n >= 4 else "INSUFFICIENT_COMPLETE_TRAIN_METERS",
            "donor_ids": donor_ids,
            "target_ids": target_ids,
            "selection_rule": "sorted eligible IDs; donor first n, target next n; n=min(8,eligible//2)",
        }
    return {
        "sites": sites,
        "total_donor_count": total_donors,
        "total_target_count": total_targets,
    }


def _selected_target_monthly(
    selection: Mapping[str, Any],
    stats: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    out: dict[str, Any] = {"months": MONTHS_2016, "sites": {}}
    for site, site_selection in selection["sites"].items():
        target_ids = list(site_selection["target_ids"])
        totals_by_month: list[list[float]] = []
        counts_by_month: list[list[int]] = []
        for month_index in range(12):
            totals_by_month.append(
                [float(stats[building_id]["monthly_sum"][month_index]) for building_id in target_ids]
            )
            counts_by_month.append(
                [int(stats[building_id]["monthly_count"][month_index]) for building_id in target_ids]
            )
        out["sites"][site] = {
            "target_ids": target_ids,
            "totals": totals_by_month,
            "counts": counts_by_month,
        }
    return out


def _timestamp_qc(timestamps: pd.Series) -> dict[str, Any]:
    invalid_count = int(timestamps.isna().sum())
    expected = pd.date_range(DEFAULT_START, DEFAULT_END, freq="h")
    duplicate_count = int(timestamps.duplicated().sum())
    diffs = timestamps.diff().dropna()
    one_hour = pd.Timedelta(hours=1)
    offstep_count = int((diffs != one_hour).sum())
    matches_expected_grid = (
        invalid_count == 0
        and len(timestamps) == len(expected)
        and duplicate_count == 0
        and offstep_count == 0
        and bool(timestamps.reset_index(drop=True).equals(pd.Series(expected)))
    )
    in_2016 = (timestamps >= DEFAULT_START) & (
        timestamps <= pd.Timestamp("2016-12-31 23:00:00")
    )
    in_2017 = (timestamps >= pd.Timestamp("2017-01-01 00:00:00")) & (
        timestamps <= DEFAULT_END
    )
    return {
        "expected_start": str(DEFAULT_START),
        "expected_end": str(DEFAULT_END),
        "expected_total_rows": int(len(expected)),
        "total_rows": int(len(timestamps)),
        "actual_start": None if len(timestamps) == 0 else str(timestamps.iloc[0]),
        "actual_end": None if len(timestamps) == 0 else str(timestamps.iloc[-1]),
        "invalid_timestamp_count": invalid_count,
        "duplicate_count": duplicate_count,
        "offstep_count": offstep_count,
        "rows_2016": int(in_2016.sum()),
        "expected_rows_2016": EXPECTED_2016_ROWS,
        "rows_2017": int(in_2017.sum()),
        "matches_expected_grid": bool(matches_expected_grid),
    }
