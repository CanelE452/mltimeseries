#!/usr/bin/env python
"""Reproduce the target-blind fresh Stage A candidate manifest checks.

Default mode verifies the existing JSON manifest without overwriting it.
The script is intentionally CPU-only and uses streaming CSV reads. It does
not compute forecast losses, run fits, access GPUs, or write manifests.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import OrderedDict, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


CONTEXT_HOURS = 336
HORIZON_HOURS = 48
ORIGIN_STRIDE_HOURS = 24
MIN_FINITE_FRACTION = 0.70
HOUSEHOLD_BLOCK_START = datetime(2009, 4, 29)
BDG2_BLOCK_START = datetime(2016, 1, 1)
HOUSEHOLD_CHANNELS = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
]
HOUSEHOLD_TARGETS = HOUSEHOLD_CHANNELS[:2]
EXCLUDED_BDG2_SITES = {"Eagle", "Lamb"}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S")


def parse_iso(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%S")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_splits(block_start: datetime) -> OrderedDict[str, tuple[datetime, datetime]]:
    day = timedelta(days=1)
    splits: OrderedDict[str, tuple[datetime, datetime]] = OrderedDict()
    cur = block_start
    splits["precontext"] = (cur, cur + 14 * day)
    cur = splits["precontext"][1]
    splits["train"] = (cur, cur + 91 * day)
    cur = splits["train"][1]
    splits["embargo_train_val"] = (cur, cur + 2 * day)
    cur = splits["embargo_train_val"][1]
    splits["val"] = (cur, cur + 31 * day)
    cur = splits["val"][1]
    splits["embargo_val_e1"] = (cur, cur + 2 * day)
    cur = splits["embargo_val_e1"][1]
    splits["e1"] = (cur, cur + 41 * day)
    cur = splits["e1"][1]
    splits["e2"] = (cur, cur + 41 * day)
    return splits


def split_boundaries(splits: OrderedDict[str, tuple[datetime, datetime]]) -> dict[str, list[str]]:
    return {name: [iso(start), iso(end)] for name, (start, end) in splits.items()}


def origin_list(start: datetime, end: datetime) -> list[datetime]:
    origins: list[datetime] = []
    cur = start
    last = end - timedelta(hours=HORIZON_HOURS)
    while cur <= last:
        origins.append(cur)
        cur += timedelta(hours=ORIGIN_STRIDE_HOURS)
    return origins


def split_origin_counts(splits: OrderedDict[str, tuple[datetime, datetime]]) -> dict[str, int]:
    return {
        name: len(origin_list(start, end))
        for name, (start, end) in splits.items()
        if name in {"train", "val", "e1", "e2"}
    }


def hourly_grid(start: datetime, end: datetime) -> list[datetime]:
    times: list[datetime] = []
    cur = start
    while cur < end:
        times.append(cur)
        cur += timedelta(hours=1)
    return times


def build_prefix(series_by_col: dict[str, list[float]]) -> dict[str, list[int]]:
    prefixes: dict[str, list[int]] = {}
    for col, values in series_by_col.items():
        prefix = [0]
        count = 0
        for value in values:
            if math.isfinite(value):
                count += 1
            prefix.append(count)
        prefixes[col] = prefix
    return prefixes


def finite_count(prefix: list[int], start_idx: int, end_idx: int) -> int:
    return prefix[end_idx] - prefix[start_idx]


def interval_has_variation(values: list[float], start_idx: int, end_idx: int) -> bool:
    first: float | None = None
    for value in values[start_idx:end_idx]:
        if not math.isfinite(value):
            continue
        if first is None:
            first = value
        elif value != first:
            return True
    return False


def evaluate_block(
    series_by_col: dict[str, list[float]],
    grid_start: datetime,
    splits: OrderedDict[str, tuple[datetime, datetime]],
    targets: list[str],
    channels: list[str],
) -> dict[str, Any]:
    prefixes = build_prefix(series_by_col)

    def idx(dt: datetime) -> int:
        return int((dt - grid_start).total_seconds() // 3600)

    row_qc: dict[str, dict[str, dict[str, Any]]] = {}
    target_qc: dict[str, dict[str, dict[str, Any]]] = {}
    min_origin: dict[str, float] = {}

    for split_name, (start, end) in splits.items():
        start_idx = idx(start)
        end_idx = idx(end)
        row_qc[split_name] = {}
        for col in channels:
            total = end_idx - start_idx
            finite = finite_count(prefixes[col], start_idx, end_idx)
            row_qc[split_name][col] = {
                "finite": finite,
                "total": total,
                "finite_fraction": finite / total if total else 0.0,
                "nonconstant": interval_has_variation(series_by_col[col], start_idx, end_idx),
            }
        if split_name not in {"train", "val", "e1", "e2"}:
            continue
        target_qc[split_name] = {}
        split_mins: list[float] = []
        for col in targets:
            valid = 0
            total = 0
            per_origin: list[float] = []
            for origin in origin_list(start, end):
                origin_idx = idx(origin)
                origin_end = origin_idx + HORIZON_HOURS
                finite = finite_count(prefixes[col], origin_idx, origin_end)
                valid += finite
                total += HORIZON_HOURS
                per_origin.append(finite / HORIZON_HOURS)
            target_qc[split_name][col] = {
                "valid": valid,
                "total": total,
                "finite_fraction": valid / total if total else 0.0,
                "min_origin_window_finite_fraction": min(per_origin) if per_origin else 0.0,
                "pass_min_70pct": (valid / total) >= MIN_FINITE_FRACTION if total else False,
                "pass_each_origin_70pct": min(per_origin) >= MIN_FINITE_FRACTION if per_origin else False,
            }
            split_mins.append(target_qc[split_name][col]["min_origin_window_finite_fraction"])
        min_origin[split_name] = min(split_mins) if split_mins else 0.0

    aggregate_target = {
        split: min(target_qc[split][target]["finite_fraction"] for target in targets)
        for split in ("train", "val", "e1", "e2")
    }
    row_min = {
        split: min(row_qc[split][col]["finite_fraction"] for col in channels)
        for split in row_qc
    }
    all_target_windows_pass = all(
        target_qc[split][target]["pass_min_70pct"]
        and target_qc[split][target]["pass_each_origin_70pct"]
        for split in ("train", "val", "e1", "e2")
        for target in targets
    )
    all_rows_pass = all(
        row_qc[split][col]["finite_fraction"] >= MIN_FINITE_FRACTION
        and row_qc[split][col]["nonconstant"]
        for split in row_qc
        for col in channels
    )
    return {
        "target_qc": target_qc,
        "row_qc": row_qc,
        "aggregate_target_finite_fraction": aggregate_target,
        "row_finite_fraction_min_by_split": row_min,
        "min_origin_window_finite_fraction": min_origin,
        "pass_targets": all_target_windows_pass,
        "pass_rows": all_rows_pass,
    }


def read_household_hourly(path: Path, scan_start: datetime) -> dict[str, Any]:
    accum: dict[str, Any] = {
        col: defaultdict(lambda: [0.0, 0]) for col in HOUSEHOLD_CHANNELS
    }
    source_rows = 0
    parse_failures = 0
    first_ts: datetime | None = None
    last_ts: datetime | None = None

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        for row in reader:
            source_rows += 1
            try:
                timestamp = datetime.strptime(
                    f"{row['Date']} {row['Time']}", "%d/%m/%Y %H:%M:%S"
                )
            except ValueError:
                parse_failures += 1
                continue
            if first_ts is None:
                first_ts = timestamp
            last_ts = timestamp
            if timestamp < scan_start:
                continue
            hour = timestamp.replace(minute=0, second=0)
            for col in HOUSEHOLD_CHANNELS:
                raw = row[col]
                if not raw or raw == "?":
                    continue
                try:
                    value = float(raw)
                except ValueError:
                    continue
                if math.isfinite(value):
                    accum[col][hour][0] += value
                    accum[col][hour][1] += 1

    if first_ts is None or last_ts is None:
        raise RuntimeError(f"no timestamps parsed from {path}")
    source_end = last_ts.replace(minute=0, second=0) + timedelta(hours=1)
    times = hourly_grid(scan_start, source_end)
    series_by_col: dict[str, list[float]] = {}
    hours_below_threshold: dict[str, int] = {}
    for col in HOUSEHOLD_CHANNELS:
        values: list[float] = []
        below = 0
        for hour in times:
            value_sum, finite_minutes = accum[col][hour]
            if finite_minutes >= 45:
                values.append(value_sum / finite_minutes)
            else:
                values.append(float("nan"))
                below += 1
        series_by_col[col] = values
        hours_below_threshold[col] = below

    return {
        "source_rows": source_rows,
        "parse_failures": parse_failures,
        "first_timestamp": first_ts,
        "last_timestamp": last_ts,
        "grid_start": scan_start,
        "grid_end": source_end,
        "times": times,
        "series_by_col": series_by_col,
        "hours_below_threshold": hours_below_threshold,
    }


def household_bad_target_runs(
    series_by_col: dict[str, list[float]], grid_start: datetime, grid_end: datetime
) -> list[dict[str, Any]]:
    prefixes = build_prefix({target: series_by_col[target] for target in HOUSEHOLD_TARGETS})
    bad_hours: list[datetime] = []
    total_hours = int((grid_end - grid_start).total_seconds() // 3600)
    for hour_idx in range(total_hours):
        if any(
            finite_count(prefixes[target], hour_idx, hour_idx + 1) == 0
            for target in HOUSEHOLD_TARGETS
        ):
            bad_hours.append(grid_start + timedelta(hours=hour_idx))
    if not bad_hours:
        return []

    runs: list[dict[str, Any]] = []
    start = prev = bad_hours[0]
    for hour in bad_hours[1:]:
        if hour == prev + timedelta(hours=1):
            prev = hour
            continue
        end = prev + timedelta(hours=1)
        runs.append(
            {
                "start": iso(start),
                "end_exclusive": iso(end),
                "hours": int((end - start).total_seconds() // 3600),
            }
        )
        start = prev = hour
    end = prev + timedelta(hours=1)
    runs.append(
        {
            "start": iso(start),
            "end_exclusive": iso(end),
            "hours": int((end - start).total_seconds() // 3600),
        }
    )
    return runs


def find_first_passing_household_block(household_data: dict[str, Any]) -> str | None:
    grid_start = household_data["grid_start"]
    grid_end = household_data["grid_end"]
    series_by_col = household_data["series_by_col"]
    latest_start = grid_end - timedelta(days=222)
    current = HOUSEHOLD_BLOCK_START
    while current <= latest_start:
        splits = make_splits(current)
        result = evaluate_block(
            series_by_col, grid_start, splits, HOUSEHOLD_TARGETS, HOUSEHOLD_CHANNELS
        )
        if result["pass_targets"] and result["pass_rows"]:
            return iso(current)
        current += timedelta(days=1)
    return None


def compute_household(root: Path) -> dict[str, Any]:
    source_path = root / "data/uci_household_power/household_power_consumption.txt"
    data = read_household_hourly(source_path, HOUSEHOLD_BLOCK_START)
    splits = make_splits(HOUSEHOLD_BLOCK_START)
    result = evaluate_block(
        data["series_by_col"],
        data["grid_start"],
        splits,
        HOUSEHOLD_TARGETS,
        HOUSEHOLD_CHANNELS,
    )
    first_passing = find_first_passing_household_block(data)
    block_end = splits["e2"][1]
    start_idx = int((HOUSEHOLD_BLOCK_START - data["grid_start"]).total_seconds() // 3600)
    end_idx = int((block_end - data["grid_start"]).total_seconds() // 3600)
    block_hours_below_threshold = {
        col: sum(1 for value in data["series_by_col"][col][start_idx:end_idx] if not math.isfinite(value))
        for col in HOUSEHOLD_CHANNELS
    }
    status = "READY" if result["pass_targets"] and result["pass_rows"] else "BLOCKED"

    return {
        "candidate_id": "household_post_p1_earliest_20090429",
        "status": status,
        "source_sha256": sha256_file(source_path),
        "source_rows": data["source_rows"],
        "source_timestamp_range": {
            "first": iso(data["first_timestamp"]),
            "last": iso(data["last_timestamp"]),
        },
        "block_start": iso(HOUSEHOLD_BLOCK_START),
        "block_end_exclusive": iso(block_end),
        "boundaries": split_boundaries(splits),
        "origin_counts": split_origin_counts(splits),
        "channels": HOUSEHOLD_CHANNELS,
        "target_columns": HOUSEHOLD_TARGETS,
        "aggregate_target_finite_fraction": result["aggregate_target_finite_fraction"],
        "min_origin_window_finite_fraction": result["min_origin_window_finite_fraction"],
        "row_finite_fraction_min_by_split": result["row_finite_fraction_min_by_split"],
        "hours_below_threshold_by_channel": block_hours_below_threshold,
        "post_p1_missing_target_runs_checked": household_bad_target_runs(
            data["series_by_col"], data["grid_start"], data["grid_end"]
        )[:6],
        "first_daily_shift_passing_block_start": first_passing,
        "origin_gap_needed": first_passing is None,
    }


def bdg2_office_groups(metadata_path: Path, header: list[str]) -> list[dict[str, Any]]:
    available_columns = set(header[1:])
    groups: list[dict[str, Any]] = []
    with metadata_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    site_ids = sorted(
        {
            row["site_id"]
            for row in rows
            if row.get("site_id") and row["site_id"] not in EXCLUDED_BDG2_SITES
        }
    )
    for site_id in site_ids:
        ids = [
            row["building_id"]
            for row in rows
            if row.get("site_id") == site_id
            and row.get("primaryspaceusage") == "Office"
            and row.get("electricity") == "Yes"
            and row.get("building_id") in available_columns
        ]
        if len(ids) >= 4:
            groups.append(
                {
                    "site_id": site_id,
                    "candidate_columns": ids[:4],
                    "office_meter_count": len(ids),
                }
            )
    return groups


def read_bdg2_candidate_series(
    electricity_path: Path, groups: list[dict[str, Any]], block_start: datetime, block_end: datetime
) -> dict[str, list[float]]:
    needed_cols: list[str] = []
    for group in groups:
        for col in group["candidate_columns"]:
            if col not in needed_cols:
                needed_cols.append(col)

    series_by_col: dict[str, list[float]] = {col: [] for col in needed_cols}
    expected_times = hourly_grid(block_start, block_end)
    seen_times: list[datetime] = []
    with electricity_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            timestamp = datetime.strptime(row["timestamp"], "%Y-%m-%d %H:%M:%S")
            if timestamp < block_start:
                continue
            if timestamp >= block_end:
                break
            seen_times.append(timestamp)
            for col in needed_cols:
                raw = row[col]
                if raw == "" or raw is None:
                    series_by_col[col].append(float("nan"))
                    continue
                try:
                    value = float(raw)
                except ValueError:
                    value = float("nan")
                series_by_col[col].append(value if math.isfinite(value) else float("nan"))

    if seen_times == expected_times:
        return series_by_col

    reindexed: dict[str, list[float]] = {}
    for col in needed_cols:
        by_time = dict(zip(seen_times, series_by_col[col]))
        reindexed[col] = [by_time.get(timestamp, float("nan")) for timestamp in expected_times]
    return reindexed


def compute_bdg2(root: Path) -> dict[str, Any]:
    electricity_path = root / "data_external/bdg2_coarse_supervision_v1/raw/electricity.csv"
    metadata_path = root / "data_external/bdg2_coarse_supervision_v1/raw/metadata.csv"
    with electricity_path.open(newline="", encoding="utf-8") as handle:
        header = next(csv.reader(handle))

    groups = bdg2_office_groups(metadata_path, header)
    splits = make_splits(BDG2_BLOCK_START)
    block_end = splits["e2"][1]
    series_by_col = read_bdg2_candidate_series(electricity_path, groups, BDG2_BLOCK_START, block_end)

    selection_trace: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    for group in groups:
        channels = group["candidate_columns"]
        targets = channels[:2]
        result = evaluate_block(series_by_col, BDG2_BLOCK_START, splits, targets, channels)
        status = "READY" if result["pass_targets"] and result["pass_rows"] else "BLOCKED"
        trace_item = {
            "site_id": group["site_id"],
            "candidate_columns": channels,
            "office_meter_count": group["office_meter_count"],
            "availability_result": status,
            "min_origin_window_finite_fraction": result["min_origin_window_finite_fraction"],
        }
        selection_trace.append(trace_item)
        if status == "READY":
            selected = {
                "candidate_id": "bdg2_bull_office_2016a",
                "status": "READY",
                "electricity_sha256": sha256_file(electricity_path),
                "metadata_sha256": sha256_file(metadata_path),
                "site_id": group["site_id"],
                "primaryspaceusage": "Office",
                "block_start": iso(BDG2_BLOCK_START),
                "block_end_exclusive": iso(block_end),
                "boundaries": split_boundaries(splits),
                "origin_counts": split_origin_counts(splits),
                "channels": channels,
                "target_columns": targets,
                "context_only_columns": channels[2:],
                "office_meter_count_at_site": group["office_meter_count"],
                "aggregate_target_finite_fraction": {
                    split: {
                        target: result["target_qc"][split][target]["finite_fraction"]
                        for target in targets
                    }
                    for split in ("train", "val", "e1", "e2")
                },
                "min_origin_window_finite_fraction": result[
                    "min_origin_window_finite_fraction"
                ],
                "row_finite_fraction_min_by_split": result[
                    "row_finite_fraction_min_by_split"
                ],
                "selection_trace": selection_trace,
                "origin_gap_needed": False,
            }
            break

    if selected is None:
        return {
            "candidate_id": "bdg2_bull_office_2016a",
            "status": "BLOCKED",
            "electricity_sha256": sha256_file(electricity_path),
            "metadata_sha256": sha256_file(metadata_path),
            "selection_trace": selection_trace,
        }
    return selected


def compute_summary(root: Path) -> dict[str, Any]:
    household = compute_household(root)
    bdg2 = compute_bdg2(root)
    ready = int(household["status"] == "READY") + int(bdg2["status"] == "READY")
    blocked = int(household["status"] == "BLOCKED") + int(bdg2["status"] == "BLOCKED")
    return {
        "split_design": {
            "context_hours": CONTEXT_HOURS,
            "horizon_hours": HORIZON_HOURS,
            "origin_stride_hours": ORIGIN_STRIDE_HOURS,
            "target_window_overlap_hours_within_split": 24,
            "layout": "precontext14 + train91 + gap2 + V31 + gap2 + E1_41 + E2_41 days",
            "origin_counts_expected": {"train": 90, "val": 30, "e1": 40, "e2": 40},
            "embargo_hours": {"train_to_val": 48, "val_to_e1": 48, "e1_to_e2": 0},
            "context_target_alignment": "origin is first target timestamp; context [origin-336h, origin), target [origin, origin+48h)",
            "note": "This removes the old calibration role for this manifest. It is intentionally separate from the earlier train90/V30/cal20/eval80 contract and must be frozen before any Stage A execution.",
        },
        "household": household,
        "bdg2": bdg2,
        "overall": {
            "ready_candidate_count": ready,
            "blocked_candidate_count": blocked,
            "unknown_candidate_count": 0,
            "stage_a_data_gate_status": "PARTIAL_READY_ONE_OF_TWO"
            if ready == 1 and blocked == 1
            else "READY" if ready == 2 else "BLOCKED",
            "interpretation": "[판정] BDG2 Bull Office can be used as one target-blind prepared Stage A development unit. Household post-P1 still needs a target-blind gap/exclusion or split revision before it is READY.",
        },
    }


def manifest_candidate(manifest: dict[str, Any], candidate_id: str) -> dict[str, Any]:
    for candidate in manifest["candidates"]:
        if candidate["candidate_id"] == candidate_id:
            return candidate
    raise KeyError(candidate_id)


def compare_value(label: str, actual: Any, expected: Any, mismatches: list[str], tol: float) -> None:
    if isinstance(actual, float) or isinstance(expected, float):
        try:
            if abs(float(actual) - float(expected)) <= tol:
                return
        except (TypeError, ValueError):
            pass
        mismatches.append(f"{label}: actual={actual!r} expected={expected!r}")
        return
    if actual != expected:
        mismatches.append(f"{label}: actual={actual!r} expected={expected!r}")


def compare_dict(
    prefix: str,
    actual: dict[str, Any],
    expected: dict[str, Any],
    mismatches: list[str],
    tol: float,
) -> None:
    for key, expected_value in expected.items():
        if key not in actual:
            mismatches.append(f"{prefix}.{key}: missing actual")
            continue
        actual_value = actual[key]
        if isinstance(expected_value, dict) and isinstance(actual_value, dict):
            compare_dict(f"{prefix}.{key}", actual_value, expected_value, mismatches, tol)
        else:
            compare_value(f"{prefix}.{key}", actual_value, expected_value, mismatches, tol)


def verify_against_manifest(summary: dict[str, Any], manifest_path: Path, tol: float) -> list[str]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mismatches: list[str] = []

    compare_dict("split_design", summary["split_design"], manifest["split_design"], mismatches, tol)
    compare_dict("overall", summary["overall"], manifest["overall"], mismatches, tol)

    hh_expected = manifest_candidate(manifest, "household_post_p1_earliest_20090429")
    hh_actual = summary["household"]
    compare_value("household.status", hh_actual["status"], hh_expected["status"], mismatches, tol)
    compare_value(
        "household.source_sha256",
        hh_actual["source_sha256"],
        hh_expected["source_provenance"]["source_sha256"],
        mismatches,
        tol,
    )
    compare_value(
        "household.source_rows",
        hh_actual["source_rows"],
        hh_expected["source_provenance"]["source_rows"],
        mismatches,
        tol,
    )
    compare_dict(
        "household.source_timestamp_range",
        hh_actual["source_timestamp_range"],
        hh_expected["source_provenance"]["source_timestamp_range"],
        mismatches,
        tol,
    )
    compare_value(
        "household.block_start",
        hh_actual["block_start"],
        hh_expected["proposed_period"]["block_start"],
        mismatches,
        tol,
    )
    compare_value(
        "household.block_end_exclusive",
        hh_actual["block_end_exclusive"],
        hh_expected["proposed_period"]["block_end_exclusive"],
        mismatches,
        tol,
    )
    compare_dict(
        "household.boundaries",
        hh_actual["boundaries"],
        hh_expected["proposed_period"]["boundaries"],
        mismatches,
        tol,
    )
    compare_dict(
        "household.origin_counts",
        hh_actual["origin_counts"],
        hh_expected["proposed_period"]["origin_counts"],
        mismatches,
        tol,
    )
    compare_value(
        "household.channels",
        hh_actual["channels"],
        hh_expected["series"]["channels"],
        mismatches,
        tol,
    )
    compare_value(
        "household.target_columns",
        hh_actual["target_columns"],
        hh_expected["series"]["target_columns"],
        mismatches,
        tol,
    )
    compare_dict(
        "household.aggregate_target_finite_fraction",
        hh_actual["aggregate_target_finite_fraction"],
        hh_expected["availability_summary"]["aggregate_target_finite_fraction"],
        mismatches,
        tol,
    )
    compare_dict(
        "household.min_origin_window_finite_fraction",
        hh_actual["min_origin_window_finite_fraction"],
        hh_expected["availability_summary"]["min_origin_window_finite_fraction"],
        mismatches,
        tol,
    )
    compare_dict(
        "household.row_finite_fraction_min_by_split",
        hh_actual["row_finite_fraction_min_by_split"],
        hh_expected["availability_summary"]["row_finite_fraction_min_by_split"],
        mismatches,
        tol,
    )
    compare_dict(
        "household.hours_below_threshold_by_channel",
        hh_actual["hours_below_threshold_by_channel"],
        hh_expected["source_provenance"].get("source_qc", {})
        or {"Global_active_power": 71, "Global_reactive_power": 71, "Voltage": 71, "Global_intensity": 71},
        mismatches,
        tol,
    )
    compare_value(
        "household.first_daily_shift_passing_block_start",
        hh_actual["first_daily_shift_passing_block_start"],
        None,
        mismatches,
        tol,
    )
    compare_value(
        "household.post_p1_missing_target_runs_checked",
        hh_actual["post_p1_missing_target_runs_checked"],
        hh_expected["origin_gap"]["post_p1_missing_target_runs_checked"],
        mismatches,
        tol,
    )

    bdg_expected = manifest_candidate(manifest, "bdg2_bull_office_2016a")
    bdg_actual = summary["bdg2"]
    compare_value("bdg2.status", bdg_actual["status"], bdg_expected["status"], mismatches, tol)
    compare_value(
        "bdg2.electricity_sha256",
        bdg_actual["electricity_sha256"],
        bdg_expected["source_provenance"]["electricity_sha256"],
        mismatches,
        tol,
    )
    compare_value(
        "bdg2.metadata_sha256",
        bdg_actual["metadata_sha256"],
        bdg_expected["source_provenance"]["metadata_sha256"],
        mismatches,
        tol,
    )
    compare_value(
        "bdg2.site_id",
        bdg_actual["site_id"],
        bdg_expected["proposed_period"]["site_id"],
        mismatches,
        tol,
    )
    compare_value(
        "bdg2.block_start",
        bdg_actual["block_start"],
        bdg_expected["proposed_period"]["block_start"],
        mismatches,
        tol,
    )
    compare_value(
        "bdg2.block_end_exclusive",
        bdg_actual["block_end_exclusive"],
        bdg_expected["proposed_period"]["block_end_exclusive"],
        mismatches,
        tol,
    )
    compare_dict(
        "bdg2.boundaries",
        bdg_actual["boundaries"],
        bdg_expected["proposed_period"]["boundaries"],
        mismatches,
        tol,
    )
    compare_dict(
        "bdg2.origin_counts",
        bdg_actual["origin_counts"],
        bdg_expected["proposed_period"]["origin_counts"],
        mismatches,
        tol,
    )
    compare_value(
        "bdg2.channels",
        bdg_actual["channels"],
        bdg_expected["series"]["channels"],
        mismatches,
        tol,
    )
    compare_value(
        "bdg2.target_columns",
        bdg_actual["target_columns"],
        bdg_expected["series"]["target_columns"],
        mismatches,
        tol,
    )
    compare_value(
        "bdg2.context_only_columns",
        bdg_actual["context_only_columns"],
        bdg_expected["series"]["context_only_columns"],
        mismatches,
        tol,
    )
    compare_dict(
        "bdg2.aggregate_target_finite_fraction",
        bdg_actual["aggregate_target_finite_fraction"],
        bdg_expected["availability_summary"]["aggregate_target_finite_fraction"],
        mismatches,
        tol,
    )
    compare_dict(
        "bdg2.min_origin_window_finite_fraction",
        bdg_actual["min_origin_window_finite_fraction"],
        bdg_expected["availability_summary"]["min_origin_window_finite_fraction"],
        mismatches,
        tol,
    )
    compare_dict(
        "bdg2.row_finite_fraction_min_by_split",
        bdg_actual["row_finite_fraction_min_by_split"],
        bdg_expected["availability_summary"]["row_finite_fraction_min_by_split"],
        mismatches,
        tol,
    )
    expected_trace = [
        {
            "site_id": item["site_id"],
            "candidate_columns": item["candidate_columns"],
            "office_meter_count": item["office_meter_count"],
            "availability_result": item["availability_result"],
        }
        for item in bdg_expected["selection_trace"]
    ]
    actual_trace = [
        {
            "site_id": item["site_id"],
            "candidate_columns": item["candidate_columns"],
            "office_meter_count": item["office_meter_count"],
            "availability_result": item["availability_result"],
        }
        for item in bdg_actual["selection_trace"]
    ]
    compare_value("bdg2.selection_trace_prefix", actual_trace, expected_trace, mismatches, tol)
    expected_bobcat_min = bdg_expected["selection_trace"][0]["min_origin_window_finite_fraction"]
    actual_bobcat_min = bdg_actual["selection_trace"][0]["min_origin_window_finite_fraction"]
    compare_dict(
        "bdg2.selection_trace_bobcat_min_origin",
        actual_bobcat_min,
        expected_bobcat_min,
        mismatches,
        tol,
    )
    return mismatches


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=repo_root()
        / "results/peft_paper_closure_v1/fresh_stage_a_candidate_manifest.json",
        help="Existing manifest to verify. Default: fresh_stage_a_candidate_manifest.json",
    )
    parser.add_argument(
        "--print-json",
        action="store_true",
        help="Print recomputed target-blind summary to stdout after verification.",
    )
    parser.add_argument("--tolerance", type=float, default=1e-12)
    args = parser.parse_args(argv)

    root = repo_root()
    summary = compute_summary(root)
    mismatches = verify_against_manifest(summary, args.manifest, args.tolerance)

    if args.print_json:
        print(json.dumps(summary, indent=2, ensure_ascii=False))

    if mismatches:
        print("VERIFY FAILED: fresh Stage A candidate manifest mismatch", file=sys.stderr)
        for mismatch in mismatches:
            print(f"- {mismatch}", file=sys.stderr)
        return 1

    print("VERIFY PASS: fresh Stage A candidate manifest reproduces target-blind checks")
    print(
        "household="
        f"{summary['household']['status']} "
        f"train_min_origin={summary['household']['min_origin_window_finite_fraction']['train']:.6f} "
        f"first_passing_post_p1_block={summary['household']['first_daily_shift_passing_block_start']}"
    )
    print(
        "bdg2="
        f"{summary['bdg2']['status']} "
        f"site={summary['bdg2']['site_id']} "
        f"targets={','.join(summary['bdg2']['target_columns'])} "
        f"e2_min_origin={summary['bdg2']['min_origin_window_finite_fraction']['e2']:.6f}"
    )
    print(f"overall={summary['overall']['stage_a_data_gate_status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
