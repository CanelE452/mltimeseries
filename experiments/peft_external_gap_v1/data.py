"""Prepare external hourly panels for the PEFT gap screen.

The split contract is intentionally narrow: model selection receives only the
fit archive, while calibration/evaluation labels live in a separate holdout
archive.  All normalization statistics are fit on the train period only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


VERSION = "peft_external_gap_v1.real.20260908"
CONTEXT = 336
HORIZON = 48
ORIGIN_STRIDE_HOURS = 24
SPLIT_DAYS = {"train": 64, "val": 14, "cal": 14, "eval": 84}
MIN_TARGET_FINITE_FRACTION = 0.70
HOUSEHOLD_MIN_FINITE_MINUTES_PER_HOUR = 45
QUANTILES = np.array(
    [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
     0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99],
    dtype=np.float64,
)


@dataclass(frozen=True)
class PanelSpec:
    name: str
    source_path: Path
    source_url: str
    download_url: str
    citation: str
    license_note: str
    channels: tuple[str, ...]
    target_indices: tuple[int, int]
    parser: str
    source_notes: tuple[str, ...]


BIKE_SPEC = PanelSpec(
    name="bike",
    source_path=Path("data_external/uci_bike_sharing/raw/hour.csv"),
    source_url="https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset",
    download_url="https://archive.ics.uci.edu/static/public/275/bike%2Bsharing%2Bdataset.zip",
    citation="Fanaee-T, Hadi, and Gama, Joao. Bike Sharing Dataset. UCI Machine Learning Repository, 2013.",
    license_note="UCI public dataset; repository metadata should be consulted for the current terms before redistribution.",
    channels=("casual", "registered", "temp", "hum", "windspeed"),
    target_indices=(0, 1),
    parser="hour.csv dteday+hr hourly rows; cnt retained only for source consistency audit, not as a model input.",
    source_notes=("cnt is excluded because it is the sum of the two targets.", "Weather/calendar futures are not provided."),
)

HOUSEHOLD_SPEC = PanelSpec(
    name="household",
    source_path=Path("data/uci_household_power/household_power_consumption.txt"),
    source_url="https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption",
    download_url="https://archive.ics.uci.edu/static/public/235/individual%2Bhousehold%2Belectric%2Bpower%2Bconsumption.zip",
    citation="Hebrail, Georges, and Berard, Alice. Individual household electric power consumption. UCI Machine Learning Repository, 2012.",
    license_note="UCI public dataset; repository metadata should be consulted for the current terms before redistribution.",
    channels=("Global_active_power", "Global_reactive_power", "Voltage", "Global_intensity"),
    target_indices=(0, 1),
    parser="semicolon-delimited minute rows aggregated to hourly means with per-column finite-minute thresholds.",
    source_notes=("Energy sub-metering channels are not used.", "Hourly values are means of observed minute measurements, not energy sums."),
)


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_float(value: object) -> float:
    if value is None:
        return math.nan
    if isinstance(value, (int, float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else math.nan
    text = str(value).strip()
    if text in {"", "?", "NA", "N/A", "nan", "NaN", "NULL", "null"}:
        return math.nan
    text = text.replace(",", ".")
    try:
        number = float(text)
    except ValueError:
        return math.nan
    return number if math.isfinite(number) else math.nan


def _floor_hour(value: datetime) -> datetime:
    return value.replace(minute=0, second=0, microsecond=0)


def _encode_timestamps(timestamps: Sequence[datetime]) -> np.ndarray:
    return np.array([value.isoformat(timespec="seconds") for value in timestamps], dtype=np.str_)


def _manifest_array(value: dict) -> np.ndarray:
    return np.array(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False), dtype=np.str_)


def _average_duplicate_rows(
    timestamps: Sequence[datetime],
    values: np.ndarray,
) -> tuple[list[datetime], np.ndarray, dict]:
    if len(timestamps) != len(values):
        raise ValueError("timestamps and values length mismatch")
    if values.ndim != 2:
        raise ValueError("values must be a [T, C] array")
    order = np.argsort(np.array([ts.timestamp() for ts in timestamps], dtype=np.float64), kind="mergesort")
    grouped: dict[datetime, list[np.ndarray]] = {}
    for index in order:
        grouped.setdefault(timestamps[int(index)], []).append(np.asarray(values[int(index)], dtype=np.float64))

    out_timestamps: list[datetime] = []
    out_values: list[np.ndarray] = []
    duplicate_groups = 0
    duplicate_rows = 0
    for timestamp in sorted(grouped):
        rows = np.vstack(grouped[timestamp])
        if len(rows) > 1:
            duplicate_groups += 1
            duplicate_rows += len(rows) - 1
        finite = np.isfinite(rows)
        counts = finite.sum(axis=0)
        sums = np.where(finite, rows, 0.0).sum(axis=0)
        merged = np.full(rows.shape[1], np.nan, dtype=np.float64)
        np.divide(sums, counts, out=merged, where=counts > 0)
        out_timestamps.append(timestamp)
        out_values.append(merged)

    metadata = {
        "duplicate_timestamp_groups": int(duplicate_groups),
        "duplicate_rows_collapsed": int(duplicate_rows),
    }
    return out_timestamps, np.asarray(out_values, dtype=np.float32), metadata


def _insert_full_hour_grid(
    timestamps: Sequence[datetime],
    values: np.ndarray,
    fill_value: float | int | bool = math.nan,
) -> tuple[list[datetime], np.ndarray, int]:
    if not timestamps:
        raise ValueError("cannot build an hourly grid from an empty source")
    if len(timestamps) != len(values):
        raise ValueError("timestamps and values length mismatch")
    for timestamp in timestamps:
        if timestamp != _floor_hour(timestamp):
            raise ValueError(f"timestamp is not hourly-aligned: {timestamp!r}")

    start, end = min(timestamps), max(timestamps)
    hours = int((end - start).total_seconds() // 3600) + 1
    full = [start + timedelta(hours=i) for i in range(hours)]
    index = {timestamp: i for i, timestamp in enumerate(timestamps)}
    out = np.full((hours, values.shape[1]), fill_value, dtype=values.dtype)
    missing = 0
    for row, timestamp in enumerate(full):
        source_index = index.get(timestamp)
        if source_index is None:
            missing += 1
        else:
            out[row] = values[source_index]
    return full, out, missing


def _finalize_hour_aggregation(
    sums_by_hour: dict[datetime, np.ndarray],
    counts_by_hour: dict[datetime, np.ndarray],
    columns: Sequence[str],
    min_finite_per_hour: int,
) -> dict:
    timestamps = sorted(sums_by_hour)
    values = np.full((len(timestamps), len(columns)), np.nan, dtype=np.float32)
    finite_minutes = np.zeros((len(timestamps), len(columns)), dtype=np.int16)
    for row, timestamp in enumerate(timestamps):
        counts = counts_by_hour[timestamp].astype(np.int64)
        finite_minutes[row] = counts
        ok = counts >= min_finite_per_hour
        means = np.full(len(columns), np.nan, dtype=np.float64)
        np.divide(sums_by_hour[timestamp], counts, out=means, where=counts > 0)
        values[row, ok] = means[ok]
    return {"timestamps": timestamps, "values": values, "finite_minutes": finite_minutes}


def aggregate_minute_rows_to_hourly(
    rows: Iterable[dict],
    columns: Sequence[str],
    min_finite_per_hour: int = HOUSEHOLD_MIN_FINITE_MINUTES_PER_HOUR,
) -> dict:
    sums_by_hour: dict[datetime, np.ndarray] = {}
    counts_by_hour: dict[datetime, np.ndarray] = {}
    for row in rows:
        timestamp = _floor_hour(row["timestamp"])
        if timestamp not in sums_by_hour:
            sums_by_hour[timestamp] = np.zeros(len(columns), dtype=np.float64)
            counts_by_hour[timestamp] = np.zeros(len(columns), dtype=np.int64)
        for column_index, column in enumerate(columns):
            value = _parse_float(row.get(column))
            if math.isfinite(value):
                sums_by_hour[timestamp][column_index] += value
                counts_by_hour[timestamp][column_index] += 1
    return _finalize_hour_aggregation(sums_by_hour, counts_by_hour, columns, min_finite_per_hour)


def load_bike_hourly(path: Path | str = BIKE_SPEC.source_path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    timestamps: list[datetime] = []
    rows: list[list[float]] = []
    cnt_mismatch_count = 0
    cnt_checked = 0
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames = list(reader.fieldnames or [])
        for row in reader:
            timestamp = datetime.strptime(f"{row['dteday']} {int(row['hr']):02d}", "%Y-%m-%d %H")
            values = [_parse_float(row[column]) for column in BIKE_SPEC.channels]
            cnt = _parse_float(row.get("cnt"))
            if all(math.isfinite(value) for value in values[:2]) and math.isfinite(cnt):
                cnt_checked += 1
                if abs((values[0] + values[1]) - cnt) > 1e-5:
                    cnt_mismatch_count += 1
            timestamps.append(timestamp)
            rows.append(values)

    unique_ts, unique_values, duplicate_qc = _average_duplicate_rows(timestamps, np.asarray(rows, dtype=np.float32))
    grid_ts, grid_values, missing_hours = _insert_full_hour_grid(unique_ts, unique_values)
    metadata = {
        "source_rows": int(len(rows)),
        "source_columns": fieldnames,
        "cnt_consistency_checked_rows": int(cnt_checked),
        "cnt_mismatch_count": int(cnt_mismatch_count),
        "missing_hours_inserted": int(missing_hours),
        **duplicate_qc,
    }
    return {"timestamps": grid_ts, "values": grid_values, "metadata": metadata}


def load_household_hourly(path: Path | str = HOUSEHOLD_SPEC.source_path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    sums_by_hour: dict[datetime, np.ndarray] = {}
    counts_by_hour: dict[datetime, np.ndarray] = {}
    raw_rows = 0
    parse_failures = 0
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter=";")
        fieldnames = list(reader.fieldnames or [])
        for row in reader:
            raw_rows += 1
            try:
                timestamp = datetime.strptime(f"{row['Date']} {row['Time']}", "%d/%m/%Y %H:%M:%S")
            except (KeyError, ValueError):
                parse_failures += 1
                continue
            hour = _floor_hour(timestamp)
            if hour not in sums_by_hour:
                sums_by_hour[hour] = np.zeros(len(HOUSEHOLD_SPEC.channels), dtype=np.float64)
                counts_by_hour[hour] = np.zeros(len(HOUSEHOLD_SPEC.channels), dtype=np.int64)
            for column_index, column in enumerate(HOUSEHOLD_SPEC.channels):
                value = _parse_float(row.get(column))
                if math.isfinite(value):
                    sums_by_hour[hour][column_index] += value
                    counts_by_hour[hour][column_index] += 1

    aggregated = _finalize_hour_aggregation(
        sums_by_hour,
        counts_by_hour,
        HOUSEHOLD_SPEC.channels,
        HOUSEHOLD_MIN_FINITE_MINUTES_PER_HOUR,
    )
    grid_ts, grid_values, missing_hours = _insert_full_hour_grid(aggregated["timestamps"], aggregated["values"])
    _, grid_minutes, _ = _insert_full_hour_grid(aggregated["timestamps"], aggregated["finite_minutes"], fill_value=0)
    metadata = {
        "source_rows": int(raw_rows),
        "source_columns": fieldnames,
        "parse_failures": int(parse_failures),
        "missing_hours_inserted": int(missing_hours),
        "min_finite_minutes_per_hour": int(HOUSEHOLD_MIN_FINITE_MINUTES_PER_HOUR),
        "hours_below_threshold_by_channel": {
            channel: int((grid_minutes[:, index] < HOUSEHOLD_MIN_FINITE_MINUTES_PER_HOUR).sum())
            for index, channel in enumerate(HOUSEHOLD_SPEC.channels)
        },
        "finite_minute_count_by_channel": {
            channel: int(grid_minutes[:, index].sum())
            for index, channel in enumerate(HOUSEHOLD_SPEC.channels)
        },
    }
    return {"timestamps": grid_ts, "values": grid_values, "finite_minutes": grid_minutes, "metadata": metadata}


def _first_complete_day_start(timestamps: Sequence[datetime]) -> datetime:
    if not timestamps:
        raise ValueError("empty timestamp sequence")
    present = set(timestamps)
    first = min(timestamps)
    candidate = datetime(first.year, first.month, first.day)
    if first > candidate:
        candidate += timedelta(days=1)
    last = max(timestamps)
    while candidate + timedelta(hours=23) <= last:
        if all(candidate + timedelta(hours=hour) in present for hour in range(24)):
            return candidate
        candidate += timedelta(days=1)
    raise ValueError("no complete day exists in timestamp grid")


def temporal_contract(timestamps: Sequence[datetime]) -> dict:
    start = _first_complete_day_start(timestamps)
    pre_end = start + timedelta(hours=CONTEXT)
    train_end = pre_end + timedelta(days=SPLIT_DAYS["train"])
    val_end = train_end + timedelta(days=SPLIT_DAYS["val"])
    cal_end = val_end + timedelta(days=SPLIT_DAYS["cal"])
    eval_end = cal_end + timedelta(days=SPLIT_DAYS["eval"])
    boundaries = {
        "precontext": (start, pre_end),
        "train": (pre_end, train_end),
        "val": (train_end, val_end),
        "cal": (val_end, cal_end),
        "eval": (cal_end, eval_end),
    }
    origins: dict[str, list[datetime]] = {}
    for split in ("train", "val", "cal", "eval"):
        split_start, split_end = boundaries[split]
        last_origin = split_end - timedelta(hours=HORIZON)
        current = split_start
        values: list[datetime] = []
        while current <= last_origin:
            values.append(current)
            current += timedelta(hours=ORIGIN_STRIDE_HOURS)
        origins[split] = values
    return {
        "context": CONTEXT,
        "horizon": HORIZON,
        "origin_stride_hours": ORIGIN_STRIDE_HOURS,
        "target_window_overlap_hours": max(0, HORIZON - ORIGIN_STRIDE_HOURS),
        "boundaries": boundaries,
        "origins": origins,
    }


def causal_ffill_context(values: np.ndarray, fit_median: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    fit_median = np.asarray(fit_median, dtype=np.float32)
    if values.ndim != 2 or fit_median.shape != (values.shape[1],):
        raise ValueError("Expected values [T, C] and fit_median [C]")
    if not np.isfinite(fit_median).all():
        raise ValueError("fit_median must be finite for causal fill fallback")

    filled = np.empty_like(values, dtype=np.float32)
    last = fit_median.astype(np.float32).copy()
    for row in range(values.shape[0]):
        observed = np.isfinite(values[row])
        last[observed] = values[row, observed]
        filled[row] = last
    return filled


def _timestamp_index(timestamps: Sequence[datetime]) -> dict[datetime, int]:
    return {timestamp: index for index, timestamp in enumerate(timestamps)}


def _end_index(index: dict[datetime, int], timestamps: Sequence[datetime], end: datetime) -> int:
    if end in index:
        return index[end]
    if timestamps and end == timestamps[-1] + timedelta(hours=1):
        return len(timestamps)
    raise ValueError(f"split boundary is outside the hourly grid: {end!r}")


def _boundary_indices(timestamps: Sequence[datetime], boundaries: dict) -> dict[str, tuple[int, int]]:
    index = _timestamp_index(timestamps)
    out: dict[str, tuple[int, int]] = {}
    for name, (start, end) in boundaries.items():
        if start not in index:
            raise ValueError(f"split boundary is outside the hourly grid: {start!r}")
        out[name] = (index[start], _end_index(index, timestamps, end))
    return out


def _origin_indices(timestamps: Sequence[datetime], origins: Sequence[datetime]) -> np.ndarray:
    index = _timestamp_index(timestamps)
    missing = [origin for origin in origins if origin not in index]
    if missing:
        raise ValueError(f"origin is outside the hourly grid: {missing[0]!r}")
    return np.array([index[origin] for origin in origins], dtype=np.int64)


def _fit_stats(values: np.ndarray, train_bounds: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train = values[train_bounds[0]: train_bounds[1]].astype(np.float64)
    finite = np.isfinite(train)
    observed_counts = finite.sum(axis=0)
    if np.any(observed_counts == 0):
        bad = np.where(observed_counts == 0)[0].tolist()
        raise ValueError(f"train period has no finite observations for columns {bad}")
    means = np.nanmean(train, axis=0)
    medians = np.nanmedian(train, axis=0)
    stds = np.nanstd(train, axis=0)
    if (not np.isfinite(means).all()) or (not np.isfinite(medians).all()) or (not np.isfinite(stds).all()):
        raise ValueError("train fit statistics must be finite")
    if np.any(stds <= 0):
        bad = np.where(stds <= 0)[0].tolist()
        raise ValueError(f"train fit std must be positive for every channel; bad columns {bad}")
    return means.astype(np.float32), stds.astype(np.float32), medians.astype(np.float32)


def _loss_mask(raw_values: np.ndarray, target_indices: Sequence[int]) -> np.ndarray:
    mask = np.zeros(raw_values.shape, dtype=bool)
    for target_index in target_indices:
        mask[:, int(target_index)] = np.isfinite(raw_values[:, int(target_index)])
    return mask


def _target_window_qc(raw_values: np.ndarray, origins: dict[str, np.ndarray], channels: Sequence[str], target_indices: Sequence[int]) -> dict:
    qc: dict[str, dict] = {}
    for split, origin_values in origins.items():
        split_qc = {}
        for target_index in target_indices:
            total = int(len(origin_values) * HORIZON)
            valid = 0
            for origin in origin_values:
                valid += int(np.isfinite(raw_values[int(origin): int(origin) + HORIZON, int(target_index)]).sum())
            fraction = (valid / total) if total else math.nan
            split_qc[channels[int(target_index)]] = {
                "valid": int(valid),
                "total": int(total),
                "finite_fraction": float(fraction),
                "pass_min_70pct": bool(fraction >= MIN_TARGET_FINITE_FRACTION),
            }
        qc[split] = split_qc
    return qc


def _split_row_qc(raw_values: np.ndarray, boundaries: dict[str, tuple[int, int]], channels: Sequence[str]) -> dict:
    qc: dict[str, dict] = {}
    for split, (start, end) in boundaries.items():
        rows = raw_values[start:end]
        split_qc = {}
        for column_index, channel in enumerate(channels):
            finite = int(np.isfinite(rows[:, column_index]).sum())
            total = int(rows.shape[0])
            split_qc[channel] = {
                "finite": finite,
                "total": total,
                "finite_fraction": float(finite / total) if total else math.nan,
            }
        qc[split] = split_qc
    return qc


def _contract_json(contract: dict) -> dict:
    return {
        "context": int(contract["context"]),
        "horizon": int(contract["horizon"]),
        "origin_stride_hours": int(contract["origin_stride_hours"]),
        "target_window_overlap_hours": int(contract["target_window_overlap_hours"]),
        "boundaries": {
            name: [start.isoformat(timespec="seconds"), end.isoformat(timespec="seconds")]
            for name, (start, end) in contract["boundaries"].items()
        },
        "origin_counts": {name: len(values) for name, values in contract["origins"].items()},
        "first_origin": {name: values[0].isoformat(timespec="seconds") if values else None for name, values in contract["origins"].items()},
        "last_origin": {name: values[-1].isoformat(timespec="seconds") if values else None for name, values in contract["origins"].items()},
    }


def _source_manifest(spec: PanelSpec) -> dict:
    path = spec.source_path
    manifest = {
        "panel": spec.name,
        "source_path": str(path),
        "source_exists": path.exists(),
        "source_sha256": sha256_file(path) if path.exists() else None,
        "source_url": spec.source_url,
        "download_url": spec.download_url,
        "citation": spec.citation,
        "license_note": spec.license_note,
        "parser": spec.parser,
        "source_notes": list(spec.source_notes),
        "channels": list(spec.channels),
        "target_indices": list(spec.target_indices),
    }
    archive_candidates = {
        "bike": Path("data_external/uci_bike_sharing/bike_sharing_dataset.zip"),
        "household": Path("data/uci_household_power/_zip/household_power_consumption.zip"),
    }
    archive = archive_candidates.get(spec.name)
    if archive and archive.exists():
        manifest["archive_path"] = str(archive)
        manifest["archive_sha256"] = sha256_file(archive)
    return manifest


def _archive_common_arrays(
    context_values: np.ndarray,
    target_values: np.ndarray,
    observed_mask: np.ndarray,
    target_loss_mask: np.ndarray,
    timestamps: Sequence[datetime],
    channels: Sequence[str],
    target_indices: Sequence[int],
    fit_mean: np.ndarray,
    fit_std: np.ndarray,
    fit_median: np.ndarray,
    archive_manifest: dict,
) -> dict:
    return {
        "context_values": context_values.astype(np.float32),
        "target_values": target_values.astype(np.float32),
        "observed_mask": observed_mask.astype(bool),
        "target_loss_mask": target_loss_mask.astype(bool),
        "timestamps": _encode_timestamps(timestamps),
        "channels": np.array(list(channels), dtype=np.str_),
        "target_indices": np.array(target_indices, dtype=np.int64),
        "fit_mean": fit_mean.astype(np.float32),
        "fit_std": fit_std.astype(np.float32),
        "fit_median": fit_median.astype(np.float32),
        "context": np.array(CONTEXT, dtype=np.int64),
        "horizon": np.array(HORIZON, dtype=np.int64),
        "origin_stride_hours": np.array(ORIGIN_STRIDE_HOURS, dtype=np.int64),
        "quantiles": QUANTILES.astype(np.float64),
        "manifest_json": _manifest_array(archive_manifest),
    }


def write_panel_archives(
    panel: str,
    timestamps: Sequence[datetime],
    raw_values: np.ndarray,
    channels: Sequence[str],
    target_indices: Sequence[int],
    output_dir: Path | str,
    source: dict,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fit_path = output_dir / f"{panel}_fit.npz"
    holdout_path = output_dir / f"{panel}_holdout.npz"
    if fit_path.exists() or holdout_path.exists():
        raise FileExistsError(f"refusing to overwrite prepared archives for panel {panel!r}")

    raw_values = np.asarray(raw_values, dtype=np.float32)
    if raw_values.ndim != 2:
        raise ValueError("raw_values must be [T, C]")
    if len(timestamps) != raw_values.shape[0]:
        raise ValueError("timestamps and raw_values length mismatch")
    if list(target_indices) != [0, 1]:
        raise ValueError("target_indices must be [0, 1] for this screen")
    if len(channels) != raw_values.shape[1]:
        raise ValueError("channels and raw_values column count mismatch")
    if any(timestamps[index] >= timestamps[index + 1] for index in range(len(timestamps) - 1)):
        raise ValueError("timestamps must be strictly increasing after duplicate handling")
    if any((timestamps[index + 1] - timestamps[index]) != timedelta(hours=1) for index in range(len(timestamps) - 1)):
        raise ValueError("timestamps must be a complete hourly grid before archive writing")

    contract = temporal_contract(timestamps)
    required_end = contract["boundaries"]["eval"][1]
    if required_end > timestamps[-1] + timedelta(hours=1):
        raise ValueError(f"source is too short for the fixed 190-day contract ending {required_end.isoformat()}")
    full_boundaries = _boundary_indices(timestamps, contract["boundaries"])
    full_origins = {split: _origin_indices(timestamps, values) for split, values in contract["origins"].items()}
    archive_start, archive_end = full_boundaries["precontext"][0], full_boundaries["eval"][1]
    timestamps = list(timestamps[archive_start:archive_end])
    raw_values = raw_values[archive_start:archive_end]
    boundaries = {
        name: (int(start - archive_start), int(end - archive_start))
        for name, (start, end) in full_boundaries.items()
    }
    origins = {split: values - archive_start for split, values in full_origins.items()}

    fit_mean, fit_std, fit_median = _fit_stats(raw_values, boundaries["train"])
    context_values = causal_ffill_context(raw_values, fit_median)
    if not np.isfinite(context_values).all():
        raise ValueError("context_values must be finite after causal fill")

    observed_mask = np.isfinite(raw_values)
    target_loss_mask = _loss_mask(raw_values, target_indices)
    target_qc = _target_window_qc(raw_values, origins, channels, target_indices)
    failing = [
        (split, target)
        for split, per_target in target_qc.items()
        for target, metrics in per_target.items()
        if not metrics["pass_min_70pct"]
    ]
    if failing:
        raise ValueError(f"target finite ratio below 70% for {failing}")

    row_qc = _split_row_qc(raw_values, boundaries, channels)
    contract_for_json = _contract_json(contract)
    base_manifest = {
        "version": VERSION,
        "panel": panel,
        "dataset": panel,
        "channels": list(channels),
        "target_indices": [int(value) for value in target_indices],
        "target_columns": [channels[int(value)] for value in target_indices],
        "context": CONTEXT,
        "horizon": HORIZON,
        "origin_stride_hours": ORIGIN_STRIDE_HOURS,
        "data_semantics": {
            "origin_index": "first target timestamp; context is [origin-context, origin), target is [origin, origin+horizon)",
            "context_values": "causal forward-fill of raw hourly values with train-only median fallback for leading gaps",
            "target_values": "raw hourly values with NaNs preserved",
            "target_loss_mask": "true only for target columns with finite raw values",
            "fit_statistics": "mean/std/median fit on observed train-period values only",
        },
        "contract": contract_for_json,
        "boundary_indices": {name: [int(start), int(end)] for name, (start, end) in boundaries.items()},
        "origin_counts": {name: int(len(values)) for name, values in origins.items()},
        "source": source,
        "qc": {"target_windows": target_qc, "split_rows": row_qc},
    }

    holdout_manifest = {
        **base_manifest,
        "archive_role": "holdout_cal_eval_only_after_selection",
        "forbidden_for": ["training", "validation_selection", "S0_smoke"],
        "included_origins": ["cal", "eval"],
    }
    holdout_arrays = _archive_common_arrays(
        context_values,
        raw_values,
        observed_mask,
        target_loss_mask,
        timestamps,
        channels,
        target_indices,
        fit_mean,
        fit_std,
        fit_median,
        holdout_manifest,
    )
    holdout_arrays["cal_origins"] = origins["cal"]
    holdout_arrays["eval_origins"] = origins["eval"]
    np.savez_compressed(holdout_path, **holdout_arrays)
    holdout_sha = sha256_file(holdout_path)

    fit_end = boundaries["val"][1]
    fit_manifest = {
        **base_manifest,
        "archive_role": "fit_train_val_only",
        "forbidden_for": ["calibration", "evaluation"],
        "included_origins": ["train", "val"],
        "holdout_archive_path": str(holdout_path),
        "holdout_archive_sha256": holdout_sha,
    }
    fit_arrays = _archive_common_arrays(
        context_values[:fit_end],
        raw_values[:fit_end],
        observed_mask[:fit_end],
        target_loss_mask[:fit_end],
        timestamps[:fit_end],
        channels,
        target_indices,
        fit_mean,
        fit_std,
        fit_median,
        fit_manifest,
    )
    fit_arrays["train_origins"] = origins["train"]
    fit_arrays["val_origins"] = origins["val"]
    np.savez_compressed(fit_path, **fit_arrays)
    fit_sha = sha256_file(fit_path)

    return {
        "panel": panel,
        "fit_path": str(fit_path),
        "holdout_path": str(holdout_path),
        "fit_sha256": fit_sha,
        "holdout_sha256": holdout_sha,
        "fit_shape": [int(fit_end), int(raw_values.shape[1])],
        "holdout_shape": [int(raw_values.shape[0]), int(raw_values.shape[1])],
        "origin_counts": {name: int(len(values)) for name, values in origins.items()},
        "target_qc": target_qc,
        "row_qc": row_qc,
        "contract": contract_for_json,
        "source": source,
    }


def build_panel(panel: str) -> tuple[PanelSpec, list[datetime], np.ndarray, dict]:
    if panel == "bike":
        loaded = load_bike_hourly(BIKE_SPEC.source_path)
        return BIKE_SPEC, loaded["timestamps"], loaded["values"], loaded["metadata"]
    if panel == "household":
        loaded = load_household_hourly(HOUSEHOLD_SPEC.source_path)
        return HOUSEHOLD_SPEC, loaded["timestamps"], loaded["values"], loaded["metadata"]
    raise ValueError(f"unknown panel {panel!r}")


def prepare(output_dir: Path | str) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite existing manifest: {manifest_path}")

    panels = {}
    files = {}
    all_qc_passed = True
    for panel in ("bike", "household"):
        spec, timestamps, values, source_qc = build_panel(panel)
        source = _source_manifest(spec)
        source["source_qc"] = source_qc
        summary = write_panel_archives(
            panel=panel,
            timestamps=timestamps,
            raw_values=values,
            channels=spec.channels,
            target_indices=spec.target_indices,
            output_dir=output_dir,
            source=source,
        )
        panels[panel] = summary
        files[f"{panel}_fit"] = {"path": summary["fit_path"], "sha256": summary["fit_sha256"]}
        files[f"{panel}_holdout"] = {"path": summary["holdout_path"], "sha256": summary["holdout_sha256"]}
        for per_split in summary["target_qc"].values():
            for metrics in per_split.values():
                all_qc_passed = all_qc_passed and bool(metrics["pass_min_70pct"])

    manifest = {
        "version": VERSION,
        "created_utc": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "all_qc_passed": bool(all_qc_passed),
        "datasets": ["bike", "household"],
        "files": files,
        "panels": panels,
        "data_contract": {
            "fit_archives": "train+val only; no cal/eval origin arrays or holdout labels for selection",
            "holdout_archives": "cal/eval origins and labels; may be read only after method/LR/checkpoint selection is fixed",
            "context_target_alignment": "origin is first target timestamp; context [origin-336, origin), target [origin, origin+48)",
            "missing_policy": "targets preserve NaN; context causally forward-filled with train-median fallback; masks carry loss eligibility",
            "score_columns": "only target_indices [0,1] are scored",
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    manifest["manifest_sha256"] = sha256_file(manifest_path)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/peft_external_gap_v1/prepared"))
    args = parser.parse_args(argv)
    manifest = prepare(args.output)
    print(json.dumps({
        "completed": True,
        "manifest_path": manifest["manifest_path"],
        "manifest_sha256": manifest["manifest_sha256"],
        "all_qc_passed": manifest["all_qc_passed"],
        "files": manifest["files"],
    }, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
