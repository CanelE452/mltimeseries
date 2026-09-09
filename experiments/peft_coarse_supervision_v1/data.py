from __future__ import annotations

import argparse
from collections.abc import Mapping
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

for _thread_env in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_thread_env, "2")

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
STUDY = "peft_coarse_supervision_v1"
PLAN = ROOT / "_docs/notes/tsfm_topics/17_coarse_supervision_learning_plan_20260908.md"
DATA_ENTRY = ROOT / "data_external/bdg2_coarse_supervision_v1"
DEFAULT_OUTPUT = ROOT / "runs/peft_coarse_supervision_v1/data"
SITE_NAMES = ("Eagle", "Lamb")
SITE_TO_INT = {"Eagle": 0, "Lamb": 1}
TRAIN_MONTHS = [(2016, month) for month in range(2, 13)]
VALIDATION_MONTHS = [(2017, month) for month in range(1, 4)]
EVALUATION_MONTHS = [(2017, month) for month in range(4, 7)]
RAW_END_MONTH = (2017, 6)
CONTEXT = 512
MAX_HORIZON = 744
EXPECTED_START = datetime(2016, 1, 1, 0, 0, 0)
EXPECTED_END = datetime(2017, 12, 31, 23, 0, 0)
EXPECTED_TOTAL_ROWS = 17544
EXPECTED_2016_ROWS = 8784
OUTPUT_FILES = (
    "train.npz",
    "validation.npz",
    "evaluation_inputs.npz",
    "evaluation_truth.npz",
    "donor_template.npz",
    "metadata.json",
)


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(root: str | Path = ROOT, output_dir: str | Path | None = None) -> dict[str, Any]:
    root = Path(root)
    data_entry = root / "data_external/bdg2_coarse_supervision_v1"
    return build_from_sources(
        metadata_path=data_entry / "raw/metadata.csv",
        raw_path=data_entry / "raw/electricity.csv",
        qc_path=data_entry / "qc.json",
        audit_path=data_entry / "independent_audit.json",
        output_dir=root / "runs/peft_coarse_supervision_v1/data" if output_dir is None else output_dir,
        plan_path=root / "_docs/notes/tsfm_topics/17_coarse_supervision_learning_plan_20260908.md",
        root=root,
    )


def validate(root: str | Path = ROOT, contract_path: str | Path | None = None) -> dict[str, Any]:
    root = Path(root).resolve()
    metadata_path = (
        root / "runs/peft_coarse_supervision_v1/data/metadata.json"
        if contract_path is None
        else Path(contract_path)
    )
    if metadata_path.is_dir():
        metadata_path = metadata_path / "metadata.json"
    metadata = _read_json(metadata_path)
    output_dir = metadata_path.parent
    if "data_build_contract_sha256" in metadata:
        contract = _read_json(output_dir / "data_build_contract.json")
        if sha256(output_dir / "data_build_contract.json") != metadata["data_build_contract_sha256"]:
            raise AssertionError("Data build contract hash changed")
        _verify_data_build_contract(contract, root)
    if "source_hashes" in metadata:
        _verify_record_hashes(metadata["source_hashes"], root, label="Prepared data input")
    if "source_code_hashes" in metadata:
        _verify_record_hashes(metadata["source_code_hashes"], root, label="Prepared data source")
    data_files: dict[str, dict[str, str]] = {}
    for key, filename in {
        "train": "train.npz",
        "validation": "validation.npz",
        "evaluation_inputs": "evaluation_inputs.npz",
        "evaluation_truth": "evaluation_truth.npz",
        "donor_template": "donor_template.npz",
        "metadata": "metadata.json",
    }.items():
        path = output_dir / filename
        if not path.exists():
            raise FileNotFoundError(path)
        actual = sha256(path)
        if filename != "metadata.json":
            expected = metadata["output_files"][filename]["sha256"]
            if actual != expected:
                raise AssertionError(f"Prepared file hash changed: {path}")
        data_files[key] = {"path": str(path.resolve()), "sha256": actual}
    run_root = root / "runs/peft_coarse_supervision_v1"
    return {
        "completed": bool(metadata.get("completed", False)),
        "data": data_files,
        "paths": {
            "cache": str((run_root / "cache/cache.npz").resolve()),
            "cache_result": str((run_root / "cache/result.json").resolve()),
            "ridge_head": str((run_root / "ridge/head_weights.npz").resolve()),
            "ridge_result": str((run_root / "ridge/result.json").resolve()),
        },
        "metadata": metadata,
    }


def _verify_record_hashes(record: Mapping[str, str], root: Path, *, label: str) -> None:
    for recorded_path, expected in record.items():
        path = Path(recorded_path)
        if not path.is_absolute():
            path = root / recorded_path
        if not path.exists():
            raise FileNotFoundError(path)
        actual = sha256(path)
        if actual != expected:
            raise AssertionError(f"{label} changed: {recorded_path}")


def _verify_data_build_contract(contract: Mapping[str, Any], root: Path) -> None:
    _verify_record_hashes(contract["source_hashes"], root, label="Data preparation source")
    _verify_record_hashes(contract["input_hashes"], root, label="Data preparation input")
    plan_path = root / contract["plan_path"] if not Path(contract["plan_path"]).is_absolute() else Path(contract["plan_path"])
    if sha256(plan_path) != contract["plan_sha256"]:
        raise AssertionError("Data preparation plan changed")


def build_from_sources(
    *,
    metadata_path: str | Path,
    raw_path: str | Path,
    qc_path: str | Path,
    audit_path: str | Path,
    output_dir: str | Path,
    plan_path: str | Path = PLAN,
    root: str | Path | None = None,
) -> dict[str, Any]:
    metadata_path = Path(metadata_path)
    raw_path = Path(raw_path)
    qc_path = Path(qc_path)
    audit_path = Path(audit_path)
    output_dir = Path(output_dir)
    plan_path = Path(plan_path)
    source_path = Path(__file__).resolve()
    root_path = Path(root).resolve() if root is not None else source_path.parents[2]
    output_dir.mkdir(parents=True, exist_ok=True)
    _refuse_existing_outputs(output_dir)

    qc = _unwrap_qc(_read_json(qc_path))
    if qc.get("decision") != "PASS":
        raise ValueError(f"Entry QC must pass before data preparation, found {qc.get('decision')}")
    audit = _read_json(audit_path)
    if audit.get("verdict") != "PASS":
        raise ValueError("Independent data-entry audit must pass before data preparation")

    selection = _selection_from_qc(qc)
    data_build_contract = _data_build_contract(
        metadata_path=metadata_path,
        raw_path=raw_path,
        qc_path=qc_path,
        audit_path=audit_path,
        output_dir=output_dir,
        plan_path=plan_path,
        source_path=source_path,
        root=root_path,
        selection=selection,
    )
    _write_json_once(output_dir / "data_build_contract.json", data_build_contract)

    raw = _read_selected_raw(raw_path, selection["all_selected_ids"])
    metadata_rows = _read_metadata(metadata_path, selection)
    monthly = _monthly_totals(raw, selection["target_ids"], selection["donor_ids"])
    template = _donor_template(raw, selection)
    rows = _build_rows(raw, selection, monthly, template)
    gates = _validation_gates(rows)

    _save_split(output_dir / "train.npz", rows["train"], include_total=True)
    _save_split(output_dir / "validation.npz", rows["validation"], include_total=True)
    _save_split(output_dir / "evaluation_inputs.npz", rows["evaluation"], include_total=False)
    _save_truth(output_dir / "evaluation_truth.npz", rows["evaluation"])
    _save_donor_template(output_dir / "donor_template.npz", template, selection)

    metadata_record = _metadata_record(
        data_build_contract=data_build_contract,
        metadata_rows=metadata_rows,
        selection=selection,
        raw=raw,
        rows=rows,
        gates=gates,
        output_dir=output_dir,
        output_files=_hash_outputs(output_dir),
    )
    _write_json_once(output_dir / "metadata.json", metadata_record)
    _verify_data_build_contract(data_build_contract, root_path)
    return validate(root_path, output_dir / "metadata.json")


def _data_build_contract(
    *,
    metadata_path: Path,
    raw_path: Path,
    qc_path: Path,
    audit_path: Path,
    output_dir: Path,
    plan_path: Path,
    source_path: Path,
    root: Path,
    selection: dict[str, Any],
) -> dict[str, Any]:
    return {
        "completed": True,
        "stage": "coarse_supervision_data_build",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "plan_path": _safe_rel(plan_path, root),
        "plan_sha256": sha256(plan_path),
        "source_hashes": {
            _safe_rel(source_path, root): sha256(source_path),
        },
        "input_hashes": {
            _safe_rel(metadata_path, root): sha256(metadata_path),
            _safe_rel(raw_path, root): sha256(raw_path),
            _safe_rel(qc_path, root): sha256(qc_path),
            _safe_rel(audit_path, root): sha256(audit_path),
        },
        "output_dir": str(output_dir.resolve()),
        "selection": {
            "site_names": list(SITE_NAMES),
            "donor_ids": selection["donor_ids_by_site"],
            "target_ids": selection["target_ids_by_site"],
            "global_target_ids": selection["target_ids"],
        },
        "leakage_boundary": {
            "contract_written_before_numeric_2017_processing": True,
            "target_fine_history_used_for_inputs": False,
            "target_2016_monthly_totals_used": True,
            "target_2017_previous_completed_month_totals_allowed": True,
            "target_current_month_total_in_evaluation_inputs": False,
            "source_energy_after_2017_06_processed": False,
        },
    }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json_once(path: Path, record: dict[str, Any]) -> None:
    if path.exists():
        existing = _read_json(path)
        comparable = dict(record)
        comparable.pop("created_at_utc", None)
        existing_cmp = dict(existing)
        existing_cmp.pop("created_at_utc", None)
        if existing_cmp != comparable:
            raise FileExistsError(f"output already exists with different content: {path}")
        return
    path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _refuse_existing_outputs(output_dir: Path) -> None:
    existing = [name for name in OUTPUT_FILES if (output_dir / name).exists()]
    if existing:
        raise FileExistsError(f"output already exists: {existing}")


def _safe_rel(path: Path, root: Path) -> str:
    try:
        return Path(path).resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(Path(path).resolve())


def _unwrap_qc(record: dict[str, Any]) -> dict[str, Any]:
    return record["qc"] if "qc" in record else record


def _selection_from_qc(qc: dict[str, Any]) -> dict[str, Any]:
    sites = qc["selection"]["sites"]
    donor_ids_by_site = {site: list(sites[site]["donor_ids"]) for site in SITE_NAMES}
    target_ids_by_site = {site: list(sites[site]["target_ids"]) for site in SITE_NAMES}
    donor_ids = [bid for site in SITE_NAMES for bid in donor_ids_by_site[site]]
    target_ids = [bid for site in SITE_NAMES for bid in target_ids_by_site[site]]
    if len(set(donor_ids + target_ids)) != len(donor_ids) + len(target_ids):
        raise ValueError("Selected donor/target IDs must be unique")
    return {
        "donor_ids_by_site": donor_ids_by_site,
        "target_ids_by_site": target_ids_by_site,
        "donor_ids": donor_ids,
        "target_ids": target_ids,
        "all_selected_ids": donor_ids + target_ids,
        "target_to_index": {building_id: index for index, building_id in enumerate(target_ids)},
    }


def _read_metadata(metadata_path: Path, selection: dict[str, Any]) -> dict[str, dict[str, str]]:
    selected = set(selection["all_selected_ids"])
    rows: dict[str, dict[str, str]] = {}
    with metadata_path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            building_id = row.get("building_id", "")
            if building_id in selected:
                rows[building_id] = {
                    key: row.get(key, "")
                    for key in [
                        "building_id",
                        "site_id",
                        "primaryspaceusage",
                        "sub_primaryspaceusage",
                        "sqm",
                        "timezone",
                        "electricity",
                    ]
                }
    missing = sorted(selected.difference(rows))
    if missing:
        raise ValueError(f"Metadata missing selected IDs: {missing}")
    return rows


def _read_selected_raw(raw_path: Path, selected_ids: list[str]) -> dict[str, Any]:
    values = {building_id: [] for building_id in selected_ids}
    timestamps_to_june: list[datetime] = []
    full_timestamps: list[datetime] = []
    parse_until = _month_end(*RAW_END_MONTH)
    with raw_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        if "timestamp" not in header:
            raise ValueError("Raw meter file missing timestamp column")
        timestamp_index = header.index("timestamp")
        missing = [building_id for building_id in selected_ids if building_id not in header]
        if missing:
            raise ValueError(f"Raw meter file missing selected columns: {missing}")
        indices = {building_id: header.index(building_id) for building_id in selected_ids}
        for row in reader:
            ts = datetime.strptime(row[timestamp_index], "%Y-%m-%d %H:%M:%S")
            full_timestamps.append(ts)
            if ts > parse_until:
                continue
            timestamps_to_june.append(ts)
            for building_id, column_index in indices.items():
                values[building_id].append(_parse_float(row[column_index]))
    _verify_full_timestamp_grid(full_timestamps)
    arrays = {building_id: np.asarray(series, dtype=np.float64) for building_id, series in values.items()}
    timestamps = np.asarray(timestamps_to_june, dtype="datetime64[h]")
    expected_to_june = int((_month_end(2017, 6) - EXPECTED_START).total_seconds() // 3600) + 1
    if len(timestamps) != expected_to_june:
        raise ValueError("Unexpected row count through 2017-06")
    return {
        "timestamps": timestamps,
        "values": arrays,
        "timestamp_qc": {
            "total_rows": len(full_timestamps),
            "start": full_timestamps[0].strftime("%Y-%m-%d %H:%M:%S"),
            "end": full_timestamps[-1].strftime("%Y-%m-%d %H:%M:%S"),
            "rows_through_2017_06": len(timestamps),
            "source_energy_after_2017_06_processed": False,
        },
    }


def _verify_full_timestamp_grid(timestamps: list[datetime]) -> None:
    if len(timestamps) != EXPECTED_TOTAL_ROWS:
        raise ValueError(f"Expected {EXPECTED_TOTAL_ROWS} timestamp rows, found {len(timestamps)}")
    expected = EXPECTED_START
    for actual in timestamps:
        if actual != expected:
            raise ValueError(f"Unexpected timestamp grid at {actual}, expected {expected}")
        expected += timedelta(hours=1)
    if timestamps[-1] != EXPECTED_END:
        raise ValueError("Unexpected timestamp end")


def _parse_float(text: str) -> float:
    text = text.strip()
    if text == "":
        return float("nan")
    try:
        return float(text)
    except ValueError:
        return float("nan")


def _month_start(year: int, month: int) -> np.datetime64:
    return np.datetime64(f"{year:04d}-{month:02d}-01T00", "h")


def _next_month(year: int, month: int) -> tuple[int, int]:
    return (year + 1, 1) if month == 12 else (year, month + 1)


def _month_end(year: int, month: int) -> datetime:
    next_year, next_month = _next_month(year, month)
    return datetime(next_year, next_month, 1) - timedelta(hours=1)


def _month_hours(year: int, month: int) -> int:
    start = datetime(year, month, 1)
    end = _month_end(year, month)
    return int((end - start).total_seconds() // 3600) + 1


def _month_key(year: int, month: int) -> str:
    return f"{year:04d}-{month:02d}"


def _mask_for_month(timestamps: np.ndarray, year: int, month: int) -> np.ndarray:
    start = _month_start(year, month)
    next_year, next_month = _next_month(year, month)
    end = _month_start(next_year, next_month)
    return (timestamps >= start) & (timestamps < end)


def _monthly_totals(
    raw: dict[str, Any],
    target_ids: list[str],
    donor_ids: list[str],
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {building_id: {} for building_id in target_ids + donor_ids}
    for year, month in [(2016, m) for m in range(1, 13)] + [(2017, m) for m in range(1, 7)]:
        mask = _mask_for_month(raw["timestamps"], year, month)
        expected = _month_hours(year, month)
        if int(mask.sum()) != expected:
            raise ValueError(f"Month {year}-{month:02d} has unexpected timestamp count")
        key = _month_key(year, month)
        for building_id in out:
            values = raw["values"][building_id][mask]
            finite_nonnegative = np.isfinite(values) & (values >= 0.0)
            valid = bool(finite_nonnegative.all())
            total = float(math.fsum(values[finite_nonnegative].tolist())) if valid else float("nan")
            out[building_id][key] = {
                "valid": valid,
                "count": int(finite_nonnegative.sum()),
                "expected_count": expected,
                "total": total,
                "mean": float(total / expected) if valid else float("nan"),
            }
    return out


def _donor_template(raw: dict[str, Any], selection: dict[str, Any]) -> np.ndarray:
    template_sum = np.zeros((len(SITE_NAMES), 12, 7, 24), dtype=np.float64)
    template_count = np.zeros((len(SITE_NAMES), 12, 7, 24), dtype=np.int64)
    timestamps = raw["timestamps"]
    mask_2016 = (timestamps >= np.datetime64("2016-01-01T00", "h")) & (
        timestamps <= np.datetime64("2016-12-31T23", "h")
    )
    py_datetimes = [datetime.utcfromtimestamp(ts.astype("datetime64[s]").astype(int)) for ts in timestamps[mask_2016]]
    for site in SITE_NAMES:
        site_index = SITE_TO_INT[site]
        for donor_id in selection["donor_ids_by_site"][site]:
            values = raw["values"][donor_id][mask_2016]
            if not (np.isfinite(values).all() and (values >= 0.0).all() and values.sum() > 0.0):
                raise ValueError(f"Selected donor is not complete in 2016: {donor_id}")
            annual_mean = float(values.sum() / EXPECTED_2016_ROWS)
            normalized = values / annual_mean
            for value, ts in zip(normalized, py_datetimes):
                template_sum[site_index, ts.month - 1, ts.weekday(), ts.hour] += float(value)
                template_count[site_index, ts.month - 1, ts.weekday(), ts.hour] += 1
    if np.any(template_count == 0):
        raise ValueError("Donor template has empty calendar cells")
    return (template_sum / template_count).astype(np.float32)


def _build_rows(
    raw: dict[str, Any],
    selection: dict[str, Any],
    monthly: dict[str, dict[str, Any]],
    template: np.ndarray,
) -> dict[str, dict[str, Any]]:
    target_ids = selection["target_ids"]
    target_to_index = selection["target_to_index"]
    template_means = _template_month_means(template)
    splits = {
        "train": TRAIN_MONTHS,
        "validation": VALIDATION_MONTHS,
        "evaluation": EVALUATION_MONTHS,
    }
    return {
        split: _build_split_rows(
            split=split,
            months=months,
            raw=raw,
            target_ids=target_ids,
            target_to_index=target_to_index,
            monthly=monthly,
            template=template,
            template_means=template_means,
        )
        for split, months in splits.items()
    }


def _build_split_rows(
    *,
    split: str,
    months: list[tuple[int, int]],
    raw: dict[str, Any],
    target_ids: list[str],
    target_to_index: dict[str, int],
    monthly: dict[str, dict[str, Any]],
    template: np.ndarray,
    template_means: dict[tuple[int, int, int], float],
) -> dict[str, Any]:
    contexts: list[np.ndarray] = []
    profiles: list[np.ndarray] = []
    totals: list[float] = []
    label_valid: list[bool] = []
    horizons: list[int] = []
    sites: list[int] = []
    target_indices: list[int] = []
    target_names: list[str] = []
    month_names: list[str] = []
    truth: list[np.ndarray] = []

    for year, month in months:
        horizon = _month_hours(year, month)
        origin = _month_start(year, month)
        for target_id in target_ids:
            site = target_id.split("_", 1)[0]
            site_index = SITE_TO_INT[site]
            scale = _target_scale(monthly, target_id)
            context = _context_proxy(
                origin, monthly, target_id, scale, site_index, template, template_means
            )
            profile = _future_profile(
                year, month, origin, monthly, target_id, scale, site_index, template, template_means
            )
            label = monthly[target_id][_month_key(year, month)]
            padded_truth = _fine_truth(raw, target_id, year, month) if split == "evaluation" else None

            contexts.append(context)
            profiles.append(profile)
            totals.append(float(label["total"]) if label["valid"] else float("nan"))
            label_valid.append(bool(label["valid"]))
            horizons.append(horizon)
            sites.append(site_index)
            target_indices.append(target_to_index[target_id])
            target_names.append(target_id)
            month_names.append(_month_key(year, month))
            if padded_truth is not None:
                truth.append(padded_truth)

    result: dict[str, Any] = {
        "context": np.vstack(contexts).astype(np.float32),
        "horizon": np.asarray(horizons, dtype=np.int64),
        "scale": np.asarray([_target_scale(monthly, target_id) for target_id in target_names], dtype=np.float64),
        "site": np.asarray(sites, dtype=np.int64),
        "target": np.asarray(target_indices, dtype=np.int64),
        "target_id": np.asarray(target_names, dtype="<U128"),
        "month": np.asarray(month_names, dtype="<U7"),
        "profile": np.vstack(profiles).astype(np.float32),
        "total": np.asarray(totals, dtype=np.float64),
        "label_valid": np.asarray(label_valid, dtype=bool),
    }
    if split == "evaluation":
        result["truth"] = np.vstack(truth).astype(np.float64)
        result["fine_valid_count"] = np.asarray(
            [int(np.isfinite(row[:horizon]).sum()) for row, horizon in zip(result["truth"], horizons)],
            dtype=np.int64,
        )
    return result


def _target_scale(monthly: dict[str, dict[str, Any]], target_id: str) -> float:
    totals = [monthly[target_id][_month_key(2016, month)]["total"] for month in range(1, 13)]
    if not all(np.isfinite(total) for total in totals):
        raise ValueError(f"Target has incomplete 2016 monthly scale: {target_id}")
    scale = float(sum(totals) / EXPECTED_2016_ROWS)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"Target scale must be positive: {target_id}")
    return scale


def _context_proxy(
    origin: np.datetime64,
    monthly: dict[str, dict[str, Any]],
    target_id: str,
    scale: float,
    site_index: int,
    template: np.ndarray,
    template_means: dict[tuple[int, int, int], float],
) -> np.ndarray:
    values = []
    start = origin - np.timedelta64(CONTEXT, "h")
    for offset in range(CONTEXT):
        ts = start + np.timedelta64(offset, "h")
        year, month, day, hour = _parts(ts)
        shape = _template_value(site_index, year, month, day, hour, template, template_means)
        values.append(shape * _known_month_mean(monthly, target_id, year, month, scale))
    return np.asarray(values, dtype=np.float32)


def _future_profile(
    year: int,
    month: int,
    origin: np.datetime64,
    monthly: dict[str, dict[str, Any]],
    target_id: str,
    scale: float,
    site_index: int,
    template: np.ndarray,
    template_means: dict[tuple[int, int, int], float],
) -> np.ndarray:
    horizon = _month_hours(year, month)
    mean = _last_complete_month_mean_before(monthly, target_id, origin, scale)
    values = np.full(MAX_HORIZON, np.nan, dtype=np.float32)
    month_shape = [
        _template_value(
            site_index, *_parts(origin + np.timedelta64(offset, "h")), template, template_means
        )
        for offset in range(horizon)
    ]
    values[:horizon] = np.asarray(month_shape, dtype=np.float32) * float(mean)
    return values


def _fine_truth(raw: dict[str, Any], target_id: str, year: int, month: int) -> np.ndarray:
    horizon = _month_hours(year, month)
    mask = _mask_for_month(raw["timestamps"], year, month)
    values = raw["values"][target_id][mask]
    out = np.full(MAX_HORIZON, np.nan, dtype=np.float64)
    valid = np.isfinite(values) & (values >= 0.0)
    out[:horizon][valid] = values[valid]
    return out


def _parts(ts: np.datetime64) -> tuple[int, int, int, int]:
    py = datetime.utcfromtimestamp(ts.astype("datetime64[s]").astype(int))
    return py.year, py.month, py.day, py.hour


def _template_value(
    site_index: int,
    year: int,
    month: int,
    day: int,
    hour: int,
    template: np.ndarray,
    template_means: Mapping[tuple[int, int, int], float],
) -> float:
    ts = datetime(year, month, day, hour)
    raw_value = float(template[site_index, month - 1, ts.weekday(), hour])
    month_mean = template_means[(site_index, year, month)]
    if not np.isfinite(month_mean) or month_mean <= 0.0:
        raise ValueError("Template monthly mean must be positive")
    return raw_value / month_mean


def _template_month_means(template: np.ndarray) -> dict[tuple[int, int, int], float]:
    means: dict[tuple[int, int, int], float] = {}
    for site_index in range(len(SITE_NAMES)):
        for year in (2016, 2017):
            for month in range(1, 13):
                means[(site_index, year, month)] = _template_month_mean(site_index, year, month, template)
    return means


def _template_month_mean(site_index: int, year: int, month: int, template: np.ndarray) -> float:
    horizon = _month_hours(year, month)
    origin = _month_start(year, month)
    values = []
    for offset in range(horizon):
        y, m, d, h = _parts(origin + np.timedelta64(offset, "h"))
        ts = datetime(y, m, d, h)
        values.append(float(template[site_index, m - 1, ts.weekday(), h]))
    return float(np.mean(values))


def _known_month_mean(
    monthly: dict[str, dict[str, Any]],
    target_id: str,
    year: int,
    month: int,
    fallback_scale: float,
) -> float:
    key = _month_key(year, month)
    if key in monthly[target_id] and monthly[target_id][key]["valid"]:
        return float(monthly[target_id][key]["mean"])
    origin = _month_start(*_next_month(year, month))
    return _last_complete_month_mean_before(monthly, target_id, origin, fallback_scale)


def _last_complete_month_mean_before(
    monthly: dict[str, dict[str, Any]],
    target_id: str,
    origin: np.datetime64,
    fallback_scale: float,
) -> float:
    year, month, _, _ = _parts(origin)
    year, month = (year - 1, 12) if month == 1 else (year, month - 1)
    while (year, month) >= (2016, 1):
        key = _month_key(year, month)
        if key in monthly[target_id] and monthly[target_id][key]["valid"]:
            return float(monthly[target_id][key]["mean"])
        year, month = (year - 1, 12) if month == 1 else (year, month - 1)
    return float(fallback_scale)


def _save_split(path: Path, rows: dict[str, Any], *, include_total: bool) -> None:
    if path.exists():
        raise FileExistsError(f"output already exists: {path}")
    arrays = {
        "context": rows["context"],
        "horizon": rows["horizon"],
        "scale": rows["scale"],
        "site": rows["site"],
        "target_id": rows["target_id"],
        "month": rows["month"],
        "profile": rows["profile"],
    }
    if include_total:
        arrays["total"] = rows["total"]
        arrays["label_valid"] = rows["label_valid"]
    np.savez_compressed(path, **arrays)


def _save_truth(path: Path, rows: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"output already exists: {path}")
    np.savez_compressed(
        path,
        target=rows["truth"],
        horizon=rows["horizon"],
        site=rows["site"],
        target_index=rows["target"],
        target_id=rows["target_id"],
        month=rows["month"],
    )


def _save_donor_template(path: Path, template: np.ndarray, selection: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"output already exists: {path}")
    np.savez_compressed(
        path,
        template=template.astype(np.float32),
        site_names=np.asarray(SITE_NAMES, dtype="<U16"),
        months=np.asarray([f"{month:02d}" for month in range(1, 13)], dtype="<U2"),
        weekdays=np.arange(7, dtype=np.int64),
        hours=np.arange(24, dtype=np.int64),
        donor_ids_by_site=np.asarray(
            ["|".join(selection["donor_ids_by_site"][site]) for site in SITE_NAMES], dtype="<U1024"
        ),
    )


def _validation_gates(rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    validation = _monthly_gate(rows["validation"], "label_valid")
    evaluation = _evaluation_gate(rows["evaluation"])
    return {
        "validation": validation,
        "evaluation": evaluation,
        "scope": (
            "pre-fit structural coverage only: validation monthly-total labels and "
            "evaluation fine-label availability; no forecast score, model output, or model choice"
        ),
    }


def _monthly_gate(rows: dict[str, Any], valid_key: str) -> dict[str, Any]:
    sites: dict[str, Any] = {}
    for site_name, site_index in SITE_TO_INT.items():
        mask = rows["site"] == site_index
        target_ids = sorted(set(rows["target_id"][mask].tolist()))
        valid = rows[valid_key][mask]
        required = min(16, 2 * len(target_ids))
        per_target = {
            target_id: int((valid & (rows["target_id"][mask] == target_id)).sum())
            for target_id in target_ids
        }
        complete = int(valid.sum())
        sites[site_name] = {
            "complete_rows": complete,
            "total_rows": int(mask.sum()),
            "required_complete_rows": required,
            "per_target_complete_rows": per_target,
            "passed": complete >= required and all(count >= 1 for count in per_target.values()),
        }
    return {"passed": all(site["passed"] for site in sites.values()), "sites": sites}


def _evaluation_gate(rows: dict[str, Any]) -> dict[str, Any]:
    row_valid = rows["fine_valid_count"] >= np.ceil(rows["horizon"] * 0.8).astype(np.int64)
    enriched = dict(rows)
    enriched["row_valid"] = row_valid
    gate = _monthly_gate(enriched, "row_valid")
    gate["row_valid_rule"] = "fine_valid_count >= ceil(0.8 * horizon)"
    gate["fine_valid_count_summary"] = {
        "min": int(rows["fine_valid_count"].min()),
        "max": int(rows["fine_valid_count"].max()),
    }
    return gate


def _metadata_record(
    *,
    data_build_contract: dict[str, Any],
    metadata_rows: dict[str, dict[str, str]],
    selection: dict[str, Any],
    raw: dict[str, Any],
    rows: dict[str, dict[str, Any]],
    gates: dict[str, Any],
    output_dir: Path,
    output_files: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    completed = bool(gates["validation"]["passed"] and gates["evaluation"]["passed"])
    split_counts = {
        name: {
            "rows": int(data["context"].shape[0]),
            "label_valid_rows": int(data["label_valid"].sum()) if "label_valid" in data else None,
            "horizon_values": sorted(set(int(x) for x in data["horizon"].tolist())),
        }
        for name, data in rows.items()
    }
    return {
        "completed": completed,
        "decision": "PASS" if completed else "INSUFFICIENT_STRUCTURAL_COVERAGE",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "schema_version": "coarse_supervision_v1",
        "data_build_contract_sha256": sha256(output_dir / "data_build_contract.json"),
        "source_hashes": data_build_contract["input_hashes"],
        "source_code_hashes": data_build_contract["source_hashes"],
        "id_mapping": {
            "site_names": list(SITE_NAMES),
            "site_to_int": SITE_TO_INT,
            "donor_ids_by_site": selection["donor_ids_by_site"],
            "target_ids_by_site": selection["target_ids_by_site"],
            "global_target_ids": selection["target_ids"],
            "target_to_index": selection["target_to_index"],
            "metadata_rows": metadata_rows,
        },
        "grid": {
            "context": CONTEXT,
            "max_horizon": MAX_HORIZON,
            "train_months": [_month_key(*month) for month in TRAIN_MONTHS],
            "validation_months": [_month_key(*month) for month in VALIDATION_MONTHS],
            "evaluation_months": [_month_key(*month) for month in EVALUATION_MONTHS],
            "raw_timestamp_qc": raw["timestamp_qc"],
            "row_order": "month-major, then global target_id order",
            "target_current_month_total_in_evaluation_inputs": False,
            "source_energy_after_2017_06_processed": False,
        },
        "validation_gate": gates,
        "split_counts": split_counts,
        "no_target_fine_stats": True,
        "output_files": output_files,
    }


def _hash_outputs(output_dir: Path) -> dict[str, dict[str, Any]]:
    return {
        filename: {
            "path": str((output_dir / filename).resolve()),
            "sha256": sha256(output_dir / filename),
            "bytes": (output_dir / filename).stat().st_size,
        }
        for filename in OUTPUT_FILES
        if filename != "metadata.json"
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    print(json.dumps(run(args.root, args.output), ensure_ascii=False))


if __name__ == "__main__":
    main()
