"""Prepare Study33 decision-transfer data archives for Jena and BMRA.

Owned scope: data preparation only. The script reuses the validated Study20
archive writer after setting study-specific runtime globals. It reads raw
source data and metadata/QC only; it never reads model predictions or forecast
losses from the proposed periods.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
from datetime import datetime, timedelta, timezone
import json
import math
import os
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from experiments.peft_external_gap_v1.data import PanelSpec, sha256_file
from experiments.peft_fullft_reference_v3 import data as archive_data

ROOT = Path(__file__).resolve().parents[2]
STUDY = "peft_decision_transfer_v1"
VERSION = "peft_decision_transfer_v1.prepare.20260911"
TARGET_INDICES = (0, 1)
EXPECTED_CHANNELS = 4
BLOCK_DAYS = int(archive_data.BLOCK_DAYS)
HORIZON = int(archive_data.HORIZON)
CONTEXT = int(archive_data.CONTEXT)
ORIGIN_STRIDE_HOURS = int(archive_data.ORIGIN_STRIDE_HOURS)
MIN_TARGET_FINITE_FRACTION = float(archive_data.MIN_TARGET_FINITE_FRACTION)

EPISODE_STARTS = {
    "dev": {"jena": datetime(2021, 5, 4), "bmra": datetime(2022, 1, 4)},
    "test": {"jena": datetime(2022, 5, 4), "bmra": datetime(2023, 1, 4)},
}

JENA_FILES = {
    "dev": (
        ROOT / "data/jena_mpi_roof/mpi_roof_2021a.csv",
        ROOT / "data/jena_mpi_roof/mpi_roof_2021b.csv",
    ),
    "test": (
        ROOT / "data/jena_mpi_roof/mpi_roof_2022a.csv",
        ROOT / "data/jena_mpi_roof/mpi_roof_2022b.csv",
    ),
}
JENA_CHANNELS = ("p (mbar)", "T (degC)", "Tpot (K)", "Tdew (degC)")

BMRA_DATA = ROOT / "data_external/ucp_path_pilot_v1/extracted/bmra/BMRA_Data/standardFormat/BMRA_data.parquet"
BMRA_METADATA = ROOT / "data_external/ucp_path_pilot_v1/extracted/bmra/BMRA_Data/standardFormat/BMRA_metadata.parquet"
BMRA_CHANNELS = ("E_BNWKW-1", "E_BRYBW-1", "E_BURBO", "E_DALSW-1")

KNOWN_EXPOSURES = {
    "jena": [
        {
            "study": "peft_contribution_freeze_v1",
            "path": "runs/peft_contribution_freeze_v1/prepared/summary.json",
            "start": "2023-05-04T00:00:00",
            "end_exclusive": "2024-01-01T00:00:00",
            "label_use": "train/V/cal/E target labels used in Study31 development diagnostics",
        },
        {
            "study": "peft_adaptation_scope_v1",
            "path": "runs/peft_adaptation_scope_v1/prepared/manifest.json",
            "start": "2024-09-07T00:00:00",
            "end_exclusive": "2025-01-01T00:00:00",
            "label_use": "Jena 2024 validation/evaluation target labels exposed in S1",
        },
    ],
    "bmra": [
        {
            "study": "ucp_path_pilot_v1",
            "path": "data_external/ucp_path_pilot_v1/processed/panel_test.npz",
            "start": "2019-01-01T00:00:00",
            "end_exclusive": "2022-01-01T00:00:00",
            "label_use": "raw source was processed into 2019 train, 2020 val, 2021 test UCP panels",
        },
    ],
}


def rel(path: Path | str) -> str:
    path = Path(path).resolve()
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def abspath(path: Path | str) -> str:
    return str(Path(path).resolve())


def write_json_x(path: Path, payload: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    return sha256_file(path)


def parse_float(value: str | None) -> float:
    if value is None:
        return math.nan
    text = str(value).strip()
    if text in {"", "?", "nan", "NaN", "NA", "null", "None"}:
        return math.nan
    try:
        return float(text)
    except ValueError:
        return math.nan


def require_files(paths: Sequence[Path]) -> None:
    missing = [abspath(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(missing)


def file_record(path: Path) -> dict:
    return {"path": abspath(path), "relative_path": rel(path), "sha256": sha256_file(path), "bytes": int(path.stat().st_size)}


def resource_snapshot() -> dict:
    if os.name != "nt":
        return {"source": "non-windows", "gate_note": "Windows commit gate unavailable here"}

    class MemoryStatusEx(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = MemoryStatusEx()
    status.dwLength = ctypes.sizeof(status)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
    return {
        "source": "GlobalMemoryStatusEx",
        "physical_available_gib": float(status.ullAvailPhys / (1024**3)),
        "commit_available_gib": float(status.ullAvailPageFile / (1024**3)),
        "memory_load_percent": int(status.dwMemoryLoad),
    }


def require_cpu_resources() -> dict:
    snap = resource_snapshot()
    physical = float(snap.get("physical_available_gib", math.inf))
    commit = float(snap.get("commit_available_gib", math.inf))
    if physical < 5.0 or commit < 6.0:
        raise RuntimeError(f"CPU data gate failed: physical={physical:.2f}GiB commit={commit:.2f}GiB")
    snap["gate"] = {"physical_available_gib_min": 5.0, "commit_available_gib_min": 6.0, "passed": True}
    return snap


def block_times(start: datetime) -> list[datetime]:
    return [start + timedelta(hours=i) for i in range(BLOCK_DAYS * 24)]


def block_end(start: datetime) -> datetime:
    return start + timedelta(days=BLOCK_DAYS)


def iso(value: datetime) -> str:
    return value.isoformat(timespec="seconds")


def overlaps(a_start: str, a_end: str, b_start: str, b_end: str) -> bool:
    return datetime.fromisoformat(a_start) < datetime.fromisoformat(b_end) and datetime.fromisoformat(b_start) < datetime.fromisoformat(a_end)


def row_qc(values: np.ndarray, timestamps: Sequence[datetime], channels: Sequence[str]) -> dict:
    return {
        "rows": int(values.shape[0]),
        "start": iso(timestamps[0]),
        "end": iso(timestamps[-1]),
        "end_exclusive": iso(timestamps[-1] + timedelta(hours=1)),
        "frequency_seconds": 3600,
        "missing_cells_by_channel": {name: int(np.isnan(values[:, i]).sum()) for i, name in enumerate(channels)},
        "finite_fraction_by_channel": {name: float(np.isfinite(values[:, i]).mean()) for i, name in enumerate(channels)},
        "nonconstant_by_channel": {
            name: bool(np.unique(values[np.isfinite(values[:, i]), i]).shape[0] > 1) for i, name in enumerate(channels)
        },
    }


def canonicalize_top_hour_rows(records: Iterable[tuple[datetime, Sequence[float]]]) -> tuple[dict[datetime, np.ndarray], dict]:
    by_time: dict[datetime, np.ndarray] = {}
    duplicate_exact = 0
    duplicate_conflict = 0
    for timestamp, row in records:
        value = np.asarray(row, dtype=np.float32)
        if timestamp in by_time:
            if np.array_equal(by_time[timestamp], value, equal_nan=True):
                duplicate_exact += 1
                continue
            duplicate_conflict += 1
            previous = by_time[timestamp]
            conflict = ~(np.isclose(previous, value, equal_nan=True))
            merged = previous.copy()
            merged[conflict] = np.nan
            by_time[timestamp] = merged
            continue
        by_time[timestamp] = value
    ordered = sorted(by_time)
    offstep = sum((b - a) != timedelta(hours=1) for a, b in zip(ordered, ordered[1:]))
    return by_time, {
        "unique_top_hour_rows": len(ordered),
        "duplicate_exact_rows_removed": duplicate_exact,
        "duplicate_conflict_rows_nan_merged": duplicate_conflict,
        "source_offstep_count_after_dedup": int(offstep),
        "source_first_top_hour": iso(ordered[0]) if ordered else None,
        "source_last_top_hour": iso(ordered[-1]) if ordered else None,
    }


def load_jena(episode: str) -> tuple[PanelSpec, list[datetime], np.ndarray, dict]:
    paths = JENA_FILES[episode]
    require_files(paths)
    start = EPISODE_STARTS[episode]["jena"]
    end = block_end(start)
    raw_rows = 0
    top_hour_seen = 0
    parse_failures = 0
    sentinel_count = 0
    headers: list[list[str]] = []
    records: list[tuple[datetime, list[float]]] = []
    for path in paths:
        with path.open("r", encoding="latin1", newline="") as stream:
            reader = csv.DictReader(stream)
            fieldnames = list(reader.fieldnames or [])
            headers.append(fieldnames)
            missing = [name for name in ("Date Time", *JENA_CHANNELS) if name not in fieldnames]
            if missing:
                raise ValueError(f"{path} missing required columns: {missing}")
            for row in reader:
                raw_rows += 1
                try:
                    timestamp = datetime.strptime(row["Date Time"].strip(), "%d.%m.%Y %H:%M:%S")
                except (KeyError, ValueError):
                    parse_failures += 1
                    continue
                if timestamp.minute != 0 or timestamp.second != 0:
                    continue
                top_hour_seen += 1
                values = []
                for name in JENA_CHANNELS:
                    value = parse_float(row[name])
                    if math.isfinite(value) and value <= -9999.0 + 1e-6:
                        value = math.nan
                        sentinel_count += 1
                    values.append(value)
                records.append((timestamp, values))
    by_time, duplicate_qc = canonicalize_top_hour_rows(records)
    timestamps = block_times(start)
    values = np.full((len(timestamps), EXPECTED_CHANNELS), np.nan, dtype=np.float32)
    for i, timestamp in enumerate(timestamps):
        if timestamp in by_time:
            values[i] = by_time[timestamp]
    missing_required_hours = [timestamp for timestamp in timestamps if timestamp not in by_time]
    metadata = {
        "episode": episode,
        "source_files": [file_record(path) for path in paths],
        "source_rows": int(raw_rows),
        "top_of_hour_rows_seen": int(top_hour_seen),
        "parse_failures": int(parse_failures),
        "raw_headers_consistent": bool(len(headers) == len(paths) and all(header == headers[0] for header in headers)),
        "source_columns": headers[0],
        "downsample_rule": "keep only top-of-hour rows; no six-row averaging",
        "sentinel_rule": "values <= -9999 are treated as NaN",
        "sentinel_count_selected_channels": int(sentinel_count),
        "duplicate_rule": "exact duplicate top-of-hour rows are collapsed; conflicting duplicate cells become NaN",
        **duplicate_qc,
        "required_block": {"start": iso(start), "end_exclusive": iso(end), "hours": len(timestamps)},
        "required_missing_hour_count": int(len(missing_required_hours)),
        "required_missing_hour_examples": [iso(t) for t in missing_required_hours[:10]],
        "block_row_qc": row_qc(values, timestamps, JENA_CHANNELS),
    }
    spec = PanelSpec(
        name="jena",
        source_path=Path("data/jena_mpi_roof"),
        source_url="https://www.bgc-jena.mpg.de/wetter/",
        download_url="local files data/jena_mpi_roof/mpi_roof_2021a.csv..mpi_roof_2022b.csv",
        citation="Max Planck Institute for Biogeochemistry Jena roof weather observations.",
        license_note="Local research copy; verify upstream redistribution terms before packaging raw files.",
        channels=JENA_CHANNELS,
        target_indices=TARGET_INDICES,
        parser="latin1 csv.DictReader; date format %d.%m.%Y %H:%M:%S; top-of-hour rows only; required block reindexed hourly with missing hours left as NaN",
        source_notes=(
            "Four-channel order follows Study31 Jena parser.",
            "First two columns are scored targets; next two are context-only covariates masked from target loss/scoring.",
            "2021/2022 periods are new target-label windows under checked local study histories, but source/pretraining exposure is not ruled out.",
        ),
    )
    return spec, timestamps, values, metadata


def load_bmra(episode: str) -> tuple[PanelSpec, list[datetime], np.ndarray, dict]:
    require_files([BMRA_DATA, BMRA_METADATA])
    start = EPISODE_STARTS[episode]["bmra"]
    end = block_end(start)
    table = pq.read_table(BMRA_DATA, columns=["Time", *BMRA_CHANNELS])
    df = table.to_pandas()
    if "Time" not in df.columns:
        df = df.reset_index()
    if "Time" not in df.columns:
        raise ValueError("BMRA Time column unavailable after reset_index")
    df["Time"] = pd.to_datetime(df["Time"])
    duplicate_raw = int(df["Time"].duplicated().sum())
    df = df.set_index("Time").sort_index()
    raw_range = {"start": str(df.index.min()), "end": str(df.index.max())}
    block = df.loc[(df.index >= pd.Timestamp(start)) & (df.index < pd.Timestamp(end)), list(BMRA_CHANNELS)]
    hourly = block.resample("1h").mean()
    expected_index = pd.date_range(pd.Timestamp(start), periods=BLOCK_DAYS * 24, freq="1h")
    hourly = hourly.reindex(expected_index)
    timestamps = [ts.to_pydatetime().replace(tzinfo=None) for ts in expected_index]
    values = hourly.to_numpy(dtype=np.float32)
    metadata_frame = pd.read_parquet(BMRA_METADATA)
    selected_meta = metadata_frame[metadata_frame["BMU ID"].isin(BMRA_CHANNELS)].copy()
    selected_meta["startDate"] = pd.to_datetime(selected_meta["startDate"], errors="coerce").astype(str)
    selected_meta["endDate"] = pd.to_datetime(selected_meta["endDate"], errors="coerce").astype(str)
    metadata = {
        "episode": episode,
        "source_files": [file_record(BMRA_DATA), file_record(BMRA_METADATA)],
        "raw_rows_total": int(pq.ParquetFile(BMRA_DATA).metadata.num_rows),
        "raw_time_range": raw_range,
        "raw_duplicate_timestamp_count": duplicate_raw,
        "raw_frequency": "30min original BMRA Time grid",
        "hourly_aggregation_rule": "mean of same-hour 30min rows using available finite values; hour is NaN for a channel only when all contributing raw values are missing",
        "selection_rule": "fixed from pre-run availability audit: active E_ generator columns, all proposed split parts finite>=0.70 and nonconstant; first four by BMU ID among passing columns; no forecast performance used",
        "selected_metadata": selected_meta.to_dict(orient="records"),
        "required_block": {"start": iso(start), "end_exclusive": iso(end), "hours": len(timestamps)},
        "block_halfhour_rows_seen": int(block.shape[0]),
        "expected_halfhour_rows": int(len(timestamps) * 2),
        "required_missing_hour_count_all_channels": int(hourly.isna().all(axis=1).sum()),
        "required_missing_hour_examples": [str(ts) for ts in hourly.index[hourly.isna().all(axis=1)][:10]],
        "block_row_qc": row_qc(values, timestamps, BMRA_CHANNELS),
    }
    spec = PanelSpec(
        name="bmra",
        source_path=Path("data_external/ucp_path_pilot_v1/extracted/bmra/BMRA_Data/standardFormat/BMRA_data.parquet"),
        source_url="local BMRA standardFormat parquet from ucp_path_pilot_v1",
        download_url="local file data_external/ucp_path_pilot_v1/extracted/bmra/BMRA_Data/standardFormat/BMRA_data.parquet",
        citation="BMRA generation/unit time series as locally prepared for ucp_path_pilot_v1.",
        license_note="Local research copy; verify upstream redistribution terms before packaging raw files.",
        channels=BMRA_CHANNELS,
        target_indices=TARGET_INDICES,
        parser="pyarrow Parquet reader; raw 30min Time grid aggregated to hourly mean; required block reindexed hourly with missing hours left as NaN",
        source_notes=(
            "First two fixed E_ generator columns are scored targets; next two are context-only covariates.",
            "2022/2023 periods are after the checked local UCP 2019/2020/2021 processed panels, but raw/source exposure is not ruled out.",
        ),
    )
    return spec, timestamps, values, metadata


def source_manifest(panel: str, episode: str, spec: PanelSpec, loader_metadata: dict) -> dict:
    files = loader_metadata.get("source_files", [])
    primary = Path(files[0]["path"]) if files else ROOT / spec.source_path
    primary_hash = files[0]["sha256"] if files else sha256_file(primary)
    return {
        "panel": panel,
        "episode": episode,
        "source_path": abspath(primary),
        "source_sha256": primary_hash,
        "raw_source_path": abspath(primary),
        "raw_sha256": primary_hash,
        "source_files": files,
        "source_url": spec.source_url,
        "download_url": spec.download_url,
        "citation": spec.citation,
        "license_note": spec.license_note,
        "parser": spec.parser,
        "source_notes": list(spec.source_notes),
        "channels": list(spec.channels),
        "target_indices": list(spec.target_indices),
        "loader_metadata": loader_metadata,
    }


def configure_writer(episode: str) -> None:
    archive_data.STUDY = STUDY
    archive_data.VERSION = VERSION
    archive_data.PANEL_STARTS = dict(EPISODE_STARTS[episode])


def dataset_manifest(summary: dict) -> dict:
    source = summary["source"]
    return {
        "dataset": summary["dataset"],
        "panel": summary["panel"],
        "fit_path": summary["fit_path"],
        "holdout_path": summary["holdout_path"],
        "fit_sha256": summary["fit_sha256"],
        "holdout_sha256": summary["holdout_sha256"],
        "fit_data_path": summary["fit_data_path"],
        "holdout_data_path": summary["holdout_data_path"],
        "fit_data_sha256": summary["fit_data_sha256"],
        "holdout_data_sha256": summary["holdout_data_sha256"],
        "fit_std": summary["fit_std"],
        "fit_mean": summary["fit_mean"],
        "fit_median": summary["fit_median"],
        "context_fallback_median": summary["context_fallback_median"],
        "target_indices": summary["target_indices"],
        "target_columns": summary["target_columns"],
        "channels": summary["channels"],
        "origin_counts": summary["origin_counts"],
        "boundary_indices": summary["boundary_indices"],
        "boundaries": summary["contract"]["boundaries"],
        "contract": summary["contract"],
        "target_qc": summary["target_qc"],
        "row_qc": summary["row_qc"],
        "source_path": source.get("source_path"),
        "source_sha256": source.get("source_sha256"),
        "raw_source_path": source.get("raw_source_path", source.get("source_path")),
        "raw_sha256": source.get("raw_sha256", source.get("source_sha256")),
        "source": source,
    }


def exposure_audit(panel: str, contract: dict) -> dict:
    proposed_start = contract["block_start"]
    proposed_end = contract["block_end_exclusive"]
    previous = []
    for item in KNOWN_EXPOSURES[panel]:
        previous.append({**item, "overlaps_proposed_block": overlaps(proposed_start, proposed_end, item["start"], item["end_exclusive"])})
    return {
        "proposed_block": {"start": proposed_start, "end_exclusive": proposed_end},
        "known_previous_target_label_exposures_checked": previous,
        "any_known_overlap": any(item["overlaps_proposed_block"] for item in previous),
        "freshness_claim": "new target-label period under checked local studies only; not a new raw source and FM pretraining overlap is unknown",
    }


def min_target_finite(summary: dict) -> float:
    values = []
    for split in summary["target_qc"].values():
        for metrics in split.values():
            values.append(float(metrics["finite_fraction"]))
    return float(min(values))


def independent_archive_check(path: Path, panel: str, role: str) -> dict:
    with np.load(path, allow_pickle=False) as z:
        manifest = json.loads(z["manifest_json"].item())
        if manifest["dataset"] != panel or manifest["panel"] != panel:
            raise AssertionError(f"{path} dataset/panel mismatch")
        if z["context_values"].shape != z["target_values"].shape:
            raise AssertionError(f"{path} context/target shape mismatch")
        if z["context_values"].shape[1] != EXPECTED_CHANNELS:
            raise AssertionError(f"{path} channel count mismatch")
        if int(z["context"]) != CONTEXT or int(z["horizon"]) != HORIZON:
            raise AssertionError(f"{path} context/horizon mismatch")
        target_indices = z["target_indices"].astype(np.int64)
        if target_indices.tolist() != [0, 1]:
            raise AssertionError(f"{path} target index mismatch")
        expected_loss = np.zeros_like(z["target_values"], dtype=bool)
        expected_loss[:, target_indices] = np.isfinite(z["target_values"][:, target_indices])
        if not np.array_equal(z["target_loss_mask"].astype(bool), expected_loss):
            raise AssertionError(f"{path} target_loss_mask mismatch")
        if not np.array_equal(z["observed_mask"].astype(bool), np.isfinite(z["target_values"])):
            raise AssertionError(f"{path} observed_mask mismatch")
        if not np.isfinite(z["context_values"]).all():
            raise AssertionError(f"{path} context contains non-finite values")
        present = set(z.files)
        if role == "fit":
            required, forbidden = {"train_origins", "val_origins"}, {"cal_origins", "eval_origins"}
            expected_rows = (archive_data.SPLIT_DAYS["precontext"] + archive_data.SPLIT_DAYS["train"] + archive_data.SPLIT_DAYS["embargo_train_val"] + archive_data.SPLIT_DAYS["val"]) * 24
        else:
            required, forbidden = {"cal_origins", "eval_origins"}, {"train_origins", "val_origins"}
            expected_rows = BLOCK_DAYS * 24
        if not required.issubset(present) or forbidden.intersection(present):
            raise AssertionError(f"{path} origin role leakage")
        if z["target_values"].shape[0] != expected_rows:
            raise AssertionError(f"{path} row count mismatch")
        origin_counts = {name.removesuffix("_origins"): int(len(z[name])) for name in sorted(required)}
        return {
            "path": rel(path),
            "sha256": sha256_file(path),
            "role": role,
            "shape": list(map(int, z["target_values"].shape)),
            "origin_counts": origin_counts,
            "manifest_archive_role": manifest["archive_role"],
        }


def chronology_audit(contract: dict) -> dict:
    first = contract["first_origin"]
    last_end = contract["last_target_end_exclusive"]
    pairs = [("train", "val"), ("val", "cal"), ("cal", "eval")]
    between = {}
    for left, right in pairs:
        between[f"{left}_to_{right}"] = {
            "left_last_target_end_exclusive": last_end[left],
            "right_first_origin": first[right],
            "target_windows_overlap": datetime.fromisoformat(last_end[left]) > datetime.fromisoformat(first[right]),
            "gap_hours": int((datetime.fromisoformat(first[right]) - datetime.fromisoformat(last_end[left])).total_seconds() // 3600),
        }
    return {
        "block_start": contract["block_start"],
        "block_end_exclusive": contract["block_end_exclusive"],
        "boundaries": contract["boundaries"],
        "first_origin": first,
        "last_origin": contract["last_origin"],
        "last_target_end_exclusive": last_end,
        "origin_counts": contract["origin_counts"],
        "within_split_target_overlap_hours": int(contract["target_window_overlap_hours"]),
        "between_split_target_overlap": between,
    }


def prepare(output_dir: Path | str) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    audit_path = output_dir / "data_audit.json"
    if summary_path.exists() or audit_path.exists():
        raise FileExistsError(f"refusing to overwrite existing summary/audit in {output_dir}")

    resource = require_cpu_resources()
    data: dict[str, dict[str, dict]] = {"dev": {}, "test": {}}
    panels: dict[str, dict] = {"dev": {}, "test": {}}
    audits: dict[str, dict] = {"dev": {}, "test": {}}
    files: dict[str, dict] = {}
    all_qc_passed = True

    for episode in ("dev", "test"):
        configure_writer(episode)
        episode_dir = output_dir / episode
        episode_dir.mkdir(parents=True, exist_ok=True)
        loaders = {"jena": load_jena, "bmra": load_bmra}
        for panel, loader in loaders.items():
            spec, timestamps, values, loader_metadata = loader(episode)
            source = source_manifest(panel, episode, spec, loader_metadata)
            summary = archive_data.write_panel_archives(panel, timestamps, values, spec.channels, spec.target_indices, episode_dir, source)
            spec_summary = dataset_manifest(summary)
            data[episode][panel] = spec_summary
            panels[episode][panel] = summary
            files[f"{episode}_{panel}_fit"] = {"path": summary["fit_path"], "sha256": summary["fit_sha256"]}
            files[f"{episode}_{panel}_holdout"] = {"path": summary["holdout_path"], "sha256": summary["holdout_sha256"]}
            min_finite = min_target_finite(summary)
            all_qc_passed = all_qc_passed and bool(min_finite >= MIN_TARGET_FINITE_FRACTION)
            audits[episode][panel] = {
                "source_qc": loader_metadata,
                "exposure": exposure_audit(panel, summary["contract"]),
                "chronology": chronology_audit(summary["contract"]),
                "min_target_window_finite_fraction": min_finite,
                "archive_checks": {
                    "fit": independent_archive_check(ROOT / summary["fit_path"], panel, "fit"),
                    "holdout": independent_archive_check(ROOT / summary["holdout_path"], panel, "holdout"),
                },
            }

    data_contract = {
        "fit_archives": "train+validation only; physically truncated at validation end with no cal/eval origin arrays",
        "holdout_archives": "full 242-day block with calibration/evaluation origins; read only after selection gates allow forecast stage",
        "context_target_alignment": "origin is first target timestamp; context [origin-336h, origin), target [origin, origin+48h)",
        "origin_stride_hours": ORIGIN_STRIDE_HOURS,
        "target_window_overlap_hours_within_split": HORIZON - ORIGIN_STRIDE_HOURS,
        "between_split_target_overlap": "none across train/V and V/cal due 48h embargo; cal last target end equals eval first origin under half-open windows",
        "missing_policy": "target_values preserve raw hourly NaNs; context_values use precontext-only median fallback then causal forward fill; no future backfill and no target imputation",
        "fit_statistics": "mean/std/median fitted from observed train interval rows only",
        "score_columns": "only target_indices [0,1] are scored",
        "forecast_subsample_note": "archives keep all 80 E origins; run.py may predeclaredly forecast eval_origins[::4] for the bounded initial screen",
    }
    summary = {
        "version": VERSION,
        "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "study": STUDY,
        "all_qc_passed": bool(all_qc_passed),
        "data": data,
        "files": files,
        "data_contract": data_contract,
        "exposure_summary": {
            "claim_limit": "new target-label periods under checked local study history; not new raw sources; FM pretraining overlap unknown",
            "selection_limit": "BMRA channel selection used future-window availability/QC only, not forecast performance; claims must disclose this availability screen",
            "checked_previous_sources": KNOWN_EXPOSURES,
        },
        "resource_snapshot": resource,
    }
    summary_sha = write_json_x(summary_path, summary)
    audit = {
        "version": VERSION,
        "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "study": STUDY,
        "summary_path": rel(summary_path),
        "summary_sha256": summary_sha,
        "audits": audits,
        "all_qc_passed": bool(all_qc_passed),
        "source_hashes": {
            rel(path): sha256_file(path)
            for path in [Path(__file__), Path(__file__).with_name("PURPOSE_DATA.md"), Path(__file__).with_name("panel.py")]
            if path.exists()
        },
    }
    audit_sha = write_json_x(audit_path, audit)
    summary["summary_path"] = rel(summary_path)
    summary["summary_sha256"] = summary_sha
    summary["data_audit_path"] = rel(audit_path)
    summary["data_audit_sha256"] = audit_sha
    return summary


def record_failure(output_dir: Path, exc: BaseException) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(rel(path) for path in output_dir.rglob("*") if path.is_file())
    payload = {
        "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "study": STUDY,
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "existing_files_at_failure": existing,
    }
    target = output_dir / "preparation_failures.json"
    if target.exists():
        index = 2
        while (output_dir / f"preparation_failures_{index:02d}.json").exists():
            index += 1
        target = output_dir / f"preparation_failures_{index:02d}.json"
    write_json_x(target, payload)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs") / STUDY / "prepared")
    args = parser.parse_args(argv)
    try:
        summary = prepare(args.output)
    except BaseException as exc:
        record_failure(args.output, exc)
        raise
    print(json.dumps({
        "completed": True,
        "summary_path": summary["summary_path"],
        "summary_sha256": summary["summary_sha256"],
        "data_audit_path": summary["data_audit_path"],
        "data_audit_sha256": summary["data_audit_sha256"],
        "all_qc_passed": summary["all_qc_passed"],
        "data": {
            episode: {
                panel: {
                    "fit_path": spec["fit_path"],
                    "fit_sha256": spec["fit_sha256"],
                    "holdout_path": spec["holdout_path"],
                    "holdout_sha256": spec["holdout_sha256"],
                    "origin_counts": spec["origin_counts"],
                }
                for panel, spec in panels.items()
            }
            for episode, panels in summary["data"].items()
        },
    }, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
