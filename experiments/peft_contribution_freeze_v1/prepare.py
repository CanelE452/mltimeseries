"""Prepare Study31 BDG2/Jena four-channel hourly data archives.

Owned scope: data preparation only.  This script reuses the frozen Study20
archive writer after setting the study name and panel starts at runtime.
It does not read model predictions or E performance outputs.
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
from typing import Sequence

import numpy as np

from experiments.peft_external_gap_v1.data import PanelSpec, sha256_file
from experiments.peft_fullft_reference_v3 import data as archive_data

ROOT = Path(__file__).resolve().parents[2]
STUDY = "peft_contribution_freeze_v1"
VERSION = "peft_contribution_freeze_v1.prepare.20260911"

PANEL_STARTS = {"bdg2": datetime(2017, 5, 4), "jena": datetime(2023, 5, 4)}
TARGET_INDICES = (0, 1)
EXPECTED_CHANNELS = 4

BDG2_RAW = ROOT / "data_external/bdg2_coarse_supervision_v1/raw/electricity.csv"
BDG2_METADATA = ROOT / "data_external/bdg2_coarse_supervision_v1/raw/metadata.csv"
BDG2_QC = ROOT / "data_external/bdg2_coarse_supervision_v1/qc.json"
BDG2_CHANNELS = (
    "Eagle_office_Elias",
    "Eagle_office_Elvis",
    "Eagle_office_Flossie",
    "Eagle_office_Francis",
)

JENA_FILES = (
    ROOT / "data/jena_mpi_roof/mpi_roof_2023a.csv",
    ROOT / "data/jena_mpi_roof/mpi_roof_2023b.csv",
)
JENA_CHANNELS = ("p (mbar)", "T (degC)", "Tpot (K)", "Tdew (degC)")

PRIOR_EVALS = {
    "bdg2": [
        {
            "study": "peft_coarse_supervision_v1",
            "source": "_docs/notes/tsfm_topics/04_objective_observation/17_coarse_supervision_results_20260908.md",
            "start": "2017-04-01T00:00:00",
            "end_exclusive": "2017-07-01T00:00:00",
            "note": "BDG2 raw/QC were previously inspected; documented fine E used Apr-Jun 2017 and later energy was reserved.",
        }
    ],
    "jena": [
        {
            "study": "peft_adaptation_scope_v1",
            "source": "runs/peft_adaptation_scope_v1/prepared/manifest.json",
            "start": "2024-09-07T00:00:00",
            "end_exclusive": "2025-01-01T00:00:00",
            "note": "Jena 2024 was previously exposed; Jena 2023 is a different period in the same source family.",
        }
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


def file_record(path: Path) -> dict:
    return {"path": abspath(path), "relative_path": rel(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def require_files(paths: Sequence[Path]) -> None:
    missing = [abspath(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(missing)


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
        "physical_available_gib": status.ullAvailPhys / (1024**3),
        "commit_available_gib": status.ullAvailPageFile / (1024**3),
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


def hourly_qc(timestamps: Sequence[datetime]) -> dict:
    if not timestamps:
        raise ValueError("empty timestamps")
    duplicate_count = len(timestamps) - len(set(timestamps))
    ordered = sorted(timestamps)
    sort_changes = sum(a != b for a, b in zip(timestamps, ordered))
    offstep_count = sum((b - a) != timedelta(hours=1) for a, b in zip(ordered, ordered[1:]))
    if duplicate_count or offstep_count:
        raise ValueError(f"not a complete hourly grid: duplicates={duplicate_count}, offstep={offstep_count}")
    return {
        "rows": len(ordered),
        "start": ordered[0].isoformat(timespec="seconds"),
        "end": ordered[-1].isoformat(timespec="seconds"),
        "frequency_seconds": 3600,
        "duplicate_count": duplicate_count,
        "offstep_count": offstep_count,
        "sort_changes": sort_changes,
    }


def bdg2_selection_qc() -> dict:
    qc = json.loads(BDG2_QC.read_text(encoding="utf-8"))
    eagle_targets = qc["qc"]["selection"]["sites"]["Eagle"]["target_ids"]
    if tuple(eagle_targets[:4]) != BDG2_CHANNELS:
        raise AssertionError("BDG2 channel order no longer matches the QC selected target order")
    return {
        "qc_path": abspath(BDG2_QC),
        "qc_sha256": sha256_file(BDG2_QC),
        "selection_basis": "first four Eagle target_ids from the pre-existing QC selected-target order; no current E performance used",
        "selected_eagle_targets_prefix4": list(eagle_targets[:4]),
        "source_energy_after_2017_06_processed": qc["qc"].get("leakage_contract", {}).get("source_energy_after_2017_06_processed"),
    }


def load_bdg2() -> tuple[PanelSpec, list[datetime], np.ndarray, dict]:
    require_files([BDG2_RAW, BDG2_METADATA, BDG2_QC])
    timestamps: list[datetime] = []
    rows: list[list[float]] = []
    parse_failures = 0
    raw_rows = 0
    with BDG2_RAW.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames = list(reader.fieldnames or [])
        missing = [name for name in ("timestamp", *BDG2_CHANNELS) if name not in fieldnames]
        if missing:
            raise ValueError(f"BDG2 missing required columns: {missing}")
        for row in reader:
            raw_rows += 1
            try:
                timestamp = datetime.strptime(row["timestamp"], "%Y-%m-%d %H:%M:%S")
            except (KeyError, ValueError):
                parse_failures += 1
                continue
            timestamps.append(timestamp)
            rows.append([parse_float(row[name]) for name in BDG2_CHANNELS])
    order = np.argsort(np.asarray(timestamps, dtype="datetime64[s]"))
    timestamps = [timestamps[int(i)] for i in order]
    values = np.asarray(rows, dtype=np.float32)[order]
    metadata = {
        "source_rows": raw_rows,
        "source_columns_count": len(fieldnames),
        "parse_failures": parse_failures,
        "hourly_grid": hourly_qc(timestamps),
        "selection_qc": bdg2_selection_qc(),
        "missing_by_channel": {name: int(np.isnan(values[:, i]).sum()) for i, name in enumerate(BDG2_CHANNELS)},
    }
    spec = PanelSpec(
        name="bdg2",
        source_path=Path("data_external/bdg2_coarse_supervision_v1/raw/electricity.csv"),
        source_url="https://github.com/buds-lab/building-data-genome-project-2",
        download_url="https://media.githubusercontent.com/media/buds-lab/building-data-genome-project-2/9b97ccbe90096aff42ed4fd6493bf7ae692d7118/data/meters/raw/electricity.csv",
        citation="Building Data Genome Project 2 public building electricity meter data.",
        license_note="Public research data; verify upstream redistribution terms before packaging raw files.",
        channels=BDG2_CHANNELS,
        target_indices=TARGET_INDICES,
        parser="streaming csv.DictReader over raw electricity.csv; local-naive hourly timestamps; blank cells to NaN",
        source_notes=(
            "First two fixed Eagle office meters are scored targets.",
            "Next two fixed Eagle office meters are context-only covariates masked from target loss/scoring.",
            "Late-2017 E is nonoverlapping with the documented Apr-Jun 2017 fine E, but source/QC were previously exposed.",
        ),
    )
    return spec, timestamps, values, metadata


def load_jena() -> tuple[PanelSpec, list[datetime], np.ndarray, dict]:
    require_files(list(JENA_FILES))
    timestamps: list[datetime] = []
    rows: list[list[float]] = []
    raw_rows = 0
    parse_failures = 0
    sentinel_count = 0
    headers: list[list[str]] = []
    for path in JENA_FILES:
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
                values = []
                for name in JENA_CHANNELS:
                    value = parse_float(row[name])
                    if math.isfinite(value) and value <= -9999.0 + 1e-6:
                        value = math.nan
                        sentinel_count += 1
                    values.append(value)
                timestamps.append(timestamp)
                rows.append(values)
    order = np.argsort(np.asarray(timestamps, dtype="datetime64[s]"))
    timestamps = [timestamps[int(i)] for i in order]
    values = np.asarray(rows, dtype=np.float32)[order]
    duplicate_exact_rows_removed = 0
    if len(timestamps) != len(set(timestamps)):
        deduped_timestamps: list[datetime] = []
        deduped_rows: list[np.ndarray] = []
        last_timestamp: datetime | None = None
        last_values: np.ndarray | None = None
        for timestamp, value in zip(timestamps, values):
            if last_timestamp == timestamp:
                if last_values is None or not np.array_equal(last_values, value, equal_nan=True):
                    raise ValueError(f"Jena duplicate timestamp has conflicting values: {timestamp!r}")
                duplicate_exact_rows_removed += 1
                continue
            deduped_timestamps.append(timestamp)
            deduped_rows.append(value)
            last_timestamp = timestamp
            last_values = value
        timestamps = deduped_timestamps
        values = np.asarray(deduped_rows, dtype=np.float32)
    metadata = {
        "source_rows": raw_rows,
        "top_of_hour_rows": len(timestamps),
        "parse_failures": parse_failures,
        "raw_headers_consistent": len(headers) == 2 and headers[0] == headers[1],
        "source_columns": headers[0],
        "downsample_rule": "keep only top-of-hour rows; no six-row averaging",
        "sentinel_rule": "values <= -9999 treated as NaN",
        "sentinel_count_selected_channels": sentinel_count,
        "duplicate_exact_top_of_hour_rows_removed": duplicate_exact_rows_removed,
        "hourly_grid": hourly_qc(timestamps),
        "missing_by_channel": {name: int(np.isnan(values[:, i]).sum()) for i, name in enumerate(JENA_CHANNELS)},
    }
    spec = PanelSpec(
        name="jena",
        source_path=Path("data/jena_mpi_roof/mpi_roof_2023a.csv"),
        source_url="https://www.bgc-jena.mpg.de/wetter/",
        download_url="local files data/jena_mpi_roof/mpi_roof_2023a.csv and mpi_roof_2023b.csv",
        citation="Max Planck Institute for Biogeochemistry Jena roof weather observations.",
        license_note="Local research copy; verify upstream redistribution terms before packaging raw files.",
        channels=JENA_CHANNELS,
        target_indices=TARGET_INDICES,
        parser="latin1 csv.DictReader over 2023a+2023b; date format %d.%m.%Y %H:%M:%S; top-of-hour rows only",
        source_notes=(
            "Four-channel order follows the previous Jena parser's first four numeric variables.",
            "First two columns are scored targets; next two are context-only covariates masked from target loss/scoring.",
            "Jena 2024 was previously exposed; this is a nonoverlapping 2023 period in the same source family.",
        ),
    )
    return spec, timestamps, values, metadata


def source_manifest(panel: str, spec: PanelSpec, loader_metadata: dict) -> dict:
    files = (
        [file_record(BDG2_RAW), file_record(BDG2_METADATA), file_record(BDG2_QC)]
        if panel == "bdg2"
        else [file_record(path) for path in JENA_FILES]
    )
    raw_path = BDG2_RAW if panel == "bdg2" else JENA_FILES[0]
    return {
        "panel": panel,
        "source_path": abspath(raw_path),
        "source_sha256": sha256_file(raw_path),
        "raw_source_path": abspath(raw_path),
        "raw_sha256": sha256_file(raw_path),
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


def configure_writer() -> None:
    archive_data.STUDY = STUDY
    archive_data.VERSION = VERSION
    archive_data.PANEL_STARTS = dict(PANEL_STARTS)


def overlaps(a_start: str, a_end: str, b_start: str, b_end: str) -> bool:
    return datetime.fromisoformat(a_start) < datetime.fromisoformat(b_end) and datetime.fromisoformat(b_start) < datetime.fromisoformat(a_end)


def exposure_audit(panel: str, contract: dict) -> dict:
    eval_start, eval_end = contract["boundaries"]["eval"]
    proposed_start = eval_start if isinstance(eval_start, str) else eval_start.isoformat(timespec="seconds")
    proposed_end = eval_end if isinstance(eval_end, str) else eval_end.isoformat(timespec="seconds")
    previous = []
    for item in PRIOR_EVALS[panel]:
        previous.append({
            **item,
            "overlaps_proposed_eval": overlaps(proposed_start, proposed_end, item["start"], item["end_exclusive"]),
        })
    return {
        "proposed_eval": {"start": proposed_start, "end_exclusive": proposed_end},
        "known_previous_evaluations": previous,
        "any_known_eval_overlap": any(item["overlaps_proposed_eval"] for item in previous),
        "freshness_claim_allowed": False,
        "recommended_wording": "nonoverlapping evaluation block from an already inspected local source family; FM pretraining overlap unknown",
    }


def resolve_output(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def summary_from_existing(panel: str, output: Path) -> dict:
    fit_path = output / f"{panel}_fit.npz"
    holdout_path = output / f"{panel}_holdout.npz"
    if not fit_path.exists() or not holdout_path.exists():
        raise FileNotFoundError(f"cannot reuse incomplete {panel} archive pair")
    with np.load(fit_path, allow_pickle=False) as fit:
        manifest = json.loads(fit["manifest_json"].item())
        summary = {
            "panel": panel,
            "dataset": panel,
            "fit_path": rel(fit_path),
            "holdout_path": rel(holdout_path),
            "fit_data_path": rel(fit_path),
            "holdout_data_path": rel(holdout_path),
            "fit_sha256": sha256_file(fit_path),
            "holdout_sha256": sha256_file(holdout_path),
            "fit_data_sha256": sha256_file(fit_path),
            "holdout_data_sha256": sha256_file(holdout_path),
            "fit_shape": list(map(int, fit["target_values"].shape)),
            "holdout_shape": None,
            "origin_counts": manifest["origin_counts"],
            "boundary_indices": manifest["boundary_indices"],
            "target_qc": manifest["qc"]["target_windows"],
            "row_qc": manifest["qc"]["split_rows"],
            "contract": manifest["contract"],
            "source": manifest["source"],
            "channels": [str(value) for value in fit["channels"]],
            "target_indices": fit["target_indices"].astype(np.int64).tolist(),
            "target_columns": manifest["target_columns"],
            "fit_mean": fit["fit_mean"].astype(float).tolist(),
            "fit_std": fit["fit_std"].astype(float).tolist(),
            "fit_median": fit["fit_median"].astype(float).tolist(),
            "context_fallback_median": fit["context_fallback_median"].astype(float).tolist(),
            "reused_existing_partial_archive": True,
        }
    with np.load(holdout_path, allow_pickle=False) as holdout:
        holdout_manifest = json.loads(holdout["manifest_json"].item())
        if holdout_manifest["dataset"] != panel or holdout_manifest["origin_counts"] != summary["origin_counts"]:
            raise AssertionError(f"{panel} existing fit/holdout manifests disagree")
        summary["holdout_shape"] = list(map(int, holdout["target_values"].shape))
    return summary


def validate_archives(panel: str, summary: dict) -> dict:
    out = {}
    for role, required, forbidden, expected_shape in (
        ("fit", ("train_origins", "val_origins"), ("cal_origins", "eval_origins"), (3312, EXPECTED_CHANNELS)),
        ("holdout", ("cal_origins", "eval_origins"), ("train_origins", "val_origins"), (5808, EXPECTED_CHANNELS)),
    ):
        path = resolve_output(summary[f"{role}_path"])
        if sha256_file(path) != summary[f"{role}_sha256"]:
            raise AssertionError(f"{panel} {role} hash mismatch")
        with np.load(path, allow_pickle=False) as z:
            manifest = json.loads(z["manifest_json"].item())
            if manifest["dataset"] != panel or manifest["panel"] != panel:
                raise AssertionError(f"{panel} {role} manifest panel mismatch")
            if z["target_values"].shape != expected_shape or z["context_values"].shape != expected_shape:
                raise AssertionError(f"{panel} {role} shape mismatch")
            if z["channels"].shape != (EXPECTED_CHANNELS,) or z["target_indices"].astype(np.int64).tolist() != [0, 1]:
                raise AssertionError(f"{panel} {role} channel/target contract mismatch")
            if any(name not in z.files for name in required) or any(name in z.files for name in forbidden):
                raise AssertionError(f"{panel} {role} origin role leakage")
            target = z["target_values"].astype(np.float32)
            expected_loss = np.zeros_like(target, dtype=bool)
            expected_loss[:, z["target_indices"].astype(np.int64)] = np.isfinite(target[:, z["target_indices"].astype(np.int64)])
            if not np.array_equal(z["target_loss_mask"].astype(bool), expected_loss):
                raise AssertionError(f"{panel} {role} target_loss_mask mismatch")
            if not np.array_equal(z["observed_mask"].astype(bool), np.isfinite(target)):
                raise AssertionError(f"{panel} {role} observed_mask mismatch")
            train_start, train_end = manifest["boundary_indices"]["train"]
            train = target[int(train_start):int(train_end)].astype(np.float64)
            mean = np.nanmean(train, axis=0).astype(np.float32)
            std = np.nanstd(train, axis=0).astype(np.float32)
            median = np.nanmedian(train, axis=0).astype(np.float32)
            stats_error = {
                "fit_mean_max_abs_error": float(np.max(np.abs(mean - z["fit_mean"].astype(np.float32)))),
                "fit_std_max_abs_error": float(np.max(np.abs(std - z["fit_std"].astype(np.float32)))),
                "fit_median_max_abs_error": float(np.max(np.abs(median - z["fit_median"].astype(np.float32)))),
            }
            if max(stats_error.values()) > 1e-6:
                raise AssertionError(f"{panel} {role} train-only stats mismatch {stats_error}")
            pre_bounds = tuple(int(v) for v in manifest["boundary_indices"]["precontext"])
            context_expected, fallback_expected = archive_data.causal_context_values(target, pre_bounds)
            context_error = float(np.max(np.abs(context_expected - z["context_values"].astype(np.float32))))
            fallback_error = float(np.max(np.abs(fallback_expected - z["context_fallback_median"].astype(np.float32))))
            if context_error > 0 or fallback_error > 0 or not np.isfinite(z["context_values"]).all():
                raise AssertionError(f"{panel} {role} causal context replay mismatch")
            times = [datetime.fromisoformat(str(value)) for value in z["timestamps"]]
            hourly_qc(times)
            out[role] = {
                "path": abspath(path),
                "sha256": sha256_file(path),
                "shape": list(map(int, target.shape)),
                "included_origins": list(manifest["included_origins"]),
                "origin_counts": {name.replace("_origins", ""): int(len(z[name])) for name in required},
                "stats_replay": stats_error,
                "context_replay": {"context_values_max_abs_error": context_error, "context_fallback_max_abs_error": fallback_error},
                "masks_verified": True,
                "hourly_grid_verified": True,
            }
    return out


def dataset_record(summary: dict) -> dict:
    fit_path = resolve_output(summary["fit_path"])
    holdout_path = resolve_output(summary["holdout_path"])
    contract = summary["contract"]
    source = summary["source"]
    return {
        "dataset": summary["dataset"],
        "panel": summary["panel"],
        "fit_data": {"path": abspath(fit_path), "sha256": summary["fit_sha256"]},
        "holdout_data": {"path": abspath(holdout_path), "sha256": summary["holdout_sha256"]},
        "fit_data_path": abspath(fit_path),
        "holdout_data_path": abspath(holdout_path),
        "fit_data_sha256": summary["fit_sha256"],
        "holdout_data_sha256": summary["holdout_sha256"],
        "source": {"path": source["source_path"], "sha256": source["source_sha256"], "files": source["source_files"]},
        "provenance": {
            "source_url": source["source_url"],
            "download_url": source["download_url"],
            "citation": source["citation"],
            "license_note": source["license_note"],
            "parser": source["parser"],
            "source_notes": source["source_notes"],
            "previous_exposure": exposure_audit(summary["panel"], contract),
        },
        "start": contract["block_start"],
        "end_exclusive": contract["block_end_exclusive"],
        "boundaries": contract["boundaries"],
        "origin_counts": summary["origin_counts"],
        "boundary_indices": summary["boundary_indices"],
        "channels": summary["channels"],
        "target_indices": summary["target_indices"],
        "target_columns": summary["target_columns"],
        "fit_shape": summary["fit_shape"],
        "holdout_shape": summary["holdout_shape"],
        "target_qc": summary["target_qc"],
        "row_qc": summary["row_qc"],
        "fit_mean": summary["fit_mean"],
        "fit_std": summary["fit_std"],
        "fit_median": summary["fit_median"],
        "context_fallback_median": summary["context_fallback_median"],
    }


def prepare(output: Path | str = ROOT / "runs/peft_contribution_freeze_v1/prepared") -> dict:
    configure_writer()
    output = Path(output).resolve()
    run_dir = output.parent
    run_dir.mkdir(parents=True, exist_ok=True)
    if output.exists() and any(output.iterdir()):
        allowed_partial = {f"{panel}_{role}.npz" for panel in ("bdg2", "jena") for role in ("fit", "holdout")}
        existing_names = {item.name for item in output.iterdir()}
        if not existing_names.issubset(allowed_partial):
            raise FileExistsError(f"refusing to overwrite existing prepared dir: {output}")
    output.mkdir(parents=True, exist_ok=True)
    audit_path = run_dir / "data_audit.json"
    summary_path = output / "summary.json"
    manifest_path = output / "manifest.json"
    if audit_path.exists() or summary_path.exists() or manifest_path.exists():
        raise FileExistsError("refusing to overwrite existing Study31 data outputs")

    resources_before = require_cpu_resources()
    loaders = {"bdg2": load_bdg2, "jena": load_jena}
    panels = {}
    datasets = {}
    validations = {}
    all_qc_passed = True
    for panel in ("bdg2", "jena"):
        existing_fit = output / f"{panel}_fit.npz"
        existing_holdout = output / f"{panel}_holdout.npz"
        if existing_fit.exists() or existing_holdout.exists():
            if not (existing_fit.exists() and existing_holdout.exists()):
                raise FileExistsError(f"refusing to overwrite incomplete existing archive pair for {panel}")
            summary = summary_from_existing(panel, output)
        else:
            spec, timestamps, values, loader_metadata = loaders[panel]()
            if len(spec.channels) != EXPECTED_CHANNELS:
                raise AssertionError(f"{panel} is not four-channel")
            summary = archive_data.write_panel_archives(
                panel=panel,
                timestamps=timestamps,
                raw_values=values,
                channels=spec.channels,
                target_indices=spec.target_indices,
                output_dir=output,
                source=source_manifest(panel, spec, loader_metadata),
            )
        panels[panel] = summary
        datasets[panel] = dataset_record(summary)
        validations[panel] = validate_archives(panel, summary)
        for split_qc in summary["target_qc"].values():
            for metrics in split_qc.values():
                all_qc_passed = all_qc_passed and bool(metrics["pass_min_70pct"])

    fixed_blocks = {panel: archive_data._contract_json(archive_data.temporal_contract(panel)) for panel in ("bdg2", "jena")}
    summary_payload = {
        "version": VERSION,
        "study": STUDY,
        "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "domain_context": {
            "domain": "hourly multivariate time-series forecasting",
            "downstream_decision": "single-trajectory LoRA contribution/freezing diagnostic",
            "quality_bar": "development ML; temporal leakage zero tolerance; scored target finite window fraction >=70%",
        },
        "all_qc_passed": all_qc_passed,
        "datasets": datasets,
        "panels": panels,
        "data_contract": {
            "context": archive_data.CONTEXT,
            "horizon": archive_data.HORIZON,
            "origin_stride_hours": archive_data.ORIGIN_STRIDE_HOURS,
            "expected_origin_counts": archive_data.EXPECTED_ORIGIN_COUNTS,
            "fixed_blocks": fixed_blocks,
            "fit_archives": "train+validation only; physically truncated at validation end; no cal/eval origin arrays",
            "holdout_archives": "calibration+evaluation origins only; read only after selection is frozen",
            "score_columns": "target_indices [0,1] only",
            "missing_policy": "target_values preserve raw NaNs; context_values use precontext-only median fallback and causal forward fill",
            "fit_statistics": "mean/std/median fit from observed train rows only",
            "source_freshness": "not wholly unseen; known previous E windows are audited for nonoverlap only",
        },
        "resource_gate_before_prepare": resources_before,
    }
    manifest_sha = write_json_x(manifest_path, summary_payload)
    summary_sha = write_json_x(summary_path, summary_payload)
    audit_payload = {
        "version": VERSION,
        "study": STUDY,
        "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "scope": "data preparation only; no GPU/training/prediction/E performance read",
        "outputs": {
            "prepared_dir": abspath(output),
            "manifest": {"path": abspath(manifest_path), "sha256": manifest_sha},
            "summary": {"path": abspath(summary_path), "sha256": summary_sha},
            "data_audit": {"path": abspath(audit_path)},
        },
        "source_hashes": {
            "bdg2_raw": sha256_file(BDG2_RAW),
            "bdg2_metadata": sha256_file(BDG2_METADATA),
            "bdg2_qc": sha256_file(BDG2_QC),
            "jena_2023a": sha256_file(JENA_FILES[0]),
            "jena_2023b": sha256_file(JENA_FILES[1]),
        },
        "prepared_hashes": {name: {"fit": item["fit_data"]["sha256"], "holdout": item["holdout_data"]["sha256"]} for name, item in datasets.items()},
        "validations": validations,
        "leakage_checks": {name: item["provenance"]["previous_exposure"] for name, item in datasets.items()},
        "resource_gate_before_prepare": resources_before,
        "resource_snapshot_after_prepare": resource_snapshot(),
        "decision": "PASS" if all_qc_passed else "FAIL",
    }
    audit_sha = write_json_x(audit_path, audit_payload)
    return {
        **summary_payload,
        "manifest_path": abspath(manifest_path),
        "manifest_sha256": manifest_sha,
        "summary_path": abspath(summary_path),
        "summary_sha256": summary_sha,
        "data_audit_path": abspath(audit_path),
        "data_audit_sha256": audit_sha,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/peft_contribution_freeze_v1/prepared")
    args = parser.parse_args(argv)
    result = prepare(args.output)
    print(json.dumps({
        "completed": True,
        "study": STUDY,
        "all_qc_passed": result["all_qc_passed"],
        "manifest_path": result["manifest_path"],
        "manifest_sha256": result["manifest_sha256"],
        "summary_path": result["summary_path"],
        "summary_sha256": result["summary_sha256"],
        "data_audit_path": result["data_audit_path"],
        "data_audit_sha256": result["data_audit_sha256"],
        "datasets": {
            name: {
                "fit_data": item["fit_data"],
                "holdout_data": item["holdout_data"],
                "origin_counts": item["origin_counts"],
                "target_columns": item["target_columns"],
                "start": item["start"],
                "end_exclusive": item["end_exclusive"],
            }
            for name, item in result["datasets"].items()
        },
    }, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
