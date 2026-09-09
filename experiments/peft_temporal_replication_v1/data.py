"""Prepare the non-overlapping temporal replication block for study 13.

This module reuses the study-12 source loaders and archive schema, but crops
each source to the next 190-day block: first complete day + 190 days through
first complete day + 380 days.  The resulting NPZ contract intentionally
matches ``peft_external_gap_v1`` so the same runner can consume it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Sequence

import numpy as np

from experiments.peft_external_gap_v1 import data as base_data


VERSION = "peft_temporal_replication_v1.real.20260908"
STUDY = "peft_temporal_replication_v1"
PREVIOUS_STUDY = "peft_external_gap_v1"
PLAN = "_docs/notes/tsfm_topics/13_peft_temporal_replication_plan_20260908.md"
PREVIOUS_PLAN = "_docs/notes/tsfm_topics/12_peft_external_gap_plan_20260908.md"

CONTEXT = base_data.CONTEXT
HORIZON = base_data.HORIZON
ORIGIN_STRIDE_HOURS = base_data.ORIGIN_STRIDE_HOURS
SPLIT_DAYS = dict(base_data.SPLIT_DAYS)
MIN_TARGET_FINITE_FRACTION = base_data.MIN_TARGET_FINITE_FRACTION
QUANTILES = base_data.QUANTILES
REPLICATION_OFFSET_DAYS = 190
BLOCK_DAYS = 190

EXPECTED_BOUNDARIES = {
    "bike": {
        "precontext": ["2011-07-10T00:00:00", "2011-07-24T00:00:00"],
        "train": ["2011-07-24T00:00:00", "2011-09-26T00:00:00"],
        "val": ["2011-09-26T00:00:00", "2011-10-10T00:00:00"],
        "cal": ["2011-10-10T00:00:00", "2011-10-24T00:00:00"],
        "eval": ["2011-10-24T00:00:00", "2012-01-16T00:00:00"],
    },
    "household": {
        "precontext": ["2007-06-25T00:00:00", "2007-07-09T00:00:00"],
        "train": ["2007-07-09T00:00:00", "2007-09-11T00:00:00"],
        "val": ["2007-09-11T00:00:00", "2007-09-25T00:00:00"],
        "cal": ["2007-09-25T00:00:00", "2007-10-09T00:00:00"],
        "eval": ["2007-10-09T00:00:00", "2008-01-01T00:00:00"],
    },
}


def sha256_file(path: Path | str) -> str:
    return base_data.sha256_file(path)


def _utc_now() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _index_for(timestamps: Sequence[datetime], value: datetime) -> int:
    try:
        return {timestamp: index for index, timestamp in enumerate(timestamps)}[value]
    except KeyError as error:
        raise ValueError(f"timestamp boundary is not present in the complete hourly grid: {value!r}") from error


def crop_temporal_replication_block(
    timestamps: Sequence[datetime],
    raw_values: np.ndarray,
    *,
    offset_days: int = REPLICATION_OFFSET_DAYS,
    block_days: int = BLOCK_DAYS,
) -> tuple[list[datetime], np.ndarray, dict]:
    """Return the fixed non-overlapping replication block.

    ``timestamps`` must already be a complete hourly grid, with missing source
    hours represented as NaN rows in ``raw_values``.
    """

    raw_values = np.asarray(raw_values, dtype=np.float32)
    if raw_values.ndim != 2 or len(timestamps) != raw_values.shape[0]:
        raise ValueError("Expected timestamps length to match raw_values [T, C]")
    if any(timestamps[index] >= timestamps[index + 1] for index in range(len(timestamps) - 1)):
        raise ValueError("timestamps must be strictly increasing")
    if any((timestamps[index + 1] - timestamps[index]) != timedelta(hours=1) for index in range(len(timestamps) - 1)):
        raise ValueError("timestamps must be a complete hourly grid")

    first_complete = base_data._first_complete_day_start(timestamps)
    previous_start = first_complete
    previous_end = previous_start + timedelta(days=offset_days)
    replication_start = previous_end
    replication_end = replication_start + timedelta(days=block_days)
    start_index = _index_for(timestamps, replication_start)
    end_index = _index_for(timestamps, replication_end)
    if end_index - start_index != block_days * 24:
        raise ValueError("replication crop length does not match the fixed 190-day block")
    if previous_end != replication_start:
        raise AssertionError("replication block must start exactly where the previous block ended")

    cropped_timestamps = list(timestamps[start_index:end_index])
    cropped_values = raw_values[start_index:end_index].copy()
    metadata = {
        "replication_index": 1,
        "base_first_complete_day": first_complete.isoformat(timespec="seconds"),
        "previous_block": [previous_start.isoformat(timespec="seconds"), previous_end.isoformat(timespec="seconds")],
        "replication_block": [replication_start.isoformat(timespec="seconds"), replication_end.isoformat(timespec="seconds")],
        "replication_start": replication_start.isoformat(timespec="seconds"),
        "replication_end_exclusive": replication_end.isoformat(timespec="seconds"),
        "offset_days_from_first_complete_day": int(offset_days),
        "block_days": int(block_days),
        "raw_timestep_overlap_with_previous_block": 0,
        "source_start_index": int(start_index),
        "source_end_index_exclusive": int(end_index),
    }
    return cropped_timestamps, cropped_values, metadata


def _with_temporal_version_write(*args, **kwargs) -> dict:
    previous_version = base_data.VERSION
    base_data.VERSION = VERSION
    try:
        return base_data.write_panel_archives(*args, **kwargs)
    finally:
        base_data.VERSION = previous_version


def write_replication_panel_archives(
    panel: str,
    timestamps: Sequence[datetime],
    raw_values: np.ndarray,
    channels: Sequence[str],
    target_indices: Sequence[int],
    output_dir: Path | str,
    source: dict,
) -> dict:
    cropped_timestamps, cropped_values, replication = crop_temporal_replication_block(timestamps, raw_values)
    source_with_replication = dict(source)
    source_with_replication["temporal_replication"] = replication
    summary = _with_temporal_version_write(
        panel=panel,
        timestamps=cropped_timestamps,
        raw_values=cropped_values,
        channels=channels,
        target_indices=target_indices,
        output_dir=output_dir,
        source=source_with_replication,
    )
    summary["replication"] = replication
    return summary


def _hash_existing(path: Path, root: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return {"path": str(path.relative_to(root) if path.is_relative_to(root) else path), "sha256": sha256_file(path)}


def previous_study_references(project_root: Path | str = Path(".")) -> dict:
    root = Path(project_root).resolve()
    previous_prepared = root / "runs" / PREVIOUS_STUDY / "prepared"
    previous_manifest_path = previous_prepared / "manifest.json"
    previous_manifest = json.loads(previous_manifest_path.read_text(encoding="utf-8"))

    protected = {
        "source_data_py": _hash_existing(root / "experiments" / PREVIOUS_STUDY / "data.py", root),
        "source_raw_py": _hash_existing(root / "experiments" / PREVIOUS_STUDY / "raw.py", root),
        "prepared_manifest": _hash_existing(previous_manifest_path, root),
    }
    for key, info in previous_manifest["files"].items():
        protected[f"prepared_{key}"] = _hash_existing(root / info["path"], root)
        if protected[f"prepared_{key}"]["sha256"] != info["sha256"]:
            raise AssertionError(f"previous prepared hash mismatch for {key}")

    results_dir = root / "results" / PREVIOUS_STUDY
    result_files = {}
    if results_dir.exists():
        for file in sorted(path for path in results_dir.rglob("*") if path.is_file()):
            result_files[file.relative_to(root).as_posix()] = sha256_file(file)

    return {
        "study": PREVIOUS_STUDY,
        "manifest_reported_sha256": sha256_file(previous_manifest_path),
        "protected_files": protected,
        "result_file_hashes": result_files,
        "note": "Previous study source/prepared/result artifacts were hashed for preservation only; performance contents were not parsed.",
    }


def _expected_boundaries_match(panel: str, summary: dict) -> None:
    expected = EXPECTED_BOUNDARIES.get(panel)
    if expected is None:
        return
    actual = summary["contract"]["boundaries"]
    if actual != expected:
        raise AssertionError(f"{panel} replication boundaries changed: {actual!r} != {expected!r}")


def build_panel(panel: str) -> tuple[base_data.PanelSpec, list[datetime], np.ndarray, dict]:
    return base_data.build_panel(panel)


def prepare(output_dir: Path | str = Path("runs/peft_temporal_replication_v1/prepared"), project_root: Path | str = Path(".")) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite existing manifest: {manifest_path}")

    root = Path(project_root).resolve()
    previous = previous_study_references(root)
    panels = {}
    files = {}
    all_qc_passed = True
    for panel in ("bike", "household"):
        spec, timestamps, values, source_qc = build_panel(panel)
        source = base_data._source_manifest(spec)
        source["source_qc"] = source_qc
        summary = write_replication_panel_archives(
            panel=panel,
            timestamps=timestamps,
            raw_values=values,
            channels=spec.channels,
            target_indices=spec.target_indices,
            output_dir=output_dir,
            source=source,
        )
        _expected_boundaries_match(panel, summary)
        panels[panel] = summary
        files[f"{panel}_fit"] = {"path": summary["fit_path"], "sha256": summary["fit_sha256"]}
        files[f"{panel}_holdout"] = {"path": summary["holdout_path"], "sha256": summary["holdout_sha256"]}
        for per_split in summary["target_qc"].values():
            for metrics in per_split.values():
                all_qc_passed = all_qc_passed and bool(metrics["pass_min_70pct"])

    manifest = {
        "version": VERSION,
        "study": STUDY,
        "created_utc": _utc_now(),
        "all_qc_passed": bool(all_qc_passed),
        "datasets": ["bike", "household"],
        "files": files,
        "panels": panels,
        "previous_study_reference": previous,
        "source_contract": {
            "data_wrapper_path": "experiments/peft_temporal_replication_v1/data.py",
            "data_wrapper_sha256": sha256_file(Path(__file__)),
            "reused_loader_path": "experiments/peft_external_gap_v1/data.py",
            "reused_loader_sha256": sha256_file(Path(base_data.__file__)),
        },
        "data_contract": {
            "fit_archives": "train+val only; no cal/eval origin arrays or holdout labels for selection",
            "holdout_archives": "cal/eval origins and labels; may be read only after method/LR/checkpoint selection is fixed",
            "context_target_alignment": "origin is first target timestamp; context [origin-336, origin), target [origin, origin+48)",
            "temporal_replication": "first complete day + 190 days through +380 days; raw timestamps do not overlap study 12 block",
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
    parser.add_argument("--output", type=Path, default=Path("runs/peft_temporal_replication_v1/prepared"))
    parser.add_argument("--project-root", type=Path, default=Path("."))
    args = parser.parse_args(argv)
    manifest = prepare(args.output, args.project_root)
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
