"""Prepare fixed future-block hourly panels for study20.

The archives produced here keep the old study12 Panel NPZ interface while
moving study20 to the predetermined third chronological block.  Fit archives
contain only train/validation origins and rows through the validation boundary.
Holdout archives contain the full block with only calibration/evaluation
origins, so model selection cannot read future labels by opening the fit file.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np

from experiments.peft_external_gap_v1 import data as source_data


ROOT = Path(__file__).resolve().parents[2]
STUDY = "peft_fullft_reference_v2"
VERSION = "peft_fullft_reference_v2.data.20260909"

CONTEXT = 336
HORIZON = 48
ORIGIN_STRIDE_HOURS = 24
PANEL_STARTS = {
    "bike": datetime(2012, 1, 16),
    "household": datetime(2008, 1, 1),
}
SPLIT_DAYS = {
    "precontext": 14,
    "train": 91,
    "embargo_train_val": 2,
    "val": 31,
    "embargo_val_cal": 2,
    "cal": 21,
    "eval": 81,
}
BLOCK_DAYS = sum(SPLIT_DAYS.values())
EXPECTED_ORIGIN_COUNTS = {"train": 90, "val": 30, "cal": 20, "eval": 80}
MIN_TARGET_FINITE_FRACTION = 0.70

QUANTILES = source_data.QUANTILES
BIKE_SPEC = source_data.BIKE_SPEC
HOUSEHOLD_SPEC = source_data.HOUSEHOLD_SPEC
PanelSpec = source_data.PanelSpec
sha256_file = source_data.sha256_file


def _relative(path: Path | str) -> str:
    path = Path(path).resolve()
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)


def _encode_timestamps(timestamps: Sequence[datetime]) -> np.ndarray:
    return np.array([value.isoformat(timespec="seconds") for value in timestamps], dtype=np.str_)


def _manifest_array(value: dict) -> np.ndarray:
    return np.array(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False), dtype=np.str_)


def _float_list(values: Sequence[float] | np.ndarray) -> list[float]:
    return [float(value) for value in np.asarray(values, dtype=np.float64)]


def _hourly_origins(start: datetime, end: datetime) -> list[datetime]:
    last_origin = end - timedelta(hours=HORIZON)
    origins: list[datetime] = []
    current = start
    while current <= last_origin:
        origins.append(current)
        current += timedelta(hours=ORIGIN_STRIDE_HOURS)
    return origins


def temporal_contract(panel: str) -> dict:
    """Return the fixed study20 chronology for one panel."""

    if panel not in PANEL_STARTS:
        raise ValueError(f"unknown panel {panel!r}")
    current = PANEL_STARTS[panel]
    boundaries: dict[str, tuple[datetime, datetime]] = {}
    for name, days in SPLIT_DAYS.items():
        end = current + timedelta(days=days)
        boundaries[name] = (current, end)
        current = end

    origins = {
        split: _hourly_origins(*boundaries[split])
        for split in ("train", "val", "cal", "eval")
    }
    counts = {name: len(values) for name, values in origins.items()}
    if counts != EXPECTED_ORIGIN_COUNTS:
        raise AssertionError(f"fixed origin counts drifted: {counts}")
    return {
        "context": CONTEXT,
        "horizon": HORIZON,
        "origin_stride_hours": ORIGIN_STRIDE_HOURS,
        "target_window_overlap_hours": max(0, HORIZON - ORIGIN_STRIDE_HOURS),
        "block_days": BLOCK_DAYS,
        "block_start": boundaries["precontext"][0],
        "block_end_exclusive": boundaries["eval"][1],
        "boundaries": boundaries,
        "origins": origins,
    }


def _contract_json(contract: dict) -> dict:
    boundaries = contract["boundaries"]
    origins = contract["origins"]
    return {
        "context": int(contract["context"]),
        "horizon": int(contract["horizon"]),
        "origin_stride_hours": int(contract["origin_stride_hours"]),
        "target_window_overlap_hours": int(contract["target_window_overlap_hours"]),
        "block_days": int(contract["block_days"]),
        "block_start": contract["block_start"].isoformat(timespec="seconds"),
        "block_end_exclusive": contract["block_end_exclusive"].isoformat(timespec="seconds"),
        "boundaries": {
            name: [start.isoformat(timespec="seconds"), end.isoformat(timespec="seconds")]
            for name, (start, end) in boundaries.items()
        },
        "split_duration_days": {
            name: int((end - start).total_seconds() // 86400)
            for name, (start, end) in boundaries.items()
        },
        "origin_counts": {name: len(values) for name, values in origins.items()},
        "first_origin": {
            name: values[0].isoformat(timespec="seconds") if values else None
            for name, values in origins.items()
        },
        "last_origin": {
            name: values[-1].isoformat(timespec="seconds") if values else None
            for name, values in origins.items()
        },
        "last_target_end_exclusive": {
            name: (values[-1] + timedelta(hours=HORIZON)).isoformat(timespec="seconds") if values else None
            for name, values in origins.items()
        },
        "between_split_embargo_hours": {
            "train_to_val": int((boundaries["val"][0] - boundaries["train"][1]).total_seconds() // 3600),
            "val_to_cal": int((boundaries["cal"][0] - boundaries["val"][1]).total_seconds() // 3600),
            "cal_to_eval": int((boundaries["eval"][0] - boundaries["cal"][1]).total_seconds() // 3600),
        },
    }


def _require_hourly_grid(timestamps: Sequence[datetime]) -> None:
    if not timestamps:
        raise ValueError("timestamps are empty")
    for left, right in zip(timestamps, timestamps[1:]):
        if right <= left:
            raise ValueError("timestamps must be strictly increasing")
        if right - left != timedelta(hours=1):
            raise ValueError("timestamps must be a complete hourly grid")


def _timestamp_index(timestamps: Sequence[datetime]) -> dict[datetime, int]:
    return {timestamp: index for index, timestamp in enumerate(timestamps)}


def _end_index(index: dict[datetime, int], timestamps: Sequence[datetime], end: datetime) -> int:
    if end in index:
        return index[end]
    if timestamps and end == timestamps[-1] + timedelta(hours=1):
        return len(timestamps)
    raise ValueError(f"split boundary is outside the hourly grid: {end!r}")


def _boundary_indices(timestamps: Sequence[datetime], boundaries: dict[str, tuple[datetime, datetime]]) -> dict[str, tuple[int, int]]:
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


def causal_context_values(values: np.ndarray, precontext_bounds: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Causally fill context values with a precontext-only fallback median."""

    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("values must be [T, C]")
    start, end = (int(precontext_bounds[0]), int(precontext_bounds[1]))
    if not (0 <= start < end <= len(values)):
        raise ValueError("precontext bounds are outside values")
    precontext = values[start:end].astype(np.float64)
    observed_counts = np.isfinite(precontext).sum(axis=0)
    if np.any(observed_counts == 0):
        bad = np.where(observed_counts == 0)[0].tolist()
        raise ValueError(f"precontext has no finite observations for columns {bad}")
    fallback = np.nanmedian(precontext, axis=0).astype(np.float32)
    if not np.isfinite(fallback).all():
        raise ValueError("precontext fallback medians must be finite")

    filled = np.empty_like(values, dtype=np.float32)
    last = fallback.copy()
    for row in range(values.shape[0]):
        observed = np.isfinite(values[row])
        last[observed] = values[row, observed]
        filled[row] = last
    return filled, fallback


def _fit_stats(values: np.ndarray, train_bounds: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train = np.asarray(values[train_bounds[0]:train_bounds[1]], dtype=np.float64)
    finite = np.isfinite(train)
    observed_counts = finite.sum(axis=0)
    if np.any(observed_counts == 0):
        bad = np.where(observed_counts == 0)[0].tolist()
        raise ValueError(f"train interval has no finite observations for columns {bad}")
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
        index = int(target_index)
        mask[:, index] = np.isfinite(raw_values[:, index])
    return mask


def _target_window_qc(
    raw_values: np.ndarray,
    origins: dict[str, np.ndarray],
    channels: Sequence[str],
    target_indices: Sequence[int],
) -> dict:
    qc: dict[str, dict] = {}
    for split, split_origins in origins.items():
        split_qc = {}
        for target_index in target_indices:
            total = int(len(split_origins) * HORIZON)
            valid = 0
            for origin in split_origins:
                valid += int(np.isfinite(raw_values[int(origin):int(origin) + HORIZON, int(target_index)]).sum())
            fraction = valid / total if total else math.nan
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
    context_fallback_median: np.ndarray,
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
        "context_fallback_median": context_fallback_median.astype(np.float32),
        "context": np.array(CONTEXT, dtype=np.int64),
        "horizon": np.array(HORIZON, dtype=np.int64),
        "origin_stride_hours": np.array(ORIGIN_STRIDE_HOURS, dtype=np.int64),
        "quantiles": QUANTILES.astype(np.float64),
        "manifest_json": _manifest_array(archive_manifest),
    }


def source_manifest(spec: PanelSpec) -> dict:
    """Return raw-source provenance using the parser metadata from study12."""

    manifest = source_data._source_manifest(spec)
    manifest["raw_source_path"] = manifest["source_path"]
    manifest["raw_sha256"] = manifest["source_sha256"]
    manifest["reused_parser_namespace"] = "experiments.peft_external_gap_v1.data"
    return manifest


def build_panel(panel: str) -> tuple[PanelSpec, list[datetime], np.ndarray, dict]:
    if panel == "bike":
        loaded = source_data.load_bike_hourly(BIKE_SPEC.source_path)
        return BIKE_SPEC, loaded["timestamps"], loaded["values"], loaded["metadata"]
    if panel == "household":
        loaded = source_data.load_household_hourly(HOUSEHOLD_SPEC.source_path)
        return HOUSEHOLD_SPEC, loaded["timestamps"], loaded["values"], loaded["metadata"]
    raise ValueError(f"unknown panel {panel!r}")


def _validate_archive_inputs(
    panel: str,
    timestamps: Sequence[datetime],
    raw_values: np.ndarray,
    channels: Sequence[str],
    target_indices: Sequence[int],
) -> None:
    if panel not in PANEL_STARTS:
        raise ValueError(f"unknown panel {panel!r}")
    if raw_values.ndim != 2:
        raise ValueError("raw_values must be [T, C]")
    if len(timestamps) != raw_values.shape[0]:
        raise ValueError("timestamps and raw_values length mismatch")
    if len(channels) != raw_values.shape[1]:
        raise ValueError("channels and raw_values column count mismatch")
    if list(target_indices) != [0, 1]:
        raise ValueError("target_indices must be [0, 1] for study20")
    _require_hourly_grid(timestamps)


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

    timestamps = list(timestamps)
    raw_values = np.asarray(raw_values, dtype=np.float32)
    channels = list(channels)
    target_indices = [int(value) for value in target_indices]
    _validate_archive_inputs(panel, timestamps, raw_values, channels, target_indices)

    contract = temporal_contract(panel)
    required_end = contract["boundaries"]["eval"][1]
    if required_end > timestamps[-1] + timedelta(hours=1):
        raise ValueError(f"source is too short for the fixed study20 block ending {required_end.isoformat(timespec='seconds')}")

    full_boundaries = _boundary_indices(timestamps, contract["boundaries"])
    full_origins = {split: _origin_indices(timestamps, values) for split, values in contract["origins"].items()}
    archive_start, archive_end = full_boundaries["precontext"][0], full_boundaries["eval"][1]
    cropped_timestamps = timestamps[archive_start:archive_end]
    cropped_values = raw_values[archive_start:archive_end]
    boundaries = {
        name: (int(start - archive_start), int(end - archive_start))
        for name, (start, end) in full_boundaries.items()
    }
    origins = {split: values - archive_start for split, values in full_origins.items()}

    fit_mean, fit_std, fit_median = _fit_stats(cropped_values, boundaries["train"])
    context_values, context_fallback_median = causal_context_values(cropped_values, boundaries["precontext"])
    if not np.isfinite(context_values).all():
        raise ValueError("context_values must be finite after causal context fill")

    observed_mask = np.isfinite(cropped_values)
    target_loss_mask = _loss_mask(cropped_values, target_indices)
    target_qc = _target_window_qc(cropped_values, origins, channels, target_indices)
    failing_targets = [
        (split, target)
        for split, per_target in target_qc.items()
        for target, metrics in per_target.items()
        if not metrics["pass_min_70pct"]
    ]
    if failing_targets:
        raise ValueError(f"target finite ratio below 70% for {failing_targets}")
    row_qc = _split_row_qc(cropped_values, boundaries, channels)

    contract_for_json = _contract_json(contract)
    base_manifest = {
        "version": VERSION,
        "study": STUDY,
        "panel": panel,
        "dataset": panel,
        "channels": channels,
        "target_indices": target_indices,
        "target_columns": [channels[index] for index in target_indices],
        "context": CONTEXT,
        "horizon": HORIZON,
        "origin_stride_hours": ORIGIN_STRIDE_HOURS,
        "data_semantics": {
            "origin_index": "first target timestamp; context is [origin-context, origin), target is [origin, origin+horizon)",
            "context_values": "causal forward-fill of raw hourly values; leading gaps use medians fit only on the fixed precontext interval before train origins",
            "target_values": "raw hourly values with NaNs preserved",
            "observed_mask": "finite raw cells for every channel",
            "target_loss_mask": "true only for finite raw cells in target columns",
            "fit_statistics": "mean/std/median fit on observed train-period rows only",
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
        cropped_values,
        observed_mask,
        target_loss_mask,
        cropped_timestamps,
        channels,
        target_indices,
        fit_mean,
        fit_std,
        fit_median,
        context_fallback_median,
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
        "holdout_archive_path": _relative(holdout_path),
        "holdout_archive_sha256": holdout_sha,
    }
    fit_arrays = _archive_common_arrays(
        context_values[:fit_end],
        cropped_values[:fit_end],
        observed_mask[:fit_end],
        target_loss_mask[:fit_end],
        cropped_timestamps[:fit_end],
        channels,
        target_indices,
        fit_mean,
        fit_std,
        fit_median,
        context_fallback_median,
        fit_manifest,
    )
    fit_arrays["train_origins"] = origins["train"]
    fit_arrays["val_origins"] = origins["val"]
    np.savez_compressed(fit_path, **fit_arrays)
    fit_sha = sha256_file(fit_path)

    origin_counts = {name: int(len(values)) for name, values in origins.items()}
    boundary_indices = {name: [int(start), int(end)] for name, (start, end) in boundaries.items()}
    return {
        "panel": panel,
        "dataset": panel,
        "fit_path": _relative(fit_path),
        "holdout_path": _relative(holdout_path),
        "fit_data_path": _relative(fit_path),
        "holdout_data_path": _relative(holdout_path),
        "fit_sha256": fit_sha,
        "holdout_sha256": holdout_sha,
        "fit_data_sha256": fit_sha,
        "holdout_data_sha256": holdout_sha,
        "fit_shape": [int(fit_end), int(cropped_values.shape[1])],
        "holdout_shape": [int(cropped_values.shape[0]), int(cropped_values.shape[1])],
        "origin_counts": origin_counts,
        "boundary_indices": boundary_indices,
        "target_qc": target_qc,
        "row_qc": row_qc,
        "contract": contract_for_json,
        "source": source,
        "channels": channels,
        "target_indices": target_indices,
        "target_columns": [channels[index] for index in target_indices],
        "fit_mean": _float_list(fit_mean),
        "fit_std": _float_list(fit_std),
        "fit_median": _float_list(fit_median),
        "context_fallback_median": _float_list(context_fallback_median),
    }


def _dataset_manifest(summary: dict) -> dict:
    source = summary["source"]
    return {
        "dataset": summary["dataset"],
        "panel": summary["panel"],
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
        "source_path": source.get("source_path"),
        "source_sha256": source.get("source_sha256"),
        "raw_source_path": source.get("raw_source_path", source.get("source_path")),
        "raw_sha256": source.get("raw_sha256", source.get("source_sha256")),
    }


def prepare(output_dir: Path | str) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite existing manifest: {manifest_path}")

    panels: dict[str, dict] = {}
    datasets: dict[str, dict] = {}
    files: dict[str, dict] = {}
    all_qc_passed = True
    for panel in ("bike", "household"):
        spec, timestamps, values, source_qc = build_panel(panel)
        source = source_manifest(spec)
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
        datasets[panel] = _dataset_manifest(summary)
        files[f"{panel}_fit"] = {"path": summary["fit_data_path"], "sha256": summary["fit_data_sha256"]}
        files[f"{panel}_holdout"] = {"path": summary["holdout_data_path"], "sha256": summary["holdout_data_sha256"]}
        for per_split in summary["target_qc"].values():
            for metrics in per_split.values():
                all_qc_passed = all_qc_passed and bool(metrics["pass_min_70pct"])

    data_contract = {
        "fit_archives": "train+validation only; physically truncated at validation end with no cal/eval origin arrays",
        "holdout_archives": "full 242-day block with calibration/evaluation origins; read only after global selection is frozen",
        "fixed_blocks": {
            panel: _contract_json(temporal_contract(panel))
            for panel in ("bike", "household")
        },
        "context_target_alignment": "origin is first target timestamp; context [origin-336h, origin), target [origin, origin+48h)",
        "origin_stride_hours": ORIGIN_STRIDE_HOURS,
        "target_window_overlap_hours": HORIZON - ORIGIN_STRIDE_HOURS,
        "missing_policy": "target_values preserve raw NaNs; context_values use precontext-only median fallback then causal forward fill, never future backfill",
        "fit_statistics": "mean/std/median fitted from observed train interval rows only",
        "score_columns": "only target_indices [0,1] are scored",
    }
    manifest = {
        "version": VERSION,
        "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "study": STUDY,
        "all_qc_passed": bool(all_qc_passed),
        "datasets": datasets,
        "files": files,
        "panels": panels,
        "data_contract": data_contract,
        "source_contract": {
            "parser_reused_from": "experiments.peft_external_gap_v1.data",
            "raw_hashes_recorded": True,
            "pretraining_overlap": "UNKNOWN",
            "holdout_selection": "next unused local temporal block fixed by timestamps; no magnitude-based choice",
        },
    }
    with manifest_path.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    manifest["manifest_path"] = _relative(manifest_path)
    manifest["manifest_sha256"] = sha256_file(manifest_path)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs") / STUDY / "prepared")
    args = parser.parse_args(argv)
    manifest = prepare(args.output)
    print(json.dumps({
        "completed": True,
        "manifest_path": manifest["manifest_path"],
        "manifest_sha256": manifest["manifest_sha256"],
        "all_qc_passed": manifest["all_qc_passed"],
        "datasets": {
            name: {
                "fit_data_path": item["fit_data_path"],
                "fit_data_sha256": item["fit_data_sha256"],
                "holdout_data_path": item["holdout_data_path"],
                "holdout_data_sha256": item["holdout_data_sha256"],
                "origin_counts": item["origin_counts"],
            }
            for name, item in manifest["datasets"].items()
        },
    }, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
