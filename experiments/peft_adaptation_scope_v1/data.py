"""Data contract for the PEFT adaptation-scope pilot.

The prepared arrays are deliberately small and explicit: one recent continuous
116-day segment per panel, temporal splits, and train-only scale statistics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PANELS = {
    "ettm2": {
        "path": Path("data/ETT-small/ETTm2.csv"),
        "encoding": "utf-8",
        "date_format": "%Y-%m-%d %H:%M:%S",
        "context": 384,
        "horizon": 96,
        "expected_channels": 7,
        "sentinel": None,
    },
    "jena": {
        "path": Path("data/jena_mpi_roof/mpi_roof_2024.csv"),
        "encoding": "latin1",
        "date_format": "%d.%m.%Y %H:%M:%S",
        "context": 576,
        "horizon": 144,
        "expected_channels": 21,
        "sentinel": -9999.0,
    },
}

SPLIT_DAYS = {"pre_context": 4, "fit": 64, "validation": 16, "evaluation": 32}
TRAIN_STRIDE = 16
STD_FLOOR = 1e-6


@dataclass(frozen=True)
class PreparedPanel:
    panel: str
    path: Path
    values: np.ndarray
    timestamps: np.ndarray
    channels: np.ndarray
    train_origins: np.ndarray
    val_origins: np.ndarray
    eval_origins: np.ndarray
    fit_mean: np.ndarray
    fit_std: np.ndarray
    context: int
    horizon: int
    rows_per_day: int
    manifest: dict[str, Any]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _first_time_column(frame: pd.DataFrame) -> str:
    for column in frame.columns:
        if not pd.api.types.is_numeric_dtype(frame[column]):
            return str(column)
    raise ValueError("No timestamp-like non-numeric column found.")


def _complete_day_end_index(times: pd.Series, freq: pd.Timedelta) -> int:
    latest_complete_end = None
    for index, timestamp in enumerate(times):
        next_time = timestamp + freq
        if next_time == next_time.normalize():
            latest_complete_end = index + 1
    if latest_complete_end is None:
        raise ValueError("Could not find a complete day boundary.")
    return int(latest_complete_end)


def _take_regular_origins(start: int, stop_exclusive: int, horizon: int, stride: int) -> np.ndarray:
    last_origin = stop_exclusive - horizon
    if last_origin < start:
        return np.zeros(0, dtype=np.int64)
    return np.arange(start, last_origin + 1, stride, dtype=np.int64)


def _load_panel(project_root: Path, panel: str) -> PreparedPanel:
    cfg = PANELS[panel]
    source = project_root / cfg["path"]
    frame = pd.read_csv(source, encoding=str(cfg["encoding"]))
    time_column = _first_time_column(frame)
    times = pd.to_datetime(frame[time_column], format=str(cfg["date_format"]), errors="raise")
    sort_changes = 0
    if not times.is_monotonic_increasing:
        order = np.argsort(times.to_numpy())
        sort_changes = int(np.count_nonzero(order != np.arange(len(order))))
        frame = frame.iloc[order].reset_index(drop=True)
        times = times.iloc[order].reset_index(drop=True)

    duplicate_timestamps = int(times.duplicated().sum())
    if duplicate_timestamps:
        raise ValueError(f"{panel}: duplicate timestamps are not allowed: {duplicate_timestamps}")

    numeric_columns = [c for c in frame.columns if c != time_column]
    non_numeric = [c for c in numeric_columns if not pd.api.types.is_numeric_dtype(frame[c])]
    if non_numeric:
        raise ValueError(f"{panel}: non-numeric target columns: {non_numeric}")
    if len(numeric_columns) != int(cfg["expected_channels"]):
        raise ValueError(f"{panel}: expected {cfg['expected_channels']} channels, got {len(numeric_columns)}")

    raw_values = frame[numeric_columns].to_numpy(dtype=np.float32)
    sentinel_count = 0
    if cfg["sentinel"] is not None:
        sentinel_mask = raw_values <= float(cfg["sentinel"]) + 1e-6
        sentinel_count = int(sentinel_mask.sum())
        raw_values = raw_values.copy()
        raw_values[sentinel_mask] = np.nan

    diffs = times.diff().dropna()
    if diffs.empty:
        raise ValueError(f"{panel}: not enough timestamps")
    freq = diffs.mode().iloc[0]
    irregular_steps = int((diffs != freq).sum())
    if irregular_steps:
        raise ValueError(f"{panel}: irregular timestamp gaps detected: {irregular_steps}")
    rows_per_day_float = pd.Timedelta(days=1) / freq
    rows_per_day = int(rows_per_day_float)
    if rows_per_day != rows_per_day_float:
        raise ValueError(f"{panel}: frequency does not divide one day: {freq}")

    context = int(cfg["context"])
    horizon = int(cfg["horizon"])
    required_rows = sum(SPLIT_DAYS.values()) * rows_per_day
    end = _complete_day_end_index(times, freq)
    start = end - required_rows
    if start < 0:
        raise ValueError(f"{panel}: need {required_rows} recent rows, found only {end}")

    values = raw_values[start:end].astype(np.float32, copy=False)
    local_times = times.iloc[start:end].dt.strftime("%Y-%m-%d %H:%M:%S").to_numpy(dtype="<U19")

    pre_rows = SPLIT_DAYS["pre_context"] * rows_per_day
    fit_rows = SPLIT_DAYS["fit"] * rows_per_day
    val_rows = SPLIT_DAYS["validation"] * rows_per_day
    eval_rows = SPLIT_DAYS["evaluation"] * rows_per_day
    fit_start = pre_rows
    fit_end = fit_start + fit_rows
    val_start = fit_end
    val_end = val_start + val_rows
    eval_start = val_end
    eval_end = eval_start + eval_rows

    fit_block = values[fit_start:fit_end]
    fit_mean = np.nanmean(fit_block, axis=0, dtype=np.float64).astype(np.float32)
    fit_std_raw = np.nanstd(fit_block, axis=0, dtype=np.float64).astype(np.float32)
    constant_channels = np.nonzero(~np.isfinite(fit_std_raw) | (fit_std_raw < STD_FLOOR))[0].astype(np.int64)
    eligible_mask = np.ones(len(numeric_columns), dtype=bool)
    eligible_mask[constant_channels] = False
    excluded_constant_channels = [str(numeric_columns[i]) for i in constant_channels.tolist()]
    if not eligible_mask.any():
        raise ValueError(f"{panel}: all channels are constant or invalid in the fit block")
    if not eligible_mask.all():
        values = values[:, eligible_mask]
        numeric_columns = [str(c) for c, keep in zip(numeric_columns, eligible_mask) if keep]
        fit_mean = fit_mean[eligible_mask]
        fit_std_raw = fit_std_raw[eligible_mask]
    fit_std = fit_std_raw.astype(np.float32)

    train_origins = _take_regular_origins(fit_start, fit_end, horizon, TRAIN_STRIDE)
    val_origins = _take_regular_origins(val_start, val_end, horizon, horizon)
    eval_origins = _take_regular_origins(eval_start, eval_end, horizon, horizon)

    label_nan = {
        "fit": int(np.isnan(values[fit_start:fit_end]).sum()),
        "validation": int(np.isnan(values[val_start:val_end]).sum()),
        "evaluation": int(np.isnan(values[eval_start:eval_end]).sum()),
    }
    manifest: dict[str, Any] = {
        "panel": panel,
        "source_path": str(cfg["path"]).replace("\\", "/"),
        "source_sha256": sha256_file(source),
        "source_rows": int(len(frame)),
        "source_columns": [str(c) for c in frame.columns],
        "encoding": str(cfg["encoding"]),
        "date_format": str(cfg["date_format"]),
        "time_column": time_column,
        "original_channels": [str(c) for c in frame.columns if c != time_column],
        "channels": [str(c) for c in numeric_columns],
        "excluded_constant_fit_channels": excluded_constant_channels,
        "frequency_seconds": int(freq.total_seconds()),
        "rows_per_day": rows_per_day,
        "segment_source_start_row": int(start),
        "segment_source_end_row_exclusive": int(end),
        "segment_start_timestamp": str(local_times[0]),
        "segment_end_timestamp": str(local_times[-1]),
        "context": context,
        "horizon": horizon,
        "split_days": dict(SPLIT_DAYS),
        "split_rows": {
            "pre_context": [0, pre_rows],
            "fit": [fit_start, fit_end],
            "validation": [val_start, val_end],
            "evaluation": [eval_start, eval_end],
        },
        "train_stride": TRAIN_STRIDE,
        "val_eval_stride": horizon,
        "origin_counts": {
            "train": int(train_origins.size),
            "validation": int(val_origins.size),
            "evaluation": int(eval_origins.size),
        },
        "sort_changes": sort_changes,
        "duplicate_timestamps": duplicate_timestamps,
        "irregular_steps": irregular_steps,
        "sentinel_rule": None if cfg["sentinel"] is None else f"values <= {cfg['sentinel']} treated as missing",
        "sentinel_count": sentinel_count,
        "missing_values": {
            "all_segment": int(np.isnan(values).sum()),
            **label_nan,
        },
        "constant_fit_channels": excluded_constant_channels,
        "fit_mean_sha256": sha256_bytes(fit_mean.tobytes()),
        "fit_std_sha256": sha256_bytes(fit_std.tobytes()),
        "values_sha256": sha256_bytes(values.tobytes()),
        "decision_context": "development forecast screen for Chronos-2 adaptation-scope research",
        "leakage_controls": [
            "temporal train < validation < evaluation labels",
            "fit_mean and fit_std use only the fit label block",
            "validation/evaluation labels are never used in feature scaling",
            "future variables are masked for foundation-model inputs",
        ],
    }

    return PreparedPanel(
        panel=panel,
        path=source,
        values=values,
        timestamps=local_times,
        channels=np.asarray(numeric_columns, dtype="<U64"),
        train_origins=train_origins,
        val_origins=val_origins,
        eval_origins=eval_origins,
        fit_mean=fit_mean,
        fit_std=fit_std,
        context=context,
        horizon=horizon,
        rows_per_day=rows_per_day,
        manifest=manifest,
    )


def save_panel(panel: PreparedPanel, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{panel.panel}.npz"
    manifest_json = json.dumps(panel.manifest, ensure_ascii=False)
    np.savez_compressed(
        path,
        values=panel.values,
        timestamps=panel.timestamps,
        channels=panel.channels,
        train_origins=panel.train_origins,
        val_origins=panel.val_origins,
        eval_origins=panel.eval_origins,
        fit_mean=panel.fit_mean,
        fit_std=panel.fit_std,
        context=np.asarray(panel.context, dtype=np.int64),
        horizon=np.asarray(panel.horizon, dtype=np.int64),
        rows_per_day=np.asarray(panel.rows_per_day, dtype=np.int64),
        manifest_json=np.asarray(manifest_json, dtype=f"<U{len(manifest_json)}"),
    )
    return path


def load_prepared(path: Path) -> PreparedPanel:
    archive = np.load(path, allow_pickle=False)
    manifest = json.loads(str(archive["manifest_json"].item()))
    return PreparedPanel(
        panel=str(manifest["panel"]),
        path=Path(manifest["source_path"]),
        values=archive["values"].astype(np.float32, copy=False),
        timestamps=archive["timestamps"],
        channels=archive["channels"],
        train_origins=archive["train_origins"].astype(np.int64, copy=False),
        val_origins=archive["val_origins"].astype(np.int64, copy=False),
        eval_origins=archive["eval_origins"].astype(np.int64, copy=False),
        fit_mean=archive["fit_mean"].astype(np.float32, copy=False),
        fit_std=archive["fit_std"].astype(np.float32, copy=False),
        context=int(archive["context"]),
        horizon=int(archive["horizon"]),
        rows_per_day=int(archive["rows_per_day"]),
        manifest=manifest,
    )


def context_for_origin(panel: PreparedPanel, origins: np.ndarray) -> np.ndarray:
    offsets = np.arange(-panel.context, 0, dtype=np.int64)
    return panel.values[origins[:, None] + offsets[None, :]]


def targets_for_origin(panel: PreparedPanel, origins: np.ndarray) -> np.ndarray:
    offsets = np.arange(0, panel.horizon, dtype=np.int64)
    return panel.values[origins[:, None] + offsets[None, :]]


def evenly_cap(values: np.ndarray, cap: int | None) -> np.ndarray:
    if cap is None or cap <= 0 or values.size <= cap:
        return values
    positions = np.linspace(0, values.size - 1, int(cap))
    return values[np.unique(np.rint(positions).astype(np.int64))]


def prepare(project_root: Path, output_dir: Path) -> dict[str, Any]:
    panels = {}
    written = {}
    for panel_name in PANELS:
        panel = _load_panel(project_root, panel_name)
        written[panel_name] = str(save_panel(panel, output_dir))
        panels[panel_name] = panel.manifest

    manifest = {
        "version": "peft_adaptation_scope_v1.real_s1.20260908",
        "created_by": "experiments.peft_adaptation_scope_v1.data.prepare",
        "panels": panels,
        "files": written,
        "quality_bar": "development research screen with publication-style leakage controls",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("runs/peft_adaptation_scope_v1/prepared"))
    args = parser.parse_args()
    manifest = prepare(args.project_root.resolve(), args.output.resolve())
    for panel, meta in manifest["panels"].items():
        counts = meta["origin_counts"]
        print(
            f"{panel}: rows={meta['source_rows']} segment={meta['segment_start_timestamp']}.."
            f"{meta['segment_end_timestamp']} C={len(meta['channels'])} "
            f"origins train/val/eval={counts['train']}/{counts['validation']}/{counts['evaluation']} "
            f"missing={meta['missing_values']['all_segment']} sentinel={meta['sentinel_count']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
