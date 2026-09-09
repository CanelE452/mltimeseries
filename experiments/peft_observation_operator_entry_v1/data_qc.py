"""USCRN pre-data provenance and support-QC for observation-operator entry."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
import urllib.request
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
STUDY = "uscrn_operator_entry_v1"
PLAN = "_docs/notes/tsfm_topics/16_observation_operator_entry_plan_20260908.md"
DEFAULT_OUTPUT = Path("data_external") / STUDY
STATION = "AZ_Tucson_11_W"
YEAR = 2024
REQUEST_TIMEOUT_SECONDS = 30
WHOLE_TIMEOUT_SECONDS = 180
MAX_FILE_BYTES = 32 * 1024 * 1024
MAX_TOTAL_BYTES = 64 * 1024 * 1024
MISSING_SENTINEL_THRESHOLD = -9000.0


SOURCES = {
    "hourly": {
        "product": "hourly02",
        "url": "https://www.ncei.noaa.gov/pub/data/uscrn/products/hourly02/2024/CRNH0203-2024-AZ_Tucson_11_W.txt",
        "filename": "CRNH0203-2024-AZ_Tucson_11_W.txt",
        "role": "hourly T_CALC last-5-minute average and T_HR_AVG whole-hour average",
    },
    "subhourly": {
        "product": "subhourly01",
        "url": "https://www.ncei.noaa.gov/pub/data/uscrn/products/subhourly01/2024/CRNS0101-05-2024-AZ_Tucson_11_W.txt",
        "filename": "CRNS0101-05-2024-AZ_Tucson_11_W.txt",
        "role": "5-minute AIR_TEMPERATURE average ending at UTC_TIME",
    },
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json_once_or_same(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        if existing != text:
            raise FileExistsError(f"Refusing to overwrite different file: {path}")
        return
    path.write_text(text, encoding="utf-8")


def _rel(path: Path, root: Path) -> str:
    path = path.resolve()
    root = root.resolve()
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def build_prefetch_contract(root: Path = ROOT, output_dir: Path | None = None) -> dict:
    root = Path(root).resolve()
    output = (root / DEFAULT_OUTPUT if output_dir is None else Path(output_dir)).resolve()
    source_entries = {
        key: {
            "url": value["url"],
            "filename": value["filename"],
            "product": value["product"],
            "station": STATION,
            "year": YEAR,
            "role": value["role"],
        }
        for key, value in SOURCES.items()
    }
    return {
        "completed": False,
        "study": STUDY,
        "plan_path": PLAN,
        "plan_sha256": sha256(root / PLAN),
        "script_path": _rel(Path(__file__), root),
        "script_sha256": sha256(Path(__file__)),
        "output_dir": _rel(output, root) if output.is_relative_to(root) else str(output),
        "expected_no_selection": True,
        "download_limits": {
            "request_timeout_seconds": REQUEST_TIMEOUT_SECONDS,
            "whole_timeout_seconds": WHOLE_TIMEOUT_SECONDS,
            "max_file_bytes": MAX_FILE_BYTES,
            "max_total_bytes": MAX_TOTAL_BYTES,
            "sequential_stream": True,
        },
        "sources": source_entries,
        "schema": {
            "station": STATION,
            "year": YEAR,
            "missing_sentinel_threshold": MISSING_SENTINEL_THRESHOLD,
            "support": {
                "hourly_T_CALC": "last_5_minutes_ending_at_UTC_TIME",
                "hourly_T_HR_AVG": "60_minutes_ending_at_UTC_TIME",
                "subhourly_AIR_TEMPERATURE": "5_minutes_ending_at_UTC_TIME",
                "T_MAX_T_MIN": "nonlinear_10_second_sensor_extrema_not_public_5min_mean_extrema",
            },
            "fields": {
                "hourly": ["WBANNO", "UTC_DATE", "UTC_TIME", "T_CALC", "T_HR_AVG"],
                "subhourly": ["WBANNO", "UTC_DATE", "UTC_TIME", "AIR_TEMPERATURE"],
            },
            "qc": [
                "utc_join_duplicates_grid_missing",
                "hourly_T_CALC_vs_subhourly_AIR_TEMPERATURE_exact_difference",
                "mean_12_valid_subhourly_bins_vs_hourly_T_HR_AVG",
                "exclude_hours_whose_support_reaches_before_available_subhourly_file",
            ],
        },
    }


def write_prefetch_contract(root: Path = ROOT, output_dir: Path | None = None) -> dict:
    contract = build_prefetch_contract(root, output_dir)
    output = (Path(root).resolve() / DEFAULT_OUTPUT if output_dir is None else Path(output_dir).resolve())
    _write_json_once_or_same(output / "contract.json", contract)
    return contract


def _known_existing_sources(qc_path: Path) -> dict[str, dict]:
    if not qc_path.exists():
        return {}
    qc = _read_json(qc_path)
    if not qc.get("completed"):
        return {}
    return qc.get("source_files", {})


def _validate_raw_preconditions(raw_dir: Path, known_sources: dict[str, dict]) -> None:
    if not raw_dir.exists():
        return
    partials = sorted(raw_dir.glob("*.part"))
    if partials:
        raise FileExistsError(f"Refusing to continue with partial download present: {partials[0]}")
    expected_names = {entry["filename"] for entry in SOURCES.values()}
    for path in raw_dir.iterdir():
        if path.is_dir():
            raise FileExistsError(f"Refusing unexpected directory in raw folder: {path}")
        if path.name not in expected_names:
            raise FileExistsError(f"Refusing unexpected raw file: {path}")
        key = next(k for k, entry in SOURCES.items() if entry["filename"] == path.name)
        known = known_sources.get(key)
        if not known or known.get("sha256") != sha256(path) or known.get("url") != SOURCES[key]["url"]:
            raise FileExistsError(f"Refusing uncontracted existing raw file: {path}")


def _fetch_url(
    url: str,
    destination: Path,
    *,
    max_bytes: int = MAX_FILE_BYTES,
    timeout_seconds: int = REQUEST_TIMEOUT_SECONDS,
    deadline_monotonic: float | None = None,
) -> dict:
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing raw file: {destination}")
    part = destination.with_name(destination.name + ".part")
    if part.exists():
        raise FileExistsError(f"Refusing to overwrite partial download: {part}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    bytes_seen = 0
    try:
        with urllib.request.urlopen(url, timeout=timeout_seconds) as response:
            content_length = response.headers.get("Content-Length")
            parsed_length = int(content_length) if content_length and content_length.isdigit() else None
            if parsed_length is not None and parsed_length > max_bytes:
                raise ValueError(f"Content-Length exceeds file limit for {url}: {parsed_length}")
            with part.open("wb") as f:
                while True:
                    if deadline_monotonic is not None and time.monotonic() > deadline_monotonic:
                        raise TimeoutError(f"Whole-download timeout exceeded while fetching {url}")
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    bytes_seen += len(chunk)
                    if bytes_seen > max_bytes:
                        raise ValueError(f"Downloaded bytes exceed file limit for {url}: {bytes_seen}")
                    f.write(chunk)
            if parsed_length is not None and bytes_seen != parsed_length:
                raise ValueError(f"Content-Length mismatch for {url}: expected {parsed_length}, got {bytes_seen}")
            part.replace(destination)
            return {
                "url": url,
                "response_url": response.geturl(),
                "status": getattr(response, "status", None) or response.getcode(),
                "content_length_header": parsed_length,
                "last_modified": response.headers.get("Last-Modified"),
                "bytes": bytes_seen,
                "sha256": sha256(destination),
                "reused_existing": False,
            }
    except Exception:
        raise


def _timestamp(date_text: str, time_text: str) -> pd.Timestamp:
    return pd.Timestamp(datetime.strptime(date_text + time_text.zfill(4), "%Y%m%d%H%M"))


def _is_valid_temperature(value: float) -> bool:
    return math.isfinite(value) and value > MISSING_SENTINEL_THRESHOLD


def _load_hourly(path: Path) -> pd.DataFrame:
    rows = []
    with Path(path).open("r", encoding="ascii", errors="strict") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) < 10:
                raise ValueError(f"hourly line {line_no} has fewer than 10 fields")
            rows.append(
                {
                    "utc": _timestamp(parts[1], parts[2]),
                    "wban": parts[0],
                    "crx_vn": parts[5],
                    "t_calc": float(parts[8]),
                    "t_hr_avg": float(parts[9]),
                }
            )
    return pd.DataFrame(rows, columns=["utc", "wban", "crx_vn", "t_calc", "t_hr_avg"])


def _load_subhourly(path: Path) -> pd.DataFrame:
    rows = []
    with Path(path).open("r", encoding="ascii", errors="strict") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) < 9:
                raise ValueError(f"subhourly line {line_no} has fewer than 9 fields")
            rows.append({"utc": _timestamp(parts[1], parts[2]), "wban": parts[0], "crx_vn": parts[5], "air_temperature": float(parts[8])})
    return pd.DataFrame(rows, columns=["utc", "wban", "crx_vn", "air_temperature"])


def _grid_report(stamps: Iterable[pd.Timestamp], step: timedelta, *, year: int = YEAR) -> dict:
    stamps = list(stamps)
    if not stamps:
        return {"rows": 0, "unique_rows": 0, "expected_by_span": 0, "missing_by_span": 0, "off_step_count": 0, "year_outlier_count": 0}
    unique = sorted(set(stamps))
    expected = int((unique[-1].to_pydatetime() - unique[0].to_pydatetime()) / step) + 1
    step_seconds = int(step.total_seconds())
    off_step = []
    year_outliers = []
    for stamp in unique:
        dt = stamp.to_pydatetime()
        seconds_since_midnight = dt.hour * 3600 + dt.minute * 60 + dt.second
        if dt.microsecond or seconds_since_midnight % step_seconds:
            off_step.append(stamp)
        if dt.year != year:
            year_outliers.append(stamp)
    return {
        "rows": len(stamps),
        "unique_rows": len(unique),
        "first_utc": unique[0].isoformat(sep=" "),
        "last_utc": unique[-1].isoformat(sep=" "),
        "expected_by_span": expected,
        "missing_by_span": expected - len(unique),
        "off_step_count": len(off_step),
        "off_step_examples": [stamp.isoformat(sep=" ") for stamp in off_step[:5]],
        "year_outlier_count": len(year_outliers),
        "year_outlier_examples": [stamp.isoformat(sep=" ") for stamp in year_outliers[:5]],
    }


def _unique_records(df: pd.DataFrame, value_column: str) -> dict[pd.Timestamp, float]:
    counts = Counter(df["utc"])
    return {row.utc: float(getattr(row, value_column)) for row in df.itertuples() if counts[row.utc] == 1}


def _diff_summary(diffs: list[float]) -> dict:
    if not diffs:
        return {
            "count": 0,
            "exact_zero_count": 0,
            "max_abs_diff": None,
            "mean_abs_diff": None,
            "p50_abs_diff": None,
            "p95_abs_diff": None,
            "abs_diff_ge_0_1_count": 0,
            "histogram_abs_diff_bins": {"[0,0.05)": 0, "[0.05,0.1)": 0, "[0.1,0.2)": 0, "[0.2,0.5)": 0, "[0.5,inf)": 0},
        }
    arr = np.abs(np.asarray(diffs, dtype=np.float64))
    return {
        "count": int(arr.size),
        "exact_zero_count": int(np.count_nonzero(arr == 0.0)),
        "max_abs_diff": float(arr.max()),
        "mean_abs_diff": float(arr.mean()),
        "p50_abs_diff": float(np.quantile(arr, 0.5)),
        "p95_abs_diff": float(np.quantile(arr, 0.95)),
        "abs_diff_ge_0_1_count": int(np.count_nonzero(arr >= 0.1)),
        "histogram_abs_diff_bins": {
            "[0,0.05)": int(np.count_nonzero((arr >= 0.0) & (arr < 0.05))),
            "[0.05,0.1)": int(np.count_nonzero((arr >= 0.05) & (arr < 0.1))),
            "[0.1,0.2)": int(np.count_nonzero((arr >= 0.1) & (arr < 0.2))),
            "[0.2,0.5)": int(np.count_nonzero((arr >= 0.2) & (arr < 0.5))),
            "[0.5,inf)": int(np.count_nonzero(arr >= 0.5)),
        },
    }


def compute_qc(hourly_path: Path, subhourly_path: Path) -> dict:
    hourly = _load_hourly(hourly_path)
    subhourly = _load_subhourly(subhourly_path)
    hourly_counts = Counter(hourly["utc"])
    sub_counts = Counter(subhourly["utc"])
    hourly_wbans = sorted(str(value) for value in hourly["wban"].dropna().unique())
    subhourly_wbans = sorted(str(value) for value in subhourly["wban"].dropna().unique())
    hourly_crx = sorted(str(value) for value in hourly["crx_vn"].dropna().unique())
    subhourly_crx = sorted(str(value) for value in subhourly["crx_vn"].dropna().unique())
    hourly_unique = _unique_records(hourly, "t_calc")
    hourly_avg_unique = _unique_records(hourly, "t_hr_avg")
    sub_unique = _unique_records(subhourly, "air_temperature")

    pair_diffs = []
    paired_valid = 0
    for stamp, t_calc in hourly_unique.items():
        if stamp in sub_unique and _is_valid_temperature(t_calc) and _is_valid_temperature(sub_unique[stamp]):
            paired_valid += 1
            pair_diffs.append(t_calc - sub_unique[stamp])
    pair_summary = _diff_summary(pair_diffs)

    sub_min = min(subhourly["utc"]) if len(subhourly) else None
    mean_diffs = []
    complete_hours = 0
    incomplete_hours = 0
    support_before_file = 0
    for stamp, t_hr_avg in hourly_avg_unique.items():
        expected = [stamp - pd.Timedelta(minutes=offset) for offset in range(55, -1, -5)]
        if sub_min is not None and any(item < sub_min for item in expected):
            support_before_file += 1
            continue
        values = [sub_unique.get(item) for item in expected]
        if (
            len(values) == 12
            and all(value is not None and _is_valid_temperature(value) for value in values)
            and _is_valid_temperature(t_hr_avg)
        ):
            complete_hours += 1
            mean_diffs.append(float(np.mean(values)) - t_hr_avg)
        else:
            incomplete_hours += 1
    mean_summary = _diff_summary(mean_diffs)

    return {
        "completed": True,
        "station": STATION,
        "year": YEAR,
        "grid": {
            "hourly_rows": int(len(hourly)),
            "hourly_unique_rows": int(hourly["utc"].nunique()),
            "subhourly_rows": int(len(subhourly)),
            "subhourly_unique_rows": int(subhourly["utc"].nunique()),
            "hourly": _grid_report(hourly["utc"], timedelta(hours=1)),
            "subhourly": _grid_report(subhourly["utc"], timedelta(minutes=5)),
        },
        "identity": {
            "hourly_wbannos": hourly_wbans,
            "subhourly_wbannos": subhourly_wbans,
            "same_single_wban": len(hourly_wbans) == 1 and hourly_wbans == subhourly_wbans,
            "hourly_crx_versions": hourly_crx,
            "subhourly_crx_versions": subhourly_crx,
            "same_single_crx_version": len(hourly_crx) == 1 and hourly_crx == subhourly_crx,
        },
        "duplicates": {
            "hourly_duplicate_stamps": int(sum(1 for count in hourly_counts.values() if count > 1)),
            "hourly_duplicate_extra_rows": int(sum(count - 1 for count in hourly_counts.values() if count > 1)),
            "subhourly_duplicate_stamps": int(sum(1 for count in sub_counts.values() if count > 1)),
            "subhourly_duplicate_extra_rows": int(sum(count - 1 for count in sub_counts.values() if count > 1)),
        },
        "missing": {
            "hourly_t_calc_missing": int(sum(not _is_valid_temperature(v) for v in hourly["t_calc"])),
            "hourly_t_hr_avg_missing": int(sum(not _is_valid_temperature(v) for v in hourly["t_hr_avg"])),
            "subhourly_air_temperature_missing": int(sum(not _is_valid_temperature(v) for v in subhourly["air_temperature"])),
        },
        "hourly_vs_subhourly_t_calc": {
            "meaning": "compare hourly T_CALC with subhourly AIR_TEMPERATURE at identical UTC end timestamp",
            "paired_valid_count": int(paired_valid),
            "exact_equal_count": pair_summary["exact_zero_count"],
            **{k: v for k, v in pair_summary.items() if k not in {"exact_zero_count"}},
        },
        "subhourly_mean12_vs_hourly_t_hr_avg": {
            "meaning": "mean of twelve valid 5-minute AIR_TEMPERATURE bins ending at hourly UTC_TIME versus hourly T_HR_AVG",
            "complete_valid_hours": int(complete_hours),
            "incomplete_hours": int(incomplete_hours),
            "first_hour_excluded_count": int(support_before_file),
            **mean_summary,
        },
        "nonlinear_max_min_note": "T_MAX/T_MIN are not treated as max/min of public 5-minute AIR_TEMPERATURE means.",
    }


def _reuse_existing(key: str, path: Path, known_sources: dict[str, dict]) -> dict:
    known = known_sources.get(key)
    if not known:
        raise FileExistsError(f"Refusing uncontracted existing raw file: {path}")
    current_sha = sha256(path)
    if known.get("sha256") != current_sha or known.get("url") != SOURCES[key]["url"]:
        raise FileExistsError(f"Existing raw file does not match saved QC provenance: {path}")
    reused = dict(known)
    reused["reused_existing"] = True
    return reused


def _download_sources(
    raw_dir: Path,
    *,
    known_sources: dict[str, dict],
    fetcher: Callable[..., dict] | None,
) -> tuple[dict[str, Path], dict[str, dict]]:
    fetch = fetcher or _fetch_url
    deadline = time.monotonic() + WHOLE_TIMEOUT_SECONDS
    paths: dict[str, Path] = {}
    metadata: dict[str, dict] = {}
    downloaded_total = 0
    raw_dir.mkdir(parents=True, exist_ok=True)
    for key in ("hourly", "subhourly"):
        entry = SOURCES[key]
        target = raw_dir / entry["filename"]
        paths[key] = target
        if target.exists():
            metadata[key] = _reuse_existing(key, target, known_sources)
            continue
        meta = fetch(
            entry["url"],
            target,
            max_bytes=MAX_FILE_BYTES,
            timeout_seconds=REQUEST_TIMEOUT_SECONDS,
            deadline_monotonic=deadline,
        )
        if not target.exists():
            raise FileNotFoundError(f"Fetcher did not create expected raw file: {target}")
        if meta.get("sha256") != sha256(target):
            raise AssertionError(f"Fetcher sha256 metadata does not match downloaded file: {target}")
        downloaded_total += int(meta.get("bytes", target.stat().st_size))
        if downloaded_total > MAX_TOTAL_BYTES:
            raise ValueError(f"Total downloaded bytes exceed limit: {downloaded_total}")
        metadata[key] = meta
    return paths, metadata


def _existing_qc_is_reusable(output: Path, contract: dict) -> dict | None:
    qc_path = output / "qc.json"
    if not qc_path.exists():
        return None
    qc = _read_json(qc_path)
    if qc.get("contract_sha256") != sha256(output / "contract.json"):
        return None
    for key, entry in SOURCES.items():
        path = output / "raw" / entry["filename"]
        meta = qc.get("source_files", {}).get(key)
        if not path.exists() or not meta or meta.get("url") != entry["url"] or meta.get("sha256") != sha256(path):
            return None
    return qc


def run(
    root: Path = ROOT,
    output_dir: Path | None = None,
    *,
    fetcher: Callable[..., dict] | None = None,
) -> dict:
    root = Path(root).resolve()
    output = (root / DEFAULT_OUTPUT if output_dir is None else Path(output_dir)).resolve()
    output.mkdir(parents=True, exist_ok=True)
    contract = write_prefetch_contract(root, output)
    reusable = _existing_qc_is_reusable(output, contract)
    if reusable is not None:
        return reusable
    known_sources = _known_existing_sources(output / "qc.json")
    raw_dir = output / "raw"
    _validate_raw_preconditions(raw_dir, known_sources)
    paths, source_meta = _download_sources(raw_dir, known_sources=known_sources, fetcher=fetcher)
    qc = compute_qc(paths["hourly"], paths["subhourly"])
    qc.update(
        {
            "contract_sha256": sha256(output / "contract.json"),
            "source_files": source_meta,
            "paths": {key: str(path.relative_to(output).as_posix()) for key, path in paths.items()},
        }
    )
    _write_json_once_or_same(output / "qc.json", qc)
    return qc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download bounded USCRN Tucson 2024 files and write observation-operator QC.")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    qc = run(root=args.root, output_dir=args.output)
    print(json.dumps({"completed": qc["completed"], "output": str((args.output or args.root / DEFAULT_OUTPUT).resolve())}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
