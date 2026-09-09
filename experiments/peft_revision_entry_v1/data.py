"""Validated ALFRED interval snapshots for the frozen revision-entry contract."""

from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import defaultdict
import csv
from dataclasses import dataclass
from datetime import date, timedelta
import hashlib
import json
import math
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]
LOWER = date(1990, 1, 1)
UPPER = date(2025, 6, 30)


@dataclass(frozen=True)
class Record:
    start: date
    end: date
    value: float


class VintageSeries:
    def __init__(self, name: str, records: dict[int, list[Record]], vintage_dates,
                 coverage_start: date = LOWER, coverage_end: date = UPPER):
        self.name = name
        self.coverage_start, self.coverage_end = coverage_start, coverage_end
        self.vintage_dates = tuple(sorted(set(vintage_dates)))
        self._records = {}
        self._starts = {}
        for event, values in records.items():
            ordered = tuple(sorted(values, key=lambda r: r.start))
            for i, record in enumerate(ordered):
                if record.end < record.start:
                    raise ValueError(f"Reversed interval: {name}, {event}, {record}")
                if not math.isfinite(record.value) or record.value <= 0:
                    raise ValueError(f"Invalid level: {name}, {event}, {record}")
                if i and ordered[i-1].end >= record.start:
                    raise ValueError(f"Overlapping inclusive intervals: {name}, {event}")
            self._records[event] = ordered
            self._starts[event] = tuple(r.start for r in ordered)
        self.events = tuple(sorted(self._records))

    def records(self, event: int) -> tuple[Record, ...]:
        return self._records.get(event, ())

    def snapshot(self, event: int, asof: date) -> float:
        if not self.coverage_start <= asof <= self.coverage_end:
            return float("nan")
        index = bisect_right(self._starts.get(event, ()), asof) - 1
        if index < 0:
            return float("nan")
        record = self._records[event][index]
        return record.value if asof <= record.end else float("nan")

    def growth(self, event: int, asof: date) -> float:
        current, previous = self.snapshot(event, asof), self.snapshot(event-1, asof)
        return 100 * (math.log(current) - math.log(previous))

    def first_growth(self, event: int) -> tuple[date | None, float]:
        candidates = {max(self.coverage_start, r.start) for t in (event-1, event) for r in self.records(t)}
        for asof in sorted(candidates):
            value = self.growth(event, asof)
            if math.isfinite(value):
                return asof, value
        return None, float("nan")


def parse_csv(path: Path, name: str, vintage_dates) -> VintageSeries:
    records = defaultdict(list)
    with path.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        expected = ["period_start_date", name, "realtime_start_date", "realtime_end_date"]
        if reader.fieldnames != expected:
            raise ValueError(f"Unexpected ALFRED schema: {reader.fieldnames}")
        for index, row in enumerate(reader, start=2):
            try:
                event_date = date.fromisoformat(row["period_start_date"])
                if event_date.day != 1:
                    raise ValueError("Monthly event must use day one")
                event = event_date.year*12 + event_date.month-1
                start = date.fromisoformat(row["realtime_start_date"])
                end = date.fromisoformat(row["realtime_end_date"]) if row["realtime_end_date"] else date.max
                records[event].append(Record(start, end, float(row[name])))
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid CSV row {index}: {error}") from error
    if not records:
        raise ValueError("No ALFRED observations")
    return VintageSeries(name, records, vintage_dates)


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_series(raw_dir: Path, series: str) -> VintageSeries:
    raw_dir = Path(raw_dir)
    receipt = json.loads((raw_dir/"browser_fetch_receipt.json").read_text(encoding="utf-8"))
    if receipt["status"] != "PASS":
        raise ValueError("Official browser archive import has not passed")
    entry = receipt["series"][series]
    if file_sha(raw_dir/f"{series}.zip") != entry["archive"]["sha256"]:
        raise ValueError("Archive hash differs from import receipt")
    for member in entry["members"]:
        if file_sha(raw_dir/series/member["name"]) != member["sha256"]:
            raise ValueError("Extracted member hash differs from import receipt")
    readme = (raw_dir/series/"README.txt").read_text(encoding="utf-8-sig")
    if f"Series ID: {series}" not in readme or "Output Format: Observations by Real-Time Period" not in readme:
        raise ValueError("Wrong series or output type in README")
    vintage_text = readme.split("Vintage Dates Specified:", 1)[1]
    vintages = tuple(date.fromisoformat(v) for v in re.findall(r"^\d{4}-\d{2}-\d{2}$", vintage_text, re.MULTILINE))
    expected = sorted(set(entry["request_fields"]["form[selected_vintage_dates][]"]) | {LOWER.isoformat(), UPPER.isoformat()})
    if [v.isoformat() for v in vintages] != expected:
        raise ValueError("README vintage selection differs from complete request")
    return parse_csv(raw_dir/series/"obs._by_real-time_period.csv", series, vintages)


def raw_qc(raw_dir: Path) -> dict:
    report = {"status": "PASS", "scope": "raw archive/schema/interval checks only; derived causal coverage is checked separately",
              "source_sha256": file_sha(Path(__file__)), "series": {}}
    for name in ("PAYEMS", "INDPRO"):
        series = load_series(raw_dir, name)
        expected = tuple(range(1989*12+11, 2024*12+12))
        if series.events != expected:
            raise ValueError(f"Missing or out-of-range event months: {name}")
        rows = [r for t in series.events for r in series.records(t)]
        gaps = []
        for event in series.events:
            for a, b in zip(series.records(event), series.records(event)[1:]):
                if a.end < UPPER and a.end + timedelta(days=1) < b.start:
                    gaps.append({"event": event, "after": a.end.isoformat(), "before": b.start.isoformat()})
        if gaps:
            raise ValueError(f"Missing intervals inside complete vintage request: {name}, {gaps[:3]}")
        report["series"][name] = {"csv_rows": len(rows), "event_months": len(series.events),
                                  "event_start": "1989-12-01", "event_end": "2024-12-01",
                                  "requested_vintages_including_boundaries": len(series.vintage_dates),
                                  "earliest_record_start": min(r.start for r in rows).isoformat(),
                                  "latest_record_start": max(r.start for r in rows).isoformat(),
                                  "open_ended_intervals": sum(r.end == date.max for r in rows),
                                  "finite_ends_after_cutoff": sum(UPPER < r.end < date.max for r in rows),
                                  "overlapping_intervals": 0, "internal_interval_gaps": len(gaps),
                                  "invalid_levels": 0, "readme_vintage_selection_verified": True,
                                  "csv_sha256": file_sha(raw_dir/name/"obs._by_real-time_period.csv")}
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=ROOT/"data_external/alfred_revision_entry_v1/complete")
    parser.add_argument("--output", type=Path, default=ROOT/"results/peft_revision_entry_v1/data_qc.json")
    args = parser.parse_args()
    try:
        result = raw_qc(args.raw_dir)
    except Exception as error:
        result = {"status": "FAIL", "error": f"{type(error).__name__}: {error}", "source_sha256": file_sha(Path(__file__))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "PASS" else 1)
