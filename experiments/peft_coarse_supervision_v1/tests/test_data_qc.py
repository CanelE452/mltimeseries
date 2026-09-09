from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _metadata_rows(site: str, count: int, timezone: str) -> list[dict[str, str]]:
    return [
        {
            "building_id": f"{site}_office_{idx:02d}",
            "site_id": site,
            "primaryspaceusage": "Office",
            "timezone": timezone,
            "electricity": "Yes",
        }
        for idx in range(count)
    ]


def _write_metadata(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "building_id",
                "site_id",
                "primaryspaceusage",
                "timezone",
                "electricity",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_raw(
    path: Path,
    building_ids: list[str],
    *,
    bad_2016: dict[str, float | str] | None = None,
    bad_2017: dict[str, float | str] | None = None,
    drop_timestamp: str | None = None,
) -> None:
    timestamps = pd.date_range("2016-01-01 00:00:00", "2017-12-31 23:00:00", freq="h")
    frame = pd.DataFrame({"timestamp": timestamps.strftime("%Y-%m-%d %H:%M:%S")})
    for col_index, building_id in enumerate(building_ids):
        frame[building_id] = np.full(len(frame), float(col_index + 1))
    if bad_2016:
        first_2016 = frame.index[frame["timestamp"] == "2016-02-01 00:00:00"][0]
        for building_id, value in bad_2016.items():
            if isinstance(value, str):
                frame[building_id] = frame[building_id].astype(object)
            frame.loc[first_2016, building_id] = value
    if bad_2017:
        first_2017 = frame.index[frame["timestamp"] == "2017-02-01 00:00:00"][0]
        for building_id, value in bad_2017.items():
            if isinstance(value, str):
                frame[building_id] = frame[building_id].astype(object)
            frame.loc[first_2017, building_id] = value
    if drop_timestamp:
        frame = frame[frame["timestamp"] != drop_timestamp]
    frame.to_csv(path, index=False)


def test_compute_qc_selects_sorted_train_complete_meters_and_ignores_2017_values(tmp_path: Path) -> None:
    from experiments.peft_coarse_supervision_v1.data_qc import compute_qc

    rows = _metadata_rows("Eagle", 10, "US/Eastern") + _metadata_rows(
        "Lamb", 10, "Europe/London"
    )
    metadata_path = tmp_path / "metadata.csv"
    raw_path = tmp_path / "electricity.csv"
    _write_metadata(metadata_path, rows)
    building_ids = [row["building_id"] for row in rows]
    _write_raw(raw_path, building_ids, bad_2017={building_ids[0]: -9999.0})

    qc = compute_qc(
        metadata_path,
        raw_path,
        expected_counts={"Eagle": 10, "Lamb": 10},
        expected_timezones={"Eagle": "US/Eastern", "Lamb": "Europe/London"},
    )

    assert qc["decision"] == "PASS"
    assert qc["timestamp_qc"]["rows_2016"] == 8784
    assert qc["timestamp_qc"]["rows_2017"] == 8760
    assert qc["candidate_train_2016_qc"][building_ids[0]]["eligible"] is True
    assert qc["candidate_train_2016_qc"][building_ids[0]]["negative_count"] == 0
    assert qc["selection"]["sites"]["Eagle"]["donor_ids"] == building_ids[:5]
    assert qc["selection"]["sites"]["Eagle"]["target_ids"] == building_ids[5:10]
    assert qc["selected_target_monthly_2016"]["months"] == [f"2016-{m:02d}" for m in range(1, 13)]
    assert len(qc["selected_target_monthly_2016"]["sites"]["Eagle"]["totals"]) == 12
    assert len(qc["selected_target_monthly_2016"]["sites"]["Eagle"]["totals"][0]) == 5


def test_compute_qc_rejects_metadata_count_or_timezone_mismatch(tmp_path: Path) -> None:
    from experiments.peft_coarse_supervision_v1.data_qc import compute_qc

    rows = _metadata_rows("Eagle", 8, "US/Eastern") + _metadata_rows(
        "Lamb", 8, "Europe/London"
    )
    metadata_path = tmp_path / "metadata.csv"
    raw_path = tmp_path / "electricity.csv"
    _write_metadata(metadata_path, rows)
    _write_raw(raw_path, [row["building_id"] for row in rows])

    with pytest.raises(ValueError, match="metadata count"):
        compute_qc(
            metadata_path,
            raw_path,
            expected_counts={"Eagle": 9, "Lamb": 8},
            expected_timezones={"Eagle": "US/Eastern", "Lamb": "Europe/London"},
        )


def test_compute_qc_rejects_non_contiguous_timestamp_grid(tmp_path: Path) -> None:
    from experiments.peft_coarse_supervision_v1.data_qc import compute_qc

    rows = _metadata_rows("Eagle", 8, "US/Eastern") + _metadata_rows(
        "Lamb", 8, "Europe/London"
    )
    metadata_path = tmp_path / "metadata.csv"
    raw_path = tmp_path / "electricity.csv"
    _write_metadata(metadata_path, rows)
    _write_raw(raw_path, [row["building_id"] for row in rows], drop_timestamp="2016-07-01 00:00:00")

    with pytest.raises(ValueError, match="timestamp grid"):
        compute_qc(
            metadata_path,
            raw_path,
            expected_counts={"Eagle": 8, "Lamb": 8},
            expected_timezones={"Eagle": "US/Eastern", "Lamb": "Europe/London"},
        )


def test_compute_qc_reports_insufficient_without_lowering_train_only_criterion(tmp_path: Path) -> None:
    from experiments.peft_coarse_supervision_v1.data_qc import compute_qc

    rows = _metadata_rows("Eagle", 8, "US/Eastern") + _metadata_rows(
        "Lamb", 8, "Europe/London"
    )
    metadata_path = tmp_path / "metadata.csv"
    raw_path = tmp_path / "electricity.csv"
    _write_metadata(metadata_path, rows)
    building_ids = [row["building_id"] for row in rows]
    _write_raw(
        raw_path,
        building_ids,
        bad_2016={
            "Eagle_office_00": "",
            "Eagle_office_01": -1.0,
            "Eagle_office_02": "",
        },
    )

    qc = compute_qc(
        metadata_path,
        raw_path,
        expected_counts={"Eagle": 8, "Lamb": 8},
        expected_timezones={"Eagle": "US/Eastern", "Lamb": "Europe/London"},
    )

    assert qc["decision"] == "INSUFFICIENT_COMPLETE_TRAIN_METERS"
    assert qc["selection"]["sites"]["Eagle"]["n"] == 2
    assert qc["candidate_train_2016_qc"]["Eagle_office_00"]["missing_count"] == 1
    assert qc["candidate_train_2016_qc"]["Eagle_office_01"]["negative_count"] == 1
