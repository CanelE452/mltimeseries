from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from experiments.peft_observation_operator_entry_v1 import data_qc as dq


class _FakeResponse:
    def __init__(self, chunks: list[bytes], *, content_length: int | None = None, fail_after_chunks: int | None = None):
        self._chunks = list(chunks)
        self._reads = 0
        self._fail_after_chunks = fail_after_chunks
        self.headers = {}
        if content_length is not None:
            self.headers["Content-Length"] = str(content_length)
        self.status = 200

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def geturl(self) -> str:
        return "https://example.test/uscrn.txt"

    def getcode(self) -> int:
        return 200

    def read(self, size: int) -> bytes:
        if self._fail_after_chunks is not None and self._reads >= self._fail_after_chunks:
            raise TimeoutError("fixture stream stopped")
        self._reads += 1
        if self._chunks:
            return self._chunks.pop(0)
        return b""


def _root_with_plan(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    plan = root / "_docs" / "notes" / "tsfm_topics" / "16_observation_operator_entry_plan_20260908.md"
    plan.parent.mkdir(parents=True)
    plan.write_text("fixed USCRN pre-data plan\n", encoding="utf-8")
    return root


def _hourly_row(stamp: datetime, t_calc: float, t_hr_avg: float) -> str:
    return (
        f"53152 {stamp:%Y%m%d} {stamp:%H%M} {stamp:%Y%m%d} {stamp:%H%M} "
        f"2.623 -110.950 32.233 {t_calc:.1f} {t_hr_avg:.1f} 9.9 0.0 0.0"
    )


def _subhourly_row(stamp: datetime, air_temperature: float) -> str:
    return (
        f"53152 {stamp:%Y%m%d} {stamp:%H%M} {stamp:%Y%m%d} {stamp:%H%M} "
        f"2.623 -110.950 32.233 {air_temperature:.1f} 0.0"
    )


def _fixture_bytes() -> dict[str, bytes]:
    start = datetime(2024, 1, 1, 0, 0)
    sub_rows: list[str] = [_subhourly_row(start, 0.5)]
    one = start + timedelta(hours=1)
    two = start + timedelta(hours=2)
    for minutes in range(5, 61, 5):
        sub_rows.append(_subhourly_row(start + timedelta(minutes=minutes), 1.0))
    for minutes in range(65, 121, 5):
        sub_rows.append(_subhourly_row(start + timedelta(minutes=minutes), 2.0))
    hourly_rows = [
        _hourly_row(start, 0.5, -9999.0),
        _hourly_row(one, 1.0, 1.0),
        _hourly_row(two, 2.0, 2.2),
    ]
    return {
        "hourly": ("\n".join(hourly_rows) + "\n").encode("ascii"),
        "subhourly": ("\n".join(sub_rows) + "\n").encode("ascii"),
    }


def test_run_writes_prefetch_contract_before_download_and_reports_temperature_support(tmp_path: Path):
    root = _root_with_plan(tmp_path)
    output = tmp_path / "external" / "uscrn_operator_entry_v1"
    payloads = _fixture_bytes()
    call_order: list[str] = []

    def fake_fetch(url: str, destination: Path, *, max_bytes: int, timeout_seconds: int, deadline_monotonic: float):
        assert (output / "contract.json").exists(), "prefetch contract must be durable before network fetch"
        key = "hourly" if "hourly02" in url else "subhourly"
        destination.write_bytes(payloads[key])
        call_order.append(key)
        return {
            "url": url,
            "response_url": url,
            "content_length_header": len(payloads[key]),
            "last_modified": "fixture",
            "bytes": len(payloads[key]),
            "sha256": hashlib.sha256(payloads[key]).hexdigest(),
            "reused_existing": False,
        }

    qc = dq.run(root=root, output_dir=output, fetcher=fake_fetch)

    contract = json.loads((output / "contract.json").read_text(encoding="utf-8"))
    assert call_order == ["hourly", "subhourly"]
    assert contract["expected_no_selection"] is True
    assert contract["sources"]["hourly"]["url"].endswith("CRNH0203-2024-AZ_Tucson_11_W.txt")
    assert contract["schema"]["support"]["hourly_T_CALC"] == "last_5_minutes_ending_at_UTC_TIME"
    assert qc["completed"] is True
    assert qc["grid"]["hourly_rows"] == 3
    assert qc["grid"]["subhourly_rows"] == 25
    assert qc["identity"]["hourly_wbannos"] == ["53152"]
    assert qc["identity"]["subhourly_wbannos"] == ["53152"]
    assert qc["identity"]["same_single_wban"] is True
    assert qc["identity"]["hourly_crx_versions"] == ["2.623"]
    assert qc["grid"]["hourly"]["off_step_count"] == 0
    assert qc["grid"]["subhourly"]["year_outlier_count"] == 0
    assert qc["hourly_vs_subhourly_t_calc"]["paired_valid_count"] == 3
    assert qc["hourly_vs_subhourly_t_calc"]["exact_equal_count"] == 3
    assert qc["subhourly_mean12_vs_hourly_t_hr_avg"]["complete_valid_hours"] == 2
    assert qc["subhourly_mean12_vs_hourly_t_hr_avg"]["first_hour_excluded_count"] == 1
    assert qc["subhourly_mean12_vs_hourly_t_hr_avg"]["abs_diff_ge_0_1_count"] == 1
    assert (output / "qc.json").exists()


def test_qc_reports_duplicates_and_missing_without_filling_values(tmp_path: Path):
    hourly = tmp_path / "hourly.txt"
    subhourly = tmp_path / "subhourly.txt"
    stamp = datetime(2024, 1, 1, 1, 0)
    hourly.write_text(_hourly_row(stamp, -9999.0, 4.2) + "\n", encoding="ascii")
    rows = [_subhourly_row(stamp - timedelta(minutes=55 - i * 5), 4.2) for i in range(12)]
    rows[-1] = _subhourly_row(stamp, -9999.0)
    rows.append(rows[0])
    subhourly.write_text("\n".join(rows) + "\n", encoding="ascii")

    qc = dq.compute_qc(hourly, subhourly)

    assert qc["duplicates"]["subhourly_duplicate_extra_rows"] == 1
    assert qc["missing"]["hourly_t_calc_missing"] == 1
    assert qc["missing"]["subhourly_air_temperature_missing"] == 1
    assert qc["hourly_vs_subhourly_t_calc"]["paired_valid_count"] == 0
    assert qc["subhourly_mean12_vs_hourly_t_hr_avg"]["incomplete_hours"] == 1
    assert qc["subhourly_mean12_vs_hourly_t_hr_avg"]["complete_valid_hours"] == 0


def test_run_rejects_partial_or_uncontracted_raw_without_overwrite(tmp_path: Path):
    root = _root_with_plan(tmp_path)
    output = tmp_path / "external" / "uscrn_operator_entry_v1"
    raw = output / "raw"
    raw.mkdir(parents=True)
    unknown = raw / "CRNH0203-2024-AZ_Tucson_11_W.txt"
    unknown.write_text("uncontracted local data\n", encoding="ascii")

    with pytest.raises(FileExistsError, match="uncontracted"):
        dq.run(root=root, output_dir=output, fetcher=lambda *args, **kwargs: None)

    assert unknown.read_text(encoding="ascii") == "uncontracted local data\n"

    output2 = tmp_path / "external" / "with_partial"
    raw2 = output2 / "raw"
    raw2.mkdir(parents=True)
    partial = raw2 / "CRNS0101-05-2024-AZ_Tucson_11_W.txt.part"
    partial.write_text("partial bytes\n", encoding="ascii")

    with pytest.raises(FileExistsError, match="partial"):
        dq.run(root=root, output_dir=output2, fetcher=lambda *args, **kwargs: None)

    assert partial.read_text(encoding="ascii") == "partial bytes\n"


def test_fetch_preserves_partial_bytes_when_stream_fails(monkeypatch, tmp_path: Path):
    target = tmp_path / "raw.txt"

    def fake_urlopen(url: str, timeout: int):
        return _FakeResponse([b"abc"], fail_after_chunks=1)

    monkeypatch.setattr(dq.urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(TimeoutError, match="fixture stream stopped"):
        dq._fetch_url("https://example.test/raw.txt", target, max_bytes=100, timeout_seconds=1)

    assert not target.exists()
    assert target.with_name("raw.txt.part").read_bytes() == b"abc"


def test_fetch_rejects_truncated_content_length_before_accepting_complete_file(monkeypatch, tmp_path: Path):
    target = tmp_path / "raw.txt"

    def fake_urlopen(url: str, timeout: int):
        return _FakeResponse([b"abc"], content_length=8)

    monkeypatch.setattr(dq.urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(ValueError, match="Content-Length"):
        dq._fetch_url("https://example.test/raw.txt", target, max_bytes=100, timeout_seconds=1)

    assert not target.exists()
    assert target.with_name("raw.txt.part").read_bytes() == b"abc"
