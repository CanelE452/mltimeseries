from datetime import datetime, timedelta

import numpy as np

from experiments.peft_external_gap_v1 import data


def _hourly_timestamps(start: datetime, hours: int) -> list[datetime]:
    return [start + timedelta(hours=i) for i in range(hours)]


def test_temporal_contract_counts_origins_and_keeps_targets_inside_each_split():
    timestamps = _hourly_timestamps(datetime(2020, 1, 1), 220 * 24)

    contract = data.temporal_contract(timestamps)

    assert contract["boundaries"]["precontext"] == (datetime(2020, 1, 1), datetime(2020, 1, 15))
    assert len(contract["origins"]["train"]) == 63
    assert len(contract["origins"]["val"]) == 13
    assert len(contract["origins"]["cal"]) == 13
    assert len(contract["origins"]["eval"]) == 83
    for split in ("train", "val", "cal", "eval"):
        start, end = contract["boundaries"][split]
        for origin in contract["origins"][split]:
            assert start <= origin
            assert origin + timedelta(hours=data.HORIZON) <= end
    assert contract["origin_stride_hours"] == 24
    assert contract["target_window_overlap_hours"] == 24


def test_causal_context_fill_uses_only_past_values_and_leading_train_median():
    values = np.array(
        [
            [np.nan, np.nan],
            [1.0, np.nan],
            [np.nan, 5.0],
            [3.0, np.nan],
        ],
        dtype=np.float32,
    )
    fit_median = np.array([10.0, 20.0], dtype=np.float32)

    filled = data.causal_ffill_context(values, fit_median)

    np.testing.assert_allclose(filled[:, 0], [10.0, 1.0, 1.0, 3.0])
    np.testing.assert_allclose(filled[:, 1], [20.0, 20.0, 5.0, 5.0])
    assert np.isnan(values[0, 0])
    assert np.isnan(values[0, 1])


def test_household_hourly_mean_requires_45_finite_minutes_per_column():
    rows = []
    base = datetime(2020, 1, 1)
    for minute in range(60):
        rows.append(
            {
                "timestamp": base + timedelta(minutes=minute),
                "a": 1.0 if minute < 45 else np.nan,
                "b": 2.0 if minute < 44 else np.nan,
            }
        )

    hourly = data.aggregate_minute_rows_to_hourly(rows, ["a", "b"], min_finite_per_hour=45)

    assert hourly["timestamps"] == [base]
    assert hourly["finite_minutes"][0, 0] == 45
    assert hourly["finite_minutes"][0, 1] == 44
    assert hourly["values"][0, 0] == 1.0
    assert np.isnan(hourly["values"][0, 1])


def test_build_archives_omits_holdout_targets_from_fit_archive(tmp_path):
    start = datetime(2020, 1, 1)
    hours = 220 * 24
    timestamps = _hourly_timestamps(start, hours)
    raw_values = np.arange(hours * 5, dtype=np.float32).reshape(hours, 5)
    raw_values[400, 0] = np.nan

    summary = data.write_panel_archives(
        panel="demo",
        timestamps=timestamps,
        raw_values=raw_values,
        channels=["y0", "y1", "x0", "x1", "x2"],
        target_indices=[0, 1],
        output_dir=tmp_path,
        source={"source_path": "memory://demo", "source_sha256": "0" * 64},
    )

    assert summary["fit_path"].endswith("demo_fit.npz")
    assert summary["holdout_path"].endswith("demo_holdout.npz")
    with np.load(tmp_path / "demo_fit.npz", allow_pickle=False) as fit:
        assert "cal_origins" not in fit.files
        assert "eval_origins" not in fit.files
        assert fit["context_values"].shape[1] == 5
        assert np.isfinite(fit["context_values"]).all()
        assert np.isnan(fit["target_values"][400, 0])
        assert fit["target_loss_mask"].shape == fit["target_values"].shape
    with np.load(tmp_path / "demo_holdout.npz", allow_pickle=False) as holdout:
        assert "train_origins" not in holdout.files
        assert "val_origins" not in holdout.files
        assert "cal_origins" in holdout.files
        assert "eval_origins" in holdout.files
        assert holdout["target_indices"].tolist() == [0, 1]
