from datetime import datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.peft_external_gap_v1 import train as shared_train
from experiments.peft_fullft_reference_v1 import data


def _hourly_timestamps(start: datetime, hours: int) -> list[datetime]:
    return [start + timedelta(hours=index) for index in range(hours)]


def _values(hours: int, channels: int) -> np.ndarray:
    time = np.arange(hours, dtype=np.float32)
    columns = [time + (column + 1) * 100 for column in range(channels)]
    return np.stack(columns, axis=1)


def test_fixed_temporal_contract_starts_after_exposed_blocks_and_counts_daily_origins():
    expected = {
        "bike": {
            "precontext": ("2012-01-16T00:00:00", "2012-01-30T00:00:00"),
            "train": ("2012-01-30T00:00:00", "2012-04-30T00:00:00"),
            "embargo_train_val": ("2012-04-30T00:00:00", "2012-05-02T00:00:00"),
            "val": ("2012-05-02T00:00:00", "2012-06-02T00:00:00"),
            "embargo_val_cal": ("2012-06-02T00:00:00", "2012-06-04T00:00:00"),
            "cal": ("2012-06-04T00:00:00", "2012-06-25T00:00:00"),
            "eval": ("2012-06-25T00:00:00", "2012-09-14T00:00:00"),
        },
        "household": {
            "precontext": ("2008-01-01T00:00:00", "2008-01-15T00:00:00"),
            "train": ("2008-01-15T00:00:00", "2008-04-15T00:00:00"),
            "embargo_train_val": ("2008-04-15T00:00:00", "2008-04-17T00:00:00"),
            "val": ("2008-04-17T00:00:00", "2008-05-18T00:00:00"),
            "embargo_val_cal": ("2008-05-18T00:00:00", "2008-05-20T00:00:00"),
            "cal": ("2008-05-20T00:00:00", "2008-06-10T00:00:00"),
            "eval": ("2008-06-10T00:00:00", "2008-08-30T00:00:00"),
        },
    }

    for panel, boundaries in expected.items():
        contract = data.temporal_contract(panel)

        assert {
            name: tuple(value.isoformat(timespec="seconds") for value in pair)
            for name, pair in contract["boundaries"].items()
        } == boundaries
        assert {name: len(values) for name, values in contract["origins"].items()} == {
            "train": 90,
            "val": 30,
            "cal": 20,
            "eval": 80,
        }
        assert contract["origin_stride_hours"] == 24
        assert contract["target_window_overlap_hours"] == 24
        for split, origins in contract["origins"].items():
            start, end = contract["boundaries"][split]
            assert origins[0] == start
            assert origins[-1] + timedelta(hours=data.HORIZON) == end
            assert all(origin + timedelta(hours=data.HORIZON) <= end for origin in origins)
            assert all((right - left) == timedelta(hours=24) for left, right in zip(origins, origins[1:]))

        assert contract["origins"]["train"][-1] + timedelta(hours=data.HORIZON) <= contract["boundaries"]["val"][0]
        assert contract["origins"]["val"][-1] + timedelta(hours=data.HORIZON) <= contract["boundaries"]["cal"][0]
        assert contract["origins"]["cal"][-1] + timedelta(hours=data.HORIZON) <= contract["boundaries"]["eval"][0]


def test_context_imputation_uses_precontext_median_for_leading_gaps_then_only_past_values():
    values = np.array(
        [
            [np.nan, np.nan],
            [2.0, 10.0],
            [4.0, np.nan],
            [np.nan, 30.0],
            [np.nan, np.nan],
            [1000.0, 1000.0],
        ],
        dtype=np.float32,
    )

    filled, fallback = data.causal_context_values(values, (0, 4))

    np.testing.assert_allclose(fallback, [3.0, 20.0])
    np.testing.assert_allclose(filled[:, 0], [3.0, 2.0, 4.0, 4.0, 4.0, 1000.0])
    np.testing.assert_allclose(filled[:, 1], [20.0, 10.0, 10.0, 30.0, 30.0, 1000.0])

    changed_future = values.copy()
    changed_future[5] = [9999.0, 9999.0]
    changed_filled, changed_fallback = data.causal_context_values(changed_future, (0, 4))
    np.testing.assert_array_equal(changed_fallback, fallback)
    np.testing.assert_array_equal(changed_filled[:5], filled[:5])


def test_write_archives_uses_train_interval_statistics_and_shared_panel_keys(tmp_path):
    hours = data.BLOCK_DAYS * 24
    timestamps = _hourly_timestamps(datetime(2012, 1, 16), hours)
    raw_values = _values(hours, 5)
    raw_values[0, 2] = np.nan
    raw_values[400, 0] = np.nan
    raw_values[data.BLOCK_DAYS * 24 - 10 :, 0] += 1_000_000

    summary = data.write_panel_archives(
        panel="bike",
        timestamps=timestamps,
        raw_values=raw_values,
        channels=["casual", "registered", "temp", "hum", "windspeed"],
        target_indices=[0, 1],
        output_dir=tmp_path,
        source={"source_path": "memory://bike", "source_sha256": "0" * 64, "raw_sha256": "0" * 64},
    )

    assert summary["origin_counts"] == {"train": 90, "val": 30, "cal": 20, "eval": 80}
    assert summary["fit_shape"] == [3312, 5]
    assert summary["holdout_shape"] == [5808, 5]
    train_start, train_end = summary["boundary_indices"]["train"]
    expected_train = raw_values[train_start:train_end].astype(np.float64)
    np.testing.assert_allclose(summary["fit_mean"], np.nanmean(expected_train, axis=0).astype(np.float32))
    np.testing.assert_allclose(summary["fit_std"], np.nanstd(expected_train, axis=0).astype(np.float32))
    np.testing.assert_allclose(summary["fit_median"], np.nanmedian(expected_train, axis=0).astype(np.float32))

    with np.load(tmp_path / "bike_fit.npz", allow_pickle=False) as fit:
        assert "cal_origins" not in fit.files
        assert "eval_origins" not in fit.files
        assert fit["context_values"].shape == (3312, 5)
        assert fit["target_values"].shape == (3312, 5)
        assert np.isfinite(fit["context_values"]).all()
        assert np.isnan(fit["target_values"][400, 0])
        np.testing.assert_array_equal(fit["observed_mask"], np.isfinite(fit["target_values"]))
        expected_loss_mask = np.zeros_like(fit["target_values"], dtype=bool)
        expected_loss_mask[:, [0, 1]] = np.isfinite(fit["target_values"][:, [0, 1]])
        np.testing.assert_array_equal(fit["target_loss_mask"], expected_loss_mask)

    fit_panel = shared_train.Panel(tmp_path / "bike_fit.npz", "fit")
    assert fit_panel.dataset == "bike"
    assert {name: len(origins) for name, origins in fit_panel.origins.items()} == {"train": 90, "val": 30}

    with np.load(tmp_path / "bike_holdout.npz", allow_pickle=False) as holdout:
        assert "train_origins" not in holdout.files
        assert "val_origins" not in holdout.files
        assert set(["cal_origins", "eval_origins"]).issubset(holdout.files)
        assert holdout["context_values"].shape == (5808, 5)
        assert holdout["target_values"].shape == (5808, 5)

    holdout_panel = shared_train.Panel(tmp_path / "bike_holdout.npz", "holdout")
    assert {name: len(origins) for name, origins in holdout_panel.origins.items()} == {"cal": 20, "eval": 80}


def test_prepare_manifest_records_paths_hashes_stats_and_refuses_overwrite(tmp_path, monkeypatch):
    def fake_build_panel(panel):
        channels = 5 if panel == "bike" else 4
        start = data.PANEL_STARTS[panel]
        spec = SimpleNamespace(name=panel, channels=tuple(f"c{i}" for i in range(channels)), target_indices=(0, 1))
        return spec, _hourly_timestamps(start, data.BLOCK_DAYS * 24), _values(data.BLOCK_DAYS * 24, channels), {
            "source_rows": data.BLOCK_DAYS * 24,
            "missing_hours_inserted": 0,
        }

    def fake_source_manifest(spec):
        return {
            "panel": spec.name,
            "source_path": f"memory://{spec.name}",
            "source_exists": True,
            "source_sha256": spec.name * 8,
            "raw_sha256": spec.name * 8,
            "source_qc": {"source_rows": data.BLOCK_DAYS * 24},
        }

    monkeypatch.setattr(data, "build_panel", fake_build_panel)
    monkeypatch.setattr(data, "source_manifest", fake_source_manifest)

    manifest = data.prepare(tmp_path)

    assert manifest["all_qc_passed"]
    assert manifest["data_contract"]["fit_archives"].startswith("train+validation")
    assert set(manifest["datasets"]) == {"bike", "household"}
    bike = manifest["datasets"]["bike"]
    household = manifest["datasets"]["household"]
    assert bike["fit_data_path"].endswith("bike_fit.npz")
    assert bike["holdout_data_path"].endswith("bike_holdout.npz")
    assert len(bike["fit_data_sha256"]) == 64
    assert len(household["holdout_data_sha256"]) == 64
    assert bike["target_indices"] == [0, 1]
    assert bike["channels"] == ["c0", "c1", "c2", "c3", "c4"]
    assert len(household["fit_std"]) == 4
    assert bike["origin_counts"] == {"train": 90, "val": 30, "cal": 20, "eval": 80}
    assert manifest["panels"]["bike"]["source"]["raw_sha256"] == "bike" * 8
    assert manifest["panels"]["bike"]["contract"]["origin_counts"]["eval"] == 80
    assert (tmp_path / "manifest.json").exists()

    with pytest.raises(FileExistsError):
        data.prepare(tmp_path)
