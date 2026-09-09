from datetime import datetime, timedelta
import json

import numpy as np

from experiments.peft_temporal_replication_v1 import data


def _hours(start, count):
    return [start + timedelta(hours=i) for i in range(count)]


def test_replication_crop_starts_190_days_after_first_complete_day_and_has_fixed_length():
    timestamps = _hours(datetime(2020, 1, 1), 500 * 24)
    values = np.arange(len(timestamps) * 3, dtype=np.float32).reshape(len(timestamps), 3)

    cropped_ts, cropped_values, meta = data.crop_temporal_replication_block(timestamps, values)

    assert cropped_ts[0] == datetime(2020, 7, 9)
    assert cropped_ts[-1] == datetime(2021, 1, 14, 23)
    assert cropped_values.shape == (190 * 24, 3)
    assert meta["previous_block"] == ["2020-01-01T00:00:00", "2020-07-09T00:00:00"]
    assert meta["replication_block"] == ["2020-07-09T00:00:00", "2021-01-15T00:00:00"]
    assert meta["raw_timestep_overlap_with_previous_block"] == 0


def test_write_replication_archives_preserves_external_gap_schema_and_new_metadata(tmp_path):
    start = datetime(2020, 1, 1)
    timestamps = _hours(start, 500 * 24)
    t = np.arange(len(timestamps), dtype=np.float32)
    values = np.stack([1 + t / 1000, 2 + t / 1000, np.sin(t / 24), np.cos(t / 24), t % 24], axis=1).astype(np.float32)

    summary = data.write_replication_panel_archives(
        panel="demo",
        timestamps=timestamps,
        raw_values=values,
        channels=["y0", "y1", "x0", "x1", "x2"],
        target_indices=[0, 1],
        output_dir=tmp_path,
        source={"source_path": "memory://demo", "source_sha256": "0" * 64},
    )

    with np.load(tmp_path / "demo_fit.npz", allow_pickle=False) as fit:
        meta = json.loads(str(fit["manifest_json"].item()))
        assert meta["version"] == data.VERSION
        assert meta["dataset"] == "demo"
        assert meta["source"]["temporal_replication"]["replication_index"] == 1
        assert "cal_origins" not in fit.files
        assert fit["context_values"].shape == (2208, 5)
        assert fit["train_origins"].shape == (63,)
        assert fit["val_origins"].shape == (13,)
    with np.load(tmp_path / "demo_holdout.npz", allow_pickle=False) as holdout:
        assert "train_origins" not in holdout.files
        assert holdout["context_values"].shape == (4560, 5)
        assert holdout["cal_origins"].shape == (13,)
        assert holdout["eval_origins"].shape == (83,)
    assert summary["replication"]["replication_start"] == "2020-07-09T00:00:00"
