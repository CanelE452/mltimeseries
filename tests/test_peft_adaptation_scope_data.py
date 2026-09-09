from pathlib import Path

import numpy as np

from experiments.peft_adaptation_scope_v1 import data
from experiments.peft_adaptation_scope_v1.modeling import forecast_scores


def test_origin_splits_are_temporal_and_leakage_checked(tmp_path):
    manifest = data.prepare(Path.cwd(), tmp_path)
    for panel_name in ("ettm2", "jena"):
        panel = data.load_prepared(tmp_path / f"{panel_name}.npz")
        split = panel.manifest["split_rows"]
        assert panel.train_origins.min() >= split["fit"][0]
        assert panel.train_origins.max() + panel.horizon <= split["fit"][1]
        assert panel.val_origins.min() >= split["validation"][0]
        assert panel.val_origins.max() + panel.horizon <= split["validation"][1]
        assert panel.eval_origins.min() >= split["evaluation"][0]
        assert panel.eval_origins.max() + panel.horizon <= split["evaluation"][1]
        assert np.isfinite(panel.fit_std).all()
        assert panel.values.shape[1] == len(panel.channels)
        with np.load(tmp_path / f"{panel_name}.npz", allow_pickle=False) as archive:
            assert archive["manifest_json"].dtype.kind == "U"
    assert manifest["panels"]["ettm2"]["origin_counts"] == {"train": 379, "validation": 16, "evaluation": 32}
    assert manifest["panels"]["jena"]["origin_counts"] == {"train": 568, "validation": 16, "evaluation": 32}


def test_forecast_score_target_macro_handles_missing_values():
    q = np.asarray([0.1, 0.5, 0.9], dtype=np.float32)
    y = np.asarray([[[1.0, np.nan], [2.0, 3.0]]], dtype=np.float32)
    pred = np.repeat(y[:, :, None, :], len(q), axis=2)
    pred[~np.isfinite(pred)] = 0.0
    metrics, per = forecast_scores(pred, y, np.ones(2, dtype=np.float32), q)
    assert metrics["scaled_2pinball"] >= 0.0
    assert "per_channel_scaled_2pinball" in metrics
    assert per.shape == (1, 2)
