import json
from pathlib import Path

import numpy as np

from experiments.peft_external_gap_v1 import raw


def test_direct_ridge_recovers_known_linear_rule_with_masks():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(40, 6))
    coef = np.array([0.5, -0.25, 0.1, 0.0, 0.2, -0.3])
    y = 1.5 + x @ coef
    y_windows = np.stack([y, y + 2.0], axis=1)[:, :, None]
    mask = np.ones_like(y_windows, dtype=bool)
    mask[:3, 1, 0] = False

    model = raw.fit_direct_ridge(x, y_windows, mask, alpha=0.0)
    pred = raw.predict_direct_ridge(model, x[:5])

    np.testing.assert_allclose(pred[:5, 0, 0], y[:5], atol=1e-10)
    np.testing.assert_allclose(pred[:5, 1, 0], y[:5] + 2.0, atol=1e-10)


def test_fit_candidate_predictions_do_not_depend_on_validation_targets():
    rng = np.random.default_rng(5)
    x_train = rng.normal(size=(30, 4))
    x_val = rng.normal(size=(8, 4))
    y_train = rng.normal(size=(30, 2, 3))
    y_val = rng.normal(size=(8, 2, 3))
    mask_train = np.ones_like(y_train, dtype=bool)
    mask_val = np.ones_like(y_val, dtype=bool)
    quantiles = np.array([0.1, 0.5, 0.9], dtype=np.float64)

    a = raw.fit_candidate(x_train, y_train, mask_train, x_val, y_val, mask_val, alpha=0.1, quantiles=quantiles)
    b = raw.fit_candidate(x_train, y_train, mask_train, x_val, y_val + 1000.0, mask_val, alpha=0.1, quantiles=quantiles)

    np.testing.assert_allclose(a["val_predictions"], b["val_predictions"])
    assert a["val_score"] != b["val_score"]


def test_run_refuses_nonempty_output_and_writes_expected_keys(tmp_path: Path):
    fit_path = tmp_path / "demo_fit.npz"
    holdout_path = tmp_path / "demo_holdout.npz"
    output = tmp_path / "raw"
    raw._write_fixture_archives(fit_path, holdout_path)
    output.mkdir()
    (output / "keep.txt").write_text("x", encoding="utf-8")

    try:
        raw.run(fit_path, holdout_path, output)
    except FileExistsError as error:
        assert "non-empty" in str(error)
    else:
        raise AssertionError("run must refuse to overwrite a non-empty output directory")

    output2 = tmp_path / "raw2"
    result = raw.run(fit_path, holdout_path, output2)
    assert result["completed"] is True
    assert result["selected_lambda"] in [0.1, 10.0, 1000.0]
    assert result["fit_data_sha256"]
    assert result["holdout_data_sha256"]
    with np.load(output2 / "predictions.npz", allow_pickle=False) as pred:
        expected = {
            "val_predictions", "cal_predictions", "eval_predictions",
            "val_target", "cal_target", "eval_target",
            "val_origins", "cal_origins", "eval_origins",
            "quantiles", "target_indices",
        }
        assert expected <= set(pred.files)
        assert pred["val_predictions"].shape[1] == 2
    saved = json.loads((output2 / "result.json").read_text(encoding="utf-8"))
    assert saved["validation_target_used_for_fit"] is False
    assert saved["cal_eval_target_used_for_selection"] is False
