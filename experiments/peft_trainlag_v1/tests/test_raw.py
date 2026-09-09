import json
from pathlib import Path

import numpy as np

from experiments.peft_trainlag_v1 import data, raw


def test_ols_fit_recovers_known_three_feature_linear_rule():
    rng = np.random.default_rng(11)
    features = rng.normal(size=(12, data.H, 3))
    coefficient = np.array([0.5, 0.25, -0.75])
    target = 1.25 + np.tensordot(features, coefficient, axes=([-1], [0]))

    fit = raw.ols_fit(features, target)

    assert abs(fit["intercept"] - 1.25) < 1e-10
    np.testing.assert_allclose(fit["coefficient"], coefficient, atol=1e-10)


def test_fit_raw_predictions_do_not_depend_on_val_eval_targets():
    archive = data.build_archive(0)
    selection = data.select_lags_from_train(archive["context_train"], archive["target_train"])
    features = {
        split: data.make_lag_features(archive[f"context_{split}"], selection["selected_lags"])
        for split in ("train", "val", "eval")
    }
    quantiles = archive["quantiles"].astype(np.float64)
    original = raw.fit_raw(features, archive["target_train"], quantiles)
    perturbed_targets = {
        "train": archive["target_train"].copy(),
        "val": np.full_like(archive["target_val"], 1234.0),
        "eval": np.full_like(archive["target_eval"], -4321.0),
    }
    perturbed = raw.fit_raw(features, perturbed_targets["train"], quantiles)

    np.testing.assert_array_equal(original["predictions"]["val"], perturbed["predictions"]["val"])
    np.testing.assert_array_equal(original["predictions"]["eval"], perturbed["predictions"]["eval"])


def test_run_writes_raw_trainlag_outputs_without_oof_or_hpo(tmp_path: Path):
    generated = tmp_path / "data"
    outputs = tmp_path / "raw"
    data.generate(generated)

    result = raw.run(generated / "Q00_c0.npz", generated / "Q00_c0_lags.json", outputs)

    assert result["completed"] is True
    assert result["known_lag_dictionary_used"] is False
    assert result["no_oof"] is True
    assert result["no_hpo"] is True
    assert result["validation_target_used_for_fit"] is False
    assert result["data_sha256"]
    assert result["lag_selection_sha256"]
    assert result["source_sha256"]
    assert "in-sample optimistic" in result["limitation"]

    with np.load(outputs / "predictions.npz", allow_pickle=False) as predictions:
        assert predictions["train_predictions"].shape == (64, 21, 16)
        assert predictions["val_predictions"].shape == (128, 21, 16)
        assert predictions["eval_predictions"].shape == (512, 21, 16)
        assert predictions["coefficient"].shape == (3,)
        np.testing.assert_array_equal(
            predictions["val_episode_ids"][:3],
            np.asarray([
                "trainlag_b2026090810_val_c00_e00000",
                "trainlag_b2026090810_val_c00_e00001",
                "trainlag_b2026090810_val_c00_e00002",
            ]),
        )

    saved = json.loads((outputs / "result.json").read_text(encoding="utf-8"))
    assert saved["residual_distribution"] == "empirical full-train residual quantiles; no OOF; in-sample optimistic"


def test_run_refuses_to_overwrite_nonempty_output(tmp_path: Path):
    generated = tmp_path / "data"
    outputs = tmp_path / "raw"
    data.generate(generated)
    outputs.mkdir()
    (outputs / "keep.txt").write_text("preserve me", encoding="utf-8")
    try:
        raw.run(generated / "Q00_c0.npz", generated / "Q00_c0_lags.json", outputs)
    except FileExistsError as error:
        assert "non-empty" in str(error)
    else:
        raise AssertionError("RAW run must refuse to overwrite a non-empty output directory")


def test_load_lag_selection_rejects_out_of_scope_declarations(tmp_path: Path):
    selection = data.select_lags_from_train(
        np.zeros((2, 3, data.L), dtype=np.float32),
        np.zeros((2, data.H), dtype=np.float32),
    )
    selection["data_sha256"] = "not-used-in-this-test"
    path = tmp_path / "lag.json"
    selection["oracle_or_manifest_used"] = True
    path.write_text(json.dumps(selection), encoding="utf-8")
    try:
        raw.load_lag_selection(path)
    except AssertionError as error:
        assert "out-of-scope" in str(error)
    else:
        raise AssertionError("lag selection that declares oracle use must be rejected")
