import json
from pathlib import Path

import numpy as np

from experiments.peft_shift_mechanism_v1 import data as original_data
from experiments.peft_trainlag_v1 import data


def test_select_lags_uses_train_target_and_tie_breaks_to_smaller_lag():
    rng = np.random.default_rng(7)
    context = rng.normal(size=(20, 3, data.L)).astype(np.float32)
    target = rng.normal(size=(20, data.H)).astype(np.float32)
    leads = np.arange(data.H)
    for channel, lag in enumerate((20, 33, 64)):
        context[:, channel, data.L + leads - lag] = target
    context[:, 0, data.L + leads - 40] = target

    selection = data.select_lags_from_train(context, target, candidate_lags=np.arange(16, 66))

    assert selection["selected_lags"] == {"Y": 20, "U": 33, "V": 64}
    assert selection["n_future_labels"] == 20 * data.H
    assert selection["input_scope"]["validation_eval_oracle_manifest_used"] is False
    assert selection["input_scope"]["context_y_used_as_extra_label"] is False


def test_make_lag_features_alignment_and_fully_observed_guard():
    context = np.arange(2 * 3 * data.L, dtype=np.float32).reshape(2, 3, data.L)
    features = data.make_lag_features(context, {"Y": 16, "U": 32, "V": 128})
    assert features.shape == (2, data.H, 3)
    for lead in range(data.H):
        np.testing.assert_array_equal(features[:, lead, 0], context[:, 0, data.L + lead - 16])
        np.testing.assert_array_equal(features[:, lead, 1], context[:, 1, data.L + lead - 32])
        np.testing.assert_array_equal(features[:, lead, 2], context[:, 2, data.L + lead - 128])
    try:
        data.make_lag_features(context, {"Y": 15, "U": 32, "V": 48})
    except ValueError as error:
        assert "fully observed" in str(error)
    else:
        raise AssertionError("lag 15 would require future input at the last horizon step")


def test_generate_writes_three_fresh_q00_archives_and_lag_json(tmp_path: Path):
    summary = data.generate(tmp_path)
    assert len(summary["files"]) == 3
    assert len(summary["lag_selection_files"]) == 3
    assert summary["all_qc_passed"] is True
    assert summary["fresh_same_family_not_original_source"] is True

    with np.load(tmp_path / "Q00_c0.npz", allow_pickle=False) as archive:
        assert archive["context_train"].shape == (64, 3, 256)
        assert archive["context_val"].shape == (128, 3, 256)
        assert archive["context_eval"].shape == (512, 3, 256)
        assert archive["target_train"].shape == (64, 16)
        assert archive["oracle_quantiles_eval"].shape == (512, 21, 16)
        assert archive["raw_features_eval"].shape == (512, 16, 3)
        assert archive["selected_lags"].shape == (3,)
        assert archive["manifest_json"].dtype.kind == "U"
        assert str(archive["episode_ids_train"][0]).startswith("trainlag_b2026090810_train_c00")
        assert str(archive["episode_ids_val"][0]).startswith("trainlag_b2026090810_val_c00")
        manifest = json.loads(str(archive["manifest_json"].item()))

    selection = json.loads((tmp_path / "Q00_c0_lags.json").read_text(encoding="utf-8"))
    assert {"input_array_hash", "selected_lags", "candidates", "candidate_lags", "scores", "fit_seconds", "data_sha256"} <= set(selection)
    assert selection["candidates"] == selection["candidate_lags"]
    assert selection["known_lag_dictionary_used"] is False
    assert selection["validation_or_eval_used"] is False
    assert selection["oracle_or_manifest_used"] is False
    assert manifest["input_scope"]["raw_feature_order"] == "selected train-only lag features in channel order Y,U,V"
    assert manifest["episode_id_namespace"] == "trainlag_b2026090810"
    assert manifest["theory"]["true_lags_for_diagnostic_only"] == {"Y": 32, "U": 48, "V": 48}


def test_fresh_seed_is_different_but_validation_and_eval_are_shared(tmp_path: Path):
    data.generate(tmp_path)
    fresh = np.load(tmp_path / "Q00_c0.npz", allow_pickle=False)
    original = original_data.build_archive("Q00", 0)
    assert not np.allclose(fresh["context_train"], original["context_train"])
    fresh.close()

    with np.load(tmp_path / "Q00_c0.npz", allow_pickle=False) as c0, np.load(tmp_path / "Q00_c2.npz", allow_pickle=False) as c2:
        np.testing.assert_allclose(c0["context_val"], c2["context_val"])
        np.testing.assert_allclose(c0["target_eval"], c2["target_eval"])
        assert not np.allclose(c0["context_train"], c2["context_train"])
        assert set(c0["episode_ids_train"]).isdisjoint(set(c0["episode_ids_val"]))
        assert set(c0["episode_ids_train"]).isdisjoint(set(c0["episode_ids_eval"]))
