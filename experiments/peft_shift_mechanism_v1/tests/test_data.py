import json
from pathlib import Path

import numpy as np

from experiments.peft_shift_mechanism_v1 import data


def test_generator_shapes_pickle_free_and_qc(tmp_path: Path):
    summary = data.generate(tmp_path)
    assert len(summary["files"]) == 12
    assert summary["all_qc_passed"] is True

    path = tmp_path / "Q00_c0.npz"
    with np.load(path, allow_pickle=False) as archive:
        assert archive["context_train"].shape == (64, 3, 256)
        assert archive["context_val"].shape == (128, 3, 256)
        assert archive["context_eval"].shape == (512, 3, 256)
        assert archive["target_train"].shape == (64, 16)
        assert archive["oracle_mean_eval"].shape == (512, 16)
        assert archive["oracle_quantiles_eval"].shape == (512, 21, 16)
        assert archive["raw_features_eval"].shape == (512, 16, 9)
        assert archive["context_train"].dtype == np.float32
        assert archive["target_train"].dtype == np.float32
        assert archive["manifest_json"].dtype.kind == "U"
        np.testing.assert_allclose(archive["quantiles"], data.QUANTILES)
        manifest = json.loads(str(archive["manifest_json"].item()))

    assert manifest["input_scope"]["future_observed_inputs"] == []
    assert manifest["input_scope"]["future_supervision"] == ["Y_future"]


def test_raw_feature_alignment_and_oracle_residual(tmp_path: Path):
    data.generate(tmp_path)
    with np.load(tmp_path / "Q10_c1.npz", allow_pickle=False) as archive:
        context = archive["context_eval"]
        target = archive["target_eval"]
        oracle = archive["oracle_mean_eval"]
        raw = archive["raw_features_eval"]

    for lead in range(data.H):
        np.testing.assert_allclose(raw[:, lead, 0], context[:, 0, data.L + lead - 32])
        np.testing.assert_allclose(raw[:, lead, 1], context[:, 0, data.L + lead - 48])
        np.testing.assert_allclose(raw[:, lead, 2], context[:, 0, data.L + lead - 64])
        np.testing.assert_allclose(raw[:, lead, 3], context[:, 1, data.L + lead - 32])
        np.testing.assert_allclose(raw[:, lead, 6], context[:, 2, data.L + lead - 32])

    residual = target - oracle
    assert abs(float(residual.mean())) < 0.06
    assert abs(float(residual.var()) - data.SIGMA**2) < 0.08


def test_common_random_numbers_and_shared_validation_eval(tmp_path: Path):
    data.generate(tmp_path)
    with np.load(tmp_path / "Q00_c0.npz", allow_pickle=False) as q00, np.load(tmp_path / "Q01_c0.npz", allow_pickle=False) as q01:
        np.testing.assert_allclose(q00["context_train"][:, 1:], q01["context_train"][:, 1:])
        assert not np.allclose(q00["context_train"][:, 0], q01["context_train"][:, 0])

    with np.load(tmp_path / "Q11_c0.npz", allow_pickle=False) as c0, np.load(tmp_path / "Q11_c2.npz", allow_pickle=False) as c2:
        np.testing.assert_allclose(c0["context_val"], c2["context_val"])
        np.testing.assert_allclose(c0["target_eval"], c2["target_eval"])
        assert not np.allclose(c0["context_train"], c2["context_train"])
        assert set(c0["episode_ids_train"]).isdisjoint(set(c0["episode_ids_val"]))
        assert set(c0["episode_ids_train"]).isdisjoint(set(c0["episode_ids_eval"]))


def test_deterministic_regeneration(tmp_path: Path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    data.generate(first)
    data.generate(second)
    for path in sorted(first.glob("*.npz")):
        with np.load(path, allow_pickle=False) as a, np.load(second / path.name, allow_pickle=False) as b:
            for key in a.files:
                np.testing.assert_array_equal(a[key], b[key])
