from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
import pytest


def test_dual_ridge_matches_primal_solution() -> None:
    from experiments.peft_coarse_supervision_v1.ridge import _fit_ridge_dual

    x = np.array([[1.0, 2.0, -1.0], [0.5, -0.5, 3.0], [2.0, 0.0, 1.0]], dtype=np.float64)
    y = np.array([1.5, -2.0, 0.25], dtype=np.float64)
    alpha = 0.7

    got = _fit_ridge_dual(x, y, alpha)
    expected = np.linalg.solve(x.T @ x + alpha * np.eye(x.shape[1]), x.T @ y)
    np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-10)


def test_head_design_matches_monthly_mean_of_linear_point_head() -> None:
    from experiments.peft_coarse_supervision_v1.ridge import (
        _head_design,
        _head_predict_point,
        _vector_to_head,
    )

    hidden = np.zeros((2, 47, 768), dtype=np.float32)
    hidden[:, :, 0] = 1.0
    hidden[:, :, 1] = np.arange(47, dtype=np.float32)
    base_point = np.full((2, 752), 10.0, dtype=np.float32)
    native_scale = np.array([2.0, 4.0], dtype=np.float64)
    coarse_scale = np.array([5.0, 8.0], dtype=np.float64)
    horizon = np.array([17, 32], dtype=np.int64)
    vector = np.zeros(16 * 768 + 16, dtype=np.float64)
    weight, bias = _vector_to_head(vector)
    weight[:, 0] = 0.25
    weight[:, 1] = 0.01
    bias[:] = -0.5
    vector = np.concatenate([weight.reshape(-1), bias])

    design = _head_design(hidden, native_scale, coarse_scale, horizon)
    point = _head_predict_point(base_point, hidden, native_scale, weight, bias)
    observed = []
    for row, h in enumerate(horizon):
        observed.append((point[row, :h].mean() - base_point[row, :h].mean()) / coarse_scale[row])

    np.testing.assert_allclose(design @ vector, np.asarray(observed), rtol=1e-6, atol=1e-6)


def _write_npz(path: Path, **arrays: np.ndarray) -> None:
    np.savez(path, **arrays)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fixture_contract(tmp_path: Path) -> Path:
    data_dir = tmp_path / "data"
    cache_dir = tmp_path / "cache"
    data_dir.mkdir()
    cache_dir.mkdir()
    n_train = 8
    n_val = 8
    target_ids = np.asarray(["Eagle_target_00", "Eagle_target_01", "Lamb_target_00", "Lamb_target_01"], dtype="<U64")
    train_target_ids = np.resize(target_ids, n_train)
    val_target_ids = np.resize(target_ids, n_val)
    train_site = np.asarray([0, 0, 1, 1, 0, 0, 1, 1], dtype=np.int64)
    val_site = train_site.copy()
    horizon = np.full(n_train, 24, dtype=np.int64)
    val_horizon = np.full(n_val, 24, dtype=np.int64)
    scale = np.ones(n_train, dtype=np.float64)
    val_scale = np.ones(n_val, dtype=np.float64)
    context = np.ones((n_train, 512), dtype=np.float32)
    val_context = np.ones((n_val, 512), dtype=np.float32)
    profile = np.ones((n_train, 744), dtype=np.float32) * 2.0
    val_profile = np.ones((n_val, 744), dtype=np.float32) * 2.0
    total = np.full(n_train, 48.0, dtype=np.float64)
    val_total = np.full(n_val, 48.0, dtype=np.float64)
    _write_npz(
        data_dir / "train.npz",
        context=context,
        horizon=horizon,
        scale=scale,
        site=train_site,
        target_id=train_target_ids,
        month=np.asarray(["2016-02"] * n_train, dtype="<U7"),
        profile=profile,
        total=total,
        label_valid=np.ones(n_train, dtype=bool),
    )
    _write_npz(
        data_dir / "validation.npz",
        context=val_context,
        horizon=val_horizon,
        scale=val_scale,
        site=val_site,
        target_id=val_target_ids,
        month=np.asarray(["2017-01", "2017-02"] * 4, dtype="<U7"),
        profile=val_profile,
        total=val_total,
        label_valid=np.ones(n_val, dtype=bool),
    )
    cache_path = cache_dir / "cache.npz"
    hidden_train = np.zeros((n_train, 47, 768), dtype=np.float32)
    hidden_val = np.zeros((n_val, 47, 768), dtype=np.float32)
    base_train = np.ones((n_train, 752), dtype=np.float32)
    base_val = np.ones((n_val, 752), dtype=np.float32)
    _write_npz(
        cache_path,
        train_hidden=hidden_train,
        train_base_point=base_train,
        train_native_scale=np.ones(n_train, dtype=np.float64),
        train_horizon=horizon,
        train_site=train_site,
        train_target_id=train_target_ids,
        train_month=np.asarray(["2016-02"] * n_train, dtype="<U7"),
        validation_hidden=hidden_val,
        validation_base_point=base_val,
        validation_native_scale=np.ones(n_val, dtype=np.float64),
        validation_horizon=val_horizon,
        validation_site=val_site,
        validation_target_id=val_target_ids,
        validation_month=np.asarray(["2017-01", "2017-02"] * 4, dtype="<U7"),
    )
    contract = {
        "completed": True,
        "data": {
            "train": {"path": str(data_dir / "train.npz"), "sha256": _sha(data_dir / "train.npz")},
            "validation": {"path": str(data_dir / "validation.npz"), "sha256": _sha(data_dir / "validation.npz")},
        },
        "paths": {"cache": str(cache_path), "cache_result": str(cache_dir / "result.json")},
    }
    contract_path = tmp_path / "study_contract.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    (cache_dir / "result.json").write_text(
        json.dumps(
            {
                "completed": True,
                "contract_sha256": _sha(contract_path),
                "cache_sha256": _sha(cache_path),
                "data_input_hashes": {
                    "train": contract["data"]["train"]["sha256"],
                    "validation": contract["data"]["validation"]["sha256"],
                },
            }
        ),
        encoding="utf-8",
    )
    return contract_path


def test_run_writes_selected_weights_and_result_without_eval_data(tmp_path: Path) -> None:
    from experiments.peft_coarse_supervision_v1.ridge import run

    contract_path = _fixture_contract(tmp_path)
    result = run(contract_path=contract_path, output_dir=tmp_path / "ridge")

    assert result["completed"] is True
    assert result["simple_policy"] in {"PROFILE", "F0", "COARSE_LIFT", "FROZEN_HEAD"}
    with np.load(tmp_path / "ridge" / "head_weights.npz", allow_pickle=False) as head:
        assert head["weight"].shape == (16, 768)
        assert head["weight"].dtype == np.float32
        assert head["bias"].shape == (16,)
    with np.load(tmp_path / "ridge" / "coarse_lift_weights.npz", allow_pickle=False) as lift:
        assert lift["weight"].shape == (2, 7)
    saved = json.loads((tmp_path / "ridge" / "result.json").read_text(encoding="utf-8"))
    assert saved["contract_sha256"]
    assert saved["cache_sha256"]
    assert saved["cache_result_sha256"]
    assert saved["head_weights_sha256"] == result["head_weights_sha256"]
    assert saved["prepare_validate_used"] is False
    assert set(saved["validation_scores"]) == {"PROFILE", "F0", "COARSE_LIFT", "FROZEN_HEAD"}
    assert saved["validation_scores"]["FROZEN_HEAD"]["deployed_fp32_by_lambda"]


def test_run_refuses_cache_identity_mismatch(tmp_path: Path) -> None:
    from experiments.peft_coarse_supervision_v1.ridge import run

    contract_path = _fixture_contract(tmp_path)
    cache_path = tmp_path / "cache" / "cache.npz"
    with np.load(cache_path, allow_pickle=False) as cache:
        arrays = {key: cache[key] for key in cache.files}
    arrays["validation_month"] = np.asarray(["2099-01"] * len(arrays["validation_month"]), dtype="<U7")
    np.savez(cache_path, **arrays)
    cache_result = tmp_path / "cache" / "result.json"
    record = json.loads(cache_result.read_text(encoding="utf-8"))
    record["cache_sha256"] = _sha(cache_path)
    cache_result.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ValueError, match="cache identity"):
        run(contract_path=contract_path, output_dir=tmp_path / "ridge")


def test_run_refuses_stale_cache_result(tmp_path: Path) -> None:
    from experiments.peft_coarse_supervision_v1.ridge import run

    contract_path = _fixture_contract(tmp_path)
    cache_result = tmp_path / "cache" / "result.json"
    record = json.loads(cache_result.read_text(encoding="utf-8"))
    record["cache_sha256"] = "0" * 64
    cache_result.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(AssertionError, match="cache provenance"):
        run(contract_path=contract_path, output_dir=tmp_path / "ridge")


def test_monthly_score_rejects_nonfinite_prediction_for_valid_label() -> None:
    from experiments.peft_coarse_supervision_v1.ridge import _score_monthly_average

    data = {
        "label_valid": np.asarray([True, True], dtype=bool),
        "total": np.asarray([24.0, 24.0], dtype=np.float64),
        "horizon": np.asarray([24, 24], dtype=np.int64),
        "scale": np.asarray([1.0, 1.0], dtype=np.float64),
        "site": np.asarray([0, 0], dtype=np.int64),
        "target_id": np.asarray(["a", "b"], dtype="<U4"),
    }

    with pytest.raises(FloatingPointError, match="Nonfinite prediction"):
        _score_monthly_average(np.asarray([1.0, np.nan], dtype=np.float64), data)
