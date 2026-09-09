from __future__ import annotations

import argparse
from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any

for _thread_env in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_thread_env, "2")

import numpy as np

LAMBDAS = (0.001, 0.1, 10.0)
SITE_NAMES = ("Eagle", "Lamb")
SIMPLE_POLICY_ORDER = ("PROFILE", "F0", "COARSE_LIFT", "FROZEN_HEAD")
HEAD_FEATURES = 16 * 768 + 16
MAX_HORIZON = 744
PATCHED_OUTPUT = 752


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(contract_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    contract_path = Path(contract_path).resolve()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _refuse_existing_outputs(output_dir)

    contract, contract_sha256, prepare_validated = _load_and_validate_contract(contract_path)
    paths = _paths_from_contract(contract)
    train = _load_npz(paths["train"])
    validation = _load_npz(paths["validation"])
    cache = _load_npz(paths["cache"])
    cache_result = _verify_input_hashes(contract, contract_sha256, paths)
    _verify_cache_identity(train, validation, cache)

    train_cache = _split_cache(cache, "train")
    validation_cache = _split_cache(cache, "validation")
    coarse = _fit_select_coarse_lift(train, validation, train_cache, validation_cache)
    head = _fit_select_head(train, validation, train_cache, validation_cache)
    baseline_scores = {
        "PROFILE": _score_monthly_average(_row_mean(validation["profile"], validation["horizon"]), validation),
        "F0": _score_monthly_average(
            _row_mean(validation_cache["base_point"], validation["horizon"]), validation
        ),
    }
    per_v_score = {
        "PROFILE": baseline_scores["PROFILE"],
        "F0": baseline_scores["F0"],
        "COARSE_LIFT": coarse["selected_score"],
        "FROZEN_HEAD": head["selected_score"],
    }
    simple_policy = _choose_policy(per_v_score)

    head_path = output_dir / "head_weights.npz"
    coarse_path = output_dir / "coarse_lift_weights.npz"
    np.savez_compressed(
        head_path,
        weight=head["weight"].astype(np.float32),
        bias=head["bias"].astype(np.float32),
        selected_lambda=np.asarray(head["selected_lambda"], dtype=np.float64),
    )
    np.savez_compressed(
        coarse_path,
        weight=coarse["weight"].astype(np.float64),
        selected_lambda=np.asarray(coarse["selected_lambda"], dtype=np.float64),
        site_names=np.asarray(SITE_NAMES, dtype="<U16"),
    )
    contract_after, contract_sha256_after, _ = _load_and_validate_contract(contract_path)
    if contract_after != contract or contract_sha256_after != contract_sha256:
        raise AssertionError("The study contract changed during ridge fitting")

    result = {
        "completed": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract_path": str(contract_path.resolve()),
        "contract_sha256": contract_sha256,
        "prepare_validate_used": prepare_validated,
        "cache_path": str(paths["cache"].resolve()),
        "cache_result_path": str(paths["cache_result"].resolve()),
        "cache_sha256": cache_result["cache_sha256"],
        "cache_result_sha256": sha256(paths["cache_result"]),
        "head_weights_sha256": sha256(head_path),
        "coarse_weights_sha256": sha256(coarse_path),
        "selected_head_lambda": head["selected_lambda"],
        "selected_coarse_lambda": coarse["selected_lambda"],
        "simple_policy": simple_policy,
        "simple_policy_score": per_v_score[simple_policy],
        "validation_scores": {
            "PROFILE": baseline_scores["PROFILE"],
            "F0": baseline_scores["F0"],
            "COARSE_LIFT": {
                "selected": coarse["selected_score"],
                "by_lambda": coarse["scores_by_lambda"],
            },
            "FROZEN_HEAD": {
                "selected": head["selected_score"],
                "by_lambda": head["scores_by_lambda"],
                "deployed_fp32_by_lambda": head["deployed_scores_by_lambda"],
                "fp64_design_identity_max_abs_by_lambda": head["design_identity_max_abs_by_lambda"],
            },
        },
        "perVscore": per_v_score,
        "lambdas": list(LAMBDAS),
        "score_aggregation": "site -> target_id -> valid months macro MSE of monthly average error divided by coarse scale",
    }
    _write_json_once(output_dir / "result.json", result)
    return result


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _load_and_validate_contract(contract_path: Path) -> tuple[dict[str, Any], str, bool]:
    contract_path = contract_path.resolve()
    root = Path(__file__).resolve().parents[2]
    canonical = (root / "runs/peft_coarse_supervision_v1/study_contract.json").resolve()
    if contract_path == canonical:
        from . import prepare

        contract = prepare.validate(root=root, contract_path=contract_path)
        return contract, sha256(contract_path), True
    return _read_json(contract_path), sha256(contract_path), False


def _write_json_once(path: Path, record: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"output already exists: {path}")
    path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _refuse_existing_outputs(output_dir: Path) -> None:
    existing = [
        name
        for name in ("head_weights.npz", "coarse_lift_weights.npz", "result.json")
        if (output_dir / name).exists()
    ]
    if existing:
        raise FileExistsError(f"output already exists: {existing}")


def _paths_from_contract(contract: Mapping[str, Any]) -> dict[str, Path]:
    data = contract.get("data", {})
    paths = contract.get("paths", {})
    try:
        return {
            "train": Path(data["train"]["path"]),
            "validation": Path(data["validation"]["path"]),
            "cache": Path(paths["cache"]),
            "cache_result": Path(paths["cache_result"]),
        }
    except KeyError as exc:
        raise ValueError("Contract must expose data.train, data.validation, paths.cache, and paths.cache_result") from exc


def _verify_input_hashes(
    contract: Mapping[str, Any],
    contract_sha256: str,
    paths: Mapping[str, Path],
) -> dict[str, Any]:
    data = contract.get("data", {})
    for key in ("train", "validation"):
        expected = data.get(key, {}).get("sha256")
        if expected is None:
            raise AssertionError(f"Contract missing prepared data hash: {key}")
        if sha256(paths[key]) != expected:
            raise AssertionError(f"Input data hash changed: {key}")
    result = _read_json(paths["cache_result"])
    actual_cache = sha256(paths["cache"])
    expected_inputs = {key: data[key]["sha256"] for key in ("train", "validation")}
    if (
        not result.get("completed")
        or result.get("contract_sha256") != contract_sha256
        or result.get("cache_sha256") != actual_cache
        or result.get("data_input_hashes") != expected_inputs
    ):
        raise AssertionError("Frozen cache provenance is incomplete or has changed")
    return result


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _split_cache(cache: Mapping[str, np.ndarray], prefix: str) -> dict[str, np.ndarray]:
    return {
        "hidden": cache[f"{prefix}_hidden"],
        "base_point": cache[f"{prefix}_base_point"],
        "native_scale": cache[f"{prefix}_native_scale"],
        "horizon": cache[f"{prefix}_horizon"],
        "site": cache[f"{prefix}_site"],
        "target_id": cache[f"{prefix}_target_id"],
        "month": cache[f"{prefix}_month"],
    }


def _verify_cache_identity(
    train: Mapping[str, np.ndarray],
    validation: Mapping[str, np.ndarray],
    cache: Mapping[str, np.ndarray],
) -> None:
    for prefix, split in (("train", train), ("validation", validation)):
        for key in ("horizon", "site", "target_id", "month"):
            cache_key = f"{prefix}_{key}"
            if cache_key not in cache:
                raise ValueError(f"cache identity missing {cache_key}")
            if not np.array_equal(cache[cache_key], split[key]):
                raise ValueError(f"cache identity mismatch: {cache_key}")


def _fit_ridge_dual(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 2 or y.shape != (x.shape[0],):
        raise ValueError("Ridge inputs must be X[n,p] and y[n]")
    if x.shape[0] == 0:
        raise ValueError("Ridge needs at least one observation")
    if not np.isfinite(x).all() or not np.isfinite(y).all() or not np.isfinite(alpha) or alpha < 0:
        raise ValueError("Ridge inputs must be finite and alpha non-negative")
    kernel = x @ x.T
    coef_dual = np.linalg.solve(kernel + float(alpha) * np.eye(x.shape[0], dtype=np.float64), y)
    return x.T @ coef_dual


def _alpha(lambda_value: float, x: np.ndarray) -> float:
    trace = float(np.sum(x * x))
    n = max(1, x.shape[0])
    if trace <= 0.0:
        return float(lambda_value)
    return float(lambda_value) * trace / n


def _fit_select_coarse_lift(
    train: Mapping[str, np.ndarray],
    validation: Mapping[str, np.ndarray],
    train_cache: Mapping[str, np.ndarray],
    validation_cache: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    x_train = _coarse_features(train, train_cache["base_point"])
    x_validation = _coarse_features(validation, validation_cache["base_point"])
    y_train = train["total"] / (train["horizon"] * train["scale"])
    valid_train = train["label_valid"].astype(bool)
    scores: dict[str, float] = {}
    weights: dict[float, np.ndarray] = {}
    predictions: dict[float, np.ndarray] = {}
    for lambda_value in LAMBDAS:
        site_weights = np.zeros((len(SITE_NAMES), x_train.shape[1]), dtype=np.float64)
        for site_index in range(len(SITE_NAMES)):
            mask = valid_train & (train["site"] == site_index)
            site_x = x_train[mask]
            site_y = y_train[mask]
            site_weights[site_index] = _fit_ridge_dual(site_x, site_y, _alpha(lambda_value, site_x))
        predicted_normalized = np.asarray(
            [x_validation[row] @ site_weights[int(validation["site"][row])] for row in range(len(x_validation))],
            dtype=np.float64,
        )
        predicted_average = predicted_normalized * validation["scale"]
        score = _score_monthly_average(predicted_average, validation)
        scores[str(lambda_value)] = score
        weights[lambda_value] = site_weights
        predictions[lambda_value] = predicted_average
    selected_lambda = _select_lambda(scores)
    return {
        "selected_lambda": selected_lambda,
        "selected_score": scores[str(selected_lambda)],
        "scores_by_lambda": scores,
        "weight": weights[selected_lambda],
        "validation_average": predictions[selected_lambda],
    }


def _coarse_features(data: Mapping[str, np.ndarray], base_point: np.ndarray) -> np.ndarray:
    rows = len(data["horizon"])
    features = np.zeros((rows, 7), dtype=np.float64)
    for row in range(rows):
        horizon = int(data["horizon"][row])
        scale = float(data["scale"][row])
        context = data["context"][row].astype(np.float64)
        point = base_point[row, :horizon].astype(np.float64)
        first = context[:72]
        last = context[-72:]
        features[row] = np.array(
            [
                1.0,
                float(context.mean() / scale),
                float(context.std() / scale),
                float(last.mean() / scale),
                float((last.mean() - first.mean()) / scale),
                float(point.mean() / scale),
                float(horizon / MAX_HORIZON),
            ],
            dtype=np.float64,
        )
    return features


def _fit_select_head(
    train: Mapping[str, np.ndarray],
    validation: Mapping[str, np.ndarray],
    train_cache: Mapping[str, np.ndarray],
    validation_cache: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    train_design = _head_design(
        train_cache["hidden"], train_cache["native_scale"], train["scale"], train["horizon"]
    )
    validation_design = _head_design(
        validation_cache["hidden"],
        validation_cache["native_scale"],
        validation["scale"],
        validation["horizon"],
    )
    train_f0_average = _row_mean(train_cache["base_point"], train["horizon"])
    validation_f0_average = _row_mean(validation_cache["base_point"], validation["horizon"])
    y_train = train["total"] / (train["horizon"] * train["scale"]) - train_f0_average / train["scale"]
    valid_train = train["label_valid"].astype(bool)
    scores: dict[str, float] = {}
    deployed_scores: dict[str, float] = {}
    identity_gaps: dict[str, float] = {}
    weights: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    for lambda_value in LAMBDAS:
        x = train_design[valid_train]
        y = y_train[valid_train]
        vector = _fit_ridge_dual(x, y, _alpha(lambda_value, x))
        weight, bias = _vector_to_head(vector)
        weight32 = weight.astype(np.float32)
        bias32 = bias.astype(np.float32)
        point = _head_predict_point_fp32(
            validation_cache["base_point"], validation_cache["hidden"], validation_cache["native_scale"], weight32, bias32
        )
        predicted_average = _row_mean(point, validation["horizon"])
        score = _score_monthly_average(predicted_average, validation)
        scores[str(lambda_value)] = score
        deployed_scores[str(lambda_value)] = score
        weights[lambda_value] = (weight32, bias32)
        fp32_vector = np.concatenate([weight32.astype(np.float64).reshape(-1), bias32.astype(np.float64)])
        design_residual = validation_design @ fp32_vector
        deployed_residual = (predicted_average - validation_f0_average) / validation["scale"]
        identity_gaps[str(lambda_value)] = float(np.max(np.abs(design_residual - deployed_residual)))
    selected_lambda = _select_lambda(scores)
    weight, bias = weights[selected_lambda]
    return {
        "selected_lambda": selected_lambda,
        "selected_score": scores[str(selected_lambda)],
        "scores_by_lambda": scores,
        "deployed_scores_by_lambda": deployed_scores,
        "design_identity_max_abs_by_lambda": identity_gaps,
        "weight": weight,
        "bias": bias,
    }


def _head_design(
    hidden: np.ndarray,
    native_scale: np.ndarray,
    coarse_scale: np.ndarray,
    horizon: np.ndarray,
) -> np.ndarray:
    hidden = np.asarray(hidden, dtype=np.float64)
    if hidden.ndim != 3 or hidden.shape[1:] != (47, 768):
        raise ValueError("hidden must have shape [N,47,768]")
    design = np.zeros((hidden.shape[0], HEAD_FEATURES), dtype=np.float64)
    for row in range(hidden.shape[0]):
        h = int(horizon[row])
        if h < 1 or h > MAX_HORIZON:
            raise ValueError("horizon must be in 1..744")
        factor = float(native_scale[row]) / float(coarse_scale[row]) / h
        for step in range(h):
            block = step // 16
            offset = step % 16
            start = offset * 768
            design[row, start : start + 768] += factor * hidden[row, block]
            design[row, 16 * 768 + offset] += factor
    return design


def _vector_to_head(vector: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vector = np.asarray(vector, dtype=np.float64)
    if vector.shape != (HEAD_FEATURES,):
        raise ValueError("Head vector must have 12304 parameters")
    weight = vector[: 16 * 768].reshape(16, 768).copy()
    bias = vector[16 * 768 :].copy()
    return weight, bias


def _head_predict_point(
    base_point: np.ndarray,
    hidden: np.ndarray,
    native_scale: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
) -> np.ndarray:
    residual = np.zeros_like(base_point, dtype=np.float64)
    for block in range(hidden.shape[1]):
        start = block * 16
        end = start + 16
        residual[:, start:end] = hidden[:, block, :].astype(np.float64) @ weight.T + bias
    return base_point.astype(np.float64) + native_scale[:, None].astype(np.float64) * residual


def _head_predict_point_fp32(
    base_point: np.ndarray,
    hidden: np.ndarray,
    native_scale: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
) -> np.ndarray:
    base32 = np.asarray(base_point, dtype=np.float32)
    hidden32 = np.asarray(hidden, dtype=np.float32)
    scale32 = np.asarray(native_scale, dtype=np.float32)
    weight32 = np.asarray(weight, dtype=np.float32)
    bias32 = np.asarray(bias, dtype=np.float32)
    residual = np.zeros_like(base32, dtype=np.float32)
    for block in range(hidden32.shape[1]):
        start = block * 16
        end = start + 16
        residual[:, start:end] = hidden32[:, block, :] @ weight32.T + bias32
    point = base32 + scale32[:, None] * residual
    if not np.isfinite(point).all():
        raise FloatingPointError("Nonfinite deployed head prediction")
    return point


def _row_mean(values: np.ndarray, horizon: np.ndarray) -> np.ndarray:
    values64 = np.asarray(values, dtype=np.float64)
    return np.asarray([float(values64[row, : int(h)].mean()) for row, h in enumerate(horizon)], dtype=np.float64)


def _score_monthly_average(predicted_average: np.ndarray, data: Mapping[str, np.ndarray]) -> float:
    valid = data["label_valid"].astype(bool)
    if not np.isfinite(predicted_average[valid]).all():
        raise FloatingPointError("Nonfinite prediction for a valid monthly label")
    truth_average = data["total"] / data["horizon"]
    if not np.isfinite(truth_average[valid]).all() or not np.isfinite(data["scale"][valid]).all():
        raise FloatingPointError("Nonfinite validation label or scale")
    errors = ((predicted_average - truth_average) / data["scale"]) ** 2
    return _site_target_month_macro(errors, data["site"], data["target_id"], valid)


def _site_target_month_macro(
    errors: np.ndarray,
    sites: np.ndarray,
    target_ids: np.ndarray,
    valid: np.ndarray,
) -> float:
    site_scores = []
    for site in sorted(set(int(x) for x in sites.tolist())):
        site_mask = sites == site
        target_scores = []
        for target_id in sorted(set(target_ids[site_mask].tolist())):
            mask = site_mask & (target_ids == target_id) & valid
            if not bool(mask.any()):
                raise ValueError(f"No valid validation label for site {site} target {target_id}")
            target_scores.append(float(np.mean(errors[mask])))
        if target_scores:
            site_scores.append(float(np.mean(target_scores)))
    if not site_scores:
        raise ValueError("No valid validation labels for macro score")
    return float(np.mean(site_scores))


def _select_lambda(scores: Mapping[str, float]) -> float:
    return min(LAMBDAS, key=lambda value: (scores[str(value)], value))


def _choose_policy(scores: Mapping[str, float]) -> str:
    return min(SIMPLE_POLICY_ORDER, key=lambda arm: (scores[arm], SIMPLE_POLICY_ORDER.index(arm)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.contract, args.output), ensure_ascii=False))


if __name__ == "__main__":
    main()
