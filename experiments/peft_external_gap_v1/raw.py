"""Train-only ridge RAW baseline for the external PEFT gap screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from experiments.peft_external_gap_v1 import data as data_contract


DEFAULT_LAMBDAS = (0.1, 10.0, 1000.0)


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _ridge_solve(x: np.ndarray, y: np.ndarray, alpha: float) -> tuple[float, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 1 or x.shape[0] != y.shape[0]:
        raise ValueError("Expected x [N, F] and y [N]")
    if x.shape[0] == 0:
        raise ValueError("cannot fit ridge with zero observed labels")

    x_mean = x.mean(axis=0)
    y_mean = float(y.mean())
    xc = x - x_mean
    yc = y - y_mean
    if alpha == 0:
        coef = np.linalg.lstsq(xc, yc, rcond=None)[0]
    elif x.shape[0] < x.shape[1]:
        kernel = xc @ xc.T
        kernel.flat[:: kernel.shape[0] + 1] += float(alpha)
        dual = np.linalg.solve(kernel, yc)
        coef = xc.T @ dual
    else:
        gram = xc.T @ xc
        rhs = xc.T @ yc
        gram.flat[:: gram.shape[0] + 1] += float(alpha)
        coef = np.linalg.solve(gram, rhs)
    intercept = y_mean - float(x_mean @ coef)
    return intercept, coef


def fit_direct_ridge(x: np.ndarray, y: np.ndarray, mask: np.ndarray, alpha: float) -> dict:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if x.ndim != 2 or y.ndim != 3 or mask.shape != y.shape or x.shape[0] != y.shape[0]:
        raise ValueError("Expected x [N, F], y/mask [N, target, horizon]")
    if alpha < 0:
        raise ValueError("alpha must be non-negative")

    targets, horizon = y.shape[1], y.shape[2]
    coef = np.zeros((targets, horizon, x.shape[1]), dtype=np.float64)
    intercept = np.zeros((targets, horizon), dtype=np.float64)
    n_obs = np.zeros((targets, horizon), dtype=np.int64)
    for target in range(targets):
        for step in range(horizon):
            observed = mask[:, target, step] & np.isfinite(y[:, target, step])
            n_obs[target, step] = int(observed.sum())
            intercept[target, step], coef[target, step] = _ridge_solve(x[observed], y[observed, target, step], alpha)
    return {"coef": coef, "intercept": intercept, "n_obs": n_obs, "alpha": float(alpha)}


def predict_direct_ridge(model: dict, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    coef = np.asarray(model["coef"], dtype=np.float64)
    intercept = np.asarray(model["intercept"], dtype=np.float64)
    if x.ndim != 2 or coef.ndim != 3 or intercept.shape != coef.shape[:2] or x.shape[1] != coef.shape[2]:
        raise ValueError("model/x shape mismatch")
    return np.einsum("nf,thf->nth", x, coef) + intercept[None, :, :]


def _pinball_score(predictions: np.ndarray, target: np.ndarray, mask: np.ndarray, quantiles: np.ndarray) -> float:
    predictions = np.asarray(predictions, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool) & np.isfinite(target)
    quantiles = np.asarray(quantiles, dtype=np.float64)
    if predictions.shape != (target.shape[0], target.shape[1], len(quantiles), target.shape[2]):
        raise ValueError("Expected predictions [N, target, quantile, horizon]")

    error = np.where(mask, target, 0.0)[:, :, None, :] - predictions
    q = quantiles[None, None, :, None]
    losses = 2.0 * np.maximum(q * error, (q - 1.0) * error)
    target_scores = []
    for target_index in range(target.shape[1]):
        valid = mask[:, target_index, :]
        denom = int(valid.sum()) * len(quantiles)
        if denom:
            target_losses = np.where(valid[:, None, :], losses[:, target_index, :, :], 0.0)
            target_scores.append(float(target_losses.sum() / denom))
    if not target_scores:
        raise ValueError("no observed target cells available for scoring")
    return float(np.mean(target_scores))


def _residual_offsets(model: dict, x_train: np.ndarray, y_train: np.ndarray, mask_train: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
    point = predict_direct_ridge(model, x_train)
    residual = y_train - point
    offsets = np.zeros((y_train.shape[1], len(quantiles)), dtype=np.float64)
    for target in range(y_train.shape[1]):
        observed = mask_train[:, target, :] & np.isfinite(residual[:, target, :])
        if not observed.any():
            raise ValueError(f"no train residuals for target {target}")
        offsets[target] = np.quantile(residual[:, target, :][observed], quantiles)
    return offsets


def _predict_quantiles(model: dict, x: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    point = predict_direct_ridge(model, x)
    return point[:, :, None, :] + offsets[None, :, :, None]


def fit_candidate(
    x_train: np.ndarray,
    y_train: np.ndarray,
    mask_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    mask_val: np.ndarray,
    alpha: float,
    quantiles: np.ndarray,
) -> dict:
    model = fit_direct_ridge(x_train, y_train, mask_train, alpha=alpha)
    offsets = _residual_offsets(model, x_train, y_train, mask_train, quantiles)
    val_predictions = _predict_quantiles(model, x_val, offsets)
    return {
        "alpha": float(alpha),
        "model": model,
        "residual_offsets": offsets,
        "val_predictions": val_predictions,
        "val_score": _pinball_score(val_predictions, y_val, mask_val, quantiles),
    }


def _load_npz(path: Path | str) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def _read_manifest(archive: dict) -> dict:
    value = archive.get("manifest_json")
    if value is None:
        raise ValueError("archive is missing manifest_json")
    return json.loads(str(value.item()))


def _window_targets(archive: dict, origins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    target_values = np.asarray(archive["target_values"], dtype=np.float64)
    target_loss_mask = np.asarray(archive["target_loss_mask"], dtype=bool)
    target_indices = np.asarray(archive["target_indices"], dtype=np.int64)
    horizon = int(archive["horizon"])
    y = np.full((len(origins), len(target_indices), horizon), np.nan, dtype=np.float64)
    mask = np.zeros(y.shape, dtype=bool)
    for row, origin in enumerate(np.asarray(origins, dtype=np.int64)):
        sl = slice(int(origin), int(origin) + horizon)
        y[row] = target_values[sl][:, target_indices].T
        mask[row] = target_loss_mask[sl][:, target_indices].T
    return y, mask


def _features(archive: dict, origins: np.ndarray) -> np.ndarray:
    context_values = np.asarray(archive["context_values"], dtype=np.float64)
    fit_mean = np.asarray(archive["fit_mean"], dtype=np.float64)
    fit_std = np.asarray(archive["fit_std"], dtype=np.float64)
    context = int(archive["context"])
    origins = np.asarray(origins, dtype=np.int64)
    if np.any(origins < context):
        raise ValueError("origin lacks full historical context")
    rows = []
    for origin in origins:
        window = context_values[int(origin) - context: int(origin)]
        rows.append(((window - fit_mean[None, :]) / fit_std[None, :]).reshape(-1))
    x = np.asarray(rows, dtype=np.float64)
    return x / math.sqrt(x.shape[1])


def _scaled_target(archive: dict, origins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y, mask = _window_targets(archive, origins)
    target_indices = np.asarray(archive["target_indices"], dtype=np.int64)
    mean = np.asarray(archive["fit_mean"], dtype=np.float64)[target_indices]
    std = np.asarray(archive["fit_std"], dtype=np.float64)[target_indices]
    return (y - mean[None, :, None]) / std[None, :, None], mask


def _split_arrays(archive: dict, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    origins = np.asarray(archive[f"{split}_origins"], dtype=np.int64)
    x = _features(archive, origins)
    y, mask = _scaled_target(archive, origins)
    return origins, x, y, mask


def _raw_target(archive: dict, origins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return _window_targets(archive, origins)


def _unscale_predictions(predictions: np.ndarray, archive: dict) -> np.ndarray:
    target_indices = np.asarray(archive["target_indices"], dtype=np.int64)
    mean = np.asarray(archive["fit_mean"], dtype=np.float64)[target_indices]
    std = np.asarray(archive["fit_std"], dtype=np.float64)[target_indices]
    return predictions * std[None, :, None, None] + mean[None, :, None, None]


def _assert_archive_contract(fit: dict, holdout: dict) -> tuple[dict, dict]:
    fit_manifest = _read_manifest(fit)
    holdout_manifest = _read_manifest(holdout)
    if fit_manifest.get("archive_role") != "fit_train_val_only":
        raise ValueError("fit archive role must be fit_train_val_only")
    if holdout_manifest.get("archive_role") != "holdout_cal_eval_only_after_selection":
        raise ValueError("holdout archive role must be holdout_cal_eval_only_after_selection")
    for key in ("channels", "target_indices", "fit_mean", "fit_std", "fit_median", "quantiles"):
        if not np.array_equal(fit[key], holdout[key]):
            raise ValueError(f"fit and holdout archives disagree on {key}")
    forbidden_fit_keys = {"cal_origins", "eval_origins"}
    forbidden_holdout_keys = {"train_origins", "val_origins"}
    if forbidden_fit_keys & set(fit):
        raise ValueError("fit archive exposes holdout origins")
    if forbidden_holdout_keys & set(holdout):
        raise ValueError("holdout archive exposes fit origins")
    return fit_manifest, holdout_manifest


def run(
    fit_data: Path | str,
    holdout_data: Path | str,
    output: Path | str,
    lambdas: Sequence[float] = DEFAULT_LAMBDAS,
) -> dict:
    start_time = time.perf_counter()
    fit_data = Path(fit_data)
    holdout_data = Path(holdout_data)
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to write into non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    fit = _load_npz(fit_data)
    holdout = _load_npz(holdout_data)
    fit_manifest, holdout_manifest = _assert_archive_contract(fit, holdout)
    quantiles = np.asarray(fit["quantiles"], dtype=np.float64)

    train_origins, x_train, y_train, mask_train = _split_arrays(fit, "train")
    val_origins, x_val, y_val, mask_val = _split_arrays(fit, "val")
    candidates = []
    for alpha in lambdas:
        candidates.append(fit_candidate(x_train, y_train, mask_train, x_val, y_val, mask_val, float(alpha), quantiles))
    best = sorted(candidates, key=lambda item: (item["val_score"], item["alpha"]))[0]

    cal_origins, x_cal, _y_cal_scaled, _mask_cal_scaled = _split_arrays(holdout, "cal")
    eval_origins, x_eval, _y_eval_scaled, _mask_eval_scaled = _split_arrays(holdout, "eval")
    val_target, val_mask = _raw_target(fit, val_origins)
    cal_target, cal_mask = _raw_target(holdout, cal_origins)
    eval_target, eval_mask = _raw_target(holdout, eval_origins)

    val_predictions = _unscale_predictions(best["val_predictions"], fit).astype(np.float32)
    cal_predictions = _unscale_predictions(_predict_quantiles(best["model"], x_cal, best["residual_offsets"]), holdout).astype(np.float32)
    eval_predictions = _unscale_predictions(_predict_quantiles(best["model"], x_eval, best["residual_offsets"]), holdout).astype(np.float32)

    model_path = output / "model.npz"
    np.savez_compressed(
        model_path,
        coef=np.asarray(best["model"]["coef"], dtype=np.float64),
        intercept=np.asarray(best["model"]["intercept"], dtype=np.float64),
        n_obs=np.asarray(best["model"]["n_obs"], dtype=np.int64),
        residual_offsets_scaled=np.asarray(best["residual_offsets"], dtype=np.float64),
        selected_lambda=np.array(best["alpha"], dtype=np.float64),
        quantiles=quantiles.astype(np.float64),
    )
    model_sha = sha256_file(model_path)

    prediction_path = output / "predictions.npz"
    np.savez_compressed(
        prediction_path,
        val_predictions=val_predictions,
        cal_predictions=cal_predictions,
        eval_predictions=eval_predictions,
        val_target=val_target.astype(np.float32),
        cal_target=cal_target.astype(np.float32),
        eval_target=eval_target.astype(np.float32),
        val_target_mask=val_mask.astype(bool),
        cal_target_mask=cal_mask.astype(bool),
        eval_target_mask=eval_mask.astype(bool),
        train_origins=train_origins.astype(np.int64),
        val_origins=val_origins.astype(np.int64),
        cal_origins=cal_origins.astype(np.int64),
        eval_origins=eval_origins.astype(np.int64),
        quantiles=quantiles.astype(np.float64),
        target_indices=np.asarray(fit["target_indices"], dtype=np.int64),
        selected_lambda=np.array(best["alpha"], dtype=np.float64),
        residual_offsets_scaled=np.asarray(best["residual_offsets"], dtype=np.float32),
    )
    prediction_sha = sha256_file(prediction_path)

    result = {
        "completed": True,
        "method": "RAW_FULL_PAST_RIDGE",
        "dataset": fit_manifest.get("panel"),
        "selected_lambda": float(best["alpha"]),
        "lambda_candidates": [float(value) for value in lambdas],
        "candidate_val_scores": [
            {"lambda": float(candidate["alpha"]), "val_score": float(candidate["val_score"])}
            for candidate in candidates
        ],
        "val_score": float(best["val_score"]),
        "fit_data_path": str(fit_data),
        "fit_data_sha256": sha256_file(fit_data),
        "holdout_data_path": str(holdout_data),
        "holdout_data_sha256": sha256_file(holdout_data),
        "source_sha256": sha256_file(Path(__file__)),
        "model_path": str(model_path),
        "model_sha256": model_sha,
        "predictions_path": str(prediction_path),
        "predictions_sha256": prediction_sha,
        "quantiles": [float(value) for value in quantiles],
        "target_indices": [int(value) for value in np.asarray(fit["target_indices"], dtype=np.int64)],
        "target_columns": fit_manifest.get("target_columns"),
        "ridge_objective": "observed standardized target SSE + lambda * ||coef||^2; intercept unpenalized; lambda not divided by n_obs",
        "residual_distribution": "train fitted residual empirical quantiles, pooled over horizons within each target; no OOF and no outside calibration",
        "validation_target_used_for_fit": False,
        "cal_eval_target_used_for_selection": False,
        "fit_manifest_holdout_sha256": fit_manifest.get("holdout_archive_sha256"),
        "holdout_archive_role": holdout_manifest.get("archive_role"),
        "n_train_origins": int(len(train_origins)),
        "n_val_origins": int(len(val_origins)),
        "n_cal_origins": int(len(cal_origins)),
        "n_eval_origins": int(len(eval_origins)),
        "wall_seconds": float(time.perf_counter() - start_time),
        "model_shapes": {
            "coef": list(np.asarray(best["model"]["coef"]).shape),
            "intercept": list(np.asarray(best["model"]["intercept"]).shape),
            "n_obs": np.asarray(best["model"]["n_obs"], dtype=np.int64).tolist(),
        },
    }
    _write_json(output / "result.json", result)
    return result


def _write_fixture_archives(fit_path: Path, holdout_path: Path) -> None:
    fit_path = Path(fit_path)
    holdout_path = Path(holdout_path)
    fit_path.parent.mkdir(parents=True, exist_ok=True)
    parent = fit_path.parent.resolve(strict=False)
    holdout_parent = holdout_path.parent.resolve(strict=False)
    if holdout_parent != parent:
        raise ValueError("fixture fit and holdout archives must share one resolved parent directory")
    fit_resolved = fit_path.resolve(strict=False)
    holdout_resolved = holdout_path.resolve(strict=False)
    if fit_resolved.parent != parent or holdout_resolved.parent != parent:
        raise ValueError("fixture output paths must stay directly under their resolved parent directory")
    panel = fit_path.name.removesuffix("_fit.npz")
    tmp_dir = (parent / f"_{panel}_fixture_build").resolve(strict=False)
    if tmp_dir.parent != parent:
        raise ValueError("fixture temporary directory must stay under the output parent")
    if tmp_dir.exists():
        raise FileExistsError(f"fixture temporary directory already exists: {tmp_dir}")
    tmp_dir.mkdir(parents=True)

    hours = 220 * 24
    start = data_contract.datetime(2020, 1, 1)
    timestamps = [start + data_contract.timedelta(hours=i) for i in range(hours)]
    t = np.arange(hours, dtype=np.float32)
    values = np.stack(
        [
            20.0 + np.sin(t / 24.0),
            30.0 + np.cos(t / 24.0),
            0.5 + 0.1 * np.sin(t / 168.0),
            0.4 + 0.1 * np.cos(t / 168.0),
            0.2 + 0.01 * (t % 24),
        ],
        axis=1,
    ).astype(np.float32)
    data_contract.write_panel_archives(
        panel=panel,
        timestamps=timestamps,
        raw_values=values,
        channels=["target0", "target1", "x0", "x1", "x2"],
        target_indices=[0, 1],
        output_dir=tmp_dir,
        source={"source_path": "memory://raw-fixture", "source_sha256": "0" * 64},
    )
    generated_fit = tmp_dir / f"{panel}_fit.npz"
    generated_holdout = tmp_dir / f"{panel}_holdout.npz"
    if fit_resolved.exists() or holdout_resolved.exists():
        raise FileExistsError("fixture target archive already exists")
    generated_fit.replace(fit_resolved)
    generated_holdout.replace(holdout_resolved)
    tmp_dir.rmdir()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-data", "--fit", dest="fit_data", type=Path, required=True)
    parser.add_argument("--holdout-data", "--holdout", dest="holdout_data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lambda", dest="lambdas", type=float, action="append")
    args = parser.parse_args(argv)
    result = run(args.fit_data, args.holdout_data, args.output, lambdas=tuple(args.lambdas or DEFAULT_LAMBDAS))
    print(json.dumps({
        "completed": result["completed"],
        "dataset": result["dataset"],
        "selected_lambda": result["selected_lambda"],
        "val_score": result["val_score"],
        "result_path": str(args.output / "result.json"),
    }, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
