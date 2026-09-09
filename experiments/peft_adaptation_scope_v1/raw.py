"""Direct multi-output ridge with purged, forward-only residual calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

# Set before NumPy/BLAS initialization; this CPU baseline must not compete with GPU training.
for _thread_variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_thread_variable] = "2"

import numpy as np

from .data import load_prepared


QUANTILES = np.asarray([0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                        0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99], dtype=np.float64)
LAMBDA_GRID = (1e-3, 1e-1, 10.0)


def _sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _lag_set(horizon):
    return np.unique(np.asarray(list(range(1, 17)) + [horizon, 2 * horizon, 3 * horizon], dtype=np.int64))


def _make_features(panel, origins):
    lags = _lag_set(panel.horizon)
    if lags.max() > panel.context or np.min(origins) < lags.max():
        raise ValueError("All raw features must lie inside the observed context")
    return panel.values[origins[:, None] - lags[None, :]].reshape(len(origins), -1).astype(np.float64)


def _targets(panel, origins):
    values = panel.values[origins[:, None] + np.arange(panel.horizon)[None, :]]
    return values.transpose(0, 2, 1).astype(np.float64)


def _fit_preprocessing(train_x):
    valid = np.isfinite(train_x)
    count = valid.sum(axis=0)
    fill = np.divide(np.where(valid, train_x, 0.0).sum(axis=0), count,
                     out=np.zeros(train_x.shape[1]), where=count > 0)
    filled = np.where(valid, train_x, fill)
    mean, std = filled.mean(axis=0), filled.std(axis=0)
    std = np.where(std < 1e-12, 1.0, std)
    return {"fill": fill, "mean": mean, "std": std,
            "all_missing_features": int((count == 0).sum())}


def _transform(raw, preprocessing):
    filled = np.where(np.isfinite(raw), raw, preprocessing["fill"])
    return (filled - preprocessing["mean"]) / preprocessing["std"]


def _ridge_fit(x, y, ridge):
    """One factorization per distinct label mask, shared across all matching outputs."""
    valid = np.isfinite(y)
    masks, mask_ids = np.unique(valid.T, axis=0, return_inverse=True)
    coefficients = np.empty((x.shape[1], y.shape[1]), dtype=np.float64)
    intercept = np.empty(y.shape[1], dtype=np.float64)
    for mask_id, mask in enumerate(masks):
        columns = np.flatnonzero(mask_ids == mask_id)
        if mask.sum() < 3:
            raise ValueError("A direct target/lead has fewer than three observed fit labels")
        selected_x, selected_y = x[mask], y[np.ix_(mask, columns)]
        x_mean, y_mean = selected_x.mean(axis=0), selected_y.mean(axis=0)
        centered_x, centered_y = selected_x - x_mean, selected_y - y_mean
        if centered_x.shape[0] < centered_x.shape[1]:
            gram = centered_x @ centered_x.T
            gram.flat[::len(gram) + 1] += ridge
            coef = centered_x.T @ np.linalg.solve(gram, centered_y)
        else:
            gram = centered_x.T @ centered_x
            gram.flat[::len(gram) + 1] += ridge
            coef = np.linalg.solve(gram, centered_x.T @ centered_y)
        coefficients[:, columns] = coef
        intercept[columns] = y_mean - x_mean @ coef
    return coefficients, intercept


def _rolling_folds(origins, horizon):
    if len(origins) < 20 or np.any(np.diff(origins) <= 0):
        raise ValueError("RAW requires at least 20 strictly increasing training origins")
    folds = []
    for test in np.array_split(np.arange(len(origins)), 5)[1:]:
        first_test_origin = int(origins[test[0]])
        train = np.flatnonzero(origins + horizon <= first_test_origin)
        if len(train) < 3:
            raise ValueError("Insufficient past labels after purging overlap with OOF validation")
        if origins[train].max() + horizon > first_test_origin or train.max() >= test.min():
            raise AssertionError("Forward-only OOF information boundary violated")
        folds.append((train, test))
    return folds


def _score(prediction, target, scale):
    prediction = np.asarray(prediction, dtype=np.float64)
    if not np.isfinite(prediction).all():
        raise FloatingPointError("Nonfinite RAW quantile forecast")
    error = target[:, :, None, :] - prediction
    q = QUANTILES[None, None, :, None]
    valid = np.isfinite(target)
    loss = np.where(valid[:, :, None, :], 2 * np.maximum(q * error, (q - 1) * error), 0.0)
    loss /= np.asarray(scale, dtype=np.float64)[None, :, None, None]
    sums, counts = loss.sum(axis=(2, 3)), valid.sum(axis=2) * len(QUANTILES)
    by_channel = sums.sum(axis=0) / counts.sum(axis=0)
    per_origin = np.divide(sums, counts, out=np.full(sums.shape, np.nan), where=counts > 0)
    return float(by_channel.mean()), per_origin.astype(np.float32)


def _save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def run(data_path, output_dir, additional_ridge=None):
    start = time.perf_counter()
    ridge_grid = list(LAMBDA_GRID)
    if additional_ridge is not None:
        if not np.isfinite(additional_ridge) or additional_ridge <= 0 or additional_ridge in ridge_grid:
            raise ValueError("The one additional ridge penalty must be positive, finite and new")
        ridge_grid = sorted(ridge_grid + [float(additional_ridge)])
    data_path, output_dir = Path(data_path), Path(output_dir)
    if (output_dir / "metrics.json").exists():
        raise FileExistsError(f"Refusing to replace completed RAW results: {output_dir}")
    panel = load_prepared(data_path)
    train_origins, val_origins, eval_origins = panel.train_origins, panel.val_origins, panel.eval_origins
    train_raw = _make_features(panel, train_origins)
    val_raw = _make_features(panel, val_origins)
    # Evaluation labels are deliberately read only after lambda selection below.
    train_target = _targets(panel, train_origins)
    train_y = train_target.reshape(len(train_origins), -1)
    val_target = _targets(panel, val_origins)
    folds = _rolling_folds(train_origins, panel.horizon)
    fold_matrices, fold_contracts = [], []
    for train, test in folds:
        preprocessing = _fit_preprocessing(train_raw[train])
        fold_matrices.append((train, test, _transform(train_raw[train], preprocessing),
                              _transform(train_raw[test], preprocessing)))
        fold_contracts.append({
            "train_origin_count": len(train), "oof_origin_count": len(test),
            "max_train_label_end_exclusive": int((train_origins[train] + panel.horizon).max()),
            "first_oof_origin": int(train_origins[test[0]]), "last_oof_origin": int(train_origins[test[-1]]),
            "purge_verified": True, "feature_fill_sha256": _sha(preprocessing["fill"]),
            "feature_mean_sha256": _sha(preprocessing["mean"]),
            "feature_std_sha256": _sha(preprocessing["std"]),
            "all_missing_features": preprocessing["all_missing_features"],
        })
    final_preprocessing = _fit_preprocessing(train_raw)
    train_x = _transform(train_raw, final_preprocessing)
    val_x = _transform(val_raw, final_preprocessing)
    trials, best = [], None
    for ridge in ridge_grid:
        trial_start = time.perf_counter()
        oof = np.full_like(train_y, np.nan)
        for train, test, fit_x, test_x in fold_matrices:
            coef, intercept = _ridge_fit(fit_x, train_y[train], ridge)
            oof[test] = test_x @ coef + intercept
        residual = train_y - oof
        residual_counts = np.isfinite(residual).sum(axis=0)
        if np.any(residual_counts < 3):
            raise ValueError("Insufficient forward-only OOF residuals for a target/lead")
        offsets = np.nanquantile(residual, QUANTILES, axis=0).T.reshape(
            panel.values.shape[1], panel.horizon, len(QUANTILES)).transpose(0, 2, 1)
        coef, intercept = _ridge_fit(train_x, train_y, ridge)
        val_mean = (val_x @ coef + intercept).reshape(len(val_origins), panel.values.shape[1], panel.horizon)
        val_pred = (val_mean[:, :, None, :] + offsets[None]).astype(np.float32)
        val_score, val_loss = _score(val_pred, val_target, panel.fit_std)
        trial = {"ridge": ridge, "val_score": val_score,
                 "wall_seconds": time.perf_counter() - trial_start,
                 "oof_residual_count_min_per_channel_lead": int(residual_counts.min()),
                 "oof_residual_count_max_per_channel_lead": int(residual_counts.max())}
        trials.append(trial)
        print(json.dumps({"event": "raw_validation", "panel": panel.panel, **trial}), flush=True)
        if best is None or val_score < best["val_score"]:
            best = {**trial, "coef": coef, "intercept": intercept, "offsets": offsets,
                    "val_pred": val_pred, "val_loss": val_loss}
    eval_x = _transform(_make_features(panel, eval_origins), final_preprocessing)
    eval_mean = (eval_x @ best["coef"] + best["intercept"]).reshape(
        len(eval_origins), panel.values.shape[1], panel.horizon)
    eval_pred = (eval_mean[:, :, None, :] + best["offsets"][None]).astype(np.float32)
    eval_score, eval_loss = _score(eval_pred, _targets(panel, eval_origins), panel.fit_std)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "predictions.npz", val_pred=best["val_pred"], eval_pred=eval_pred,
                        quantiles=QUANTILES, residual_offsets=best["offsets"],
                        val_origins=val_origins, eval_origins=eval_origins,
                        val_per_origin_channel_loss=best["val_loss"], eval_per_origin_channel_loss=eval_loss)
    np.savez_compressed(output_dir / "ridge_model.npz", coefficients=best["coef"], intercept=best["intercept"],
                        feature_fill=final_preprocessing["fill"], feature_mean=final_preprocessing["mean"],
                        feature_std=final_preprocessing["std"], residual_offsets=best["offsets"])
    metrics = {
        "completed": True, "method": "RAW_VARX_RIDGE", "panel": panel.panel,
        "selected_ridge": best["ridge"], "selected_step": 0, "val_score": best["val_score"],
        "eval_score": eval_score, "lambda_selection": trials,
        "origin_counts": {"train": len(train_origins), "validation": len(val_origins), "evaluation": len(eval_origins)},
        "train_origins_sha256": _sha(train_origins), "wall_seconds": time.perf_counter() - start,
        "rawknobs": {"feature_lags": _lag_set(panel.horizon).tolist(),
                     "feature_count": train_x.shape[1], "channels": panel.channels.tolist(),
                     "direct_outputs": train_y.shape[1], "lambda_grid": ridge_grid,
                     "additional_ridge": additional_ridge,
                     "origin_caps": None, "cpu_threads": 2,
                     "ridge_objective": "sum squared errors + lambda * squared standardized-feature coefficients; unpenalized intercept",
                     "preprocessing": "Per-feature imputation and scaling fitted separately on each OOF past-fit subset; final fit uses all fit origins",
                     "oof": "Five chronological blocks; first is warmup, next four use expanding past fit with H-label overlap purged",
                     "quantile_calibration": "Separate channel-and-lead empirical quantiles of forward-only OOF residuals; no lead pooling",
                     "point_prediction": "Direct ridge conditional-mean estimate; empirical quantile offsets create the distribution forecast",
                     "f0_raw_lag_residual": "Not implemented; this run is RAW_VARX_RIDGE only"},
        "oof_folds": fold_contracts,
        "audits": {"oof_future_label_exclusion": True, "oof_preprocessing_fit_only": True,
                   "all_fm_train_origins_used": True, "lambda_selected_on_validation_only": True,
                   "eval_labels_read_after_selection": True},
        "coefficient_count": int(best["coef"].size + best["intercept"].size),
        "calibration_offset_count": int(best["offsets"].size),
    }
    _save_json(output_dir / "metrics.json", metrics)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--additional-ridge", type=float,
                        help="One predeclared extra penalty for a separately preserved boundary rescue")
    args = parser.parse_args()
    metrics = run(args.data, args.output, args.additional_ridge)
    print(json.dumps({"method": metrics["method"], "panel": metrics["panel"],
                      "val_score": metrics["val_score"], "eval_score": metrics["eval_score"],
                      "selected_ridge": metrics["selected_ridge"], "wall_seconds": metrics["wall_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
