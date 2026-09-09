"""Train-lag RAW baseline with no known-lag dictionary and no OOF calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from .data import CHANNELS, H, make_lag_features


def episode_loss(prediction: np.ndarray, target: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
    error = target[:, None, :] - prediction
    q = quantiles[None, :, None]
    return (2 * np.maximum(q * error, (q - 1) * error)).mean(axis=(1, 2))


def load_lag_selection(path: Path) -> dict[str, Any]:
    selection = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {
        "input_array_hash",
        "selected_lags",
        "candidates",
        "candidate_lags",
        "scores",
        "fit_seconds",
        "data_sha256",
        "known_lag_dictionary_used",
        "validation_or_eval_used",
        "oracle_or_manifest_used",
    }
    if not required <= set(selection):
        raise AssertionError(f"lag selection JSON is missing keys: {sorted(required - set(selection))}")
    if selection["known_lag_dictionary_used"] or selection["validation_or_eval_used"] or selection["oracle_or_manifest_used"]:
        raise AssertionError("lag selection JSON declares an out-of-scope input")
    if set(selection["selected_lags"]) != set(CHANNELS):
        raise AssertionError("lag selection must provide exactly Y/U/V lags")
    return selection


def ols_fit(features: np.ndarray, target: np.ndarray) -> dict[str, np.ndarray | float]:
    x = np.asarray(features, dtype=np.float64).reshape(-1, len(CHANNELS))
    y = np.asarray(target, dtype=np.float64).reshape(-1)
    if x.shape[0] != y.shape[0]:
        raise ValueError("feature and target rows differ")
    design = np.column_stack((np.ones(x.shape[0], dtype=np.float64), x))
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    fitted = design @ coef
    return {
        "intercept": float(coef[0]),
        "coefficient": coef[1:].astype(np.float64),
        "train_fitted": fitted.reshape(target.shape),
        "rank": int(np.linalg.matrix_rank(design)),
    }


def ols_predict(fit: dict[str, np.ndarray | float], features: np.ndarray) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    return float(fit["intercept"]) + np.tensordot(x, np.asarray(fit["coefficient"], dtype=np.float64), axes=([-1], [0]))


def fit_raw(features: dict[str, np.ndarray], target_train: np.ndarray, quantiles: np.ndarray) -> dict[str, Any]:
    started = time.perf_counter()
    fit = ols_fit(features["train"], target_train)
    residuals = np.asarray(target_train, dtype=np.float64) - np.asarray(fit["train_fitted"], dtype=np.float64)
    offsets = np.quantile(residuals.reshape(-1), quantiles)
    mean_predictions = {
        split: ols_predict(fit, values)
        for split, values in features.items()
    }
    predictions = {
        split: mean[:, None, :] + offsets[None, :, None]
        for split, mean in mean_predictions.items()
    }
    return {
        "predictions": predictions,
        "mean_predictions": mean_predictions,
        "offsets": offsets.astype(np.float64),
        "intercept": float(fit["intercept"]),
        "coefficient": np.asarray(fit["coefficient"], dtype=np.float64),
        "rank": int(fit["rank"]),
        "train_residual_mean": float(residuals.mean()),
        "train_residual_var": float(residuals.var()),
        "fit_seconds": float(time.perf_counter() - started),
        "residual_distribution": "empirical full-train residual quantiles; no OOF; in-sample optimistic",
    }


def _load_arrays(data_path: Path, selection: dict[str, Any]) -> dict[str, Any]:
    with np.load(data_path, allow_pickle=False) as archive:
        quantiles = archive["quantiles"].astype(np.float64)
        context = {split: archive[f"context_{split}"].astype(np.float64) for split in ("train", "val", "eval")}
        target = {split: archive[f"target_{split}"].astype(np.float64) for split in ("train", "val", "eval")}
        episode_ids = {split: np.asarray(archive[f"episode_ids_{split}"]) for split in ("train", "val", "eval")}
        saved_features = {
            split: archive[f"raw_features_{split}"].astype(np.float64)
            for split in ("train", "val", "eval")
            if f"raw_features_{split}" in archive.files
        }
    features = {
        split: make_lag_features(context[split], selection["selected_lags"]).astype(np.float64)
        for split in ("train", "val", "eval")
    }
    for split, values in saved_features.items():
        if not np.array_equal(values.astype(np.float32), features[split].astype(np.float32)):
            raise AssertionError(f"saved raw_features_{split} does not match selected_lags")
    return {"quantiles": quantiles, "features": features, "target": target, "episode_ids": episode_ids}


def run(data_path: Path, lag_selection_path: Path, output: Path) -> dict[str, Any]:
    started = time.perf_counter()
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty RAW output directory: {output}")
    selection = load_lag_selection(lag_selection_path)
    data_sha256 = _sha256(Path(data_path))
    lag_selection_sha256 = _sha256(Path(lag_selection_path))
    if data_sha256 != selection["data_sha256"]:
        raise AssertionError("lag selection file does not match the data archive hash")
    arrays = _load_arrays(Path(data_path), selection)
    fit = fit_raw(arrays["features"], arrays["target"]["train"], arrays["quantiles"])
    predictions = fit["predictions"]
    losses = {
        split: episode_loss(predictions[split], arrays["target"][split], arrays["quantiles"])
        for split in ("train", "val", "eval")
    }
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "predictions.npz",
        train_predictions=predictions["train"].astype(np.float32),
        val_predictions=predictions["val"].astype(np.float32),
        eval_predictions=predictions["eval"].astype(np.float32),
        train_target=arrays["target"]["train"].astype(np.float32),
        val_target=arrays["target"]["val"].astype(np.float32),
        eval_target=arrays["target"]["eval"].astype(np.float32),
        train_episode_ids=arrays["episode_ids"]["train"],
        val_episode_ids=arrays["episode_ids"]["val"],
        eval_episode_ids=arrays["episode_ids"]["eval"],
        train_episode_losses=losses["train"].astype(np.float64),
        val_episode_losses=losses["val"].astype(np.float64),
        eval_episode_losses=losses["eval"].astype(np.float64),
        quantiles=arrays["quantiles"].astype(np.float32),
        residual_quantile_offsets=fit["offsets"].astype(np.float64),
        selected_lags=np.asarray([selection["selected_lags"][channel] for channel in CHANNELS], dtype=np.int64),
        intercept=np.asarray(fit["intercept"], dtype=np.float64),
        coefficient=fit["coefficient"].astype(np.float64),
    )
    result = {
        "completed": True,
        "method": "RAW_TRAINLAG",
        "data": str(Path(data_path)),
        "lag_selection": str(Path(lag_selection_path)),
        "data_sha256": data_sha256,
        "lag_selection_sha256": lag_selection_sha256,
        "source_sha256": _sha256(Path(__file__)),
        "selected_lags": selection["selected_lags"],
        "candidate_lags": selection["candidate_lags"],
        "lag_input_array_hash": selection["input_array_hash"],
        "val_score": float(losses["val"].mean()),
        "eval_score": float(losses["eval"].mean()),
        "train_score": float(losses["train"].mean()),
        "intercept": fit["intercept"],
        "coefficient": fit["coefficient"].tolist(),
        "rank": fit["rank"],
        "residual_quantile_offsets": fit["offsets"].tolist(),
        "train_residual_mean": fit["train_residual_mean"],
        "train_residual_var": fit["train_residual_var"],
        "fit_seconds": fit["fit_seconds"],
        "wall_seconds": float(time.perf_counter() - started),
        "fit_kind": "intercept plus selected train-lag Y/U/V features, full-train OLS",
        "residual_distribution": fit["residual_distribution"],
        "known_lag_dictionary_used": False,
        "no_ridge": True,
        "no_hpo": True,
        "no_oof": True,
        "validation_target_used_for_fit": False,
        "eval_target_used_for_fit": False,
        "oracle_or_manifest_used_for_fit": False,
        "context_y_used_as_extra_label": False,
        "training_future_labels": int(arrays["target"]["train"].size),
        "limitation": "Full-train residual empirical quantiles are in-sample optimistic relative to OOF calibration.",
    }
    (output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--lag-selection", "--lag-file", dest="lag_selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.data, args.lag_selection, args.output), allow_nan=False), flush=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
