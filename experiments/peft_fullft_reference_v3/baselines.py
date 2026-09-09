"""Train-only probabilistic seasonal and ridge references; no neural imports."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

for _thread_key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_thread_key] = "2"

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


LAMBDAS = (0.01, 1.0, 100.0)


def fit_ridge(x, y, lam):
    """Minimize sum squared errors + lam * ||standardized coefficients||²."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or len(x) < 2:
        raise ValueError("Ridge requires matching nonempty feature and output matrices")
    if not np.isfinite(x).all() or not np.isfinite(lam) or lam <= 0 or np.isinf(y).any():
        raise ValueError("Invalid ridge data or positive penalty")
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale == 0] = 1.0
    z = (x - mean) / scale
    valid = np.isfinite(y)
    counts = valid.sum(axis=0)
    if np.any(counts < 2):
        raise ValueError("Every target/lead needs at least two observed training labels")
    groups = {}
    for column in range(y.shape[1]):
        groups.setdefault(valid[:, column].tobytes(), []).append(column)
    coefficients = np.empty((x.shape[1], y.shape[1]), dtype=np.float64)
    intercept = np.empty(y.shape[1], dtype=np.float64)
    for columns in groups.values():
        rows = valid[:, columns[0]]
        group_z = z[rows]
        group_mean = group_z.mean(axis=0)
        centered_z = group_z - group_mean
        observed_y = y[np.ix_(rows, columns)]
        target_mean = observed_y.mean(axis=0)
        gram = centered_z @ centered_z.T
        gram.flat[::len(gram) + 1] += lam
        dual = np.linalg.solve(gram, observed_y - target_mean)
        coefficients[:, columns] = centered_z.T @ dual
        intercept[columns] = target_mean - group_mean @ coefficients[:, columns]
    return {"feature_mean": mean, "feature_scale": scale, "coefficients": coefficients,
            "intercept": intercept, "observed_counts": counts, "lambda": np.array(lam),
            "mask_groups": np.array(len(groups))}


def predict_ridge(model, x):
    x = np.asarray(x, dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError("Prediction features must be finite")
    return ((x - model["feature_mean"]) / model["feature_scale"]) @ model["coefficients"] + model["intercept"]


def oof_folds(origins, horizon):
    origins = np.asarray(origins, dtype=np.int64)
    if origins.ndim != 1 or not 61 <= len(origins) <= 90 or np.any(np.diff(origins) <= 0):
        raise ValueError("OOF requires 61..90 strictly chronological train origins")
    folds = []
    for start, stop in ((30, 60), (60, len(origins))):
        forecast_rows = np.arange(start, stop)
        fit_rows = np.flatnonzero((np.arange(len(origins)) < start) &
                                  (origins + horizon <= origins[start]))
        if len(fit_rows) < 2:
            raise ValueError("Insufficient fully arrived OOF labels")
        folds.append((fit_rows, forecast_rows))
    return folds


def ridge_oof_predictions(x, y, origins, horizon, lam):
    x, y = np.asarray(x), np.asarray(y)
    predictions, indices, audit = [], [], []
    for fit_rows, forecast_rows in oof_folds(origins, horizon):
        model = fit_ridge(x[fit_rows], y[fit_rows].reshape(len(fit_rows), -1), lam)
        predictions.append(predict_ridge(model, x[forecast_rows]).reshape((len(forecast_rows),) + y.shape[1:]))
        indices.append(forecast_rows)
        audit.append({"fit_rows": fit_rows.tolist(), "forecast_rows": forecast_rows.tolist(),
                      "fit_target_end_exclusive": int(np.max(np.asarray(origins)[fit_rows] + horizon)),
                      "forecast_first_origin": int(np.asarray(origins)[forecast_rows[0]]),
                      "observed_counts": model["observed_counts"].tolist(),
                      "mask_groups": int(model["mask_groups"])})
    return np.concatenate(predictions), np.concatenate(indices), audit


def residual_quantiles(residual, quantiles):
    residual, quantiles = np.asarray(residual), np.asarray(quantiles)
    if residual.ndim != 3 or np.isinf(residual).any():
        raise ValueError("Residuals must have origin/target/lead axes and finite observed values")
    offsets, counts = [], []
    for target in range(residual.shape[1]):
        values = residual[:, target, :].ravel()
        values = values[np.isfinite(values)]
        if not len(values):
            raise ValueError("Each target needs observed calibration residuals")
        offsets.append(np.quantile(values, quantiles, method="linear"))
        counts.append(len(values))
    return np.asarray(offsets), np.asarray(counts, dtype=np.int64)


def weekly_predictions(values, origins, target_indices, horizon):
    if horizon > 168 or np.min(origins) < 168:
        raise ValueError("Weekly reference must use fully historical input")
    return np.stack([values[o - 168:o - 168 + horizon, target_indices].T for o in origins]).astype(np.float64)


def score_predictions(prediction, target, fit_std, quantiles):
    prediction = np.sort(np.asarray(prediction, dtype=np.float64), axis=2)
    target, fit_std, quantiles = map(np.asarray, (target, fit_std, quantiles))
    if prediction.shape != (len(target), target.shape[1], len(quantiles), target.shape[2]):
        raise ValueError("Prediction and target dimensions disagree")
    if not np.isfinite(prediction).all() or not np.isfinite(fit_std).all() or np.any(fit_std <= 0):
        raise ValueError("Invalid prediction or target scale")
    valid = np.isfinite(target)
    count = valid.sum(axis=(0, 2))
    if np.any(count == 0):
        raise ValueError("Every scored target needs observed labels")
    error = np.where(valid[:, :, None, :], target[:, :, None, :] - prediction, 0.)
    losses = 2 * np.maximum(quantiles[None, None, :, None] * error,
                            (quantiles[None, None, :, None] - 1) * error)
    sums = losses.sum(axis=(0, 2, 3)) / fit_std
    target_score = sums / (count * len(quantiles))
    med, lo, hi = [int(np.argmin(abs(quantiles - value))) for value in (.5, .1, .9)]
    target_mse = np.where(valid, (target - prediction[:, :, med]) ** 2, 0.).sum(axis=(0, 2)) / count
    coverage = ((target >= prediction[:, :, lo]) & (target <= prediction[:, :, hi]) & valid).sum(axis=(0, 2)) / count
    return {"score": float(target_score.mean()), "target_scores": target_score.tolist(),
            "median_mse": float(target_mse.mean()), "target_median_mse": target_mse.tolist(),
            "coverage80": float(coverage.mean()), "target_coverage80": coverage.tolist(),
            "observed_target_cells": count.tolist()}


def windows(panel, split):
    origins = panel["origins"][split]
    context, horizon = panel["context"], panel["horizon"]
    x = np.stack([panel["context_values"][o - context:o].T.reshape(-1) for o in origins])
    y = np.stack([panel["target_values"][o:o + horizon, panel["target_indices"]].T for o in origins])
    return x, y


def probabilistic_prediction(model, panel, split):
    if str(model["method"]) == "SEASONAL_WEEK":
        point = weekly_predictions(panel["context_values"], panel["origins"][split],
                                   panel["target_indices"], panel["horizon"])
    else:
        x, _ = windows(panel, split)
        point = predict_ridge(model, x).reshape(len(x), len(panel["target_indices"]), panel["horizon"])
    return point[:, :, None, :] + model["residual_quantiles"][None, :, :, None]


def fit_candidates(panel):
    x, target = windows(panel, "train")
    _, val_target = windows(panel, "val")
    origins, horizon, quantiles = panel["origins"]["train"], panel["horizon"], panel["quantiles"]
    oof_rows = np.concatenate([future for _, future in oof_folds(origins, horizon)])
    models, records, predictions = [], [], []
    for index, lam in enumerate((None,) + LAMBDAS):
        started = time.perf_counter()
        if lam is None:
            point = weekly_predictions(panel["context_values"], origins[oof_rows], panel["target_indices"], horizon)
            residual = target[oof_rows] - point
            model, fold_audit = {"method": np.array("SEASONAL_WEEK")}, []
        else:
            point, rows, fold_audit = ridge_oof_predictions(x, target, origins, horizon, lam)
            if not np.array_equal(rows, oof_rows):
                raise AssertionError("All reference candidates must use the same residual origins")
            residual = target[rows] - point
            model = fit_ridge(x, target.reshape(len(target), -1), lam)
            model["method"] = np.array("CONTEXT_RIDGE")
        offsets, counts = residual_quantiles(residual, quantiles)
        model.update(residual_quantiles=offsets, quantiles=quantiles.copy(),
                     target_indices=panel["target_indices"].copy(), fit_std=panel["fit_std"].copy(),
                     context=np.array(panel["context"]), horizon=np.array(horizon),
                     residual_counts=counts)
        prediction = probabilistic_prediction(model, panel, "val")
        metrics = score_predictions(prediction, val_target, panel["fit_std"][panel["target_indices"]], quantiles)
        records.append({"index": index, "method": str(model["method"]), "lambda": lam,
                        "validation": metrics, "residual_rows": oof_rows.tolist(),
                        "residual_cells_per_target": counts.tolist(), "oof_folds": fold_audit,
                        "wall_seconds": time.perf_counter() - started})
        models.append(model)
        predictions.append(prediction)
    return models, records, np.stack(predictions)


def load_panel(path, stage):
    names = ("train", "val") if stage == "fit" else ("cal", "eval")
    forbidden = ("cal", "eval") if stage == "fit" else ("train", "val")
    with np.load(path, allow_pickle=False) as archive:
        if any(name + "_origins" in archive for name in forbidden):
            raise ValueError("Archive mixes fitting and holdout origins")
        panel = {key: archive[key].copy() for key in (
            "context_values", "target_values", "target_indices", "quantiles", "fit_std", "timestamps", "channels")}
        panel["context"], panel["horizon"] = int(archive["context"]), int(archive["horizon"])
        panel["origins"] = {name: archive[name + "_origins"].astype(np.int64) for name in names}
        panel["metadata"] = json.loads(archive["manifest_json"].item())
    values, target = panel["context_values"], panel["target_values"]
    if values.shape != target.shape or values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Finite causal context and original target shapes disagree")
    if (panel["context"], panel["horizon"]) != (336, 48) or panel["quantiles"].shape != (21,):
        raise ValueError("Expected L336/H48 with native 21 quantiles")
    if np.any(np.diff(panel["quantiles"]) <= 0) or np.any(panel["fit_std"] <= 0):
        raise ValueError("Invalid native quantile grid or training scale")
    for split, origins in panel["origins"].items():
        if not len(origins) or np.any(np.diff(origins) <= 0) or origins[0] < panel["context"] or origins[-1] + panel["horizon"] > len(values):
            raise ValueError("Origins violate context/target chronology")
        if "boundary_indices" in panel["metadata"]:
            left, right = panel["metadata"]["boundary_indices"][split]
            if origins[0] < left or origins[-1] + panel["horizon"] > right:
                raise ValueError("Target window crosses split boundary")
    if stage == "fit" and panel["origins"]["train"][-1] + panel["horizon"] > panel["origins"]["val"][0]:
        raise ValueError("Training labels overlap validation")
    return panel


def _entries(contract):
    datasets = contract["datasets"]
    if isinstance(datasets, dict):
        return [(name, entry) for name, entry in sorted(datasets.items())]
    return [(entry["dataset"], entry) for entry in datasets]


def _path(value, root):
    value = Path(value)
    return value if value.is_absolute() else root / value


def validate_global_selection(global_path, contract_sha256, artifacts):
    from experiments.peft_fullft_reference_v3.contract import digest
    if not Path(global_path).is_file():
        raise ValueError("Global choices must be frozen before reading holdout")
    global_selection = json.loads(Path(global_path).read_text(encoding="utf-8"))
    if (global_selection.get("completed") is not True or
            global_selection.get("global_choices_frozen") is not True or
            global_selection.get("contract_sha256") != contract_sha256):
        raise ValueError("Global selection is incomplete or belongs to another contract")
    for dataset, paths in artifacts.items():
        entry = global_selection.get("baselines", {}).get(dataset, {})
        for key, path in paths.items():
            if entry.get(key) != digest(path):
                raise ValueError("Global baseline model/selection hash mismatch")


def fit_stage(contract_path, output):
    from experiments.peft_fullft_reference_v3.contract import ROOT, digest, read_contract, save_json
    contract = read_contract(contract_path, verify="core")
    contract_hash = digest(contract_path)
    summary, started = {}, time.perf_counter()
    for dataset, entry in _entries(contract):
        directory = Path(output) / dataset
        if (directory / "result.json").exists():
            raise ValueError("Refusing to overwrite completed baseline fit")
        fit_path = _path(entry["fit_data_path"], ROOT)
        input_hash = digest(fit_path)
        panel = load_panel(fit_path, "fit")
        models, records, predictions = fit_candidates(panel)
        selected = min(range(len(records)), key=lambda index: (records[index]["validation"]["score"], index))
        directory.mkdir(parents=True, exist_ok=True)
        model_path = directory / "model.npz"
        np.savez_compressed(model_path, **models[selected])
        _, target = windows(panel, "val")
        np.savez_compressed(directory / "V_predictions.npz", candidate_predictions=predictions,
                            val_predictions=predictions[selected], val_target=target,
                            val_origins=panel["origins"]["val"],
                            quantiles=panel["quantiles"], target_indices=panel["target_indices"],
                            fit_std=panel["fit_std"], val_timestamps=panel["timestamps"][panel["origins"]["val"]])
        if digest(fit_path) != input_hash:
            raise AssertionError("Fitting input changed while fitting baselines")
        selection = {"dataset": dataset, "completed": True, "contract_sha256": contract_hash,
                     "selected_index": selected, "selected_candidate": records[selected], "candidates": records,
                     "model_sha256": digest(model_path), "fit_data_sha256": input_hash,
                     "V_predictions_sha256": digest(directory / "V_predictions.npz"),
                     "selection_rule": "minimum V score; exact tie uses seasonal then ascending lambda",
                     "uses_calibration_labels": False, "uses_evaluation_labels": False,
                     "residual_protocol": "train origins 30..89 only; ridge expanding OOF with horizon purge; target-wise lead pooling",
                     "ridge_objective": "sum observed squared errors + lambda times standardized coefficient squared norm; intercept unpenalized"}
        save_json(directory / "selection.json", selection)
        result = {"dataset": dataset, "completed": True, "stage": "fit",
                  "status": "BASELINE_FIT_COMPLETE_GLOBAL_SELECTION_PENDING", "contract_sha256": contract_hash,
                  "candidate_count": 4, "ridge_oof_fits": 6, "ridge_final_fits": 3,
                  "new_neural_fits": 0, "seed": None, "train_origins": len(panel["origins"]["train"]),
                  "val_origins": len(panel["origins"]["val"]), "selected_index": selected,
                  "selection_sha256": digest(directory / "selection.json"), "model_sha256": digest(model_path),
                  "wall_seconds": sum(record["wall_seconds"] for record in records)}
        save_json(directory / "result.json", result)
        summary[dataset] = result
    save_json(Path(output) / "fit_summary.json", {"completed": True, "datasets": summary,
                                                "wall_seconds": time.perf_counter() - started})
    return summary


def forecast_stage(contract_path, output):
    from experiments.peft_fullft_reference_v3.contract import ROOT, STUDY, digest, read_contract, save_json
    contract = read_contract(contract_path, verify="core")
    contract_hash = digest(contract_path)
    entries = _entries(contract)
    artifacts = {dataset: {"model_sha256": Path(output) / dataset / "model.npz",
                           "selection_sha256": Path(output) / dataset / "selection.json"} for dataset, _ in entries}
    global_path = ROOT / "runs" / STUDY / "selection.json"
    validate_global_selection(global_path, contract_hash, artifacts)
    summary = {}
    for dataset, entry in entries:
        directory = Path(output) / dataset
        if (directory / "forecast_result.json").exists():
            raise ValueError("Refusing to overwrite completed baseline holdout predictions")
        selection = json.loads((directory / "selection.json").read_text(encoding="utf-8"))
        if (selection["contract_sha256"] != contract_hash or
                selection["model_sha256"] != digest(directory / "model.npz")):
            raise ValueError("Local model belongs to another contract or selection")
        with np.load(directory / "model.npz", allow_pickle=False) as archive:
            model = {key: archive[key].copy() for key in archive.files}
        holdout_path = _path(entry["holdout_data_path"], ROOT)
        before = digest(holdout_path)
        if before != entry["holdout_data_sha256"]:
            raise ValueError("Holdout hash differs from frozen contract")
        panel = load_panel(holdout_path, "forecast")
        for key in ("target_indices", "fit_std", "quantiles", "context", "horizon"):
            if not np.array_equal(model[key], panel[key]):
                raise AssertionError("Fit/holdout model coordinate mismatch")
        paths = {}
        for split, label in (("cal", "C"), ("eval", "E")):
            prediction = probabilistic_prediction(model, panel, split)
            _, target = windows(panel, split)
            destination = directory / f"{label}_predictions.npz"
            np.savez_compressed(destination, prediction=prediction, target=target, mask=np.isfinite(target),
                                origins=panel["origins"][split], timestamps=panel["timestamps"][panel["origins"][split]],
                                quantiles=panel["quantiles"], target_indices=panel["target_indices"],
                                fit_std=panel["fit_std"], **{split + "_predictions": prediction, split + "_target": target,
                                                          split + "_origins": panel["origins"][split]})
            paths[label] = {"path": str(destination), "sha256": digest(destination), "origins": len(prediction)}
        if digest(holdout_path) != before:
            raise AssertionError("Holdout input changed during forecasting")
        validate_global_selection(global_path, contract_hash, artifacts)
        result = {"dataset": dataset, "completed": True, "stage": "forecast", "contract_sha256": contract_hash,
                  "global_selection_sha256": digest(global_path), "holdout_data_sha256": before,
                  "uses_calibration_for_fitting": False, "predictions": paths}
        save_json(directory / "forecast_result.json", result)
        summary[dataset] = result
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--stage", choices=("fit", "forecast"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = (fit_stage if args.stage == "fit" else forecast_stage)(args.contract, args.output)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
