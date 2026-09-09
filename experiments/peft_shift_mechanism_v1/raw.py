"""Episode-OOF ridge and frozen-forecast residual baselines."""

import argparse
import json
from pathlib import Path
import time

import numpy as np


def episode_loss(prediction, target, quantiles):
    error = target[:, None, :] - prediction
    q = quantiles[None, :, None]
    return (2 * np.maximum(q * error, (q - 1) * error)).mean(axis=(1, 2))


def ridge_fit(x, y, alpha):
    x, y = x.reshape(-1, x.shape[-1]).astype(np.float64), y.reshape(-1).astype(np.float64)
    mean, scale = x.mean(0), x.std(0)
    scale[scale < 1e-12] = 1.0
    standardized = (x - mean) / scale
    center = y.mean()
    coef = np.linalg.solve(standardized.T @ standardized + alpha * np.eye(x.shape[-1]),
                           standardized.T @ (y - center))
    return mean, scale, center, coef


def ridge_predict(fit, x):
    mean, scale, center, coef = fit
    return ((x - mean) / scale) @ coef + center


def fit_candidate(features, target, bases, alpha, quantiles):
    n = len(target["train"])
    oof = np.full(target["train"].shape, np.nan, dtype=np.float64)
    for held in np.array_split(np.arange(n), 4):
        fit_ids = np.setdiff1d(np.arange(n), held)
        model = ridge_fit(features["train"][fit_ids],
                          target["train"][fit_ids] - bases["train"][fit_ids], alpha)
        oof[held] = bases["train"][held] + ridge_predict(model, features["train"][held])
    assert np.isfinite(oof).all()
    # One pooled residual distribution: the generator's innovation is homoscedastic.
    offsets = np.quantile((target["train"] - oof).reshape(-1), quantiles)
    model = ridge_fit(features["train"], target["train"] - bases["train"], alpha)
    predictions = {split: (bases[split] + ridge_predict(model, features[split]))[:, None, :]
                   + offsets[None, :, None] for split in ("val", "eval")}
    return predictions, offsets, model


def run(data_path, f0_path, output):
    started = time.perf_counter()
    with np.load(data_path, allow_pickle=False) as data:
        features = {s: data[f"raw_features_{s}"].astype(np.float64) for s in ("train", "val", "eval")}
        target = {s: data[f"target_{s}"].astype(np.float64) for s in features}
        quantiles = data["quantiles"].astype(np.float64)
    with np.load(f0_path, allow_pickle=False) as f0:
        med = int(np.argmin(abs(quantiles - .5)))
        frozen_medians = {s: f0[f"{s}_predictions"][:, med].astype(np.float64) for s in features}
    output.mkdir(parents=True, exist_ok=True)
    result = {}
    for method in ("RAW", "F0_RAW"):
        t0 = time.perf_counter()
        bases = frozen_medians if method == "F0_RAW" else {s: np.zeros_like(target[s]) for s in target}
        candidates = []
        for alpha in (.001, .1, 10.0):
            pred, offsets, model = fit_candidate(features, target, bases, alpha, quantiles)
            candidates.append((float(episode_loss(pred["val"], target["val"], quantiles).mean()),
                               alpha, pred, offsets, model))
        best = min(candidates, key=lambda value: value[0])
        val_score, alpha, pred, offsets, model = best
        method_dir = output / method
        method_dir.mkdir(exist_ok=True)
        np.savez_compressed(method_dir / "predictions.npz", val_predictions=pred["val"],
                            eval_predictions=pred["eval"], val_target=target["val"], eval_target=target["eval"],
                            residual_quantile_offsets=offsets, feature_mean=model[0], feature_scale=model[1],
                            intercept=model[2], coefficient=model[3])
        item = {"completed": True, "method": method, "alpha": alpha, "val_score": val_score,
                "eval_score": float(episode_loss(pred["eval"], target["eval"], quantiles).mean()),
                "candidates": [{"alpha": c[1], "val_score": c[0]} for c in candidates],
                "wall_seconds": time.perf_counter() - t0,
                "fold_unit": "episode", "fold_count": 4, "residual_distribution": "pooled OOF future-label residuals",
                "training_future_labels": int(target["train"].size), "context_used_as_extra_label": False,
                "known_noise_sigma_used": False, "dictionary": "named Y,U,V × lags32,48,64"}
        (method_dir / "result.json").write_text(json.dumps(item, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        result[method] = item
    result["wall_seconds"] = time.perf_counter() - started
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--f0", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.data, args.f0, args.output), allow_nan=False))


if __name__ == "__main__":
    main()
