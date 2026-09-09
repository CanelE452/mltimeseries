"""Audit the completed external-source screen and compare fixed selected procedures."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import time

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "2"

import numpy as np


STUDY = "peft_external_gap_v1"
BOOTSTRAP_REPLICATES = 4000
BOOTSTRAP_SEED = 2026090812
CONFIDENCE = .975
DELTA = .01
DATASETS = ("bike", "household")
SEEDS = (12000, 12001, 12002)
GRIDS = {"H_MLP": (1e-4, 3e-4, 1e-3), "H_FULL": (3e-5, 1e-4, 3e-4),
         "OFF_LORA": (1e-5, 3e-5, 1e-4)}
COUNTS = {"F0": 0, "H_MLP": 589301, "H_FULL": 3653280, "OFF_LORA": 1206912}
CSV_FIELDS = ["source", "role", "method", "seed", "lr", "procedure", "score",
              "relative_improvement_over_f0", "coverage80", "width80_scaled",
              "median_mse_scaled", "crossing_before_sort", "best_step", "val_score",
              "trainable", "path"]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def compare_float(actual, expected, label, rtol=2e-6, atol=1e-8):
    if not np.isfinite(actual) or not np.isfinite(expected) or not np.isclose(actual, expected, rtol=rtol, atol=atol):
        raise AssertionError(f"{label}: {actual!r} != {expected!r}")


def forecast_arrays(prediction, target, scale, quantiles):
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    scale = np.asarray(scale, dtype=np.float64)
    quantiles = np.asarray(quantiles, dtype=np.float64)
    if (target.ndim != 3 or prediction.shape != (*target.shape[:2], len(quantiles), target.shape[-1])
            or scale.shape != (target.shape[1],)):
        raise ValueError("Expected predictions [origins, targets, quantiles, horizon] and target [origins, targets, horizon]")
    if not np.isfinite(prediction).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("Forecasts and positive train scales must be finite")
    if not np.isfinite(quantiles).all() or np.any(np.diff(quantiles) <= 0) or np.any((quantiles <= 0) | (quantiles >= 1)):
        raise ValueError("Quantiles must strictly increase inside (0, 1)")
    if np.isinf(target).any():
        raise ValueError("Only NaN represents missing targets")
    return prediction, target, scale, quantiles


def score_prediction(prediction, target, scale, quantiles):
    prediction, target, scale, quantiles = forecast_arrays(prediction, target, scale, quantiles)
    valid = np.isfinite(target)
    error = np.where(valid, target, 0)[:, :, None, :] - prediction
    q = quantiles[None, None, :, None]
    losses = 2 * np.maximum(q * error, (q - 1) * error) / scale[None, :, None, None]
    sums = np.where(valid[:, :, None, :], losses, 0).sum(axis=(2, 3))
    counts = valid.sum(axis=-1) * len(quantiles)
    return float(macro_score(sums, counts)), sums, counts


def macro_score(sums, counts):
    sums, counts = np.asarray(sums, dtype=np.float64), np.asarray(counts, dtype=np.float64)
    if sums.shape != counts.shape or sums.ndim < 2 or np.any(counts < 0):
        raise ValueError("Sufficient statistics must match and end in [origin, target]")
    totals, denominators = sums.sum(axis=-2), counts.sum(axis=-2)
    if np.any(denominators <= 0) or not np.isfinite(totals).all() or not np.isfinite(denominators).all():
        raise ValueError("Every fixed target must have observed cells")
    return np.mean(totals / denominators, axis=-1)


def qcal_offsets(prediction, target, quantiles):
    prediction, target, _, quantiles = forecast_arrays(prediction, target, np.ones(target.shape[1]), quantiles)
    offsets = np.empty((target.shape[1], len(quantiles)), dtype=np.float64)
    for channel in range(target.shape[1]):
        valid = np.isfinite(target[:, channel])
        if not valid.any():
            raise ValueError("Calibration must observe every fixed target")
        for index, q in enumerate(quantiles):
            residuals = (target[:, channel] - prediction[:, channel, index])[valid]
            offsets[channel, index] = np.quantile(residuals, q, method="linear")
    return offsets


def apply_qcal(prediction, offsets):
    prediction, offsets = np.asarray(prediction, dtype=np.float64), np.asarray(offsets, dtype=np.float64)
    if prediction.ndim != 4 or offsets.shape != prediction.shape[1:3] or not np.isfinite(offsets).all():
        raise ValueError("QCAL offsets must match [target, quantile]")
    return np.sort(prediction + offsets[None, :, :, None], axis=2)


def score_details(prediction, target, scale, quantiles, crossing_prediction=None):
    prediction, target, scale, quantiles = forecast_arrays(prediction, target, scale, quantiles)
    score, sums, counts = score_prediction(prediction, target, scale, quantiles)
    valid = np.isfinite(target)
    denominator = valid.sum(axis=(0, 2))
    indices = []
    for q in (.1, .5, .9):
        hits = np.flatnonzero(np.isclose(quantiles, q, atol=1e-7, rtol=0))
        if len(hits) != 1:
            raise ValueError("Quantile grid must contain q10/q50/q90")
        indices.append(int(hits[0]))
    lo, med, hi = indices
    covered = (target >= prediction[:, :, lo]) & (target <= prediction[:, :, hi])
    widths = (prediction[:, :, hi] - prediction[:, :, lo]) / scale[None, :, None]
    mse = ((prediction[:, :, med] - np.where(valid, target, 0)) / scale[None, :, None]) ** 2
    by_target = lambda values: (np.where(valid, values, 0).sum(axis=(0, 2)) / denominator)
    coverage, width, median_mse = by_target(covered), by_target(widths), by_target(mse)
    cross = prediction if crossing_prediction is None else np.asarray(crossing_prediction)
    if cross.shape != prediction.shape or not np.isfinite(cross).all():
        raise ValueError("Crossing diagnostic must share the same forecast shape")
    crossing = (np.diff(cross, axis=2) < 0).mean(axis=(0, 2, 3))
    per_target = sums.sum(axis=0) / counts.sum(axis=0)
    err = np.where(valid, target, 0)[:, :, None, :] - prediction
    q = quantiles[None, None, :, None]
    losses = 2 * np.maximum(q * err, (q - 1) * err) / scale[None, :, None, None]
    horizon_sums = np.where(valid[:, :, None, :], losses, 0).sum(axis=(0, 2)).T
    horizon_counts = (valid.sum(axis=0) * len(quantiles)).T
    return {"score": score, "target_scores": per_target.tolist(), "valid_cells": denominator.tolist(),
            "coverage80": float(coverage.mean()), "coverage80_by_target": coverage.tolist(),
            "width80_scaled": float(width.mean()), "width80_scaled_by_target": width.tolist(),
            "median_mse_scaled": float(median_mse.mean()), "median_mse_scaled_by_target": median_mse.tolist(),
            "crossing_before_sort": float(crossing.mean()), "crossing_by_target": crossing.tolist(),
            "horizon_target_loss_sums": horizon_sums.tolist(),
            "horizon_target_loss_counts": horizon_counts.tolist()}


def moving_block_weights(n, block, replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED):
    if not 1 <= block <= n or replicates < 1:
        raise ValueError("Invalid moving-block specification")
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n - block + 1, size=(replicates, int(np.ceil(n / block))))
    indices = (starts[..., None] + np.arange(block)).reshape(replicates, -1)[:, :n]
    weights = np.zeros((replicates, n), dtype=np.float64)
    np.add.at(weights, (np.arange(replicates)[:, None], indices), 1)
    return weights


def paired_effect(left_sums, right_sums, f0_sums, counts, weights, confidence=CONFIDENCE):
    left, right, f0, counts, weights = [np.asarray(x, dtype=np.float64)
                                       for x in (left_sums, right_sums, f0_sums, counts, weights)]
    if (left.shape != right.shape or left.ndim != 3 or left.shape[0] != 3
            or f0.shape != left.shape[1:] or counts.shape != f0.shape or weights.shape[1:] != (f0.shape[0],)):
        raise ValueError("Pair three fixed optimizer seeds across the same [daily origin, target] cells")
    if not all(np.isfinite(value).all() for value in (left, right, f0, counts, weights)):
        raise ValueError("Effect sufficient statistics must be finite")
    full_counts = np.broadcast_to(counts, left.shape)
    denom = float(macro_score(f0, counts))
    if denom <= 0:
        raise ValueError("F0 score must be positive")
    seed_values = (macro_score(left, full_counts) - macro_score(right, full_counts)) / denom
    sampled_counts = weights @ counts
    sampled_left = np.einsum("bn,snt->sbt", weights, left)
    sampled_right = np.einsum("bn,snt->sbt", weights, right)
    sampled_f0 = weights @ f0
    valid = np.all(sampled_counts > 0, axis=1)
    sampled_denom = np.mean(np.divide(sampled_f0, sampled_counts,
                                     out=np.full_like(sampled_f0, np.nan), where=sampled_counts > 0), axis=-1)
    valid &= np.isfinite(sampled_denom) & (sampled_denom > 0)
    if not valid.any():
        raise ValueError("No bootstrap draws represent every fixed target")
    delta = np.mean((sampled_left[:, valid] - sampled_right[:, valid]) / sampled_counts[None, valid], axis=(0, 2))
    draws = delta / sampled_denom[valid]
    tail = (1 - confidence) / 2
    ci = np.quantile(draws, [tail, 1-tail]).tolist()
    if ci[0] > DELTA and np.all(seed_values > 0):
        decision = "conditional_practical_improvement"
    elif ci[0] >= -DELTA and ci[1] <= DELTA:
        decision = "conditional_practical_equivalence"
    elif ci[1] < -DELTA and np.all(seed_values < 0):
        decision = "conditional_practical_harm"
    else:
        decision = "inconclusive"
    return {"value": float(seed_values.mean()), "seed_values": seed_values.tolist(), "ci": ci,
            "confidence": confidence, "decision": decision, "practical_threshold": DELTA,
            "valid_resamples": int(valid.sum()), "discarded_resamples": int((~valid).sum()),
            "requested_resamples": len(weights), "positive_means": "right procedure improves over left",
            "scope": "date blocks conditional on three fitted optimizer seeds and frozen development selections"}


def source_effects(statistics):
    effects = {}
    for procedure in ("SORT", "QCAL"):
        current = statistics[procedure]
        f0, counts = current["F0"][0]
        h = np.stack([value[0] for value in current["H"]])
        lora = np.stack([value[0] for value in current["OFF_LORA"]])
        raw = np.broadcast_to(current["RAW"][0][0], h.shape)
        for role in current.values():
            if any(not np.array_equal(value[1], counts) for value in role):
                raise AssertionError("Every procedure must score the same observed cells")
        n = len(counts)
        effects[procedure] = {}
        for block in (3, 7, 14):
            weights = moving_block_weights(n, block)
            effect = paired_effect(h, lora, f0, counts, weights,
                                   CONFIDENCE if procedure == "QCAL" else .95)
            effect.update({"block_days": block, "definition": f"(S_H_{procedure} - S_LORA_{procedure}) / S_F0_{procedure}",
                           "inferential_role": "primary" if procedure == "QCAL" and block == 7 else "descriptive sensitivity"})
            effects[procedure][str(block)] = effect
        if procedure == "QCAL":
            effects["raw_over_lora"] = paired_effect(lora, raw, f0, counts, moving_block_weights(n, 7), .95)
            effects["raw_over_lora"].update({"definition": "(mean_seed S_LORA_QCAL - S_RAW_QCAL) / S_F0_QCAL",
                                              "inferential_role": "descriptive, outside the primary family"})
            raw_score = float(macro_score(current["RAW"][0][0], counts))
            lora_score = float(macro_score(lora, np.broadcast_to(counts, lora.shape)).mean())
            effects["raw_veto"] = {"applies": raw_score < lora_score, "raw_score": raw_score,
                                    "lora_mean_score": lora_score,
                                    "criterion": "Conservative point-estimate screen: S_RAW_QCAL < mean_seed S_LORA_QCAL; no significance claim"}
    return {"primary": effects["QCAL"]["7"], "block_sensitivity": effects["QCAL"],
            "sort_effect": effects["SORT"]["7"], "sort_block_sensitivity": effects["SORT"],
            "raw_over_lora": effects["raw_over_lora"], "raw_veto": effects["raw_veto"]}


def verify_file_hashes(root, hashes, checked):
    if not hashes:
        raise AssertionError("Expected a nonempty file provenance map")
    for name, expected in hashes.items():
        path = (Path(root) / name).resolve()
        key = (str(path), expected)
        if key not in checked:
            if not path.is_file() or sha256_file(path) != expected:
                raise AssertionError(f"Protected file hash mismatch: {path}")
            checked.add(key)


def validate_guard(path, guard_directory=False):
    path = Path(path)
    directory = path if guard_directory else path / "guard"
    status = read_json(directory / "status.json")
    if not status.get("completed") or status.get("returncode") != 0 or status.get("reasons"):
        raise AssertionError(f"Incomplete or failed guard: {path}")
    if (path / "failure.json").exists() or (directory / "safety_stop.json").exists():
        raise AssertionError(f"Failure artifacts cannot accompany a completed run: {path}")
    if not resource_samples([path], guard_directory):
        raise AssertionError(f"Guard has no actual resource measurements: {path}")
    return status


def resource_samples(paths, guard_directory=False):
    required = {"timestamp", "available_ram_gib", "available_commit_gib", "child_tree_rss_gib", "git_process_count"}
    samples = []
    for path in paths:
        directory = Path(path) if guard_directory else Path(path) / "guard"
        filename = directory / "resource_log.jsonl"
        if not filename.is_file():
            continue
        for line in filename.read_text(encoding="utf-8").splitlines():
            if line.strip():
                sample = json.loads(line)
                if required <= set(sample):
                    samples.append(sample)
    return samples


def summarize_resources(paths, scope, guard_directory=False):
    samples = resource_samples(paths, guard_directory)
    if not samples:
        raise AssertionError("Cannot report resource safety without measurement samples")
    gpus = [gpu for sample in samples for gpu in (sample.get("gpus") or [])]
    return {"scope": scope, "sample_count": len(samples), "job_count": len(paths),
            "first_sample_utc": min(sample["timestamp"] for sample in samples),
            "last_sample_utc": max(sample["timestamp"] for sample in samples),
            "min_available_ram_gib": min(sample["available_ram_gib"] for sample in samples),
            "min_available_commit_gib": min(sample["available_commit_gib"] for sample in samples),
            "max_child_tree_rss_gib": max(sample["child_tree_rss_gib"] for sample in samples),
            "max_git_process_count": max(sample["git_process_count"] for sample in samples),
            "max_gpu_memory_used_mib": max((gpu["memory_used_mib"] for gpu in gpus), default=None),
            "max_gpu_temperature_c": max((gpu["temperature_c"] for gpu in gpus), default=None),
            "guard_failures": 0, "safety_stops": 0}


def trial_key(entry):
    return entry["dataset"], entry["method"], int(entry["seed"]), float(entry["lr"])


def verify_entries(trials, selection, development_choices, forecasts, root):
    root = Path(root)
    lookup = {trial_key(entry): entry for entry in trials}
    if len(trials) != 28 or len(lookup) != 28:
        raise AssertionError("Expected exactly 28 unique fit trial entries")
    if (selection.get("completed") is not True or selection.get("global_choices_frozen") is not True
            or selection.get("fit_trial_count") != 28 or selection.get("choices") != development_choices):
        raise AssertionError("Selection must freeze every development choice after all 28 fits")
    if set(development_choices) != set(DATASETS):
        raise AssertionError("Both fixed source choices must be present")
    expected = set()
    expected_selected = {}
    for dataset in DATASETS:
        expected.add((dataset, "F0", 12000, 0.))
        base = lookup.get((dataset, "F0", 12000, 0.))
        if base is None:
            raise AssertionError("Each source must have one F0")
        expected_selected[(dataset, "F0", 12000)] = {**base, "role": "F0"}
        for method, rates in GRIDS.items():
            expected.update((dataset, method, 12000, lr) for lr in rates)
        choices = development_choices[dataset]
        if set(choices) != {"H", "OFF_LORA"}:
            raise AssertionError("H and LoRA must each have a validation-only choice")
        for role, methods in (("H", ("H_MLP", "H_FULL")), ("OFF_LORA", ("OFF_LORA",))):
            keys = [(dataset, method, 12000, lr) for method in methods for lr in GRIDS[method]]
            if any(key not in lookup for key in keys):
                raise AssertionError("A development LR candidate is missing")
            candidates = [lookup[key] for key in keys]
            chosen = min(candidates, key=lambda entry: (entry["val_score"], entry["method"], entry["lr"]))
            if choices[role] != chosen:
                raise AssertionError("Development choice is not the exact sorted-score validation argmin")
            for seed in SEEDS:
                key = (dataset, chosen["method"], seed, chosen["lr"])
                expected.add(key)
                if key not in lookup:
                    raise AssertionError("A selected repeat is missing or changed family/LR")
                expected_selected[(dataset, role, seed)] = {**lookup[key], "role": role}
    if set(lookup) != expected:
        raise AssertionError("Unexpected source/method/LR/seed key in full fit set")
    fields = {"dataset", "method", "seed", "lr", "path", "val_score", "guard_seconds", "result_sha256", "checkpoint_sha256"}
    for entry in trials:
        if set(entry) != fields or not np.isfinite(entry["val_score"]) or entry["guard_seconds"] <= 0:
            raise AssertionError("Invalid full trial metadata")
        canonical = root / "runs" / STUDY / "trials" / entry["dataset"] / entry["method"] / f"lr_{entry['lr']:.0e}_seed_{entry['seed']}"
        if (root / entry["path"]).resolve() != canonical.resolve():
            raise AssertionError("Fit path is not canonical")
    selected = selection.get("selected", [])
    selected_lookup = {(entry["dataset"], entry["role"], entry["seed"]): entry for entry in selected}
    if len(selected) != 14 or selected_lookup != expected_selected:
        raise AssertionError("Expected exactly 14 selected FM entries identical to their trial metadata")
    forecast_lookup = {(entry["dataset"], entry["role"], entry["seed"]): entry for entry in forecasts}
    if len(forecasts) != 14 or set(forecast_lookup) != set(expected_selected):
        raise AssertionError("Expected exactly 14 selected held-out forecast jobs")
    for key, entry in forecast_lookup.items():
        selected_entry = expected_selected[key]
        if set(entry) != set(selected_entry) | {"fit_path", "fit_result_sha256", "forecast_guard_seconds"}:
            raise AssertionError("Forecast metadata schema mismatch")
        for name, value in selected_entry.items():
            if name not in ("path", "result_sha256") and entry[name] != value:
                raise AssertionError(f"Forecast changed selected fit metadata: {name}")
        if entry["fit_path"] != selected_entry["path"] or entry["fit_result_sha256"] != selected_entry["result_sha256"]:
            raise AssertionError("Forecast does not reference its selected checkpoint's own fit")
        canonical = root / "runs" / STUDY / "forecasts" / key[0] / f"{key[1]}_seed_{key[2]}"
        if (root / entry["path"]).resolve() != canonical.resolve() or entry["forecast_guard_seconds"] <= 0:
            raise AssertionError("Noncanonical forecast path or invalid forecast cost")
    return lookup, selected_lookup, forecast_lookup


def array_hash(*arrays):
    digest = hashlib.sha256()
    for value in arrays:
        array = np.ascontiguousarray(value)
        digest.update(str((array.shape, array.dtype.str)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def load_prediction_archive(path, panel, splits, quantiles=None, with_unsorted=False):
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    q = arrays["quantiles"]
    if q.shape != (21,) or (quantiles is not None and not np.array_equal(q, quantiles)):
        raise AssertionError("Native 21-quantile grid changed")
    if not np.array_equal(arrays["target_indices"], panel.target_indices):
        raise AssertionError("Prediction target-index contract changed")
    if not np.array_equal(arrays["target_channels"], np.asarray(panel.channels)[panel.target_indices]):
        raise AssertionError("Prediction target names/order changed")
    for split in splits:
        origins = panel.origins[split]
        target = panel.targets(split)
        if (not np.array_equal(arrays[split + "_origins"], origins)
                or not np.array_equal(arrays[split + "_timestamps"], panel.timestamps[origins])):
            raise AssertionError(f"Prediction origin/timestamp mismatch: {split}")
        if not np.array_equal(arrays[split + "_target"], target, equal_nan=True):
            raise AssertionError(f"Original observed targets or missing mask changed: {split}")
        prediction = arrays[split + "_predictions"]
        forecast_arrays(prediction, target, panel.fit_std[panel.target_indices], q)
        if not np.array_equal(prediction, np.sort(prediction, axis=2)):
            raise AssertionError("Selection/evaluation forecasts must use the common SORT procedure")
        if with_unsorted:
            unsorted = arrays[split + "_unsorted_predictions"]
            if not np.array_equal(prediction, np.sort(unsorted, axis=2)):
                raise AssertionError("Saved native quantiles do not reproduce the sorted forecast")
    return arrays


def verify_fit(root, entry, panel, frozen, checked):
    path = Path(root) / entry["path"]
    guard = validate_guard(path)
    compare_float(entry["guard_seconds"], guard["elapsed_seconds"], "Fit guard elapsed seconds", rtol=0, atol=1e-8)
    if sha256_file(path / "result.json") != entry["result_sha256"]:
        raise AssertionError("Fit result does not match frozen trial metadata")
    meta = read_json(path / "result.json")
    if (meta.get("completed") is not True or meta.get("smoke") is not False or meta.get("stage") != "fit"
            or meta.get("holdout_file_opened") is not False):
        raise AssertionError("Invalid fit completion or fit/holdout boundary")
    for name in ("dataset", "method", "seed", "lr"):
        if meta[name] != entry[name]:
            raise AssertionError(f"Fit identity mismatch: {name}")
    for name, value in {"seed_index": entry["seed"] - 12000, "context": 336, "horizon": 48,
                        "micro_groups": 4, "effective_groups": 8, "autocast": "bfloat16",
                        "autocast_weight_cache": False, "weight_dtype": "float32", "tf32": False,
                        "torch_threads": 2, "interop_threads": 1, "weight_decay": 0., "dropout": 0.,
                        "gradient_clip_norm": 1., "trainable": COUNTS[entry["method"]],
                        "trainable_parameters": COUNTS[entry["method"]]}.items():
        if meta[name] != value:
            raise AssertionError(f"Fit numerical/training contract mismatch: {name}")
    if meta["channels"] != panel.channels or meta["target_indices"] != panel.target_indices.tolist():
        raise AssertionError("Fit input/target channel map mismatch")
    for name in ("fit_mean", "fit_std", "fit_median"):
        if not np.array_equal(meta[name], getattr(panel, name)):
            raise AssertionError(f"Fit train-only statistic mismatch: {name}")
    if meta["stats_sha256"] != panel.stats_hash or meta["fit_data_sha256"] != sha256_file(panel.path):
        raise AssertionError("Fit data/statistic provenance mismatch")
    if meta["data_sha256"] != meta["fit_data_sha256"] or meta["plan_sha256"] != frozen["plan_sha256"]:
        raise AssertionError("Fit data/plan hash mismatch")
    for source, expected in meta["source_hashes"].items():
        if frozen["sources"].get(source) != expected:
            raise AssertionError("Fit source is not from the frozen study contract")
    for hashes in (meta["source_hashes"], meta["native_source_hashes"], meta["protected_hashes"], meta["cache_array_hashes"]):
        verify_file_hashes(root, hashes, checked)
    verify_file_hashes(Path(meta["checkpoint"]), meta["native_checkpoint_hashes"], checked)
    if (meta["checkpoint_sha256"] != entry["checkpoint_sha256"]
            or sha256_file(path / "best_trainable.pt") != meta["checkpoint_sha256"]
            or sha256_file(path / "predictions.npz") != meta["predictions_sha256"]):
        raise AssertionError("Fit's own saved adaptive checkpoint or prediction hash changed")
    stages = [0] if entry["method"] == "F0" else list(range(0, 201, 40))
    history = meta["validation_history"]
    if [item["step"] for item in history] != stages or meta["steps_completed"] != stages[-1]:
        raise AssertionError("Bounded optimizer/checkpoint evaluation schedule changed")
    if any(not np.isfinite(item["val_score"]) for item in history):
        raise AssertionError("Nonfinite checkpoint-selection score")
    best = min(history, key=lambda item: item["val_score"])
    if meta["best_step"] != best["step"]:
        raise AssertionError("Saved checkpoint is not the first validation argmin")
    compare_float(meta["val_score"], best["val_score"], "Checkpoint selection score")
    compare_float(meta["val_score"], entry["val_score"], "Trial selection score")
    audit = meta["audits"]
    for name in ("zero_update_identity", "padded_group_identity_checked", "trainable_map_verified",
                 "frozen_parameters_verified", "checkpoint_reload_verified", "checkpoint_includes_all_trainable_parameters"):
        if audit.get(name) is not True:
            raise AssertionError(f"Fit audit did not pass: {name}")
    if audit["zero_update_normalized_max_abs"] > 1e-5:
        raise AssertionError("Cache/direct initial identity failed")
    if audit["frozen_before_sha256"] != audit["frozen_after_training_sha256"]:
        raise AssertionError("Frozen weights changed")
    if entry["method"] != "F0":
        if (audit.get("finite_nonzero_gradient_verified") is not True
                or audit["initial_adaptation_sha256"] == audit["before_restore_adaptation_sha256"]
                or audit["frozen_before_sha256"] != audit["frozen_after_first_update_sha256"]):
            raise AssertionError("Optimizer did not update its declared trainable parameters only")
    if entry["method"] == "OFF_LORA":
        expected = [f"encoder.block.{b}.layer.{l}.self_attention.{part}" for b in range(12)
                    for l in (0, 1) for part in ("q", "k", "v", "o")]
        expected.append("output_patch_embedding.output_layer")
        if meta["module_map"] != expected:
            raise AssertionError("Official LoRA native 97-module map changed")
    steps = 0 if entry["method"] == "F0" else 200
    samples = np.random.default_rng(entry["seed"]).integers(len(panel.origins["train"]), size=(steps, 8))
    if meta["sampler_sha256"] != array_hash(panel.origins["train"][samples]):
        raise AssertionError("Shared seeded train-origin sampler changed")
    for split in ("train", "val"):
        if meta["origin_hashes"][split] != array_hash(panel.origins[split]) or meta["origin_counts"][split] != len(panel.origins[split]):
            raise AssertionError("Fit split origins changed")
    cache = Path(meta["cache"])
    manifest = read_json(cache / "manifest.json")
    cache_contract = manifest["contract"]
    signature = hashlib.sha256(json.dumps(cache_contract, sort_keys=True).encode()).hexdigest()
    if not manifest.get("completed") or signature != meta["cache_contract_sha256"]:
        raise AssertionError("Frozen feature cache manifest mismatch")
    for name, expected in {"data_sha256": meta["fit_data_sha256"], "source_hashes": meta["source_hashes"],
                            "native_source_hashes": meta["native_source_hashes"], "plan_sha256": meta["plan_sha256"],
                            "checkpoint_hashes": meta["native_checkpoint_hashes"], "dataset": entry["dataset"],
                            "context": 336, "horizon": 48, "micro_groups": 4, "smoke": False,
                            "stats_sha256": panel.stats_hash, "origins": {k: v.tolist() for k, v in panel.origins.items()}}.items():
        if cache_contract[name] != expected:
            raise AssertionError(f"Cache contract mismatch: {name}")
    verify_file_hashes(cache, manifest["array_hashes"], checked)
    origins = np.unique(np.concatenate(list(panel.origins.values())))
    if not np.array_equal(np.load(cache / "origins.npy", allow_pickle=False), origins):
        raise AssertionError("Frozen feature cache origins differ from fit/validation")
    shapes = {"hidden": (len(origins), len(panel.channels), 3, 768),
              "base_norm": (len(origins), len(panel.channels), 21, 48),
              "loc": (len(origins), len(panel.channels), 1), "scale": (len(origins), len(panel.channels), 1)}
    for name, expected in shapes.items():
        value = np.load(cache / (name + ".npy"), mmap_mode="r", allow_pickle=False)
        if value.shape != expected:
            raise AssertionError("Cache must retain every input channel and all three future patches")
        del value
    arrays = load_prediction_archive(path / "predictions.npz", panel, ("val",))
    score, sums, counts = score_prediction(arrays["val_predictions"], arrays["val_target"], panel.fit_std[panel.target_indices], arrays["quantiles"])
    compare_float(score, meta["val_score"], "Recomputed selected validation loss")
    if (not np.allclose(sums, arrays["val_loss_sums"], rtol=1e-10, atol=1e-8)
            or not np.array_equal(counts, arrays["val_valid_counts"])):
        raise AssertionError("Saved validation loss sums/observed counts do not match predictions")
    return meta


def verify_forecast(root, entry, fit, panel, frozen, selection_sha, contract_sha, checked):
    path = Path(root) / entry["path"]
    guard = validate_guard(path)
    compare_float(entry["forecast_guard_seconds"], guard["elapsed_seconds"], "Forecast guard cost", rtol=0, atol=1e-8)
    if sha256_file(path / "result.json") != entry["result_sha256"]:
        raise AssertionError("Forecast result hash mismatch")
    meta = read_json(path / "result.json")
    if meta.get("completed") is not True or meta.get("stage") != "forecast":
        raise AssertionError("Incomplete held-out forecast")
    for name in ("dataset", "method", "role", "seed", "lr"):
        if meta[name] != entry[name]:
            raise AssertionError(f"Forecast identity mismatch: {name}")
    if (Path(meta["fit_trial"]).resolve() != (Path(root) / entry["fit_path"]).resolve()
            or meta["fit_result_sha256"] != entry["fit_result_sha256"]
            or meta["fit_checkpoint_sha256"] != fit["checkpoint_sha256"]
            or meta["fit_data_sha256"] != fit["fit_data_sha256"]
            or meta["holdout_data_sha256"] != sha256_file(panel.path)):
        raise AssertionError("Forecast did not use the selected fit and held-out data")
    if (meta["selection_sha256"] != selection_sha or meta["study_contract_sha256"] != contract_sha
            or meta["plan_sha256"] != frozen["plan_sha256"] or meta["source_hashes"] != fit["source_hashes"]
            or meta["native_checkpoint_hashes"] != fit["native_checkpoint_hashes"]):
        raise AssertionError("Forecast selection/source/pretrained provenance changed")
    for name, value in {"optimizer_steps": 0, "model_unchanged": True,
                        "global_selection_verified_before_holdout_load": True, "cache_created_this_trial": False,
                        "context": 336, "horizon": 48, "micro_groups": 4, "autocast": "bfloat16",
                        "autocast_weight_cache": False, "weight_dtype": "float32", "tf32": False}.items():
        if meta[name] != value:
            raise AssertionError(f"Forecast numerical/deployment contract mismatch: {name}")
    if (meta["stats_sha256"] != panel.stats_hash or meta["stats_sha256"] != fit["stats_sha256"]
            or meta["fit_std"] != panel.fit_std.tolist() or meta["channels"] != panel.channels
            or meta["target_indices"] != panel.target_indices.tolist()
            or Path(meta["fit_cache"]).resolve() != Path(fit["cache"]).resolve()
            or meta["best_step"] != fit["best_step"] or meta["trainable"] != fit["trainable"]):
        raise AssertionError("Forecast train statistics, cache, checkpoint or channel map mismatch")
    if (meta.get("checkpoint_reload_verified") is not True
            or meta["restored_adaptation_sha256"] != fit["audits"]["restored_adaptation_sha256"]):
        raise AssertionError("Forecast loaded parameters differ from the fit's restored checkpoint")
    verify_file_hashes(root, meta["protected_hashes"], checked)
    if sha256_file(path / "predictions.npz") != meta["predictions_sha256"]:
        raise AssertionError("Forecast prediction archive hash mismatch")
    arrays = load_prediction_archive(path / "predictions.npz", panel, ("cal", "eval"), with_unsorted=True)
    for split in ("cal", "eval"):
        info = meta["splits"][split]
        if (info["origin_sha256"] != array_hash(panel.origins[split])
                or info["target_sha256"] != array_hash(arrays[split + "_target"])
                or info["origins"] != len(panel.origins[split])
                or info["predictions_shape"] != list(arrays[split + "_predictions"].shape)
                or info["target_shape"] != list(arrays[split + "_target"].shape)):
            raise AssertionError("Forecast split shape/original target/origin audit mismatch")
        compare_float(info["unsorted_crossing"], float((np.diff(arrays[split + "_unsorted_predictions"], axis=2) < 0).mean()), "Native crossing diagnostic")
    return meta, arrays


def load_panels(root, dataset, checked):
    from .train import Panel

    prepared = Path(root) / "runs" / STUDY / "prepared"
    fit = Panel(prepared / f"{dataset}_fit.npz", "fit")
    holdout = Panel(prepared / f"{dataset}_holdout.npz", "forecast")
    archives = []
    for panel in (fit, holdout):
        with np.load(panel.path, allow_pickle=False) as archive:
            arrays = {name: archive[name].copy() for name in archive.files}
        archives.append(arrays)
        valid = np.isfinite(panel.target_values)
        mask = valid.copy()
        inactive = np.setdiff1d(np.arange(len(panel.channels)), panel.target_indices)
        mask[:, inactive] = False
        if not np.array_equal(arrays["observed_mask"], valid) or not np.array_equal(arrays["target_loss_mask"], mask):
            raise AssertionError("Prepared original/missing/target-only mask contract changed")
        timestamps = panel.timestamps.astype("datetime64[s]")
        if np.any(np.diff(timestamps) != np.timedelta64(1, "h")):
            raise AssertionError("Prepared timestamps must retain a complete hourly grid")
        if int(arrays["origin_stride_hours"]) != 24:
            raise AssertionError("Origin stride differs from the fixed daily sampling design")
        for split, origins in panel.origins.items():
            start, end = panel.metadata["boundary_indices"][split]
            expected = np.arange(start, end - 48 + 1, 24)
            if not np.array_equal(origins, expected):
                raise AssertionError("Origins were capped, changed, or cross their own target split boundary")
            if np.any(np.isfinite(panel.targets(split)).mean(axis=(0, 2)) < .70):
                raise AssertionError("A fixed target split failed its predeclared missingness gate")
        origin_counts = {"train": 63, "val": 13, "cal": 13, "eval": 83}
        if panel.metadata["origin_counts"] != origin_counts:
            raise AssertionError("Fixed 64/14/14/84-day split origin counts changed")
        if panel.metadata["contract"]["target_window_overlap_hours"] != 24:
            raise AssertionError("Daily H48 forecasts must overlap by 24 hours")
        for name, days in (("precontext", 14), ("train", 64), ("val", 14), ("cal", 14), ("eval", 84)):
            start, end = panel.metadata["boundary_indices"][name]
            if end - start != days * 24:
                raise AssertionError("A fixed temporal split duration changed")
    fit_arrays, holdout_arrays = archives
    n_fit = len(fit.target_values)
    if n_fit != fit.metadata["boundary_indices"]["val"][1]:
        raise AssertionError("Fit archive includes observations after validation selection")
    for name in ("context_values", "target_values", "timestamps", "observed_mask", "target_loss_mask"):
        if not np.array_equal(fit_arrays[name], holdout_arrays[name][:n_fit], equal_nan=fit_arrays[name].dtype.kind == "f"):
            raise AssertionError(f"Fit archive is not the exact held-out archive's historical prefix: {name}")
    for name in ("channels", "target_indices", "fit_mean", "fit_std", "fit_median", "quantiles"):
        if not np.array_equal(fit_arrays[name], holdout_arrays[name]):
            raise AssertionError(f"Fit/holdout train-statistic or channel mismatch: {name}")
    if fit.metadata["holdout_archive_sha256"] != sha256_file(holdout.path):
        raise AssertionError("Holdout was not fixed in the original fit data contract")
    start, end = fit.metadata["boundary_indices"]["train"]
    train = fit.target_values[start:end].astype(np.float64)
    for name, operation in (("fit_mean", np.nanmean), ("fit_std", np.nanstd), ("fit_median", np.nanmedian)):
        expected = operation(train, axis=0).astype(np.float32).astype(np.float64)
        if not np.array_equal(getattr(fit, name), expected):
            raise AssertionError(f"Scaling was not fitted exclusively to original train observations: {name}")
    raw = holdout.target_values
    for channel in range(len(holdout.channels)):
        observed = np.isfinite(raw[:, channel])
        previous = np.maximum.accumulate(np.where(observed, np.arange(len(raw)), -1))
        expected = np.where(previous >= 0, raw[np.maximum(previous, 0), channel], holdout.fit_median[channel])
        if not np.array_equal(holdout.context_values[:, channel], expected):
            raise AssertionError("Context filling used future measurements or changed observed inputs")
    source = fit.metadata["source"]
    verify_file_hashes(root, {source["source_path"]: source["source_sha256"]}, checked)
    if "archive_path" in source:
        verify_file_hashes(root, {source["archive_path"]: source["archive_sha256"]}, checked)
    return fit, holdout, fit_arrays, holdout_arrays


def load_raw(root, dataset, fit_panel, holdout_panel, fit_arrays, holdout_arrays):
    from . import raw as raw_api

    path = Path(root) / "runs" / STUDY / "raw" / dataset
    created = not (path / "result.json").exists()
    if created:
        raw_api.run(fit_panel.path, holdout_panel.path, path)
    meta = read_json(path / "result.json")
    if (meta.get("completed") is not True or meta["dataset"] != dataset or meta["method"] != "RAW_FULL_PAST_RIDGE"
            or meta["validation_target_used_for_fit"] is not False or meta["cal_eval_target_used_for_selection"] is not False):
        raise AssertionError("RAW completion or fit/selection boundary mismatch")
    if (meta["fit_data_sha256"] != sha256_file(fit_panel.path)
            or meta["holdout_data_sha256"] != sha256_file(holdout_panel.path)
            or meta["fit_manifest_holdout_sha256"] != meta["holdout_data_sha256"]
            or meta["source_sha256"] != sha256_file(Path(raw_api.__file__))
            or meta["predictions_sha256"] != sha256_file(path / "predictions.npz")
            or meta["model_sha256"] != sha256_file(path / "model.npz")):
        raise AssertionError("RAW source/data/prediction/model provenance mismatch")
    if meta["lambda_candidates"] != [.1, 10., 1000.] or meta["wall_seconds"] <= 0:
        raise AssertionError("RAW fixed lambda search or cost record changed")
    with np.load(path / "predictions.npz", allow_pickle=False) as archive:
        predictions = {key: archive[key].copy() for key in archive.files}
    with np.load(path / "model.npz", allow_pickle=False) as archive:
        saved = {key: archive[key].copy() for key in archive.files}
    q = fit_arrays["quantiles"].astype(np.float64)
    if not np.array_equal(q, predictions["quantiles"]) or not np.array_equal(q, saved["quantiles"]):
        raise AssertionError("RAW does not use the same native quantile grid")
    replay_started = time.perf_counter()
    _, x_train, y_train, mask_train = raw_api._split_arrays(fit_arrays, "train")
    _, x_val, y_val, mask_val = raw_api._split_arrays(fit_arrays, "val")
    expected_x = np.stack([fit_panel.context_values[o-336:o] for o in fit_panel.origins["train"]]).astype(np.float64)
    expected_x = ((expected_x - fit_panel.fit_mean) / fit_panel.fit_std).reshape(len(x_train), -1) / np.sqrt(336 * len(fit_panel.channels))
    if not np.array_equal(x_train, expected_x) or not np.array_equal(mask_train, np.isfinite(fit_panel.targets("train"))):
        raise AssertionError("RAW changed the common past inputs or original target mask")
    expected_y = (fit_panel.targets("train").astype(np.float64) - fit_panel.fit_mean[fit_panel.target_indices][None, :, None]) / fit_panel.fit_std[fit_panel.target_indices][None, :, None]
    if not np.array_equal(y_train, expected_y, equal_nan=True):
        raise AssertionError("RAW fitted labels differ from standardized original train targets")
    candidates = [raw_api.fit_candidate(x_train, y_train, mask_train, x_val, y_val, mask_val, alpha, q)
                  for alpha in (.1, 10., 1000.)]
    if [entry["lambda"] for entry in meta["candidate_val_scores"]] != [.1, 10., 1000.]:
        raise AssertionError("RAW validation candidate set is incomplete")
    for actual, candidate in zip(meta["candidate_val_scores"], candidates):
        compare_float(actual["val_score"], candidate["val_score"], "Replayed RAW candidate validation score", rtol=1e-10, atol=1e-12)
    best = min(candidates, key=lambda candidate: (candidate["val_score"], candidate["alpha"]))
    if meta["selected_lambda"] != best["alpha"] or float(saved["selected_lambda"]) != best["alpha"]:
        raise AssertionError("RAW lambda is not the validation-only argmin")
    compare_float(meta["val_score"], best["val_score"], "RAW selected validation score")
    for key in ("coef", "intercept", "n_obs"):
        if not np.allclose(saved[key], best["model"][key], rtol=1e-10, atol=1e-12):
            raise AssertionError("RAW saved coefficients do not reproduce train-only fitting")
    if not np.allclose(saved["residual_offsets_scaled"], best["residual_offsets"], rtol=1e-10, atol=1e-12):
        raise AssertionError("RAW distribution offsets were not fitted to train residuals")
    for target in range(2):
        for h in range(48):
            mask = mask_train[:, target, h]
            x, y = x_train[mask], y_train[mask, target, h]
            beta, intercept = saved["coef"][target, h], saved["intercept"][target, h]
            error = x @ beta + intercept - y
            gradient = x.T @ error + best["alpha"] * beta
            if not np.allclose(gradient, 0, rtol=0, atol=1e-7) or abs(error.sum()) > 1e-7:
                raise AssertionError("RAW coefficients violate unpenalized-intercept SSE ridge normal equations")
    for split, panel, arrays in (("val", fit_panel, fit_arrays), ("cal", holdout_panel, holdout_arrays), ("eval", holdout_panel, holdout_arrays)):
        target = panel.targets(split)
        prediction = predictions[split + "_predictions"]
        if (not np.array_equal(predictions[split + "_origins"], panel.origins[split])
                or not np.array_equal(predictions[split + "_target"], target, equal_nan=True)
                or not np.array_equal(predictions[split + "_target_mask"], np.isfinite(target))):
            raise AssertionError("RAW holdout cell/target/mask differs from the FM evaluation")
        forecast_arrays(prediction, target, panel.fit_std[panel.target_indices], q)
        x = raw_api._features(arrays, panel.origins[split])
        point = np.einsum("nf,thf->nth", x, saved["coef"]) + saved["intercept"][None]
        expected = point[:, :, None, :] + saved["residual_offsets_scaled"][None, :, :, None]
        expected = expected * panel.fit_std[panel.target_indices][None, :, None, None] + panel.fit_mean[panel.target_indices][None, :, None, None]
        if not np.array_equal(prediction, expected.astype(np.float32)):
            raise AssertionError("RAW predictions do not reproduce the saved train-only model")
        if not np.array_equal(prediction, np.sort(prediction, axis=2)):
            raise AssertionError("RAW quantiles crossed")
    value, _, _ = score_prediction(predictions["val_predictions"], predictions["val_target"], fit_panel.fit_std[fit_panel.target_indices], q)
    compare_float(value, meta["val_score"], "RAW saved raw-space validation score")
    return meta, predictions, path, {"created_this_analysis": created, "fit_wall_seconds": meta["wall_seconds"],
                                     "replay_audit_wall_seconds": time.perf_counter() - replay_started,
                                     "all_lambda_train_only_replay": True, "normal_equations_verified": True}


def write_csv(path, rows, fields):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def run(root):
    from .run_study import contract as runner_contract, verify_smoke

    started = time.perf_counter()
    root = Path(root).resolve()
    study, output = root / "runs" / STUDY, root / "results" / STUDY
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Preserve previous complete or partial analysis outputs; automatic overwrite is forbidden")
    completed = read_json(study / "completed.json")
    fit_completed = read_json(study / "fit_completed.json")
    if (completed.get("completed") is not True or completed.get("fit_trials") != 28 or completed.get("forecasts") != 14
            or fit_completed.get("completed") is not True or fit_completed.get("fit_trials") != 28
            or fit_completed.get("adaptation_fits") != 26 or fit_completed.get("frozen_cache_trials") != 2):
        raise AssertionError("Holdout analysis requires all 28 fits, frozen selection and 14 completed forecasts")
    frozen = read_json(study / "study_contract.json")
    if frozen != runner_contract(root) or read_json(study / "smoke_contract.json") != frozen:
        raise AssertionError("Current source/plan/data contract differs from the production and S0 contracts")
    smoke = read_json(study / "smoke_completed.json")
    if smoke.get("completed") is not True:
        raise AssertionError("Current S0 is incomplete")
    verify_smoke(root, smoke["trials"])
    selection = read_json(study / "selection.json")
    contract_sha = sha256_file(study / "study_contract.json")
    selection_sha = sha256_file(study / "selection.json")
    if selection["study_contract_sha256"] != contract_sha:
        raise AssertionError("Selection was not frozen on this study contract")
    trials, forecasts = read_json(study / "trials.json"), read_json(study / "forecasts.json")
    choices = read_json(study / "development_selection.json")
    _, selected_lookup, _ = verify_entries(trials, selection, choices, forecasts, root)
    raw_paths = [study / "raw" / dataset for dataset in DATASETS]
    if not all((path / "result.json").is_file() for path in raw_paths):
        raise AssertionError("Run both RAW baselines under their separate CPU guards before production analysis")
    raw_guard_paths = [study / "raw_guards" / dataset for dataset in DATASETS]
    raw_guards = {dataset: validate_guard(path, guard_directory=True) for dataset, path in zip(DATASETS, raw_guard_paths)}
    fit_seconds = sum(entry["guard_seconds"] for entry in trials)
    forecast_seconds = sum(entry["forecast_guard_seconds"] for entry in forecasts)
    compare_float(fit_seconds, completed["fit_guard_seconds"], "Completed fit cost")
    compare_float(fit_seconds, fit_completed["fit_guard_seconds"], "Fit completion cost")
    compare_float(forecast_seconds, completed["forecast_guard_seconds"], "Completed forecast cost")
    analysis_sources = {str(Path(__file__).resolve()): sha256_file(__file__)}
    checked = set()
    panels = {dataset: load_panels(root, dataset, checked) for dataset in DATASETS}
    fit_results = {}
    for entry in trials:
        fit_results[trial_key(entry)] = verify_fit(root, entry, panels[entry["dataset"]][0], frozen, checked)
    initial_by_key, samplers = {}, {}
    for entry in trials:
        meta = fit_results[trial_key(entry)]
        initial_key = (entry["dataset"], entry["method"], entry["seed"])
        initial = meta["audits"]["initial_adaptation_sha256"]
        if initial_by_key.setdefault(initial_key, initial) != initial:
            raise AssertionError("LR candidates changed their shared model initialization")
        if entry["method"] != "F0":
            seed_key = (entry["dataset"], entry["seed"])
            if samplers.setdefault(seed_key, meta["sampler_sha256"]) != meta["sampler_sha256"]:
                raise AssertionError("Methods used different train samples for the same optimizer seed")
        f0 = fit_results[(entry["dataset"], "F0", 12000, 0.)]
        compare_float(meta["validation_history"][0]["val_score"], f0["val_score"], "Method step0 versus source F0", rtol=2e-6, atol=1e-8)
        if Path(meta["cache"]).resolve() != Path(f0["cache"]).resolve():
            raise AssertionError("Methods within a source must share the exact frozen feature cache")
    selected_rows, trial_rows, diagnostics, statistics, source_outputs = [], [], {}, {}, {}
    forecast_results, artifact_hashes, raw_costs = {}, {}, {}
    selected_trial_keys = {trial_key(entry) for entry in selection["selected"]}
    for entry in trials:
        meta = fit_results[trial_key(entry)]
        grid = GRIDS.get(entry["method"], (0.,))
        trial_rows.append({"source": entry["dataset"], "stage": "GPU_FIT_SORT_SELECTION", "method": entry["method"],
                           "seed": entry["seed"], "lr_or_lambda": entry["lr"], "val_score": meta["val_score"],
                           "best_step": meta["best_step"], "selected_for_forecast": trial_key(entry) in selected_trial_keys,
                           "lr_boundary": entry["method"] != "F0" and entry["lr"] in (min(grid), max(grid)),
                           "step_boundary": entry["method"] != "F0" and meta["best_step"] in (0, 200), "trainable": meta["trainable"],
                           "fit_wall_seconds": meta["wall_seconds"], "guard_seconds": entry["guard_seconds"], "path": entry["path"]})
        diagnostics[entry["path"]] = {"validation_history": meta["validation_history"], "best_step": meta["best_step"],
                                       "selected_for_forecast": trial_key(entry) in selected_trial_keys,
                                       "initial_adaptation_sha256": meta["audits"]["initial_adaptation_sha256"]}
        for name in ("result.json", "predictions.npz", "best_trainable.pt", "trial_contract.json", "guard/status.json", "guard/resource_log.jsonl"):
            artifact_hashes[str(root / entry["path"] / name)] = sha256_file(root / entry["path"] / name)

    def add_procedures(dataset, role, method, seed, lr, path, meta, arrays, trainable):
        holdout = panels[dataset][1]
        q = arrays["quantiles"]
        if not np.array_equal(q, panels[dataset][2]["quantiles"]):
            raise AssertionError("FM and RAW must use the exact same native quantile levels")
        scale = holdout.fit_std[holdout.target_indices]
        cal, target_cal = arrays["cal_predictions"], arrays["cal_target"]
        forecast, target = arrays["eval_predictions"], arrays["eval_target"]
        offsets = qcal_offsets(cal, target_cal, q)
        unsorted = arrays.get("eval_unsorted_predictions", forecast)
        key = f"{dataset}/{role}/seed_{seed}"
        details = {"calibration_offsets": offsets.tolist(), "calibration_observed_cells": np.isfinite(target_cal).sum(axis=(0, 2)).tolist(),
                   "offset_estimation": "Per-target and per-quantile empirical q residual on C_cal, horizon pooled; fixed before evaluation"}
        for procedure, prediction in (("SORT", forecast), ("QCAL", apply_qcal(forecast, offsets))):
            _, sums, counts = score_prediction(prediction, target, scale, q)
            statistics.setdefault(dataset, {}).setdefault(procedure, {}).setdefault(role, []).append((sums, counts))
            detail = score_details(prediction, target, scale, q, crossing_prediction=unsorted)
            details[procedure] = detail
            selected_rows.append({"source": dataset, "role": role, "method": method, "seed": seed, "lr": lr,
                                  "procedure": procedure, "score": detail["score"], "relative_improvement_over_f0": None,
                                  "coverage80": detail["coverage80"], "width80_scaled": detail["width80_scaled"],
                                  "median_mse_scaled": detail["median_mse_scaled"], "crossing_before_sort": detail["crossing_before_sort"],
                                  "best_step": meta.get("best_step", ""), "val_score": meta["val_score"], "trainable": trainable,
                                  "path": str(Path(path).resolve().relative_to(root)).replace("\\", "/")})
        diagnostics[key] = details

    # Keep each method's seed axis in its fixed numeric order, irrespective of file ordering.
    forecasts = sorted(forecasts, key=lambda entry: (entry["dataset"], entry["role"], entry["seed"]))
    for entry in forecasts:
        fit = fit_results[trial_key(entry)]
        forecast_meta, arrays = verify_forecast(root, entry, fit, panels[entry["dataset"]][1], frozen, selection_sha, contract_sha, checked)
        forecast_results[entry["path"]] = forecast_meta
        add_procedures(entry["dataset"], entry["role"], entry["method"], entry["seed"], entry["lr"],
                       root / entry["path"], fit, arrays, fit["trainable"])
        for name in ("result.json", "predictions.npz", "trial_contract.json", "guard/status.json", "guard/resource_log.jsonl"):
            artifact_hashes[str(root / entry["path"] / name)] = sha256_file(root / entry["path"] / name)
    for dataset in DATASETS:
        raw_meta, arrays, path, cost = load_raw(root, dataset, *panels[dataset])
        raw_costs[dataset] = cost
        add_procedures(dataset, "RAW", raw_meta["method"], -1, raw_meta["selected_lambda"], path, raw_meta, arrays, "")
        trial_rows.append({"source": dataset, "stage": "CPU_RAW_SORT_SELECTION", "method": raw_meta["method"], "seed": -1,
                           "lr_or_lambda": raw_meta["selected_lambda"], "val_score": raw_meta["val_score"], "best_step": "",
                           "selected_for_forecast": True, "lr_boundary": raw_meta["selected_lambda"] in (.1, 1000.),
                           "step_boundary": "", "trainable": "", "fit_wall_seconds": raw_meta["wall_seconds"],
                           "guard_seconds": "", "path": str(path.relative_to(root)).replace("\\", "/")})
        diagnostics[f"{dataset}/RAW_selection"] = raw_meta
        for name in ("result.json", "predictions.npz", "model.npz"):
            artifact_hashes[str(path / name)] = sha256_file(path / name)
        for name in ("status.json", "resource_log.jsonl"):
            guard_file = study / "raw_guards" / dataset / name
            artifact_hashes[str(guard_file)] = sha256_file(guard_file)
        source_outputs[dataset] = source_effects(statistics[dataset])
        source_outputs[dataset]["selected_head_family"] = choices[dataset]["H"]["method"]
        source_outputs[dataset]["selected_head_lr"] = choices[dataset]["H"]["lr"]
        source_outputs[dataset]["selected_lora_lr"] = choices[dataset]["OFF_LORA"]["lr"]
        source_outputs[dataset]["optimizer_seeds"] = list(SEEDS)
        source_outputs[dataset]["target_names"] = np.asarray(panels[dataset][0].channels)[panels[dataset][0].target_indices].tolist()
    expected_rows = {(dataset, role, seed, procedure) for dataset in DATASETS for role in ("F0", "H", "OFF_LORA", "RAW")
                     for seed in ((12000,) if role == "F0" else ((-1,) if role == "RAW" else SEEDS)) for procedure in ("SORT", "QCAL")}
    if len(selected_rows) != 32 or {(row["source"], row["role"], row["seed"], row["procedure"]) for row in selected_rows} != expected_rows:
        raise AssertionError("Expected exactly 32 selected SORT/QCAL procedure rows")
    if len(trial_rows) != 30:
        raise AssertionError("Full fit report must preserve 28 GPU and 2 RAW selection rows")
    baselines = {(row["source"], row["procedure"]): row["score"] for row in selected_rows if row["role"] == "F0"}
    for row in selected_rows:
        row["relative_improvement_over_f0"] = 1 - row["score"] / baselines[(row["source"], row["procedure"])]
    effects = {"completed": True, "sources": source_outputs, "primary_family_size": 2,
               "primary_confidence_each": CONFIDENCE, "bootstrap_replicates": BOOTSTRAP_REPLICATES,
               "bootstrap_seed": BOOTSTRAP_SEED, "primary_block_days": 7, "moving_block_variant": "noncircular overlapping blocks; truncate final block to original origin count",
               "overall_gate": "enter_module_hypothesis_research" if all(value["primary"]["decision"] == "conditional_practical_improvement" and not value["raw_veto"]["applies"] for value in source_outputs.values()) else "internal_peft_need_not_established_by_this_screen",
               "scope": "Two fixed real-source windows, static adapters and three fixed optimizer seeds; conditional temporal bootstrap, not whole selection/training uncertainty",
               "selection_limitation": "SORT selects checkpoints/family/LR before common QCAL; this compares selected procedures, not optimally calibrated model families. H has six development candidates versus three LoRA candidates.",
               "calibration_limitation": "Fixed horizon-pooled empirical quantile offsets on 13 dependent daily calibration origins; no conformal coverage guarantee"}
    fit_paths, forecast_paths = [root / e["path"] for e in trials], [root / e["path"] for e in forecasts]
    resources = {"new_fit_gpu": summarize_resources(fit_paths, "28 new production fit guards: 26 adaptation fits and 2 F0/cache jobs"),
                 "new_forecast_gpu": summarize_resources(forecast_paths, "14 new selected holdout inference guards, no optimizer updates"),
                 "new_all_production_gpu": summarize_resources(fit_paths + forecast_paths, "All 42 new production GPU guards; CPU analysis and S0 excluded"),
                 "new_raw_cpu": summarize_resources(raw_guard_paths, "Two new RAW CPU guards, distinct from GPU and the enclosing analysis guard", guard_directory=True)}
    unused = [e for e in trials if trial_key(e) not in selected_trial_keys]
    costs = {"completed": True, "fit_trials": 28, "adaptation_fits": 26, "f0_cache_jobs": 2, "forecast_jobs": 14,
             "new_production_gpu_jobs": 42, "fit_guard_seconds": fit_seconds, "forecast_guard_seconds": forecast_seconds,
             "production_gpu_guard_seconds": fit_seconds + forecast_seconds,
             "runner_invocation_wall_seconds": completed["invocation_wall_seconds"],
             "fit_completion_invocation_wall_seconds": fit_completed["invocation_wall_seconds"],
             "unused_fit_count": len(unused), "unused_fit_guard_seconds_included": sum(e["guard_seconds"] for e in unused),
             "optimizer_seconds_included": sum(meta["optimizer_seconds_total"] for meta in fit_results.values()),
             "raw_cpu": raw_costs, "raw_cpu_fit_seconds": sum(value["fit_wall_seconds"] for value in raw_costs.values()),
             "raw_cpu_jobs": 2, "raw_cpu_guard_seconds": sum(value["elapsed_seconds"] for value in raw_guards.values()),
             "raw_cpu_replay_seconds": sum(value["replay_audit_wall_seconds"] for value in raw_costs.values()),
             "s0_separate": {"jobs": 8, "invocation_wall_seconds": smoke["wall_seconds"],
                              "guard_seconds": sum(e["guard_seconds"] for e in smoke["trials"])},
             "cost_scope": "Each phase is separately reported; invocation wall and child guard durations overlap and must not be added. RAW fits use two separate CPU guards; RAW replay is inside CPU analysis wall."}
    if frozen != runner_contract(root):
        raise AssertionError("Source/plan/data contract changed during analysis")
    verify_file_hashes(root, artifact_hashes, set())
    verify_file_hashes(root, analysis_sources, set())
    costs["analysis_wall_seconds_before_serialization"] = time.perf_counter() - started
    output.mkdir(parents=True, exist_ok=True)
    selected_rows.sort(key=lambda row: (row["source"], row["role"], row["seed"], row["procedure"]))
    write_csv(output / "selected_results.csv", selected_rows, CSV_FIELDS)
    write_csv(output / "all_results.csv", trial_rows, list(trial_rows[0]))
    for name, value in (("effects.json", effects), ("costs.json", costs), ("resource_summary.json", resources), ("diagnostics.json", diagnostics)):
        write_json(output / name, value)
    outputs = {path.name: sha256_file(path) for path in sorted(output.iterdir()) if path.is_file()}
    verification = {"passed": True, "completed": True, "fit_trial_count": 28, "adaptation_fit_count": 26,
                    "forecast_count": 14, "selected_native_count": 14, "selected_procedure_rows": 32,
                    "all_fit_rows": 30, "raw_count": 2, "optimizer_seeds": list(SEEDS),
                    "study_contract_sha256": contract_sha, "selection_sha256": selection_sha,
                    "analysis_sources": analysis_sources, "artifact_hashes": artifact_hashes, "output_hashes": outputs,
                    "checks": ["exact trial, candidate, selection and forecast identities", "all 42 GPU guards completed without failure artifacts",
                               "current S0/source/plan/data/checkpoint contracts", "observed target masks and train-only scaling replay",
                               "all three frozen future feature patches and common numerical group path", "checkpoint, initialization and sampler provenance",
                               "saved prediction and loss/count reconstruction", "RAW full candidate train-only replay and ridge normal equations",
                               "QCAL exclusively on C_cal; paired day-block sufficient-statistic bootstrap"]}
    write_json(output / "verification.json", verification)
    print(json.dumps({"passed": True, "selected_rows": 32, "overall_gate": effects["overall_gate"], "output": str(output)}), flush=True)
    return verification


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    run(args.root)


if __name__ == "__main__":
    main()
