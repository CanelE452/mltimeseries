"""Verify the completed train-only lag study and report paired conditional effects."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from statistics import NormalDist
import time

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "2"

import numpy as np

from experiments.peft_module_ablation_v1.analyse import (
    compare_float, episode_loss, expected_map, read_json, resource_samples,
    sha256_file, summarize_resources, validate_adaptation_checkpoint_file,
    validate_prediction_archive, write_json,
)
from . import data as data_api
from . import raw as raw_api
from .run_study import PLAN, RATES, STUDY, contract as runner_contract, verify_smoke


BOOTSTRAP_REPLICATES = 4000
BOOTSTRAP_SEED = 2026090810
DELTA = .01
FM_METHODS = ("F0", "ATTN", "ALIGN_F0", "ALIGN_ATTN")
TRAINED = ("ATTN", "ALIGN_ATTN")
CSV_FIELDS = ["method", "corpus", "lr", "input_mode", "score", "val_score",
              "relative_improvement_over_f0", "median_mse", "oracle_mean_error_mse",
              "oracle_mean_error_mse_over_variance", "coverage80", "width80", "crossing",
              "best_step", "trainable", "wall", "path", "origin"]


def trial_key(entry):
    return int(entry["corpus"]), entry["method"], float(entry["lr"])


def expected_trial_keys():
    return {(c, method, lr) for c in range(3) for method in FM_METHODS
            for lr in (RATES if method in TRAINED else (.001,))}


def verify_entries(trials, selected, choices, root):
    lookup = {trial_key(entry): entry for entry in trials}
    if len(trials) != 18 or set(lookup) != expected_trial_keys():
        raise AssertionError("Expected exactly 18 unique GPU trial keys")
    for key, entry in lookup.items():
        if set(entry) != {"condition", "corpus", "method", "input_mode", "lr", "path", "val_score"}:
            raise AssertionError("Unexpected trial metadata fields")
        corpus, method, lr = key
        expected_mode = "aligned" if method.startswith("ALIGN_") else "raw"
        expected_path = Path(root) / "runs" / STUDY / "trials" / f"Q00_c{corpus}" / method / f"lr_{lr:.0e}"
        if (entry["condition"], entry["input_mode"]) != ("Q00", expected_mode):
            raise AssertionError("Trial condition/input mode mismatch")
        if (Path(root) / entry["path"]).resolve() != expected_path.resolve():
            raise AssertionError("Noncanonical trial path")
        if not np.isfinite(float(entry["val_score"])):
            raise AssertionError("Nonfinite validation selection score")
    if set(choices) != set(TRAINED):
        raise AssertionError("Missing or unexpected LR selection procedure")
    for method in TRAINED:
        choice = choices[method]
        expected = [lookup[(0, method, lr)] for lr in RATES]
        if set(choice) != {"lr", "candidates"} or len(choice["candidates"]) != 2:
            raise AssertionError("LR choice must contain both corpus0 candidates")
        actual = {trial_key(entry): entry for entry in choice["candidates"]}
        if actual != {trial_key(entry): entry for entry in expected}:
            raise AssertionError("LR candidates must exactly reproduce full trial entries")
        best = min(expected, key=lambda entry: (entry["val_score"], entry["lr"]))
        if choice["lr"] != best["lr"]:
            raise AssertionError("LR choice is not the corpus0 validation argmin")
    expected_selected = {(c, method, choices[method]["lr"] if method in TRAINED else .001)
                         for c in range(3) for method in FM_METHODS}
    if len(selected) != 12 or {trial_key(entry) for entry in selected} != expected_selected:
        raise AssertionError("Expected exactly 12 selected FM keys with fixed corpus0 choices")
    if any(entry != lookup[trial_key(entry)] for entry in selected):
        raise AssertionError("Selected metadata differs from its original trial")
    return lookup


def bootstrap_weights(n, replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED):
    return np.random.default_rng(seed).multinomial(n, np.full(n, 1 / n), size=replicates) / n


def effect_ratio(numerator, denominator, weights, confidence=.975):
    numerator, denominator = np.asarray(numerator, dtype=np.float64), np.asarray(denominator, dtype=np.float64)
    if numerator.shape != denominator.shape or numerator.ndim != 2 or numerator.shape[0] != 3:
        raise ValueError("Effects require matching [three fitted corpora, paired episodes] arrays")
    if weights.ndim != 2 or weights.shape[1] != numerator.shape[1]:
        raise ValueError("Bootstrap must resample the common episode axis")
    if not np.isfinite(numerator).all() or not np.isfinite(denominator).all():
        raise ValueError("Effect inputs must be finite")
    curve, scale = numerator.mean(axis=0), denominator.mean(axis=0)
    denominators = weights @ scale
    if np.any(denominators <= 0) or np.any(denominator.mean(axis=1) <= 0):
        raise ValueError("Bootstrap and fitted-corpus denominators must be positive")
    draws = (weights @ curve) / denominators
    tail = (1 - confidence) / 2
    return {"value": float(curve.mean() / scale.mean()),
            "corpus_values": (numerator.mean(axis=1) / denominator.mean(axis=1)).tolist(),
            "ci": np.quantile(draws, [tail, 1 - tail]).tolist(), "confidence": confidence,
            "scope": "paired episodes conditional on three fitted corpus/optimizer and validation selections"}


def classify_effect(effect, delta=DELTA):
    low, high = effect["ci"]
    corpus_values = np.asarray(effect["corpus_values"])
    if low > delta and np.all(corpus_values > 0):
        return "repeated_practical_improvement"
    if high < -delta and np.all(corpus_values < 0):
        return "repeated_practical_harm"
    if low >= -delta and high <= delta:
        return "conditional_practical_equivalence"
    return "inconclusive"


def contrast_family(losses, rates, weights):
    def stack(method):
        lr = rates[method] if method in TRAINED else .001
        return np.stack([losses[(c, method, lr)] for c in range(3)])
    f0, attn, aligned, adapted = (stack(method) for method in FM_METHODS)
    effects = {}
    for name, numerator, definition in (
        ("alignment_over_attention", attn - aligned, "(S_ATTN - S_ALIGN_F0) / S_F0"),
        ("attention_after_alignment", aligned - adapted, "(S_ALIGN_F0 - S_ALIGN_ATTN) / S_F0"),
    ):
        effect = effect_ratio(numerator, f0, weights)
        effect.update({"decision": classify_effect(effect), "definition": definition})
        effects[name] = effect
    return {"effects": effects, "lr_by_method": rates, "positive_means_improvement": True}


def summarize_scores(prediction, target, quantiles, oracle_mean):
    med, lower, upper = [int(np.argmin(abs(quantiles - value))) for value in (.5, .1, .9)]
    median = prediction[:, med]
    error = float(np.mean((median - oracle_mean) ** 2))
    variance = float(np.var(oracle_mean))
    return {"median_mse": float(np.mean((median - target) ** 2)), "oracle_mean_error_mse": error,
            "oracle_mean_error_mse_over_variance": error / variance if variance > 0 else None,
            "coverage80": float(np.mean((target >= prediction[:, lower]) & (target <= prediction[:, upper]))),
            "width80": float(np.mean(prediction[:, upper] - prediction[:, lower])),
            "crossing": float(np.mean(np.diff(prediction, axis=1) < 0))}


def verify_lag_selection(arrays, saved, data_hash):
    reproduced = data_api.select_lags_from_train(arrays["context_train"], arrays["target_train"])
    if saved["data_sha256"] != data_hash:
        raise AssertionError("Lag reference data hash mismatch")
    for key, value in reproduced.items():
        if key == "fit_seconds":
            continue
        if key in ("scores", "pearson_correlations"):
            if set(saved.get(key, {})) != set(value) or any(
                not np.allclose(saved[key][channel], values, rtol=1e-12, atol=1e-15)
                for channel, values in value.items()
            ):
                raise AssertionError(f"Train-only lag selector replay mismatch: {key}")
            continue
        if saved.get(key) != value:
            raise AssertionError(f"Train-only lag selector replay mismatch: {key}")
    if saved["candidate_lags"] != list(range(16, 129)) or saved["n_future_labels"] != 1024:
        raise AssertionError("Lag dictionary or supervised-label count changed")
    lags = [saved["selected_lags"][channel] for channel in data_api.CHANNELS]
    if not np.array_equal(arrays["selected_lags"], lags):
        raise AssertionError("NPZ selected lag mismatch")
    if not np.array_equal(arrays["candidate_lags"], saved["candidate_lags"]):
        raise AssertionError("NPZ candidate range mismatch")
    for split in ("train", "val", "eval"):
        context = arrays[f"context_{split}"]
        direct = np.stack([context[:, channel, 256 + np.arange(16) - lag]
                           for channel, lag in enumerate(lags)], axis=-1)
        if not np.array_equal(direct, arrays[f"raw_features_{split}"]):
            raise AssertionError(f"Original past-index feature mismatch: {split}")
    return reproduced


def load_data(study, corpus):
    path = study / "data" / f"Q00_c{corpus}.npz"
    lag_path = study / "data" / f"Q00_c{corpus}_lags.json"
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    manifest = json.loads(arrays["manifest_json"].item())
    if (manifest["condition"], manifest["corpus"], manifest["base_seed"]) != ("Q00", corpus, data_api.BASE_SEED):
        raise AssertionError("Fresh data identity mismatch")
    for split, n in data_api.N_BY_SPLIT.items():
        if arrays[f"context_{split}"].shape != (n, 3, 256) or arrays[f"target_{split}"].shape != (n, 16):
            raise AssertionError("Data split shape mismatch")
        ids = arrays[f"episode_ids_{split}"]
        if ids.shape != (n,) or len(np.unique(ids)) != n:
            raise AssertionError("Missing or duplicate episode IDs")
        if not all(str(value).startswith(f"trainlag_b{data_api.BASE_SEED}_{split}_") for value in ids):
            raise AssertionError("Fresh data reused the old episode ID namespace")
        for prefix in ("context", "target", "oracle_mean", "oracle_quantiles", "raw_features"):
            if not np.isfinite(arrays[f"{prefix}_{split}"]).all():
                raise AssertionError("Nonfinite prepared array")
    for a, b in (("train", "val"), ("train", "eval"), ("val", "eval")):
        if np.intersect1d(arrays[f"episode_ids_{a}"], arrays[f"episode_ids_{b}"]).size:
            raise AssertionError("Episode IDs overlap across splits")
    if not np.array_equal(arrays["quantiles"], data_api.QUANTILES):
        raise AssertionError("Native quantile grid mismatch")
    generator = manifest["generator"]
    z = np.asarray([NormalDist().inv_cdf(float(q)) for q in arrays["quantiles"]])
    for split in ("train", "val", "eval"):
        context = arrays[f"context_{split}"].astype(np.float64)
        lead = 256 + np.arange(16)
        oracle = generator["a"] * context[:, 0, lead - generator["self_lag"]]
        oracle += generator["b"] * (np.cos(generator["theta_radians"]) * context[:, 1, lead - generator["driver_lag"]]
                                    + np.sin(generator["theta_radians"]) * context[:, 2, lead - generator["driver_lag"]])
        if not np.allclose(oracle, arrays[f"oracle_mean_{split}"], rtol=1e-5, atol=5e-7):
            raise AssertionError("Oracle conditional mean does not match its original past indices")
        oracle_q = oracle[:, None, :] + generator["sigma"] * z[None, :, None]
        if not np.allclose(oracle_q, arrays[f"oracle_quantiles_{split}"], rtol=1e-5, atol=8e-7):
            raise AssertionError("Oracle quantiles do not match the declared conditional distribution")
    saved = read_json(lag_path)
    verify_lag_selection(arrays, saved, sha256_file(path))
    return arrays, saved, path, lag_path


def verify_shared_splits(datasets):
    for split in ("val", "eval"):
        for prefix in ("context", "target", "oracle_mean", "oracle_quantiles", "episode_ids"):
            key = f"{prefix}_{split}"
            if any(not np.array_equal(datasets[0][key], other[key]) for other in datasets[1:]):
                raise AssertionError(f"Paired corpus evaluation/validation arrays differ: {key}")
    hashes = [data_api.array_hash(item["context_train"], item["target_train"]) for item in datasets]
    if len(set(hashes)) != 3:
        raise AssertionError("Training corpora are not independent array samples")


def validate_guard(path):
    status = read_json(path / "guard/status.json")
    if not status.get("completed") or status.get("returncode") != 0 or status.get("reasons"):
        raise AssertionError(f"Failed or interrupted GPU guard: {path}")
    if (path / "failure.json").exists() or (path / "guard/safety_stop.json").exists():
        raise AssertionError(f"Failure artifact must not be reported as successful: {path}")
    if not (path / "guard/resource_log.jsonl").is_file() or not resource_samples([path]):
        raise AssertionError(f"Missing resource measurements: {path}")
    return status


def verify_prediction(path, arrays, meta):
    with np.load(path / "predictions.npz", allow_pickle=False) as saved:
        for split in ("val", "eval"):
            if saved[f"{split}_episode_losses"].shape != (len(arrays[f"target_{split}"]),):
                raise AssertionError("Saved per-episode loss vector has the wrong shape")
    archive = validate_prediction_archive(path, arrays)
    for split in ("val", "eval"):
        if not np.isfinite(archive[f"{split}_predictions"]).all():
            raise AssertionError("Nonfinite saved predictions")
        compare_float(archive[f"{split}_losses"].mean(), meta[f"{split}_score"], f"{split} score", path)
    return archive


def verify_file_hashes(root, hashes, checked):
    if not hashes:
        raise AssertionError("Empty protected-file hash contract")
    for name, expected in hashes.items():
        path = (root / name).resolve()
        key = (str(path), expected)
        if key not in checked:
            if sha256_file(path) != expected:
                raise AssertionError(f"Protected source/data/model/cache changed: {path}")
            checked.add(key)


def verify_input_scope(scope, mode, lags):
    if (scope["input_mode"], scope["selected_lags"], scope["Y_past_unchanged"],
        scope["Y_future_mask"], scope["additional_original_future_observations"],
        scope["Y_lag_used_by_FM"]) != (mode, lags, True, 0, 0, False):
        raise AssertionError("FM information contract changed")
    expected = None
    if mode == "aligned":
        expected = {name: {"lag": lag, "retained_past_source_range": [0, 255 - lag],
                           "future_source_indices": (256 + np.arange(16) - lag).tolist(),
                           "padding": "NaN", "original_recent_values_dropped": lag - 16}
                    for name, lag in lags.items() if name in ("U", "V")}
    if scope["driver_alignment"] != expected:
        raise AssertionError("Retimed input must reference the declared original past indices")


def verify_fm_metadata(root, path, entry, arrays, lag, data_path, lag_path, checked):
    meta, guard = read_json(path / "result.json"), validate_guard(path)
    c, method, lr = trial_key(entry)
    trained = method in TRAINED
    base = "ATTN_ONLY" if trained else "F0"
    mode, count, updates = entry["input_mode"], 1179648 if trained else 0, 200 if trained else 0
    required_true = ("completed", "wrapped_completed", "inner_training_completed")
    if any(meta.get(name) is not True for name in required_true):
        raise AssertionError("Wrapper completion gate did not pass")
    expected = {"method": method, "base_method": base, "input_mode": mode,
                "seed": 8100 + c, "lr": lr, "smoke": False, "steps_completed": updates,
                "trainable": count, "trainable_parameters": count, "head_lr": None,
                "head_only_updates": 0, "phase_transitions": [], "context": 256, "horizon": 16,
                "channel_order": ["Y", "U", "V"], "target_channel": "Y", "micro_groups": 4,
                "effective_groups": 8, "autocast": "bfloat16", "autocast_weight_cache": False,
                "weight_dtype": "float32", "dropout": 0., "gradient_clip_norm": 1.,
                "weight_decay": 0., "torch_threads": 2, "interop_threads": 1,
                "episode_counts": data_api.N_BY_SPLIT,
                "data_sha256": sha256_file(data_path), "lag_file_sha256": sha256_file(lag_path),
                "lag_selection_input_sha256": lag["input_array_hash"], "selected_lags": lag["selected_lags"]}
    for name, value in expected.items():
        if meta[name] != value:
            raise AssertionError(f"FM result contract mismatch: {name} at {path}")
    if (meta["metadata"]["condition"], meta["metadata"]["corpus"], meta["metadata"]["base_seed"]) != (
            "Q00", c, data_api.BASE_SEED):
        raise AssertionError("FM result data metadata mismatch")
    if entry["val_score"] != meta["val_score"]:
        raise AssertionError("Trial listing score differs from the completed result")
    validate_adaptation_checkpoint_file(path, meta)
    verify_file_hashes(root, meta["source_hashes"], checked)
    wrapper = meta["wrapper_contract"]
    if wrapper != read_json(path / "wrapper_contract.json"):
        raise AssertionError("Saved wrapper contract was replaced")
    for name in ("method", "base_method", "data_sha256", "lag_file_sha256", "lag_selection_input_sha256",
                 "trainable_parameters", "module_map"):
        if wrapper[name] != meta[name]:
            raise AssertionError(f"Wrapper/result contract mismatch: {name}")
    if wrapper["plan_sha256"] != sha256_file(root / PLAN) or wrapper["new_residual_head"]:
        raise AssertionError("Wrapper plan or residual-head contract changed")
    for field in ("source_hashes", "native_source_hashes", "protected_hashes"):
        verify_file_hashes(root, wrapper[field], checked)
    verify_input_scope(wrapper["input_scope"], mode, lag["selected_lags"])
    verify_input_scope(meta["audits"]["input_scope"], mode, lag["selected_lags"])
    for name in ("zero_update_identity", "frozen_parameters_verified", "checkpoint_reload_verified",
                 "checkpoint_includes_all_adaptive_parameters", "trainable_map_verified"):
        if meta["audits"].get(name) is not True:
            raise AssertionError(f"Native-loop audit failed: {name}")
    audit = meta["audits"]
    if audit["backbone_before_sha256"] != audit["backbone_after_training_sha256"]:
        raise AssertionError("Frozen parameters changed within the trial")
    model_audit = meta["wrapper_audits"]["model"]
    names = expected_map("ATTN_ONLY") if trained else []
    if meta["module_map"] != names or model_audit["module_map"] != names:
        raise AssertionError("Unexpected adaptation module map")
    if model_audit["adaptive_count"] != count or not model_audit["residual_head_absent"]:
        raise AssertionError("Unexpected trainable count or extra head")
    sampler = np.random.default_rng(8100 + c).integers(64, size=(updates, 8)).astype(np.int64)
    if meta["sampler_sha256"] != hashlib.sha256(sampler.tobytes()).hexdigest():
        raise AssertionError("Sampler no longer follows the paired seed and update budget")
    if trained:
        expected_seeds = {name: int.from_bytes(hashlib.sha256(f"{8100+c}:{name}:rank=8".encode()).digest()[:8],
                                              "little") % (2**63 - 1) for name in names}
        for field in ("module_subset_verified", "native_pretrained_parameters_frozen",
                      "canonical_a_bitwise_verified", "zero_b_verified"):
            if model_audit.get(field) is not True:
                raise AssertionError(f"Attention initialization audit failed: {field}")
        if meta["module_initialization_seeds"] != expected_seeds or model_audit["module_seeds"] != expected_seeds:
            raise AssertionError("Attention initialization seeds differ from the canonical paired definition")
        expected_parameters = {f"base.{name}.lora_{letter}.default.weight" for name in names for letter in ("A", "B")}
        if set(meta["trainable_names"]) != expected_parameters or len(meta["trainable_names"]) != 192:
            raise AssertionError("Unexpected attention trainable parameter names")
        if meta["lora_rank"] != 8 or meta["lora_alpha"] != 16 or meta["cache_created_this_trial"]:
            raise AssertionError("Attention rank/scaling/cache path changed")
        if meta["wrapper_audits"]["before_restore_lora_sha256"] == meta["wrapper_audits"]["initial_lora_sha256"]:
            raise AssertionError("Attention parameters did not change during training")
    elif meta["trainable_names"] or meta["module_initialization_seeds"] or meta["best_step"] != 0:
        raise AssertionError("F0 has an adaptation or nonzero selected step")
    cache_path = Path(meta["cache"])
    cache_root = root / "runs" / STUDY / "cache" / f"Q00_c{c}" / mode
    if not cache_path.resolve().is_relative_to(cache_root.resolve()):
        raise AssertionError("FM trial reused an incompatible or old cache")
    cache = read_json(cache_path / "manifest.json")
    cache_contract = cache["contract"]
    if not cache["completed"]:
        raise AssertionError("Incomplete frozen-feature cache")
    for name in ("data_sha256", "lag_file_sha256", "lag_selection_input_sha256", "source_hashes", "plan_sha256",
                 "checkpoint_hashes", "input_scope"):
        if cache_contract[name] != wrapper[name]:
            raise AssertionError(f"Frozen cache contract mismatch: {name}")
    for name, expected in {"study": STUDY, "counts": data_api.N_BY_SPLIT, "micro_groups": 4,
                           "encoder_rows": 12, "target_rows": 4, "autocast": "bfloat16",
                           "autocast_weight_cache": False, "weight_dtype": "float32",
                           "tf32": False, "smoke": False}.items():
        if cache_contract[name] != expected:
            raise AssertionError(f"Cache batching/precision/data scope mismatch: {name}")
    expected_arrays = {f"{split}_{field}.npy" for split in ("train", "val", "eval")
                       for field in ("hidden", "norm", "loc", "scale")}
    if set(cache["array_sha256"]) != expected_arrays:
        raise AssertionError("Cache hash index does not contain all twelve frozen arrays")
    verify_file_hashes(cache_path, cache["array_sha256"], checked)
    verify_file_hashes(root, meta["wrapper_audits"]["cache_hashes"], checked)
    verify_file_hashes(Path(cache_contract["checkpoint"]), cache_contract["checkpoint_hashes"], checked)
    history = meta["validation_history"]
    expected_steps = list(range(0, 201, 40)) if trained else [0]
    if [row["step"] for row in history] != expected_steps:
        raise AssertionError("Missing or unexpected validation checkpoint")
    best = min(history, key=lambda row: (row["val_score"], row["step"]))
    if meta["best_step"] != best["step"]:
        raise AssertionError("Saved checkpoint is not the earliest validation argmin")
    compare_float(meta["val_score"], best["val_score"], "best validation", path)
    return meta, guard


def verify_paired_initialization(metadata):
    for c in range(3):
        reference = metadata[(c, "ATTN", RATES[0])]
        for method in TRAINED:
            for lr in RATES:
                actual = metadata[(c, method, lr)]
                f0_method = "ALIGN_F0" if method == "ALIGN_ATTN" else "F0"
                if actual["validation_history"][0]["val_score"] != metadata[(c, f0_method, .001)]["val_score"]:
                    raise AssertionError("Attention step0 validation differs from its matching frozen input/cache path")
                for name in ("sampler_sha256", "module_initialization_seeds", "trainable_names"):
                    if actual[name] != reference[name]:
                        raise AssertionError(f"Paired raw/aligned initialization differs: {name}")
                if actual["audits"]["initial_adaptation_sha256"] != reference["audits"]["initial_adaptation_sha256"]:
                    raise AssertionError("Raw/aligned initial adapter tensors differ")
                for field in ("module_a_initial_sha256", "module_b_initial_sha256"):
                    if actual["wrapper_audits"]["model"][field] != reference["wrapper_audits"]["model"][field]:
                        raise AssertionError(f"Per-module initial tensor hashes differ: {field}")


def raw_provenance(root, data_path, lag_path):
    return {"data_sha256": sha256_file(data_path), "lag_sha256": sha256_file(lag_path),
            "source_sha256": sha256_file(root / "experiments" / STUDY / "raw.py"),
            "data_source_sha256": sha256_file(root / "experiments" / STUDY / "data.py"),
            "plan_sha256": sha256_file(root / PLAN)}


def load_or_fit_raw(root, corpus, arrays, lag, data_path, lag_path):
    path = root / "runs" / STUDY / "raw" / f"Q00_c{corpus}"
    contract = raw_provenance(root, data_path, lag_path)
    provenance_path = path / "analysis_provenance.json"
    if (path / "result.json").exists():
        provenance = read_json(provenance_path)
        if provenance["inputs"] != contract:
            raise AssertionError("Preserved RAW artifact has different inputs or source")
        for name, expected in provenance["outputs"].items():
            if sha256_file(path / name) != expected:
                raise AssertionError("Preserved RAW artifact changed")
    else:
        if path.exists() and any(path.iterdir()):
            raise RuntimeError("Preserve incomplete RAW output; automatic overwrite is forbidden")
        raw_api.run(data_path, lag_path, path)
        write_json(provenance_path, {"inputs": contract, "outputs": {
            name: sha256_file(path / name) for name in ("result.json", "predictions.npz")}})
    meta = read_json(path / "result.json")
    if not meta["completed"] or meta["method"] != "RAW_TRAINLAG":
        raise AssertionError("RAW fit did not complete")
    if (meta["data_sha256"], meta["lag_selection_sha256"], meta["source_sha256"]) != (
            contract["data_sha256"], contract["lag_sha256"], contract["source_sha256"]):
        raise AssertionError("RAW saved source/data/lag hash mismatch")
    if meta["selected_lags"] != lag["selected_lags"] or meta["lag_input_array_hash"] != lag["input_array_hash"]:
        raise AssertionError("RAW did not use the common train-only lag estimate")
    for name in ("no_ridge", "no_hpo", "no_oof"):
        if meta[name] is not True:
            raise AssertionError("RAW fitting procedure changed")
    for name in ("known_lag_dictionary_used", "validation_target_used_for_fit", "eval_target_used_for_fit",
                 "oracle_or_manifest_used_for_fit", "context_y_used_as_extra_label"):
        if meta[name] is not False:
            raise AssertionError("RAW used unplanned information")
    if meta["training_future_labels"] != 1024 or meta["candidate_lags"] != list(range(16, 129)):
        raise AssertionError("RAW supervision or candidate scope changed")
    archive = verify_prediction(path, arrays, meta)
    features = {split: arrays[f"raw_features_{split}"].astype(np.float64) for split in ("train", "val", "eval")}
    reproduced = raw_api.fit_raw(features, arrays["target_train"], arrays["quantiles"].astype(np.float64))
    with np.load(path / "predictions.npz", allow_pickle=False) as saved:
        for split in ("train", "val", "eval"):
            if not np.array_equal(saved[f"{split}_predictions"], reproduced["predictions"][split].astype(np.float32)):
                raise AssertionError("RAW predictions do not reproduce the declared train-only OLS fit")
        for name, value in (("coefficient", reproduced["coefficient"]), ("intercept", reproduced["intercept"]),
                            ("residual_quantile_offsets", reproduced["offsets"])):
            if not np.allclose(saved[name], value, rtol=1e-12, atol=1e-12):
                raise AssertionError(f"RAW saved fitting parameter mismatch: {name}")
    for name in ("intercept", "rank", "train_residual_mean", "train_residual_var"):
        compare_float(meta[name], reproduced[name], f"RAW fit {name}", path)
    return path, meta, archive


def make_record(method, corpus, lr, mode, archive, arrays, meta, path, origin):
    return {"method": method, "corpus": corpus, "lr": lr, "input_mode": mode,
            "score": float(archive["eval_losses"].mean()), "val_score": float(archive["val_losses"].mean()),
            "relative_improvement_over_f0": None,
            **summarize_scores(archive["eval_predictions"], arrays["target_eval"], arrays["quantiles"], arrays["oracle_mean_eval"]),
            "best_step": meta.get("best_step"), "trainable": meta.get("trainable_parameters"),
            "wall": meta["wall_seconds"], "path": str(path), "origin": origin}


def load_completed_study(root):
    study = root / "runs" / STUDY
    completed = read_json(study / "completed.json")
    if any(completed.get(key) != value for key, value in
           {"completed": True, "fit_count": 12, "f0_count": 6, "trial_count": 18}.items()):
        raise AssertionError("Analysis requires all 18 GPU trials to complete first")
    contract = read_json(study / "study_contract.json")
    if contract != runner_contract(root) or read_json(study / "smoke_contract.json") != contract:
        raise AssertionError("Current source/plan/data/prior or S0 contract no longer matches")
    smoke = read_json(study / "smoke_completed.json")
    if not smoke["completed"] or not smoke["same_attention_initialization_and_sampler"]:
        raise AssertionError("S0 validation did not complete")
    verify_smoke(root, smoke["trials"])
    for entry in smoke["trials"]:
        smoke_path = root / entry["path"]
        smoke_lr = 1e-4 if entry["method"] in TRAINED else .001
        expected_path = study / "smoke/Q00_c0" / entry["method"] / f"lr_{smoke_lr:.0e}"
        if smoke_path.resolve() != expected_path.resolve() or entry["corpus"] != 0 or entry["lr"] != smoke_lr:
            raise AssertionError("S0 must use the declared four corpus0 smoke trials")
        if (smoke_path / "failure.json").exists() or (smoke_path / "guard/safety_stop.json").exists():
            raise AssertionError("S0 contains a failure artifact")
        meta = read_json(smoke_path / "result.json")
        if (not meta["smoke"] or meta["episode_counts"] != {"train": 8, "val": 8, "eval": 8}
                or meta["input_mode"] != entry["input_mode"]):
            raise AssertionError("S0 method/input/batching scope changed")
        if (meta["data_sha256"] != sha256_file(study / "data/Q00_c0.npz") or
            meta["lag_file_sha256"] != sha256_file(study / "data/Q00_c0_lags.json") or
            meta["wrapper_contract"]["plan_sha256"] != sha256_file(root / PLAN)):
            raise AssertionError("S0 is not linked to the current data, lag selection and plan")
        for name in ("zero_update_identity", "future_target_isolation", "masked_future_isolation",
                     "group_isolation", "original_input_unchanged", "original_past_indices_verified"):
            if meta["audits"].get(name) is not True:
                raise AssertionError(f"S0 information audit failed: {name}")
        gradients = meta["wrapper_audits"]["first_B_gradients"]
        if entry["method"] in TRAINED and (len(gradients) != 96 or max(gradients.values()) <= 0):
            raise AssertionError("S0 did not observe finite nonzero attention gradients")
    choices = read_json(study / "selection.json")
    trials, selected = read_json(study / "trials.json"), read_json(study / "selected.json")
    lookup = verify_entries(trials, selected, choices, root)
    actual = {path.resolve() for path in (study / "trials").glob("Q00_c*/*/lr_*") if path.is_dir()}
    if actual != {(root / entry["path"]).resolve() for entry in trials}:
        raise AssertionError("Unlisted, failed, or unexpected trial directory exists")
    return study, completed, contract, lookup, selected, choices


def write_csv(path, records):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(records)


def run(root):
    started = time.perf_counter()
    root = Path(root).resolve()
    study, completed, contract, entries, selected, choices = load_completed_study(root)
    loaded = [load_data(study, corpus) for corpus in range(3)]
    verify_shared_splits([item[0] for item in loaded])
    data_manifest = read_json(study / "data/manifest.json")
    if not data_manifest["all_qc_passed"]:
        raise AssertionError("Prepared-data QC failed")
    records, losses, metadata, guards, artifacts, checked = [], {}, {}, {}, {}, set()
    trial_paths = []
    for key, entry in sorted(entries.items()):
        c, method, lr = key
        arrays, lag, data_path, lag_path = loaded[c]
        if (data_manifest["files"][data_path.name]["sha256"] != sha256_file(data_path) or
            data_manifest["lag_selection_files"][lag_path.name]["sha256"] != sha256_file(lag_path)):
            raise AssertionError("Generation manifest data/lag hash mismatch")
        path = root / entry["path"]
        meta, guard = verify_fm_metadata(root, path, entry, arrays, lag, data_path, lag_path, checked)
        archive = verify_prediction(path, arrays, meta)
        records.append(make_record(method, c, lr, entry["input_mode"], archive, arrays, meta, entry["path"], "new_gpu"))
        losses[key], metadata[key], guards[key] = archive["eval_losses"], meta, guard
        artifacts[entry["path"]] = {name: sha256_file(path / name) for name in ("result.json", "predictions.npz", "best_adaptation.pt")}
        trial_paths.append(path)
    verify_paired_initialization(metadata)
    cpu_costs = []
    for c, (arrays, lag, data_path, lag_path) in enumerate(loaded):
        path, meta, archive = load_or_fit_raw(root, c, arrays, lag, data_path, lag_path)
        relative = path.relative_to(root).as_posix()
        records.append(make_record("RAW", c, None, "selected_lag_features", archive, arrays, meta, relative, "new_cpu"))
        losses[(c, "RAW", None)] = archive["eval_losses"]
        artifacts[relative] = {name: sha256_file(path / name) for name in ("result.json", "predictions.npz", "analysis_provenance.json")}
        cpu_costs.append({"corpus": c, "raw_fit_seconds": meta["fit_seconds"], "raw_wall_seconds": meta["wall_seconds"],
                          "lag_fit_seconds": lag["fit_seconds"], "path": relative})
        oracle = {f"{split}_predictions": arrays[f"oracle_quantiles_{split}"].astype(np.float64) for split in ("val", "eval")}
        for split in ("val", "eval"):
            oracle[f"{split}_losses"] = episode_loss(oracle[f"{split}_predictions"], arrays[f"target_{split}"], arrays["quantiles"])
        records.append(make_record("ORACLE", c, None, "true_conditional_distribution", oracle, arrays,
                                   {"wall_seconds": 0., "trainable_parameters": 0}, data_path.relative_to(root).as_posix(), "diagnostic"))
        losses[(c, "ORACLE", None)] = oracle["eval_losses"]
    chosen_keys = {trial_key(entry) for entry in selected} | {(c, method, None) for c in range(3) for method in ("RAW", "ORACLE")}
    chosen = [row for row in records if (row["corpus"], row["method"], row["lr"]) in chosen_keys]
    if len(records) != 24 or len(chosen) != 18 or len(chosen_keys) != 18:
        raise AssertionError("Incomplete all/selected reporting grid")
    for row in records:
        f0 = float(losses[(row["corpus"], "F0", .001)].mean())
        row["relative_improvement_over_f0"] = (f0 - row["score"]) / f0
    weights = bootstrap_weights(512)
    rates = {method: choices[method]["lr"] for method in TRAINED}
    primary = contrast_family(losses, rates, weights)
    fixed = contrast_family(losses, {method: 3e-5 for method in TRAINED}, weights)
    denominator = np.stack([losses[(c, "F0", .001)] for c in range(3)])
    gap = effect_ratio(np.stack([losses[(c, "RAW", None)] - losses[(c, "ORACLE", None)] for c in range(3)]),
                       denominator, weights, confidence=.95)
    gap.update({"definition": "(S_RAW - S_ORACLE) / S_F0", "practically_near_oracle": gap["ci"][1] < DELTA,
                "multiplicity": "Separate exploratory diagnostic; not part of the two-primary-contrast family"})
    guard_seconds = sum(guard["elapsed_seconds"] for guard in guards.values())
    compare_float(guard_seconds, completed["trial_guard_seconds"], "total GPU guard time", study)
    if contract != runner_contract(root):
        raise AssertionError("Protected original/new source, plan, or data changed during analysis")
    output = root / "results" / STUDY
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "all_results.csv", records)
    write_csv(output / "selected_results.csv", chosen)
    write_json(output / "effects.json", {
        "primary": primary, "fixed_lr_descriptive": fixed, "raw_oracle_diagnostic": gap,
        "choices": choices, "delta": DELTA,
        "selected_optimization_boundaries": [{"method": row["method"], "corpus": row["corpus"],
            "lr": row["lr"], "lr_at_lower_candidate": row["lr"] == RATES[0],
            "lr_at_upper_candidate": row["lr"] == RATES[-1], "best_step": row["best_step"],
            "best_at_zero": row["best_step"] == 0, "best_at_last_update": row["best_step"] == 200}
            for row in chosen if row["method"] in TRAINED],
        "bootstrap": {"replicates": 4000, "seed": BOOTSTRAP_SEED, "paired_episodes": 512,
                      "fitted_corpora": 3, "primary_family_size": 2, "primary_confidence": .975,
                      "raw_diagnostic_confidence": .95, "resample_training_or_validation": False},
        "interpretation_limits": [
            "Fresh random samples from the same stationary Gaussian family are not a new external data source.",
            "ALIGN_F0 fits lag selection on target training labels; only FM weights remain frozen.",
            "Retiming, missing padding, normalization, retained context and future-covariate access change jointly.",
            "Removing the small oracle lag dictionary does not remove stationary additive or one-lag-per-channel inductive biases.",
            "RAW uses optimistic in-sample residual quantiles, without OOF calibration; failure cannot establish absent information.",
            "Conditional intervals do not include full training-corpus or validation-selection uncertainty.",
        ]})
    gpu_costs = [{"path": entries[key]["path"], "method": key[1], "corpus": key[0], "lr": key[2],
                  "selected": key in chosen_keys, "model_wall_seconds": meta["wall_seconds"],
                  "wrapper_wall_seconds": meta["wrapper_wall_seconds"], "guard_seconds": guards[key]["elapsed_seconds"],
                  "peak_cuda_gib": meta["peak_cuda_gib"], "best_step": meta["best_step"]}
                 for key, meta in metadata.items()]
    write_json(output / "costs.json", {"new_gpu_trials": 18, "new_gpu_fits": 12, "new_f0_cache_trials": 6,
               "reused_gpu_trials": 0, "trial_guard_seconds": guard_seconds,
               "completed_invocation_wall_seconds": completed["invocation_wall_seconds"], "gpu_trials": gpu_costs,
               "cpu_raw_fits": cpu_costs, "cpu_raw_wall_seconds": sum(item["raw_wall_seconds"] for item in cpu_costs),
               "cpu_lag_fit_seconds": sum(item["lag_fit_seconds"] for item in cpu_costs),
               "analysis_wall_seconds": time.perf_counter() - started,
               "note": "All 12 LR fits and six frozen/cache trials are charged; CPU lag/RAW and analysis costs are separate."})
    resource = summarize_resources(resource_samples(trial_paths), "Exactly 18 new production GPU trial guards; S0/CPU and old runs excluded", "new")
    resource.update({"gpu_trial_count": 18, "reused_trial_count": 0,
                     "cpu_resource_scope": "CPU RAW fits run inside the separately reported root analysis guard"})
    write_json(output / "resource_summary.json", resource)
    verification = {"passed": True, "all_rows": 24, "selected_rows": 18, "gpu_trials": 18,
                    "raw_fits": 3, "oracle_rows": 3, "source_plan_data_prior_s0_contract_matched": True,
                    "exact_trial_and_selection_keys": True, "corpus0_validation_argmin_verified": True,
                    "lag_selector_train_hash_and_full_scores_reproduced": True,
                    "original_past_source_indices_verified": True, "paired_initialization_sampler_verified": True,
                    "own_adaptation_checkpoint_and_native_cache_hashes_verified": True,
                    "saved_prediction_scores_recomputed": True, "raw_train_only_ols_predictions_reproduced": True,
                    "shared_evaluation_arrays_verified": True, "all_gpu_guards_exit0": True,
                    "analysis_sha256": sha256_file(Path(__file__)), "artifacts": artifacts,
                    "contract_sha256": hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest(),
                    "outputs": {name: sha256_file(output / name) for name in
                                ("all_results.csv", "selected_results.csv", "effects.json", "costs.json", "resource_summary.json")}}
    write_json(output / "verification.json", verification)
    print(json.dumps({"passed": True, "all_rows": 24, "selected_rows": 18, "output": str(output)}), flush=True)
    return verification


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    run(Path(__file__).resolve().parents[2])


if __name__ == "__main__":
    main()
