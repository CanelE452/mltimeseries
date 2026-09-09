"""Analyse Q00 native LoRA module-deletion ablations after guarded completion."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "2"

import numpy as np

from .run_study import METHODS, PARENT, RATES, STUDY
from .run_study import contract as runner_contract
from .run_study import reference_paths


PRIMARY_LR = 3e-5
REFERENCE_LORA_METHOD = "OFF_LORA"
REFERENCE_LABEL = "BOTH"
OUTPUT_MODULE = "output_patch_embedding.output_layer"
BOOTSTRAP_REPLICATES = 4000
BOOTSTRAP_CONFIDENCE = 0.975
BOOTSTRAP_SEED = 2026090809
DELTA = 0.01

CSV_FIELDS = [
    "method",
    "corpus",
    "lr",
    "score",
    "median_mse",
    "oracle_mean_error_mse",
    "coverage80",
    "width80",
    "crossing",
    "exploratoryself/U/Vcoeffs",
    "best_step",
    "val_score",
    "wall",
    "trainable",
    "path",
    "origin",
]

EXPECTED_PREDICTION_KEYS = {
    "val_predictions",
    "eval_predictions",
    "val_target",
    "eval_target",
    "val_episode_losses",
    "eval_episode_losses",
    "val_episode_ids",
    "eval_episode_ids",
    "quantiles",
}

EXPECTED_RESULT_KEYS = {
    "completed",
    "method",
    "seed",
    "lr",
    "smoke",
    "metadata",
    "val_score",
    "best_step",
    "eval_score",
    "steps_completed",
    "trainable",
    "trainable_parameters",
    "trainable_names",
    "module_map",
    "module_initialization_seeds",
    "lora_rank",
    "lora_alpha",
    "head_lr",
    "head_only_updates",
    "phase_transitions",
    "final_optimizer_states",
    "wall_seconds",
    "peak_cuda_gib",
    "peak_cuda_reserved_gib",
    "optimizer_seconds_per_step",
    "optimizer_seconds_per_step_after_first",
    "optimizer_seconds_total",
    "sampler_sha256",
    "data_sha256",
    "source_sha256",
    "source_hashes",
    "frozen_s1_source_hashes",
    "cache",
    "cache_created_this_trial",
    "cache_generation_seconds",
    "checkpoint_sha256",
    "audits",
    "validation",
    "evaluation",
    "validation_history",
    "boundaries",
    "episode_counts",
    "context",
    "horizon",
    "channel_order",
    "target_channel",
    "micro_groups",
    "effective_groups",
    "autocast",
    "autocast_weight_cache",
    "weight_dtype",
    "dropout",
    "gradient_clip_norm",
    "weight_decay",
    "torch_threads",
    "interop_threads",
    "training_loss",
    "selection_metric",
    "scope",
    "packages",
}

EXPECTED_WRAPPER_KEYS = EXPECTED_RESULT_KEYS | {
    "wrapped_completed",
    "inner_training_completed",
    "wrapper_contract",
    "wrapper_audits",
    "wrapper_wall_seconds",
}


def expected_map(method):
    attention = [
        f"encoder.block.{block}.layer.{layer}.self_attention.{part}"
        for block in range(12)
        for layer in (0, 1)
        for part in ("q", "k", "v", "o")
    ]
    if method == "OUT_ONLY":
        return [OUTPUT_MODULE]
    if method == "ATTN_ONLY":
        return attention
    if method == REFERENCE_LORA_METHOD:
        return [*attention, OUTPUT_MODULE]
    raise ValueError(method)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def episode_loss(prediction, target, quantiles):
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    quantiles = np.asarray(quantiles, dtype=np.float64)
    error = target[:, None, :] - prediction
    q = quantiles[None, :, None]
    return (2 * np.maximum(q * error, (q - 1) * error)).mean(axis=(1, 2))


def bootstrap_weights(episode_count, replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED):
    rng = np.random.default_rng(seed)
    return rng.multinomial(
        episode_count,
        np.ones(episode_count, dtype=np.float64) / episode_count,
        size=replicates,
    ).astype(np.float64) / episode_count


def effect_ratio(numerator, denominator, weights, confidence=BOOTSTRAP_CONFIDENCE):
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if numerator.shape != denominator.shape or numerator.ndim != 2:
        raise ValueError("numerator and denominator must both be [corpus, episode]")
    if weights.shape[1] != numerator.shape[1]:
        raise ValueError("bootstrap weights must share the episode axis")
    curve = numerator.mean(axis=0)
    denom = denominator.mean(axis=0)
    if np.any(denom <= 0):
        raise ValueError("denominator episode losses must be positive")
    draws = (weights @ curve) / (weights @ denom)
    tail = (1 - confidence) / 2
    low, high = np.quantile(draws, [tail, 1 - tail])
    corpus_values = numerator.mean(axis=1) / denominator.mean(axis=1)
    return {
        "value": float(curve.mean() / denom.mean()),
        "corpus_values": corpus_values.tolist(),
        "ci": [float(low), float(high)],
        "confidence": confidence,
        "scope": "paired evaluation episodes conditional on three fitted corpus/optimizer repetitions",
    }


def classify_effect(effect, delta=DELTA):
    low, high = effect["ci"]
    corpus_values = np.asarray(effect["corpus_values"], dtype=np.float64)
    if low > delta and np.all(corpus_values > 0):
        return "repeated_practical_harm"
    if high < -delta and np.all(corpus_values < 0):
        return "practical_improvement"
    if low >= -delta and high <= delta:
        return "conditional_practical_equivalence"
    return "inconclusive"


def _metadata_item(value):
    item = value.item()
    if isinstance(item, bytes):
        item = item.decode("utf-8")
    return json.loads(item)


def load_data_arrays(study, corpus):
    path = study / "data" / f"Q00_c{corpus}.npz"
    with np.load(path, allow_pickle=False) as data:
        context_eval = data["context_eval"].astype(np.float64)
        quantiles = data["quantiles"].astype(np.float64)
        result = {
            "path": path,
            "quantiles": quantiles,
            "target_val": data["target_val"].astype(np.float64),
            "target_eval": data["target_eval"].astype(np.float64),
            "oracle_mean_eval": data["oracle_mean_eval"].astype(np.float64),
            "episode_ids_val": np.asarray(data["episode_ids_val"]),
            "episode_ids_eval": np.asarray(data["episode_ids_eval"]),
            "manifest": _metadata_item(data["manifest_json"]),
        }
    generator = result["manifest"]["generator"]
    context = int(generator["context_length"])
    horizon = int(generator["horizon"])
    leads = np.arange(horizon)
    self_lag = int(generator["self_lag"])
    driver_lag = int(generator.get("driver_lag", 48))
    theta = float(generator["theta_radians"])
    components = np.stack(
        (
            0.5 * context_eval[:, 0, context + leads - self_lag],
            np.sqrt(0.39) * np.cos(theta) * context_eval[:, 1, context + leads - driver_lag],
            np.sqrt(0.39) * np.sin(theta) * context_eval[:, 2, context + leads - driver_lag],
        ),
        axis=-1,
    )
    if not np.allclose(components.sum(axis=-1), result["oracle_mean_eval"], rtol=1e-5, atol=5e-7):
        raise AssertionError(f"Oracle component reconstruction mismatch for corpus {corpus}")
    result["component_design"] = np.column_stack(
        (np.ones(result["target_eval"].size), components.reshape(-1, 3))
    )
    return result


def validate_guard(path):
    guard_path = path / "guard" / "status.json"
    guard = read_json(guard_path)
    if not guard.get("completed") or guard.get("returncode") != 0 or guard.get("reasons"):
        raise AssertionError(f"Guard failed or stopped: {guard_path}")
    return guard


def validate_prediction_archive(path, data):
    with np.load(path / "predictions.npz", allow_pickle=False) as archive:
        keys = set(archive.files)
        if not EXPECTED_PREDICTION_KEYS <= keys:
            missing = sorted(EXPECTED_PREDICTION_KEYS - keys)
            raise AssertionError(f"Prediction archive missing keys at {path}: {missing}")
        val_prediction = archive["val_predictions"].astype(np.float64)
        eval_prediction = archive["eval_predictions"].astype(np.float64)
        if not np.array_equal(archive["val_target"], data["target_val"]):
            raise AssertionError(f"Validation target mismatch at {path}")
        if not np.array_equal(archive["eval_target"], data["target_eval"]):
            raise AssertionError(f"Evaluation target mismatch at {path}")
        if not np.array_equal(archive["val_episode_ids"], data["episode_ids_val"]):
            raise AssertionError(f"Validation episode-id mismatch at {path}")
        if not np.array_equal(archive["eval_episode_ids"], data["episode_ids_eval"]):
            raise AssertionError(f"Evaluation episode-id mismatch at {path}")
        if not np.array_equal(archive["quantiles"].astype(np.float64), data["quantiles"]):
            raise AssertionError(f"Quantile grid mismatch at {path}")
        val_losses = archive["val_episode_losses"].astype(np.float64)
        eval_losses = archive["eval_episode_losses"].astype(np.float64)
    if eval_prediction.shape != (len(data["episode_ids_eval"]), len(data["quantiles"]), data["target_eval"].shape[1]):
        raise AssertionError(f"Evaluation prediction shape mismatch at {path}: {eval_prediction.shape}")
    if val_prediction.shape != (len(data["episode_ids_val"]), len(data["quantiles"]), data["target_val"].shape[1]):
        raise AssertionError(f"Validation prediction shape mismatch at {path}: {val_prediction.shape}")
    recomputed_val = episode_loss(val_prediction, data["target_val"], data["quantiles"])
    recomputed_eval = episode_loss(eval_prediction, data["target_eval"], data["quantiles"])
    if not np.allclose(recomputed_val, val_losses, rtol=1e-6, atol=1e-7):
        raise AssertionError(f"Saved validation episode losses mismatch at {path}")
    if not np.allclose(recomputed_eval, eval_losses, rtol=1e-6, atol=1e-7):
        raise AssertionError(f"Saved evaluation episode losses mismatch at {path}")
    return {
        "keys": sorted(keys),
        "val_predictions": val_prediction,
        "eval_predictions": eval_prediction,
        "val_losses": recomputed_val,
        "eval_losses": recomputed_eval,
    }


def compare_float(actual, expected, label, path, atol=1e-7):
    if not np.isclose(float(actual), float(expected), rtol=1e-6, atol=atol):
        raise AssertionError(f"{label} mismatch at {path}: {actual} != {expected}")


def validate_adaptation_checkpoint_file(path, meta):
    checkpoint = path / "best_adaptation.pt"
    if not checkpoint.exists():
        raise AssertionError(f"Missing saved adaptation checkpoint at {path}")
    if sha256_file(checkpoint) != meta["checkpoint_sha256"]:
        raise AssertionError(f"Saved adaptation checkpoint hash mismatch at {path}")


def validate_original_pretrained_contract(meta, root, path):
    audits = meta["audits"]
    if not audits.get("frozen_parameters_verified") or not audits.get("checkpoint_reload_verified"):
        raise AssertionError(f"Frozen-parameter or checkpoint-reload audit failed at {path}")
    cache = Path(meta["cache"])
    manifest = read_json(cache / "manifest.json")
    contract = manifest["contract"]
    if not manifest.get("completed") or contract["data_sha256"] != meta["data_sha256"]:
        raise AssertionError(f"Original cache contract mismatch at {path}")
    shared_source = root / "experiments" / PARENT / "train.py"
    if contract["source_sha256"] != sha256_file(shared_source):
        raise AssertionError(f"Original cache source hash mismatch at {path}")
    checkpoint = Path(contract["checkpoint"])
    if not checkpoint.is_absolute():
        checkpoint = root / checkpoint
    if sha256_file(checkpoint / "config.json") != contract["config_sha256"]:
        raise AssertionError(f"Original checkpoint config hash mismatch at {path}")
    for filename, expected in manifest["checkpoint_weights"].items():
        if sha256_file(checkpoint / filename) != expected:
            raise AssertionError(f"Original checkpoint weight hash mismatch at {path}: {filename}")


def validate_result_json(path, expected_method, expected_corpus, expected_lr, data_sha256, require_wrapper):
    meta = read_json(path / "result.json")
    keys = set(meta)
    expected_keys = EXPECTED_WRAPPER_KEYS if require_wrapper else EXPECTED_RESULT_KEYS
    if keys != expected_keys:
        raise AssertionError(
            f"Result keyset mismatch at {path}: missing={sorted(expected_keys - keys)} "
            f"extra={sorted(keys - expected_keys)}"
        )
    if not meta["completed"]:
        raise AssertionError(f"Incomplete result at {path}")
    if require_wrapper and (not meta["wrapped_completed"] or not meta["inner_training_completed"]):
        raise AssertionError(f"Wrapper did not complete at {path}")
    if meta["method"] != expected_method or meta["metadata"]["corpus"] != expected_corpus:
        raise AssertionError(f"Method/corpus mismatch at {path}")
    if meta["seed"] != 7100 + expected_corpus or not np.isclose(meta["lr"], expected_lr, rtol=0, atol=1e-12):
        raise AssertionError(f"Seed/LR mismatch at {path}")
    if meta["smoke"]:
        raise AssertionError(f"Main analysis received a smoke result at {path}")
    if meta["data_sha256"] != data_sha256:
        raise AssertionError(f"Data hash mismatch at {path}")
    validate_adaptation_checkpoint_file(path, meta)
    return meta


def validate_reference(path, label, corpus, lr, data_sha256, root):
    expected_method = "F0" if label == "F0" else REFERENCE_LORA_METHOD
    guard = validate_guard(path)
    meta = validate_result_json(path, expected_method, corpus, lr, data_sha256, require_wrapper=False)
    validate_original_pretrained_contract(meta, root, path)
    return meta, guard


def validate_new_ablation(path, method, corpus, lr, data_sha256, root, both_meta):
    guard = validate_guard(path)
    meta = validate_result_json(path, method, corpus, lr, data_sha256, require_wrapper=True)
    validate_original_pretrained_contract(meta, root, path)
    if meta["cache_created_this_trial"]:
        raise AssertionError(f"Production trial created cache at {path}")
    if meta["steps_completed"] != 200:
        raise AssertionError(f"Expected 200 updates at {path}")
    if meta["head_lr"] is not None or meta["head_only_updates"] != 0 or meta["phase_transitions"]:
        raise AssertionError(f"Residual-head or phased training path used at {path}")
    if meta["sampler_sha256"] != both_meta["sampler_sha256"]:
        raise AssertionError(f"Sampler differs from paired BOTH trial at {path}")
    if meta["module_map"] != expected_map(method):
        raise AssertionError(f"Module map differs from planned {method} subset at {path}")
    if method == "OUT_ONLY" and len(meta["module_map"]) != 1:
        raise AssertionError(f"OUT_ONLY must retain exactly one module at {path}")
    if method == "ATTN_ONLY" and (len(meta["module_map"]) != 96 or OUTPUT_MODULE in meta["module_map"]):
        raise AssertionError(f"ATTN_ONLY must retain 96 attention modules and no output projection at {path}")
    wrapper = meta["wrapper_contract"]
    if wrapper["module_map"] != meta["module_map"] or wrapper["trainable_parameters"] != meta["trainable_parameters"]:
        raise AssertionError(f"Wrapper contract does not match saved result at {path}")
    if wrapper["common_training"]["steps"] != 200 or wrapper["common_training"]["new_training_loop"]:
        raise AssertionError(f"Wrapper training contract changed at {path}")
    audit = meta["wrapper_audits"]
    if audit["module_map"] != meta["module_map"] or not audit["residual_head_absent"]:
        raise AssertionError(f"Wrapper model audit failed at {path}")
    retained = set(meta["module_map"])
    both_seeds = both_meta["module_initialization_seeds"]
    new_seeds = meta["module_initialization_seeds"]
    if set(new_seeds) != retained:
        raise AssertionError(f"Unexpected retained seed keys at {path}")
    for name in retained:
        if new_seeds[name] != both_seeds[name]:
            raise AssertionError(f"Retained module seed differs from BOTH at {path}: {name}")
    return meta, guard


def summarize_scores(prediction, target, quantiles, oracle_mean, component_design):
    med = int(np.argmin(abs(quantiles - 0.5)))
    lower = int(np.argmin(abs(quantiles - 0.1)))
    upper = int(np.argmin(abs(quantiles - 0.9)))
    median = prediction[:, med]
    coeffs = np.linalg.lstsq(component_design, median.reshape(-1), rcond=None)[0]
    return {
        "median_mse": float(np.mean((target - median) ** 2)),
        "oracle_mean_error_mse": float(np.mean((oracle_mean - median) ** 2)),
        "coverage80": float(np.mean((target >= prediction[:, lower]) & (target <= prediction[:, upper]))),
        "width80": float(np.mean(prediction[:, upper] - prediction[:, lower])),
        "crossing": float(np.mean(np.diff(prediction, axis=1) < 0)),
        "exploratoryself/U/Vcoeffs": (
            f"self={coeffs[1]:.8g};U={coeffs[2]:.8g};V={coeffs[3]:.8g}"
        ),
    }


def make_record(label, corpus, lr, score, summary, meta, path, origin):
    return {
        "method": label,
        "corpus": corpus,
        "lr": lr,
        "score": score,
        "median_mse": summary["median_mse"],
        "oracle_mean_error_mse": summary["oracle_mean_error_mse"],
        "coverage80": summary["coverage80"],
        "width80": summary["width80"],
        "crossing": summary["crossing"],
        "exploratoryself/U/Vcoeffs": summary["exploratoryself/U/Vcoeffs"],
        "best_step": meta["best_step"],
        "val_score": meta["val_score"],
        "wall": meta["wall_seconds"],
        "trainable": meta["trainable_parameters"],
        "path": str(path),
        "origin": origin,
    }


def load_trial(path, label, expected_method, corpus, lr, data, data_sha256, origin, root, both_meta=None):
    if origin == "new":
        meta, guard = validate_new_ablation(path, expected_method, corpus, lr, data_sha256, root, both_meta)
    else:
        meta, guard = validate_reference(path, label, corpus, lr, data_sha256, root)
    archive = validate_prediction_archive(path, data)
    compare_float(archive["val_losses"].mean(), meta["val_score"], "validation score", path)
    compare_float(archive["eval_losses"].mean(), meta["eval_score"], "evaluation score", path)
    summary = summarize_scores(
        archive["eval_predictions"],
        data["target_eval"],
        data["quantiles"],
        data["oracle_mean_eval"],
        data["component_design"],
    )
    row = make_record(
        label,
        corpus,
        lr,
        float(archive["eval_losses"].mean()),
        summary,
        meta,
        path,
        origin,
    )
    return row, archive["eval_losses"], meta, guard, archive["keys"]


def expected_new_trial_keys():
    return {(corpus, method, lr) for corpus in range(3) for method in METHODS for lr in RATES}


def validate_smoke_reproduction(root, study):
    smoke = read_json(study / "smoke_completed.json")
    if not smoke.get("completed") or not smoke.get("both_reproduces_parent_s0"):
        raise AssertionError("S0 wrapper control did not verify BOTH reproduction")
    if {entry["method"] for entry in smoke["trials"]} != {*METHODS, REFERENCE_LORA_METHOD}:
        raise AssertionError("S0 wrapper control has the wrong method set")
    if len(smoke["trials"]) != 3:
        raise AssertionError("S0 wrapper control must contain exactly three trials")
    for entry in smoke["trials"]:
        guard = validate_guard(root / entry["path"])
        if guard.get("returncode") != 0:
            raise AssertionError(f"S0 guard failed for {entry['path']}")
    return smoke


def load_completed_study(root):
    study = root / "runs" / STUDY
    parent = root / "runs" / PARENT
    validate_smoke_reproduction(root, study)
    completed = read_json(study / "completed.json")
    if not completed.get("completed") or completed.get("fit_count") != 12:
        raise AssertionError("Expected the module-ablation study to have exactly 12 completed fits")
    saved_contract = read_json(study / "study_contract.json")
    current_contract = runner_contract(root)
    if saved_contract != current_contract:
        raise AssertionError("Current runner source/reference contract differs from the saved study contract")
    trials = read_json(study / "trials.json")
    trial_lookup = {(int(e["corpus"]), e["method"], float(e["lr"])): e for e in trials}
    if set(trial_lookup) != expected_new_trial_keys() or len(trials) != 12:
        raise AssertionError("Missing, duplicate or unexpected module-ablation trial key")
    selection = read_json(study / "selection.json")
    if set(selection) != set(METHODS):
        raise AssertionError("Selection JSON must contain exactly OUT_ONLY and ATTN_ONLY")
    for method in METHODS:
        candidates = [trial_lookup[(0, method, lr)] for lr in RATES]
        expected_lr = min(candidates, key=lambda item: item["val_score"])["lr"]
        selected = selection[method]
        if set(selected) != {"lr", "candidates"} or len(selected["candidates"]) != len(RATES):
            raise AssertionError(f"Selection entry has the wrong shape for {method}")
        if not np.isclose(selected["lr"], expected_lr, rtol=0, atol=1e-12):
            raise AssertionError(f"Selection LR is not the corpus0 validation argmin for {method}")
        candidate_keys = set()
        for candidate in selected["candidates"]:
            key = (int(candidate["corpus"]), candidate["method"], float(candidate["lr"]))
            candidate_keys.add(key)
            if key not in trial_lookup or candidate != trial_lookup[key]:
                raise AssertionError(f"Selection candidate is not an exact trials.json copy for {method}: {key}")
        if candidate_keys != {(0, method, lr) for lr in RATES}:
            raise AssertionError(f"Selection candidates have wrong keys for {method}")
    return study, parent, completed, saved_contract, trial_lookup, selection


def score_relative_to_f0(records, f0_scores):
    items = []
    for row in records:
        f0 = f0_scores[int(row["corpus"])]
        items.append({
            "method": row["method"],
            "corpus": int(row["corpus"]),
            "lr": float(row["lr"]),
            "score": float(row["score"]),
            "f0_score": float(f0),
            "relative_to_f0": float((row["score"] - f0) / f0),
            "origin": row["origin"],
        })
    return items


def contrast_family(losses, lr_by_method, weights):
    f0 = np.stack([losses[("F0", corpus, 0.001)] for corpus in range(3)])
    both = np.stack([losses[(REFERENCE_LABEL, corpus, PRIMARY_LR)] for corpus in range(3)])
    out = np.stack([losses[("OUT_ONLY", corpus, lr_by_method["OUT_ONLY"])] for corpus in range(3)])
    attn = np.stack([losses[("ATTN_ONLY", corpus, lr_by_method["ATTN_ONLY"])] for corpus in range(3)])
    attention_deleted = effect_ratio(out - both, f0, weights, BOOTSTRAP_CONFIDENCE)
    output_deleted = effect_ratio(attn - both, f0, weights, BOOTSTRAP_CONFIDENCE)
    attention_deleted["decision"] = classify_effect(attention_deleted)
    output_deleted["decision"] = classify_effect(output_deleted)
    return {
        "attention_deletion_penalty": attention_deleted,
        "output_deletion_penalty": output_deleted,
        "lr_by_method": {key: float(value) for key, value in lr_by_method.items()},
        "definition": {
            "attention_deletion_penalty": "(S_OUT_ONLY - S_BOTH) / S_F0",
            "output_deletion_penalty": "(S_ATTN_ONLY - S_BOTH) / S_F0",
            "positive_value": "the ablated method is worse than BOTH relative to F0",
        },
    }


def resource_samples(paths):
    required = {
        "timestamp",
        "available_ram_gib",
        "available_commit_gib",
        "child_tree_rss_gib",
        "git_process_count",
    }
    samples = []
    for path in paths:
        log = path / "guard" / "resource_log.jsonl"
        if not log.exists():
            continue
        for line in log.read_text(encoding="utf-8").splitlines():
            if line.strip():
                sample = json.loads(line)
                if required <= set(sample):
                    samples.append(sample)
    return samples


def summarize_resources(samples, scope, origin):
    if not samples:
        return {
            "scope": scope,
            "origin": origin,
            "sample_count": 0,
            "first_sample_utc": None,
            "last_sample_utc": None,
            "min_available_ram_gib": None,
            "min_available_commit_gib": None,
            "max_child_tree_rss_gib": None,
            "max_git_process_count": None,
            "max_gpu_memory_used_mib": None,
            "max_gpu_temperature_c": None,
            "guard_failures": 0,
            "safety_stops": 0,
        }
    gpu_samples = [gpu for sample in samples for gpu in sample.get("gpus", [])]
    return {
        "scope": scope,
        "origin": origin,
        "sample_count": len(samples),
        "first_sample_utc": min(sample["timestamp"] for sample in samples),
        "last_sample_utc": max(sample["timestamp"] for sample in samples),
        "min_available_ram_gib": min(sample["available_ram_gib"] for sample in samples),
        "min_available_commit_gib": min(sample["available_commit_gib"] for sample in samples),
        "max_child_tree_rss_gib": max(sample["child_tree_rss_gib"] for sample in samples),
        "max_git_process_count": max(sample["git_process_count"] for sample in samples),
        "max_gpu_memory_used_mib": max((gpu["memory_used_mib"] for gpu in gpu_samples), default=None),
        "max_gpu_temperature_c": max((gpu["temperature_c"] for gpu in gpu_samples), default=None),
        "guard_failures": 0,
        "safety_stops": 0,
    }


def resource_summary_by_origin(paths_by_origin):
    return {
        "scope": (
            "Resource summaries are separated by artifact origin. Use 'new' for the 12 module-ablation "
            "fits; 'reused' describes original F0/BOTH reference guards and must not be reported as new-run safety."
        ),
        "new": summarize_resources(
            resource_samples(paths_by_origin["new"]),
            "12 new module-ablation production trial guards only",
            "new",
        ),
        "reused": summarize_resources(
            resource_samples(paths_by_origin["reused"]),
            "read-reused original Q00 F0/BOTH reference guards only",
            "reused",
        ),
    }


def run(root):
    root = Path(root)
    study, parent, completed, saved_contract, trial_lookup, selection = load_completed_study(root)
    output = root / "results" / STUDY
    output.mkdir(parents=True, exist_ok=True)
    refs = reference_paths(root)

    records = []
    losses = {}
    result_meta = {}
    guards = {}
    prediction_keysets = {}
    paths_by_origin = {"new": [], "reused": []}
    data_hashes = {}

    for corpus in range(3):
        data = load_data_arrays(parent, corpus)
        data_sha = saved_contract["references"][
            f"runs/{PARENT}/data/Q00_c{corpus}.npz"
        ]
        if sha256_file(data["path"]) != data_sha:
            raise AssertionError(f"Q00_c{corpus} data hash mismatch")
        data_hashes[f"Q00_c{corpus}.npz"] = data_sha

        f0_path = refs[(corpus, "F0")]
        f0_row, f0_loss, f0_meta, f0_guard, f0_keys = load_trial(
            f0_path,
            "F0",
            "F0",
            corpus,
            0.001,
            data,
            data_sha,
            "reused",
            root,
        )
        records.append(f0_row)
        losses[("F0", corpus, 0.001)] = f0_loss
        result_meta[("F0", corpus, 0.001)] = f0_meta
        guards[("F0", corpus, 0.001)] = f0_guard
        prediction_keysets[("F0", corpus, 0.001)] = f0_keys
        paths_by_origin["reused"].append(f0_path)

        both_path = refs[(corpus, REFERENCE_LORA_METHOD)]
        both_row, both_loss, both_meta, both_guard, both_keys = load_trial(
            both_path,
            REFERENCE_LABEL,
            REFERENCE_LORA_METHOD,
            corpus,
            PRIMARY_LR,
            data,
            data_sha,
            "reused",
            root,
        )
        records.append(both_row)
        losses[(REFERENCE_LABEL, corpus, PRIMARY_LR)] = both_loss
        result_meta[(REFERENCE_LABEL, corpus, PRIMARY_LR)] = both_meta
        guards[(REFERENCE_LABEL, corpus, PRIMARY_LR)] = both_guard
        prediction_keysets[(REFERENCE_LABEL, corpus, PRIMARY_LR)] = both_keys
        paths_by_origin["reused"].append(both_path)

        for method in METHODS:
            for lr in RATES:
                entry = trial_lookup[(corpus, method, lr)]
                canonical = study / "trials" / f"Q00_c{corpus}" / method / f"lr_{lr:.0e}"
                if (root / entry["path"]).resolve() != canonical.resolve():
                    raise AssertionError(f"Noncanonical trial path for {method}/c{corpus}/lr{lr}")
                row, loss, meta, guard, keys = load_trial(
                    canonical,
                    method,
                    method,
                    corpus,
                    lr,
                    data,
                    data_sha,
                    "new",
                    root,
                    both_meta=both_meta,
                )
                if not np.isclose(entry["val_score"], meta["val_score"], rtol=0, atol=1e-12):
                    raise AssertionError(f"trials.json val_score differs at {canonical}")
                records.append(row)
                losses[(method, corpus, lr)] = loss
                result_meta[(method, corpus, lr)] = meta
                guards[(method, corpus, lr)] = guard
                prediction_keysets[(method, corpus, lr)] = keys
                paths_by_origin["new"].append(canonical)

    if len(records) != 18 or sum(row["origin"] == "new" for row in records) != 12:
        raise AssertionError("Expected exactly 18 CSV rows: 12 new + 6 reused")
    if {row["method"] for row in records} != {"F0", REFERENCE_LABEL, *METHODS}:
        raise AssertionError("Unexpected method labels in CSV records")

    weights = bootstrap_weights(512, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED)
    primary = contrast_family(
        losses,
        {"OUT_ONLY": PRIMARY_LR, "ATTN_ONLY": PRIMARY_LR},
        weights,
    )
    selected_lrs = {method: float(selection[method]["lr"]) for method in METHODS}
    secondary = contrast_family(losses, selected_lrs, weights)
    f0_scores = {
        corpus: losses[("F0", corpus, 0.001)].mean()
        for corpus in range(3)
    }
    descriptive = score_relative_to_f0(records, f0_scores)

    for row in records:
        row["path"] = str(Path(row["path"]).relative_to(root)).replace("\\", "/")

    csv_path = output / "selected_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(records)

    costs = []
    for row in records:
        key = (
            row["method"],
            int(row["corpus"]),
            float(row["lr"]),
        )
        if row["method"] == REFERENCE_LABEL:
            key = (REFERENCE_LABEL, int(row["corpus"]), PRIMARY_LR)
        meta = result_meta[key]
        guard = guards[key]
        costs.append({
            "path": row["path"],
            "method": row["method"],
            "corpus": int(row["corpus"]),
            "lr": float(row["lr"]),
            "origin": row["origin"],
            "model_wall_seconds": float(meta["wall_seconds"]),
            "guard_elapsed_seconds": float(guard["elapsed_seconds"]),
            "peak_cuda_gib": meta.get("peak_cuda_gib"),
            "best_step": meta["best_step"],
        })
    new_cost = [item for item in costs if item["origin"] == "new"]
    reused_cost = [item for item in costs if item["origin"] == "reused"]
    write_json(output / "costs.json", {
        "new_trial_count": len(new_cost),
        "reused_reference_count": len(reused_cost),
        "new_guard_total_seconds": float(sum(item["guard_elapsed_seconds"] for item in new_cost)),
        "reused_reference_guard_total_seconds": float(sum(item["guard_elapsed_seconds"] for item in reused_cost)),
        "new_model_wall_seconds": float(sum(item["model_wall_seconds"] for item in new_cost)),
        "reused_reference_model_wall_seconds": float(sum(item["model_wall_seconds"] for item in reused_cost)),
        "trial_costs": costs,
        "completed_invocation_wall_seconds": completed.get("invocation_wall_seconds"),
        "completed_trial_guard_seconds": completed.get("trial_guard_seconds"),
        "cost_note": "Original F0/BOTH references are read-reused; they are reported separately from the 12 new fits.",
    })
    write_json(output / "resource_summary.json", resource_summary_by_origin(paths_by_origin))
    write_json(output / "effects.json", {
        "primary": primary,
        "secondary": secondary,
        "choices": {
            "primary_fixed_lr": {method: PRIMARY_LR for method in METHODS},
            "secondary_corpus0_validation": selection,
        },
        "descriptive_f0_relative_scores": descriptive,
        "bootstrap": {
            "replicates": BOOTSTRAP_REPLICATES,
            "confidence_per_contrast": BOOTSTRAP_CONFIDENCE,
            "family_size": 2,
            "seed": BOOTSTRAP_SEED,
            "paired_episode_count": 512,
        },
        "delta": DELTA,
        "interpretation_limits": [
            "The intervals are conditional on the three fitted corpus/optimizer repetitions.",
            "The exploratory self/U/V coefficients are OLS projections of median forecasts, not module mechanisms.",
            "This Q00 module-deletion result does not establish an equal-parameter model ranking.",
        ],
    })
    verification = {
        "passed": True,
        "csv_rows": len(records),
        "new_trial_count": sum(row["origin"] == "new" for row in records),
        "reused_reference_count": sum(row["origin"] == "reused" for row in records),
        "source_reference_contract_matched": True,
        "saved_contract_sha256": hashlib.sha256(
            json.dumps(saved_contract, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "analysis_sha256": sha256_file(Path(__file__)),
        "csv_sha256": sha256_file(csv_path),
        "exact_new_trial_keys_verified": True,
        "selection_corpus0_validation_argmin_verified": True,
        "baseline_and_new_guards_exit0": True,
        "s0_both_reproduction_verified": True,
        "wrapped_completed_new_trials": True,
        "prediction_targets_grid_episode_ids_matched": True,
        "raw_mean_2pinball_recomputed": True,
        "data_hashes": data_hashes,
        "prediction_keysets": {
            f"{method}/c{corpus}/lr{lr:.0e}": keys
            for (method, corpus, lr), keys in sorted(
                prediction_keysets.items(), key=lambda item: (item[0][1], item[0][0], item[0][2])
            )
        },
        "module_checks": {
            "out_only_modules": len(expected_map("OUT_ONLY")),
            "attn_only_modules": len(expected_map("ATTN_ONLY")),
            "attention_and_output_maps_disjoint": not bool(
                set(expected_map("OUT_ONLY")) & set(expected_map("ATTN_ONLY"))
            ),
            "union_is_original_both_map": set(expected_map("OUT_ONLY")) | set(expected_map("ATTN_ONLY"))
            == set(expected_map(REFERENCE_LORA_METHOD)),
            "retained_initialization_seeds_matched_both": True,
            "steps200_no_head_path_sampler_and_checkpoint_verified": True,
            "trial_checkpoint_hashes_verified_against_own_best_adaptation_file": True,
            "shared_pretrained_contract_verified_by_original_checkpoint_cache_source_hashes": True,
        },
        "output_files": {
            "selected_results_csv": str(csv_path.relative_to(root)).replace("\\", "/"),
            "effects_json": f"results/{STUDY}/effects.json",
            "costs_json": f"results/{STUDY}/costs.json",
            "resource_summary_json": f"results/{STUDY}/resource_summary.json",
            "verification_json": f"results/{STUDY}/verification.json",
        },
    }
    write_json(output / "verification.json", verification)
    print(json.dumps({
        "passed": True,
        "rows": len(records),
        "new_trials": 12,
        "csv": str(csv_path.relative_to(root)).replace("\\", "/"),
    }, allow_nan=False))
    return verification


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    run(Path(__file__).resolve().parents[2])


if __name__ == "__main__":
    main()
