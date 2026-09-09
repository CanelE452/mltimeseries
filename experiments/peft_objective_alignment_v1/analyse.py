"""Verify the bounded objective diagnostic before scoring its saved forecasts."""

import argparse
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import time

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "2"

import numpy as np

from experiments.peft_external_gap_v1 import analyse as numerical


STUDY = "peft_objective_alignment_v1"
DATASETS = ("bike", "household")
ARMS = ("NATIVE", "NORM_ALIGNED", "RAW_ALIGNED")
PROCEDURES = ("SORT", "QCAL")
BOOTSTRAP_SEED = 2026090815
REPLICATES = 4000
CONFIDENCE = .975
DELTA = .01
read_json, write_json, sha = numerical.read_json, numerical.write_json, numerical.sha256_file


def paired_effect(left, right, f0, counts, block_days=7, stride_days=1):
    left, right, f0, counts = [np.asarray(x, dtype=np.float64) for x in (left, right, f0, counts)]
    if (left.ndim != 2 or left.shape != right.shape or left.shape != f0.shape or left.shape != counts.shape
            or left.shape[1] not in (1, 2) or stride_days not in (1, 2)
            or not all(np.isfinite(x).all() for x in (left, right, f0, counts)) or np.any(counts < 0)):
        raise ValueError("Pair the same observed origin/target cells for one fixed optimizer seed")
    denominator = float(numerical.macro_score(f0, counts))
    if denominator <= 0:
        raise ValueError("The matching F0 denominator must be positive")
    value = float((numerical.macro_score(left, counts) - numerical.macro_score(right, counts)) / denominator)
    block_origins = int(np.ceil(block_days / stride_days))
    weights = numerical.moving_block_weights(len(counts), block_origins, REPLICATES, BOOTSTRAP_SEED)
    sampled_counts, sampled_f0 = weights @ counts, weights @ f0
    sampled_denominator = np.divide(sampled_f0, sampled_counts, out=np.full_like(sampled_f0, np.nan), where=sampled_counts > 0).mean(axis=1)
    valid = (sampled_counts > 0).all(axis=1) & np.isfinite(sampled_denominator) & (sampled_denominator > 0)
    if not valid.any():
        raise ValueError("No bootstrap draw observes every fixed target")
    draws = ((weights[valid] @ (left-right)) / sampled_counts[valid]).mean(axis=1) / sampled_denominator[valid]
    ci = np.quantile(draws, [(1-CONFIDENCE)/2, 1-(1-CONFIDENCE)/2]).tolist()
    return {"value": value, "ci": ci, "confidence": CONFIDENCE, "bootstrap_seed": BOOTSTRAP_SEED,
            "requested_resamples": REPLICATES, "valid_resamples": int(valid.sum()),
            "discarded_resamples": int((~valid).sum()), "requested_block_days": block_days,
            "stride_days": stride_days, "block_origins": block_origins,
            "target_span_days": (block_origins-1)*stride_days + 2, "origin_count": len(counts),
            "positive_means": "left objective has greater forecast loss than right objective",
            "scope": "Conditional date uncertainty for saved seed12000 fits on previously consumed development blocks; excludes training uncertainty and is not independent confirmation"}


def comparison_effects(left, right, f0, counts):
    result = {"daily": {str(block): paired_effect(left, right, f0, counts, block) for block in (3, 7, 14)},
              "nonoverlapping": {}, "by_target": []}
    for label, offset in (("even", 0), ("odd", 1)):
        result["nonoverlapping"][label] = {}
        for block in (3, 7, 14):
            part = [x[offset::2] for x in (left, right, f0, counts)]
            item = paired_effect(*part, block_days=block, stride_days=2)
            item.update({"origin_indexing": "zero-based", "target_windows_overlap": False,
                         "inferential_role": "descriptive sensitivity; context overlap and serial dependence remain"})
            result["nonoverlapping"][label][str(block)] = item
    for channel in range(2):
        result["by_target"].append(paired_effect(*[x[:, channel:channel+1] for x in (left, right, f0, counts)]))
    result["inferential_role"] = "Primary only for SORT NORM_ALIGNED versus RAW_ALIGNED, two-target macro, daily seven-day blocks; otherwise descriptive"
    return result


def alignment_gate(effects):
    if set(effects) != set(DATASETS):
        raise AssertionError("The gate requires exactly the two pre-fixed development sources")
    checks = {source: effects[source]["SORT"]["NORM_ALIGNED_vs_RAW_ALIGNED"]["daily"]["7"]["ci"][0] > DELTA for source in DATASETS}
    passed = all(checks.values())
    return {"decision": "RAW_ALIGNMENT_PRACTICALLY_RELEVANT" if passed else "CLOSE_CURRENT_OBJECTIVE_ALIGNMENT_SCREEN",
            "both_source_lower_bounds_exceed_one_percent": passed, "source_checks": checks,
            "threshold": DELTA, "new_method_demonstrated": False,
            "scope": "Two reused development blocks, LR3e-5, seed12000 and 200 updates; neither a novel PEFT success nor a universal statement about transformed losses"}


def remember_files(root, paths, hashes):
    for name in paths:
        path = (Path(root) / name).resolve()
        value = sha(path)
        if str(path) in hashes and hashes[str(path)] != value:
            raise AssertionError(f"File changed during analysis: {path}")
        hashes[str(path)] = value


def write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize_attempt_resources(records, scope):
    summary = numerical.summarize_resources([Path(r["guard_path"]) for r in records], scope, guard_directory=True)
    summary["guard_failures"] = sum(not r["completed"] for r in records)
    summary["safety_stops"] = sum(r["state"] == "safety_stop" for r in records)
    return summary


def score_procedures(arrays, scale):
    q = arrays["quantiles"]
    offsets = numerical.qcal_offsets(arrays["cal_predictions"], arrays["cal_target"], q)
    records, stats = {}, {}
    for procedure in PROCEDURES:
        prediction = arrays["eval_predictions"] if procedure == "SORT" else numerical.apply_qcal(arrays["eval_predictions"], offsets)
        records[procedure] = numerical.score_details(prediction, arrays["eval_target"], scale, q, arrays["eval_unsorted_predictions"])
        _, sums, counts = numerical.score_prediction(prediction, arrays["eval_target"], scale, q)
        stats[procedure] = (sums, counts)
        records[procedure]["origin_loss_sums"] = sums.tolist()
        records[procedure]["origin_valid_counts"] = counts.tolist()
    return records, stats, offsets


def verify_calibration(meta, panel, path):
    value = meta["gradient_calibration"]
    if read_json(path/"gradient_calibration.json") != value or sha(path/"gradient_calibration.json") != meta["gradient_calibration_sha256"]:
        raise AssertionError("Initial gradient record changed")
    indices = np.linspace(0, len(panel.origins["train"])-1, 8).astype(int)
    origins = panel.origins["train"][indices]
    if (value["origin_indices"] != indices.tolist() or value["origins"] != origins.tolist()
            or value["origin_sha256"] != numerical.array_hash(origins)
            or value["source_parameter_sha256"] != meta["audits"]["initial_adaptation_sha256"]
            or value["source_frozen_sha256"] != meta["audits"]["frozen_before_sha256"]
            or value["optimizer_steps"] != 0):
        raise AssertionError("Gradient calibration did not use the fixed initial train-only origins")
    if not all(value.get(key) is True for key in ("rng_unchanged", "parameters_unchanged", "gradients_cleared")):
        raise AssertionError("Calibration state was not restored")
    for key in ("norms", "multipliers", "objective_losses"):
        if set(value[key]) != set(ARMS) or any(not np.isfinite(v) or v <= 0 for v in value[key].values()):
            raise AssertionError(f"Invalid initial objective {key}")
    if value["multipliers"]["NATIVE"] != 1:
        raise AssertionError("NATIVE must retain its exact original loss scale")
    for arm in ARMS:
        numerical.compare_float(value["multipliers"][arm], value["norms"]["NATIVE"]/value["norms"][arm], "Initial gradient norm multiplier", rtol=1e-12, atol=0)
    expected_pairs = {f"{left}__{right}" for i, left in enumerate(ARMS) for right in ARMS[i+1:]}
    if set(value["cosines"]) != expected_pairs or any(not np.isfinite(v) or abs(v) > 1+1e-12 for v in value["cosines"].values()):
        raise AssertionError("Invalid paired initial gradient cosines")
    jacobian = value["jacobian_summary"]
    count = np.isfinite(np.stack([panel.target_values[o:o+48, panel.target_indices].T for o in origins])).sum()*21
    ordered = [jacobian[k] for k in ("min", "median", "p95", "max")]
    if (jacobian["count"] != count or not np.isfinite(ordered).all() or min(ordered) <= 0
            or ordered != sorted(ordered) or not 0 <= value["crossing_fraction"] <= 1):
        raise AssertionError("Invalid initial Jacobian or quantile crossing diagnostic")
    steps = meta["steps_completed"]
    norms = np.asarray(meta["gradient_norms"], dtype=np.float64)
    if norms.shape != (steps,) or not np.isfinite(norms).all() or np.any(norms < 0) or norms.max() <= 0:
        raise AssertionError("Every optimizer step must retain its pre-clipping gradient norm")
    if meta["clipping_steps"] != (np.flatnonzero(norms > 1)+1).tolist():
        raise AssertionError("Clipping count does not reconstruct from pre-clipping norms")
    numerical.compare_float(meta["audits"]["maximum_gradient_norm"], float(norms.max()), "Maximum training gradient norm", rtol=0, atol=0)
    if meta["loss_multiplier"] != value["multipliers"][meta["arm"]]:
        raise AssertionError("The objective's multiplier changed after initial calibration")
    unscaled, scaled = [np.asarray(meta[name], dtype=np.float64) for name in ("unscaled_training_losses", "scaled_training_losses")]
    if (unscaled.shape != (steps,) or scaled.shape != (steps,) or not np.isfinite(unscaled).all()
            or not np.isfinite(scaled).all() or not np.allclose(scaled, unscaled*meta["loss_multiplier"], rtol=2e-6, atol=1e-8)):
        raise AssertionError("Training did not retain the fixed loss multiplier")
    return value


def replay_native(meta, arrays, reference):
    original = read_json(reference/"result.json")
    for key in ("sampler_sha256", "validation_history", "best_step", "val_score"):
        if meta[key] != original[key]:
            raise AssertionError(f"NATIVE replay differs from its original fit: {key}")
    for key in ("initial_adaptation_sha256", "restored_adaptation_sha256"):
        if meta["audits"][key] != original["audits"][key]:
            raise AssertionError(f"NATIVE tensor replay failed: {key}")
    with np.load(reference/"predictions.npz", allow_pickle=False) as old:
        for key in ("val_predictions", "val_target", "val_origins", "val_timestamps", "quantiles", "target_indices", "target_channels", "val_loss_sums", "val_valid_counts"):
            if not np.array_equal(arrays[key], old[key], equal_nan=arrays[key].dtype.kind in "fc"):
                raise AssertionError(f"NATIVE validation array replay failed: {key}")
    record = meta["native_replay"]
    if (record.get("passed") is not True or not all(record["checks"].values()) or record["val_predictions_max_abs"] != 0
            or Path(record["reference_trial"]).resolve() != reference.resolve()
            or record["reference_result_sha256"] != sha(reference/"result.json")
            or record["reference_predictions_sha256"] != sha(reference/"predictions.npz")):
        raise AssertionError("Saved NATIVE replay record disagrees with independent reconstruction")
    return record


def verify_fit(root, contract, entry, panel, checked, hashes, smoke=False):
    from . import run_study

    meta = run_study.verify_fit(root, contract, entry, smoke=smoke)
    path, dataset = root/entry["path"], contract["datasets"][entry["dataset"]]
    for key, expected in {"seed": 12000, "seed_index": 0, "method": "OFF_LORA", "context": 336, "horizon": 48,
                          "micro_groups": 4, "effective_groups": 8, "autocast": "bfloat16", "autocast_weight_cache": False,
                          "weight_dtype": "float32", "tf32": False, "weight_decay": 0., "dropout": 0., "torch_threads": 2,
                          "interop_threads": 1, "gradient_clip_norm": 1., "trainable": 1206912, "trainable_parameters": 1206912,
                          "cache_created_this_trial": False, "holdout_file_opened": False}.items():
        if meta[key] != expected:
            raise AssertionError(f"Fixed fit numerical contract changed: {key}")
    if meta["training_loss"] != entry["arm"] or meta["source_hashes"] != contract["source_hashes"]:
        raise AssertionError("Fit objective or frozen source identity changed")
    if meta["fit_data_sha256"] != dataset["fit_data_sha256"] or meta["data_sha256"] != sha(panel.path):
        raise AssertionError("Fit data file changed")
    if meta["channels"] != panel.channels or meta["target_indices"] != panel.target_indices.tolist() or meta["stats_sha256"] != panel.stats_hash:
        raise AssertionError("Original input/target channel order or train statistics changed")
    for key in ("fit_mean", "fit_std", "fit_median"):
        if not np.array_equal(meta[key], getattr(panel, key)):
            raise AssertionError(f"Train-only statistic changed: {key}")
    steps, interval = (5, 5) if smoke else (200, 40)
    history = meta["validation_history"]
    if [x["step"] for x in history] != list(range(0, steps+1, interval)) or meta["steps_completed"] != steps:
        raise AssertionError("Fixed checkpoint-selection cadence changed")
    if not all(np.isfinite(x["val_score"]) for x in history):
        raise AssertionError("Nonfinite validation score")
    best = min(history, key=lambda x: x["val_score"])
    if meta["best_step"] != best["step"]:
        raise AssertionError("Checkpoint is not the first validation argmin")
    numerical.compare_float(meta["val_score"], best["val_score"], "Selected validation score")
    samples = np.random.default_rng(12000).integers(len(panel.origins["train"]), size=(steps, 8))
    if meta["sampler_sha256"] != numerical.array_hash(panel.origins["train"][samples]):
        raise AssertionError("Paired deterministic sampler changed")
    for split, origins in panel.origins.items():
        if meta["origin_counts"][split] != len(origins) or meta["origin_hashes"][split] != numerical.array_hash(origins):
            raise AssertionError("Fit training/selection origins changed")
    expected_counts = np.isfinite(panel.targets("train")).sum(axis=(0, 2)).tolist()
    if meta["train_valid_counts"] != expected_counts or expected_counts != dataset["smoke_train_valid_counts" if smoke else "train_valid_counts"]:
        raise AssertionError("Full train-origin target-cell denominators changed")
    expected_modules = [f"encoder.block.{block}.layer.{layer}.self_attention.{part}" for block in range(12) for layer in (0, 1) for part in ("q", "k", "v", "o")]
    expected_modules.append("output_patch_embedding.output_layer")
    if meta["module_map"] != expected_modules:
        raise AssertionError("The native 97-module LoRA map changed")
    audit = meta["audits"]
    for key in ("padded_group_identity_checked", "trainable_map_verified", "checkpoint_includes_all_trainable_parameters", "gradient_calibration_verified", "source_unchanged"):
        if audit.get(key) is not True:
            raise AssertionError(f"Fit audit did not pass: {key}")
    if (audit["zero_update_normalized_max_abs"] > 1e-5
            or not audit["frozen_before_sha256"] == audit["frozen_after_training_sha256"] == audit["frozen_after_first_update_sha256"]
            or audit["initial_adaptation_sha256"] == audit["before_restore_adaptation_sha256"]):
        raise AssertionError("Initial identity, frozen-parameter protection or actual optimizer update failed")
    for mapping in (meta["protected_hashes"], meta["native_source_hashes"], meta["cache_array_hashes"]):
        numerical.verify_file_hashes(root, mapping, checked)
        remember_files(root, mapping, hashes)
    numerical.verify_file_hashes(Path(meta["checkpoint"]), meta["native_checkpoint_hashes"], checked)
    cache = (root/dataset["smoke_cache" if smoke else "cache"]).resolve()
    manifest = read_json(cache/"manifest.json")
    signature = hashlib.sha256(json.dumps(manifest["contract"], sort_keys=True).encode()).hexdigest()
    if Path(meta["cache"]).resolve() != cache or signature != meta["cache_contract_sha256"] or not manifest.get("completed"):
        raise AssertionError("Original read-only feature cache changed")
    expected_cache_hashes = {str((cache/name).resolve()): digest for name, digest in manifest["array_hashes"].items()}
    expected_cache_hashes[str(cache/"manifest.json")] = sha(cache/"manifest.json")
    if meta["cache_array_hashes"] != expected_cache_hashes:
        raise AssertionError("Every original cache array and manifest must remain protected")
    arrays = numerical.load_prediction_archive(path/"predictions.npz", panel, ("val",), panel.quantiles)
    score, sums, counts = numerical.score_prediction(arrays["val_predictions"], arrays["val_target"], panel.fit_std[panel.target_indices], arrays["quantiles"])
    numerical.compare_float(score, meta["val_score"], "Reconstructed raw SORT validation score")
    if not np.array_equal(counts, arrays["val_valid_counts"]) or not np.allclose(sums, arrays["val_loss_sums"], rtol=1e-10, atol=1e-8):
        raise AssertionError("Saved validation loss/count statistics fail independent reconstruction")
    verify_calibration(meta, panel, path)
    if entry["arm"] == "NATIVE":
        replay_native(meta, arrays, (root/dataset["smoke_reference_fit" if smoke else "native_reference_fit"]).resolve())
        if read_json(path/"native_replay.json") != meta["native_replay"]:
            raise AssertionError("NATIVE replay artifact changed")
        remember_files(root, [path/"native_replay.json"], hashes)
    elif meta["native_replay"].get("applicable") is not False:
        raise AssertionError("A changed objective cannot claim NATIVE replay equality")
    remember_files(root, [path/name for name in ("result.json", "trial_contract.json", "predictions.npz", "best_trainable.pt", "gradient_calibration.json", "guard/status.json", "guard/resource_log.jsonl")], hashes)
    return meta


def verify_entry_sets(trials, forecasts, selection, done, contract_sha):
    expected = {(dataset, arm) for dataset in DATASETS for arm in ARMS}
    for entries, name in ((trials, "fits"), (forecasts, "forecasts")):
        if len(entries) != 6 or {(e["dataset"], e["arm"]) for e in entries} != expected:
            raise AssertionError(f"Exactly six unique {name} are required before future scoring")
        if any((e["method"], e["seed"], e["lr"], e["smoke"]) != ("OFF_LORA", 12000, 3e-5, False) for e in entries):
            raise AssertionError("The fixed objective experiment cannot silently change the optimizer budget")
    if (selection.get("completed") is not True or selection.get("global_choices_frozen") is not True
            or selection.get("fit_trial_count") != 6 or selection.get("study_contract_sha256") != contract_sha
            or selection.get("selected") != trials):
        raise AssertionError("Every completed fit must match the globally frozen selection exactly")
    for key, value in {"completed": True, "fits": 6, "forecasts": 6, "reused_forecasts": 2,
                       "study_contract_sha256": contract_sha}.items():
        if done.get(key) != value:
            raise AssertionError(f"Production must finish before future scoring: {key}")
    for key, entries in (("fit_guard_seconds", trials), ("forecast_guard_seconds", forecasts)):
        numerical.compare_float(done[key], sum(e["guard_seconds"] for e in entries), key, rtol=1e-12, atol=1e-8)


def verify_paired_fits(metas):
    if set(metas) != {(dataset, arm) for dataset in DATASETS for arm in ARMS}:
        raise AssertionError("Every objective requires paired initialization and sampling evidence")
    for dataset in DATASETS:
        reference = metas[(dataset, "NATIVE")]
        for arm in ARMS[1:]:
            value = metas[(dataset, arm)]
            for key in ("sampler_sha256", "train_valid_counts", "gradient_calibration", "cache_array_hashes"):
                if value[key] != reference[key]:
                    raise AssertionError(f"Objective arms differ before optimization: {dataset}/{key}")
            for key in ("initial_adaptation_sha256", "frozen_before_sha256"):
                if value["audits"][key] != reference["audits"][key]:
                    raise AssertionError("Paired initial model tensors differ")
            if value["validation_history"][0] != reference["validation_history"][0]:
                raise AssertionError("The common step-zero validation path differs")


def verify_forecast(root, contract, entry, fit, panel, checked, hashes):
    from . import run_study

    meta = run_study.verify_forecast(root, contract, entry)
    path = root/entry["path"]
    dataset = contract["datasets"][entry["dataset"]]
    expected_fit = root/"runs"/STUDY/"trials"/entry["dataset"]/entry["arm"]
    if (Path(meta["fit_trial"]).resolve() != expected_fit.resolve()
            or meta["fit_data_sha256"] != dataset["fit_data_sha256"]
            or meta["holdout_data_sha256"] != dataset["holdout_data_sha256"]
            or meta["restored_adaptation_sha256"] != fit["audits"]["restored_adaptation_sha256"]
            or meta["frozen_before_sha256"] != meta["frozen_after_sha256"]
            or meta["frozen_before_sha256"] != fit["audits"]["frozen_before_sha256"]
            or Path(meta["fit_cache"]).resolve() != Path(fit["cache"]).resolve()):
        raise AssertionError("Forecast did not restore its own selected fit, native weights or original data")
    for key in ("channels", "target_indices", "fit_std", "stats_sha256", "context", "horizon", "micro_groups",
                "autocast", "autocast_weight_cache", "weight_dtype", "tf32", "trainable", "trainable_parameters"):
        if meta[key] != fit[key]:
            raise AssertionError(f"Forecast numerical path changed from fitting: {key}")
    if meta["cache_created_this_trial"] is not False:
        raise AssertionError("The objective diagnostic cannot replace the original cache")
    for mapping in (meta["protected_hashes"], meta["native_source_hashes"], meta["parent_source_hashes"]):
        numerical.verify_file_hashes(root, mapping, checked)
        remember_files(root, mapping, hashes)
    numerical.validate_guard(path)
    arrays = numerical.load_prediction_archive(path/"predictions.npz", panel, ("cal", "eval"), panel.quantiles, with_unsorted=True)
    verify_split_metadata(meta, arrays)
    remember_files(root, [path/name for name in ("result.json", "trial_contract.json", "predictions.npz", "guard/status.json", "guard/resource_log.jsonl")], hashes)
    return meta, arrays


def verify_split_metadata(meta, arrays):
    for split in ("cal", "eval"):
        saved = meta["splits"][split]
        for key, value in {"origins": len(arrays[f"{split}_origins"]),
                           "predictions_shape": list(arrays[f"{split}_predictions"].shape),
                           "target_shape": list(arrays[f"{split}_target"].shape),
                           "origin_sha256": numerical.array_hash(arrays[f"{split}_origins"]),
                           "target_sha256": numerical.array_hash(arrays[f"{split}_target"])}.items():
            if saved[key] != value:
                raise AssertionError(f"Future prediction split metadata changed: {split}/{key}")
        numerical.compare_float(saved["unsorted_crossing"], float((np.diff(arrays[f"{split}_unsorted_predictions"], axis=2) < 0).mean()), "Unsorted quantile crossing", rtol=0, atol=0)


def load_f0(root, contract, dataset, panel, checked, hashes):
    descriptor = contract["datasets"][dataset]
    path = (root/descriptor["f0_forecast"]).resolve()
    meta = read_json(path/"result.json")
    fit_path = Path(meta["fit_trial"])
    fit = read_json(fit_path/"result.json")
    for value in (meta, fit):
        if (value.get("completed") is not True or value["dataset"] != dataset or value["method"] != "F0"
                or value["seed"] != 12000 or value["lr"] != 0 or value["trainable"] != 0 or value["best_step"] != 0):
            raise AssertionError("The reused baseline must be the original unadapted F0")
        numerical.verify_file_hashes(root, value["protected_hashes"], checked)
        remember_files(root, value["protected_hashes"], hashes)
    if (meta["fit_result_sha256"] != sha(fit_path/"result.json")
            or meta["fit_checkpoint_sha256"] != sha(fit_path/"best_trainable.pt")
            or meta["predictions_sha256"] != sha(path/"predictions.npz")
            or meta["restored_adaptation_sha256"] != fit["audits"]["restored_adaptation_sha256"]
            or meta["fit_data_sha256"] != descriptor["fit_data_sha256"]
            or meta["holdout_data_sha256"] != descriptor["holdout_data_sha256"]
            or meta["stats_sha256"] != panel.stats_hash
            or not np.array_equal(meta["fit_std"], panel.fit_std)):
        raise AssertionError("Reused F0 does not match its own fit and the original holdout archive")
    for key, expected in {"optimizer_steps": 0, "model_unchanged": True, "checkpoint_reload_verified": True,
                          "global_selection_verified_before_holdout_load": True, "cache_created_this_trial": False,
                          "context": 336, "horizon": 48, "micro_groups": 4, "autocast": "bfloat16",
                          "autocast_weight_cache": False, "weight_dtype": "float32", "tf32": False}.items():
        if meta[key] != expected:
            raise AssertionError(f"Reused F0 numerical contract changed: {key}")
    numerical.verify_file_hashes(root, fit["native_source_hashes"], checked)
    remember_files(root, fit["native_source_hashes"], hashes)
    numerical.validate_guard(path)
    numerical.validate_guard(fit_path)
    arrays = numerical.load_prediction_archive(path/"predictions.npz", panel, ("cal", "eval"), panel.quantiles, with_unsorted=True)
    verify_split_metadata(meta, arrays)
    remember_files(root, [path/"result.json", path/"predictions.npz", fit_path/"result.json", fit_path/"best_trainable.pt"], hashes)
    return fit, arrays, path


def attempt_audit(root, hashes):
    study = root/"runs"/STUDY
    records, lookup = [], {}
    modules = {f"experiments.{STUDY}.train": "fit", f"experiments.{STUDY}.forecast": "forecast",
               f"experiments.{STUDY}.analyse": "analysis_cpu"}
    current_cpu = []
    for path in sorted(study.rglob("status.json")):
        value = read_json(path)
        matches = [module for module in modules if module in value.get("command", [])]
        if not matches:
            continue
        if len(matches) != 1:
            raise AssertionError("A guard cannot represent multiple objective stages")
        stage = modules[matches[0]]
        if "finished_at" not in value:
            if stage == "analysis_cpu":
                current_cpu.append(str(path.parent))
                continue
            raise AssertionError("A fit or forecast attempt is still running")
        identity = (value["guard_pid"], value["started_at"])
        if identity in lookup:
            raise AssertionError("A physical guard attempt was counted more than once")
        lookup[identity] = (value, sha(path))
        completed = value.get("completed") is True
        if completed and (value.get("returncode") != 0 or value.get("reasons")):
            raise AssertionError("Successful guard has failure evidence")
        directory = path.parent
        files = [item for item in directory.iterdir() if item.is_file()]
        if (directory.parent/"failure.json").is_file():
            files.append(directory.parent/"failure.json")
        remember_files(root, files, hashes)
        records.append({"guard_path": str(directory), "stage": stage, "smoke": "--smoke" in value["command"],
                        "completed": completed, "state": value["state"], "returncode": value["returncode"],
                        "reasons": value["reasons"], "started_at": value["started_at"], "finished_at": value["finished_at"],
                        "elapsed_seconds": value["elapsed_seconds"]})
    phases = {}
    for name, stage, smoke, required in (("production_fit", "fit", False, 6), ("production_forecast", "forecast", False, 6),
                                          ("s0_fit", "fit", True, 6), ("completed_analysis_cpu", "analysis_cpu", False, None)):
        phase = [record for record in records if record["stage"] == stage and record["smoke"] is smoke]
        success = sum(record["completed"] for record in phase)
        if required is not None and success != required:
            raise AssertionError(f"Successful physical guards disagree with the fixed execution budget: {name}")
        phases[name] = {"attempts": len(phase), "successful_attempts": success, "unsuccessful_attempts": len(phase)-success,
                        "all_attempt_guard_seconds": sum(r["elapsed_seconds"] for r in phase),
                        "successful_guard_seconds": sum(r["elapsed_seconds"] for r in phase if r["completed"]),
                        "unsuccessful_guard_seconds": sum(r["elapsed_seconds"] for r in phase if not r["completed"])}
    invocations, recorded = [], set()
    for path in sorted((study/"invocations").glob("*.json")):
        value = read_json(path)
        if (value.get("study_contract_sha256") != sha(study/"study_contract.json")
                or "finished_at" not in value or "wall_seconds" not in value):
            raise AssertionError("Every runner invocation must retain its final wall time and frozen contract")
        numerical.compare_float(value["wall_seconds"], value["finished_at"]-value["started_at"], "Invocation wall clock", rtol=1e-6, atol=.02)
        for attempt in value["attempts"]:
            status = attempt.get("guard_status")
            if status is None:
                raise AssertionError("A runner attempt is missing its preserved status")
            identity = (status["guard_pid"], status["started_at"])
            if identity not in lookup or lookup[identity] != (status, attempt["guard_status_sha256"]) or identity in recorded:
                raise AssertionError("Attempt ledger and unique physical guard evidence disagree")
            recorded.add(identity)
        remember_files(root, [path], hashes)
        invocations.append({"path": str(path), "sha256": sha(path), "record": value})
    expected_recorded = {identity for identity, (value, _) in lookup.items() if f"experiments.{STUDY}.analyse" not in value["command"]}
    if recorded != expected_recorded or {x["record"]["phase"] for x in invocations} != {"smoke", "production"}:
        raise AssertionError("All fit, forecast and S0 attempts must be represented in both-stage runner ledgers")
    return {"phases": phases, "attempt_records": records, "runner_invocations": invocations,
            "current_analysis_guard_not_yet_finalized": current_cpu,
            "scope": "Current study only; original 12/13/14 training, caches and two reused F0 forecasts are sunk costs excluded. Final current analysis guard cost must be added by the outer completion audit."}


def run(root):
    from . import prepare, run_study
    from experiments.peft_external_gap_v1.train import Panel

    started = time.perf_counter()
    root = Path(root).resolve()
    study, output = root/"runs"/STUDY, root/"results"/STUDY
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Preserve an existing or partial analysis; never silently overwrite it")
    done = read_json(study/"completed.json")
    if done.get("completed") is not True or (done.get("fits"), done.get("forecasts"), done.get("reused_forecasts")) != (6, 6, 2):
        raise AssertionError("All six fits and six forecasts must finish before analysis")
    contract = prepare.validate_contract(root)
    contract_sha = sha(study/"study_contract.json")
    trials, forecasts = read_json(study/"trials.json"), read_json(study/"forecasts.json")
    selection = run_study.verify_selection(root, contract)
    verify_entry_sets(trials, forecasts, selection, done, contract_sha)
    smoke = run_study.verify_smoke(root, contract)
    if smoke != read_json(study/"smoke_trials.json"):
        raise AssertionError("S0 completion and full trial ledger disagree")
    checked, hashes = set(), {}
    numerical.verify_file_hashes(root, contract["protected_hashes"], checked)
    remember_files(root, contract["protected_hashes"], hashes)
    remember_files(root, [study/name for name in ("study_contract.json", "completed.json", "trials.json", "forecasts.json", "selection.json", "smoke_trials.json", "smoke_completed.json")], hashes)
    fits = {}
    for is_smoke, entries in ((True, smoke), (False, trials)):
        panels = {dataset: Panel(root/contract["datasets"][dataset]["fit_data_path"], "fit", is_smoke) for dataset in DATASETS}
        metas = {(entry["dataset"], entry["arm"]): verify_fit(root, contract, entry, panels[entry["dataset"]], checked, hashes, is_smoke) for entry in entries}
        verify_paired_fits(metas)
        if not is_smoke:
            fits = metas
    audit = attempt_audit(root, hashes)
    for key in ("fit", "forecast"):
        numerical.compare_float(audit["phases"][f"production_{key}"]["successful_guard_seconds"], done[f"{key}_guard_seconds"], "Physical guard cost versus completion ledger")
    # Every S0 and production NATIVE replay has now passed, before any C/E array is opened below.
    panels = {dataset: Panel(root/contract["datasets"][dataset]["holdout_data_path"], "forecast") for dataset in DATASETS}
    rows, sources, diagnostics, all_stats = [], {}, {"completed": True, "sources": {}}, {}
    forecast_lookup = {(e["dataset"], e["arm"]): e for e in forecasts}
    for dataset in DATASETS:
        panel = panels[dataset]
        if len(panel.origins["cal"]) != 13 or len(panel.origins["eval"]) != 83 or not np.all(np.diff(panel.origins["eval"]) == 24):
            raise AssertionError("Original 13-calibration/83-daily-evaluation origin contract changed")
        if panel.metadata["archive_role"] != "holdout_cal_eval_only_after_selection":
            raise AssertionError("Expected the isolated original holdout archive")
        sources[dataset], all_stats[dataset], diagnostics["sources"][dataset] = {}, {}, {}
        f0, arrays, f0_path = load_f0(root, contract, dataset, panel, checked, hashes)
        for arm in ("F0", *ARMS):
            if arm != "F0":
                entry = forecast_lookup[(dataset, arm)]
                _, arrays = verify_forecast(root, contract, entry, fits[(dataset, arm)], panel, checked, hashes)
                fit, path = fits[(dataset, arm)], root/entry["path"]
                diagnostics["sources"][dataset][arm] = {key: fit[key] for key in ("gradient_calibration", "gradient_norms", "clipping_steps", "validation_history", "native_replay", "unscaled_training_losses", "scaled_training_losses", "loss_multiplier", "best_step", "val_score")}
            else:
                fit, path = f0, f0_path
            records, stats, offsets = score_procedures(arrays, panel.fit_std[panel.target_indices])
            all_stats[dataset][arm] = stats
            sources[dataset][arm] = {"procedures": records, "qcal_offsets": offsets.tolist(), "target_channels": np.asarray(panel.channels)[panel.target_indices].tolist()}
            for procedure, record in records.items():
                baseline = sources[dataset]["F0"]["procedures"][procedure]["score"]
                rows.append({"source": dataset, "arm": arm, "method": "F0" if arm == "F0" else "OFF_LORA",
                             "seed": 12000, "lr": 0. if arm == "F0" else 3e-5, "procedure": procedure,
                             "score": record["score"], "val_score": fit["val_score"], "best_step": fit["best_step"],
                             "trainable": fit["trainable"], "path": str(path.relative_to(root)),
                             "relative_improvement_over_f0": (baseline-record["score"])/baseline,
                             **{key: record[key] for key in ("coverage80", "width80_scaled", "median_mse_scaled", "crossing_before_sort")}})
    effects = {}
    for dataset in DATASETS:
        effects[dataset] = {}
        for procedure in PROCEDURES:
            f0_sums, counts = all_stats[dataset]["F0"][procedure]
            if any(not np.array_equal(all_stats[dataset][arm][procedure][1], counts) for arm in ARMS):
                raise AssertionError("All objectives must use identical observed target cells")
            effects[dataset][procedure] = {}
            for left, right in (("NORM_ALIGNED", "RAW_ALIGNED"), ("NATIVE", "NORM_ALIGNED"), ("NATIVE", "RAW_ALIGNED")):
                effects[dataset][procedure][f"{left}_vs_{right}"] = comparison_effects(all_stats[dataset][left][procedure][0], all_stats[dataset][right][procedure][0], f0_sums, counts)
    effects = {"completed": True, "sources": effects, "gate": alignment_gate(effects), "bootstrap_seed": BOOTSTRAP_SEED,
               "primary_confidence_each": CONFIDENCE, "primary_family_size": 2, "primary_procedure": "SORT",
               "primary_comparison": "NORM_ALIGNED_vs_RAW_ALIGNED", "primary_block_days": 7,
               "secondary_scope": "QCAL, NATIVE comparisons, per-target, 3/14-day and parity intervals are descriptive; no additional family-wide guarantee"}
    gpu_records = [record for record in audit["attempt_records"] if record["stage"] != "analysis_cpu"]
    resources = {"all_gpu_attempts": summarize_attempt_resources(gpu_records, "All new fit/forecast/S0 attempts including failed preserved attempts"),
                 "successful_gpu_attempts": summarize_attempt_resources([r for r in gpu_records if r["completed"]], "Successful new fit/forecast/S0 attempts only")}
    cpu_records = [record for record in audit["attempt_records"] if record["stage"] == "analysis_cpu"]
    resources["prior_completed_cpu_analysis_attempts"] = summarize_attempt_resources(cpu_records, "Prior finalized CPU analysis attempts; current enclosing analysis is still running") if cpu_records else None
    costs = {"completed": True, **audit, "new_fit_jobs": 6, "new_forecast_jobs": 6, "s0_fit_jobs": 6,
             "reused_f0_forecasts": 2, "new_optimizer_updates": 1200, "s0_optimizer_updates": 30,
             "optimizer_seconds": sum(fit["optimizer_seconds_total"] for fit in fits.values()),
             "all_runner_invocation_wall_seconds": sum(item["record"]["wall_seconds"] for item in audit["runner_invocations"]),
             "analysis_seconds_before_final_hash_verification_and_output": time.perf_counter()-started}
    prepare.validate_contract(root)
    for filename, digest in hashes.items():
        if sha(Path(filename)) != digest:
            raise AssertionError(f"Protected analysis input changed: {filename}")
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output/"selected_results.csv", rows)
    for filename, value in (("effects.json", effects), ("diagnostics.json", diagnostics), ("metrics.json", {"completed": True, "sources": sources}),
                            ("costs.json", costs), ("resource_summary.json", resources)):
        write_json(output/filename, value)
    names = ("selected_results.csv", "effects.json", "diagnostics.json", "metrics.json", "costs.json", "resource_summary.json")
    verification = {"completed": True, "passed": True, "study": STUDY, "selected_procedure_rows": len(rows),
                    "new_fits": 6, "new_forecasts": 6, "reused_forecasts": 2, "s0_count": 6,
                    "native_replay_verified": True, "paired_initialization_sampler_and_gradient_calibration_verified": True,
                    "validation_and_evaluation_scores_recomputed": True, "missing_target_counts_recomputed": True,
                    "all_attempts_preserved_and_counted": True, "study_contract_sha256": contract_sha,
                    "artifact_hashes": hashes, "analysis_source_hashes": {str(Path(__file__).resolve()): sha(Path(__file__))},
                    "output_hashes": {name: sha(output/name) for name in names},
                    "scope": "Reused development blocks, conditional date uncertainty with one optimizer seed; no new-method claim"}
    write_json(output/"verification.json", verification)
    return {"completed": True, "passed": True, "selected_procedure_rows": len(rows), "gate": effects["gate"], "output": str(output)}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    args = cli.parse_args()
    print(json.dumps(run(args.root)))


if __name__ == "__main__":
    main()
