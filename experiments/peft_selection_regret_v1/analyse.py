"""Audit saved-checkpoint selection regret without fitting or changing selectors."""

import argparse
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import time

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_key] = "2"

import numpy as np

from experiments.peft_external_gap_v1 import analyse as numerical


STUDY = "peft_selection_regret_v1"
BOOTSTRAP_SEED = 2026090814
REPLICATES = 4000
CONFIDENCE = .9875
DELTA = .01
SELECTORS = ("F0", "FIXED_LOW", "LORA_V", "LORA_RECENT7", "ALL_V", "ALL_RECENT7", "HEAD_V")
SIMPLE_RULES = ("F0", "FIXED_LOW", "LORA_RECENT7", "ALL_V", "ALL_RECENT7", "HEAD_V")
REFERENCE_ANALYSIS_SHA256 = "e0d32f04674a45d02df58ce27eb4f432ea88f576780db3bfb554c9e90e5c702a"
read_json, write_json, sha = numerical.read_json, numerical.write_json, numerical.sha256_file


def paired_effect(left, right, f0, counts, block_days=7, stride_days=1, confidence=CONFIDENCE):
    left, right, f0, counts = [np.asarray(value, dtype=np.float64) for value in (left, right, f0, counts)]
    if (left.ndim != 2 or left.shape[1] != 2 or left.shape != right.shape or left.shape != f0.shape
            or left.shape != counts.shape or stride_days not in (1, 2)
            or not all(np.isfinite(v).all() for v in (left, right, f0, counts))
            or np.any(counts < 0)):
        raise ValueError("Pair the same two targets and observed cells within one fixed optimizer seed")
    block_origins = int(np.ceil(block_days / stride_days))
    weights = numerical.moving_block_weights(len(counts), block_origins, replicates=REPLICATES, seed=BOOTSTRAP_SEED)
    denominator = float(numerical.macro_score(f0, counts))
    if denominator <= 0:
        raise ValueError("The matching F0 score must be positive")
    value = float((numerical.macro_score(left, counts) - numerical.macro_score(right, counts)) / denominator)
    sampled_counts = weights @ counts
    sampled_f0 = weights @ f0
    valid = (sampled_counts > 0).all(axis=1)
    sampled_denominator = np.divide(sampled_f0, sampled_counts, out=np.full_like(sampled_f0, np.nan),
                                    where=sampled_counts > 0).mean(axis=1)
    valid &= np.isfinite(sampled_denominator) & (sampled_denominator > 0)
    if not valid.any():
        raise ValueError("No bootstrap draw observes both fixed targets")
    numerator = ((weights[valid] @ (left - right)) / sampled_counts[valid]).mean(axis=1)
    draws = numerator / sampled_denominator[valid]
    ci = np.quantile(draws, [(1-confidence)/2, 1-(1-confidence)/2]).tolist()
    decision = ("conditional_practical_selection_harm" if ci[0] > DELTA else
                "conditional_practical_fixed_low_harm" if ci[1] < -DELTA else
                "conditional_practical_equivalence" if ci[0] >= -DELTA and ci[1] <= DELTA else "inconclusive")
    return {"value": value, "ci": ci, "confidence": confidence, "decision": decision,
            "practical_threshold": DELTA, "bootstrap_seed": BOOTSTRAP_SEED,
            "requested_resamples": REPLICATES, "valid_resamples": int(valid.sum()),
            "discarded_resamples": int((~valid).sum()), "requested_block_days": block_days,
            "stride_days": stride_days, "block_origins": block_origins,
            "target_span_days": (block_origins - 1) * stride_days + 2,
            "origin_count": len(counts), "positive_means": "left selector is worse than right selector",
            "scope": "Conditional date-block uncertainty for stored seed12000 checkpoints and pre-fixed V selectors; excludes training and selector-estimation uncertainty"}


def comparison_effects(left, right, f0, counts):
    result = {"daily": {}, "nonoverlapping": {}}
    for block in (3, 7, 14):
        effect = paired_effect(left, right, f0, counts, block_days=block)
        effect["inferential_role"] = "primary only for SORT LORA_V versus FIXED_LOW at 7 days; otherwise descriptive"
        result["daily"][str(block)] = effect
    for label, offset in (("even", 0), ("odd", 1)):
        subset = slice(offset, None, 2)
        result["nonoverlapping"][label] = {}
        for block in (3, 7, 14):
            effect = paired_effect(left[subset], right[subset], f0[subset], counts[subset], block_days=block, stride_days=2)
            effect.update({"inferential_role": "descriptive sensitivity; overlapping contexts and temporal dependence remain",
                           "origin_indexing": "zero-based before subsampling", "target_windows_overlap": False})
            result["nonoverlapping"][label][str(block)] = effect
    return result


def oracle_diagnostics(candidate_scores, candidates, selectors, f0_score):
    by_id = {item["candidate_id"]: item for item in candidates}
    universes = {"ALL": list(by_id),
                 "LORA": [key for key, item in by_id.items() if item["method"] == "OFF_LORA"],
                 "HEAD": [key for key, item in by_id.items() if item["method"] in ("H_FULL", "H_MLP")]}
    if len(by_id) != 10 or {key: len(value) for key, value in universes.items()} != {"ALL": 10, "LORA": 3, "HEAD": 6}:
        raise AssertionError("Keep the fixed all-ten, LoRA-three and head-six hindsight universes")
    oracles = {name: min(keys, key=lambda key: (candidate_scores[key], key)) for name, keys in universes.items()}
    result = {}
    for selector, candidate_id in selectors.items():
        family = "LORA" if selector in ("FIXED_LOW", "LORA_V", "LORA_RECENT7") else "HEAD" if selector == "HEAD_V" else "ALL"
        if candidate_id not in universes[family]:
            raise AssertionError("A selector's regret universe must contain its selected candidate")
        result[selector] = {"candidate_id": candidate_id, "family": family,
                            "family_oracle_candidate_id": oracles[family], "all_oracle_candidate_id": oracles["ALL"],
                            "family_regret_over_f0": (candidate_scores[candidate_id] - candidate_scores[oracles[family]]) / f0_score,
                            "all_regret_over_f0": (candidate_scores[candidate_id] - candidate_scores[oracles["ALL"]]) / f0_score}
    return {"oracles": {name: {"candidate_id": key, "score": candidate_scores[key], "candidate_count": len(universes[name])}
                         for name, key in oracles.items()}, "selectors": result,
            "point_estimates_only": True, "confidence_intervals_computed": False,
            "scope": "Hindsight minima over already scored candidates; unavailable at deployment and not an implementable selector"}


def selection_gate(cell_effects, selector_scores, cell_sources):
    if len(cell_effects) != 4 or set(cell_effects) != set(selector_scores) or set(cell_effects) != set(cell_sources):
        raise AssertionError("The gate must include exactly four fixed block/source cells")
    if sorted(cell_sources.values()) != ["bike", "bike", "household", "household"]:
        raise AssertionError("Four cells represent two time blocks of each of the same two sources")
    positive, negative, supported = [], [], []
    checks = {}
    for cell, value in cell_effects.items():
        primary = value["SORT"]["LORA_V_vs_FIXED_LOW"]
        recent = value["SORT"]["LORA_RECENT7_vs_FIXED_LOW"]
        harmful = primary["daily"]["7"]["ci"][0] > DELTA
        opposite = primary["daily"]["7"]["ci"][1] < -DELTA
        recent_fails = recent["daily"]["7"]["value"] > DELTA
        nonoverlap_positive = all(item["nonoverlapping"][parity]["7"]["value"] > 0
                                  for item in (primary, recent) for parity in ("even", "odd"))
        if harmful:
            positive.append(cell)
        if opposite:
            negative.append(cell)
        if harmful and recent_fails and nonoverlap_positive:
            supported.append(cell)
        checks[cell] = {"primary_selection_harm_exceeds_one_percent": harmful,
                        "opposite_fixed_low_harm_exceeds_one_percent": opposite,
                        "recent7_point_harm_exceeds_one_percent": recent_fails,
                        "both_selectors_positive_on_both_nonoverlapping_subsets": nonoverlap_positive}
    simple = {}
    for rule in SIMPLE_RULES:
        values = {cell: (scores[rule] - min(scores["LORA_V"], scores["FIXED_LOW"])) / scores["F0"]
                  for cell, scores in selector_scores.items()}
        simple[rule] = {"values_over_f0": values, "avoids_point_loss_within_one_percent_in_all_cells": all(v <= DELTA for v in values.values())}
    veto_rules = [rule for rule, value in simple.items() if value["avoids_point_loss_within_one_percent_in_all_cells"]]
    repeated = {cell_sources[cell] for cell in supported} == {"bike", "household"}
    enter = repeated and bool(negative) and not veto_rules
    return {"decision": "enter_bounded_selection_mechanism_research" if enter else "close_current_B_branch",
            "primary_harm_cells": positive, "opposite_fixed_low_harm_cells": negative,
            "harm_cells_passing_recent_and_nonoverlap_checks": supported,
            "harm_repeats_across_both_sources": repeated, "cell_checks": checks,
            "simple_rule_veto": bool(veto_rules), "simple_rule_veto_rules": veto_rules,
            "simple_rule_diagnostics": simple, "simple_rule_comparison_is_descriptive": True,
            "scope": "Exploratory gate conditional on four cells from two sources and stored seed12000 checkpoints; a gate failure does not prove that all selectors or PEFT methods are unnecessary"}


def remember_files(root, paths, hashes):
    for path in paths:
        path = (Path(root) / path).resolve()
        digest = sha(path)
        if str(path) in hashes and hashes[str(path)] != digest:
            raise AssertionError(f"An artifact changed during analysis: {path}")
        hashes[str(path)] = digest


def verify_validation(root, cell, candidate, panel, checked, hashes):
    from . import prepare

    result_path = root / candidate["fit_result_path"]
    meta = read_json(result_path)
    for key in ("dataset", "method", "seed", "lr", "best_step"):
        if meta[key] != candidate[key]:
            raise AssertionError(f"Original candidate fit identity changed: {key}")
    if (not meta.get("completed") or meta.get("stage") != "fit" or meta.get("smoke")
            or candidate["seed"] != 12000 or meta["fit_data_sha256"] != cell["fit_data_sha256"]
            or meta["checkpoint_sha256"] != candidate["checkpoint_sha256"]
            or meta["audits"]["restored_adaptation_sha256"] != candidate["restored_adaptation_sha256"]):
        raise AssertionError("Use only each stored seed0 validation-best checkpoint")
    if cell["study_number"] == 13 and meta.get("wrapped_completed") is not True:
        raise AssertionError("Original temporal fit wrapper did not complete")
    numerical.validate_guard(root / candidate["fit_path"])
    paths = {candidate["fit_result_path"]: candidate["fit_result_sha256"],
             candidate["fit_predictions_path"]: candidate["fit_predictions_sha256"],
             candidate["checkpoint_path"]: candidate["checkpoint_sha256"]}
    numerical.verify_file_hashes(root, paths, checked)
    remember_files(root, paths, hashes)
    arrays = numerical.load_prediction_archive(root / candidate["fit_predictions_path"], panel, ("val",), panel.quantiles)
    score, sums, counts = numerical.score_prediction(arrays["val_predictions"], arrays["val_target"],
                                                    panel.fit_std[panel.target_indices], arrays["quantiles"])
    if len(sums) != 13 or not np.array_equal(counts, arrays["val_valid_counts"]):
        raise AssertionError("Each V must retain 13 origins and original observed-target counts")
    if not np.allclose(sums, arrays["val_loss_sums"], rtol=1e-10, atol=1e-8):
        raise AssertionError("Original V loss arrays do not reconstruct from saved predictions")
    numerical.compare_float(score, candidate["reported_val_score"], "V prediction score versus saved fit")
    numerical.compare_float(score, candidate["val_full_score"], "V prediction score versus frozen selector")
    recent = float(numerical.macro_score(sums[-7:], counts[-7:]))
    numerical.compare_float(recent, candidate["val_recent7_score"], "Recent-seven V reconstruction")
    if (prepare.sha256_array(arrays["val_loss_sums"].astype(np.float64)) != candidate["val_loss_sums_sha256"]
            or prepare.sha256_array(arrays["val_valid_counts"].astype(np.int64)) != candidate["val_valid_counts_sha256"]
            or candidate["val_origin_count"] != 13 or candidate["val_recent_origin_count"] != 7):
        raise AssertionError("Frozen V sufficient-statistic metadata changed")
    stages = [0] if candidate["method"] == "F0" else list(range(0, 201, 40))
    if [item["step"] for item in meta["validation_history"]] != stages:
        raise AssertionError("The diagnostic cannot substitute an unrecorded intermediate checkpoint")
    best = min(meta["validation_history"], key=lambda item: item["val_score"])
    if candidate["best_step"] != best["step"]:
        raise AssertionError("The preserved checkpoint is not its original first V argmin")
    return meta, {"score": score, "recent7_score": recent, "loss_sums": sums, "valid_counts": counts}


def verify_selector_reconstruction(root, cell, validation):
    candidates = list(cell["candidates"].values())
    priority = {"F0": 0, "OFF_LORA": 1, "H_FULL": 2, "H_MLP": 3}
    full = lambda item: item["val_full_score"]
    recent = lambda item: item["val_recent7_score"]
    lora = [item for item in candidates if item["method"] == "OFF_LORA"]
    head = [item for item in candidates if item["method"] in ("H_FULL", "H_MLP")]
    picks = {"F0": next(item for item in candidates if item["method"] == "F0"),
             "FIXED_LOW": next(item for item in lora if item["lr"] == 1e-5),
             "LORA_V": min(lora, key=lambda item: (full(item), item["lr"])),
             "LORA_RECENT7": min(lora, key=lambda item: (recent(item), item["lr"])),
             "HEAD_V": min(head, key=lambda item: (full(item), item["method"], item["lr"])),
             "ALL_V": min(candidates, key=lambda item: (full(item), priority[item["method"]], item["lr"])),
             "ALL_RECENT7": min(candidates, key=lambda item: (recent(item), priority[item["method"]], item["lr"]))}
    selected = {name: item["candidate_id"] for name, item in picks.items()}
    if selected != cell["selectors"]:
        raise AssertionError("A selector does not follow its fixed V-only argmin and tie rule")
    original = read_json(root / cell["selection_json_path"])["choices"][cell["dataset"]]
    for name, role in (("HEAD_V", "H"), ("LORA_V", "OFF_LORA")):
        if any(picks[name][key] != original[role][key] for key in ("method", "lr")):
            raise AssertionError("Full-V head/LoRA choices must match the original deployment selection")
    for name, item in picks.items():
        view = cell["selected"][name]
        if any(view[key] != item[key] for key in ("candidate_id", "method", "lr", "seed", "val_full_score", "val_recent7_score")):
            raise AssertionError("Detailed selector metadata differs from its selected candidate")
    if set(validation) != set(cell["candidates"]):
        raise AssertionError("V reconstruction must cover every candidate, including step0 candidates")
    return selected


def verify_forecast(root, cell, candidate, path, panel, contract, checked, hashes, smoke=False):
    from . import forecast

    path = Path(path).resolve()
    meta = read_json(path / "result.json")
    fit = read_json(root / candidate["fit_result_path"])
    new = smoke or candidate["needs_forecast"]
    numerical.validate_guard(path)
    for key in ("dataset", "method", "seed", "lr"):
        if meta[key] != candidate[key]:
            raise AssertionError(f"Forecast candidate identity changed: {key}")
    if (not meta.get("completed") or meta["fit_result_sha256"] != candidate["fit_result_sha256"]
            or meta["fit_checkpoint_sha256"] != candidate["checkpoint_sha256"]
            or meta["fit_data_sha256"] != cell["fit_data_sha256"]
            or meta["holdout_data_sha256"] != cell["holdout_data_sha256"]
            or meta["restored_adaptation_sha256"] != candidate["restored_adaptation_sha256"]
            or not meta.get("checkpoint_reload_verified") or not meta.get("model_unchanged")
            or meta["optimizer_steps"] != 0 or sha(path / "predictions.npz") != meta["predictions_sha256"]):
        raise AssertionError("Inference must restore the exact unchanged candidate without optimizer updates")
    if new:
        expected = {"stage": "diagnostic_forecast", "study": STUDY, "smoke": smoke,
                    "cell_id": cell["cell_id"], "candidate_id": candidate["candidate_id"],
                    "original_study": cell["study"], "study_number": cell["study_number"],
                    "plan_sha256": contract["plan_sha256"],
                    "selection_contract_sha256": sha(root / "runs" / STUDY / "selection_contract.json"),
                    "original_selection_sha256": cell["selection_json_sha256"],
                    "original_study_contract_sha256": cell["study_contract_sha256"],
                    "no_training": True, "diagnostic_only": True, "diagnostic_contract_verified_before_holdout_load": True,
                    "original_deployment_selection_gate_modified": False,
                    "cache_created_this_trial": False, "cache_used_for_forecast": False,
                    "context": 336, "horizon": 48, "micro_groups": 4, "autocast": "bfloat16",
                    "autocast_weight_cache": False, "weight_dtype": "float32", "tf32": False,
                    "torch_threads": 2, "interop_threads": 1, "dropout": 0.,
                    "trainable": numerical.COUNTS[candidate["method"]]}
        for name, value in expected.items():
            if meta.get(name) != value:
                raise AssertionError(f"Diagnostic inference contract mismatch: {name}")
        if (meta["after_inference_adaptation_sha256"] != candidate["restored_adaptation_sha256"]
                or meta["frozen_before_sha256"] != meta["frozen_after_sha256"]):
            raise AssertionError("Diagnostic inference altered model parameters")
        if meta["frozen_before_sha256"] != fit["audits"]["frozen_after_training_sha256"]:
            raise AssertionError("Diagnostic started from different frozen weights")
        source_hashes = {record["path"]: record["sha256"] for key, record in contract["source_hashes"].items() if key != "plan14"}
        if meta["source_hashes"] != source_hashes:
            raise AssertionError("Inference source provenance differs from the frozen selector contract")
        if meta["native_source_hashes"] != fit["native_source_hashes"]:
            raise AssertionError("Diagnostic native implementation differs from its original fitted candidate")
        for name in ("source_unchanged", "weights_unchanged", "checkpoint_reload_verified", "no_optimizer", "model_unchanged", "numerical_path_unchanged"):
            if meta["audits"].get(name) is not True:
                raise AssertionError(f"Inference audit failed: {name}")
        forecast.numerical_identity()
    else:
        reuse = candidate["reuse_forecast"]
        if (path != (root / reuse["path"]).resolve() or sha(path / "result.json") != reuse["result_sha256"]
                or meta["predictions_sha256"] != reuse["predictions_sha256"]
                or meta["selection_sha256"] != cell["selection_json_sha256"]):
            raise AssertionError("Reused forecast differs from its original verified selection")
        if cell["study_number"] == 13 and meta.get("wrapped_completed") is not True:
            raise AssertionError("Reused temporal forecast wrapper did not complete")
    if meta["stats_sha256"] != panel.stats_hash or meta["target_indices"] != panel.target_indices.tolist() or meta["channels"] != panel.channels:
        raise AssertionError("Forecast normalization or target map changed")
    for values in (meta["protected_hashes"], meta["source_hashes"], fit["native_source_hashes"]):
        numerical.verify_file_hashes(root, values, checked)
    arrays = numerical.load_prediction_archive(path / "predictions.npz", panel, ("cal", "eval"), panel.quantiles, with_unsorted=True)
    if smoke:
        if meta["audits"].get("reference_predictions_equal") is not True:
            raise AssertionError("S0 reference comparison did not complete")
        comparison = forecast.compare_reference(arrays, root / candidate["reuse_forecast"]["predictions_path"], meta["use_arcsinh"])
        if comparison != meta["reference_comparison"]:
            raise AssertionError("Recomputed S0 raw/normalized reference comparison differs from its saved audit")
    elif new and meta["reference_comparison"] is not None:
        raise AssertionError("A previously unseen candidate cannot claim an existing forecast reference")
    remember_files(root, [path / name for name in ("result.json", "predictions.npz", "trial_contract.json", "guard/status.json", "guard/resource_log.jsonl")], hashes)
    return meta, arrays


def attempt_audit(root, hashes):
    study = root / "runs" / STUDY
    records, identities, status_lookup = [], set(), {}
    for status_path in sorted(study.rglob("status.json")):
        status = read_json(status_path)
        if f"experiments.{STUDY}.forecast" not in status.get("command", []):
            continue
        identity = (status["guard_pid"], status["started_at"])
        if identity in identities:
            raise AssertionError("The same physical guard attempt appears twice in the diagnostic namespace")
        identities.add(identity)
        status_lookup[identity] = status
        if "finished_at" not in status or "elapsed_seconds" not in status:
            raise AssertionError("A diagnostic inference attempt is still unfinished")
        directory = status_path.parent
        files = [path for path in directory.iterdir() if path.is_file()]
        if (directory.parent / "failure.json").is_file():
            files.append(directory.parent / "failure.json")
        remember_files(root, files, hashes)
        records.append({"guard_path": str(directory), "smoke": "--smoke" in status["command"],
                        "completed": status.get("completed") is True, "state": status["state"],
                        "returncode": status["returncode"], "reasons": status["reasons"],
                        "started_at": status["started_at"], "finished_at": status["finished_at"],
                        "elapsed_seconds": status["elapsed_seconds"]})
    phases = {}
    for name, smoke, required in (("production", False, 28), ("s0", True, 4)):
        phase = [record for record in records if record["smoke"] is smoke]
        if sum(record["completed"] for record in phase) != required:
            raise AssertionError("Successful guard counts do not match the fixed 28-new/4-S0 execution")
        phases[name] = {"attempts": len(phase), "successful_attempts": required,
                        "unsuccessful_attempts": len(phase)-required,
                        "all_attempt_guard_seconds": sum(r["elapsed_seconds"] for r in phase),
                        "successful_guard_seconds": sum(r["elapsed_seconds"] for r in phase if r["completed"]),
                        "unsuccessful_guard_seconds": sum(r["elapsed_seconds"] for r in phase if not r["completed"]),
                        "observed_guard_span_seconds": (max(datetime.fromisoformat(r["finished_at"]) for r in phase)
                                                        - min(datetime.fromisoformat(r["started_at"]) for r in phase)).total_seconds(),
                        "span_scope": "First guard start to last guard finish, including intervening gaps; excludes runner work outside those endpoints"}
    invocations = []
    for path in sorted((study / "invocations").glob("*.json")):
        remember_files(root, [path], hashes)
        value = read_json(path)
        if (value.get("selection_contract_sha256") != sha(study / "selection_contract.json")
                or "finished_at" not in value or "wall_seconds" not in value):
            raise AssertionError("Every diagnostic invocation requires its frozen selection and final wall-time record")
        for attempt in value["attempts"]:
            status = attempt.get("guard_status")
            if status is None or status_lookup.get((status["guard_pid"], status["started_at"])) != status:
                raise AssertionError("A recorded inference attempt is missing its preserved guard evidence")
        invocations.append({"path": str(path), "sha256": sha(path), "record": value})
    if not invocations:
        raise AssertionError("Both S0 and production invocation records must be retained")
    return {"phases": phases, "attempt_records": records, "runner_invocations": invocations}


def write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize_attempt_resources(records, scope):
    result = numerical.summarize_resources([Path(record["guard_path"]) for record in records], scope, guard_directory=True)
    result["guard_failures"] = sum(not record["completed"] for record in records)
    result["safety_stops"] = sum(record["state"] == "safety_stop" for record in records)
    return result


def run(root):
    from . import prepare
    from .run_study import verify_entry, verify_smoke
    from experiments.peft_external_gap_v1.train import Panel

    started = time.perf_counter()
    root = Path(root).resolve()
    study, output = root / "runs" / STUDY, root / "results" / STUDY
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Preserve existing complete or partial selection diagnostic outputs")
    completed = read_json(study / "completed.json")
    if (completed.get("completed") is not True or completed.get("forecasts") != 28
            or completed.get("reused_forecasts") != 12):
        raise AssertionError("All 28 unseen-candidate forecasts must finish before opening future prediction arrays")
    contract_path = study / "selection_contract.json"
    contract = prepare.validate_contract(root, contract_path)
    contract_sha = sha(contract_path)
    if completed["selection_contract_sha256"] != contract_sha or prepare.build_contract(root, study) != contract:
        raise AssertionError("Current V-only selections, sources, parent artifacts or candidate metadata changed")
    if contract["missing_optional_sources"] or sha(numerical.__file__) != REFERENCE_ANALYSIS_SHA256:
        raise AssertionError("Freeze all diagnostic sources and the unchanged original score implementation")
    source_hashes = {str(Path(__file__).resolve()): sha(__file__), str(Path(numerical.__file__).resolve()): sha(numerical.__file__)}
    checked, hashes = set(), {}
    remember_files(root, [contract_path, study / "completed.json", study / "forecasts.json", study / "smoke_completed.json"], hashes)
    smoke_entries = verify_smoke(root, contract)
    entries = read_json(study / "forecasts.json")
    expected = {candidate["candidate_id"] for cell in contract["cells"].values() for candidate in cell["candidates"].values() if candidate["needs_forecast"]}
    if len(entries) != 28 or {entry["candidate_id"] for entry in entries} != expected:
        raise AssertionError("The new forecast listing is incomplete, duplicated or outside the frozen candidate universe")
    for entry in entries:
        verify_entry(root, contract, entry)
        numerical.compare_float(entry["guard_seconds"], numerical.validate_guard(root / entry["path"])["elapsed_seconds"],
                                "Per-candidate guard elapsed time", rtol=0, atol=1e-8)
    numerical.compare_float(sum(e["guard_seconds"] for e in entries), completed["forecast_guard_seconds"], "Successful new forecast guard cost", rtol=0, atol=1e-8)
    lookup = {entry["candidate_id"]: entry for entry in entries}
    panels, fit_results, validations, selectors = {}, {}, {}, {}
    for cell_id, cell in contract["cells"].items():
        fit, holdout = Panel(root / cell["fit_data_path"], "fit"), Panel(root / cell["holdout_data_path"], "forecast")
        if fit.metadata["holdout_archive_sha256"] != cell["holdout_data_sha256"] or fit.stats_hash != holdout.stats_hash:
            raise AssertionError("Candidate holdout data and train-only statistics differ from original preparation")
        if {k: len(v) for k, v in holdout.origins.items()} != {"cal": 13, "eval": 83}:
            raise AssertionError("The complete fixed C13/E83 windows must be retained")
        for origins in (*fit.origins.values(), *holdout.origins.values()):
            if np.any(np.diff(origins) != 24):
                raise AssertionError("The fixed daily origins changed")
        for offset in (0, 1):
            if np.any(np.diff(holdout.origins["eval"][offset::2]) < 48):
                raise AssertionError("Odd/even evaluation target windows still overlap")
        panels[cell_id] = (fit, holdout)
        validations[cell_id], fit_results[cell_id] = {}, {}
        for candidate_id, candidate in cell["candidates"].items():
            fit_results[cell_id][candidate_id], validations[cell_id][candidate_id] = verify_validation(root, cell, candidate, fit, checked, hashes)
        selectors[cell_id] = verify_selector_reconstruction(root, cell, validations[cell_id])
    # Every selector has now been reconstructed from V before any diagnostic C/E prediction values are loaded.
    smoke_audits = {}
    for entry in smoke_entries:
        cell = contract["cells"][entry["cell_id"]]
        candidate = cell["candidates"][entry["candidate_id"]]
        meta, arrays = verify_forecast(root, cell, candidate, root / entry["path"], panels[entry["cell_id"]][1], contract, checked, hashes, smoke=True)
        numerical.compare_float(entry["guard_seconds"], numerical.validate_guard(root / entry["path"])["elapsed_seconds"],
                                "Per-S0 guard elapsed time", rtol=0, atol=1e-8)
        smoke_audits[entry["cell_id"]] = meta["reference_comparison"]
        del arrays
    candidate_rows, selected_rows, diagnostics, statistics, effects, oracles = [], [], {}, {}, {}, {}
    score_by_cell = {}
    for cell_id, cell in contract["cells"].items():
        panel = panels[cell_id][1]
        scale = panel.fit_std[panel.target_indices]
        statistics[cell_id] = {procedure: {} for procedure in ("SORT", "QCAL")}
        rows_by_key = {}
        for candidate_id, candidate in cell["candidates"].items():
            new = candidate["needs_forecast"]
            path = root / (lookup[candidate_id]["path"] if new else candidate["reuse_forecast"]["path"])
            if new and path.resolve() != (root / candidate["new_forecast_output"]).resolve():
                raise AssertionError("Diagnostic forecast path differs from the pre-fixed candidate path")
            meta, arrays = verify_forecast(root, cell, candidate, path, panel, contract, checked, hashes)
            q = arrays["quantiles"]
            offset = numerical.qcal_offsets(arrays["cal_predictions"], arrays["cal_target"], q)
            validation = validations[cell_id][candidate_id]
            detail = {"cell": cell_id, "candidate_id": candidate_id, "calibration_offsets": offset.tolist(),
                      "val_origin_loss_sums": validation["loss_sums"].tolist(), "val_origin_valid_counts": validation["valid_counts"].tolist(),
                      "calibration_estimation": "C-only per-target/per-quantile offsets, horizon pooled; C QCAL score is in-sample resubstitution",
                      "new_forecast": new, "best_step": candidate["best_step"]}
            for procedure in ("SORT", "QCAL"):
                cal = arrays["cal_predictions"] if procedure == "SORT" else numerical.apply_qcal(arrays["cal_predictions"], offset)
                prediction = arrays["eval_predictions"] if procedure == "SORT" else numerical.apply_qcal(arrays["eval_predictions"], offset)
                score, sums, counts = numerical.score_prediction(prediction, arrays["eval_target"], scale, q)
                statistics[cell_id][procedure][candidate_id] = (sums, counts)
                cal_score, cal_sums, cal_counts = numerical.score_prediction(cal, arrays["cal_target"], scale, q)
                scored = numerical.score_details(prediction, arrays["eval_target"], scale, q, arrays["eval_unsorted_predictions"])
                first = float(numerical.macro_score(sums[:41], counts[:41]))
                last = float(numerical.macro_score(sums[41:], counts[41:]))
                row = {"cell": cell_id, "study_number": cell["study_number"], "dataset": cell["dataset"],
                       "candidate_id": candidate_id, "method": candidate["method"], "lr": candidate["lr"], "seed": 12000,
                       "procedure": procedure, "score": score, "val_score": candidate["val_full_score"],
                       "val_recent7_score": candidate["val_recent7_score"], "cal_score": cal_score,
                       "cal_score_is_resubstitution": procedure == "QCAL", "eval_first41_score": first,
                       "eval_last42_score": last, "coverage80": scored["coverage80"], "width80_scaled": scored["width80_scaled"],
                       "best_step": candidate["best_step"], "trainable": numerical.COUNTS[candidate["method"]],
                       "new_forecast": new, "path": path.relative_to(root).as_posix()}
                candidate_rows.append(row)
                rows_by_key[(candidate_id, procedure)] = row
                detail[procedure] = {"evaluation": scored, "cal_score": cal_score, "cal_loss_sums": cal_sums.tolist(),
                                     "cal_valid_counts": cal_counts.tolist(), "eval_loss_sums": sums.tolist(), "eval_valid_counts": counts.tolist(),
                                     "eval_first41_score": first, "eval_last42_score": last}
            diagnostics[candidate_id] = detail
            del arrays
        effects[cell_id], oracles[cell_id] = {}, {}
        for procedure in ("SORT", "QCAL"):
            current = statistics[cell_id][procedure]
            f0, counts = current[selectors[cell_id]["F0"]]
            if any(not np.array_equal(value[1], counts) for value in current.values()):
                raise AssertionError("Every candidate/selector must evaluate the same observed cells")
            effects[cell_id][procedure] = {}
            for rule in ("LORA_V", "LORA_RECENT7"):
                left, right = current[selectors[cell_id][rule]][0], current[selectors[cell_id]["FIXED_LOW"]][0]
                comparison = comparison_effects(left, right, f0, counts)
                for block, value in comparison["daily"].items():
                    value["inferential_role"] = "primary" if procedure == "SORT" and rule == "LORA_V" and block == "7" else "descriptive sensitivity"
                effects[cell_id][procedure][rule + "_vs_FIXED_LOW"] = comparison
            scores = {key: float(numerical.macro_score(value[0], counts)) for key, value in current.items()}
            f0_score = scores[selectors[cell_id]["F0"]]
            oracles[cell_id][procedure] = oracle_diagnostics(scores, list(cell["candidates"].values()), selectors[cell_id], f0_score)
            for rule in SELECTORS:
                candidate_id = selectors[cell_id][rule]
                oracle = oracles[cell_id][procedure]["selectors"][rule]
                selected_rows.append({**rows_by_key[(candidate_id, procedure)], "selector": rule,
                                      "relative_loss_over_f0": scores[candidate_id]/f0_score-1,
                                      "family_oracle": oracle["family"], "family_oracle_candidate_id": oracle["family_oracle_candidate_id"],
                                      "family_oracle_regret_over_f0": oracle["family_regret_over_f0"],
                                      "all_oracle_candidate_id": oracle["all_oracle_candidate_id"],
                                      "all_oracle_regret_over_f0": oracle["all_regret_over_f0"]})
            if procedure == "SORT":
                score_by_cell[cell_id] = {rule: scores[candidate_id] for rule, candidate_id in selectors[cell_id].items()}
    if len(candidate_rows) != 80 or len(selected_rows) != 56:
        raise AssertionError("Report every 40 candidate and 28 selector SORT/QCAL procedure exactly once")
    gate = selection_gate(effects, score_by_cell, {key: cell["dataset"] for key, cell in contract["cells"].items()})
    effect_output = {"completed": True, "cells": effects, "gate": gate, "primary_family_size": 4,
                     "primary_confidence_each": CONFIDENCE, "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_replicates": REPLICATES,
                     "primary_procedure": "SORT", "primary_comparison": "LORA_V minus FIXED_LOW, divided by the same cell's F0",
                     "primary_block_days": 7, "optimizer_seed": 12000,
                     "scope": "Exploratory four cells from two sources; no pooled temporal blocks or independent-source claim",
                     "secondary_inference": "QCAL, Recent7 and 3/14-day/odd/even intervals are descriptive; only four full-SORT 7-day comparisons define the primary family",
                     "checkpoint_limitation": "Only each stored V-best checkpoint is available; this does not measure regret over all early-stopping steps"}
    attempt = attempt_audit(root, hashes)
    numerical.compare_float(attempt["phases"]["production"]["successful_guard_seconds"], completed["forecast_guard_seconds"], "Attempt inventory versus successful forecast costs", rtol=0, atol=1e-8)
    production = [root / entry["path"] for entry in entries]
    smoke_paths = [root / entry["path"] for entry in smoke_entries]
    all_production = [record for record in attempt["attempt_records"] if not record["smoke"]]
    all_smoke = [record for record in attempt["attempt_records"] if record["smoke"]]
    resources = {"successful_new_production": numerical.summarize_resources(production, "28 successful new inference guards; original reused forecasts excluded"),
                 "all_new_production_attempts": summarize_attempt_resources(all_production, "All current diagnostic production attempts, including preserved interruptions"),
                 "successful_s0": numerical.summarize_resources(smoke_paths, "Four new reference replay guards, separate from production"),
                 "all_s0_attempts": summarize_attempt_resources(all_smoke, "All current S0 attempts, including preserved interruptions")}
    costs = {"completed": True, "new_training_updates": 0, "new_forecasts": 28, "reused_forecasts": 12, "new_s0_forecasts": 4,
             "original_training_and_forecast_sunk_costs_excluded": True,
             "final_successful_runner_invocation_wall_seconds": completed["invocation_wall_seconds"],
             "runner_wall_scope": "This field describes the final runner invocation only; all invocation records and failed/successful attempt costs are retained separately",
             **attempt, "analysis_wall_seconds_before_serialization": time.perf_counter()-started}
    prepare.verify_protected_hashes(contract, root)
    numerical.verify_file_hashes(root, hashes, set())
    numerical.verify_file_hashes(root, source_hashes, set())
    if sha(contract_path) != contract_sha:
        raise AssertionError("Frozen selector contract changed during diagnostic analysis")
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "candidate_results.csv", candidate_rows)
    write_csv(output / "selected_results.csv", selected_rows)
    for name, value in (("effects.json", effect_output), ("oracle_regret.json", oracles), ("diagnostics.json", diagnostics),
                        ("costs.json", costs), ("resource_summary.json", resources)):
        write_json(output / name, value)
    verification = {"passed": True, "completed": True, "candidate_count": 40, "candidate_procedure_rows": 80,
                    "selected_procedure_rows": 56, "selector_count": 7, "cell_count": 4, "physical_sources": 2,
                    "new_forecasts": 28, "reused_forecasts": 12, "s0_forecasts": 4, "new_training_updates": 0,
                    "selection_contract_sha256": contract_sha, "analysis_sources": source_hashes, "artifact_hashes": hashes,
                    "parent_artifacts_unchanged": True, "s0_recomputed_reference_audits": smoke_audits,
                    "output_hashes": {path.name: sha(path) for path in sorted(output.iterdir()) if path.is_file()},
                    "checks": ["exact candidate/selector/forecast identities and original source/checkpoint/parent protection",
                               "all candidate V predictions and loss counts independently reconstructed before C/E prediction reads",
                               "full and recent-seven selectors replayed with fixed tie priorities; original H/LoRA selection preserved",
                               "four S0 all-origin raw/normalized reference comparisons recomputed on CPU",
                               "same observed targets, native quantiles, SORT and C-only QCAL across all candidates",
                               "four primary paired 98.75% intervals; parity block durations and hindsight oracle distinction",
                               "all current successful and interrupted inference attempts counted separately from historical sunk costs"]}
    write_json(output / "verification.json", verification)
    print(json.dumps({"passed": True, "candidate_rows": 80, "selector_rows": 56, "gate": gate["decision"], "output": str(output)}), flush=True)
    return verification


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    run(parser.parse_args().root)


if __name__ == "__main__":
    main()
