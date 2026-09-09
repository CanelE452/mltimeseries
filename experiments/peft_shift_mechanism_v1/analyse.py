"""Recompute predictions, paired contrasts and conditional uncertainty after completion."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[name] = "2"

import numpy as np

from .raw import episode_loss, run as run_raw
from .run_study import CONDITIONS, METHODS, write_json


def effect(numerator, denominator, weights, confidence):
    curve = numerator.mean(axis=0)
    denom = denominator.mean(axis=0)
    draws = (weights @ curve) / (weights @ denom)
    tail = (1 - confidence) / 2
    low, high = np.quantile(draws, [tail, 1 - tail])
    by_corpus = numerator.mean(axis=1) / denominator.mean(axis=1)
    return {"value": float(curve.mean() / denom.mean()), "corpus_values": by_corpus.tolist(),
            "ci": [float(low), float(high)], "confidence": confidence,
            "scope": "paired evaluation episodes conditional on three fitted corpus/optimizer repetitions"}


def run(root):
    study = root / "runs/peft_shift_mechanism_v1"
    output = root / "results/peft_shift_mechanism_v1"
    output.mkdir(parents=True, exist_ok=True)
    done = json.loads((study / "completed.json").read_text(encoding="utf-8"))
    selected = json.loads((study / "selected.json").read_text(encoding="utf-8"))
    if not done["completed"] or len(selected) != 96:
        raise AssertionError("Expected 4 conditions × 3 corpora × 8 selected FM entries")
    lookup = {(e["condition"], e["corpus"], e["method"]): e for e in selected}
    expected_selected = {(q, c, m) for q in CONDITIONS for c in range(3) for m in ("F0", *METHODS)}
    if set(lookup) != expected_selected:
        raise AssertionError("Missing/duplicate/unexpected selected entry")
    selection = json.loads((study / "selection.json").read_text(encoding="utf-8"))
    expected_trials = set()
    for condition, corpus, method in sorted(expected_selected):
        entry = lookup[(condition, corpus, method)]
        lr = entry["lr"]
        canonical = f"runs/peft_shift_mechanism_v1/trials/{condition}_c{corpus}/{method}/lr_{lr:.0e}"
        if entry["path"] != canonical:
            raise AssertionError("Noncanonical selected trial path")
        meta = json.loads((root / canonical / "result.json").read_text(encoding="utf-8"))
        if meta["method"] != method or meta["seed"] != 7100 + corpus or meta["lr"] != lr:
            raise AssertionError("Selected method/seed/LR mismatch")
        if not np.isclose(entry["val_score"], meta["val_score"], rtol=0, atol=1e-12):
            raise AssertionError("Selected validation score mismatch")
        if method == "F0":
            if lr != .001:
                raise AssertionError("F0 path contract mismatch")
            expected_trials.add(canonical)
        else:
            choice = selection[f"{condition}/{method}"]
            grid = (.0003, .001) if method in ("H_LIN", "H_MLP") else (.00003, .0001)
            candidates = choice["candidates"]
            if len(candidates) != 2 or {candidate["lr"] for candidate in candidates} != set(grid):
                raise AssertionError("Incorrect HPO candidate grid")
            for candidate in candidates:
                cp = f"runs/peft_shift_mechanism_v1/trials/{condition}_c0/{method}/lr_{candidate['lr']:.0e}"
                if candidate["path"] != cp:
                    raise AssertionError("HPO candidate path mismatch")
                cm = json.loads((root / cp / "result.json").read_text(encoding="utf-8"))
                if not np.isclose(candidate["val_score"], cm["val_score"], rtol=0, atol=1e-12):
                    raise AssertionError("HPO candidate validation mismatch")
                expected_trials.add(cp)
            if lr != choice["lr"] or lr != min(candidates, key=lambda item: item["val_score"])["lr"]:
                raise AssertionError("Selected LR is not validation argmin/fixed across corpora")
            expected_trials.add(canonical)
    contract = json.loads((study / "study_contract.json").read_text(encoding="utf-8"))
    plan_path = root / "_docs/notes/tsfm_topics/08_peft_shift_mechanism_plan_20260908.md"
    if hashlib.sha256(plan_path.read_bytes()).hexdigest() != contract["plan_hash"]:
        raise AssertionError("Plan changed after execution contract")
    for file, expected in contract["sources"].items():
        if hashlib.sha256((root / file).read_bytes()).hexdigest() != expected:
            raise AssertionError(f"Source changed: {file}")
    data_manifest = json.loads((study / "data/manifest.json").read_text(encoding="utf-8"))
    if hashlib.sha256((study / "data/manifest.json").read_bytes()).hexdigest() != contract["data_manifest_sha256"]:
        raise AssertionError("Data manifest changed after execution contract")
    if not data_manifest["all_qc_passed"]:
        raise AssertionError("Data QC failed")
    records, losses, choices, costs = [], {}, {}, []
    quantiles = None
    episode_reference = None
    for condition in CONDITIONS:
        h = min((lookup[(condition, 0, method)] for method in ("H_LIN", "H_MLP")), key=lambda e: e["val_score"])["method"]
        best_fm = min((lookup[(condition, 0, method)] for method in METHODS), key=lambda e: e["val_score"])["method"]
        choices[condition] = {"head": h, "best_fm_by_c0_validation": best_fm}
        for corpus in range(3):
            data_path = study / "data" / f"{condition}_c{corpus}.npz"
            if hashlib.sha256(data_path.read_bytes()).hexdigest() != data_manifest["files"][data_path.name]["sha256"]:
                raise AssertionError("Data hash mismatch")
            if contract["data_hashes"][data_path.name] != data_manifest["files"][data_path.name]["sha256"]:
                raise AssertionError("Prepared data differs from execution contract")
            with np.load(data_path, allow_pickle=False) as data:
                target = data["target_eval"].astype(np.float64)
                val_target = data["target_val"].astype(np.float64)
                q = data["quantiles"].astype(np.float64)
                ids = data["episode_ids_eval"]
                oracle = data["oracle_quantiles_eval"].astype(np.float64)
                oracle_mean = data["oracle_mean_eval"].astype(np.float64)
                context = data["context_eval"].astype(np.float64)
                metadata = json.loads(data["manifest_json"].item())
                lag = metadata["generator"]["self_lag"]
                theta = metadata["generator"]["theta_radians"]
                leads = np.arange(16)
                components = np.stack((.5 * context[:, 0, 256 + leads - lag],
                                       np.sqrt(.39) * np.cos(theta) * context[:, 1, 256 + leads - 48],
                                       np.sqrt(.39) * np.sin(theta) * context[:, 2, 256 + leads - 48]), axis=-1)
                component_design = np.column_stack((np.ones(target.size), components.reshape(-1, 3)))
                if not np.allclose(components.sum(axis=-1), oracle_mean, rtol=1e-5, atol=5e-7):
                    raise AssertionError("Oracle component reconstruction mismatch")
            if episode_reference is None:
                episode_reference = ids
                quantiles = q
            if not np.array_equal(episode_reference, ids) or not np.array_equal(q, quantiles):
                raise AssertionError("Paired episode/grid contract mismatch")
            f0_dir = root / lookup[(condition, corpus, "F0")]["path"]
            with np.load(f0_dir / "predictions.npz", allow_pickle=False) as f0_archive:
                frozen_prediction = f0_archive["eval_predictions"].astype(np.float64)
            raw_dir = study / "raw" / f"{condition}_c{corpus}"
            run_raw(data_path, f0_dir / "predictions.npz", raw_dir)
            for method in ("F0", *METHODS, "RAW", "F0_RAW", "ORACLE"):
                if method == "ORACLE":
                    prediction, meta, path = oracle, {}, None
                else:
                    path = raw_dir / method if method in ("RAW", "F0_RAW") else root / lookup[(condition, corpus, method)]["path"]
                    meta = json.loads((path / "result.json").read_text(encoding="utf-8"))
                    if not meta["completed"]:
                        raise AssertionError(f"Incomplete {path}")
                    if method not in ("RAW", "F0_RAW"):
                        if meta["data_sha256"] != contract["data_hashes"][data_path.name]:
                            raise AssertionError("Trial used different input data")
                        guard = json.loads((path / "guard/status.json").read_text(encoding="utf-8"))
                        if not guard["completed"] or guard["returncode"] != 0:
                            raise AssertionError(f"Guard incomplete {path}")
                    with np.load(path / "predictions.npz", allow_pickle=False) as predictions:
                        prediction = predictions["eval_predictions"].astype(np.float64)
                        vp = predictions["val_predictions"].astype(np.float64)
                        if not np.array_equal(predictions["eval_target"], target):
                            raise AssertionError(f"Target mismatch {path}")
                    recomputed_val = float(episode_loss(vp, val_target, q).mean())
                    if not np.isclose(recomputed_val, meta["val_score"], rtol=1e-6, atol=1e-7):
                        raise AssertionError(f"Validation score mismatch {path}")
                if prediction.shape != (512, 21, 16) or not np.isfinite(prediction).all():
                    raise AssertionError(f"Invalid prediction {condition}/{corpus}/{method}")
                if method in METHODS and meta["best_step"] == 0 and not np.array_equal(prediction, frozen_prediction):
                    raise AssertionError(f"Selected step-zero prediction differs from F0: {condition}/{corpus}/{method}")
                loss = episode_loss(prediction, target, q)
                losses[(condition, corpus, method)] = loss
                if method != "ORACLE" and not np.isclose(loss.mean(), meta["eval_score"], rtol=1e-6, atol=1e-7):
                    raise AssertionError(f"Evaluation score mismatch {path}")
                med = int(np.argmin(abs(q - .5)))
                lower, upper = int(np.argmin(abs(q - .1))), int(np.argmin(abs(q - .9)))
                # Exploratory projection of predictions, not a causal/representation-identification test.
                component_fit = np.linalg.lstsq(component_design, prediction[:, med].reshape(-1), rcond=None)[0]
                records.append({"condition": condition, "corpus": corpus, "method": method,
                                "score": float(loss.mean()), "median_mse": float(np.mean((target - prediction[:, med]) ** 2)),
                                "oracle_mean_error_mse": float(np.mean((oracle_mean - prediction[:, med]) ** 2)),
                                "coverage80": float(np.mean((target >= prediction[:, lower]) & (target <= prediction[:, upper]))),
                                "crossing": float(np.mean(np.diff(prediction, axis=1) < 0)),
                                "exploratory_intercept": float(component_fit[0]),
                                "exploratory_self_gain": float(component_fit[1]),
                                "exploratory_u_gain": float(component_fit[2]),
                                "exploratory_v_gain": float(component_fit[3]),
                                "val_score": meta.get("val_score"), "lr": meta.get("lr"),
                                "best_step": meta.get("best_step"), "wall_seconds": meta.get("wall_seconds"),
                                "path": str(path.relative_to(root)).replace("\\", "/") if path else str(data_path.relative_to(root))})
    def gather(condition, method):
        return np.stack([losses[(condition, c, method)] for c in range(3)])
    rng = np.random.default_rng(2026090808)
    weights = rng.multinomial(512, np.ones(512) / 512, size=4000).astype(np.float64) / 512
    contrasts = {}
    for condition in CONDITIONS:
        h = choices[condition]["head"]
        f0 = gather(condition, "F0")
        head_contrast = effect(gather(condition, h) - gather(condition, "OFF_LORA"), f0, weights, .9875)
        upper = head_contrast["ci"][1]
        head_contrast["decision"] = (
            "repeat_practical_harm" if upper < -.01 and max(head_contrast["corpus_values"]) < 0
            else "practical_additional_gain_not_supported" if upper < .01
            else "practical_gain_supported" if head_contrast["ci"][0] > .01 and min(head_contrast["corpus_values"]) > 0
            else "inconclusive")
        contrasts[condition] = {"head_vs_off": head_contrast,
            "raw_vs_off": effect(gather(condition, "RAW") - gather(condition, "OFF_LORA"), f0, weights, .9875),
            "joint_minus_lp": effect(gather(condition, "JOINT") - gather(condition, "LP"), f0, weights, .9875),
            "raw_vs_validation_best_fm": effect(gather(condition, "RAW") - gather(condition, choices[condition]["best_fm_by_c0_validation"]), f0, weights, .9875)}
    d = {condition: gather(condition, "TIME") - gather(condition, "GROUP") for condition in CONDITIONS}
    # Raw-score factorial contrasts; never mix condition-specific denominators inside D.
    ones = np.ones((3, 512))
    interactions = {
        "angle": effect((d["Q01"] - d["Q00"] + d["Q11"] - d["Q10"]) / 2, ones, weights, .975),
        "self_lag": effect((d["Q10"] - d["Q00"] + d["Q11"] - d["Q01"]) / 2, ones, weights, .975),
        "nonadditive_exploratory": effect(d["Q11"] - d["Q10"] - d["Q01"] + d["Q00"], ones, weights, .95),
        "D_by_condition": {condition: effect(value, ones, weights, .95) for condition, value in d.items()},
        "time_group_both_selected_step_zero": {
            condition: all(record["best_step"] == 0 for record in records
                           if record["condition"] == condition and record["method"] in ("TIME", "GROUP"))
            for condition in CONDITIONS},
        "interpretation": "If TIME and GROUP both select step zero, a zero contrast describes the selection outcome and does not establish equal module roles or absence of an interaction under other optimization procedures.",
    }
    guard_paths = sorted((study / "trials").glob("*/*/*/guard/status.json"))
    actual_trials = {str(path.parent.parent.relative_to(root)).replace("\\", "/") for path in guard_paths}
    if len(guard_paths) != 124 or actual_trials != expected_trials:
        raise AssertionError(f"Expected 124 completed trial guards, got {len(guard_paths)}")
    resources = []
    for path in guard_paths:
        item = json.loads(path.read_text(encoding="utf-8"))
        if not item["completed"] or item["returncode"] != 0 or item["reasons"]:
            raise AssertionError(f"Failed trial {path}")
        meta = json.loads((path.parent.parent / "result.json").read_text(encoding="utf-8"))
        costs.append({"path": str(path.parent.parent.relative_to(root)), "guard_seconds": item["elapsed_seconds"],
                      "method": meta["method"], "seconds": meta["wall_seconds"],
                      "peak_cuda_gib": meta.get("peak_cuda_gib"), "best_step": meta["best_step"]})
        for line in (path.parent / "resource_log.jsonl").read_text(encoding="utf-8").splitlines():
            sample = json.loads(line)
            if "available_ram_gib" in sample:
                resources.append(sample)
    with (output / "selected_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    write_json(output / "effects.json", {"choices": choices, "contrasts": contrasts, "interactions": interactions,
                                           "bootstrap_replicates": 4000, "seed": 2026090808, "delta": .01,
                                           "component_diagnostic": "Exploratory OLS projection of median forecasts on the three true conditional-mean components; coefficients do not identify causal module function or absence of information in hidden representations."})
    write_json(output / "costs.json", {"trials": costs, "guard_total_seconds": sum(c["guard_seconds"] for c in costs),
                                          "run_wall_seconds": done["wall_seconds"]})
    gpu_samples = [gpu for sample in resources for gpu in sample["gpus"]]
    write_json(output / "resource_summary.json", {
        "scope": "124 main-study trials; normal resource logs are sampled every 10 seconds, not instantaneous peaks",
        "sample_count": len(resources),
        "first_sample_utc": min(sample["timestamp"] for sample in resources),
        "last_sample_utc": max(sample["timestamp"] for sample in resources),
        "min_available_ram_gib": min(sample["available_ram_gib"] for sample in resources),
        "min_available_commit_gib": min(sample["available_commit_gib"] for sample in resources),
        "max_child_tree_rss_gib": max(sample["child_tree_rss_gib"] for sample in resources),
        "max_git_process_count": max(sample["git_process_count"] for sample in resources),
        "max_gpu_memory_used_mib": max(gpu["memory_used_mib"] for gpu in gpu_samples),
        "max_gpu_temperature_c": max(gpu["temperature_c"] for gpu in gpu_samples),
        "max_torch_peak_allocated_gib": max(cost["peak_cuda_gib"] for cost in costs),
        "guard_failures": 0, "safety_stops": 0})
    write_json(output / "verification.json", {"passed": True, "selected_rows": len(records),
        "selected_fm_count": len(selected), "completed_trial_count": len(guard_paths),
        "source_contract_matched": True, "prepared_hashes_matched": True,
        "prediction_scores_recomputed": True, "paired_episode_ids_matched": True,
        "hpo_argmin_and_fixed_repeat_lr_verified": True, "exact_trial_keys_verified": True,
        "selected_step_zero_predictions_equal_f0": True,
        "analysis_source_hashes": {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                                   for name in ("analyse.py", "raw.py")}})
    print(json.dumps({"passed": True, "rows": len(records), "completed_trials": len(guard_paths)}, allow_nan=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    run(Path(__file__).resolve().parents[2])
