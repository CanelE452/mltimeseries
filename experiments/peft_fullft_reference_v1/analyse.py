"""Frozen study20 analysis: seed-average losses, common calibration, full costs."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time

for _variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_variable] = "2"

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.peft_external_gap_v1 import analyse as shared
from experiments.peft_fullft_reference_v1.contract import ROOT, STUDY, digest, read_contract, save_json

ARMS = ("F0", "SIMPLE", "HEAD_ONLY", "LORA", "FULL_FT")
EXPECTED_PARAMETERS = {"F0": 0, "HEAD_ONLY": 3653280, "LORA": 1206912, "FULL_FT": 119477664}
EFFECTS = {"FULL_FT_benefit_vs_LORA": ("LORA", "FULL_FT"),
           "LORA_benefit_vs_HEAD_ONLY": ("HEAD_ONLY", "LORA")}


def audit_selection(entries, selected, datasets, grids, seeds):
    identity = lambda row: (row["dataset"], row["arm"], row["lr"], row["seed"])
    expected = {(dataset, arm, lr, seed) for dataset in datasets for arm, rates in grids.items()
                for lr in rates for seed in seeds}
    if len(entries) != len(expected) or {identity(row) for row in entries} != expected:
        raise AssertionError("Incomplete or duplicate fit grid")
    if any(not np.isfinite(row["val_score"]) for row in entries):
        raise AssertionError("Nonfinite grid validation score")
    required, choices = set(), []
    for dataset in datasets:
        for arm, rates in grids.items():
            candidates = []
            for lr in rates:
                values = [next(row["val_score"] for row in entries if identity(row) == (dataset, arm, lr, seed)) for seed in seeds]
                candidates.append({"lr": lr, "mean_val_score": float(np.mean(values)), "seed_scores": values})
            best = min(candidates, key=lambda row: (row["mean_val_score"], row["lr"]))
            required.update((dataset, arm, best["lr"], seed) for seed in seeds)
            choices.append({"dataset": dataset, "arm": arm, "selected_lr": best["lr"],
                            "grid_boundary_selected": best["lr"] in (min(rates), max(rates)), "candidates": candidates})
    if len(selected) != len(required) or {identity(row) for row in selected} != required:
        raise AssertionError("Frozen selection disagrees with mean-seed V minimum / low-LR tie")
    return choices


def summarize_dataset(dataset, records, scale, quantiles, seeds, *, replicates=4000,
                      block_days=7, bootstrap_seed=2026090920, confidence=.95):
    expected = {(arm, seed) for arm in ("HEAD_ONLY", "LORA", "FULL_FT") for seed in seeds}
    expected |= {("F0", None), ("SIMPLE", None)}
    keys = [(record["arm"], record["seed"]) for record in records]
    if len(keys) != len(expected) or set(keys) != expected:
        raise AssertionError("Expected exactly three neural seeds and two deterministic references")
    reference = records[0]
    for split in ("C", "E"):
        target, origins = reference[split]["target"], reference[split]["origins"]
        if np.any(np.diff(origins) != 24):
            raise AssertionError("Bootstrap requires daily origins")
        for record in records:
            if not np.array_equal(record[split]["target"], target, equal_nan=True):
                raise AssertionError("Methods have different targets or missingness")
            if not np.array_equal(record[split]["origins"], origins):
                raise AssertionError("Methods have different origins")
            prediction = record[split]["prediction"]
            shared.forecast_arrays(prediction, target, scale, quantiles)
            if np.any(np.diff(prediction, axis=2) < 0):
                raise AssertionError("Saved primary predictions must already be SORT")
    horizon = reference["C"]["target"].shape[-1]
    if reference["C"]["origins"][-1] + horizon > reference["E"]["origins"][0]:
        raise AssertionError("Calibration and evaluation target windows overlap")
    rows, statistics, offsets = [], {}, {}
    for record in records:
        key = (record["arm"], record["seed"])
        correction = shared.qcal_offsets(record["C"]["prediction"], record["C"]["target"], quantiles)
        offsets[f"{key[0]}/{key[1]}"] = correction.tolist()
        statistics[key] = {}
        for procedure in ("SORT", "QCAL"):
            target = record["E"]["target"]
            prediction = record["E"]["prediction"]
            cross = record["E"].get("unsorted", prediction)
            if procedure == "QCAL":
                cross = prediction + correction[None, :, :, None]
                prediction = shared.apply_qcal(prediction, correction)
            details = shared.score_details(prediction, target, scale, quantiles, crossing_prediction=cross)
            _, sums, counts = shared.score_prediction(prediction, target, scale, quantiles)
            statistics[key][procedure] = (sums, counts)
            details["median_mse_raw"] = float(np.mean(np.asarray(details["median_mse_scaled_by_target"]) * np.asarray(scale) ** 2))
            row = {"dataset": dataset, "arm": key[0], "seed": key[1], "procedure": procedure, "split": "E",
                   **details, **record.get("metadata", {})}
            rows.append(row)
    arms = {}
    numeric = ("score", "coverage80", "width80_scaled", "median_mse_scaled", "median_mse_raw", "crossing_before_sort")
    for arm in ARMS:
        arms[arm] = {}
        for procedure in ("SORT", "QCAL"):
            group = sorted((row for row in rows if row["arm"] == arm and row["procedure"] == procedure),
                           key=lambda row: -1 if row["seed"] is None else row["seed"])
            values = [row["score"] for row in group]
            summary = {name + "_mean": float(np.mean([row[name] for row in group])) for name in numeric}
            summary.update(seed_values=values, seeds=[row["seed"] for row in group], seed_range=[min(values), max(values)],
                           deterministic=len(group) == 1, aggregation="mean of separately scored seed losses; not forecast ensemble")
            arms[arm][procedure] = summary
    for row in rows:
        denominator = arms["F0"][row["procedure"]]["score_mean"]
        row["improvement_over_f0"] = (denominator - row["score"]) / denominator if denominator > 0 else None
    weights = shared.moving_block_weights(len(reference["E"]["origins"]), block_days, replicates, bootstrap_seed)
    effects = {}
    for procedure in ("SORT", "QCAL"):
        effects[procedure] = {}
        f0_sums, counts = statistics[("F0", None)][procedure]
        for name, (left, right) in EFFECTS.items():
            metadata = {"left": left, "right": right, "formula": "(left loss - right loss) / F0 loss",
                        "unit": "ratio in F0 score units", "positive_means": f"{right} improves on {left}"}
            if shared.macro_score(f0_sums, counts) <= 0:
                effects[procedure][name] = {**metadata, "defined": False, "reason": "F0 score is zero"}
                continue
            lhs = np.stack([statistics[(left, seed)][procedure][0] for seed in seeds])
            rhs = np.stack([statistics[(right, seed)][procedure][0] for seed in seeds])
            for key in statistics:
                if not np.array_equal(statistics[key][procedure][1], counts):
                    raise AssertionError("Paired sufficient-statistic counts differ")
            effect = shared.paired_effect(lhs, rhs, f0_sums, counts, weights, confidence)
            effect.pop("decision")
            effect.pop("practical_threshold")
            effect.update(metadata, defined=True, seed_range=[min(effect["seed_values"]), max(effect["seed_values"])],
                          uncertainty="time blocks conditional on these three fitted seeds and frozen V selection",
                          multiple_comparison_status="four exploratory primary contrasts; no simultaneous error control")
            effects[procedure][name] = effect
    return {"dataset": dataset, "arms": arms, "effects": effects, "rows": rows, "qcal_offsets": offsets,
            "origins": {split: reference[split]["origins"].tolist() for split in ("C", "E")},
            "primary": "SORT", "secondary": "common C-only QCAL", "hard_success_gate": None,
            "bootstrap": {"replicates": replicates, "block_days": block_days, "seed": bootstrap_seed,
                          "confidence": confidence, "shared_weights_across_arms_and_seeds": True}}


def _same_number(actual, expected, label):
    if not np.isfinite(actual) or not np.isfinite(expected) or not np.isclose(actual, expected, rtol=2e-6, atol=1e-8):
        raise AssertionError(f"{label}: saved and recomputed values disagree")


def _cost_record(result, checkpoint_bytes, guard_seconds, selected):
    seconds = result["seconds"]
    return {"dataset": result["dataset"], "arm": result["arm"], "seed": result["seed"], "lr": result["lr"],
            "selected": selected, "best_step": result["best_step"], "val_score": result["val_score"],
            "trainable": result["trainable"], "checkpoint_bytes": checkpoint_bytes,
            "optimizer_seconds": seconds["optimizer"], "optimizer_seconds_per_step": result["optimizer_seconds_per_step"],
            "fit_seconds": seconds["total"], "guard_seconds": guard_seconds,
            "load_seconds": seconds["load"], "validation_seconds": seconds["validation"],
            "checkpoint_seconds": seconds["checkpoint_write_restore"],
            "restore_validation_seconds": seconds["restore_validation"],
            "peak_cuda_allocated_bytes": result["peak_cuda_allocated_bytes"],
            "peak_cuda_reserved_bytes": result["peak_cuda_reserved_bytes"],
            "rss_sample_max_bytes": result["rss_sample_max_bytes"]}


def cost_summary(records):
    output = {}
    for arm in ("HEAD_ONLY", "LORA", "FULL_FT"):
        group = [record for record in records if record["arm"] == arm]
        selected = [record for record in group if record["selected"]]
        if not group:
            raise AssertionError("Missing cost arm")
        output[arm] = {"fit_count": len(group), "selected_fit_count": len(selected),
                       "trainable": group[0]["trainable"],
                       "fit_seconds_sum": sum(row["fit_seconds"] for row in group),
                       "guard_seconds_sum": sum(row["guard_seconds"] for row in group),
                       "optimizer_seconds_sum": sum(row["optimizer_seconds"] for row in group),
                       "optimizer_seconds_per_step_mean": float(np.mean([row["optimizer_seconds_per_step"] for row in group])),
                       "checkpoint_bytes_sum": sum(row["checkpoint_bytes"] for row in group),
                       "checkpoint_bytes_median": float(np.median([row["checkpoint_bytes"] for row in group])),
                       "peak_cuda_allocated_bytes": max(row["peak_cuda_allocated_bytes"] for row in group),
                       "peak_cuda_reserved_bytes": max(row["peak_cuda_reserved_bytes"] for row in group),
                       "rss_sample_max_bytes": max(row["rss_sample_max_bytes"] for row in group),
                       "unselected_fit_seconds_sum": sum(row["fit_seconds"] for row in group if not row["selected"])}
    return {"all_neural_fit_count": len(records), "all_neural_fits": records, "by_arm": output,
            "total_neural_fit_seconds": sum(row["fit_seconds"] for row in records),
            "total_neural_guard_seconds": sum(row["guard_seconds"] for row in records),
            "guard_time_includes_fit_time": True, "rss_is_sampled_not_continuous_peak": True,
            "head_execution": "direct forward each update; no head-only feature cache",
            "parameter_count_is_not_runtime": True}


def execute(root=ROOT):
    root = Path(root).resolve()
    run = root / "runs" / STUDY
    output = root / "results" / STUDY
    contract_path = run / "study_contract.json"
    started, checked = time.perf_counter(), {}
    contract = read_contract(contract_path, verify="core")

    def checked_path(path, expected=None):
        path = Path(path)
        actual = digest(path)
        if expected is not None and actual != expected:
            raise AssertionError(f"Frozen artifact hash mismatch: {path}")
        checked[str(path)] = actual
        return path

    def document(path, expected=None):
        return json.loads(checked_path(path, expected).read_text(encoding="utf-8-sig"))

    def guard(path, expected=None):
        status = document(Path(path) / "status.json", expected)
        if not status.get("completed") or status.get("returncode") != 0 or status.get("reasons"):
            raise AssertionError("Analysis requires completed successful execution guards")
        return float(status["elapsed_seconds"])

    def arrays(path, expected):
        with np.load(checked_path(path, expected), allow_pickle=False) as archive:
            return {key: archive[key].copy() for key in archive.files}

    contract_hash = digest(contract_path)
    checked[str(contract_path)] = contract_hash
    completed = document(run / "completed.json")
    selection = document(run / "selection.json", completed["selection_sha256"])
    if (not completed.get("completed") or not selection.get("completed") or not selection.get("global_choices_frozen")
            or completed["contract_sha256"] != contract_hash or selection["contract_sha256"] != contract_hash
            or selection.get("C_or_E_used_for_selection") is not False):
        raise AssertionError("Complete forecasts and frozen validation-only selection are required")
    analysis_contract = document(run / "analysis_contract.json")
    if analysis_contract["contract_sha256"] != contract_hash:
        raise AssertionError("Analysis contract differs from trained study")
    for source, sha in analysis_contract["source_hashes"].items():
        checked_path(root / source, sha)
    read_contract(contract_path, verify="full")
    settings, seeds = contract["settings"], contract["settings"]["seeds"]
    choices = audit_selection(selection["all_trials"], selection["selected"], contract["datasets"], settings["lr_grids"], seeds)
    selected_keys = {(row["dataset"], row["arm"], row["lr"], row["seed"]) for row in selection["selected"]}
    expected_forecasts = {(dataset, "F0", seeds[0]) for dataset in contract["datasets"]}
    expected_forecasts |= {(row["dataset"], row["arm"], row["seed"]) for row in selection["selected"]}
    forecast_keys = [(row["dataset"], row["arm"], row["seed"]) for row in completed["forecasts"]]
    if len(forecast_keys) != len(expected_forecasts) or set(forecast_keys) != expected_forecasts:
        raise AssertionError("Exactly twenty selected neural forecasts are required")
    # No prepared target arrays are opened before the complete forecast/selection audit.
    panels = {}
    for dataset, entry in contract["datasets"].items():
        fit = arrays(root / entry["fit_data_path"], entry["fit_data_sha256"])
        future = arrays(root / entry["holdout_data_path"], entry["holdout_data_sha256"])
        for field in ("fit_std", "target_indices", "quantiles", "context", "horizon"):
            if not np.array_equal(fit[field], future[field]):
                raise AssertionError("Prepared fit/holdout coordinate mismatch")
        target_indices = fit["target_indices"]
        panel = {"scale": fit["fit_std"][target_indices].astype(np.float64), "quantiles": fit["quantiles"]}
        for letter, source, split, count in (("V", fit, "val", 30), ("C", future, "cal", 20), ("E", future, "eval", 80)):
            origins = source[split + "_origins"]
            if len(origins) != count or np.any(np.diff(origins) != 24):
                raise AssertionError("Prepared split has unexpected origin count/stride")
            target = np.stack([source["target_values"][o:o + 48, target_indices].T for o in origins])
            if target.shape != (count, 2, 48):
                raise AssertionError("Prepared truth shape differs from H48 / two targets")
            panel[letter] = {"target": target, "origins": origins}
        panels[dataset] = panel

    def verify_prediction(saved, panel, letter, simple=False):
        key = "prediction" if simple else "predictions"
        prediction, target, origins = saved[key], saved["target"], saved["origins"]
        if not np.array_equal(target, panel[letter]["target"], equal_nan=True) or not np.array_equal(origins, panel[letter]["origins"]):
            raise AssertionError("Saved prediction targets/origins differ from prepared source")
        score, sums, counts = shared.score_prediction(prediction, target, panel["scale"], panel["quantiles"])
        unsorted = saved.get("unsorted_predictions", prediction)
        if not np.array_equal(np.sort(unsorted, axis=2), prediction):
            raise AssertionError("Saved SORT differs from native unsorted forecast")
        if "loss_sums" in saved and (not np.allclose(saved["loss_sums"], sums, rtol=2e-6, atol=1e-8)
                                     or not np.array_equal(saved["valid_counts"], counts)):
            raise AssertionError("Saved prediction sufficient statistics differ")
        return {"prediction": prediction, "target": target, "origins": origins, "unsorted": unsorted}, score

    fits, fit_costs = {}, []
    for entry in selection["all_trials"]:
        path = root / entry["fit_dir"]
        result = document(path / "result.json", entry["fit_result_sha256"])
        dataset, arm, lr, seed = (entry[key] for key in ("dataset", "arm", "lr", "seed"))
        if (result.get("completed") is not True or result.get("smoke") is not False or result["stage"] != "fit"
                or result["contract_sha256"] != contract_hash or result["holdout_file_opened"] is not False
                or (result["dataset"], result["arm"], result["lr"], result["seed"]) != (dataset, arm, lr, seed)
                or result["steps_completed"] != 200 or result["trainable"] != EXPECTED_PARAMETERS[arm]):
            raise AssertionError("Fit identity/update scope differs from registered grid")
        checkpoint = checked_path(path / "best_trainable.pt", result["checkpoint_sha256"])
        if result["checkpoint_sha256"] != entry["checkpoint_sha256"]:
            raise AssertionError("Selection checkpoint linkage differs")
        prediction = arrays(path / "Vpredictions.npz", result["Vpredictions_sha256"])
        _, score = verify_prediction(prediction, panels[dataset], "V")
        _same_number(score, result["val_score"], "fit validation")
        _same_number(score, entry["val_score"], "selection validation")
        history = result["history"]
        if [row["step"] for row in history] != [0, 40, 80, 120, 160, 200]:
            raise AssertionError("Validation checkpoint schedule differs")
        best = min(history, key=lambda row: (row["val_score"], row["step"]))
        if best["step"] != result["best_step"]:
            raise AssertionError("Selected checkpoint is not first strict validation minimum")
        _same_number(best["val_score"], score, "restored best checkpoint")
        guard_seconds = guard(root / entry["guard_path"], entry["guard_sha256"])
        key = (dataset, arm, lr, seed)
        cost = _cost_record(result, checkpoint.stat().st_size, guard_seconds, key in selected_keys)
        fit_costs.append(cost)
        fits[str(path.resolve())] = (result, cost)

    records = {dataset: [] for dataset in contract["datasets"]}
    forecast_costs = []
    for entry in completed["forecasts"]:
        path = root / entry["path"]
        result = document(path / "result.json", entry["result_sha256"])
        if (not result["completed"] or result["contract_sha256"] != contract_hash or result["optimizer_steps"] != 0
                or result["selection_sha256"] != digest(run / "selection.json") or not result["audits"]["model_unchanged"]
                or (result["dataset"], result["arm"], result["seed"]) != (entry["dataset"], entry["arm"], entry["seed"])):
            raise AssertionError("Forecast does not match immutable selected trial")
        dataset, arm = entry["dataset"], entry["arm"]
        metadata = {"path": str(path), "trainable": result["trainable"], "best_step": result["best_step"]}
        if arm != "F0":
            fit, cost = fits[str(Path(result["fit_dir"]).resolve())]
            if fit["checkpoint_sha256"] != result["checkpoint_sha256"] or (fit["dataset"], fit["arm"], fit["seed"]) != (dataset, arm, result["seed"]):
                raise AssertionError("Forecast loaded a different fitted model")
            metadata.update({key: cost[key] for key in ("lr", "checkpoint_bytes", "optimizer_seconds_per_step", "fit_seconds",
                                                       "peak_cuda_allocated_bytes", "peak_cuda_reserved_bytes", "rss_sample_max_bytes")})
        else:
            metadata.update(lr=None, checkpoint_bytes=0, optimizer_seconds_per_step=0., fit_seconds=0.,
                            peak_cuda_allocated_bytes=result["peak_cuda_allocated_bytes"],
                            peak_cuda_reserved_bytes=result["peak_cuda_reserved_bytes"], rss_sample_max_bytes=result["rss_sample_max_bytes"])
        record = {"arm": arm, "seed": None if arm == "F0" else result["seed"], "metadata": metadata}
        for letter, split in (("C", "cal"), ("E", "eval")):
            saved = arrays(path / f"{letter}predictions.npz", result[f"{letter}predictions_sha256"])
            record[letter], score = verify_prediction(saved, panels[dataset], letter)
            _same_number(score, result[f"{split}_score"], "forecast score")
        records[dataset].append(record)
        forecast_costs.append({"dataset": dataset, "arm": arm, "seed": result["seed"], "seconds": result["seconds"],
                               "guard_seconds": guard(root / entry["guard_path"])})

    simple_costs = {}
    for dataset in contract["datasets"]:
        directory = run / "baselines" / dataset
        entry = selection["baselines"][dataset]
        choice = document(directory / "selection.json", entry["selection_sha256"])
        model_path = checked_path(directory / "model.npz", entry["model_sha256"])
        result = document(directory / "result.json")
        forecast = document(directory / "forecast_result.json")
        if (choice["model_sha256"] != entry["model_sha256"] or result["selection_sha256"] != entry["selection_sha256"]
                or forecast["global_selection_sha256"] != digest(run / "selection.json")
                or not forecast["completed"] or forecast["contract_sha256"] != contract_hash
                or choice["uses_calibration_labels"] is not False or choice["uses_evaluation_labels"] is not False):
            raise AssertionError("Simple reference provenance differs")
        saved_v = arrays(directory / "V_predictions.npz", choice["V_predictions_sha256"])
        candidate_scores = []
        for index, prediction in enumerate(saved_v["candidate_predictions"]):
            _, score = verify_prediction({"predictions": prediction, "target": saved_v["val_target"],
                                           "origins": saved_v["val_origins"]}, panels[dataset], "V")
            _same_number(score, choice["candidates"][index]["validation"]["score"], "simple candidate V")
            candidate_scores.append(score)
        selected_index = min(range(len(candidate_scores)), key=lambda index: (candidate_scores[index], index))
        if len(candidate_scores) != 4 or selected_index != choice["selected_index"]:
            raise AssertionError("Simple reference V selection differs")
        record = {"arm": "SIMPLE", "seed": None, "metadata": {"path": str(directory), "trainable": None,
                  "lr": None, "best_step": None, "checkpoint_bytes": model_path.stat().st_size,
                  "optimizer_seconds_per_step": None, "fit_seconds": result["wall_seconds"],
                  "peak_cuda_allocated_bytes": 0, "peak_cuda_reserved_bytes": 0, "rss_sample_max_bytes": None}}
        for letter in ("C", "E"):
            pred_entry = forecast["predictions"][letter]
            saved = arrays(Path(pred_entry["path"]), pred_entry["sha256"])
            record[letter], _ = verify_prediction(saved, panels[dataset], letter, simple=True)
        records[dataset].append(record)
        simple_costs[dataset] = {"candidate_count": 4, "ridge_oof_fits": 6, "ridge_final_fits": 3,
                                 "wall_seconds": result["wall_seconds"], "model_bytes": model_path.stat().st_size,
                                 "selected_index": selected_index, "selected_method": choice["selected_candidate"]["method"],
                                 "selected_lambda": choice["selected_candidate"]["lambda"]}
    datasets = {dataset: summarize_dataset(dataset, records[dataset], panel["scale"], panel["quantiles"], seeds,
                  replicates=settings["bootstrap_replicates"], block_days=settings["block_days"],
                  bootstrap_seed=settings["bootstrap_seed"], confidence=settings["confidence"])
                for dataset, panel in panels.items()}
    costs = cost_summary(fit_costs)
    costs.update(forecasts=forecast_costs, simple=simple_costs,
                 simple_fit_guard_seconds=guard(run / "guards" / "baseline_fit"),
                 simple_forecast_guard_seconds=guard(run / "guards" / "baseline_forecast"),
                 smoke_costs_included_in_54_fits=False, previous_studies_costs_included=False)
    smoke = document(run / "smoke_completed.json")
    costs["smoke_guard_seconds_separate"] = sum(guard(root / entry["guard_path"]) for entry in smoke["trials"])
    rows = [row for dataset in datasets.values() for row in dataset.pop("rows")]
    fields = ["dataset", "arm", "seed", "procedure", "split", "score", "improvement_over_f0", "coverage80",
              "width80_scaled", "median_mse_scaled", "median_mse_raw", "crossing_before_sort", "lr", "best_step",
              "trainable", "checkpoint_bytes", "optimizer_seconds_per_step", "fit_seconds", "peak_cuda_allocated_bytes",
              "peak_cuda_reserved_bytes", "rss_sample_max_bytes", "path"]
    # Verify exactly the bytes used before emitting a completed analytical result.
    for path, sha in checked.items():
        if digest(path) != sha:
            raise AssertionError(f"Input changed during analysis: {path}")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "metrics.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    summary = {"completed": True, "study": STUDY, "contract_sha256": contract_hash,
               "selection_sha256": digest(run / "selection.json"), "datasets": datasets, "selection_audit": choices,
               "costs": costs, "metric_rows": len(rows), "full_metric_records": rows,
               "audit": {"all_neural_grid_fits": len(fit_costs), "selected_neural_forecasts": len(forecast_costs),
                         "simple_forecasts": 2, "common_prepared_targets_verified": True, "selection_recomputed": True,
                         "input_hashes": checked, "metrics_csv_sha256": digest(output / "metrics.csv")},
               "wall_seconds_before_json_write": time.perf_counter() - started,
               "interpretation": {"new_method_success": False, "full_ft_is_performance_upper_bound": False,
                                  "pretraining_overlap": "UNKNOWN", "primary": "SORT", "secondary": "common QCAL",
                                  "seed_average_is_not_ensemble": True, "hard_success_gate": None,
                                  "CI_scope": "four exploratory primary time-block intervals conditional on three fitted seeds and V selection"}}
    save_json(output / "summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    result = execute(args.root)
    print(json.dumps({"completed": result["completed"], "metric_rows": result["metric_rows"],
                      "wall_seconds": result["wall_seconds_before_json_write"]}))
