"""Audit saved contracts and actual execution artifacts before reporting S1."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from .run_study import RUN
from .analyse import select_raw_result


def verify_selected_report(rows, effects, contract, selection, trials, raw_selected, run_root=RUN):
    expected = set()
    for panel in contract["panels"]:
        expected.update((panel, method, 0) for method in ["F0", "RAW_VARX_RIDGE"])
        expected.update((panel, method, seed) for method in contract["grids"] for seed in contract["seeds"])
    row_map = {}
    for row in rows:
        key = (row["panel"], row["method"], int(row["seed"]))
        assert key not in row_map, f"Duplicate selected CSV key: {key}"
        row_map[key] = row
    actual = set(row_map)
    assert actual == expected, f"Selected CSV keys differ: missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
    selected_metrics = {}
    for key, row in row_map.items():
        panel, method, seed = key
        if method == "RAW_VARX_RIDGE":
            metric = raw_selected[panel]["metric"]
            source = Path(raw_selected[panel]["choice"]["source_path"])
            lr = metric["selected_ridge"]
        else:
            lr = 0 if method == "F0" else selection[f"{panel}/{method}"]["lr"]
            matching = [metric for metric in trials if metric["panel"] == panel and metric["method"] == method
                        and metric["seed"] == seed and metric["lr"] == lr]
            assert len(matching) == 1, f"Expected one selected trial for {key}, found {len(matching)}"
            metric = matching[0]
            source = Path(run_root) / "trials" / metric["name"]
        assert float(row["lr"]) == lr, f"Selected CSV LR mismatch: {key}"
        assert Path(row["source_path"]).resolve() == source.resolve(), f"Selected CSV source mismatch: {key}"
        assert int(row["selected_step"]) == metric["selected_step"], f"Selected CSV step mismatch: {key}"
        for score in ["val_score", "eval_score"]:
            assert np.isfinite(float(row[score])) and np.isclose(
                float(row[score]), metric[score], rtol=2e-5, atol=2e-6
            ), f"Selected CSV {score} mismatch: {key}"
        selected_metrics[key] = metric
    expected_effects = {(panel, method) for panel in contract["panels"] for method in ["OFF_LORA", "FULL"]}
    effect_map = {}
    for effect in effects:
        key = (effect["panel"], effect["internal"])
        assert key not in effect_map, f"Duplicate effect key: {key}"
        effect_map[key] = effect
    assert set(effect_map) == expected_effects, "Missing or extra effect keys in final report"
    seeds = sorted(contract["seeds"])
    assert seeds == [0, 1, 2], "This registered S1 report requires all three selected seeds"
    for (panel, internal), effect in effect_map.items():
        assert effect["paired_seed_ids"] == seeds and effect["paired_seed_count"] == len(seeds), f"Incomplete seed pairing: {panel}/{internal}"
        head = min(["H_LIN", "H_MLP", "H_FULL"], key=lambda method: selection[f"{panel}/{method}"]["val_score"])
        assert effect["head_selected_on_validation"] == head, f"Effect uses wrong validation-selected head: {panel}/{internal}"
        baseline = selected_metrics[(panel, "F0", 0)]["eval_score"]
        assert np.isfinite(baseline) and baseline > 0
        deltas = np.array([(selected_metrics[(panel, head, seed)]["eval_score"]
                            - selected_metrics[(panel, internal, seed)]["eval_score"]) / baseline for seed in seeds])
        assert np.asarray(effect["seed_deltas"]).shape == deltas.shape
        assert np.allclose(effect["seed_deltas"], deltas, rtol=2e-5, atol=2e-6), f"Stale per-seed effects: {panel}/{internal}"
        assert np.isclose(effect["delta_over_F0"], deltas.mean(), rtol=2e-5, atol=2e-6), f"Stale mean effect: {panel}/{internal}"
    return {"passed": True, "selected_csv_rows": len(rows), "exact_selected_keys_verified": True,
            "selected_sources_and_scores_verified": True, "effects_verified": len(effects),
            "paired_seed_ids_verified": seeds}


def verify_optimization_boundaries(trials, contract, selection):
    boundaries = []
    for panel in contract["panels"]:
        for method, grid in contract["grids"].items():
            candidates = [metric for metric in trials if metric["panel"] == panel and metric["method"] == method and metric["seed"] == 0]
            assert sorted(metric["lr"] for metric in candidates) == sorted(grid), f"Executed LR grid differs from contract: {panel}/{method}"
            # Reproduce the runner's stable tie rule using the registered grid order.
            chosen = min(candidates, key=lambda metric: (metric["val_score"], grid.index(metric["lr"])))
            actual = selection[f"{panel}/{method}"]
            assert chosen["lr"] == actual["lr"] and chosen["val_score"] == actual["val_score"]
            selected_seeds = []
            for seed in contract["seeds"]:
                repeats = [metric for metric in trials if metric["panel"] == panel and metric["method"] == method
                           and metric["seed"] == seed and metric["lr"] == chosen["lr"]]
                assert len(repeats) == 1, f"Expected one selected seed trial: {panel}/{method}/{seed}"
                if seed != 0:
                    assert sum(metric["panel"] == panel and metric["method"] == method and metric["seed"] == seed for metric in trials) == 1
                metric = repeats[0]
                selected_seeds.append({"seed": seed, "selected_step": metric["selected_step"],
                                       "step_at_budget": metric["selected_step"] == contract["steps"]})
            boundaries.append({"panel": panel, "method": method, "lr": chosen["lr"],
                               "registered_grid": grid, "lr_at_grid_boundary": chosen["lr"] in [min(grid), max(grid)],
                               "selected_step": chosen["selected_step"], "selected_step_scope": "HPO seed 0",
                               "step_at_budget": chosen["selected_step"] == contract["steps"],
                               "selected_seeds": selected_seeds,
                               "any_seed_at_budget": any(item["step_at_budget"] for item in selected_seeds),
                               "rescue_performed": False})
    return boundaries


def main():
    contract = json.loads((RUN / "contract.json").read_text(encoding="utf-8"))
    selection = json.loads((RUN / "selection.json").read_text(encoding="utf-8"))
    progress = json.loads((RUN / "progress.json").read_text(encoding="utf-8"))
    assert progress["status"] == "completed" and progress["stage"] == "s1"
    for name, digest in contract["source_sha256"].items():
        assert hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest() == digest, name
    files = sorted((RUN / "trials").glob("*/metrics.json"))
    assert len(files) == 62, f"Expected 60 adaptation trials + 2 F0, got {len(files)}"
    result_dir = Path(__file__).resolve().parents[2] / "results/peft_adaptation_scope_v1"
    # Analysis must run first so this checks the source actually used in the reported comparison.
    with (result_dir / "selected_results.csv").open(encoding="utf-8", newline="") as handle:
        comparison_rows = list(csv.DictReader(handle))
    effects = json.loads((result_dir / "effects.json").read_text(encoding="utf-8"))
    saved_raw_choices = json.loads((result_dir / "raw_selection.json").read_text(encoding="utf-8"))
    raw_audits, raw_selected = {}, {}
    hashes, samples, resources, trials = {}, {}, [], []
    for panel in contract["panels"]:
        path = RUN / "prepared" / f"{panel}.npz"
        hashes[panel] = hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path, allow_pickle=False) as data:
            samples[panel] = data["train_origins"].astype(np.int64)
            expected_origins = {split: data[split + "_origins"].copy() for split in ["val", "eval"]}
            for split in ["val", "eval"]:
                origins = data[split + "_origins"]
                assert np.all(np.diff(origins) == int(data["horizon"]))
        chosen_raw = select_raw_result(panel)
        candidate_audits, raw_metrics = [], {}
        for candidate in chosen_raw["candidates"]:
            raw_folder = Path(candidate["source_path"])
            raw = json.loads((raw_folder / "metrics.json").read_text(encoding="utf-8"))
            assert raw["completed"]
            assert raw["train_origins_sha256"] == hashlib.sha256(samples[panel].tobytes()).hexdigest()
            assert raw["origin_counts"]["train"] == len(samples[panel])
            assert all(fold["max_train_label_end_exclusive"] <= fold["first_oof_origin"] for fold in raw["oof_folds"])
            assert all(raw["audits"][key] for key in ["oof_future_label_exclusion", "oof_preprocessing_fit_only",
                                                      "all_fm_train_origins_used", "lambda_selected_on_validation_only"])
            assert len(raw["lambda_selection"]) == len(raw["rawknobs"]["lambda_grid"])
            best_candidate = min(raw["lambda_selection"], key=lambda entry: entry["val_score"])
            assert best_candidate["ridge"] == raw["selected_ridge"] and best_candidate["val_score"] == raw["val_score"]
            with np.load(raw_folder / "predictions.npz", allow_pickle=False) as predictions:
                for split in ["val", "eval"]:
                    assert np.array_equal(predictions[split + "_origins"], expected_origins[split])
                    assert len(predictions[split + "_pred"]) == len(expected_origins[split])
            raw_metrics[raw_folder.parent.name] = raw
            candidate_audits.append({**candidate, "completed_verified": True, "train_origin_hash_verified": True,
                                     "prediction_origin_order_verified": True, "oof_purge_verified": True})
        if "raw_rescue" in raw_metrics:
            rescue_contract = json.loads((RUN / "raw_rescue/contract.json").read_text(encoding="utf-8"))
            assert raw_metrics["raw_rescue"]["rawknobs"]["lambda_grid"] == rescue_contract["grid"]
            assert raw_metrics["raw_rescue"]["oof_folds"] == raw_metrics["raw"]["oof_folds"]
            assert rescue_contract["prepared_sha256"][panel] == hashes[panel]
            for name, digest in rescue_contract["source_sha256"].items():
                assert hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest() == digest
            for name, digest in rescue_contract["original_results"][panel]["sha256"].items():
                assert hashlib.sha256((RUN / "raw" / panel / name).read_bytes()).hexdigest() == digest
        raw_rows = [row for row in comparison_rows if row["panel"] == panel and row["method"] == "RAW_VARX_RIDGE"]
        assert len(raw_rows) == 1
        row = raw_rows[0]
        assert Path(row["source_path"]).resolve() == Path(chosen_raw["source_path"]).resolve()
        assert row["raw_rescue_used"] == str(chosen_raw["raw_rescue_used"])
        assert row["penaltyboundary"] == str(chosen_raw["penaltyboundary"])
        assert float(row["lr"]) == chosen_raw["selected_ridge"]
        assert np.isclose(float(row["val_score"]), chosen_raw["val_score"], rtol=1e-12, atol=1e-12)
        assert int(row["raw_source_candidate_count"]) == chosen_raw["source_candidate_count"]
        assert int(row["raw_total_candidates_executed"]) == chosen_raw["total_candidates_executed"]
        assert saved_raw_choices[panel] == chosen_raw
        raw_selected[panel] = {"choice": chosen_raw, "metric": raw_metrics[Path(chosen_raw["source_path"]).parent.name]}
        raw_audits[panel] = {"selected_source": chosen_raw["source_path"],
                             "selected_on_validation_only": True, "analysis_csv_source_verified": True,
                             "raw_rescue_used": chosen_raw["raw_rescue_used"],
                             "penaltyboundary": chosen_raw["penaltyboundary"], "candidates": candidate_audits,
                             "original_and_rescue_preserved": "raw_rescue" in raw_metrics,
                             "total_candidates_executed": chosen_raw["total_candidates_executed"]}
    for file in files:
        metric = json.loads(file.read_text(encoding="utf-8"))
        panel = file.parent.name.split("_", 1)[0]
        assert metric["completed"] and not metric["smoke"], file
        assert metric["data_sha256"] == hashes[panel], file
        assert metric["effective_groups"] == 8 and metric["micro_groups"] == 4
        assert metric["autocast_weight_cache"] is False
        assert metric["audits"]["zero_update_identity"]
        assert metric["audits"]["frozen_after_training_verified"]
        assert metric["audits"]["best_checkpoint_reload_verified"]
        expected_steps = 0 if metric["method"] == "F0" else contract["steps"]
        assert metric["steps_completed"] == expected_steps
        indices = np.random.default_rng(metric["seed"]).integers(len(samples[panel]), size=(expected_steps, 8))
        assert metric["sampler_hash"] == hashlib.sha256(samples[panel][indices].tobytes()).hexdigest()
        guard = json.loads((file.parent / "guard/status.json").read_text())
        assert guard["completed"] is True and guard["returncode"] == 0, file
        assert not (file.parent / "failure.json").exists()
        assert not (file.parent / "guard/safety_stop.json").exists()
        resources.extend(json.loads(line) for line in (file.parent / "guard/resource_log.jsonl").read_text().splitlines() if "available_ram_gib" in line)
        trials.append({"panel": panel, "name": file.parent.name, **metric})
    boundaries = verify_optimization_boundaries(trials, contract, selection)
    selected_report_audit = verify_selected_report(comparison_rows, effects, contract, selection,
                                                  trials, raw_selected)
    gpu_samples = [g for resource in resources for g in resource["gpus"]]
    report = {"passed": True, "completed_trials": len(trials), "adaptation_trials": 60,
              "data_sha256": hashes, "resource_samples": len(resources),
              "min_available_ram_gib": min(x["available_ram_gib"] for x in resources),
              "min_available_commit_gib": min(x["available_commit_gib"] for x in resources),
              "max_git_process_count": max(x["git_process_count"] for x in resources),
              "max_child_tree_rss_gib": max(x["child_tree_rss_gib"] for x in resources),
              "max_gpu_sample_memory_mib": max(x["memory_used_mib"] for x in gpu_samples),
              "max_gpu_sample_temperature_c": max(x["temperature_c"] for x in gpu_samples),
              "max_torch_peak_allocated_gib": max(x["peak_vram_gib"] for x in trials),
              "total_trial_wall_seconds": sum(x["wall_seconds"] for x in trials),
              "optimization_boundaries": boundaries, "raw_result_audits": raw_audits,
              "selected_report_audit": selected_report_audit}
    destination = Path(__file__).resolve().parents[2] / "results/peft_adaptation_scope_v1/verification.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
