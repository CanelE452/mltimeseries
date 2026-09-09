import importlib
import importlib.util
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


SEEDS = [20000, 20001, 20002]
Q = np.array([.01, .05, .10, .15, .20, .25, .30, .35, .40, .45,
              .50, .55, .60, .65, .70, .75, .80, .85, .90, .95, .99])
EXPECTED_COUNTS = {"train": 90, "val": 30, "cal": 20, "eval": 80}
EXPECTED_BOUNDARY_INDICES = {
    "precontext": [0, 336],
    "train": [336, 2520],
    "embargo_train_val": [2520, 2568],
    "val": [2568, 3312],
    "embargo_val_cal": [3312, 3360],
    "cal": [3360, 3864],
    "eval": [3864, 5808],
}
EXPECTED_BOUNDARIES = {
    "bike": {
        "precontext": ["2012-01-16T00:00:00", "2012-01-30T00:00:00"],
        "train": ["2012-01-30T00:00:00", "2012-04-30T00:00:00"],
        "embargo_train_val": ["2012-04-30T00:00:00", "2012-05-02T00:00:00"],
        "val": ["2012-05-02T00:00:00", "2012-06-02T00:00:00"],
        "embargo_val_cal": ["2012-06-02T00:00:00", "2012-06-04T00:00:00"],
        "cal": ["2012-06-04T00:00:00", "2012-06-25T00:00:00"],
        "eval": ["2012-06-25T00:00:00", "2012-09-14T00:00:00"],
    },
    "household": {
        "precontext": ["2008-01-01T00:00:00", "2008-01-15T00:00:00"],
        "train": ["2008-01-15T00:00:00", "2008-04-15T00:00:00"],
        "embargo_train_val": ["2008-04-15T00:00:00", "2008-04-17T00:00:00"],
        "val": ["2008-04-17T00:00:00", "2008-05-18T00:00:00"],
        "embargo_val_cal": ["2008-05-18T00:00:00", "2008-05-20T00:00:00"],
        "cal": ["2008-05-20T00:00:00", "2008-06-10T00:00:00"],
        "eval": ["2008-06-10T00:00:00", "2008-08-30T00:00:00"],
    },
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.write_text(json.dumps(value), encoding="utf-8")
    return sha(path)


def fixture():
    rows = []
    for arm, values in (("F0", [2.]), ("SIMPLE", [1.5]), ("HEAD_ONLY", [2., 2., 2.]),
                        ("LORA", [-1., 1., 3.]), ("FULL_FT", [.5, .5, .5])):
        for index, value in enumerate(values):
            record = {"arm": arm, "seed": None if len(values) == 1 else SEEDS[index]}
            for split, start in (("C", 1000), ("E", 2000)):
                prediction = np.full((8, 2, 21, 2), value)
                record[split] = {"prediction": prediction, "unsorted": prediction.copy(),
                                 "target": np.zeros((8, 2, 2)), "origins": start + np.arange(8) * 24}
            rows.append(record)
    return rows


def finished_tree(root):
    study = "peft_fullft_reference_v3"
    run = root / "runs" / study
    def js(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        return write_json(path, value)
    def status(path):
        js(path / "status.json", {"completed": True, "returncode": 0, "reasons": [], "elapsed_seconds": 1.2})
        return sha(path / "status.json")
    counts = {"HEAD_ONLY": 3653280, "LORA": 1206912, "FULL_FT": 119477664}
    grids = {arm: [.01, .1, 1.] for arm in counts}
    settings = {"seeds": SEEDS, "lr_grids": grids, "bootstrap_replicates": 20,
                "block_days": 7, "bootstrap_seed": 2026090920, "confidence": .95}
    prior_audit = root / "runs" / "peft_fullft_reference_v1" / "prepared" / "independent_fit_audit.json"
    js(prior_audit, {"completed": True, "passed": True, "datasets": {
        dataset: {"boundaries": EXPECTED_BOUNDARIES[dataset], "origin_counts": EXPECTED_COUNTS}
        for dataset in ("bike", "household")
    }})
    audit_key = str(prior_audit.relative_to(root))
    contract = {"settings": settings, "datasets": {},
                "independent_fit_audit_path": audit_key,
                "independent_fit_audit_sha256": sha(prior_audit),
                "protected_hashes": {audit_key: sha(prior_audit)}}
    origins = {"V": 2568 + np.arange(30) * 24, "C": 3360 + np.arange(20) * 24,
               "E": 3864 + np.arange(80) * 24}
    common = dict(fit_std=np.ones(2), target_indices=np.array([0, 1]), quantiles=Q, context=np.array(336), horizon=np.array(48))
    for dataset in ("bike", "household"):
        directory = run / "prepared"
        directory.mkdir(parents=True, exist_ok=True)
        fit_path, future = directory / f"{dataset}_fit.npz", directory / f"{dataset}_holdout.npz"
        np.savez_compressed(fit_path, target_values=np.zeros((3312, 2)), val_origins=origins["V"], **common)
        np.savez_compressed(future, target_values=np.zeros((5808, 2)), cal_origins=origins["C"], eval_origins=origins["E"], **common)
        contract["datasets"][dataset] = {"fit_data_path": str(fit_path.relative_to(root)), "fit_data_sha256": sha(fit_path),
                                          "holdout_data_path": str(future.relative_to(root)), "holdout_data_sha256": sha(future),
                                          "origin_counts": EXPECTED_COUNTS,
                                          "boundary_indices": EXPECTED_BOUNDARY_INDICES,
                                          "boundaries": EXPECTED_BOUNDARIES[dataset]}
    contract_hash = js(run / "study_contract.json", contract)
    js(run / "analysis_contract.json", {"contract_sha256": contract_hash, "source_hashes": {}})
    def value(arm, seed):
        return {"F0": 2., "HEAD_ONLY": 2., "LORA": [-1., 1., 3.][SEEDS.index(seed)], "FULL_FT": .5}[arm]
    def saved_prediction(path, letter, point):
        n = len(origins[letter])
        pred = np.full((n, 2, 21, 48), point)
        target = np.zeros((n, 2, 48))
        np.savez_compressed(path, predictions=pred, unsorted_predictions=pred, target=target, origins=origins[letter],
                            loss_sums=np.full((n, 2), abs(point) * 21 * 48), valid_counts=np.full((n, 2), 21 * 48))
        return sha(path)
    entries, selected, baselines = [], [], {}
    for dataset in contract["datasets"]:
        for arm in grids:
            for lr in grids[arm]:
                for seed in SEEDS:
                    path = run / "trials" / dataset / arm / f"{lr}_{seed}"
                    path.mkdir(parents=True)
                    checkpoint = path / "best_trainable.pt"
                    checkpoint.write_bytes(b"synthetic checkpoint")
                    guard_path = run / "guards" / "trials" / dataset / arm / f"{lr}_{seed}"
                    guard_hash = status(guard_path)
                    score = abs(value(arm, seed))
                    result = {"completed": True, "smoke": False, "stage": "fit", "dataset": dataset, "arm": arm,
                              "lr": lr, "seed": seed, "contract_sha256": contract_hash, "holdout_file_opened": False,
                              "steps_completed": 200, "best_step": 0, "trainable": counts[arm], "val_score": score,
                              "checkpoint_sha256": sha(checkpoint), "Vpredictions_sha256": saved_prediction(path / "Vpredictions.npz", "V", value(arm, seed)),
                              "history": [{"step": step, "val_score": score} for step in (0, 40, 80, 120, 160, 200)],
                              "seconds": {"optimizer": .4, "total": 1., "load": .2, "validation": .2,
                                          "checkpoint_write_restore": .1, "restore_validation": .1},
                              "optimizer_seconds_per_step": .002, "peak_cuda_allocated_bytes": 2000000,
                              "peak_cuda_reserved_bytes": 3000000, "rss_sample_max_bytes": 4000000}
                    result_hash = js(path / "result.json", result)
                    entry = {"dataset": dataset, "arm": arm, "lr": lr, "seed": seed, "val_score": score,
                             "fit_dir": str(path.relative_to(root)), "fit_result_sha256": result_hash,
                             "checkpoint_sha256": sha(checkpoint), "guard_path": str(guard_path.relative_to(root)),
                             "guard_sha256": guard_hash}
                    entries.append(entry)
                    if lr == .01:
                        selected.append(entry)
        directory = run / "baselines" / dataset
        directory.mkdir(parents=True)
        np.savez_compressed(directory / "model.npz", marker=np.array(0))
        pred = np.full((4, 30, 2, 21, 48), 1.5)
        np.savez_compressed(directory / "V_predictions.npz", candidate_predictions=pred,
                            val_target=np.zeros((30, 2, 48)), val_origins=origins["V"])
        candidates = [
            {"index": 0, "method": "SEASONAL_WEEK", "lambda": None, "validation": {"score": 1.5}},
            {"index": 1, "method": "CONTEXT_RIDGE", "lambda": 0.01, "validation": {"score": 1.5}},
            {"index": 2, "method": "CONTEXT_RIDGE", "lambda": 1.0, "validation": {"score": 1.5}},
            {"index": 3, "method": "CONTEXT_RIDGE", "lambda": 100.0, "validation": {"score": 1.5}},
        ]
        choice = {"dataset": dataset, "completed": True, "contract_sha256": contract_hash,
                  "model_sha256": sha(directory / "model.npz"), "fit_data_sha256": contract["datasets"][dataset]["fit_data_sha256"],
                  "V_predictions_sha256": sha(directory / "V_predictions.npz"),
                  "uses_calibration_labels": False, "uses_evaluation_labels": False, "selected_index": 0,
                  "candidates": candidates,
                  "selected_candidate": {"method": "SEASONAL_WEEK", "lambda": None}}
        choice_hash = js(directory / "selection.json", choice)
        js(directory / "result.json", {"dataset": dataset, "completed": True, "stage": "fit",
                                       "status": "BASELINE_FIT_COMPLETE_GLOBAL_SELECTION_PENDING",
                                       "contract_sha256": contract_hash, "candidate_count": 4,
                                       "ridge_oof_fits": 6, "ridge_final_fits": 3, "new_neural_fits": 0,
                                       "seed": None, "train_origins": 90, "val_origins": 30,
                                       "selected_index": 0, "selection_sha256": choice_hash,
                                       "model_sha256": sha(directory / "model.npz"), "wall_seconds": .1})
        baselines[dataset] = {"model_sha256": sha(directory / "model.npz"), "selection_sha256": choice_hash}
    selection = {"completed": True, "global_choices_frozen": True, "contract_sha256": contract_hash,
                 "C_or_E_used_for_selection": False, "all_trials": entries, "selected": selected, "baselines": baselines}
    selection_hash = js(run / "selection.json", selection)
    forecasts = []
    for dataset in contract["datasets"]:
        for entry in [None] + [row for row in selected if row["dataset"] == dataset]:
            arm, seed = ("F0", SEEDS[0]) if entry is None else (entry["arm"], entry["seed"])
            path = run / "forecasts" / dataset / f"{arm}_{seed}"
            path.mkdir(parents=True)
            guard_path = run / "guards" / "forecasts" / dataset / f"{arm}_{seed}"
            status(guard_path)
            result = {"completed": True, "dataset": dataset, "arm": arm, "seed": seed, "contract_sha256": contract_hash,
                      "selection_sha256": selection_hash, "optimizer_steps": 0, "audits": {"model_unchanged": True},
                      "trainable": 0 if entry is None else counts[arm], "best_step": 0,
                      "peak_cuda_allocated_bytes": 2000000, "peak_cuda_reserved_bytes": 3000000, "rss_sample_max_bytes": 4000000,
                      "cal_score": abs(value(arm, seed)), "eval_score": abs(value(arm, seed)), "seconds": {"total": .5}}
            if entry is not None:
                result.update(fit_dir=str(root / entry["fit_dir"]), fit_result_sha256=entry["fit_result_sha256"],
                              checkpoint_sha256=entry["checkpoint_sha256"])
            for letter in ("C", "E"):
                result[f"{letter}predictions_sha256"] = saved_prediction(path / f"{letter}predictions.npz", letter, value(arm, seed))
            result_hash = js(path / "result.json", result)
            forecasts.append({"dataset": dataset, "arm": arm, "seed": seed, "path": str(path.relative_to(root)),
                              "result_sha256": result_hash, "guard_path": str(guard_path.relative_to(root))})
        directory = run / "baselines" / dataset
        result = {"completed": True, "global_selection_sha256": selection_hash, "contract_sha256": contract_hash,
                  "holdout_data_sha256": contract["datasets"][dataset]["holdout_data_sha256"],
                  "uses_calibration_for_fitting": False, "predictions": {}}
        for letter in ("C", "E"):
            path = directory / f"{letter}_predictions.npz"
            n = len(origins[letter])
            np.savez_compressed(path, prediction=np.full((n, 2, 21, 48), 1.5), target=np.zeros((n, 2, 48)), origins=origins[letter])
            result["predictions"][letter] = {"path": str(path), "sha256": sha(path)}
        js(directory / "forecast_result.json", result)
    for name in ("baseline_fit", "baseline_forecast", "synthetic_smoke"):
        status(run / "guards" / name)
    js(run / "smoke_completed.json", {"trials": [{"guard_path": str((run / "guards" / "synthetic_smoke").relative_to(root))}]})
    js(run / "completed.json", {"completed": True, "contract_sha256": contract_hash, "selection_sha256": selection_hash, "forecasts": forecasts})
    return contract


class AnalysisTests(unittest.TestCase):
    def module(self):
        name = "experiments.peft_fullft_reference_v3.analyse"
        self.assertIsNotNone(importlib.util.find_spec(name), "analysis implementation is missing")
        return importlib.import_module(name)

    def test_seed_losses_are_averaged_without_forecast_ensembling(self):
        a = self.module()
        result = a.summarize_dataset("synthetic", fixture(), np.ones(2), Q, SEEDS,
                                     replicates=40, block_days=2)
        lora = result["arms"]["LORA"]["SORT"]
        self.assertAlmostEqual(lora["score_mean"], 5. / 3)
        np.testing.assert_allclose(lora["seed_range"], [1., 3.], rtol=1e-14)
        self.assertNotAlmostEqual(lora["score_mean"], 1.)

    def test_effect_sign_and_denominator_follow_named_benefit(self):
        a = self.module()
        result = a.summarize_dataset("synthetic", fixture(), np.ones(2), Q, SEEDS,
                                     replicates=40, block_days=2)
        full = result["effects"]["SORT"]["FULL_FT_benefit_vs_LORA"]
        self.assertAlmostEqual(full["value"], ((5. / 3) - .5) / 2)
        self.assertAlmostEqual(full["seed_range"][0], .25)
        self.assertAlmostEqual(full["seed_range"][1], 1.25)
        self.assertNotIn("decision", full)
        self.assertNotIn("practical_threshold", full)
        self.assertFalse(result["effects"]["QCAL"]["FULL_FT_benefit_vs_LORA"]["defined"])

    def test_target_or_origin_misalignment_fails_before_scoring(self):
        a = self.module()
        rows = fixture()
        rows[-1]["E"]["target"][0, 0, 0] = 1
        with self.assertRaisesRegex(AssertionError, "targets"):
            a.summarize_dataset("synthetic", rows, np.ones(2), Q, SEEDS, replicates=20, block_days=2)
        rows = fixture()
        rows[-1]["C"]["origins"][0] += 1
        with self.assertRaisesRegex(AssertionError, "origins"):
            a.summarize_dataset("synthetic", rows, np.ones(2), Q, SEEDS, replicates=20, block_days=2)

    def test_calibration_offsets_do_not_use_evaluation_targets(self):
        a = self.module()
        rows = fixture()
        before = a.summarize_dataset("synthetic", rows, np.ones(2), Q, SEEDS, replicates=20, block_days=2)
        for row in rows:
            row["E"]["target"] += 1
        after = a.summarize_dataset("synthetic", rows, np.ones(2), Q, SEEDS, replicates=20, block_days=2)
        self.assertEqual(before["qcal_offsets"], after["qcal_offsets"])
        self.assertNotEqual(before["arms"]["LORA"]["QCAL"]["score_mean"], after["arms"]["LORA"]["QCAL"]["score_mean"])

    def test_grid_choice_uses_mean_seed_validation_and_low_lr_tie(self):
        a = self.module()
        entries = []
        for lr, scores in ((.01, [0., 9., 0.]), (.1, [2., 2., 2.]), (1., [1., 4., 1.])):
            entries.extend({"dataset": "d", "arm": "LORA", "lr": lr, "seed": seed, "val_score": score}
                           for seed, score in zip(SEEDS, scores))
        selected = [entry for entry in entries if entry["lr"] == .1]
        result = a.audit_selection(entries, selected, ["d"], {"LORA": [.01, .1, 1.]}, SEEDS)
        self.assertEqual(result[0]["selected_lr"], .1)
        with self.assertRaisesRegex(AssertionError, "selection"):
            a.audit_selection(entries, [entry for entry in entries if entry["lr"] == 1.], ["d"], {"LORA": [.01, .1, 1.]}, SEEDS)
        with self.assertRaisesRegex(AssertionError, "grid"):
            a.audit_selection(entries[:-1], selected, ["d"], {"LORA": [.01, .1, 1.]}, SEEDS)

    def test_completed_synthetic_pipeline_audits_all_fits_and_renders(self):
        a = self.module()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            contract = finished_tree(root)
            with patch.object(a, "read_contract", return_value=contract):
                result = a.execute(root)
            self.assertEqual(result["metric_rows"], 44)
            self.assertEqual(result["costs"]["all_neural_fit_count"], 54)
            self.assertAlmostEqual(result["costs"]["total_neural_fit_seconds"], 54.)
            self.assertEqual(result["audit"]["selected_neural_forecasts"], 20)
            self.assertEqual(len(result["audit"]["independent_fit_audit_sha256"]), 64)
            self.assertEqual(result["audit"]["fixed_chronology"]["bike"]["origin_counts"], EXPECTED_COUNTS)
            self.assertTrue(result["costs"]["simple"]["bike"]["selected_fit_time_unavailable"])
            self.assertIsNone(result["costs"]["simple"]["bike"]["selected_fit_seconds"])
            self.assertAlmostEqual(result["costs"]["simple"]["bike"]["hpo_fit_seconds"], .1)
            simple_rows = [row for row in result["full_metric_records"] if row["dataset"] == "bike" and row["arm"] == "SIMPLE"]
            self.assertTrue(all(row["selected_fit_time_unavailable"] for row in simple_rows))
            self.assertTrue(all(row["fit_seconds"] is None for row in simple_rows))
            self.assertAlmostEqual(result["datasets"]["bike"]["arms"]["LORA"]["SORT"]["score_mean"], 5. / 3)
            from experiments.peft_fullft_reference_v3.plot import render
            files = render(result, root / "synthetic_figures")
            self.assertEqual(len(files), 4)
            for name in files:
                path = Path(name)
                magic = path.read_bytes()[:8]
                self.assertTrue(magic.startswith(b"%PDF") if path.suffix == ".pdf" else magic == b"\x89PNG\r\n\x1a\n")

    def test_execute_rejects_forecast_not_linked_to_exact_selected_fit(self):
        a = self.module()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            contract = finished_tree(root)
            run = root / "runs" / "peft_fullft_reference_v3"
            selection = read_json(run / "selection.json")
            completed = read_json(run / "completed.json")
            selected = next(row for row in selection["selected"]
                            if row["dataset"] == "bike" and row["arm"] == "LORA" and row["seed"] == SEEDS[0])
            unselected = next(row for row in selection["all_trials"]
                              if row["dataset"] == selected["dataset"] and row["arm"] == selected["arm"]
                              and row["seed"] == selected["seed"] and row["lr"] != selected["lr"])
            forecast = next(row for row in completed["forecasts"]
                            if row["dataset"] == selected["dataset"] and row["arm"] == selected["arm"]
                            and row["seed"] == selected["seed"])
            result_path = root / forecast["path"] / "result.json"
            result = read_json(result_path)
            result["fit_dir"] = str(root / unselected["fit_dir"])
            result["fit_result_sha256"] = unselected["fit_result_sha256"]
            result["checkpoint_sha256"] = unselected["checkpoint_sha256"]
            forecast["result_sha256"] = write_json(result_path, result)
            write_json(run / "completed.json", completed)
            with patch.object(a, "read_contract", return_value=contract):
                with self.assertRaisesRegex(AssertionError, "selected fit"):
                    a.execute(root)

    def test_execute_rejects_incomplete_or_misidentified_simple_baseline_result(self):
        a = self.module()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            contract = finished_tree(root)
            result_path = root / "runs" / "peft_fullft_reference_v3" / "baselines" / "bike" / "result.json"
            result = read_json(result_path)
            result["train_origins"] = 89
            write_json(result_path, result)
            with patch.object(a, "read_contract", return_value=contract):
                with self.assertRaisesRegex(AssertionError, "Simple reference"):
                    a.execute(root)

    def test_execute_requires_passing_frozen_independent_fit_audit(self):
        a = self.module()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            contract = finished_tree(root)
            audit_path = root / next(key for key in contract["protected_hashes"] if key.endswith("independent_fit_audit.json"))
            write_json(audit_path, {"completed": True, "passed": False})
            contract["protected_hashes"][str(audit_path.relative_to(root))] = sha(audit_path)
            contract["independent_fit_audit_sha256"] = sha(audit_path)
            with patch.object(a, "read_contract", return_value=contract):
                with self.assertRaisesRegex(AssertionError, "independent fit audit"):
                    a.execute(root)


if __name__ == "__main__":
    unittest.main()
