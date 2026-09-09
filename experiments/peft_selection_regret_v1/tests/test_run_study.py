"""CPU lifecycle fixtures; every guard call is replaced and no model is loaded."""

import copy
from contextlib import ExitStack, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.peft_adaptation_scope_v1 import guard as original_guard
from experiments.peft_selection_regret_v1 import run_study as r


def fixture(root):
    study = root / "runs" / r.STUDY
    cells, jobs, reused = {}, [], []
    rates = {"F0": (0.,), "H_MLP": (1e-4, 3e-4, 1e-3),
             "H_FULL": (3e-5, 1e-4, 3e-4), "OFF_LORA": (1e-5, 3e-5, 1e-4)}
    for number in (12, 13):
        for dataset in ("bike", "household"):
            cell_id = f"s{number}_{dataset}"
            candidates = {}
            for method, values in rates.items():
                for lr in values:
                    key = r.prepare.candidate_id(cell_id, method, lr)
                    used = method == "F0" or (method == "H_FULL" and lr == 3e-5) or (method == "OFF_LORA" and lr == 1e-4)
                    output = r.relative(root, study / "forecasts" / cell_id / key)
                    candidates[key] = {"candidate_id": key, "cell_id": cell_id, "dataset": dataset, "method": method,
                                       "lr": lr, "seed": 12000, "best_step": 0 if method == "F0" else 80,
                                       "fit_result_sha256": "fit result fixture " + key, "checkpoint_sha256": "checkpoint fixture " + key,
                                       "restored_adaptation_sha256": "restored fixture " + key, "new_forecast_output": output}
                    job = {"cell_id": cell_id, "candidate_id": key, "output": output}
                    (reused if used else jobs).append(job)
            cells[cell_id] = {"cell_id": cell_id, "dataset": dataset, "candidates": candidates,
                              "holdout_data_sha256": "holdout fixture " + cell_id,
                              "smoke_candidate": r.prepare.candidate_id(cell_id, "OFF_LORA", 1e-4)}
    value = {"completed": True, "selection_rules_frozen": True, "missing_optional_sources": [],
             "cells": cells, "unselected_forecast_jobs": jobs, "reuse_forecast_jobs": reused}
    r.write(study / "selection_contract.json", value)
    return value


class TestRunnerLifecycle(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.study = self.root / "runs" / r.STUDY
        self.contract = fixture(self.root)
        self.source = self.root / "experiments" / r.STUDY / "run_study.py"
        self.source.parent.mkdir(parents=True)
        self.source.write_text("CPU fixture source; never executed", encoding="utf-8")
        self.guard_calls, self.lock_active, self.active_children = [], False, 0
        self.addCleanup(self.assertEqual, self.active_children, 0)

    def smoke_job(self, cell_id="s12_bike"):
        candidate_id = self.contract["cells"][cell_id]["smoke_candidate"]
        return {"cell_id": cell_id, "candidate_id": candidate_id,
                "output": r.relative(self.root, self.study / "smoke" / cell_id / candidate_id)}

    def materialize(self, job, smoke=False, status=None):
        path = self.root / job["output"]
        path.mkdir(parents=True, exist_ok=True)
        candidate = self.contract["cells"][job["cell_id"]]["candidates"][job["candidate_id"]]
        (path / "predictions.npz").write_bytes(b"CPU fixture prediction bytes; contains no real data")
        result = {key: candidate[key] for key in ("cell_id", "candidate_id", "dataset", "method", "lr", "seed", "best_step")}
        result.update({"completed": True, "stage": "diagnostic_forecast", "study": r.STUDY, "smoke": smoke,
                       "selection_contract_sha256": r.sha(self.study / "selection_contract.json"),
                       "fit_result_sha256": candidate["fit_result_sha256"], "fit_checkpoint_sha256": candidate["checkpoint_sha256"],
                       "holdout_data_sha256": self.contract["cells"][job["cell_id"]]["holdout_data_sha256"],
                       "restored_adaptation_sha256": candidate["restored_adaptation_sha256"],
                       "predictions_sha256": r.sha(path / "predictions.npz"), "optimizer_steps": 0,
                       "no_training": True, "model_unchanged": True, "original_deployment_selection_gate_modified": False,
                       "audits": {name: True for name in ("source_unchanged", "weights_unchanged", "checkpoint_reload_verified",
                                  "no_optimizer", "model_unchanged", "numerical_path_unchanged", "reference_predictions_equal")},
                       "splits": {"cal": {"origins": 13}, "eval": {"origins": 83}},
                       "reference_comparison": {"passed": True, "normalized_tolerance": 1e-5,
                           "comparisons": {f"{split}_{name}": {"normalized_max_abs": 0., "raw_max_abs": 0.}
                                           for split in ("cal", "eval") for name in ("predictions", "unsorted_predictions")}}})
        r.write(path / "result.json", result)
        status = status or {"completed": True, "returncode": 0, "reasons": [], "elapsed_seconds": 2.5}
        r.write(path / "guard/status.json", status)
        return {"cell_id": job["cell_id"], "candidate_id": job["candidate_id"], "path": job["output"], "smoke": smoke,
                "result_sha256": r.sha(path / "result.json"), "predictions_sha256": result["predictions_sha256"],
                "guard_seconds": status["elapsed_seconds"]}

    def seed_smoke(self):
        entries = [self.materialize(self.smoke_job(key), True) for key in self.contract["cells"]]
        record = {"completed": True, "selection_contract_sha256": r.sha(self.study / "selection_contract.json"),
                  "trials": entries, "guard_seconds": 10., "invocation_wall_seconds": 123.}
        r.write(self.study / "smoke_completed.json", record)
        return record

    def invoke(self, smoke=False, guard=None, prepare_side_effect=None):
        owner = self

        class Lock:
            def __init__(self, path):
                owner.assertEqual(path, owner.study / ".runner.lock")

            def __enter__(self):
                owner.assertFalse(owner.lock_active)
                owner.lock_active = True

            def __exit__(self, *exc):
                owner.lock_active = False

        with ExitStack() as stack:
            stack.enter_context(patch.object(r, "__file__", str(self.source)))
            stack.enter_context(patch.object(r, "StudyLock", Lock))
            stack.enter_context(patch.object(r.sys, "argv", [str(self.source), *( ["--smoke"] if smoke else [])]))
            stack.enter_context(patch.object(r.prepare, "validate_contract", side_effect=prepare_side_effect or
                                             (lambda root, path: copy.deepcopy(self.contract))))
            mocked = stack.enter_context(patch.object(r, "run_guarded", side_effect=guard or self.successful_guard))
            stack.enter_context(redirect_stdout(io.StringIO()))
            r.main()
        return mocked

    def successful_guard(self, command, guard_path, cwd, timeout_seconds):
        self.assertTrue(self.lock_active)
        self.assertEqual(self.active_children, 0)
        self.active_children += 1
        try:
            self.assertEqual(command[:3], [r.sys.executable, "-m", f"experiments.{r.STUDY}.forecast"])
            self.assertEqual(cwd, self.root)
            self.assertEqual(timeout_seconds, 300)
            self.assertNotIn("--steps", command)
            self.assertNotIn("--lr", command)
            job = {"cell_id": command[command.index("--cell") + 1], "candidate_id": command[command.index("--candidate") + 1],
                   "output": r.relative(self.root, Path(command[command.index("--output") + 1]))}
            self.assertEqual(guard_path, self.root / job["output"] / "guard")
            persisted = self.latest_invocation()
            self.assertFalse(persisted["completed"])
            self.assertFalse(persisted["attempts"][-1]["completed"])
            self.assertEqual(persisted["attempts"][-1]["path"], job["output"])
            self.guard_calls.append(job)
            self.materialize(job, "--smoke" in command)
            return r.read(guard_path / "status.json")
        finally:
            self.active_children -= 1

    def latest_invocation(self):
        paths = sorted((self.study / "invocations").glob("*.json"), key=lambda p: int(p.stem))
        return r.read(paths[-1])

    def test_uses_original_single_child_guard_and_study_lock(self):
        self.assertIs(r.run_guarded, original_guard.run_guarded)
        self.assertIs(r.StudyLock, original_guard.StudyLock)

    def test_smoke_runs_four_serial_replays_and_reuses_verified_results(self):
        mocked = self.invoke(smoke=True)
        self.assertEqual(mocked.call_count, 4)
        self.assertEqual(len(self.guard_calls), 4)
        before = {str(path): path.read_bytes() for path in (self.study / "smoke").rglob("*") if path.is_file()}
        marker = (self.study / "smoke_completed.json").read_bytes()
        second = self.invoke(smoke=True)
        self.assertEqual(second.call_count, 0)
        self.assertEqual(len(self.latest_invocation()["reused_completed_jobs"]), 4)
        self.assertEqual(self.latest_invocation()["attempts"], [])
        self.assertEqual((self.study / "smoke_completed.json").read_bytes(), marker)
        self.assertEqual({str(path): path.read_bytes() for path in (self.study / "smoke").rglob("*") if path.is_file()}, before)

    def test_main_requires_all_smoke_provenance_before_any_child(self):
        with self.assertRaises(FileNotFoundError):
            self.invoke()
        self.assertEqual(self.guard_calls, [])
        self.assertFalse(self.latest_invocation()["completed"])
        record = self.seed_smoke()
        record["selection_contract_sha256"] = "changed selection contract"
        r.write(self.study / "smoke_completed.json", record)
        with self.assertRaisesRegex(AssertionError, "must pass S0"):
            self.invoke()
        self.assertEqual(self.guard_calls, [])
        self.assertEqual(r.read(self.study / "progress.json")["state"], "failed")

    def test_main_runs_exactly_twenty_eight_then_resumes_without_new_children(self):
        self.seed_smoke()
        self.assertEqual(self.invoke().call_count, 28)
        done = r.read(self.study / "completed.json")
        self.assertEqual((done["forecasts"], done["reused_forecasts"], done["training_runs"]), (28, 12, 0))
        self.assertTrue(self.latest_invocation()["completed"])
        self.assertEqual(len(self.latest_invocation()["attempts"]), 28)
        before = (self.study / "completed.json").read_bytes()
        self.assertEqual(self.invoke().call_count, 0)
        self.assertEqual(len(self.latest_invocation()["reused_completed_jobs"]), 28)
        self.assertEqual((self.study / "completed.json").read_bytes(), before)

    def test_partial_forecast_is_never_overwritten_or_retried(self):
        self.seed_smoke()
        job = self.contract["unselected_forecast_jobs"][0]
        path = self.root / job["output"]
        path.mkdir(parents=True)
        partial = path / "predictions.npz"
        partial.write_bytes(b"preserve interrupted output")
        with self.assertRaisesRegex(RuntimeError, "Preserve incomplete"):
            self.invoke()
        self.assertEqual(self.guard_calls, [])
        self.assertEqual(partial.read_bytes(), b"preserve interrupted output")
        self.assertFalse(self.latest_invocation()["completed"])
        self.assertEqual(self.latest_invocation()["attempts"], [])
        r.write(path / "result.json", {"completed": False, "error": "original interruption"})
        before = (path / "result.json").read_bytes()
        with self.assertRaisesRegex(RuntimeError, "Preserve incomplete"):
            self.invoke()
        self.assertEqual((path / "result.json").read_bytes(), before)

    def test_stopped_guard_attempt_is_persisted_and_not_a_completed_forecast(self):
        self.seed_smoke()

        def stopped(command, guard_path, cwd, timeout_seconds):
            self.assertFalse(self.latest_invocation()["attempts"][-1]["completed"])
            status = {"completed": False, "returncode": -1, "reasons": ["fixture resource stop"], "elapsed_seconds": 3.}
            r.write(guard_path / "status.json", status)
            return status

        with self.assertRaisesRegex(RuntimeError, "failed attempt preserved"):
            self.invoke(guard=stopped)
        invocation = self.latest_invocation()
        self.assertFalse(invocation["completed"])
        self.assertEqual(len(invocation["attempts"]), 1)
        attempt = invocation["attempts"][0]
        self.assertFalse(attempt["completed"])
        self.assertEqual(attempt["guard_status"]["reasons"], ["fixture resource stop"])
        self.assertIn("finished_at", attempt)
        self.assertEqual(attempt["guard_status_sha256"], r.sha(self.root / attempt["path"] / "guard/status.json"))
        self.assertFalse((self.study / "completed.json").exists())
        self.assertEqual(r.read(self.study / "progress.json")["state"], "failed")

    def test_guard_exception_preserves_started_attempt_and_invocation_failure(self):
        self.seed_smoke()

        def interrupted(*args, **kwargs):
            self.assertEqual(len(self.latest_invocation()["attempts"]), 1)
            raise RuntimeError("fixture guard startup failure")

        with self.assertRaisesRegex(RuntimeError, "startup failure"):
            self.invoke(guard=interrupted)
        invocation = self.latest_invocation()
        self.assertFalse(invocation["completed"])
        self.assertFalse(invocation["attempts"][0]["completed"])
        self.assertEqual(invocation["error_type"], "RuntimeError")
        self.assertIn("finished_at", invocation)
        self.assertFalse((self.study / "completed.json").exists())

    def test_guard_success_does_not_override_failed_result(self):
        self.seed_smoke()

        def incomplete_result(command, guard_path, cwd, timeout_seconds):
            status = self.successful_guard(command, guard_path, cwd, timeout_seconds)
            value = r.read(guard_path.parent / "result.json")
            value["completed"] = False
            r.write(guard_path.parent / "result.json", value)
            return status

        with self.assertRaisesRegex(AssertionError, "Incomplete diagnostic"):
            self.invoke(guard=incomplete_result)
        self.assertFalse(self.latest_invocation()["completed"])
        self.assertFalse((self.study / "completed.json").exists())

    def test_completed_result_with_failed_guard_is_not_reused(self):
        self.seed_smoke()
        job = self.contract["unselected_forecast_jobs"][0]
        self.materialize(job, status={"completed": False, "returncode": 1,
                                    "reasons": ["fixture child failure"], "elapsed_seconds": 3.})
        path = self.root / job["output"]
        before = {str(p): p.read_bytes() for p in path.rglob("*") if p.is_file()}
        with self.assertRaisesRegex(RuntimeError, "failed or partial guard"):
            self.invoke()
        self.assertEqual(self.guard_calls, [])
        self.assertEqual({str(p): p.read_bytes() for p in path.rglob("*") if p.is_file()}, before)
        self.assertEqual(self.latest_invocation()["reused_completed_jobs"], [])

    def test_saved_smoke_prediction_hash_must_match_before_main(self):
        record = self.seed_smoke()
        prediction = self.root / record["trials"][0]["path"] / "predictions.npz"
        prediction.write_bytes(b"changed CPU fixture prediction bytes")
        with self.assertRaisesRegex(AssertionError, "provenance changed"):
            self.invoke()
        self.assertEqual(self.guard_calls, [])
        self.assertFalse(self.latest_invocation()["completed"])

    def test_existing_completion_mismatch_is_rejected_without_overwriting_it(self):
        self.seed_smoke()
        for job in self.contract["unselected_forecast_jobs"]:
            self.materialize(job)
        nominal = {"completed": True, "selection_contract_sha256": r.sha(self.study / "selection_contract.json"),
                   "forecasts": 28, "reused_forecasts": 12, "forecast_guard_seconds": 70., "training_runs": 0,
                   "invocation_wall_seconds": 1.}
        for key, value in (("completed", False), ("selection_contract_sha256", "wrong"), ("forecasts", 27), ("training_runs", 1)):
            with self.subTest(key=key):
                r.write(self.study / "completed.json", {**nominal, key: value})
                before = (self.study / "completed.json").read_bytes()
                with self.assertRaisesRegex(AssertionError, "completion record differs"):
                    self.invoke()
                self.assertEqual((self.study / "completed.json").read_bytes(), before)
                self.assertFalse(self.latest_invocation()["completed"])
        self.assertEqual(self.guard_calls, [])

    def test_s0_numeric_comparison_and_complete_scope_are_required(self):
        job = self.smoke_job()
        entry = self.materialize(job, True)
        path = self.root / job["output"] / "result.json"
        nominal = r.read(path)
        mutations = [lambda x: x["reference_comparison"].update(passed=False),
                     lambda x: x["reference_comparison"].update(normalized_tolerance=.1),
                     lambda x: x["reference_comparison"]["comparisons"].pop("eval_unsorted_predictions"),
                     lambda x: x["reference_comparison"]["comparisons"]["eval_unsorted_predictions"].update(normalized_max_abs=.0001),
                     lambda x: x["reference_comparison"]["comparisons"]["eval_unsorted_predictions"].update(normalized_max_abs=-1.),
                     lambda x: x["splits"]["eval"].update(origins=82)]
        for mutate in mutations:
            value = copy.deepcopy(nominal)
            mutate(value)
            r.write(path, value)
            changed_entry = {**entry, "result_sha256": r.sha(path)}
            with self.assertRaises(AssertionError):
                r.verify_entry(self.root, self.contract, changed_entry, True)

    def test_contract_change_after_last_child_prevents_completion(self):
        altered = copy.deepcopy(self.contract)
        altered["changed_after_start"] = True
        with self.assertRaisesRegex(AssertionError, "changed during inference"):
            self.invoke(smoke=True, prepare_side_effect=[copy.deepcopy(self.contract), altered])
        self.assertEqual(len(self.guard_calls), 4)
        self.assertFalse((self.study / "smoke_completed.json").exists())
        self.assertFalse(self.latest_invocation()["completed"])


if __name__ == "__main__":
    unittest.main()
