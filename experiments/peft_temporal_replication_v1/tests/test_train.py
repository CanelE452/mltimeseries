import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.peft_external_gap_v1 import train as original
from experiments.peft_external_gap_v1.tests.test_train import panel_archive
from experiments.peft_temporal_replication_v1 import train as t


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def touch(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return t.file_hash(path)


def fixture(root):
    old, new = root / "runs" / t.PARENT_STUDY, root / "runs" / t.STUDY
    old_results = root / "results" / t.PARENT_STUDY
    old_sources = {f"experiments/{t.PARENT_STUDY}/{name}.py": touch(root / "experiments" / t.PARENT_STUDY / f"{name}.py", name)
                   for name in ("train", "data")}
    previous = {"sources": old_sources, "data_files": {"runs/peft_external_gap_v1/prepared/old.npz":
                 touch(old / "prepared/old.npz", "old original values")}, "plan_sha256": touch(root / t.PARENT_PLAN, "original plan")}
    write(old / "study_contract.json", previous)
    prior_entries, artifacts, caches = [], {}, []
    for dataset in ("bike", "household"):
        path = old / "trials" / dataset / "F0"
        cache = old / "cache" / dataset / "hidden.npy"
        cache_hash = touch(cache, "original frozen feature bytes " + dataset)
        caches.append(cache)
        meta = {"cache_array_hashes": {str(cache): cache_hash}}
        write(path / "result.json", meta)
        result_hash = t.file_hash(path / "result.json")
        artifacts[str(path / "result.json")] = result_hash
        prediction = path / "predictions.npz"
        artifacts[str(prediction)] = touch(prediction, "old prediction bytes")
        prior_entries.append({"dataset": dataset, "role": "F0", "path": path.relative_to(root).as_posix(),
                              "result_sha256": result_hash})
    write(old / "selection.json", {"selected": prior_entries})
    output_hash = touch(old_results / "effects.json", "original effects")
    analysis = root / "experiments" / t.PARENT_STUDY / "analyse.py"
    verification = {"passed": True, "completed": True, "artifact_hashes": artifacts,
                    "analysis_sources": {str(analysis): touch(analysis, "original analysis")},
                    "output_hashes": {"effects.json": output_hash},
                    "study_contract_sha256": t.file_hash(old / "study_contract.json"),
                    "selection_sha256": t.file_hash(old / "selection.json")}
    write(old_results / "verification.json", verification)
    protected = {**artifacts, **verification["analysis_sources"], str(old_results / "effects.json"): output_hash}
    for path in (old_results / "verification.json", old / "study_contract.json", old / "selection.json", root / t.PARENT_PLAN):
        protected[str(path)] = t.file_hash(path)
    sources = dict(old_sources)
    for name in ("train", "data", "raw", "run_study"):
        path = root / "experiments" / t.STUDY / f"{name}.py"
        sources[path.relative_to(root).as_posix()] = touch(path, "new " + name)
    data_files = {}
    for dataset in ("bike", "household"):
        for stage in ("fit", "holdout"):
            path = new / "prepared" / f"{dataset}_{stage}.npz"
            data_files[path.relative_to(root).as_posix()] = touch(path, "new " + dataset + stage)
    frozen = {"sources": sources, "data_files": data_files,
              "plan_sha256": touch(root / t.PLAN, "new fixed temporal plan"),
              "previous_contract_sha256": t.file_hash(old / "study_contract.json"),
              "parent_artifacts": {Path(p).relative_to(root).as_posix(): h for p, h in protected.items()}}
    write(new / "study_contract.json", frozen)
    write(new / "smoke_contract.json", frozen)
    entries, choices = [], {}
    for dataset in ("bike", "household"):
        choices[dataset] = {"H": {"method": "H_MLP", "lr": 1e-4}, "OFF_LORA": {"method": "OFF_LORA", "lr": 3e-5}}
        for role, method, lr in (("F0", "F0", 0.), ("H", "H_MLP", 1e-4), ("OFF_LORA", "OFF_LORA", 3e-5)):
            for seed in ((12000,) if role == "F0" else (12000, 12001, 12002)):
                path = new / "trials" / dataset / method / str(seed)
                meta = {"completed": True, "wrapped_completed": True, "smoke": False,
                        "dataset": dataset, "method": method, "seed": seed, "lr": lr}
                write(path / "result.json", meta)
                write(path / "guard/status.json", {"completed": True, "returncode": 0, "reasons": []})
                ckpt = touch(path / "best_trainable.pt", "new adaptive checkpoint")
                entries.append({**meta, "role": role, "path": path.relative_to(root).as_posix(),
                                "result_sha256": t.file_hash(path / "result.json"), "checkpoint_sha256": ckpt})
    selection = {"completed": True, "global_choices_frozen": True, "fit_trial_count": 28,
                 "study_contract_sha256": t.file_hash(new / "study_contract.json"), "selected": entries, "choices": choices}
    write(new / "selection.json", selection)
    return frozen, new / "selection.json", root / entries[0]["path"], caches


class TestTemporalWrapper(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        torch.set_num_threads(2)

    def test_numerical_function_bytecode_is_unchanged_and_namespace_isolated(self):
        for name in ("run_fit", "run_forecast", "prepare_cache", "direct", "predict", "scores", "construct",
                     "load_base", "information_audit", "padded_origins", "validate_fit_args"):
            self.assertEqual(getattr(t._shared, name).__code__.co_code, getattr(original, name).__code__.co_code)
            self.assertEqual(getattr(t._shared, name).__code__.co_consts, getattr(original, name).__code__.co_consts)
        self.assertEqual(original.STUDY, t.PARENT_STUDY)
        self.assertEqual(original.PLAN, t.PARENT_PLAN)
        self.assertEqual(t._shared.STUDY, t.STUDY)
        self.assertIsNot(t._shared, original)
        self.assertEqual(t.Panel.batch.__code__.co_code, original.Panel.batch.__code__.co_code)
        self.assertIs(t.file_hash, original.file_hash)

    def test_panel_loss_and_scores_match_original_on_missing_multitarget_fixture(self):
        path = self.root / "panel.npz"
        panel_archive(path)
        old, new = original.Panel(path, "fit", True), t.Panel(path, "fit", True)
        origins = new.origins["train"][:4]
        for left, right in zip(old.batch(origins, "cpu"), new.batch(origins, "cpu")):
            torch.testing.assert_close(left, right, rtol=0, atol=0, equal_nan=True)
        rng = np.random.default_rng(123)
        prediction = rng.normal(size=(4, 2, 21, 48))
        q, target = new.quantiles, new.targets("val")
        a = original.scores(prediction, target, new.fit_std[:2], q)
        b = t.scores(prediction, target, new.fit_std[:2], q)
        self.assertEqual(a[0], b[0])
        np.testing.assert_array_equal(a[1], b[1])
        np.testing.assert_array_equal(a[2], b[2])
        _, labels, _ = new.batch(origins, "cpu")
        first = torch.randn(20, 21, 48, requires_grad=True)
        second = first.detach().clone().requires_grad_(True)
        args = (labels, torch.zeros(20, 1), torch.ones(20, 1), torch.tensor(q, dtype=torch.float32), True)
        old_loss = original.native.native_pinball(first, *args)
        new_loss = t.native.native_pinball(second, *args)
        old_loss.backward(); new_loss.backward()
        torch.testing.assert_close(old_loss, new_loss, rtol=0, atol=0)
        torch.testing.assert_close(first.grad, second.grad, rtol=0, atol=0)

    def test_cli_keeps_all_original_method_budget_contracts(self):
        for smoke in (False, True):
            for method in t.METHODS:
                args = t.parser().parse_args(["fit", "--data", "x", "--output", "y", "--cache", "z",
                    "--method", method, "--lr", str(t.RATES[method][0]), "--seed", "12000",
                    "--steps", str(0 if method == "F0" else (5 if smoke else 200)),
                    "--val-every", str(5 if smoke else 40), *(["--smoke"] if smoke else [])])
                t.validate_fit_args(args)

    def test_registration_restores_atomic_state_even_when_interrupted(self):
        old_atomic, old_state = t._shared.atomic_json, t._shared.source_state
        with self.assertRaisesRegex(RuntimeError, "fixture"):
            with t.registration(self.root, []):
                self.assertIsNot(t._shared.atomic_json, old_atomic)
                self.assertIs(original.atomic_json, t.atomic_json)
                raise RuntimeError("fixture")
        self.assertIs(t._shared.atomic_json, old_atomic)
        self.assertIs(t._shared.source_state, old_state)
        self.assertEqual(original.STUDY, t.PARENT_STUDY)

    def test_selection_uses_original_real_contract_without_trainlag_path(self):
        _, selection, fit, caches = fixture(self.root)
        entry, hashes = t.validate_selection(self.root, selection, fit)
        self.assertEqual(entry["method"], "F0")
        self.assertFalse((self.root / "runs/peft_trainlag_v1").exists())
        self.assertIn(str(caches[0]), hashes)

    def test_original_cache_change_is_rejected(self):
        _, selection, fit, caches = fixture(self.root)
        caches[0].write_text("changed original cache")
        with self.assertRaisesRegex(AssertionError, "Protected file changed"):
            t.validate_selection(self.root, selection, fit)

    def test_new_holdout_change_is_rejected_before_forecasting(self):
        _, selection, fit, _ = fixture(self.root)
        data = self.root / "runs" / t.STUDY / "prepared/bike_holdout.npz"
        data.write_text("changed future input")
        with self.assertRaisesRegex(AssertionError, "Protected file changed"):
            t.validate_selection(self.root, selection, fit)

    def test_pending_wrapper_cannot_be_selected(self):
        _, selection, fit, _ = fixture(self.root)
        meta = t.read(fit / "result.json")
        meta["wrapped_completed"] = False
        write(fit / "result.json", meta)
        chosen = t.read(selection)
        chosen["selected"][0]["result_sha256"] = t.file_hash(fit / "result.json")
        write(selection, chosen)
        with self.assertRaisesRegex(AssertionError, "wrapper or guard"):
            t.validate_selection(self.root, selection, fit)

    def test_source_state_protects_both_producer_and_consumer(self):
        frozen, _, _, _ = fixture(self.root)
        consumer = self.root / "experiments" / t.STUDY / "train.py"
        producer = self.root / "experiments" / t.PARENT_STUDY / "train.py"
        data = self.root / "runs" / t.STUDY / "prepared/bike_fit.npz"
        stub = {"source_hashes": {producer.relative_to(self.root).as_posix(): t.file_hash(producer)},
                "protected_hashes": {str(data): t.file_hash(data)}, "data_sha256": t.file_hash(data),
                "plan_sha256": frozen["plan_sha256"]}
        with patch.object(t, "__file__", str(consumer)), patch.object(t.parent, "__file__", str(producer)), \
                patch.object(t, "_original_source_state", return_value=copy.deepcopy(stub)):
            state = t.source_state(self.root, SimpleNamespace(data=data, smoke=False))
        self.assertEqual(state["source_hashes"][consumer.relative_to(self.root).as_posix()], t.file_hash(consumer))
        self.assertEqual(state["wrapper_contract"]["numerical_producer_source_sha256"], t.file_hash(producer))
        self.assertEqual(state["wrapper_contract"]["offset_days"], 190)
        self.assertIn(str(data), state["protected_hashes"])

    def test_completion_is_published_only_after_wrapper_audit(self):
        output = self.root / "trial"
        output.mkdir()
        source = self.root / "source.txt"
        digest = touch(source, "preserved")
        state = {"source_hashes": {"source.txt": digest}, "protected_hashes": {str(source): digest},
                 "wrapper_contract": {"source_sha256": digest}}
        def numerical(args):
            t._shared.source_state(self.root, args)
            result = {"completed": True, "source_hashes": state["source_hashes"], "protected_hashes": state["protected_hashes"],
                      "cache_array_hashes": {str(source): digest}, "audits": {"zero_update_identity": True, "checkpoint_reload_verified": True}}
            t._shared.atomic_json(output / "result.json", result)
            pending = t.read(output / "result.json")
            self.assertFalse(pending["completed"])
            self.assertFalse(pending["wrapped_completed"])
            return result
        with patch.object(t, "source_state", return_value=state), patch.object(t._shared, "run_fit", side_effect=numerical):
            result = t.run(SimpleNamespace(output=output, command="fit"))
        self.assertTrue(result["wrapped_completed"])
        self.assertTrue(t.read(output / "result.json")["completed"])

    def test_failed_final_audit_leaves_pending_result_incomplete(self):
        output = self.root / "trial"
        output.mkdir()
        source = self.root / "source.txt"
        digest = touch(source, "preserved")
        state = {"source_hashes": {"source.txt": digest}, "protected_hashes": {str(source): digest}, "wrapper_contract": {}}
        def numerical(args):
            t._shared.source_state(self.root, args)
            result = {"completed": True}
            t._shared.atomic_json(output / "result.json", result)
            source.write_text("changed after numerical completion")
            return result
        with patch.object(t, "source_state", return_value=state), patch.object(t._shared, "run_fit", side_effect=numerical):
            with self.assertRaisesRegex(AssertionError, "Protected file changed"):
                t.run(SimpleNamespace(output=output, command="fit"))
        self.assertFalse(t.read(output / "result.json")["completed"])
        self.assertFalse(t.read(output / "result.json")["wrapped_completed"])


if __name__ == "__main__":
    unittest.main()
