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
from experiments.peft_selection_regret_v1 import forecast as f


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.25))
        self.frozen = torch.nn.Parameter(torch.tensor(2.), requires_grad=False)
        self.method, self.use_arcsinh = "OFF_LORA", True

    def encode(self, context, groups, horizon):
        if torch.is_grad_enabled():
            raise AssertionError("Inference enabled autograd")
        rows = len(context)
        loc = context.mean(dim=1, keepdim=True)
        scale = context.std(dim=1, keepdim=True) + 1
        norm = (torch.arange(21, dtype=torch.float32).reshape(1, 21, 1) / 20 + self.weight).expand(rows, 21, horizon)
        return None, norm, loc, scale


class TestDiagnosticForecast(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        torch.set_num_threads(2)

    def test_native_numeric_code_and_original_module_are_unchanged(self):
        before = original.direct
        self.assertEqual(len(f.numerical_identity()), 5)
        records = []
        with f.capture_normalization(records):
            self.assertIs(original.direct, before)
            self.assertIsNot(f._numerical.direct, before)
        self.assertIs(original.direct, before)
        self.assertEqual(len(f.numerical_identity()), 5)

    def test_capture_preserves_native_output_padding_and_target_missingness(self):
        args = SimpleNamespace(device="cpu", min_free_ram_gib=0)
        for dataset in ("bike", "household"):
            path = self.root / f"{dataset}.npz"
            panel_archive(path, dataset, "forecast")
            panel = original.Panel(path, "forecast")
            model = ToyModel()
            before = original.parameter_hash(model.named_parameters())
            for count in (1, 3, 4, 5, 13):
                panel.origins["cal"] = np.arange(336, 336 + count * 24, 24)
                untouched = panel.context_values.copy()
                expected, expected_raw = original.predict(model, panel, "cal", args, return_unsorted=True)
                actual, raw, norm, loc, scale = f.predict_with_normalization(model, panel, "cal", args)
                np.testing.assert_array_equal(actual, expected)
                np.testing.assert_array_equal(raw, expected_raw)
                np.testing.assert_array_equal(panel.context_values, untouched)
                self.assertEqual(norm.shape, (count, 2, 21, 48))
                self.assertEqual(loc.shape, (count, 2, 1))
                self.assertEqual(scale.shape, loc.shape)
                self.assertEqual(f.normalized_difference(raw, expected_raw, loc, scale, True), 0.)
                self.assertTrue(np.isnan(panel.targets("cal")[0, 0, 0]))
            self.assertEqual(original.parameter_hash(model.named_parameters()), before)
            self.assertIsNone(model.weight.grad)

    def test_capture_restores_private_function_after_failure(self):
        before = f._numerical.direct
        with self.assertRaises(RuntimeError):
            with f.capture_normalization([]):
                raise RuntimeError("fixture interrupted")
        self.assertIs(f._numerical.direct, before)

    def test_inverse_normalization_detects_error_in_both_paths(self):
        shape = (3, 2, 21, 48)
        loc, scale = np.full((3, 2, 1), 100., np.float32), np.full((3, 2, 1), 2., np.float32)
        baseline = np.full(shape, 100., np.float32)
        changed = baseline.copy()
        changed[2, 1, 20, 47] += 1
        self.assertGreater(f.normalized_difference(baseline, changed, loc, scale, True), .4)
        self.assertAlmostEqual(f.normalized_difference(baseline, changed, loc, scale, False), .5)
        changed.flat[0] = np.nan
        with self.assertRaises(ValueError):
            f.normalized_difference(baseline, changed, loc, scale, True)

    def test_reference_compares_unsorted_quantiles_and_original_masks(self):
        arrays = {"quantiles": np.arange(21), "target_indices": np.arange(2), "target_channels": np.array(["A", "B"])}
        for split, count in (("cal", 13), ("eval", 83)):
            arrays.update({f"{split}_predictions": np.zeros((count, 2, 21, 48), np.float32),
                           f"{split}_unsorted_predictions": np.zeros((count, 2, 21, 48), np.float32),
                           f"{split}_origins": np.arange(count), f"{split}_timestamps": np.arange(count).astype(str),
                           f"{split}_target": np.zeros((count, 2, 48), np.float32),
                           f"{split}_loc": np.zeros((count, 2, 1), np.float32),
                           f"{split}_scale": np.ones((count, 2, 1), np.float32)})
        arrays["eval_target"][82, 1, 47] = np.nan
        path = self.root / "reference.npz"
        np.savez(path, **arrays)
        self.assertTrue(f.compare_reference(arrays, path, True)["passed"])
        changed = copy.deepcopy(arrays)
        changed["eval_unsorted_predictions"][82, 1, 20, 47] += .1
        with self.assertRaisesRegex(AssertionError, "unsorted_predictions normalized"):
            f.compare_reference(changed, path, True)
        changed = copy.deepcopy(arrays)
        changed["eval_target"][82, 1, 47] = 0
        with self.assertRaisesRegex(AssertionError, "eval_target"):
            f.compare_reference(changed, path, True)

    def test_adaptive_checkpoint_roundtrip_keeps_frozen_parameters(self):
        model = ToyModel()
        expected = original.parameter_hash((n, p) for n, p in model.named_parameters() if p.requires_grad)
        frozen = original.native.frozen_digest(model)[0]
        checkpoint = self.root / "best_trainable.pt"
        original.legacy.save_trainable(model, checkpoint)
        with torch.no_grad():
            model.weight.add_(12)
        original.legacy.load_trainable(model, checkpoint, "cpu")
        self.assertEqual(original.parameter_hash((n, p) for n, p in model.named_parameters() if p.requires_grad), expected)
        self.assertEqual(original.native.frozen_digest(model)[0], frozen)
        torch.save({"unexpected": torch.ones(1)}, checkpoint)
        with self.assertRaisesRegex(AssertionError, "trainable map"):
            original.legacy.load_trainable(model, checkpoint, "cpu")

    def test_output_rejects_old_namespace_and_preserves_partial(self):
        with self.assertRaises(ValueError):
            f.new_output(self.root, self.root / "runs/peft_external_gap_v1/new")
        path = f.new_output(self.root, self.root / "runs" / f.STUDY / "fixture")
        (path / "predictions.npz").write_bytes(b"partial original bytes")
        with self.assertRaises(FileExistsError):
            f.new_output(self.root, path)
        self.assertEqual((path / "predictions.npz").read_bytes(), b"partial original bytes")

    def test_authorization_uses_fixed_grid_and_separates_replay_from_new_candidates(self):
        cells = {}
        for block, study in f.PARENTS.items():
            for dataset in ("bike", "household"):
                cell_id = f"s{block}_{dataset}"
                candidates = {}
                for method, rates in original.RATES.items():
                    for lr in rates:
                        key = f.prepare.candidate_id(cell_id, method, lr)
                        reuse = method == "F0" or (method == "H_FULL" and lr == 3e-5) or (method == "OFF_LORA" and lr == 1e-4)
                        candidates[key] = {"candidate_id": key, "cell_id": cell_id, "dataset": dataset, "study": study,
                                           "method": method, "lr": lr, "seed": 12000, "needs_forecast": not reuse,
                                           "reuse_forecast": {"role": "OFF_LORA" if method == "OFF_LORA" else "H"} if reuse else None}
                smoke = f.prepare.candidate_id(cell_id, "OFF_LORA", 1e-4)
                cells[cell_id] = {"dataset": dataset, "study_number": block, "study": study, "candidates": candidates,
                                  "selectors": {"LORA_V": smoke}, "smoke_candidate": smoke}
        contract = {"cells": cells}
        for cell_id, cell in cells.items():
            accepted = 0
            for key, candidate in cell["candidates"].items():
                if candidate["needs_forecast"]:
                    self.assertEqual(f.authorize_candidate(contract, cell_id, key)[1], candidate)
                    accepted += 1
                else:
                    with self.assertRaises(AssertionError):
                        f.authorize_candidate(contract, cell_id, key)
            self.assertEqual(accepted, 7)
            self.assertEqual(f.authorize_candidate(contract, cell_id, cell["smoke_candidate"], True)[1]["method"], "OFF_LORA")
            with self.assertRaises(AssertionError):
                f.authorize_candidate(contract, cell_id, f.prepare.candidate_id(cell_id, "OFF_LORA", 1e-5), True)
        altered = copy.deepcopy(contract)
        altered["cells"]["s12_bike"]["candidates"][f.prepare.candidate_id("s12_bike", "H_MLP", 1e-3)]["lr"] = .01
        with self.assertRaisesRegex(AssertionError, "ten saved"):
            f.authorize_candidate(altered, "s12_bike", f.prepare.candidate_id("s12_bike", "OFF_LORA", 1e-5))

    def test_invalid_contract_leaves_incomplete_result_and_never_loads_model(self):
        fake_source = self.root / "experiments" / f.STUDY / "forecast.py"
        fake_source.parent.mkdir(parents=True)
        fake_source.write_text("CPU fixture", encoding="utf-8")
        args = SimpleNamespace(output=str(self.root / "runs" / f.STUDY / "smoke/fail"),
                               contract="missing", cell="s12_bike", candidate="bad", smoke=True)
        with patch.object(f, "__file__", str(fake_source)), patch.object(f, "validate_inputs", side_effect=AssertionError("fixture contract mismatch")), \
                patch.object(f._numerical, "load_base", side_effect=AssertionError("CPU tests must never load native weights")) as load:
            with self.assertRaisesRegex(AssertionError, "contract mismatch"):
                f.run(args)
        self.assertFalse(load.called)
        result = json.loads((Path(args.output) / "result.json").read_text())
        self.assertFalse(result["completed"])
        self.assertFalse((Path(args.output) / "predictions.npz").exists())
        self.assertTrue((Path(args.output) / "failure.json").exists())

    def test_cli_accepts_only_canonical_cells_without_training_knobs(self):
        args = f.parser().parse_args(["--contract", "frozen.json", "--cell", "s13_household", "--candidate",
                                     "s13_household__OFF_LORA__lr_1e-05__seed_12000", "--output", "new", "--smoke"])
        self.assertTrue(args.smoke)
        self.assertFalse(hasattr(args, "steps"))
        self.assertFalse(hasattr(args, "lr"))
        self.assertFalse(hasattr(args, "checkpoint"))


if __name__ == "__main__":
    unittest.main()
