import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.peft_external_gap_v1 import train as t


QUANTILES = np.asarray([.01, *np.arange(.05, 1., .05), .99], dtype=np.float64)


def panel_archive(path, dataset="bike", stage="fit", with_holdout=False):
    channels = 5 if dataset == "bike" else 4
    values = np.arange(1056 * channels, dtype=np.float32).reshape(1056, channels) / 100
    target = values.copy()
    target[336, 0] = np.nan
    target[361:364, 1] = np.nan
    loss_mask = np.isfinite(target)
    loss_mask[:, 2:] = False
    arrays = dict(context_values=values, target_values=target,
                  observed_mask=np.isfinite(target), target_loss_mask=loss_mask, quantiles=QUANTILES,
                  channels=np.asarray([f"c{i}" for i in range(channels)]),
                  target_indices=np.asarray([0, 1]), timestamps=np.arange(1056),
                  fit_mean=np.zeros(channels), fit_std=np.arange(1, channels + 1),
                  fit_median=np.zeros(channels), context=np.asarray(336), horizon=np.asarray(48),
                  manifest_json=np.asarray(json.dumps({"dataset": dataset})))
    origins = {"train": np.arange(336, 336 + 20 * 24, 24), "val": np.asarray([864, 888, 912]),
               "cal": np.asarray([864, 888]), "eval": np.asarray([936, 960, 984])}
    for name in (("train", "val") if stage == "fit" else ("cal", "eval")):
        arrays[name + "_origins"] = origins[name]
    if with_holdout:
        arrays["eval_origins"] = origins["eval"]
    np.savez(path, **arrays)
    return arrays


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def selection_fixture(root):
    study = root / "runs" / t.STUDY
    source = root / "source.py"
    source.write_text("fixture source", encoding="utf-8")
    plan = root / t.PLAN
    plan.parent.mkdir(parents=True, exist_ok=True)
    plan.write_text("fixture plan", encoding="utf-8")
    previous = root / "runs/peft_trainlag_v1/study_contract.json"
    write_json(previous, {})
    data = study / "prepared/holdout.npz"
    data.parent.mkdir(parents=True, exist_ok=True)
    data.write_bytes(b"fixture data bytes")
    contract = {"sources": {"source.py": t.file_hash(source)}, "plan_sha256": t.file_hash(plan),
                "previous_contract_sha256": t.file_hash(previous),
                "data_files": {data.relative_to(root).as_posix(): t.file_hash(data)}}
    write_json(study / "study_contract.json", contract)
    entries, choices = [], {}
    for dataset in ("bike", "household"):
        choices[dataset] = {"H": {"method": "H_MLP", "lr": 1e-4},
                            "OFF_LORA": {"method": "OFF_LORA", "lr": 3e-5}}
        for role, method, lr in (("F0", "F0", 0.), ("H", "H_MLP", 1e-4), ("OFF_LORA", "OFF_LORA", 3e-5)):
            for seed in ((12000,) if role == "F0" else (12000, 12001, 12002)):
                path = study / "trials" / dataset / method / str(seed)
                meta = {"completed": True, "smoke": False, "dataset": dataset, "method": method, "seed": seed, "lr": lr}
                write_json(path / "result.json", meta)
                write_json(path / "guard/status.json", {"completed": True, "returncode": 0, "reasons": []})
                (path / "best_trainable.pt").write_bytes(b"fixture checkpoint")
                entries.append({**meta, "role": role, "path": path.relative_to(root).as_posix(),
                                "result_sha256": t.file_hash(path / "result.json"),
                                "checkpoint_sha256": t.file_hash(path / "best_trainable.pt")})
    selection = {"completed": True, "global_choices_frozen": True, "fit_trial_count": 28,
                 "study_contract_sha256": t.file_hash(study / "study_contract.json"),
                 "choices": choices, "selected": entries}
    selection_path = study / "selection.json"
    write_json(selection_path, selection)
    return selection_path, root / entries[0]["path"], selection, data


class TestTrainingContract(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        torch.set_num_threads(2)

    def test_padding_retains_actual_count_and_distinct_groups(self):
        for n in (1, 3, 4):
            padded, count = t.padded_origins(np.arange(n) + 336)
            self.assertEqual(count, n)
            self.assertEqual(len(padded), 4)
            np.testing.assert_array_equal(padded[:n], np.arange(n) + 336)
            self.assertTrue((padded[n:] == 335 + n).all())
        with self.assertRaises(ValueError):
            t.padded_origins([])

    def test_both_panels_mask_only_unobserved_and_non_target_labels(self):
        for dataset in ("bike", "household"):
            path = self.root / f"{dataset}.npz"
            arrays = panel_archive(path, dataset)
            panel = t.Panel(path, "fit")
            origins, _ = t.padded_origins(panel.origins["train"][:3])
            context, target, groups = panel.batch(origins, "cpu")
            c = panel.count_channels
            self.assertEqual(tuple(context.shape), (4 * c, 336))
            np.testing.assert_array_equal(groups, np.repeat(np.arange(4), c))
            expected = np.stack([arrays["target_values"][o:o + 48].T for o in origins])
            expected[:, 2:] = np.nan
            np.testing.assert_array_equal(target.reshape(4, c, 48), expected)
            np.testing.assert_array_equal(panel.context_values, arrays["context_values"])
            self.assertTrue(torch.isnan(target[0, 0]))

    def test_smoke_reads_train_origin_subset_and_purges_labels(self):
        path = self.root / "fit.npz"
        original = panel_archive(path)
        panel = t.Panel(path, "fit", smoke=True)
        np.testing.assert_array_equal(panel.origins["train"], original["train_origins"][:8])
        np.testing.assert_array_equal(panel.origins["val"], original["train_origins"][10:14])
        self.assertLessEqual(panel.origins["train"][-1] + 48, panel.origins["val"][0])
        self.assertFalse(np.isin(panel.origins["val"], original["val_origins"]).any())

    def test_mixed_archive_stage_is_rejected(self):
        path = self.root / "mixed.npz"
        panel_archive(path, with_holdout=True)
        with self.assertRaisesRegex(AssertionError, "mixes fit and holdout"):
            t.Panel(path, "fit")

    def test_target_macro_uses_valid_cell_denominators_and_sort(self):
        target = np.zeros((2, 2, 48))
        target[0, 0] = 10
        target[1, 0, 1:] = np.nan
        target[0, 1, 2:] = np.nan
        prediction = np.broadcast_to(np.linspace(1, -1, 21)[None, None, :, None], (2, 2, 21, 48)).copy()
        original = prediction.copy()
        metric, sums, counts = t.scores(prediction, target, np.asarray([2., 4.]), QUANTILES)
        expected = []
        for c in range(2):
            losses = []
            for o in range(2):
                for h in range(48):
                    if np.isfinite(target[o, c, h]):
                        e = target[o, c, h] - np.sort(prediction[o, c, :, h])
                        losses.extend(2 * np.maximum(QUANTILES * e, (QUANTILES - 1) * e) / [2, 4][c])
            expected.append(np.mean(losses))
        self.assertAlmostEqual(metric["score"], np.mean(expected), places=12)
        np.testing.assert_allclose(sums.sum(0) / counts.sum(0), expected)
        self.assertFalse(np.isclose(metric["score"], np.mean(sums / counts)))
        np.testing.assert_array_equal(prediction, original)

    def test_native_loss_masks_original_missing_and_all_non_targets(self):
        path = self.root / "fit.npz"
        panel_archive(path)
        panel = t.Panel(path, "fit")
        _, target, _ = panel.batch(panel.origins["train"][:4], "cpu")
        prediction = torch.zeros(20, 21, 48, requires_grad=True)
        loss = t.native.native_pinball(prediction, target, torch.zeros(20, 1), torch.ones(20, 1),
                                       torch.tensor(QUANTILES, dtype=torch.float32), False)
        loss.backward()
        gradient = prediction.grad.reshape(4, 5, 21, 48)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertEqual(int(torch.count_nonzero(gradient[:, 2:])), 0)
        self.assertEqual(int(torch.count_nonzero(gradient[0, 0, :, 0])), 0)
        self.assertGreater(int(torch.count_nonzero(gradient[:, :2])), 0)

    def test_prediction_preserves_three_future_patches_and_discards_padding(self):
        path = self.root / "fit.npz"
        panel_archive(path)
        panel = t.Panel(path, "fit")
        seen = []
        class Model:
            method = "H_MLP"
            use_arcsinh = False
            def eval(self):
                pass
            def encode(self, context, groups, horizon):
                seen.append((tuple(context.shape), len(torch.unique(groups)), horizon))
                hidden = torch.arange(3.)[None, :, None].expand(len(context), 3, 768)
                return hidden, torch.zeros(len(context), 21, 48), torch.zeros(len(context), 1), torch.ones(len(context), 1)
            def from_cache(self, hidden, norm, loc, scale):
                residual = hidden[:, :, :1].expand(-1, -1, 336)
                value = t.native.patch_to_quantiles(residual, 21, 16)
                return value, value
        args = SimpleNamespace(device="cpu", min_free_ram_gib=0)
        pred, unsorted = t.predict(Model(), panel, "val", args, return_unsorted=True)
        self.assertEqual(pred.shape, (3, 2, 21, 48))
        self.assertEqual(seen, [((20, 336), 4, 48)])
        np.testing.assert_array_equal(pred, unsorted)
        for k in range(3):
            self.assertTrue((pred[..., k * 16:(k + 1) * 16] == k).all())

    def test_empty_and_nonempty_trainable_checkpoint_roundtrip(self):
        for frozen in (False, True):
            model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 2))
            model[0].requires_grad_(False)
            if frozen:
                model.requires_grad_(False)
            named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
            before = t.parameter_hash(named)
            path = self.root / f"checkpoint_{frozen}.pt"
            t.legacy.save_trainable(model, path)
            with torch.no_grad():
                for _, p in named:
                    p.add_(2)
            t.legacy.load_trainable(model, path, "cpu")
            self.assertEqual(before, t.parameter_hash(named))
            self.assertEqual(set(torch.load(path, weights_only=True)), {n for n, _ in named})

    def test_runner_cli_production_and_smoke_contract(self):
        for smoke in (False, True):
            for method in t.METHODS:
                args = SimpleNamespace(method=method, lr=t.RATES[method][0], seed=12000, smoke=smoke,
                                       steps=0 if method == "F0" else (5 if smoke else 200),
                                       val_every=5 if smoke else 40, checkpoint=t.DEFAULT_CHECKPOINT)
                t.validate_fit_args(args)
                args.steps += 1
                with self.assertRaises(ValueError):
                    t.validate_fit_args(args)

    def test_selection_gate_checks_all_fourteen_and_frozen_data(self):
        path, fit, selection, data = selection_fixture(self.root)
        entry, hashes = t.validate_selection(self.root, path, fit)
        self.assertEqual(entry["method"], "F0")
        self.assertIn(str(data), hashes)
        selection["selected"].pop()
        write_json(path, selection)
        with self.assertRaisesRegex(AssertionError, "fourteen"):
            t.validate_selection(self.root, path, fit)

    def test_selection_gate_rejects_changed_holdout_bytes(self):
        path, fit, _, data = selection_fixture(self.root)
        data.write_bytes(b"changed data")
        with self.assertRaisesRegex(AssertionError, "Protected file changed"):
            t.validate_selection(self.root, path, fit)

    def test_forecast_cannot_open_panel_before_global_selection_gate(self):
        args = SimpleNamespace(selection="unused", fit_trial="unused")
        with patch.object(t, "validate_selection", side_effect=AssertionError("not selected")), \
                patch.object(t, "Panel") as panel, patch.object(t, "load_base") as load:
            with self.assertRaisesRegex(AssertionError, "not selected"):
                t.run_forecast(args)
            panel.assert_not_called()
            load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
