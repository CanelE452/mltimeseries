import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.peft_trainlag_v1 import train


torch, np = train.shared.torch, train.shared.np


class ToyNative(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gain = torch.nn.Parameter(torch.tensor(.5))
        self.num_quantiles = 21
        self.chronos_config = SimpleNamespace(output_patch_size=16, use_arcsinh=False)

    def encode(self, context, group_ids, num_output_patches=1, future_covariates=None,
               future_covariates_mask=None, future_target=None, future_target_mask=None):
        loc = context.nanmean(-1, keepdim=True)
        future = torch.zeros_like(loc)
        if future_covariates is not None:
            future = torch.where(future_covariates_mask > 0, future_covariates,
                                 torch.zeros_like(future_covariates)).mean(-1, keepdim=True)
        value = loc + future
        grouped = torch.stack([value[group_ids == group].mean() for group in group_ids])
        hidden = (self.gain * grouped).reshape(12, 1, 1).expand(12, 1, 768)
        return SimpleNamespace(last_hidden_state=hidden), (loc, torch.ones_like(loc)), None, 16

    def output_patch_embedding(self, hidden):
        return hidden[:, :, :1].expand(12, 1, 336)


class TrainLagTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def context(self):
        return torch.arange(12 * 256, dtype=torch.float32).reshape(12, 256) / 100

    def arguments(self, directory):
        return train.parser().parse_args(["--data", "data.npz", "--output", directory,
                     "--cache", str(Path(directory) / "cache"), "--method", "ATTN_ONLY",
                     "--input-mode", "aligned", "--lag-file", "lag.json", "--lr", "3e-5", "--seed", "8100"])

    def test_all_source_indices_are_observed_with_boundary_offsets(self):
        for lag in range(16, 129):
            past, future = train.source_indices(lag)
            np.testing.assert_array_equal(past, np.arange(256 - lag))
            np.testing.assert_array_equal(future + lag, 256 + np.arange(16))
            self.assertGreaterEqual(min(past.min(), future.min()), 0)
            self.assertLess(max(past.max(), future.max()), 256)
        for invalid in (15, 129, 48.5, True):
            with self.assertRaises(ValueError):
                train.source_indices(invalid)

    def test_distinct_driver_lags_nan_padding_future_mask_and_input_immutability(self):
        original = self.context()
        saved = original.clone()
        lags = {"Y": 32, "U": 16, "V": 128}
        context, future, mask = train.transformed_inputs(original, "aligned", lags)
        self.assertTrue(torch.equal(original, saved))
        self.assertTrue(torch.equal(context[::3], original[::3]))
        self.assertTrue(torch.equal(future[::3], torch.zeros((4, 16))))
        self.assertEqual(int(mask[::3].sum()), 0)
        for channel, lag in ((1, 16), (2, 128)):
            self.assertTrue(torch.isnan(context[channel::3, :lag]).all())
            self.assertTrue(torch.equal(context[channel::3, lag:], original[channel::3, :256-lag]))
            self.assertTrue(torch.equal(future[channel::3], original[channel::3, 256-lag:272-lag]))
            self.assertTrue(torch.equal(mask[channel::3], torch.ones((4, 16))))

    def test_common_lag_matches_the_previous_inference_diagnostic(self):
        from experiments.peft_shift_mechanism_v1.diagnose_alignment import aligned_inputs
        original = self.context()
        for lag in (32, 48, 64):
            expected = aligned_inputs(original.numpy().reshape(4, 3, 256), lag)
            observed = train.transformed_inputs(original, "aligned", {"Y": 32, "U": lag, "V": lag})
            for actual, reference in zip(observed, expected):
                np.testing.assert_equal(actual.numpy(), reference.reshape(actual.shape))

    def test_raw_is_the_exact_original_encode_and_rejects_external_covariates(self):
        context = self.context()
        groups = torch.arange(4).repeat_interleave(3)
        base = ToyNative()
        encode = train.encoder(train.shared.encode_y, "raw", {"Y": 32, "U": 48, "V": 64})
        for actual, expected in zip(encode(base, context, groups), train.shared.encode_y(base, context, groups)):
            self.assertTrue(torch.equal(actual, expected))
        shifted, future, mask = train.transformed_inputs(context, "raw", {})
        self.assertIs(shifted, context)
        self.assertIsNone(future)
        self.assertIsNone(mask)
        with self.assertRaises(ValueError):
            encode(base, context, groups, future_covariates=context[:, :16])

    def test_alignment_transports_model_gradients_and_preserves_Y_normalization(self):
        context = self.context().requires_grad_()
        base = ToyNative()
        groups = torch.arange(4).repeat_interleave(3)
        encode = train.encoder(train.shared.encode_y, "aligned", {"Y": 32, "U": 48, "V": 64})
        _, prediction, loc, scale = encode(base, context, groups)
        raw_values = train.shared.encode_y(base, context, groups)
        self.assertTrue(torch.equal(loc, raw_values[2]))
        self.assertTrue(torch.equal(scale, raw_values[3]))
        loss = train.shared.native_pinball(prediction, torch.zeros((4, 16)), loc, scale,
                                           torch.linspace(.01, .99, 21), False)
        loss.backward()
        self.assertTrue(torch.isfinite(base.gain.grad))
        self.assertGreater(float(base.gain.grad.abs()), 0)
        self.assertIsNotNone(context.grad)
        self.assertTrue(torch.isfinite(context.grad).all())

    def test_information_audit_preserves_known_UV_while_mutating_only_masked_targets(self):
        context = self.context()
        groups = torch.arange(4).repeat_interleave(3)
        episodes = SimpleNamespace(batch=lambda split, indices, device: (context, torch.zeros((4, 16)), groups))
        original_encode = train.shared.encode_y
        for mode in train.INPUT_MODES:
            lags = {"Y": 32, "U": 48, "V": 64}
            with patch.object(train.shared, "encode_y", train.encoder(original_encode, mode, lags)):
                result = train.information_auditor(original_encode, mode, lags)(ToyNative(), episodes, "cpu")
            self.assertTrue(result["future_target_isolation"])
            self.assertTrue(result["masked_future_isolation"])
            self.assertTrue(result["group_isolation"])
            self.assertTrue(result["original_input_unchanged"])

    def test_subset_rank_parameters_and_actual_A_B_match_full_native_lora(self):
        full_names = train.ablation.expected_map("OFF_LORA")
        attention = train.ablation.expected_map("ATTN_ONLY")
        def adapters(names):
            modules = {name: SimpleNamespace(
                lora_A={"default": torch.nn.Linear(3072 if name == train.ablation.OUTPUT_MODULE else 768, 8, bias=False)},
                lora_B={"default": torch.nn.Linear(8, 336 if name == train.ablation.OUTPUT_MODULE else 768, bias=False)})
                for name in names}
            return SimpleNamespace(named_modules=lambda: modules.items()), modules
        full, full_modules = adapters(full_names)
        partial, partial_modules = adapters(list(reversed(attention)))
        train.shared.initialize_lora(full, full_names, 8, 8100)
        seeds = train.shared.initialize_lora(partial, list(reversed(attention)), 8, 8100)
        parameters = {}
        for name in attention:
            for side in ("A", "B"):
                actual = getattr(partial_modules[name], "lora_" + side)["default"].weight
                expected = getattr(full_modules[name], "lora_" + side)["default"].weight
                self.assertTrue(torch.equal(actual, expected))
                parameters[f"base.{name}.lora_{side}.default.weight"] = actual
        audit = train.ablation.initialization_audit(attention, parameters, seeds, 8100)
        self.assertTrue(audit["canonical_a_bitwise_verified"])
        self.assertTrue(audit["zero_b_verified"])
        self.assertEqual(sum(p.numel() for p in parameters.values()), 1179648)
        self.assertEqual((len(attention), len(full_names)), (96, 97))
        self.assertNotIn(train.ablation.OUTPUT_MODULE, attention)

    def test_registration_restores_original_globals_and_cannot_stamp_inner_completion(self):
        names = ("METHODS", "INTERNAL_METHODS", "EXPECTED_PARAMETERS", "lora_targets", "ShiftModel",
                 "atomic_json", "prepare_cache", "encode_y", "information_audit", "load_adaptation")
        original = {name: getattr(train.shared, name) for name in names}
        original_heads = train.shared.HEAD_METHODS
        with tempfile.TemporaryDirectory() as directory:
            args = self.arguments(directory)
            state = {"first_B_gradients": {}, "cache_hashes": {}}
            with self.assertRaisesRegex(RuntimeError, "stop"):
                with train.registration(args, {"lags": {"Y": 32, "U": 48, "V": 64}}, state):
                    self.assertIn("ATTN_ONLY", train.shared.INTERNAL_METHODS)
                    self.assertNotIn("ATTN_ONLY", original["INTERNAL_METHODS"])
                    self.assertNotIn("ATTN_ONLY", train.shared.HEAD_METHODS)
                    train.shared.atomic_json(Path(directory) / "result.json", {"completed": True})
                    raise RuntimeError("stop")
            pending = json.loads((Path(directory) / "result.json").read_text())
            self.assertFalse(pending["completed"])
            self.assertFalse(pending["wrapped_completed"])
            self.assertTrue(pending["inner_training_completed"])
        for key, value in original.items():
            self.assertIs(getattr(train.shared, key), value)
        self.assertIs(train.shared.HEAD_METHODS, original_heads)

    def test_cache_contract_separates_mode_lag_source_and_smoke_but_not_learning_rate(self):
        episodes = SimpleNamespace(metadata={"corpus": 0}, targets={"train": np.zeros((64, 16))})
        inputs = {"data_sha256": "data", "lag_file_sha256": "lagfile", "lag_selection_input_sha256": "train",
                  "lags": {"Y": 32, "U": 48, "V": 64}, "checkpoint_hashes": {"weights": "checkpoint"},
                  "source_hashes": {"train.py": "source"}, "native_source_hashes": {}, "plan_sha256": "plan"}
        with tempfile.TemporaryDirectory() as directory:
            args = self.arguments(directory)
            signature = train.cache_signature(train.cache_contract(episodes, args, inputs))
            args.lr = 1e-4
            self.assertEqual(signature, train.cache_signature(train.cache_contract(episodes, args, inputs)))
            for attribute, changed in (("input_mode", "raw"), ("smoke", True)):
                previous = getattr(args, attribute)
                setattr(args, attribute, changed)
                self.assertNotEqual(signature, train.cache_signature(train.cache_contract(episodes, args, inputs)))
                setattr(args, attribute, previous)
            for key, changed in (("lag_file_sha256", "other"), ("source_hashes", {"train.py": "other"}),
                                 ("lags", {"Y": 32, "U": 49, "V": 64})):
                self.assertNotEqual(signature, train.cache_signature(train.cache_contract(episodes, args, {**inputs, key: changed})))

    def test_selection_binds_exact_train_arrays_without_loading_future_splits(self):
        from experiments.peft_trainlag_v1.data import select_lags_from_train
        rng = np.random.default_rng(120)
        context = rng.normal(size=(64, 3, 256)).astype(np.float32)
        target = rng.normal(size=(64, 16)).astype(np.float32)
        selected = {**select_lags_from_train(context, target), "data_sha256": "file"}
        self.assertEqual(train.validate_selection(selected, context, target, "file"), selected["selected_lags"])
        changed = target.copy()
        changed[0, 0] += 1
        with self.assertRaisesRegex(AssertionError, "input hashes"):
            train.validate_selection(selected, context, changed, "file")
        with self.assertRaisesRegex(AssertionError, "information scope"):
            train.validate_selection({**selected, "validation_or_eval_used": True}, context, target, "file")


if __name__ == "__main__":
    unittest.main()
