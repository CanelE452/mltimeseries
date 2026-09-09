import hashlib
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest

from experiments.peft_module_ablation_v1 import train


class ModuleAblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        train.shared.torch.set_num_threads(2)

    def linear_base(self):
        modules = {}
        for name in train.expected_map("OFF_LORA"):
            input_size, output_size = (3072, 336) if name == train.OUTPUT_MODULE else (768, 768)
            modules[name] = train.shared.nn.Linear(input_size, output_size, bias=False, device="meta")
        return SimpleNamespace(named_modules=lambda: modules.items())

    def adapter_base(self, names):
        modules = {}
        for name in names:
            input_size, output_size = (3072, 336) if name == train.OUTPUT_MODULE else (768, 768)
            modules[name] = SimpleNamespace(
                lora_A={"default": train.shared.nn.Linear(input_size, 8, bias=False)},
                lora_B={"default": train.shared.nn.Linear(8, output_size, bias=False)})
        return SimpleNamespace(named_modules=lambda: modules.items()), modules

    def test_disjoint_subsets_recover_the_actual_original_map_and_parameter_count(self):
        base = self.linear_base()
        full, rank = train.shared.lora_targets(base, "OFF_LORA")
        output, output_rank = train.subset_targets(base, "OUT_ONLY", train.shared.lora_targets)
        attention, attention_rank = train.subset_targets(base, "ATTN_ONLY", train.shared.lora_targets)
        self.assertEqual((len(full), len(output), len(attention)), (97, 1, 96))
        self.assertEqual((rank, output_rank, attention_rank), (8, 8, 8))
        self.assertFalse(set(output) & set(attention))
        self.assertEqual(set(output) | set(attention), set(full))
        modules = dict(base.named_modules())
        for method in train.CLI_METHODS:
            count = sum(8 * (modules[name].in_features + modules[name].out_features)
                        for name in train.expected_map(method))
            self.assertEqual(count, train.COUNTS[method])

    def test_actual_retained_a_and_b_tensors_match_both_independent_of_traversal(self):
        seed = 7100
        full_names = train.expected_map("OFF_LORA")
        full_base, full_modules = self.adapter_base(full_names)
        full_seeds = train.shared.initialize_lora(full_base, full_names, 8, seed)
        for method in train.NEW_METHODS:
            names = list(reversed(train.expected_map(method)))
            subset_base, modules = self.adapter_base(names)
            seeds = train.shared.initialize_lora(subset_base, names, 8, seed)
            parameters = {}
            for name in names:
                self.assertEqual(seeds[name], full_seeds[name])
                for side in ("A", "B"):
                    actual = getattr(modules[name], "lora_" + side)["default"].weight
                    expected = getattr(full_modules[name], "lora_" + side)["default"].weight
                    self.assertTrue(train.shared.torch.equal(actual, expected))
                    parameters[f"base.{name}.lora_{side}.default.weight"] = actual
            audit = train.initialization_audit(names, parameters, seeds, seed)
            self.assertTrue(audit["canonical_a_bitwise_verified"])
            for name in names:
                original_a = full_modules[name].lora_A["default"].weight.detach().numpy().tobytes()
                self.assertEqual(audit["module_a_initial_sha256"][name], hashlib.sha256(original_a).hexdigest())

    def test_registration_does_not_mutate_original_map_or_add_a_head_and_restores_on_error(self):
        original = {key: getattr(train.shared, key) for key in
                    ("METHODS", "INTERNAL_METHODS", "EXPECTED_PARAMETERS", "lora_targets", "ShiftModel", "atomic_json", "prepare_cache")}
        heads = train.shared.HEAD_METHODS
        base = self.linear_base()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "stop"):
                with train.registration(directory, []):
                    for method in train.NEW_METHODS:
                        self.assertIn(method, train.shared.INTERNAL_METHODS)
                        self.assertNotIn(method, train.shared.HEAD_METHODS)
                        self.assertNotIn(method, original["INTERNAL_METHODS"])
                    self.assertIs(heads, train.shared.HEAD_METHODS)
                    self.assertEqual(train.shared.lora_targets(base, "OFF_LORA"), original["lora_targets"](base, "OFF_LORA"))
                    raise RuntimeError("stop")
        for key, value in original.items():
            self.assertIs(getattr(train.shared, key), value)

    def test_inner_result_cannot_be_mistaken_for_wrapper_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            with train.registration(directory, []):
                train.shared.atomic_json(path, {"completed": True, "method": "OUT_ONLY"})
            value = json.loads(path.read_text())
        self.assertFalse(value["completed"])
        self.assertFalse(value["wrapped_completed"])
        self.assertTrue(value["inner_training_completed"])

    def test_post_training_audit_failure_keeps_completion_false(self):
        samples = train.shared.np.random.default_rng(7100).integers(8, size=(5, 8))
        result = {"completed": True, "method": "OUT_ONLY", "smoke": True,
                  "trainable_parameters": 27264, "module_map": [train.OUTPUT_MODULE],
                  "head_lr": None, "head_only_updates": 0, "phase_transitions": [],
                  "steps_completed": 5, "seed": 7100,
                  "sampler_sha256": hashlib.sha256(samples.astype(train.shared.np.int64).tobytes()).hexdigest()}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            protected = output / "protected.txt"
            protected.write_text("changed")
            with train.registration(output, []):
                train.shared.atomic_json(output / "result.json", result)
            inputs = {"protected_hashes": {str(protected): "wrong-prior-hash"}}
            with self.assertRaisesRegex(AssertionError, "Protected"):
                train.finish_result(output, result, inputs, [{}], {}, time.perf_counter())
            value = json.loads((output / "result.json").read_text())
        self.assertFalse(value["completed"])
        self.assertFalse(value["wrapped_completed"])

    def test_cli_keeps_shared_arguments_and_limits_original_off_to_smoke(self):
        arguments = ["--data", "data.npz", "--output", "trial", "--cache", "cache",
                     "--method", "OUT_ONLY", "--seed", "7100", "--lr", "3e-5"]
        args = train.parser().parse_args(arguments)
        train.validate_args(args)
        self.assertEqual((args.steps, args.val_every, args.lp_fraction), (200, 40, .4))
        args.method = "OFF_LORA"
        with self.assertRaises(ValueError):
            train.validate_args(args)
        args.smoke = True
        train.validate_args(args)


if __name__ == "__main__":
    unittest.main()
