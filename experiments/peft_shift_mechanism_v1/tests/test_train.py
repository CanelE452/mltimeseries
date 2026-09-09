from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch import nn

from experiments.peft_shift_mechanism_v1.train import (
    Episodes, ShiftModel, forecast_scores, load_adaptation, lora_targets,
    native_pinball, save_adaptation,
)


class SmallBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.output_patch_embedding = nn.Linear(768, 336)
        self.num_quantiles = 21
        self.config = SimpleNamespace(dropout_rate=.1)
        self.chronos_config = SimpleNamespace(output_patch_size=16, use_arcsinh=False)
        self.device = torch.device("cpu")

    def encode(self, context, group_ids, num_output_patches):
        hidden = context.mean(dim=1)[:, None, None].expand(-1, 1, 768)
        loc = torch.zeros((len(context), 1))
        scale = torch.ones_like(loc)
        return SimpleNamespace(last_hidden_state=hidden), (loc, scale), None, None


class PhaseState(nn.Module):
    parameters_named = ShiftModel.parameters_named
    set_lora_active = ShiftModel.set_lora_active

    def __init__(self):
        super().__init__()
        self.head = nn.Parameter(torch.tensor([3.0]))
        self.lora = nn.Parameter(torch.tensor([7.0]), requires_grad=False)
        self.adaptive_names = ("head", "lora")
        self.lora_names = ("lora",)
        self.lora_active = False


class TrainingContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_episode_batch_has_no_future_input_and_keeps_duplicate_instances_separate(self):
        arrays = {"quantiles": np.linspace(.05, .95, 21)}
        for split in ("train", "val", "eval"):
            arrays["context_" + split] = np.arange(8 * 3 * 256, dtype=np.float32).reshape(8, 3, 256)
            arrays["target_" + split] = np.zeros((8, 16), dtype=np.float32)
            arrays["episode_ids_" + split] = np.asarray([f"{split}_{i}" for i in range(8)])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episodes.npz"
            np.savez(path, **arrays)
            panel = Episodes(path, smoke=True)
            before, _, groups = panel.batch("train", [1, 1, 2, 3], "cpu")
            panel.targets["train"][:] = 999
            after, target, _ = panel.batch("train", [1, 1, 2, 3], "cpu")
        self.assertTrue(torch.equal(before, after))
        self.assertTrue(torch.equal(before[:3], before[3:6]))
        self.assertEqual(groups.tolist(), [0] * 3 + [1] * 3 + [2] * 3 + [3] * 3)
        self.assertEqual(tuple(target.shape), (4, 16))

    def test_native_y_only_loss_and_raw_selection_reduction(self):
        quantiles = torch.linspace(.05, .95, 21)
        prediction = torch.ones((4, 21, 16), requires_grad=True)
        target = torch.zeros((4, 16))
        loss = native_pinball(prediction, target, torch.zeros((4, 1)), torch.ones((4, 1)),
                              quantiles, use_arcsinh=False)
        scores, per_episode = forecast_scores(prediction.detach().numpy(), target.numpy(), quantiles.numpy())
        self.assertAlmostEqual(float(loss.detach()), 21.0, places=5)
        self.assertAlmostEqual(scores["raw_mean_2pinball"], 1.0, places=6)
        self.assertEqual(per_episode.shape, (4,))
        loss.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())

    def test_probe_is_preserved_in_direct_forecast(self):
        torch.manual_seed(4)
        model = ShiftModel(SmallBase(), "H_LIN", 4)
        context = torch.ones((12, 256))
        groups = torch.arange(4).repeat_interleave(3)
        before = model.from_context(context, groups)[0].detach()
        with torch.no_grad():
            model.probe.bias.fill_(2)
        after = model.from_context(context, groups)[0].detach()
        torch.testing.assert_close(after - before, torch.full_like(before, 2), atol=3e-7, rtol=0)

    def test_checkpoint_contains_phase_frozen_parameters(self):
        model = PhaseState()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "best.pt"
            save_adaptation(model, path, 80)
            with torch.no_grad():
                model.head.zero_()
                model.lora.zero_()
            model.set_lora_active(True)
            restored = load_adaptation(model, path, "cpu")
        self.assertEqual(restored["step"], 80)
        self.assertEqual(float(model.head.detach()), 3)
        self.assertEqual(float(model.lora.detach()), 7)
        self.assertFalse(model.lora.requires_grad)

    def test_scope_maps_use_equal_attention_budgets(self):
        attention = {}
        for block in range(12):
            for layer in (0, 1):
                for part in ("q", "k", "v", "o"):
                    attention[f"encoder.block.{block}.layer.{layer}.self_attention.{part}"] = nn.Linear(
                        768, 768, bias=False, device="meta")
        attention["output_patch_embedding.output_layer"] = nn.Linear(3072, 336, device="meta")
        base = SimpleNamespace(named_modules=lambda: attention.items())
        for method, count, rank in (("TIME", 48, 8), ("GROUP", 48, 8),
                                     ("JOINT", 96, 4), ("LP", 96, 4), ("OFF_LORA", 97, 8)):
            names, actual_rank = lora_targets(base, method)
            self.assertEqual((len(names), actual_rank), (count, rank))
            parameters = sum(rank * (attention[name].in_features + attention[name].out_features) for name in names)
            self.assertEqual(parameters, 1206912 if method == "OFF_LORA" else 589824)


if __name__ == "__main__":
    unittest.main()
