import argparse
from contextlib import nullcontext
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.peft_objective_alignment_v1 import train

torch.set_num_threads(2)


def test_equally_spaced_calibration_origins_and_fixed_runtime():
    assert train.calibration_indices(63).tolist() == [0, 8, 17, 26, 35, 44, 53, 62]
    assert train.calibration_indices(8).tolist() == list(range(8))
    with pytest.raises(ValueError):
        train.calibration_indices(7)
    for smoke, expected in ((False, (3e-5, 200, 40)), (True, (1e-5, 5, 5))):
        args = train.runtime_args(argparse.Namespace(smoke=smoke, arm="NATIVE"), "fit.npz")
        assert (args.lr, args.steps, args.val_every) == expected
        assert (args.seed, args.method) == (12000, "OFF_LORA")


def test_calibration_uses_norm_of_accumulated_vector():
    vectors = {"NATIVE": torch.tensor([3., 4.], dtype=torch.float64),
               "NORM_ALIGNED": torch.tensor([6., 8.], dtype=torch.float64),
               "RAW_ALIGNED": torch.tensor([-4., 3.], dtype=torch.float64)}
    result = train.gradient_summary(vectors)
    assert result["norms"] == {"NATIVE": 5., "NORM_ALIGNED": 10., "RAW_ALIGNED": 5.}
    assert result["multipliers"] == {"NATIVE": 1., "NORM_ALIGNED": .5, "RAW_ALIGNED": 1.}
    assert result["cosines"]["NATIVE__NORM_ALIGNED"] == 1.
    assert result["cosines"]["NATIVE__RAW_ALIGNED"] == 0.
    vectors["RAW_ALIGNED"] = torch.zeros(2, dtype=torch.float64)
    with pytest.raises(FloatingPointError):
        train.gradient_summary(vectors)


def test_objective_wrapper_uses_real_loss_api_with_target_scales():
    norm = torch.full((8, 21, 48), .2, requires_grad=True)
    target = torch.ones(8, 48)
    target[1::2] = float("nan")
    panel = SimpleNamespace(target_indices=np.array([0]), count_channels=2,
                            origins={"train": np.arange(8)}, fit_std=np.array([3., 11.]))
    quantiles = torch.linspace(.05, .95, 21)
    args = (norm, target, torch.zeros(8, 1), torch.ones(8, 1), quantiles,
            SimpleNamespace(use_arcsinh=True), panel, np.array([8 * 48]))
    for arm in train.ARMS:
        loss = train.objective(arm, *args)
        assert torch.isfinite(loss)
        gradient, = torch.autograd.grad(loss, norm)
        assert torch.count_nonzero(gradient[1::2]) == 0
        assert torch.count_nonzero(gradient[::2]) > 0


def test_gradient_probe_preserves_parameters_rng_flags_and_clears_gradients(monkeypatch):
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.value = torch.nn.Parameter(torch.tensor(.2))
            self.frozen = torch.nn.Parameter(torch.tensor(7.), requires_grad=False)
            self.child = torch.nn.Dropout(0.)
            self.child.eval()
            self.use_arcsinh = True

    model = Tiny()
    rng = torch.get_rng_state().clone()
    flags = [module.training for module in model.modules()]
    panel = SimpleNamespace(origins={"train": np.arange(8)}, count_channels=2,
                            target_indices=np.array([0, 1]), fit_std=np.array([1., 2.]))
    panel.batch = lambda origins, device: (torch.ones(8, 336), torch.ones(8, 48), torch.arange(4).repeat_interleave(2))
    monkeypatch.setattr(torch.cuda, "get_rng_state", lambda: torch.get_rng_state())
    monkeypatch.setattr(train.legacy, "precision", lambda args: nullcontext())
    monkeypatch.setattr(train.legacy, "check_resources", lambda args: None)
    monkeypatch.setattr(train.shared, "direct", lambda m, x, groups: (
        m.value.expand(8, 21, 48), None, torch.zeros(8, 1), torch.ones(8, 1)))
    result = train.calibrate_gradients(model, panel, SimpleNamespace(device="cpu"),
                                       np.array([384, 384]), torch.linspace(.05, .95, 21))
    assert result["rng_unchanged"] and result["parameters_unchanged"] and result["gradients_cleared"]
    assert torch.equal(rng, torch.get_rng_state())
    assert flags == [module.training for module in model.modules()]
    assert all(parameter.grad is None for parameter in model.parameters())
    assert model.value.item() == pytest.approx(.2)
    assert result["multipliers"]["NATIVE"] == 1.
    assert result["jacobian_summary"]["count"] == 8 * 2 * 21 * 48


def test_native_replay_rejects_prediction_or_history_drift(tmp_path):
    original = {"audits": {"initial_adaptation_sha256": "a", "restored_adaptation_sha256": "b"},
                "sampler_sha256": "s", "best_step": 40, "val_score": .3,
                "validation_history": [{"step": 0, "val_score": .4}, {"step": 40, "val_score": .3}]}
    reference = tmp_path / "old"
    reference.mkdir()
    (reference / "result.json").write_text(json.dumps(original))
    arrays = {"val_predictions": np.ones((2, 2, 21, 48), dtype=np.float32),
              "val_target": np.ones((2, 2, 48)), "val_origins": np.arange(2), "quantiles": np.arange(21),
              "val_loss_sums": np.ones((2, 2)), "val_valid_counts": np.ones((2, 2))}
    np.savez(reference / "predictions.npz", **arrays)
    actual = tmp_path / "actual.npz"
    np.savez(actual, **arrays)
    assert train.verify_native_replay(original, actual, reference)["passed"]
    arrays["val_predictions"][0, 0, 0, 0] += .01
    np.savez(actual, **arrays)
    result = train.verify_native_replay(original, actual, reference)
    assert not result["passed"] and not result["checks"]["val_predictions"]
    assert result["val_predictions_max_abs"] > 0


def test_partial_outputs_and_read_only_cache_contract(tmp_path, monkeypatch):
    monkeypatch.setattr(train, "ROOT", tmp_path)
    output = tmp_path / "runs" / train.STUDY / "trials" / "bike" / "NATIVE"
    output.mkdir(parents=True)
    (output / "failure.json").write_text("preserve")
    with pytest.raises(FileExistsError):
        train.new_output(output)
    assert (output / "failure.json").read_text() == "preserve"
    panel = SimpleNamespace(dataset="bike", channels=["a", "b"], target_indices=np.array([0, 1]),
                            stats_hash="stats", origins={"train": np.arange(8)}, smoke=False)
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "manifest.json").write_text(json.dumps({"completed": True, "contract": {"study": "wrong"}}))
    with pytest.raises(AssertionError, match="read-only"):
        train.read_only_cache(panel, cache, {"data_sha256": "d", "checkpoint_hashes": {}})
