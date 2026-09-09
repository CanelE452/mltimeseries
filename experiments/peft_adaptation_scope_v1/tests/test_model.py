from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from experiments.peft_adaptation_scope_v1.modeling import (
    AdaptationModel, forecast_scores, native_pinball, normalized_to_raw,
    patch_to_quantiles, raw_to_normalized,
)
from experiments.peft_adaptation_scope_v1.train import Panel


def test_native_loss_matches_installed_chronos_with_missing_and_padding():
    from chronos.chronos_bolt import InstanceNorm
    from chronos.chronos2.model import Chronos2Model

    torch.manual_seed(4)
    prediction = torch.randn(3, 3, 32, requires_grad=True)
    target = torch.randn(3, 27)
    target[1, 3] = float("nan")
    target[2, :] = float("nan")
    loc = torch.tensor([[2.0], [0.0], [-3.0]])
    scale = torch.tensor([[3.0], [2.0], [4.0]])
    quantiles = torch.tensor([0.1, 0.5, 0.9])
    native = SimpleNamespace(instance_norm=InstanceNorm(use_arcsinh=True), device=torch.device("cpu"),
                             quantiles=quantiles, chronos_config=SimpleNamespace(output_patch_size=16))
    expected = Chronos2Model._compute_loss(native, prediction, target, None,
                                          torch.zeros(3, 2, 16), (loc, scale), 2)
    actual = native_pinball(prediction, target, loc, scale, quantiles)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.backward()
    assert torch.isfinite(prediction.grad).all()
    assert torch.count_nonzero(prediction.grad[2]) == 0


def test_future_patches_preserve_quantile_and_lead_order():
    value = torch.arange(2 * 3 * 2 * 4).reshape(2, 3, 2 * 4)
    output = patch_to_quantiles(value, quantiles=2, patch_size=4)
    for batch in range(2):
        for quantile in range(2):
            expected = torch.cat([value[batch, patch, quantile * 4:(quantile + 1) * 4]
                                  for patch in range(3)])
            torch.testing.assert_close(output[batch, quantile], expected)


def test_affine_is_in_raw_space_and_shares_channel_parameters():
    class TinyBase(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(dropout_rate=0.1)
            self.chronos_config = SimpleNamespace(output_patch_size=2, use_arcsinh=True)
            self.num_quantiles = 3
            self.device = torch.device("cpu")

    model = AdaptationModel(TinyBase(), "AFF", channels=2)
    with torch.no_grad():
        model.log_scale.copy_(torch.tensor([2.0, 0.5]).log())
        model.offset.copy_(torch.tensor([3.0, -4.0]))
    norm = torch.arange(24, dtype=torch.float32).reshape(4, 3, 2) / 24
    loc = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    scale = torch.tensor([[2.0], [3.0], [4.0], [5.0]])
    adjusted_norm, adjusted_raw = model.from_cache(None, norm, loc, scale)
    raw = normalized_to_raw(norm, loc, scale)
    expected = raw * torch.tensor([2.0, 0.5, 2.0, 0.5])[:, None, None]
    expected += torch.tensor([3.0, -4.0, 3.0, -4.0])[:, None, None]
    torch.testing.assert_close(adjusted_raw, expected)
    torch.testing.assert_close(adjusted_norm, raw_to_normalized(expected, loc, scale))


def test_origin_group_ids_separate_duplicate_sample_instances(tmp_path):
    path = tmp_path / "panel.npz"
    values = np.arange(160 * 2, dtype=np.float32).reshape(160, 2)
    np.savez(path, values=values, context=32, horizon=16, fit_std=np.ones(2),
             channels=np.array(["a", "b"]), train_origins=np.array([32, 48]),
             val_origins=np.array([80]), eval_origins=np.array([112]))
    panel = Panel(path)
    context, target, groups = panel.batch([48, 48], "cpu")
    torch.testing.assert_close(groups, torch.tensor([0, 0, 1, 1]))
    torch.testing.assert_close(context[:2], torch.tensor(values[16:48].T))
    torch.testing.assert_close(target[:2], torch.tensor(values[48:64].T))


def test_metric_valid_denominators_equal_channel_weight():
    target = np.array([[[1.0, np.nan], [8.0, 8.0]], [[1.0, 1.0], [np.nan, 8.0]]])
    prediction = np.zeros((2, 2, 1, 2))
    metrics, individual = forecast_scores(prediction, target, np.array([1.0, 4.0]), np.array([0.5]))
    assert metrics["scaled_2pinball"] == 1.5
    np.testing.assert_array_equal(individual, np.array([[1.0, 2.0], [1.0, 2.0]]))
