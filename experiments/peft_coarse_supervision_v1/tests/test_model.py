import hashlib
import importlib
import importlib.util
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

torch.set_num_threads(2)


def module():
    name = "experiments.peft_coarse_supervision_v1.model"
    assert importlib.util.find_spec(name) is not None, "The point-model prototype is not implemented"
    return importlib.import_module(name)


class Config(SimpleNamespace):
    def to_dict(self):
        return vars(self)


class DummyBase(nn.Module):
    """Only one attention projection is used; shared CPU weights avoid a full FM."""

    def __init__(self):
        super().__init__()
        self.model_dim, self.num_quantiles = 768, 21
        self.config = Config(dropout_rate=0.1)
        self.chronos_config = SimpleNamespace(output_patch_size=16, use_arcsinh=True,
                                              context_length=8192, max_output_patches=64)
        self.device = torch.device("cpu")
        shared_weight = nn.Parameter(torch.eye(768) * 0.05)
        self.encoder = nn.Module()
        self.encoder.block = nn.ModuleList()
        for _ in range(12):
            block = nn.Module()
            block.layer = nn.ModuleList()
            for _ in range(2):
                layer = nn.Module()
                layer.self_attention = nn.Module()
                for part in ("q", "k", "v", "o"):
                    projection = nn.Linear(768, 768, bias=False, device="meta")
                    projection.weight = shared_weight
                    setattr(layer.self_attention, part, projection)
                block.layer.append(layer)
            self.encoder.block.append(block)
        self.output_patch_embedding = nn.Linear(768, 336)
        with torch.no_grad():
            self.output_patch_embedding.weight.fill_(0.0001)
            self.output_patch_embedding.bias.copy_(torch.linspace(-0.2, 0.2, 336))
        self.last_call = None

    def encode(self, context, group_ids, num_output_patches):
        self.last_call = (context.shape, group_ids.clone(), num_output_patches)
        hidden = context.mean(dim=1, keepdim=True).expand(-1, 768) * 0.1
        hidden = hidden + self.encoder.block[0].layer[0].self_attention.v(hidden)
        hidden = torch.stack([hidden * (1 + patch * 0.01) for patch in range(num_output_patches)], dim=1)
        loc = context.mean(dim=1, keepdim=True)
        scale = torch.full_like(loc, 2.0)
        return SimpleNamespace(last_hidden_state=hidden), (loc, scale), None, None


def test_native_grid_point_inverts_each_quantile_before_averaging():
    m = module()
    z = torch.linspace(-0.4, 1.2, 42).reshape(1, 21, 2).requires_grad_()
    loc, scale = torch.tensor([[3.0]]), torch.tensor([[2.0]])
    point = m.native_grid_point(z, loc, scale, use_arcsinh=True)
    expected = (3 + 2 * torch.sinh(z)).mean(dim=1)
    torch.testing.assert_close(point, expected, rtol=0, atol=0)
    assert torch.max(torch.abs(point - (3 + 2 * z.mean(dim=1).sinh()))) > 0.05
    point.sum().backward()
    torch.testing.assert_close(z.grad, 2 * z.detach().cosh() / 21)


def test_native_grid_point_without_arcsinh():
    m = module()
    z = torch.linspace(-1, 1, 63).reshape(1, 21, 3)
    result = m.native_grid_point(z, torch.tensor([[4.0]]), torch.tensor([[3.0]]), False)
    torch.testing.assert_close(result, (z * 3 + 4).mean(dim=1))


def test_coarse_loss_matches_scaled_sum_loss_and_ignores_padding():
    m = module()
    point = torch.tensor([[1.0, 2.0, 3.0, 4.0, float("nan")],
                          [4.0, 3.0, 2.0, 1.0, -1e9]], requires_grad=True)
    total, scale = torch.tensor([18.0, 14.0]), torch.tensor([2.0, 0.5])
    actual = m.coarse_loss(point, total, 4, scale)
    expected = (((point[:, :4].sum(dim=1) - total) / (4 * scale))**2).mean()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.backward()
    torch.testing.assert_close(point.grad[:, 4], torch.zeros(2))
    assert torch.isfinite(point.grad).all()


@pytest.mark.parametrize("scale", [0.0, -1.0, float("nan"), float("inf")])
def test_coarse_scale_must_be_fixed_positive_finite(scale):
    with pytest.raises(ValueError, match="scale"):
        module().coarse_loss(torch.ones(1, 16), torch.tensor([20.0]), 16, scale)


def test_coarse_loss_rejects_accidental_batch_broadcast():
    with pytest.raises(ValueError, match="total"):
        module().coarse_loss(torch.ones(2, 16), torch.ones(2, 1), 16, 1.0)


def test_head_encode_crop_scale_and_gradient():
    m = module()
    base = DummyBase()
    model = m.PointModel(base, "HEAD")
    context, groups = torch.ones(2, 32), torch.tensor([0, 1])
    hidden, point, loc, scale = model.encode(context, groups, 17)
    assert (hidden.shape, point.shape, loc.shape, scale.shape) == ((2, 2, 768), (2, 32), (2, 1), (2, 1))
    torch.testing.assert_close(model(context, groups, 17), point[:, :17], rtol=0, atol=0)
    model.load_point_head(np.zeros((16, 768), dtype=np.float32), np.ones(16, dtype=np.float32))
    prediction = model(context, groups, 17)
    torch.testing.assert_close(prediction, point[:, :17] + 2, rtol=0, atol=0)
    m.coarse_loss(prediction, torch.zeros(2), 17, 1.0).backward()
    assert model.trainable_count == 12304
    assert model.point_head.weight.grad.abs().sum() > 0
    assert all(not p.requires_grad and p.grad is None for p in base.parameters())


def test_month_horizon_uses_47_patches_but_returns_744_values():
    m = module()
    base = DummyBase()
    model = m.PointModel(base, "HEAD")
    context, groups = torch.ones(1, 32), torch.tensor([0])
    assert model(context, groups, 744).shape == (1, 744)
    assert base.last_call[2] == 47
    with pytest.raises(ValueError, match="horizon"):
        model(context, groups, 745)


def test_lora_uses_96_attention_modules_and_preserves_the_loaded_head():
    m = module()
    reference = m.PointModel(DummyBase(), "HEAD")
    adapted = m.PointModel(DummyBase(), "ATTN_LORA_FIXED_HEAD", seed=17)
    weight = np.full((16, 768), 0.0002, dtype=np.float32)
    bias = np.linspace(-0.1, 0.1, 16, dtype=np.float32)
    reference.load_point_head(weight, bias)
    with pytest.raises(RuntimeError, match="load_point_head"):
        adapted(torch.ones(1, 32), torch.tensor([0]), 17)
    adapted.load_point_head(weight, bias)
    assert len(adapted.module_map) == 96
    assert not any("output_patch_embedding" in name for name in adapted.module_map)
    assert adapted.trainable_count == 1179648
    assert not any(p.requires_grad for p in adapted.point_head.parameters())
    actual = adapted(torch.ones(1, 32), torch.tensor([0]), 17)
    expected = reference(torch.ones(1, 32), torch.tensor([0]), 17)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    first_name = adapted.module_map[0]
    layer = dict(adapted.base.named_modules())[first_name]
    seed = int.from_bytes(hashlib.sha256(f"17:{first_name}:rank=8".encode()).digest()[:8], "little") % (2**63-1)
    expected_a = torch.empty_like(layer.lora_A.default.weight)
    nn.init.kaiming_uniform_(expected_a, a=math.sqrt(5), generator=torch.Generator().manual_seed(seed))
    torch.testing.assert_close(layer.lora_A.default.weight, expected_a, rtol=0, atol=0)
    assert torch.count_nonzero(layer.lora_B.default.weight) == 0


@pytest.mark.parametrize("path", ["native_point", "fixed_point_head"])
def test_frozen_head_preserves_gradient_to_lora_without_updating_head(path):
    m = module()
    base = DummyBase()
    if path == "fixed_point_head":
        with torch.no_grad():
            base.output_patch_embedding.weight.zero_()
    model = m.PointModel(base, "ATTN_LORA_FIXED_HEAD", seed=3)
    weight = torch.zeros(16, 768) if path == "native_point" else torch.full((16, 768), .001)
    model.load_point_head(weight, torch.zeros(16))
    saved = model.point_head.weight.detach().clone()
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    prediction = model(torch.ones(1, 32), torch.tensor([0]), 17)
    m.coarse_loss(prediction, prediction.detach().sum(dim=1) + 17, 17, 1.0).backward()
    active = model.base.encoder.block[0].layer[0].self_attention.v
    assert active.lora_B.default.weight.grad.abs().sum() > 0
    assert torch.isfinite(active.lora_B.default.weight.grad).all()
    assert torch.count_nonzero(active.lora_A.default.weight.grad) == 0
    assert model.point_head.weight.grad is None and model.point_head.bias.grad is None
    optimizer.step()
    torch.testing.assert_close(saved, model.point_head.weight, rtol=0, atol=0)


def test_head_loading_rejects_transposed_or_nonfinite_weights():
    m = module()
    model = m.PointModel(DummyBase(), "HEAD")
    with pytest.raises(ValueError, match="shape"):
        model.load_point_head(torch.zeros(768, 16), torch.zeros(16))
    with pytest.raises(ValueError, match="finite"):
        model.load_point_head(torch.full((16, 768), float("nan")), torch.zeros(16))


def test_point_head_stays_float32_inside_bfloat16_autocast():
    m = module()
    model = m.PointModel(DummyBase(), "HEAD")
    generator = torch.Generator().manual_seed(28)
    model.load_point_head(torch.randn(16, 768, generator=generator) * .03,
                          torch.randn(16, generator=generator) * .03)
    context, groups = torch.full((1, 32), .12345), torch.tensor([0])
    with torch.autocast("cpu", dtype=torch.bfloat16, cache_enabled=False):
        hidden, base_point, _, scale = model.encode(context, groups, 17)
        actual = model(context, groups, 17)
    expected = (base_point + scale * torch.nn.functional.linear(
        hidden.float(), model.point_head.weight, model.point_head.bias).flatten(1))[:, :17]
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.sum().backward()
    assert model.point_head.weight.grad.abs().sum() > 0
