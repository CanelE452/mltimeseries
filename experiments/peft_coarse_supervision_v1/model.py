"""Point-regression prototype for monthly-only target supervision.

The fixed average of raw native quantile outputs is a point function, not a
claim about a conditional mean or the quantiles of an aggregate. This module
loads no data or model weights and performs no training by itself.
"""

import hashlib
import math

import torch
from torch import nn

from experiments.peft_adaptation_scope_v1.modeling import (
    deterministic_backbone, normalized_to_raw, patch_to_quantiles,
)


MODES = ("HEAD", "ATTN_LORA_FIXED_HEAD")
MAX_HORIZON = 744
EXPECTED_TRAINABLE = {"HEAD": 12304, "ATTN_LORA_FIXED_HEAD": 1179648}


def _validate_horizon(horizon, available=None):
    if not isinstance(horizon, int) or not 1 <= horizon <= MAX_HORIZON:
        raise ValueError("horizon must be an integer between 1 and 744")
    if available is not None and horizon > available:
        raise ValueError("horizon exceeds the available point predictions")


def native_grid_point(normalized, loc, scale, use_arcsinh=True):
    """Arithmetic average of the 21 outputs after their individual inversion."""
    if normalized.ndim != 3 or normalized.shape[1] != 21:
        raise ValueError("Expected normalized native outputs with shape [rows,21,time]")
    raw = normalized_to_raw(normalized, loc, scale, use_arcsinh)
    return raw.mean(dim=1)


def coarse_loss(point, monthly_total, horizon, coarse_scale):
    """MSE of monthly average hourly energy, using a supplied coarse-only scale."""
    if point.ndim != 2:
        raise ValueError("point must have shape [rows,time]")
    _validate_horizon(horizon, point.shape[-1])
    total = torch.as_tensor(monthly_total, dtype=torch.float32, device=point.device)
    scale = torch.as_tensor(coarse_scale, dtype=torch.float32, device=point.device)
    if total.ndim == 0 and len(point) == 1:
        total = total.reshape(1)
    if total.shape != (len(point),) or not torch.isfinite(total).all():
        raise ValueError("monthly_total must have one finite total per row")
    if (scale.ndim != 0 and scale.shape != (len(point),)) or not torch.isfinite(scale).all() or torch.any(scale <= 0):
        raise ValueError("coarse_scale must be positive and finite, scalar or one value per row")
    valid_point = point[:, :horizon].float()
    if not torch.isfinite(valid_point).all():
        raise ValueError("Point predictions inside the month must be finite")
    error = (valid_point.mean(dim=-1) - total / horizon) / scale
    return error.square().mean()


class PointModel(nn.Module):
    """Own the provided base; a LoRA mode mutates it by injecting adapters.

    A fixed fitted point head must be stored with any later adapter checkpoint,
    even though it is absent from the LoRA trainable parameter set.
    """

    def __init__(self, base, mode, seed=0):
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"Unknown point-model mode: {mode}")
        if (base.model_dim, base.num_quantiles, base.chronos_config.output_patch_size) != (768, 21, 16):
            raise ValueError("Use the fixed Chronos-2 768-dimensional, 21-quantile, patch16 model")
        if any(hasattr(module, "lora_A") for module in base.modules()):
            raise ValueError("The supplied base already contains LoRA adapters")
        self.base, self.mode = base, mode
        self.patch_size = 16
        self.use_arcsinh = base.chronos_config.use_arcsinh
        self.module_map, self.module_initialization_seeds = [], {}
        self.point_head_loaded = False
        base.requires_grad_(False)
        deterministic_backbone(base)
        self.point_head = nn.Linear(768, 16, device=base.device, dtype=torch.float32)
        nn.init.zeros_(self.point_head.weight)
        nn.init.zeros_(self.point_head.bias)
        self.point_head.requires_grad_(mode == "HEAD")
        if mode == "ATTN_LORA_FIXED_HEAD":
            from peft import LoraConfig, inject_adapter_in_model

            self.module_map = [f"encoder.block.{block}.layer.{layer}.self_attention.{part}"
                               for block in range(12) for layer in (0, 1) for part in ("q", "k", "v", "o")]
            modules = dict(base.named_modules())
            for name in self.module_map:
                module = modules.get(name)
                if not isinstance(module, nn.Linear) or (module.in_features, module.out_features) != (768, 768):
                    raise ValueError(f"The fixed attention module map changed: {name}")
            inject_adapter_in_model(LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0,
                                               bias="none", target_modules=self.module_map), base)
            modules = dict(base.named_modules())
            with torch.no_grad():
                for name in self.module_map:
                    value = int.from_bytes(hashlib.sha256(f"{seed}:{name}:rank=8".encode()).digest()[:8], "little") % (2**63 - 1)
                    generator = torch.Generator(device="cpu").manual_seed(value)
                    a, b = modules[name].lora_A["default"].weight, modules[name].lora_B["default"].weight
                    initial = torch.empty(a.shape, dtype=torch.float32, device="cpu")
                    nn.init.kaiming_uniform_(initial, a=math.sqrt(5), generator=generator)
                    a.copy_(initial.to(a.device))
                    b.zero_()
                    self.module_initialization_seeds[name] = value
        self.trainable_count = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        if self.trainable_count != EXPECTED_TRAINABLE[mode]:
            raise AssertionError(f"Unexpected {mode} trainable count: {self.trainable_count}")
        self.eval()

    def load_point_head(self, weight, bias):
        weight = torch.as_tensor(weight, dtype=torch.float32, device=self.point_head.weight.device)
        bias = torch.as_tensor(bias, dtype=torch.float32, device=self.point_head.bias.device)
        if weight.shape != (16, 768) or bias.shape != (16,):
            raise ValueError("Point-head shape must be weight[16,768] and bias[16]")
        if not torch.isfinite(weight).all() or not torch.isfinite(bias).all():
            raise ValueError("Point-head weights and bias must be finite")
        with torch.no_grad():
            self.point_head.weight.copy_(weight)
            self.point_head.bias.copy_(bias)
        self.point_head.requires_grad_(self.mode == "HEAD")
        self.point_head_loaded = True
        return self

    def encode(self, context, group_ids, horizon):
        _validate_horizon(horizon)
        if context.ndim != 2 or group_ids.shape != (len(context),):
            raise ValueError("Expected context[rows,time] and group_ids[rows]")
        if context.shape[-1] > self.base.chronos_config.context_length:
            raise ValueError("Proxy context would be silently truncated by the native model")
        patches = math.ceil(horizon / self.patch_size)
        encoded, (loc, scale), _, _ = self.base.encode(
            context=context, group_ids=group_ids, num_output_patches=patches)
        hidden = encoded.last_hidden_state[:, -patches:]
        normalized = patch_to_quantiles(self.base.output_patch_embedding(hidden), 21, self.patch_size)
        base_point = native_grid_point(normalized, loc, scale, self.use_arcsinh)
        return hidden, base_point, loc, scale

    def forward(self, context, group_ids, horizon):
        if self.mode == "ATTN_LORA_FIXED_HEAD" and not self.point_head_loaded:
            raise RuntimeError("Call load_point_head with the fitted reference head before LoRA forecasting")
        hidden, base_point, _, scale = self.encode(context, group_ids, horizon)
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            residual = self.point_head(hidden.float()).flatten(start_dim=1)
        point = base_point + scale * residual
        return point[:, :horizon]
