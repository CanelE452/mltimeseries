"""Chronos-2 adaptation methods with one explicit information contract."""

import hashlib
import math

import numpy as np
import torch
from torch import nn


METHODS = ("F0", "AFF", "H_LIN", "H_MLP", "H_FULL", "OFF_LORA", "FULL")
CACHED_METHODS = {"F0", "AFF", "H_LIN", "H_MLP", "H_FULL"}
EXPECTED_TRAINABLE = {"F0": 0, "H_LIN": 258384, "H_MLP": 589301,
                      "H_FULL": 3653280, "OFF_LORA": 1206912}


def deterministic_backbone(model):
    from chronos.chronos2.layers import MHA

    model.config.dropout_rate = 0.0
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.0
        if isinstance(module, MHA):
            module.dropout = 0.0
            module.config.dropout_rate = 0.0
    model.eval()


def patch_to_quantiles(output, quantiles, patch_size):
    rows, patches, _ = output.shape
    return output.reshape(rows, patches, quantiles, patch_size).permute(0, 2, 1, 3).reshape(
        rows, quantiles, patches * patch_size
    )


def normalized_to_raw(prediction, loc, scale, use_arcsinh=True):
    prediction = prediction.float()
    if use_arcsinh:
        prediction = prediction.sinh()
    return prediction * scale[:, None, :] + loc[:, None, :]


def raw_to_normalized(prediction, loc, scale, use_arcsinh=True):
    prediction = (prediction.float() - loc[:, None, :]) / scale[:, None, :]
    return prediction.asinh() if use_arcsinh else prediction


def native_pinball(prediction, target, loc, scale, quantiles, use_arcsinh=True):
    """Native mean-horizon, sum-quantile, mean-series reduction, including masked zeros."""
    target = raw_to_normalized(target[:, None, :], loc, scale, use_arcsinh)
    mask = torch.isfinite(target)
    target = torch.where(mask, target, 0.0)
    if target.shape[-1] < prediction.shape[-1]:
        padding = prediction.shape[-1] - target.shape[-1]
        target = torch.nn.functional.pad(target, (0, padding))
        mask = torch.nn.functional.pad(mask, (0, padding), value=False)
    error = target - prediction.float()
    loss = 2 * torch.abs(error * ((target <= prediction).float() - quantiles[None, :, None]))
    return (loss * mask).mean(dim=-1).sum(dim=-1).mean()


def frozen_digest(model):
    digest = hashlib.sha256()
    count = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            digest.update(name.encode())
            digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
            count += parameter.numel()
    return digest.hexdigest(), count


class AdaptationModel(nn.Module):
    def __init__(self, base, method, channels):
        super().__init__()
        if method not in METHODS:
            raise ValueError(method)
        self.base = base
        self.method = method
        self.channels = channels
        self.patch_size = base.chronos_config.output_patch_size
        self.n_quantiles = base.num_quantiles
        self.use_arcsinh = base.chronos_config.use_arcsinh
        self.module_map = []
        base.requires_grad_(False)
        if method == "AFF":
            self.log_scale = nn.Parameter(torch.zeros(channels, device=base.device))
            self.offset = nn.Parameter(torch.zeros(channels, device=base.device))
        elif method == "H_LIN":
            self.probe = nn.Linear(base.model_dim, self.n_quantiles * self.patch_size).to(base.device)
            nn.init.zeros_(self.probe.weight)
            nn.init.zeros_(self.probe.bias)
        elif method == "H_MLP":
            self.probe = nn.Sequential(
                nn.Linear(base.model_dim, 533), nn.ReLU(),
                nn.Linear(533, self.n_quantiles * self.patch_size),
            ).to(base.device)
            nn.init.zeros_(self.probe[-1].weight)
            nn.init.zeros_(self.probe[-1].bias)
        elif method == "H_FULL":
            base.output_patch_embedding.requires_grad_(True)
        elif method == "OFF_LORA":
            # Import must fail explicitly if PEFT is unavailable; full FT is never a fallback.
            from peft import LoraConfig, inject_adapter_in_model

            self.module_map = [
                name for name, module in base.named_modules()
                if isinstance(module, nn.Linear) and (
                    any(name.endswith("self_attention." + part) for part in ("q", "k", "v", "o"))
                    or name == "output_patch_embedding.output_layer"
                )
            ]
            if len(self.module_map) != 97:
                raise AssertionError(f"Expected 96 attention projections plus head, got {self.module_map}")
            inject_adapter_in_model(LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0,
                                               bias="none", target_modules=self.module_map), base)
        elif method == "FULL":
            base.requires_grad_(True)
        deterministic_backbone(base)
        self.trainable_names = [name for name, p in self.named_parameters() if p.requires_grad]
        self.trainable_count = sum(p.numel() for p in self.parameters() if p.requires_grad)
        expected = channels * 2 if method == "AFF" else EXPECTED_TRAINABLE.get(method)
        if expected is not None and self.trainable_count != expected:
            raise AssertionError(f"{method}: {self.trainable_count} != {expected}")
        if method == "FULL" and any(not p.requires_grad for p in base.parameters()):
            raise AssertionError("FULL contains frozen parameters")

    def from_cache(self, hidden, base_norm, loc, scale):
        if self.method == "H_FULL":
            prediction = patch_to_quantiles(self.base.output_patch_embedding(hidden),
                                            self.n_quantiles, self.patch_size).float()
        elif self.method in {"H_LIN", "H_MLP"}:
            residual = patch_to_quantiles(self.probe(hidden), self.n_quantiles, self.patch_size)
            prediction = base_norm.float() + residual.float()
        elif self.method == "AFF":
            raw = normalized_to_raw(base_norm, loc, scale, self.use_arcsinh)
            channel = torch.arange(raw.shape[0], device=raw.device) % self.channels
            raw = raw * self.log_scale[channel].exp()[:, None, None] + self.offset[channel, None, None]
            prediction = raw_to_normalized(raw, loc, scale, self.use_arcsinh)
        else:
            prediction = base_norm.float()
        return prediction, normalized_to_raw(prediction, loc, scale, self.use_arcsinh)

    def encode(self, context, group_ids, horizon):
        patches = math.ceil(horizon / self.patch_size)
        encoded, (loc, scale), _, _ = self.base.encode(
            context=context, group_ids=group_ids, num_output_patches=patches
        )
        hidden = encoded.last_hidden_state[:, -patches:]
        norm = patch_to_quantiles(self.base.output_patch_embedding(hidden),
                                 self.n_quantiles, self.patch_size).float()
        return hidden, norm, loc, scale

    def from_context(self, context, group_ids, horizon):
        _, norm, loc, scale = self.encode(context, group_ids, horizon)
        raw = normalized_to_raw(norm, loc, scale, self.use_arcsinh)
        return norm, raw, loc, scale


def forecast_scores(predictions, target, fit_std, quantiles):
    """Inputs N,C,Q,H and N,C,H; equal target weights after valid-cell means."""
    predictions = np.asarray(predictions, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    valid = np.isfinite(target)
    if not np.isfinite(predictions).all():
        raise FloatingPointError("Nonfinite forecast")
    error = target[:, :, None, :] - predictions
    loss = 2 * np.maximum(quantiles[None, None, :, None] * error,
                          (quantiles[None, None, :, None] - 1) * error)
    loss = np.where(valid[:, :, None, :], loss, 0.0) / fit_std[None, :, None, None]
    counts = valid.sum(axis=-1) * len(quantiles)
    per_origin_channel = np.divide(loss.sum(axis=(2, 3)), counts,
                                   out=np.full(counts.shape, np.nan), where=counts > 0)
    channel_counts = counts.sum(axis=0)
    per_channel = np.divide(loss.sum(axis=(0, 2, 3)), channel_counts,
                           out=np.full(channel_counts.shape, np.nan), where=channel_counts > 0)
    mean_prediction = predictions.mean(axis=2)
    median_prediction = predictions[:, :, int(np.argmin(abs(quantiles - 0.5))), :]
    mse = np.where(valid, (target - mean_prediction) ** 2, 0.0).sum(axis=(0, 2)) / valid.sum(axis=(0, 2))
    median_mae = np.where(valid, abs(target - median_prediction), 0.0).sum(axis=(0, 2)) / valid.sum(axis=(0, 2))
    lower = predictions[:, :, int(np.argmin(abs(quantiles - 0.1))), :]
    upper = predictions[:, :, int(np.argmin(abs(quantiles - 0.9))), :]
    coverage = ((target >= lower) & (target <= upper) & valid).sum() / valid.sum()
    width = np.where(valid, upper - lower, 0.0).sum() / valid.sum()
    return {
        "scaled_2pinball": float(np.nanmean(per_channel)),
        "per_channel_scaled_2pinball": per_channel.tolist(),
        "qmean_mse": float(np.nanmean(mse)),
        "qmean_mse_definition": "arithmetic average of the 21 predicted quantiles, not an identified conditional mean",
        "median_mae": float(np.nanmean(median_mae)),
        "interval80_coverage": float(coverage), "interval80_width": float(width),
        "quantile_crossing_rate": float((np.diff(predictions, axis=2) < 0).mean()),
    }, per_origin_channel.astype(np.float32)
