"""Objective-aligned losses for the PEFT objective alignment diagnostic."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from experiments.peft_adaptation_scope_v1 import modeling as native


NATIVE = "NATIVE"
NORM_ALIGNED = "NORM_ALIGNED"
RAW_ALIGNED = "RAW_ALIGNED"
ARMS = (NATIVE, NORM_ALIGNED, RAW_ALIGNED)


def training_denominators(
    target_values: np.ndarray,
    origins: Sequence[int] | np.ndarray,
    target_indices: Sequence[int] | np.ndarray,
    horizon: int,
) -> np.ndarray:
    """Count train valid target cells D_j over origin windows, with overlaps retained."""
    values = np.asarray(target_values)
    origins = np.asarray(origins, dtype=np.int64)
    targets = np.asarray(target_indices, dtype=np.int64)
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if origins.ndim != 1 or targets.ndim != 1 or len(targets) == 0:
        raise ValueError("origins and target_indices must be non-empty vectors")
    if len(np.unique(targets)) != len(targets) or np.any(targets < 0):
        raise ValueError("target_indices must be unique non-negative channel indices")
    if values.ndim == 2:
        if values.shape[1] <= int(targets.max(initial=0)):
            raise ValueError("target_indices exceed target_values channels")
        counts = []
        for channel in targets:
            total = 0
            for origin in origins:
                if origin < 0 or origin + horizon > values.shape[0]:
                    raise ValueError("origin window is outside target_values")
                total += int(np.isfinite(values[origin:origin + horizon, channel]).sum())
            counts.append(total)
        return np.asarray(counts, dtype=np.int64)
    if values.ndim == 3:
        if values.shape[1] <= int(targets.max(initial=0)) or horizon > values.shape[2]:
            raise ValueError("target_indices or horizon exceed target_values")
        if np.any(origins < 0) or np.any(origins >= values.shape[0]):
            raise ValueError("origin index is outside target_values")
        windows = values[origins][:, targets, :horizon]
        return np.isfinite(windows).sum(axis=(0, 2)).astype(np.int64)
    raise ValueError("target_values must be [T,C] or [N,C,H]")


def compute_loss(
    arm: str,
    norm: torch.Tensor,
    target: torch.Tensor,
    loc: torch.Tensor,
    scale: torch.Tensor,
    quantiles: torch.Tensor | np.ndarray | Sequence[float],
    use_arcsinh: bool,
    *,
    target_indices: Sequence[int] | np.ndarray,
    channel_count: int,
    train_valid_counts: Sequence[int] | np.ndarray | torch.Tensor,
    train_origin_count: int,
    train_target_scale: Sequence[float] | np.ndarray | torch.Tensor | None = None,
) -> torch.Tensor:
    """Return one microbatch loss for a declared objective-alignment arm.

    For aligned arms this is the N/B unbiased estimator of the train target-macro
    Q-mean objective.  If the caller uses two B=4 microbatches for one effective
    batch of eight origins, it should backpropagate each returned value with the
    existing `/ 2` accumulation factor.
    """
    arm = _canonical_arm(arm)
    if arm == NATIVE:
        return native.native_pinball(norm, target, loc, scale, _as_tensor(quantiles, norm), use_arcsinh)

    norm, target, loc, scale = _validate_tensors(norm, target, loc, scale)
    q = _as_tensor(quantiles, norm)
    if q.ndim != 1 or q.numel() == 0 or not torch.all((q > 0) & (q < 1)) or not torch.all(q[1:] > q[:-1]):
        raise ValueError("quantiles must be a strictly increasing vector inside (0, 1)")
    if channel_count <= 0 or norm.shape[0] % channel_count != 0:
        raise ValueError("norm rows must be divisible by channel_count")
    batch = norm.shape[0] // channel_count
    if train_origin_count <= 0 or batch <= 0:
        raise ValueError("train_origin_count and batch size must be positive")

    indices = torch.as_tensor(np.asarray(target_indices, dtype=np.int64), device=norm.device, dtype=torch.long)
    if indices.ndim != 1 or indices.numel() == 0:
        raise ValueError("target_indices must be a non-empty vector")
    if torch.unique(indices).numel() != indices.numel() or torch.any(indices < 0) or torch.any(indices >= channel_count):
        raise ValueError("target_indices must be unique legal channel ids")
    target_count = int(indices.numel())

    counts = _as_tensor(train_valid_counts, norm).reshape(-1)
    if counts.numel() != target_count or not torch.all(torch.isfinite(counts)) or torch.any(counts <= 0):
        raise ValueError("train_valid_counts must be positive and match target_indices")

    h = target.shape[-1]
    pred = norm.reshape(batch, channel_count, q.numel(), h).index_select(1, indices)
    y_raw = target.reshape(batch, channel_count, h).index_select(1, indices)
    row_loc = loc.reshape(batch, channel_count, -1).index_select(1, indices)
    row_scale = scale.reshape(batch, channel_count, -1).index_select(1, indices)
    if row_loc.shape[-1] != 1 or row_scale.shape[-1] != 1 or torch.any(row_scale <= 0):
        raise ValueError("loc/scale must provide one positive normalization scale per row")
    valid = torch.isfinite(y_raw)

    if arm == NORM_ALIGNED:
        y = ((y_raw[:, :, None, :] - row_loc[:, :, None, :]) / row_scale[:, :, None, :]).squeeze(2)
        if use_arcsinh:
            y = torch.asinh(y)
        pred_for_loss = torch.sort(pred, dim=2).values
        divisor = None
    else:
        pred_for_loss = pred
        if use_arcsinh:
            pred_for_loss = torch.sinh(pred_for_loss)
        pred_for_loss = pred_for_loss * row_scale[:, :, None, :] + row_loc[:, :, None, :]
        pred_for_loss = torch.sort(pred_for_loss, dim=2).values
        y = y_raw
        if train_target_scale is None:
            raise ValueError("RAW_ALIGNED requires train_target_scale")
        divisor = _as_tensor(train_target_scale, norm).reshape(-1)
        if divisor.numel() != target_count or not torch.all(torch.isfinite(divisor)) or torch.any(divisor <= 0):
            raise ValueError("train_target_scale must be positive and match target_indices")

    y = torch.where(valid, y, torch.zeros_like(y))
    error = y[:, :, None, :] - pred_for_loss
    pinball = 2 * torch.maximum(q[None, None, :, None] * error, (q[None, None, :, None] - 1) * error)
    if divisor is not None:
        pinball = pinball / divisor[None, :, None, None]
    qmean = pinball.mean(dim=2)
    masked = torch.where(valid, qmean, torch.zeros_like(qmean))
    per_target_sum = masked.sum(dim=(0, 2))
    return (float(train_origin_count) / float(batch)) * torch.mean(per_target_sum / counts)


def _canonical_arm(arm: str) -> str:
    if arm in ARMS:
        return arm
    raise ValueError(f"unknown objective arm {arm!r}")


def _as_tensor(value, like: torch.Tensor) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=like.device, dtype=like.dtype)
    return torch.as_tensor(value, device=like.device, dtype=like.dtype)


def _validate_tensors(norm: torch.Tensor, target: torch.Tensor, loc: torch.Tensor, scale: torch.Tensor):
    if norm.ndim != 3:
        raise ValueError("norm must be [rows, quantiles, horizon]")
    if target.shape != (norm.shape[0], norm.shape[2]):
        raise ValueError("target must be [rows, horizon]")
    if loc.shape[0] != norm.shape[0] or scale.shape[0] != norm.shape[0]:
        raise ValueError("loc/scale row counts must match norm")
    if loc.numel() != norm.shape[0] or scale.numel() != norm.shape[0]:
        raise ValueError("loc/scale must contain one scalar per row")
    if not torch.all(torch.isfinite(norm)) or not torch.all(torch.isfinite(loc)) or not torch.all(torch.isfinite(scale)):
        raise ValueError("norm, loc and scale must be finite")
    return norm, target.to(device=norm.device, dtype=norm.dtype), loc.to(device=norm.device, dtype=norm.dtype), scale.to(device=norm.device, dtype=norm.dtype)


