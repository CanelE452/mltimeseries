"""Explicit native Chronos-2 head, LoRA, and full-model update scopes."""

import hashlib
import os
from pathlib import Path

import numpy as np
import torch

from experiments.peft_adaptation_scope_v1 import modeling as native


ARMS = ("F0", "HEAD_ONLY", "LORA", "FULL_FT")
METHODS = {"F0": "F0", "HEAD_ONLY": "H_FULL", "LORA": "OFF_LORA", "FULL_FT": "FULL"}
EXPECTED_COUNTS = {"F0": 0, "HEAD_ONLY": 3653280, "LORA": 1206912, "FULL_FT": 119477664}
BASE_PARAMETERS = 119477664


def construct(base, arm, seed, channels=1):
    if arm not in ARMS:
        raise ValueError(f"Unknown adaptation scope: {arm}")
    torch.manual_seed(seed)
    model = native.AdaptationModel(base, METHODS[arm], channels)
    model.arm = arm
    audit_scope(model, arm)
    if arm == "LORA":
        expected = [f"encoder.block.{block}.layer.{layer}.self_attention.{part}" for block in range(12)
                    for layer in (0, 1) for part in ("q", "k", "v", "o")]
        expected.append("output_patch_embedding.output_layer")
        if model.module_map != expected:
            raise AssertionError("LoRA differs from the reference 97-projection map")
        if any(torch.count_nonzero(p).item() for name, p in model.named_parameters() if ".lora_B." in name):
            raise AssertionError("LoRA B is not zero initialized")
    return model


def audit_scope(model, arm):
    named = dict(model.named_parameters())
    trainable = {name for name, p in named.items() if p.requires_grad}
    count = sum(named[name].numel() for name in trainable)
    if count != EXPECTED_COUNTS[arm]:
        raise AssertionError(f"Wrong trainable count for {arm}: {count}")
    if arm == "FULL_FT" and (len(trainable) != len(named) or any(not p.requires_grad for p in model.base.parameters())):
        raise AssertionError("FULL_FT must train every native parameter")
    if arm == "HEAD_ONLY" and trainable != {name for name in named if name.startswith("base.output_patch_embedding.")}:
        raise AssertionError("HEAD_ONLY must train precisely the native residual head")
    if arm == "LORA" and any(".lora_A." not in n and ".lora_B." not in n for n in trainable):
        raise AssertionError("LORA contains non-adapter trainable parameters")
    return {"trainable": count, "total_parameters": sum(p.numel() for p in named.values()),
            "frozen_parameters": sum(p.numel() for p in named.values() if not p.requires_grad),
            "trainable_names": sorted(trainable), "full_model_update_scope_verified": arm == "FULL_FT"}


def forward(model, context, groups, horizon=48):
    _, norm, loc, scale = model.encode(context, groups, horizon)
    raw = native.normalized_to_raw(norm, loc, scale, model.use_arcsinh)
    return norm, raw, loc, scale


def parameter_digest(model, trainable=None, prefix=None):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if trainable is not None and parameter.requires_grad != trainable:
            continue
        if prefix is not None and not name.startswith(prefix):
            continue
        value = parameter.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((tuple(value.shape), value.dtype)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def snapshot_trainable(model):
    return {name: p.detach().cpu().clone() for name, p in model.named_parameters() if p.requires_grad}


class MappedBestState:
    def __init__(self, model, path):
        self.path = Path(path)
        parameters = {name: p for name, p in model.named_parameters() if p.requires_grad}
        if not parameters or any(p.dtype != torch.float32 for p in parameters.values()):
            raise ValueError("Mapped best state requires nonempty FP32 trainable parameters")
        if self.path.exists():
            raise FileExistsError("Preserve existing mapped best-state working file")
        count = sum(p.numel() for p in parameters.values())
        self.nbytes = count * np.dtype(np.float32).itemsize
        self._flat = np.memmap(self.path, dtype=np.float32, mode="w+", shape=(count,))
        self.state = {}
        offset = 0
        for name, parameter in parameters.items():
            stop = offset + parameter.numel()
            self.state[name] = torch.from_numpy(self._flat[offset:stop]).reshape(parameter.shape)
            offset = stop
        self.capture(model)

    def capture(self, model):
        if self._flat is None:
            raise RuntimeError("Mapped best state is closed")
        parameters = {name: p for name, p in model.named_parameters() if p.requires_grad}
        if set(parameters) != set(self.state) or any(
            p.shape != self.state[name].shape or p.dtype != torch.float32 for name, p in parameters.items()
        ):
            raise AssertionError("Mapped best-state parameter contract changed")
        with torch.no_grad():
            for name, parameter in parameters.items():
                self.state[name].copy_(parameter.detach())
        self._flat.flush()

    def close(self):
        if self._flat is not None:
            self._flat.flush()
            self.state.clear()
            self._flat._mmap.close()
            self._flat = None


def restore_trainable(model, state):
    parameters = {name: p for name, p in model.named_parameters() if p.requires_grad}
    if set(parameters) != set(state):
        raise AssertionError("Checkpoint trainable parameter names differ")
    for name, parameter in parameters.items():
        value = state[name]
        if value.device.type != "cpu" or value.shape != parameter.shape or value.dtype != parameter.dtype:
            raise AssertionError(f"Checkpoint tensor contract differs: {name}")
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"Nonfinite checkpoint tensor: {name}")
    with torch.no_grad():
        for name, parameter in parameters.items():
            parameter.copy_(state[name])


def write_checkpoint(state, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if path.exists() or temporary.exists():
        raise FileExistsError("Preserve existing checkpoint or partial checkpoint")
    torch.save(state, temporary)
    os.replace(temporary, path)
