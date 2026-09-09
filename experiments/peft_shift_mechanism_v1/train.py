"""Serial episode adaptation trials; launch CUDA trials through the study guard."""

import argparse
from contextlib import nullcontext
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import time
import traceback

for _variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_variable] = "2"
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import psutil
import torch
from torch import nn

from experiments.peft_adaptation_scope_v1.modeling import (
    deterministic_backbone, native_pinball, normalized_to_raw, patch_to_quantiles,
)


METHODS = ("F0", "H_LIN", "H_MLP", "OFF_LORA", "JOINT", "LP", "TIME", "GROUP")
HEAD_METHODS = {"H_LIN", "H_MLP", "JOINT", "LP", "TIME", "GROUP"}
INTERNAL_METHODS = {"OFF_LORA", "JOINT", "LP", "TIME", "GROUP"}
EXPECTED_PARAMETERS = {
    "F0": 0, "H_LIN": 258384, "H_MLP": 589301, "OFF_LORA": 1206912,
    "JOINT": 848208, "LP": 848208, "TIME": 848208, "GROUP": 848208,
}
REVISION = "29ec3766d36d6f73f0696f85560a422f50e8498c"
DEFAULT_CHECKPOINT = (Path.home() / ".cache/huggingface/hub/models--amazon--chronos-2"
                      / "snapshots" / REVISION)
MICRO_GROUPS = 4
EFFECTIVE_GROUPS = 8


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parameter_hash(named_parameters):
    digest = hashlib.sha256()
    for name, parameter in named_parameters:
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def precision(device):
    return (torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False)
            if str(device).startswith("cuda") else nullcontext())


def check_resources():
    if psutil.virtual_memory().available < 5 * 2**30:
        raise RuntimeError("RESOURCE_GUARD: less than 5 GiB available RAM")


def log_event(output, event, **fields):
    record = {"time": time.time(), "event": event, **fields}
    with (output / "progress.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, allow_nan=False) + "\n")
    print(json.dumps(record, allow_nan=False), flush=True)


class Episodes:
    def __init__(self, path, smoke=False):
        self.contexts, self.targets, self.ids = {}, {}, {}
        with np.load(path, allow_pickle=False) as data:
            self.quantiles = np.asarray(data["quantiles"], dtype=np.float64)
            for split in ("train", "val", "eval"):
                context = np.asarray(data["context_" + split], dtype=np.float32)
                target = np.asarray(data["target_" + split], dtype=np.float32)
                if context.ndim != 3 or context.shape[1:] != (3, 256):
                    raise ValueError(f"Invalid {split} context shape: {context.shape}")
                if target.shape != (len(context), 16):
                    raise ValueError(f"Invalid {split} target shape: {target.shape}")
                if not np.isfinite(context).all() or not np.isfinite(target).all():
                    raise ValueError("This complete-episode experiment requires finite inputs/targets")
                ids = np.asarray(data["episode_ids_" + split]).copy()
                if ids.shape != (len(context),) or len(np.unique(ids)) != len(ids):
                    raise ValueError(f"Invalid {split} episode IDs")
                if smoke:
                    context, target, ids = context[:8], target[:8], ids[:8]
                expected_count = 8 if smoke else {"train": 64, "val": 128, "eval": 512}[split]
                if len(context) != expected_count or len(context) % MICRO_GROUPS:
                    raise ValueError("Every split must contain complete four-episode microbatches")
                self.contexts[split], self.targets[split], self.ids[split] = context, target, ids
            self.metadata = {}
            for key in ("condition", "corpus", "corpus_id", "corpus_seed", "data_seed"):
                if key in data and data[key].size == 1:
                    self.metadata[key] = data[key].item()
            if "manifest_json" in data:
                manifest = json.loads(data["manifest_json"].item())
                self.metadata.update({key: manifest[key] for key in ("condition", "corpus", "base_seed")})
                if manifest["input_scope"]["context_rows"] != ["Y_past", "U_past", "V_past"]:
                    raise AssertionError("Dataset channel order differs from Y/U/V")
        if any(np.intersect1d(self.ids[left], self.ids[right]).size
               for left, right in (("train", "val"), ("train", "eval"), ("val", "eval"))):
            raise ValueError("Episode IDs overlap across splits")
        if self.quantiles.shape != (21,) or np.any(np.diff(self.quantiles) <= 0):
            raise ValueError("Expected an ordered grid of 21 quantiles")

    def batch(self, split, indices, device):
        if len(indices) != MICRO_GROUPS:
            raise ValueError("Encoder calls must use exactly four complete groups")
        context = torch.as_tensor(self.contexts[split][indices].reshape(12, 256), device=device)
        target = torch.as_tensor(self.targets[split][indices], device=device)
        groups = torch.arange(MICRO_GROUPS, device=device).repeat_interleave(3)
        return context, target, groups


def lora_targets(base, method):
    layers = {"TIME": (0,), "GROUP": (1,)}.get(method, (0, 1))
    names = [f"encoder.block.{block}.layer.{layer}.self_attention.{part}"
             for block in range(12) for layer in layers for part in ("q", "k", "v", "o")]
    if method == "OFF_LORA":
        names.append("output_patch_embedding.output_layer")
    modules = dict(base.named_modules())
    for name in names:
        if name not in modules or not isinstance(modules[name], nn.Linear):
            raise AssertionError(f"Missing expected LoRA linear: {name}")
        expected = (3072, 336) if name == "output_patch_embedding.output_layer" else (768, 768)
        if (modules[name].in_features, modules[name].out_features) != expected:
            raise AssertionError(f"LoRA module dimensions changed: {name}")
    rank = 4 if method in {"JOINT", "LP"} else 8
    return names, rank


def initialize_lora(base, names, rank, seed):
    modules = dict(base.named_modules())
    seeds = {}
    with torch.no_grad():
        for name in names:
            value = int.from_bytes(hashlib.sha256(f"{seed}:{name}:rank={rank}".encode()).digest()[:8],
                                   "little") % (2**63 - 1)
            generator = torch.Generator(device="cpu").manual_seed(value)
            a, b = modules[name].lora_A["default"].weight, modules[name].lora_B["default"].weight
            initial = torch.empty(a.shape, dtype=torch.float32, device="cpu")
            nn.init.kaiming_uniform_(initial, a=math.sqrt(5), generator=generator)
            a.copy_(initial.to(a.device))
            b.zero_()
            seeds[name] = value
    return seeds


class ShiftModel(nn.Module):
    def __init__(self, base, method, seed):
        super().__init__()
        self.base, self.method = base, method
        self.quantiles = base.num_quantiles
        self.patch_size = base.chronos_config.output_patch_size
        self.use_arcsinh = base.chronos_config.use_arcsinh
        self.probe = None
        self.module_map, self.module_seeds = [], {}
        self.rank = None
        base.requires_grad_(False)
        if method in HEAD_METHODS:
            if method == "H_MLP":
                self.probe = nn.Sequential(nn.Linear(768, 533), nn.ReLU(), nn.Linear(533, 336)).to(base.device)
                nn.init.zeros_(self.probe[-1].weight)
                nn.init.zeros_(self.probe[-1].bias)
            else:
                self.probe = nn.Linear(768, 336).to(base.device)
                nn.init.zeros_(self.probe.weight)
                nn.init.zeros_(self.probe.bias)
        if method in INTERNAL_METHODS:
            from peft import LoraConfig, inject_adapter_in_model

            self.module_map, self.rank = lora_targets(base, method)
            inject_adapter_in_model(LoraConfig(r=self.rank, lora_alpha=2 * self.rank,
                                               lora_dropout=0.0, bias="none",
                                               target_modules=self.module_map), base)
            self.module_seeds = initialize_lora(base, self.module_map, self.rank, seed)
        deterministic_backbone(base)
        self.adaptive_names = tuple(name for name, parameter in self.named_parameters() if parameter.requires_grad)
        self.lora_names = tuple(name for name in self.adaptive_names if ".lora_" in name)
        self.head_names = tuple(name for name in self.adaptive_names if name.startswith("probe."))
        parameters = dict(self.named_parameters())
        self.adaptive_count = sum(parameters[name].numel() for name in self.adaptive_names)
        if self.adaptive_count != EXPECTED_PARAMETERS[method]:
            raise AssertionError(f"{method}: {self.adaptive_count} != {EXPECTED_PARAMETERS[method]}")
        if len(self.lora_names) != 2 * len(self.module_map):
            raise AssertionError("Unexpected trainable LoRA tensor map")
        self.set_lora_active(method != "LP")

    def parameters_named(self, names):
        parameters = dict(self.named_parameters())
        return [(name, parameters[name]) for name in names]

    def set_lora_active(self, active):
        self.lora_active = bool(active and self.lora_names)
        for _, parameter in self.parameters_named(self.lora_names):
            parameter.requires_grad_(self.lora_active)

    def backbone_hash(self):
        adaptive = set(self.adaptive_names)
        return parameter_hash((name, p) for name, p in self.named_parameters() if name not in adaptive)

    def from_features(self, hidden_y, base_norm_y, loc_y, scale_y):
        norm = base_norm_y.float()
        if self.probe is not None:
            residual = patch_to_quantiles(self.probe(hidden_y), self.quantiles, self.patch_size)
            norm = norm + residual.float()
        return norm, normalized_to_raw(norm, loc_y, scale_y, self.use_arcsinh), loc_y, scale_y

    def from_context(self, context, groups):
        hidden, norm, loc, scale = encode_y(self.base, context, groups)
        return self.from_features(hidden, norm, loc, scale)


def encode_y(base, context, groups, **extra):
    encoded, (loc, scale), _, _ = base.encode(context=context, group_ids=groups,
                                             num_output_patches=1, **extra)
    hidden = encoded.last_hidden_state[:, -1:]
    norm = patch_to_quantiles(base.output_patch_embedding(hidden), base.num_quantiles,
                             base.chronos_config.output_patch_size).float()
    rows = torch.arange(0, len(context), 3, device=context.device)
    return hidden[rows], norm[rows], loc[rows], scale[rows]


class Cache:
    def __init__(self, path, manifest):
        self.path, self.manifest = path, manifest
        self.arrays = {split: {name: np.load(path / f"{split}_{name}.npy", mmap_mode="r")
                               for name in ("hidden", "norm", "loc", "scale")}
                       for split in ("train", "val", "eval")}

    def batch(self, split, indices, device):
        return [torch.as_tensor(self.arrays[split][name][indices], device=device)
                for name in ("hidden", "norm", "loc", "scale")]


def prepare_cache(base, episodes, args, output):
    contract = {
        "version": 1, "data_sha256": file_hash(args.data), "metadata": episodes.metadata,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "config_sha256": file_hash(Path(args.checkpoint) / "config.json"),
        "counts": {split: len(values) for split, values in episodes.targets.items()},
        "context": 256, "horizon": 16, "channel_order": ["Y", "U", "V"],
        "micro_groups": 4, "encoder_rows": 12, "probe_rows": 4,
        "autocast": "bfloat16" if args.device.startswith("cuda") else "float32",
        "weight_dtype": "float32", "autocast_weight_cache": False, "dropout": 0.0,
        "feature_inputs": "context only; no future Y/U/V values", "smoke": args.smoke,
        "source_sha256": file_hash(Path(__file__)),
    }
    signature = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()[:20]
    path = Path(args.cache) / signature
    manifest_path = path / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not manifest.get("completed") or manifest["contract"] != contract:
            raise AssertionError("Cache contract mismatch")
        return Cache(path, manifest), False
    path.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with torch.no_grad():
        for split in ("train", "val", "eval"):
            count = len(episodes.targets[split])
            shapes = {"hidden": (count, 1, 768), "norm": (count, 21, 16),
                      "loc": (count, 1), "scale": (count, 1)}
            arrays = {name: np.lib.format.open_memmap(path / f"{split}_{name}.npy", mode="w+",
                                                     dtype=np.float32, shape=shape)
                      for name, shape in shapes.items()}
            for start in range(0, count, MICRO_GROUPS):
                check_resources()
                indices = np.arange(start, start + MICRO_GROUPS)
                context, _, groups = episodes.batch(split, indices, args.device)
                with precision(args.device):
                    values = encode_y(base, context, groups)
                for name, value in zip(("hidden", "norm", "loc", "scale"), values):
                    arrays[name][indices] = value.float().cpu().numpy()
            for array in arrays.values():
                array.flush()
            del arrays
            log_event(output, "cache", split=split, completed=count,
                      seconds=time.perf_counter() - started)
    manifest = {"completed": True, "contract": contract, "wall_seconds": time.perf_counter() - started,
                "checkpoint_weights": {weight.name: file_hash(weight)
                                       for weight in sorted(Path(args.checkpoint).glob("*.safetensors"))}}
    atomic_json(manifest_path, manifest)
    return Cache(path, manifest), True


def information_audit(base, episodes, device):
    context, _, groups = episodes.batch("train", np.arange(4), device)
    with torch.no_grad(), precision(device):
        original = encode_y(base, context, groups)
        artificial = torch.full((12, 16), 123.0, device=device)
        changed_target = encode_y(base, context, groups, future_target=artificial)
        blocked_future = encode_y(base, context, groups, future_covariates=artificial,
                                  future_covariates_mask=torch.zeros_like(artificial))
        other = context.clone()
        other[3:] = 2 * other[3:] + 17
        isolated = encode_y(base, other, groups)
        swapped = context.reshape(4, 3, 256)[:, [0, 2, 1]].reshape(12, 256)
        exchanged = encode_y(base, swapped, groups)
    errors = {
        "future_target_hidden_max_abs": float((original[0] - changed_target[0]).abs().max()),
        "masked_future_hidden_max_abs": float((original[0] - blocked_future[0]).abs().max()),
        "other_group_hidden_max_abs": float((original[0][:1] - isolated[0][:1]).abs().max()),
    }
    if any(value > 1e-5 for value in errors.values()):
        raise AssertionError(f"Information isolation failed: {errors}")
    original_raw = normalized_to_raw(original[1], original[2], original[3], base.chronos_config.use_arcsinh)
    swapped_raw = normalized_to_raw(exchanged[1], exchanged[2], exchanged[3], base.chronos_config.use_arcsinh)
    return {**errors, "future_target_isolation": True, "masked_future_isolation": True,
            "group_isolation": True, "encoder_rows": 12, "groups": 4,
            "uv_swap_y_normalized_max_abs": float((original[1] - exchanged[1]).abs().max()),
            "uv_swap_y_raw_max_abs": float((original_raw - swapped_raw).abs().max()),
            "symmetry_note": "Shared channel projections and unordered group attention imply U/V-swap invariance in exact arithmetic; recorded differences include BF16 rounding. This is a structural constraint, not a learned LoRA mechanism."}


def forecast_scores(prediction, target, quantiles):
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if prediction.shape != (len(target), 21, 16) or target.shape != (len(target), 16):
        raise ValueError("Forecast/target shapes differ from the Y-only contract")
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise FloatingPointError("Nonfinite prediction or target")
    error = target[:, None, :] - prediction
    loss = 2 * np.maximum(quantiles[None, :, None] * error,
                          (quantiles[None, :, None] - 1) * error)
    per_episode = loss.mean(axis=(1, 2))
    median = prediction[:, int(np.argmin(abs(quantiles - .5))), :]
    lower = prediction[:, int(np.argmin(abs(quantiles - .1))), :]
    upper = prediction[:, int(np.argmin(abs(quantiles - .9))), :]
    return {
        "raw_mean_2pinball": float(per_episode.mean()),
        "median_mse": float(np.mean((target - median)**2)),
        "median_mae": float(np.mean(abs(target - median))),
        "qmean_proxy_mse": float(np.mean((target - prediction.mean(axis=1))**2)),
        "qmean_proxy_definition": "Arithmetic average of predicted quantiles; not an identified conditional mean",
        "interval80_coverage": float(np.mean((target >= lower) & (target <= upper))),
        "interval80_width": float(np.mean(upper - lower)),
        "quantile_crossing_rate": float(np.mean(np.diff(prediction, axis=1) < 0)),
        "per_horizon_2pinball": loss.mean(axis=(0, 1)).tolist(),
    }, per_episode


def predict(model, episodes, cache, split, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(episodes.targets[split]), MICRO_GROUPS):
            check_resources()
            indices = np.arange(start, start + MICRO_GROUPS)
            with precision(device):
                if model.method in {"F0", "H_LIN", "H_MLP"}:
                    _, raw, _, _ = model.from_features(*cache.batch(split, indices, device))
                else:
                    context, _, groups = episodes.batch(split, indices, device)
                    _, raw, _, _ = model.from_context(context, groups)
            predictions.append(raw.float().cpu().numpy())
    return np.concatenate(predictions)


def save_adaptation(model, path, step):
    state = {name: parameter.detach().cpu() for name, parameter in model.parameters_named(model.adaptive_names)}
    temporary = path.with_suffix(".tmp")
    torch.save({"parameters": state, "step": step, "lora_active": model.lora_active}, temporary)
    os.replace(temporary, path)


def load_adaptation(model, path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if set(checkpoint["parameters"]) != set(model.adaptive_names):
        raise AssertionError("Adaptive checkpoint map changed or omitted phase-frozen parameters")
    with torch.no_grad():
        for name, parameter in model.parameters_named(model.adaptive_names):
            parameter.copy_(checkpoint["parameters"][name].to(device))
    model.set_lora_active(checkpoint["lora_active"])
    return checkpoint


def optimizer_steps(optimizer):
    if optimizer is None:
        return []
    return [int(state["step"].item()) for state in optimizer.state.values() if "step" in state]


def run(args):
    started = time.perf_counter()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "result.json").exists():
        raise FileExistsError(f"Refusing to overwrite an existing trial: {output}")
    check_resources()
    source_before = file_hash(Path(__file__))
    data_before = file_hash(args.data)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed)
    if args.device.startswith("cuda"):
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("CUDA BF16 support is required")
        torch.cuda.set_per_process_memory_fraction(.67)
        torch.cuda.reset_peak_memory_stats()
        torch.backends.cuda.matmul.allow_tf32 = False
    if args.smoke:
        args.steps = min(args.steps, 5)
        args.val_every = 1
    head_only_steps = int(args.steps * args.lp_fraction)
    if args.method == "LP" and not 0 < head_only_steps < args.steps:
        raise ValueError("LP needs at least one update in each phase")
    episodes = Episodes(args.data, args.smoke)
    from chronos.chronos2.model import Chronos2Model

    old_source = Path(__file__).resolve().parents[1] / "peft_adaptation_scope_v1"
    frozen_sources = {name: file_hash(old_source / name) for name in ("modeling.py", "train.py", "data.py")}
    log_event(output, "loading", method=args.method, checkpoint=args.checkpoint)
    base = Chronos2Model.from_pretrained(args.checkpoint, local_files_only=True,
                                        dtype=torch.float32, attn_implementation="sdpa").to(args.device)
    deterministic_backbone(base)
    base.requires_grad_(False)
    if not np.allclose(episodes.quantiles, base.chronos_config.quantiles, rtol=0, atol=1e-7):
        raise AssertionError("Data and native model quantile grids differ")
    audits = (information_audit(base, episodes, args.device) if args.smoke else
              {"information_audit": "S0; production uses the same context-only four-group encoder"})
    cache, cache_created = prepare_cache(base, episodes, args, output)
    if cache.manifest["contract"]["data_sha256"] != data_before:
        raise AssertionError("Data changed while loading or creating the cache")
    torch.manual_seed(args.seed)
    model = ShiftModel(base, args.method, args.seed)
    initial_adaptation_hash = parameter_hash(model.parameters_named(model.adaptive_names))
    initial_lora_hash = parameter_hash(model.parameters_named(model.lora_names))
    backbone_before = model.backbone_hash()
    identity_errors = []
    with torch.no_grad(), precision(args.device):
        for start in (0, 4):
            indices = np.arange(start, start + 4)
            context, _, groups = episodes.batch("train", indices, args.device)
            cached = cache.batch("train", indices, args.device)
            direct, _, _, _ = model.from_context(context, groups)
            identity_errors.append(float((direct - cached[1]).abs().max()))
            if args.method in {"F0", "H_LIN", "H_MLP"}:
                probed, _, _, _ = model.from_features(*cached)
                identity_errors.append(float((probed - direct).abs().max()))
    if max(identity_errors) > 1e-5:
        raise AssertionError(f"Four-group zero-update identity failed: {identity_errors}")
    audits.update({"zero_update_identity": True, "zero_update_normalized_max_abs": max(identity_errors),
                   "zero_update_groups_checked": 8, "trainable_map_verified": True,
                   "backbone_before_sha256": backbone_before,
                   "initial_adaptation_sha256": initial_adaptation_hash})
    head_parameters = [p for _, p in model.parameters_named(model.head_names)]
    lora_parameters = [p for _, p in model.parameters_named(model.lora_names)]
    head_lr = args.lr if args.method in {"H_LIN", "H_MLP"} else 1e-3
    head_optimizer = (torch.optim.AdamW(head_parameters, lr=head_lr, weight_decay=0.0, foreach=False)
                      if head_parameters else None)
    lora_optimizer = (torch.optim.AdamW(lora_parameters, lr=args.lr, weight_decay=0.0, foreach=False)
                      if lora_parameters and args.method != "LP" else None)
    registered = {id(p) for optimizer in (head_optimizer, lora_optimizer) if optimizer is not None
                  for group in optimizer.param_groups for p in group["params"]}
    if registered != {id(p) for p in model.parameters() if p.requires_grad}:
        raise AssertionError("Initial optimizer map mismatch")
    quantiles = torch.as_tensor(episodes.quantiles, dtype=torch.float32, device=args.device)
    best_checkpoint = output / "best_adaptation.pt"
    val_pred = predict(model, episodes, cache, "val", args.device)
    val_scores, _ = forecast_scores(val_pred, episodes.targets["val"], episodes.quantiles)
    best_val, best_step = val_scores["raw_mean_2pinball"], 0
    save_adaptation(model, best_checkpoint, 0)
    records = [{"step": 0, "val_score": best_val, "lora_active": model.lora_active}]
    log_event(output, "validation", **records[-1], best_step=0)
    updates = args.steps if model.adaptive_names else 0
    samples = np.random.default_rng(args.seed).integers(len(episodes.targets["train"]),
                                                       size=(updates, EFFECTIVE_GROUPS))
    sampler_hash = hashlib.sha256(samples.astype(np.int64).tobytes()).hexdigest()
    durations, phases = [], []
    for step in range(1, updates + 1):
        check_resources()
        tick = time.perf_counter()
        if args.method == "LP" and step == head_only_steps + 1:
            unchanged = parameter_hash(model.parameters_named(model.lora_names)) == initial_lora_hash
            if not unchanged:
                raise AssertionError("LP LoRA parameters changed during the head-only phase")
            state_before = optimizer_steps(head_optimizer)
            head_optimizer_id = id(head_optimizer)
            model.set_lora_active(True)
            lora_optimizer = torch.optim.AdamW(lora_parameters, lr=args.lr, weight_decay=0.0, foreach=False)
            phases.append({"step": step, "head_only_updates": head_only_steps,
                           "head_optimizer_state_steps_before": state_before,
                           "head_optimizer_identity_preserved": id(head_optimizer) == head_optimizer_id,
                           "lora_unchanged_in_phase1": unchanged, "lora_optimizer_initial_states": 0})
            registered = {id(p) for optimizer in (head_optimizer, lora_optimizer)
                          for group in optimizer.param_groups for p in group["params"]}
            if registered != {id(p) for p in model.parameters() if p.requires_grad}:
                raise AssertionError("Phase2 optimizer map mismatch")
            log_event(output, "phase_transition", **phases[-1])
        model.train()
        for optimizer in (head_optimizer, lora_optimizer):
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        for offset in range(0, EFFECTIVE_GROUPS, MICRO_GROUPS):
            indices = samples[step - 1, offset:offset + MICRO_GROUPS]
            context, target, groups = episodes.batch("train", indices, args.device)
            with precision(args.device):
                if args.method in {"H_LIN", "H_MLP"}:
                    norm, _, loc, scale = model.from_features(*cache.batch("train", indices, args.device))
                else:
                    norm, _, loc, scale = model.from_context(context, groups)
                loss = native_pinball(norm, target, loc, scale, quantiles, model.use_arcsinh)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite training loss at update {step}")
            (loss / 2).backward()
            accumulated_loss += float(loss.detach()) / 2
        active_parameters = [p for p in model.parameters() if p.requires_grad]
        gradient_norm = torch.nn.utils.clip_grad_norm_(active_parameters, 1.0, error_if_nonfinite=True)
        for optimizer in (head_optimizer, lora_optimizer):
            if optimizer is not None:
                optimizer.step()
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        durations.append(time.perf_counter() - tick)
        if step == 1:
            audits["backbone_after_first_update_sha256"] = model.backbone_hash()
            if audits["backbone_after_first_update_sha256"] != backbone_before:
                raise AssertionError("Frozen backbone changed after the first update")
        if step % args.val_every == 0 or step == updates:
            val_pred = predict(model, episodes, cache, "val", args.device)
            val_scores, _ = forecast_scores(val_pred, episodes.targets["val"], episodes.quantiles)
            value = val_scores["raw_mean_2pinball"]
            if value < best_val:
                best_val, best_step = value, step
                save_adaptation(model, best_checkpoint, step)
            record = {"step": step, "val_score": value, "train_loss": accumulated_loss,
                      "gradient_norm": float(gradient_norm), "lora_active": model.lora_active,
                      "optimizer_seconds": durations[-1]}
            records.append(record)
            log_event(output, "validation", **record, best_step=best_step)
        elif step % 10 == 0:
            log_event(output, "training", step=step, train_loss=accumulated_loss,
                      optimizer_seconds=float(np.mean(durations[-10:])))
    audits["backbone_after_training_sha256"] = model.backbone_hash()
    audits["frozen_parameters_verified"] = audits["backbone_after_training_sha256"] == backbone_before
    if not audits["frozen_parameters_verified"]:
        raise AssertionError("Frozen backbone changed during training")
    final_optimizer_states = {"head_steps": optimizer_steps(head_optimizer),
                              "lora_steps": optimizer_steps(lora_optimizer)}
    restored = load_adaptation(model, best_checkpoint, args.device)
    val_pred = predict(model, episodes, cache, "val", args.device)
    val_scores, val_losses = forecast_scores(val_pred, episodes.targets["val"], episodes.quantiles)
    if restored["step"] != best_step or not np.isclose(val_scores["raw_mean_2pinball"], best_val,
                                                       rtol=1e-6, atol=1e-8):
        raise AssertionError("Restored best checkpoint does not reproduce validation selection")
    audits["checkpoint_reload_verified"] = True
    audits["checkpoint_includes_all_adaptive_parameters"] = True
    eval_pred = predict(model, episodes, cache, "eval", args.device)
    eval_scores, eval_losses = forecast_scores(eval_pred, episodes.targets["eval"], episodes.quantiles)
    prediction_arrays = {"val_predictions": val_pred, "eval_predictions": eval_pred,
                         "val_target": episodes.targets["val"], "eval_target": episodes.targets["eval"],
                         "val_episode_losses": val_losses, "eval_episode_losses": eval_losses,
                         "val_episode_ids": episodes.ids["val"], "eval_episode_ids": episodes.ids["eval"],
                         "quantiles": episodes.quantiles}
    if args.method == "F0":
        prediction_arrays["train_predictions"] = predict(model, episodes, cache, "train", args.device)
        prediction_arrays["train_target"] = episodes.targets["train"]
        prediction_arrays["train_episode_ids"] = episodes.ids["train"]
    np.savez_compressed(output / "predictions.npz", **prediction_arrays)
    frozen_after = {name: file_hash(old_source / name) for name in frozen_sources}
    if frozen_after != frozen_sources:
        raise AssertionError("Existing S1 source changed during this trial")
    if file_hash(Path(__file__)) != source_before or file_hash(args.data) != data_before:
        raise AssertionError("Trial source or prepared data changed during execution")
    grid = (.0003, .001) if args.method in {"H_LIN", "H_MLP"} else (.00003, .0001)
    result = {
        "completed": True, "method": args.method, "seed": args.seed, "lr": args.lr,
        "smoke": args.smoke, "metadata": episodes.metadata, "val_score": val_scores["raw_mean_2pinball"],
        "best_step": best_step, "eval_score": eval_scores["raw_mean_2pinball"],
        "steps_completed": updates, "trainable": model.adaptive_count,
        "trainable_parameters": model.adaptive_count, "trainable_names": list(model.adaptive_names),
        "module_map": model.module_map, "module_initialization_seeds": model.module_seeds,
        "lora_rank": model.rank, "lora_alpha": 2 * model.rank if model.rank else None,
        "head_lr": head_lr if head_parameters else None, "head_only_updates": head_only_steps if args.method == "LP" else 0,
        "phase_transitions": phases, "final_optimizer_states": final_optimizer_states,
        "wall_seconds": time.perf_counter() - started,
        "peak_cuda_gib": torch.cuda.max_memory_allocated() / 2**30 if args.device.startswith("cuda") else 0,
        "peak_cuda_reserved_gib": torch.cuda.max_memory_reserved() / 2**30 if args.device.startswith("cuda") else 0,
        "optimizer_seconds_per_step": float(np.mean(durations)) if durations else 0,
        "optimizer_seconds_per_step_after_first": float(np.mean(durations[1:])) if len(durations) > 1 else None,
        "optimizer_seconds_total": float(sum(durations)), "sampler_sha256": sampler_hash,
        "data_sha256": data_before, "source_sha256": source_before,
        "source_hashes": {"experiments/peft_shift_mechanism_v1/train.py": source_before,
                          **{f"experiments/peft_adaptation_scope_v1/{name}": value
                             for name, value in frozen_after.items()}},
        "frozen_s1_source_hashes": frozen_after, "cache": str(cache.path.resolve()),
        "cache_created_this_trial": cache_created, "cache_generation_seconds": cache.manifest["wall_seconds"],
        "checkpoint_sha256": file_hash(best_checkpoint), "audits": audits,
        "validation": val_scores, "evaluation": eval_scores, "validation_history": records,
        "boundaries": {"lr_at_lower_candidate": bool(np.isclose(args.lr, grid[0])),
                       "lr_at_upper_candidate": bool(np.isclose(args.lr, grid[-1])),
                       "best_at_zero": best_step == 0, "best_at_last_update": bool(updates and best_step == updates)},
        "episode_counts": {split: len(values) for split, values in episodes.targets.items()},
        "context": 256, "horizon": 16, "channel_order": ["Y", "U", "V"], "target_channel": "Y",
        "micro_groups": MICRO_GROUPS, "effective_groups": EFFECTIVE_GROUPS,
        "autocast": "bfloat16" if args.device.startswith("cuda") else "float32",
        "autocast_weight_cache": False, "weight_dtype": "float32", "dropout": 0.0,
        "gradient_clip_norm": 1.0, "weight_decay": 0.0, "torch_threads": 2, "interop_threads": 1,
        "training_loss": "Native normalized 2-pinball, mean horizon, sum quantiles, mean Y rows only",
        "selection_metric": "Raw mean 2-pinball over Y episode/horizon/quantile; no fit-std scaling",
        "scope": "S0 implementation smoke" if args.smoke else "Controlled synthetic development screen",
        "packages": {name: importlib.metadata.version(name) for name in
                     ("torch", "numpy", "transformers", "chronos-forecasting", "peft")},
    }
    atomic_json(output / "result.json", result)
    log_event(output, "completed", val_score=result["val_score"], best_step=best_step,
              wall_seconds=result["wall_seconds"], peak_cuda_gib=result["peak_cuda_gib"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--val-every", type=int, default=40)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lp-fraction", type=float, default=.4)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.steps < 0 or args.val_every < 1 or args.lr <= 0 or not 0 < args.lp_fraction < 1:
        parser.error("Invalid steps, validation interval, learning rate, or LP phase fraction")
    try:
        run(args)
    except Exception as error:
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        atomic_json(output / "failure.json", {"completed": False, "method": args.method,
                    "error_type": type(error).__name__, "error": str(error),
                    "traceback": traceback.format_exc(), "time": time.time()})
        raise


if __name__ == "__main__":
    main()
