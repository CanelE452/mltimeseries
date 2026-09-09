"""Bounded, single-process S0/S1 training; call only through the guarded runner."""

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

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch

from .modeling import (AdaptationModel, CACHED_METHODS, METHODS, deterministic_backbone,
                       forecast_scores, frozen_digest, native_pinball, patch_to_quantiles)


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_resources(args):
    import psutil

    free_gib = psutil.virtual_memory().available / 1024 ** 3
    if free_gib < args.min_free_ram_gib:
        raise RuntimeError(f"RESOURCE_GUARD: free RAM {free_gib:.2f} GiB < {args.min_free_ram_gib:.2f} GiB")


def precision(args):
    return torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False) if args.device.startswith("cuda") else nullcontext()


def log_event(output, event, **fields):
    record = {"time": time.time(), "event": event, **fields}
    with (output / "progress.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, allow_nan=False) + "\n")
    print(json.dumps(record, allow_nan=False), flush=True)


class Panel:
    def __init__(self, path, smoke=False):
        with np.load(path, allow_pickle=False) as data:
            self.values = data["values"].astype(np.float32)
            self.context = int(data["context"])
            self.horizon = int(data["horizon"])
            self.fit_std = data["fit_std"].astype(np.float64)
            self.channels = [str(x) for x in data["channels"]]
            self.origins = {split: data[split + "_origins"].astype(np.int64)
                            for split in ("train", "val", "eval")}
        if smoke:
            self.origins = {split: origins[:4] for split, origins in self.origins.items()}
        self.count_channels = self.values.shape[1]
        if self.fit_std.shape != (self.count_channels,) or not np.all(self.fit_std > 0):
            raise ValueError("Invalid fit-only scaling")
        if self.horizon % 16:
            raise ValueError("This S1 contract requires a whole number of output patches")
        for origins in self.origins.values():
            if len(origins) == 0 or origins.min() < self.context or origins.max() + self.horizon > len(self.values):
                raise ValueError("Invalid origin boundaries")

    def batch(self, origins, device):
        context = np.stack([self.values[o - self.context:o].T for o in origins])
        target = np.stack([self.values[o:o + self.horizon].T for o in origins])
        rows = len(origins) * self.count_channels
        # IDs identify sample instances, including duplicate draws of the same origin.
        groups = np.repeat(np.arange(len(origins)), self.count_channels)
        return (torch.as_tensor(context.reshape(rows, self.context), device=device),
                torch.as_tensor(target.reshape(rows, self.horizon), device=device),
                torch.as_tensor(groups, dtype=torch.long, device=device))

    def targets(self, split):
        return np.stack([self.values[o:o + self.horizon].T for o in self.origins[split]])


class FeatureCache:
    def __init__(self, path, manifest):
        self.path = path
        self.manifest = manifest
        self.arrays = {key: np.load(path / (key + ".npy"), mmap_mode="r")
                       for key in ("hidden", "base_norm", "loc", "scale", "origins")}
        self.origin_to_row = {int(origin): row for row, origin in enumerate(self.arrays["origins"])}

    def batch(self, origins, device):
        indices = [self.origin_to_row[int(origin)] for origin in origins]
        result = []
        for key in ("hidden", "base_norm", "loc", "scale"):
            values = self.arrays[key][indices]
            values = values.reshape(-1, *values.shape[2:])
            result.append(torch.as_tensor(values, device=device))
        return result


def prepare_cache(base, panel, args, output):
    origin_manifest = {split: values.tolist() for split, values in panel.origins.items()}
    contract = {
        "cache_version": 2, "data_sha256": file_hash(args.data),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "config_sha256": file_hash(Path(args.checkpoint) / "config.json"),
        "context": panel.context, "horizon": panel.horizon,
        "channels": panel.channels, "origins": origin_manifest, "smoke": args.smoke,
        "dropout": 0.0, "autocast": "bfloat16" if args.device.startswith("cuda") else "float32",
        "hidden_storage": "float32", "raw_inverse_dtype": "float32", "autocast_weight_cache": False,
        "feature_inputs": "context and isolated group IDs only; no future target/covariate",
    }
    signature = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()[:20]
    path = Path(args.cache) / signature
    manifest_path = path / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["contract"] != contract or not manifest["completed"]:
            raise AssertionError("Cache contract mismatch")
        return FeatureCache(path, manifest), False
    path.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    origins = np.unique(np.concatenate(list(panel.origins.values())))
    n, c = len(origins), panel.count_channels
    patches = panel.horizon // base.chronos_config.output_patch_size
    shapes = {"hidden": (n, c, patches, base.model_dim),
              "base_norm": (n, c, base.num_quantiles, panel.horizon),
              "loc": (n, c, 1), "scale": (n, c, 1)}
    arrays = {key: np.lib.format.open_memmap(path / (key + ".npy"), mode="w+", dtype=np.float32, shape=shape)
              for key, shape in shapes.items()}
    np.save(path / "origins.npy", origins)
    with torch.no_grad():
        for index, origin in enumerate(origins):
            check_resources(args)
            context, _, groups = panel.batch([origin], args.device)
            with precision(args):
                encoded, (loc, scale), _, _ = base.encode(
                    context=context, group_ids=groups, num_output_patches=patches)
                hidden = encoded.last_hidden_state[:, -patches:]
                norm = patch_to_quantiles(base.output_patch_embedding(hidden), base.num_quantiles,
                                         base.chronos_config.output_patch_size)
            for key, value in zip(("hidden", "base_norm", "loc", "scale"), (hidden, norm, loc, scale)):
                arrays[key][index] = value.float().cpu().numpy()
            if index == 0 or (index + 1) % 32 == 0 or index + 1 == n:
                log_event(output, "cache", completed=index + 1, total=n,
                          seconds=time.perf_counter() - start)
    for value in arrays.values():
        value.flush()
    del arrays
    manifest = {"completed": True, "contract": contract,
                "wall_seconds": time.perf_counter() - start,
                "checkpoint_weights": {weight.name: file_hash(weight)
                                       for weight in sorted(Path(args.checkpoint).glob("*.safetensors"))}}
    atomic_json(manifest_path, manifest)
    return FeatureCache(path, manifest), True


def audit_information(base, panel, args):
    """Run once per smoke trial at the real panel shape, without training/evaluation labels."""
    origins = panel.origins["train"][:2]
    context, _, groups = panel.batch(origins, args.device)
    patches = panel.horizon // base.chronos_config.output_patch_size
    c = panel.count_channels
    with torch.no_grad(), precision(args):
        def encode(ctx, ids, **extra):
            return base.encode(context=ctx, group_ids=ids, num_output_patches=patches,
                               **extra)[0].last_hidden_state
        original = encode(context, groups)
        artificial_future = torch.full((context.shape[0], panel.horizon), 123.0, device=args.device)
        changed_target = encode(context, groups, future_target=artificial_future)
        blocked_future = encode(context, groups, future_covariates=artificial_future,
                                future_covariates_mask=torch.zeros_like(artificial_future))
        modified_context = context.clone()
        modified_context[c:] = modified_context[c:] * 2 + 17
        isolated = encode(modified_context, groups)
    errors = {"future_target_hidden_max_abs": float((original - changed_target).abs().max()),
              "masked_future_hidden_max_abs": float((original - blocked_future).abs().max()),
              "other_group_hidden_max_abs": float((original[:c] - isolated[:c]).abs().max())}
    if any(value > 1e-5 for value in errors.values()):
        raise AssertionError(f"Information isolation audit failed: {errors}")
    return {**errors, "future_target_isolation": True, "masked_future_isolation": True,
            "group_isolation": True, "groups_tested": len(origins)}


def save_trainable(model, path):
    state = {name: parameter.detach().cpu() for name, parameter in model.named_parameters()
             if parameter.requires_grad}
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def load_trainable(model, path, device):
    state = torch.load(path, map_location="cpu", weights_only=True)
    parameters = dict(model.named_parameters())
    expected = {name for name, value in parameters.items() if value.requires_grad}
    if set(state) != expected:
        raise AssertionError("Checkpoint trainable map changed")
    with torch.no_grad():
        for name, value in state.items():
            parameters[name].copy_(value.to(device))


def predict(model, panel, cache, split, args):
    predictions = []
    model.eval()
    with torch.no_grad():
        origins = panel.origins[split]
        for start in range(0, len(origins), args.micro_groups):
            check_resources(args)
            selected = origins[start:start + args.micro_groups]
            with precision(args):
                if model.method in CACHED_METHODS:
                    _, raw = model.from_cache(*cache.batch(selected, args.device))
                else:
                    context, _, groups = panel.batch(selected, args.device)
                    _, raw, _, _ = model.from_context(context, groups, panel.horizon)
            predictions.append(raw[:, :, :panel.horizon].float().cpu().numpy().reshape(
                len(selected), panel.count_channels, model.n_quantiles, panel.horizon))
    return np.concatenate(predictions)


def run(args):
    start = time.perf_counter()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "metrics.json").exists():
        raise FileExistsError(f"Refusing to overwrite trial: {output}")
    check_resources(args)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed)
    if args.device.startswith("cuda"):
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("CUDA with bfloat16 support required")
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction)
        torch.cuda.reset_peak_memory_stats()
        torch.backends.cuda.matmul.allow_tf32 = False
    panel = Panel(args.data, smoke=args.smoke)
    if args.smoke:
        args.steps = min(args.steps, 5)
        args.eval_every = 1
    from chronos.chronos2.model import Chronos2Model

    log_event(output, "loading", checkpoint=args.checkpoint, method=args.method)
    base = Chronos2Model.from_pretrained(args.checkpoint, local_files_only=True,
                                        dtype=torch.float32, attn_implementation="sdpa").to(args.device)
    deterministic_backbone(base)
    base.requires_grad_(False)
    information_audit = audit_information(base, panel, args) if args.smoke else {
        "runtime_information_audit": "S0 only; production cache and sampler use the same audited code path"}
    cache, cache_created = prepare_cache(base, panel, args, output)
    # Reset after loading/caching so initialization is paired across methods and cache hit/miss.
    torch.manual_seed(args.seed)
    model = AdaptationModel(base, args.method, panel.count_channels)
    trainables = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainables, lr=args.lr, weight_decay=0.0, foreach=False) if trainables else None
    if optimizer is not None:
        registered = {id(p) for group in optimizer.param_groups for p in group["params"]}
        if registered != {id(p) for p in trainables}:
            raise AssertionError("Optimizer parameter registration mismatch")
    frozen_before, frozen_count = frozen_digest(model)
    first_origins = panel.origins["train"][:1]
    with torch.no_grad(), precision(args):
        hidden, f0_norm, loc, scale = cache.batch(first_origins, args.device)
        if args.method in CACHED_METHODS:
            initial, _ = model.from_cache(hidden, f0_norm, loc, scale)
        else:
            context, _, groups = panel.batch(first_origins, args.device)
            initial, _, _, _ = model.from_context(context, groups, panel.horizon)
        identity_error = float((initial.float() - f0_norm).abs().max())
    if identity_error > 1e-5:
        raise AssertionError(f"Zero-update identity failed: {identity_error}")
    audits = {**information_audit, "zero_update_identity": True,
              "zero_update_normalized_max_abs": identity_error,
              "trainable_count_verified": True, "optimizer_registration_verified": True,
              "frozen_parameter_count": frozen_count, "frozen_before_sha256": frozen_before}
    log_event(output, "audits", trainable=model.trainable_count, audits=audits)
    checkpoint = output / "best_trainable.pt"
    quantiles = np.asarray(base.chronos_config.quantiles, dtype=np.float64)
    quantile_tensor = torch.as_tensor(quantiles, dtype=torch.float32, device=args.device)
    val_targets = panel.targets("val")
    val_pred = predict(model, panel, cache, "val", args)
    val_metrics, _ = forecast_scores(val_pred, val_targets, panel.fit_std, quantiles)
    best_val = val_metrics["scaled_2pinball"]
    selected_step = 0
    save_trainable(model, checkpoint)
    records = [{"step": 0, "val_score": best_val}]
    log_event(output, "validation", step=0, val_score=best_val, best_val_score=best_val)
    steps = args.steps if optimizer is not None else 0
    samples = np.random.default_rng(args.seed).integers(len(panel.origins["train"]),
                                                       size=(steps, args.effective_groups))
    sampler_hash = hashlib.sha256(panel.origins["train"][samples].tobytes()).hexdigest()
    update_seconds = []
    steps_completed = 0
    for step in range(1, steps + 1):
        check_resources(args)
        tick = time.perf_counter()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        indices = samples[step - 1]
        accumulated_loss = 0.0
        for offset in range(0, args.effective_groups, args.micro_groups):
            selected = panel.origins["train"][indices[offset:offset + args.micro_groups]]
            context, target, groups = panel.batch(selected, args.device)
            with precision(args):
                if args.method in CACHED_METHODS:
                    hidden, f0_norm, loc, scale = cache.batch(selected, args.device)
                    norm, _ = model.from_cache(hidden, f0_norm, loc, scale)
                else:
                    norm, _, loc, scale = model.from_context(context, groups, panel.horizon)
                loss = native_pinball(norm, target, loc, scale, quantile_tensor, model.use_arcsinh)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite training loss at step {step}")
            weight = len(selected) / args.effective_groups
            (loss * weight).backward()
            accumulated_loss += float(loss.detach()) * weight
        grad_norm = torch.nn.utils.clip_grad_norm_(trainables, max_norm=1.0, error_if_nonfinite=True)
        optimizer.step()
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        update_seconds.append(time.perf_counter() - tick)
        steps_completed = step
        if step == 1:
            after_first, _ = frozen_digest(model)
            audits["frozen_after_first_step_sha256"] = after_first
            audits["frozen_after_first_step_verified"] = after_first == frozen_before
            if after_first != frozen_before:
                raise AssertionError("Frozen parameter changed after first optimizer update")
        if step % args.eval_every == 0 or step == steps:
            val_pred = predict(model, panel, cache, "val", args)
            val_metrics, _ = forecast_scores(val_pred, val_targets, panel.fit_std, quantiles)
            score = val_metrics["scaled_2pinball"]
            if score < best_val:
                best_val, selected_step = score, step
                save_trainable(model, checkpoint)
            records.append({"step": step, "val_score": score, "train_loss": accumulated_loss,
                            "grad_norm": float(grad_norm), "optimizer_seconds": update_seconds[-1]})
            log_event(output, "validation", **records[-1], best_val_score=best_val,
                      selected_step=selected_step)
        elif step % 10 == 0:
            log_event(output, "training", step=step, train_loss=accumulated_loss,
                      optimizer_seconds=float(np.mean(update_seconds[-10:])))
    frozen_after, _ = frozen_digest(model)
    audits["frozen_after_training_sha256"] = frozen_after
    audits["frozen_after_training_verified"] = frozen_after == frozen_before
    if frozen_after != frozen_before:
        raise AssertionError("Frozen parameter changed during training")
    load_trainable(model, checkpoint, args.device)
    val_pred = predict(model, panel, cache, "val", args)
    val_metrics, val_losses = forecast_scores(val_pred, val_targets, panel.fit_std, quantiles)
    if not np.isclose(val_metrics["scaled_2pinball"], best_val, rtol=1e-6, atol=1e-8):
        raise AssertionError("Restored checkpoint does not reproduce selected validation score")
    eval_pred = predict(model, panel, cache, "eval", args)
    eval_metrics, eval_losses = forecast_scores(eval_pred, panel.targets("eval"), panel.fit_std, quantiles)
    audits["best_checkpoint_reload_verified"] = True
    np.savez_compressed(output / "predictions.npz", val_pred=val_pred, eval_pred=eval_pred,
                        quantiles=quantiles, val_origins=panel.origins["val"],
                        eval_origins=panel.origins["eval"],
                        val_normalized_losses=val_losses, eval_normalized_losses=eval_losses,
                        val_per_origin_channel_loss=val_losses, eval_per_origin_channel_loss=eval_losses)
    packages = {name: importlib.metadata.version(name) for name in
                ("torch", "numpy", "transformers", "chronos-forecasting")}
    if args.method == "OFF_LORA":
        packages["peft"] = importlib.metadata.version("peft")
    metrics = {
        "completed": True, "method": args.method, "seed": args.seed, "lr": args.lr,
        "smoke": args.smoke, "steps_completed": steps_completed, "selected_step": selected_step,
        "val_score": val_metrics["scaled_2pinball"], "eval_score": eval_metrics["scaled_2pinball"],
        "trainable_parameters": model.trainable_count, "trainable_names": model.trainable_names,
        "module_map": model.module_map, "wall_seconds": time.perf_counter() - start,
        "peak_vram_gib": torch.cuda.max_memory_allocated() / 1024 ** 3 if args.device.startswith("cuda") else 0,
        "peak_vram_reserved_gib": torch.cuda.max_memory_reserved() / 1024 ** 3 if args.device.startswith("cuda") else 0,
        "optimizer_seconds_per_step": float(np.mean(update_seconds)) if update_seconds else 0.0,
        "optimizer_seconds_per_step_after_first": float(np.mean(update_seconds[1:])) if len(update_seconds) > 1 else None,
        "sampler_hash": sampler_hash, "audits": audits, "validation": val_metrics,
        "evaluation": eval_metrics, "validation_history": records, "packages": packages,
        "data_sha256": cache.manifest["contract"]["data_sha256"], "cache": str(cache.path.resolve()),
        "cache_created_this_trial": cache_created, "cache_generation_seconds": cache.manifest["wall_seconds"],
        "context": panel.context, "horizon": panel.horizon, "channels": panel.channels,
        "origin_counts": {split: len(origins) for split, origins in panel.origins.items()},
        "effective_groups": args.effective_groups, "micro_groups": args.micro_groups,
        "base_dropout": 0.0, "adapter_dropout": 0.0, "gradient_clip_norm": 1.0,
        "weight_decay": 0.0, "weight_dtype": "float32", "raw_inverse_dtype": "float32",
        "autocast": "bfloat16" if args.device.startswith("cuda") else "float32",
        "autocast_weight_cache": False,
        "point_summary": "quantile arithmetic average; not asserted to be conditional mean",
        "training_loss": "native normalized 2-pinball: mean H, sum Q, mean rows including missing zeros",
        "selection_metric": "fit-std-scaled 2-pinball, valid-cell mean within channel then equal channel macro",
        "scope": "development pilot; S0 smoke is not a performance experiment" if args.smoke else "S1 development pilot",
    }
    atomic_json(output / "metrics.json", metrics)
    log_event(output, "completed", val_score=metrics["val_score"], eval_score=metrics["eval_score"],
              selected_step=selected_step, wall_seconds=metrics["wall_seconds"],
              peak_vram_gib=metrics["peak_vram_gib"])
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--effective-groups", type=int, default=8)
    parser.add_argument("--micro-groups", type=int, default=1)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.67)
    parser.add_argument("--min-free-ram-gib", type=float, default=4.0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.effective_groups < 1 or args.micro_groups < 1 or args.eval_every < 1 or args.steps < 0:
        parser.error("Groups/evaluation interval must be positive and steps nonnegative")
    output = Path(args.output)
    try:
        run(args)
    except Exception as error:
        output.mkdir(parents=True, exist_ok=True)
        # Preserve the last best checkpoint. A failed trial never has completed metrics.
        atomic_json(output / "failure.json", {"completed": False, "method": args.method,
                    "error_type": type(error).__name__, "error": str(error),
                    "traceback": traceback.format_exc(), "time": time.time()})
        raise


if __name__ == "__main__":
    main()
