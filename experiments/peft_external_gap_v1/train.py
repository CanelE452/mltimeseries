"""Serial native-quantile fits and selection-gated holdout forecasts for new real panels."""

import argparse
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import time
import traceback

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "2"

import numpy as np
import torch

from experiments.peft_adaptation_scope_v1 import modeling as native
from experiments.peft_adaptation_scope_v1 import train as legacy


STUDY = "peft_external_gap_v1"
PLAN = "_docs/notes/tsfm_topics/12_peft_external_gap_plan_20260908.md"
METHODS = ("F0", "H_MLP", "H_FULL", "OFF_LORA")
RATES = {"F0": (0.,), "H_MLP": (1e-4, 3e-4, 1e-3),
         "H_FULL": (3e-5, 1e-4, 3e-4), "OFF_LORA": (1e-5, 3e-5, 1e-4)}
COUNTS = {"F0": 0, "H_MLP": 589301, "H_FULL": 3653280, "OFF_LORA": 1206912}
REVISION = "29ec3766d36d6f73f0696f85560a422f50e8498c"
DEFAULT_CHECKPOINT = Path.home() / ".cache/huggingface/hub/models--amazon--chronos-2/snapshots" / REVISION
file_hash, atomic_json = legacy.file_hash, legacy.atomic_json


def array_hash(*arrays):
    digest = hashlib.sha256()
    for value in arrays:
        array = np.ascontiguousarray(value)
        digest.update(str((array.shape, array.dtype.str)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def parameter_hash(named_parameters):
    digest = hashlib.sha256()
    for name, parameter in named_parameters:
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def log(output, event, **fields):
    record = {"time": time.time(), "event": event, **fields}
    with (Path(output) / "progress.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, allow_nan=False) + "\n")
    print(json.dumps(record, allow_nan=False), flush=True)


def padded_origins(origins):
    origins = np.asarray(origins, dtype=np.int64)
    if origins.ndim != 1 or not 1 <= len(origins) <= 4:
        raise ValueError("A native forward requires one to four complete origins")
    return np.pad(origins, (0, 4 - len(origins)), mode="edge"), len(origins)


class Panel:
    def __init__(self, path, stage, smoke=False):
        self.path, self.stage, self.smoke = Path(path), stage, smoke
        with np.load(path, allow_pickle=False) as archive:
            self.context_values = archive["context_values"].astype(np.float32)
            self.target_values = archive["target_values"].astype(np.float32)
            self.channels = [str(value) for value in archive["channels"]]
            self.target_indices = archive["target_indices"].astype(np.int64)
            self.quantiles = archive["quantiles"].astype(np.float64)
            observed_mask = archive["observed_mask"].astype(bool)
            loss_mask = archive["target_loss_mask"].astype(bool)
            self.timestamps = archive["timestamps"].copy()
            self.fit_mean = archive["fit_mean"].astype(np.float64)
            self.fit_std = archive["fit_std"].astype(np.float64)
            self.fit_median = archive["fit_median"].astype(np.float64)
            self.context, self.horizon = int(archive["context"]), int(archive["horizon"])
            self.metadata = json.loads(archive["manifest_json"].item())
            names = ("train", "val") if stage == "fit" else ("cal", "eval")
            if smoke:
                if stage != "fit":
                    raise ValueError("S0 is fit-only")
                origins = archive["train_origins"].astype(np.int64)
                if len(origins) < 14:
                    raise ValueError("S0 needs fourteen train origins for its isolated pseudo-validation")
                self.origins = {"train": origins[:8], "val": origins[10:14]}
                self.smoke_train_origin_hash = array_hash(origins)
            else:
                self.origins = {name: archive[name + "_origins"].astype(np.int64) for name in names}
            forbidden = ("cal_origins", "eval_origins") if stage == "fit" else ("train_origins", "val_origins")
            if any(name in archive for name in forbidden):
                raise AssertionError("Prepared archive mixes fit and holdout origin contracts")
        self.count_channels = len(self.channels)
        self.dataset = self.metadata["dataset"]
        expected_channels = {"bike": 5, "household": 4}
        if self.dataset not in expected_channels or self.count_channels != expected_channels[self.dataset]:
            raise ValueError("Use the declared Bike5/Household4 input channels")
        if (self.context, self.horizon) != (336, 48) or self.target_indices.shape != (2,) or len(np.unique(self.target_indices)) != 2:
            raise ValueError("Expected L336/H48 and exactly two target channels")
        if np.any(self.target_indices < 0) or np.any(self.target_indices >= self.count_channels):
            raise ValueError("Target indices are outside the input channel map")
        if self.context_values.ndim != 2 or self.context_values.shape != self.target_values.shape or self.context_values.shape[1] != self.count_channels:
            raise ValueError("Context and original target arrays must have matching [T,C] shapes")
        expected_loss_mask = np.zeros_like(self.target_values, dtype=bool)
        expected_loss_mask[:, self.target_indices] = np.isfinite(self.target_values[:, self.target_indices])
        if not np.array_equal(observed_mask, np.isfinite(self.target_values)) or not np.array_equal(loss_mask, expected_loss_mask):
            raise AssertionError("Prepared observed/target loss masks disagree with original labels")
        if self.quantiles.shape != (21,) or np.any(np.diff(self.quantiles) <= 0):
            raise ValueError("Prepared quantile grid must contain 21 increasing levels")
        if not np.isfinite(self.context_values).all() or not np.isfinite(self.fit_std).all() or np.any(self.fit_std <= 0):
            raise ValueError("Past-only imputation or train scaling is invalid")
        if len(self.timestamps) != len(self.context_values):
            raise ValueError("Timestamp and data row counts differ")
        for split, origins in self.origins.items():
            if origins.ndim != 1 or len(origins) == 0 or len(np.unique(origins)) != len(origins):
                raise ValueError("Every split needs distinct chronological origins")
            if np.any(np.diff(origins) <= 0) or origins.min() < self.context or origins.max() + self.horizon > len(self.context_values):
                raise ValueError("Origin violates the legal context/target boundaries")
            if "boundary_indices" in self.metadata:
                left, right = self.metadata["boundary_indices"]["train" if smoke else split]
                if origins.min() < left or origins.max() + self.horizon > right:
                    raise AssertionError("Target horizon crosses its declared split boundary")
        if smoke and self.origins["train"][-1] + self.horizon > self.origins["val"][0]:
            raise AssertionError("S0 pseudo-validation labels overlap its optimizer origins")
        self.stats_hash = array_hash(self.fit_mean, self.fit_std, self.fit_median, self.target_indices)

    def batch(self, origins, device):
        if len(origins) != 4:
            raise ValueError("Every encoder call must contain four complete isolated groups")
        context = np.stack([self.context_values[o - self.context:o].T for o in origins])
        target = np.stack([self.target_values[o:o + self.horizon].T for o in origins])
        inactive = np.ones(self.count_channels, dtype=bool)
        inactive[self.target_indices] = False
        target[:, inactive] = np.nan
        groups = np.repeat(np.arange(4), self.count_channels)
        return (torch.as_tensor(context.reshape(4 * self.count_channels, self.context), device=device),
                torch.as_tensor(target.reshape(4 * self.count_channels, self.horizon), device=device),
                torch.as_tensor(groups, dtype=torch.long, device=device))

    def targets(self, split):
        return np.stack([self.target_values[o:o + self.horizon, self.target_indices].T for o in self.origins[split]])


def scores(prediction, target, fit_std, quantiles):
    prediction, target = np.asarray(prediction, dtype=np.float64), np.asarray(target, dtype=np.float64)
    fit_std, quantiles = np.asarray(fit_std, dtype=np.float64), np.asarray(quantiles, dtype=np.float64)
    if prediction.shape != (len(target), 2, 21, 48) or target.shape != (len(target), 2, 48) or fit_std.shape != (2,):
        raise ValueError("Scoring uses two target channels only")
    if not np.isfinite(prediction).all() or np.any(fit_std <= 0):
        raise ValueError("Nonfinite prediction or invalid train-only scale")
    prediction = np.sort(prediction, axis=2)
    valid = np.isfinite(target)
    error = target[:, :, None, :] - prediction
    loss = np.where(valid[:, :, None, :], 2 * np.maximum(quantiles[None, None, :, None] * error,
                       (quantiles[None, None, :, None] - 1) * error), 0) / fit_std[None, :, None, None]
    loss_sums, counts = loss.sum(axis=(2, 3)), valid.sum(axis=2) * len(quantiles)
    if np.any(counts.sum(axis=0) == 0):
        raise ValueError("A target has no observed scoring labels")
    per_target = loss_sums.sum(axis=0) / counts.sum(axis=0)
    med, lo, hi = [int(np.argmin(abs(quantiles - q))) for q in (.5, .1, .9)]
    covered = ((target >= prediction[:, :, lo]) & (target <= prediction[:, :, hi]) & valid).sum(axis=(0, 2))
    cells = valid.sum(axis=(0, 2))
    width = np.where(valid, prediction[:, :, hi] - prediction[:, :, lo], 0).sum(axis=(0, 2)) / cells / fit_std
    mse = np.where(valid, (target - prediction[:, :, med])**2, 0).sum(axis=(0, 2)) / cells
    return {"score": float(per_target.mean()), "target_scores": per_target.tolist(),
            "coverage80": float((covered / cells).mean()), "target_coverage80": (covered / cells).tolist(),
            "scaled_width80": float(width.mean()), "target_scaled_width80": width.tolist(),
            "median_mse": float(mse.mean()), "target_median_mse": mse.tolist(),
            "crossing": float((np.diff(prediction, axis=2) < 0).mean())}, loss_sums, counts


def load_base(args):
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    if not args.device.startswith("cuda") or not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Guarded CUDA BF16 execution is required")
    torch.cuda.set_per_process_memory_fraction(.67)
    torch.cuda.reset_peak_memory_stats()
    torch.backends.cuda.matmul.allow_tf32 = False
    from chronos.chronos2.model import Chronos2Model

    base = Chronos2Model.from_pretrained(args.checkpoint, local_files_only=True, dtype=torch.float32,
                                        attn_implementation="sdpa").to(args.device)
    native.deterministic_backbone(base)
    base.requires_grad_(False)
    return base


def source_state(root, args):
    from chronos.chronos2.model import Chronos2Model

    paths = [Path(__file__).resolve(), root / "experiments" / STUDY / "data.py", root / PLAN]
    paths.extend(root / "experiments/peft_adaptation_scope_v1" / name for name in ("modeling.py", "train.py", "guard.py"))
    native_folder = Path(inspect.getsourcefile(Chronos2Model)).resolve().parent
    native_files = sorted(native_folder.glob("*.py"))
    checkpoint = Path(args.checkpoint).resolve()
    weights = sorted(checkpoint.glob("*.safetensors"))
    if not weights:
        raise FileNotFoundError("The native checkpoint has no local weights")
    checkpoints = [checkpoint / "config.json", *weights]
    sources = {path.relative_to(root).as_posix(): file_hash(path) for path in paths if path.suffix == ".py"}
    protected = {str(path.resolve()): file_hash(path) for path in [*paths, *native_files, *checkpoints, Path(args.data)]}
    return {"source_hashes": sources, "native_source_hashes": {str(p): file_hash(p) for p in native_files},
            "plan_sha256": file_hash(root / PLAN), "checkpoint_hashes": {p.name: file_hash(p) for p in checkpoints},
            "checkpoint": str(checkpoint), "data_sha256": file_hash(args.data), "protected_hashes": protected}


def verify_hashes(hashes):
    for path, expected in hashes.items():
        if file_hash(path) != expected:
            raise AssertionError(f"Protected file changed: {path}")


def prepare_cache(base, panel, args, state):
    contract = {"study": STUDY, "version": 1, "data_sha256": state["data_sha256"],
                "source_hashes": state["source_hashes"], "native_source_hashes": state["native_source_hashes"],
                "plan_sha256": state["plan_sha256"], "checkpoint_hashes": state["checkpoint_hashes"],
                "checkpoint": state["checkpoint"], "dataset": panel.dataset, "context": 336, "horizon": 48,
                "channels": panel.channels, "target_indices": panel.target_indices.tolist(), "stats_sha256": panel.stats_hash,
                "origins": {key: value.tolist() for key, value in panel.origins.items()}, "smoke": args.smoke,
                "micro_groups": 4, "last_batch": "repeat last origin as a separate group then discard padding",
                "weight_dtype": "float32", "autocast": "bfloat16", "autocast_weight_cache": False,
                "tf32": False, "dropout": 0., "features": "past context and isolated groups; no future values"}
    signature = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    path = Path(args.cache) / signature[:20]
    manifest_path = path / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not manifest.get("completed") or manifest["contract"] != contract:
            raise AssertionError("Frozen feature cache contract mismatch")
        verify_hashes({str(path / name): digest for name, digest in manifest["array_hashes"].items()})
        return legacy.FeatureCache(path, manifest), False, signature
    if args.method != "F0":
        raise RuntimeError("Build the matching F0 cache before adaptation")
    path.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    origins = np.unique(np.concatenate(list(panel.origins.values())))
    n, c = len(origins), panel.count_channels
    shapes = {"hidden": (n, c, 3, 768), "base_norm": (n, c, 21, 48), "loc": (n, c, 1), "scale": (n, c, 1)}
    arrays = {name: np.lib.format.open_memmap(path / (name + ".npy"), mode="w+", dtype=np.float32, shape=shape)
              for name, shape in shapes.items()}
    np.save(path / "origins.npy", origins)
    with torch.no_grad():
        for index in range(0, n, 4):
            legacy.check_resources(args)
            selected, count = padded_origins(origins[index:index + 4])
            context, _, groups = panel.batch(selected, args.device)
            with legacy.precision(args):
                encoded, (loc, scale), _, _ = base.encode(context=context, group_ids=groups, num_output_patches=3)
                hidden = encoded.last_hidden_state[:, -3:]
                norm = native.patch_to_quantiles(base.output_patch_embedding(hidden), 21, 16)
            for name, value in zip(("hidden", "base_norm", "loc", "scale"), (hidden, norm, loc, scale)):
                arrays[name][index:index + count] = value.float().cpu().numpy().reshape(4, c, *value.shape[1:])[:count]
    for array in arrays.values():
        array.flush()
    del arrays, array
    manifest = {"completed": True, "contract": contract, "wall_seconds": time.perf_counter() - started,
                "array_hashes": {p.name: file_hash(p) for p in sorted(path.glob("*.npy"))}}
    atomic_json(manifest_path, manifest)
    log(args.output, "cache_completed", origins=n, seconds=manifest["wall_seconds"])
    return legacy.FeatureCache(path, manifest), True, signature


def direct(model, context, groups):
    hidden, norm, loc, scale = model.encode(context, groups, 48)
    if model.method in native.CACHED_METHODS:
        norm, raw = model.from_cache(hidden, norm, loc, scale)
    else:
        raw = native.normalized_to_raw(norm, loc, scale, model.use_arcsinh)
    return norm, raw, loc, scale


def predict(model, panel, split, args, cache=None, return_unsorted=False):
    model.eval()
    blocks = []
    with torch.no_grad():
        for start in range(0, len(panel.origins[split]), 4):
            legacy.check_resources(args)
            selected, count = padded_origins(panel.origins[split][start:start + 4])
            with legacy.precision(args):
                if cache is not None and model.method in native.CACHED_METHODS:
                    _, raw = model.from_cache(*cache.batch(selected, args.device))
                else:
                    context, _, groups = panel.batch(selected, args.device)
                    _, raw, _, _ = direct(model, context, groups)
            prediction = raw.float().cpu().numpy().reshape(4, panel.count_channels, 21, 48)
            blocks.append(prediction[:count, panel.target_indices])
    unsorted = np.concatenate(blocks)
    sorted_prediction = np.sort(unsorted, axis=2)
    return (sorted_prediction, unsorted) if return_unsorted else sorted_prediction


def information_audit(base, panel, args):
    selected, _ = padded_origins(panel.origins["train"][:4])
    context, target, groups = panel.batch(selected, args.device)
    original_context = context.clone()
    with torch.no_grad(), legacy.precision(args):
        def encode(value, **extra):
            return base.encode(context=value, group_ids=groups, num_output_patches=3, **extra)[0].last_hidden_state
        initial = encode(context)
        artificial = torch.full_like(target, 123.)
        changed_target = encode(context, future_target=artificial)
        masked_future = encode(context, future_covariates=artificial, future_covariates_mask=torch.zeros_like(artificial))
        other = context.clone()
        other[panel.count_channels:] = 3 * other[panel.count_channels:] + 17
        isolated = encode(other)
    errors = {"future_target_hidden_max_abs": float((initial - changed_target).abs().max()),
              "masked_future_hidden_max_abs": float((initial - masked_future).abs().max()),
              "other_group_hidden_max_abs": float((initial[:panel.count_channels] - isolated[:panel.count_channels]).abs().max())}
    inactive = np.setdiff1d(np.arange(panel.count_channels), panel.target_indices)
    target_view = target.reshape(4, panel.count_channels, 48).cpu().numpy()
    raw_target = np.stack([panel.target_values[o:o + 48].T for o in selected])
    missing_preserved = np.array_equal(np.isnan(target_view[:, panel.target_indices]), np.isnan(raw_target[:, panel.target_indices]))
    non_targets_masked = bool(np.isnan(target_view[:, inactive]).all())
    if any(not math.isfinite(value) or value > 1e-5 for value in errors.values()) or not missing_preserved or not non_targets_masked or not torch.equal(context, original_context):
        raise AssertionError(f"Input/label isolation audit failed: {errors}")
    return {**errors, "future_target_isolation": True, "masked_future_isolation": True, "group_isolation": True,
            "original_target_missing_mask_preserved": missing_preserved, "non_target_future_loss_masked": non_targets_masked,
            "context_unchanged": True, "smoke_uses_train_origins_only": True,
            "smoke_original_train_origins_sha256": panel.smoke_train_origin_hash}


def construct(base, method, seed, channels):
    torch.manual_seed(seed)
    model = native.AdaptationModel(base, method, channels)
    if model.trainable_count != COUNTS[method]:
        raise AssertionError("Unexpected trainable parameter count")
    if method == "OFF_LORA":
        expected = [f"encoder.block.{block}.layer.{layer}.self_attention.{part}" for block in range(12)
                    for layer in (0, 1) for part in ("q", "k", "v", "o")]
        expected.append("output_patch_embedding.output_layer")
        if model.module_map != expected:
            raise AssertionError("Official LoRA map differs from the declared 97 projections")
        for name, parameter in model.named_parameters():
            if ".lora_B." in name and torch.count_nonzero(parameter).item() != 0:
                raise AssertionError("LoRA B must initialize at zero")
    return model


def validate_fit_args(args):
    if args.method not in METHODS or args.lr not in RATES[args.method] or args.seed not in (12000, 12001, 12002):
        raise ValueError("Use the declared method, LR grid and optimizer seed")
    expected_steps = 0 if args.method == "F0" else (5 if args.smoke else 200)
    if args.steps != expected_steps or args.val_every != (5 if args.smoke else 40):
        raise ValueError("Use adaptation200/validation40, or train-only S0 updates5/validation5; F0 has zero updates")
    if args.method == "F0" and args.seed != 12000:
        raise ValueError("F0 is evaluated once per dataset")
    if Path(args.checkpoint).resolve() != DEFAULT_CHECKPOINT.resolve():
        raise ValueError("Use the frozen local Chronos-2 revision")


def new_output(root, output):
    output = Path(output).resolve()
    if not output.is_relative_to(root / "runs" / STUDY):
        raise ValueError("Trial outputs must stay inside the new study's runs directory")
    if any((output / name).exists() for name in ("result.json", "trial_contract.json", "best_trainable.pt", "failure.json")):
        raise FileExistsError("Preserve existing complete or partial trial outputs")
    output.mkdir(parents=True, exist_ok=True)
    return output


def run_fit(args):
    started = time.perf_counter()
    validate_fit_args(args)
    root = Path(__file__).resolve().parents[2]
    output = new_output(root, args.output)
    if not Path(args.cache).resolve().is_relative_to(root / "runs" / STUDY):
        raise ValueError("Use a separate cache inside the new study")
    legacy.check_resources(args)
    state = source_state(root, args)
    panel = Panel(args.data, "fit", args.smoke)
    if Path(args.data).resolve() != root / "runs" / STUDY / "prepared" / f"{panel.dataset}_fit.npz":
        raise ValueError("Use the fixed prepared fit archive for this dataset")
    atomic_json(output / "trial_contract.json", {**state, "requested_arguments": vars(args), "stage": "fit",
                "dataset": panel.dataset, "origin_hashes": {k: array_hash(v) for k, v in panel.origins.items()}})
    base = load_base(args)
    quantiles = np.asarray(base.chronos_config.quantiles, dtype=np.float64)
    if not np.array_equal(quantiles, panel.quantiles):
        raise AssertionError("Native and prepared quantile grids differ")
    audits = information_audit(base, panel, args) if args.smoke else {"information_audit": "Same train-only S0 source contract"}
    cache, created, cache_hash = prepare_cache(base, panel, args, state)
    cache_files = {str(p.resolve()): file_hash(p) for p in [cache.path / "manifest.json", *sorted(cache.path.glob("*.npy"))]}
    model = construct(base, args.method, args.seed, panel.count_channels)
    named = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    initial_hash = parameter_hash(named)
    frozen_before, _ = native.frozen_digest(model)
    identity_errors = []
    with torch.no_grad(), legacy.precision(args):
        for origin_subset in (panel.origins["train"][:4], panel.origins["train"][-3:]):
            selected, _ = padded_origins(origin_subset)
            context, _, groups = panel.batch(selected, args.device)
            cached = cache.batch(selected, args.device)
            actual, _, _, _ = direct(model, context, groups)
            identity_errors.append(float((actual - cached[1]).abs().max()))
            if model.method in native.CACHED_METHODS:
                cached_prediction, _ = model.from_cache(*cached)
                identity_errors.append(float((actual - cached_prediction).abs().max()))
    if not all(math.isfinite(value) for value in identity_errors) or max(identity_errors) > 1e-5:
        raise AssertionError(f"Fixed-four-group zero-update identity failed: {identity_errors}")
    audits.update({"zero_update_identity": True, "zero_update_normalized_max_abs": max(identity_errors),
                   "padded_group_identity_checked": True, "initial_adaptation_sha256": initial_hash,
                   "frozen_before_sha256": frozen_before, "trainable_map_verified": True})
    parameters = [p for _, p in named]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=0., foreach=False) if parameters else None
    if optimizer is not None and {id(p) for group in optimizer.param_groups for p in group["params"]} != {id(p) for p in parameters}:
        raise AssertionError("Optimizer parameter registration mismatch")
    steps, interval = args.steps, args.val_every
    samples = np.random.default_rng(args.seed).integers(len(panel.origins["train"]), size=(steps, 8))
    sampler_hash = array_hash(panel.origins["train"][samples])
    quantile_tensor = torch.as_tensor(quantiles, dtype=torch.float32, device=args.device)
    val_target = panel.targets("val")
    val_pred = predict(model, panel, "val", args, cache)
    val_metrics, _, _ = scores(val_pred, val_target, panel.fit_std[panel.target_indices], quantiles)
    best, best_step = val_metrics["score"], 0
    checkpoint = output / "best_trainable.pt"
    legacy.save_trainable(model, checkpoint)
    history = [{"step": 0, "val_score": best}]
    log(output, "validation", **history[-1])
    durations, max_gradient = [], 0.
    for step in range(1, steps + 1):
        legacy.check_resources(args)
        tick = time.perf_counter()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_loss = 0.
        for offset in (0, 4):
            selected = panel.origins["train"][samples[step - 1, offset:offset + 4]]
            context, target, groups = panel.batch(selected, args.device)
            with legacy.precision(args):
                if model.method in native.CACHED_METHODS:
                    hidden, f0_norm, loc, scale = cache.batch(selected, args.device)
                    norm, _ = model.from_cache(hidden, f0_norm, loc, scale)
                else:
                    norm, _, loc, scale = direct(model, context, groups)
                loss = native.native_pinball(norm, target, loc, scale, quantile_tensor, model.use_arcsinh)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite native masked training loss")
            (loss / 2).backward()
            train_loss += float(loss.detach()) / 2
        gradient = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
        max_gradient = max(max_gradient, float(gradient))
        optimizer.step()
        torch.cuda.synchronize()
        durations.append(time.perf_counter() - tick)
        if step == 1:
            audits["frozen_after_first_update_sha256"] = native.frozen_digest(model)[0]
            if audits["frozen_after_first_update_sha256"] != frozen_before:
                raise AssertionError("Frozen parameters changed after an update")
        if step % interval == 0 or step == steps:
            val_pred = predict(model, panel, "val", args, cache)
            val_metrics, _, _ = scores(val_pred, val_target, panel.fit_std[panel.target_indices], quantiles)
            if val_metrics["score"] < best:
                best, best_step = val_metrics["score"], step
                legacy.save_trainable(model, checkpoint)
            history.append({"step": step, "val_score": val_metrics["score"], "train_loss": train_loss,
                            "gradient_norm": float(gradient)})
            log(output, "validation", **history[-1], best_step=best_step)
    before_restore = parameter_hash(named)
    if parameters and (max_gradient <= 0 or before_restore == initial_hash):
        raise AssertionError("No finite nonzero gradient/parameter update was observed")
    frozen_after = native.frozen_digest(model)[0]
    if frozen_after != frozen_before:
        raise AssertionError("Frozen parameters changed during training")
    legacy.load_trainable(model, checkpoint, args.device)
    restored_hash = parameter_hash(named)
    val_pred = predict(model, panel, "val", args, cache)
    val_metrics, loss_sums, counts = scores(val_pred, val_target, panel.fit_std[panel.target_indices], quantiles)
    if not np.isclose(val_metrics["score"], best, rtol=1e-6, atol=1e-8):
        raise AssertionError("Restored checkpoint does not reproduce the validation selection")
    audits.update({"finite_nonzero_gradient_verified": bool(max_gradient > 0) if parameters else None,
                   "maximum_gradient_norm": max_gradient, "before_restore_adaptation_sha256": before_restore,
                   "restored_adaptation_sha256": restored_hash,
                   "frozen_after_training_sha256": frozen_after, "frozen_parameters_verified": True,
                   "checkpoint_reload_verified": True, "checkpoint_includes_all_trainable_parameters": True,
                   "frozen_unchanged": True, "train_only_smoke": True if args.smoke else None,
                   "information_isolation": True if args.smoke else None,
                   "gradient_nonzero": bool(max_gradient > 0) if parameters else None,
                   "optimizer_steps": steps})
    np.savez_compressed(output / "predictions.npz", val_predictions=val_pred, val_target=val_target,
                        val_origins=panel.origins["val"], val_timestamps=panel.timestamps[panel.origins["val"]],
                        val_loss_sums=loss_sums, val_valid_counts=counts, quantiles=quantiles,
                        target_indices=panel.target_indices, target_channels=np.asarray(panel.channels)[panel.target_indices])
    verify_hashes(state["protected_hashes"])
    verify_hashes(cache_files)
    audits["source_unchanged"] = True
    result = {"completed": True, "stage": "fit", "method": args.method, "dataset": panel.dataset,
              "seed": args.seed, "seed_index": args.seed - 12000, "lr": args.lr, "smoke": args.smoke,
              "steps_completed": steps, "best_step": best_step, "val_score": val_metrics["score"],
              "trainable": model.trainable_count, "trainable_parameters": model.trainable_count,
              "trainable_names": model.trainable_names, "module_map": model.module_map,
              "sampler_sha256": sampler_hash, "source_hashes": state["source_hashes"],
              "native_source_hashes": state["native_source_hashes"], "plan_sha256": state["plan_sha256"],
              "fit_data_sha256": state["data_sha256"], "data_sha256": state["data_sha256"],
              "checkpoint": state["checkpoint"], "native_checkpoint_hashes": state["checkpoint_hashes"],
              "checkpoint_sha256": file_hash(checkpoint), "predictions_sha256": file_hash(output / "predictions.npz"),
              "protected_hashes": state["protected_hashes"], "cache": str(cache.path.resolve()),
              "cache_created_this_trial": created, "cache_generation_seconds": cache.manifest["wall_seconds"],
              "cache_contract_sha256": cache_hash, "cache_array_hashes": cache_files,
              "channels": panel.channels, "target_indices": panel.target_indices.tolist(),
              "fit_mean": panel.fit_mean.tolist(), "fit_std": panel.fit_std.tolist(), "fit_median": panel.fit_median.tolist(),
              "stats_sha256": panel.stats_hash, "origin_counts": {key: len(v) for key, v in panel.origins.items()},
              "origin_hashes": {key: array_hash(v) for key, v in panel.origins.items()},
              "context": 336, "horizon": 48, "micro_groups": 4, "effective_groups": 8,
              "autocast": "bfloat16", "autocast_weight_cache": False, "weight_dtype": "float32", "tf32": False,
              "gradient_clip_norm": 1., "weight_decay": 0., "dropout": 0., "torch_threads": 2, "interop_threads": 1,
              "training_loss": "Native normalized 2-pinball; non-target and originally missing labels contribute zero",
              "selection_metric": "Sorted raw quantiles, observed-cell mean within each target/train std, equal target macro",
              "holdout_file_opened": False, "validation": val_metrics, "validation_history": history, "audits": audits,
              "wall_seconds": time.perf_counter() - started, "optimizer_seconds_total": float(sum(durations)),
              "optimizer_seconds_per_step": float(np.mean(durations)) if durations else 0.,
              "peak_cuda_gib": torch.cuda.max_memory_allocated() / 2**30}
    atomic_json(output / "result.json", result)
    log(output, "fit_completed", method=args.method, val_score=result["val_score"], best_step=best_step,
        seconds=result["wall_seconds"])
    return result


def validate_selection(root, selection_path, fit_trial):
    selection_path, fit_trial = Path(selection_path).resolve(), Path(fit_trial).resolve()
    study = root / "runs" / STUDY
    if selection_path != study / "selection.json" or not fit_trial.is_relative_to(study / "trials"):
        raise ValueError("Forecasts require the final study selection and a production fit trial")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    contract_path = study / "study_contract.json"
    if not selection.get("completed") or not selection.get("global_choices_frozen") or selection.get("fit_trial_count") != 28:
        raise AssertionError("All fit and dataset choices must be frozen before any holdout forecast")
    if selection["study_contract_sha256"] != file_hash(contract_path):
        raise AssertionError("Final selection and current study contract disagree")
    entries = selection["selected"]
    expected = {(dataset, role, seed) for dataset in ("bike", "household") for role in ("F0", "H", "OFF_LORA")
                for seed in ((12000,) if role == "F0" else (12000, 12001, 12002))}
    if len(entries) != 14 or {(e["dataset"], e["role"], e["seed"]) for e in entries} != expected:
        raise AssertionError("Final selection must contain all fourteen production procedures")
    if set(selection["choices"]) != {"bike", "household"}:
        raise AssertionError("Both real datasets must have fixed choices")
    for entry in entries:
        if entry["role"] == "F0":
            if entry["method"] != "F0" or entry["lr"] != 0:
                raise AssertionError("F0 selection is not the fixed native baseline")
        else:
            chosen = selection["choices"][entry["dataset"]][entry["role"]]
            if (entry["method"], entry["lr"]) != (chosen["method"], chosen["lr"]):
                raise AssertionError("Additional seeds changed the selected family or LR")
            if entry["role"] == "H" and entry["method"] not in ("H_MLP", "H_FULL"):
                raise AssertionError("Selected H must be a declared output adaptation")
            if entry["role"] == "OFF_LORA" and entry["method"] != "OFF_LORA":
                raise AssertionError("Selected internal adapter must use the native official map")
        path = (root / entry["path"]).resolve()
        if not path.is_relative_to(study / "trials") or file_hash(path / "result.json") != entry["result_sha256"]:
            raise AssertionError("Selected fit path or result hash changed")
        if file_hash(path / "best_trainable.pt") != entry["checkpoint_sha256"]:
            raise AssertionError("Selected adaptive checkpoint changed")
        meta = json.loads((path / "result.json").read_text(encoding="utf-8"))
        guard = json.loads((path / "guard/status.json").read_text(encoding="utf-8"))
        if not meta.get("completed") or meta.get("smoke") or not guard.get("completed") or guard["returncode"] != 0 or guard["reasons"]:
            raise AssertionError("A selected fit or guard did not complete")
        if any(meta[key] != entry[key] for key in ("dataset", "method", "seed", "lr")):
            raise AssertionError("Selection does not identify its actual fit result")
    matches = [entry for entry in entries if (root / entry["path"]).resolve() == fit_trial]
    if len(matches) != 1:
        raise AssertionError("The requested fit is not uniquely selected")
    study_contract = json.loads(contract_path.read_text(encoding="utf-8"))
    protected = {str(root / key): value for key, value in
                 {**study_contract["sources"], **study_contract["data_files"]}.items()}
    protected[str(root / PLAN)] = study_contract["plan_sha256"]
    protected[str(root / "runs/peft_trainlag_v1/study_contract.json")] = study_contract["previous_contract_sha256"]
    verify_hashes(protected)
    return matches[0], {**protected, str(selection_path): file_hash(selection_path),
                        str(contract_path): file_hash(contract_path)}


def run_forecast(args):
    started = time.perf_counter()
    root = Path(__file__).resolve().parents[2]
    entry, selection_hashes = validate_selection(root, args.selection, args.fit_trial)
    fit_trial = Path(args.fit_trial).resolve()
    fit_result = json.loads((fit_trial / "result.json").read_text(encoding="utf-8"))
    if Path(args.checkpoint).resolve() != Path(fit_result["checkpoint"]).resolve():
        raise AssertionError("Forecast must use its fit's native checkpoint")
    verify_hashes(fit_result["protected_hashes"])
    verify_hashes(fit_result["cache_array_hashes"])
    output = new_output(root, args.output)
    state = source_state(root, args)
    if state["source_hashes"] != fit_result["source_hashes"] or state["checkpoint_hashes"] != fit_result["native_checkpoint_hashes"]:
        raise AssertionError("Forecast source/native checkpoint differs from the completed fit")
    panel = Panel(args.data, "forecast")
    if Path(args.data).resolve() != root / "runs" / STUDY / "prepared" / f"{entry['dataset']}_holdout.npz":
        raise ValueError("Use the globally frozen holdout archive for the selected dataset")
    with np.load(root / "runs" / STUDY / "prepared" / f"{entry['dataset']}_fit.npz", allow_pickle=False) as archive:
        fit_manifest = json.loads(archive["manifest_json"].item())
    if fit_manifest["holdout_archive_sha256"] != state["data_sha256"]:
        raise AssertionError("Holdout archive does not match the original fit preparation")
    if (panel.dataset, panel.channels, panel.target_indices.tolist(), panel.stats_hash) != (
            fit_result["dataset"], fit_result["channels"], fit_result["target_indices"], fit_result["stats_sha256"]):
        raise AssertionError("Holdout channels or train-only statistics differ from fitting")
    protected = {**state["protected_hashes"], **fit_result["protected_hashes"], **selection_hashes,
                 str(fit_trial / "result.json"): entry["result_sha256"],
                 str(fit_trial / "best_trainable.pt"): entry["checkpoint_sha256"]}
    atomic_json(output / "trial_contract.json", {**state, "stage": "forecast", "selected_entry": entry,
                "selection_hashes": selection_hashes, "protected_hashes": protected})
    base = load_base(args)
    quantiles = np.asarray(base.chronos_config.quantiles, dtype=np.float64)
    if not np.array_equal(quantiles, panel.quantiles):
        raise AssertionError("Native and held-out quantile grids differ")
    model = construct(base, entry["method"], entry["seed"], panel.count_channels)
    legacy.load_trainable(model, fit_trial / "best_trainable.pt", args.device)
    loaded_hash = parameter_hash((n, p) for n, p in model.named_parameters() if p.requires_grad)
    if loaded_hash != fit_result["audits"]["restored_adaptation_sha256"]:
        raise AssertionError("Forecast checkpoint does not reproduce the selected adaptive parameters")
    frozen_before = native.frozen_digest(model)[0]
    arrays, metadata = {"quantiles": quantiles, "target_indices": panel.target_indices,
                        "target_channels": np.asarray(panel.channels)[panel.target_indices]}, {}
    for split in ("cal", "eval"):
        prediction, unsorted = predict(model, panel, split, args, return_unsorted=True)
        target = panel.targets(split)
        for name, value in {"predictions": prediction, "unsorted_predictions": unsorted, "target": target,
                            "origins": panel.origins[split], "timestamps": panel.timestamps[panel.origins[split]]}.items():
            arrays[f"{split}_{name}"] = value
        metadata[split] = {"origins": len(panel.origins[split]), "target_shape": list(target.shape),
                           "predictions_shape": list(prediction.shape),
                           "origin_sha256": array_hash(panel.origins[split]),
                           "target_sha256": array_hash(target), "unsorted_crossing": float((np.diff(unsorted, axis=2) < 0).mean())}
    if frozen_before != native.frozen_digest(model)[0] or loaded_hash != parameter_hash((n, p) for n, p in model.named_parameters() if p.requires_grad):
        raise AssertionError("Forecast inference changed model parameters")
    np.savez_compressed(output / "predictions.npz", **arrays)
    verify_hashes(protected)
    result = {"completed": True, "stage": "forecast", "dataset": panel.dataset, "method": entry["method"],
              "role": entry["role"], "seed": entry["seed"], "lr": entry["lr"], "best_step": fit_result["best_step"],
              "fit_trial": str(fit_trial), "fit_result_sha256": entry["result_sha256"],
              "fit_checkpoint_sha256": entry["checkpoint_sha256"], "fit_data_sha256": fit_result["fit_data_sha256"],
              "holdout_data_sha256": state["data_sha256"], "source_hashes": state["source_hashes"],
              "native_checkpoint_hashes": state["checkpoint_hashes"], "plan_sha256": state["plan_sha256"],
              "selection_sha256": file_hash(args.selection), "study_contract_sha256": file_hash(root / "runs" / STUDY / "study_contract.json"),
              "fit_cache": fit_result["cache"], "cache_created_this_trial": False,
              "channels": panel.channels, "target_indices": panel.target_indices.tolist(), "fit_std": panel.fit_std.tolist(),
              "stats_sha256": panel.stats_hash, "trainable": COUNTS[entry["method"]],
              "trainable_parameters": COUNTS[entry["method"]], "context": 336, "horizon": 48,
              "micro_groups": 4, "autocast": "bfloat16", "autocast_weight_cache": False, "weight_dtype": "float32", "tf32": False,
              "optimizer_steps": 0, "model_unchanged": True, "global_selection_verified_before_holdout_load": True,
              "checkpoint_reload_verified": True, "restored_adaptation_sha256": loaded_hash,
              "predictions_sha256": file_hash(output / "predictions.npz"), "splits": metadata,
              "protected_hashes": protected, "wall_seconds": time.perf_counter() - started,
              "peak_cuda_gib": torch.cuda.max_memory_allocated() / 2**30}
    atomic_json(output / "result.json", result)
    log(output, "forecast_completed", dataset=panel.dataset, method=entry["method"], seed=entry["seed"],
        seconds=result["wall_seconds"])
    return result


def parser():
    main = argparse.ArgumentParser(description=__doc__)
    commands = main.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("fit")
    forecast = commands.add_parser("forecast")
    for command in (fit, forecast):
        command.add_argument("--data", required=True)
        command.add_argument("--output", required=True)
        command.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
        command.add_argument("--device", default="cuda")
        command.add_argument("--min-free-ram-gib", type=float, default=5.)
    fit.add_argument("--cache", required=True)
    fit.add_argument("--method", choices=METHODS, required=True)
    fit.add_argument("--lr", type=float, required=True)
    fit.add_argument("--seed", type=int, required=True)
    fit.add_argument("--steps", type=int, default=200)
    fit.add_argument("--val-every", type=int, default=40)
    fit.add_argument("--smoke", action="store_true")
    forecast.add_argument("--fit-trial", required=True)
    forecast.add_argument("--selection", required=True)
    return main


def main():
    args = parser().parse_args()
    try:
        (run_fit if args.command == "fit" else run_forecast)(args)
    except Exception as error:
        output = Path(args.output)
        root = Path(__file__).resolve().parents[2]
        if output.exists() and output.resolve().is_relative_to(root / "runs" / STUDY):
            atomic_json(output / "failure.json", {"completed": False, "stage": args.command,
                        "error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
