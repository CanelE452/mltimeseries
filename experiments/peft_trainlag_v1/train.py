"""Train-only lag alignment with the unchanged native attention-LoRA training loop."""

import argparse
from contextlib import contextmanager
import hashlib
import inspect
import json
from pathlib import Path
import time
import traceback

from experiments.peft_shift_mechanism_v1 import train as shared
from experiments.peft_module_ablation_v1 import train as ablation


METHODS = ("F0", "ATTN_ONLY")
INPUT_MODES = ("raw", "aligned")
PLAN = "_docs/notes/tsfm_topics/10_peft_trainlag_plan_20260908.md"
BASE_SEED = 2026090810
COUNTS = {"F0": 0, "ATTN_ONLY": 1179648}


def report_method(method, input_mode):
    return ("ALIGN_" if input_mode == "aligned" else "") + ("ATTN" if method == "ATTN_ONLY" else "F0")


def source_indices(lag):
    if isinstance(lag, bool) or int(lag) != lag or not 16 <= lag <= 128:
        raise ValueError("Every selected lag must be an integer in 16..128")
    lag = int(lag)
    past = shared.np.arange(lag, 256, dtype=shared.np.int64) - lag
    future = 256 + shared.np.arange(16, dtype=shared.np.int64) - lag
    if min(past.min(), future.min()) < 0 or max(past.max(), future.max()) >= 256:
        raise AssertionError("Alignment reads an originally unobserved future value")
    return past, future


def transformed_inputs(context, input_mode, lags):
    """Retiming preserves the autograd graph; only observed context values are read."""
    if context.shape != (12, 256) or input_mode not in INPUT_MODES:
        raise ValueError("Use four complete Y/U/V groups and a declared input mode")
    if input_mode == "raw":
        return context, None, None
    original = context.reshape(4, 3, 256)
    shifted = original.clone()
    future = context.new_zeros((4, 3, 16))
    mask = context.new_zeros((4, 3, 16))
    for channel, name in ((1, "U"), (2, "V")):
        lag = int(lags[name])
        source_indices(lags[name])
        shifted[:, channel, :lag] = float("nan")
        shifted[:, channel, lag:] = original[:, channel, :256 - lag]
        future[:, channel] = original[:, channel, 256 - lag:272 - lag]
        mask[:, channel] = 1
    return shifted.reshape(12, 256), future.reshape(12, 16), mask.reshape(12, 16)


def encoder(original_encode, input_mode, lags):
    def encode(base, context, groups, **extra):
        if set(extra) - {"future_target", "future_target_mask"}:
            raise ValueError("Known-future inputs must be built from observed context by this wrapper")
        shifted, future, mask = transformed_inputs(context, input_mode, lags)
        if input_mode == "aligned":
            extra = {**extra, "future_covariates": future, "future_covariates_mask": mask}
        return original_encode(base, shifted, groups, **extra)
    return encode


def input_scope(input_mode, lags):
    indices = {name: {"lag": int(lags[name]), "retained_past_source_range": [0, 255 - int(lags[name])],
                      "future_source_indices": source_indices(lags[name])[1].tolist(),
                      "padding": "NaN", "original_recent_values_dropped": int(lags[name]) - 16}
               for name in ("U", "V")}
    return {"input_mode": input_mode, "selected_lags": {name: int(lags[name]) for name in ("Y", "U", "V")},
            "Y_past_unchanged": True, "Y_future_mask": 0, "additional_original_future_observations": 0,
            "driver_alignment": indices if input_mode == "aligned" else None,
            "Y_lag_used_by_FM": False, "normalization": "Native instance normalization on supplied past; NaNs excluded",
            "intervention": "Retiming, NaN padding, retained context, normalization and native known-future path jointly change"
                            if input_mode == "aligned" else "Original finite past only",
            "channel_identity_note": "Distinct named U/V lags can break the old U/V-swap symmetry through preprocessing"}


def information_auditor(original_encode, input_mode, lags):
    def audit(base, episodes, device):
        context, _, groups = episodes.batch("train", shared.np.arange(4), device)
        original_context = context.clone()
        shifted, future, mask = transformed_inputs(context, input_mode, lags)
        native_extra = ({} if input_mode == "raw" else
                        {"future_covariates": future, "future_covariates_mask": mask})
        with shared.torch.no_grad(), shared.precision(device):
            actual = shared.encode_y(base, context, groups)
            direct = original_encode(base, shifted, groups, **native_extra)
            target_changed = shared.encode_y(base, context, groups,
                                             future_target=context.new_full((12, 16), 123))
            altered_future = (context.new_zeros((12, 16)) if future is None else future.clone())
            altered_mask = (context.new_zeros((12, 16)) if mask is None else mask)
            altered_future[altered_mask == 0] = 123
            masked = original_encode(base, shifted, groups, future_covariates=altered_future,
                                     future_covariates_mask=altered_mask)
            other = context.clone()
            other[3:] = 2 * other[3:] + 17
            isolated = shared.encode_y(base, other, groups)
            swapped = context.reshape(4, 3, 256)[:, [0, 2, 1]].reshape(12, 256)
            exchanged = shared.encode_y(base, swapped, groups)
        errors = {"wrapper_native_hidden_max_abs": float((actual[0] - direct[0]).abs().max()),
                  "future_target_hidden_max_abs": float((actual[0] - target_changed[0]).abs().max()),
                  "masked_Y_future_hidden_max_abs": float((actual[0] - masked[0]).abs().max()),
                  "other_group_hidden_max_abs": float((actual[0][:1] - isolated[0][:1]).abs().max())}
        if any(value > 1e-5 for value in errors.values()) or not shared.torch.equal(context, original_context):
            raise AssertionError(f"Input isolation or input immutability failed: {errors}")
        raw = shared.normalized_to_raw(actual[1], actual[2], actual[3], base.chronos_config.use_arcsinh)
        swap_raw = shared.normalized_to_raw(exchanged[1], exchanged[2], exchanged[3], base.chronos_config.use_arcsinh)
        return {**errors, "future_target_isolation": True, "masked_future_isolation": True,
                "group_isolation": True, "original_input_unchanged": True,
                "original_past_indices_verified": True, "raw_encode_identity_verified": input_mode == "raw",
                "known_UV_future_values_preserved_during_mask_audit": input_mode == "aligned",
                "encoder_rows": 12, "groups": 4,
                "uv_swap_y_normalized_max_abs": float((actual[1] - exchanged[1]).abs().max()),
                "uv_swap_y_raw_max_abs": float((raw - swap_raw).abs().max()),
                "symmetry_note": "Pretrained numerical diagnostic only; unequal named-channel lags can break preprocessing swap symmetry",
                "input_scope": input_scope(input_mode, lags)}
    return audit


def cache_contract(episodes, args, inputs):
    return {"version": 1, "study": "peft_trainlag_v1", "data_sha256": inputs["data_sha256"],
            "metadata": episodes.metadata, "lag_file_sha256": inputs["lag_file_sha256"],
            "lag_selection_input_sha256": inputs["lag_selection_input_sha256"],
            "input_scope": input_scope(args.input_mode, inputs["lags"]),
            "checkpoint": str(Path(args.checkpoint).resolve()), "checkpoint_hashes": inputs["checkpoint_hashes"],
            "source_hashes": inputs["source_hashes"], "native_source_hashes": inputs["native_source_hashes"],
            "plan_sha256": inputs["plan_sha256"],
            "counts": {split: len(values) for split, values in episodes.targets.items()},
            "context": 256, "horizon": 16, "channel_order": ["Y", "U", "V"],
            "micro_groups": 4, "encoder_rows": 12, "target_rows": 4,
            "autocast": "bfloat16", "autocast_weight_cache": False, "weight_dtype": "float32",
            "tf32": False, "dropout": 0.0, "smoke": args.smoke}


def cache_signature(contract):
    return hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()[:20]


def cache_loader(inputs, generated):
    def prepare(base, episodes, args, output):
        contract = cache_contract(episodes, args, inputs)
        path = Path(args.cache) / cache_signature(contract)
        manifest_path = path / "manifest.json"
        if base.model_dim != 768 or base.num_quantiles != 21:
            raise AssertionError("Unexpected native cache dimensions")
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not manifest.get("completed") or manifest["contract"] != contract:
                raise AssertionError("Input-mode/lag/data/source/model cache contract mismatch")
            for name, expected in manifest["array_sha256"].items():
                if shared.file_hash(path / name) != expected:
                    raise AssertionError(f"Cached array changed: {name}")
            created = False
        else:
            if args.method != "F0":
                raise RuntimeError("Generate and audit the matching F0 cache before adapting attention")
            path.mkdir(parents=True, exist_ok=True)
            started = time.perf_counter()
            with shared.torch.no_grad():
                for split in ("train", "val", "eval"):
                    count = len(episodes.targets[split])
                    shapes = {"hidden": (count, 1, 768), "norm": (count, 21, 16), "loc": (count, 1), "scale": (count, 1)}
                    arrays = {name: shared.np.lib.format.open_memmap(path / f"{split}_{name}.npy", mode="w+",
                              dtype=shared.np.float32, shape=shape) for name, shape in shapes.items()}
                    for start in range(0, count, 4):
                        shared.check_resources()
                        indices = shared.np.arange(start, start + 4)
                        context, _, groups = episodes.batch(split, indices, args.device)
                        with shared.precision(args.device):
                            values = shared.encode_y(base, context, groups)
                        for name, value in zip(("hidden", "norm", "loc", "scale"), values):
                            arrays[name][indices] = value.float().cpu().numpy()
                    for array in arrays.values():
                        array.flush()
                    del arrays, array
                    shared.log_event(output, "cache", split=split, completed=count,
                                     seconds=time.perf_counter() - started)
            manifest = {"completed": True, "contract": contract, "wall_seconds": time.perf_counter() - started,
                        "checkpoint_weights": {k: v for k, v in inputs["checkpoint_hashes"].items() if k.endswith(".safetensors")},
                        "array_sha256": {file.name: shared.file_hash(file) for file in sorted(path.glob("*.npy"))}}
            shared.atomic_json(manifest_path, manifest)
            created = True
        generated.update({str(file.resolve()): shared.file_hash(file) for file in
                          [manifest_path, *sorted(path.glob("*.npy"))]})
        return shared.Cache(path, manifest), created
    return prepare


@contextmanager
def registration(args, inputs, state):
    keys = ("METHODS", "INTERNAL_METHODS", "EXPECTED_PARAMETERS", "lora_targets", "ShiftModel",
            "atomic_json", "prepare_cache", "encode_y", "information_audit", "load_adaptation")
    original = {key: getattr(shared, key) for key in keys}
    result_path = (Path(args.output) / "result.json").resolve()
    hooks = []

    def targets(base, method):
        return ablation.subset_targets(base, method, original["lora_targets"])

    def construct(base, method, seed):
        model = original["ShiftModel"](base, method, seed)
        if method == "ATTN_ONLY":
            state["model"] = ablation.model_audit(model, method, seed)
            state["initial_lora_sha256"] = shared.parameter_hash(model.parameters_named(model.lora_names))
            if args.smoke:
                for name, parameter in model.parameters_named(model.lora_names):
                    if ".lora_B." not in name:
                        continue
                    def capture(gradient, name=name):
                        if name not in state["first_B_gradients"]:
                            if not shared.torch.isfinite(gradient).all():
                                raise FloatingPointError(f"Nonfinite LoRA gradient: {name}")
                            state["first_B_gradients"][name] = float(gradient.detach().abs().max())
                        return gradient
                    hooks.append(parameter.register_hook(capture))
        else:
            if model.adaptive_names or model.probe is not None:
                raise AssertionError("F0 must have no trainable parameters or new head")
            state["model"] = {"adaptive_count": 0, "residual_head_absent": True, "module_map": []}
        return model

    def restore(model, path, device):
        if model.method == "ATTN_ONLY":
            state["before_restore_lora_sha256"] = shared.parameter_hash(model.parameters_named(model.lora_names))
            if state["before_restore_lora_sha256"] == state["initial_lora_sha256"]:
                raise AssertionError("Attention parameters never changed during training")
        return original["load_adaptation"](model, path, device)

    def pending_json(path, value):
        if Path(path).resolve() == result_path:
            value = {**value, "completed": False, "wrapped_completed": False,
                     "inner_training_completed": bool(value.get("completed"))}
        return original["atomic_json"](path, value)

    try:
        shared.METHODS = (*original["METHODS"], "ATTN_ONLY")
        shared.INTERNAL_METHODS = original["INTERNAL_METHODS"] | {"ATTN_ONLY"}
        shared.EXPECTED_PARAMETERS = {**original["EXPECTED_PARAMETERS"], "ATTN_ONLY": COUNTS["ATTN_ONLY"]}
        shared.lora_targets, shared.ShiftModel = targets, construct
        shared.encode_y = encoder(original["encode_y"], args.input_mode, inputs["lags"])
        shared.information_audit = information_auditor(original["encode_y"], args.input_mode, inputs["lags"])
        shared.prepare_cache = cache_loader(inputs, state["cache_hashes"])
        shared.load_adaptation, shared.atomic_json = restore, pending_json
        yield
    finally:
        for hook in hooks:
            hook.remove()
        for key, value in original.items():
            setattr(shared, key, value)


def validate_args(args):
    if args.method not in METHODS or args.input_mode not in INPUT_MODES:
        raise ValueError("Use F0/ATTN_ONLY and raw/aligned")
    allowed_lr = (.001,) if args.method == "F0" else (3e-5, 1e-4)
    if args.lr not in allowed_lr or args.steps != 200 or args.val_every != 40:
        raise ValueError("Retain the declared LR, 200 updates, and validation interval 40")
    if not args.device.startswith("cuda") or args.lp_fraction != .4:
        raise ValueError("Use the unchanged CUDA execution path and unused default LP fraction")
    if Path(args.checkpoint).resolve() != shared.DEFAULT_CHECKPOINT.resolve():
        raise ValueError("Use the declared frozen Chronos-2 checkpoint")


def protected_hashes(paths):
    return {str(Path(path).resolve()): shared.file_hash(path) for path in paths}


def verify_hashes(hashes):
    for path, expected in hashes.items():
        if shared.file_hash(path) != expected:
            raise AssertionError(f"Protected source/input/cache changed during execution: {path}")


def selection_array_hash(*arrays):
    digest = hashlib.sha256()
    for value in arrays:
        array = shared.np.ascontiguousarray(value)
        digest.update(str(array.shape).encode())
        digest.update(str(array.dtype).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def validate_selection(selection, context_train, target_train, data_hash):
    if selection.get("data_sha256") != data_hash:
        raise AssertionError("Lag selection belongs to a different prepared dataset")
    expected = {"input_array_hash": selection_array_hash(context_train, target_train),
                "context_train_sha256": selection_array_hash(context_train),
                "target_train_sha256": selection_array_hash(target_train)}
    if any(selection.get(key) != value for key, value in expected.items()):
        raise AssertionError("Lag selection input hashes differ from train-only arrays")
    if (context_train.shape, target_train.shape) != ((64, 3, 256), (64, 16)):
        raise AssertionError("Lag estimation must use all 64 train episodes, including during S0")
    if selection.get("n_train_episodes") != 64 or selection.get("n_future_labels") != 1024:
        raise AssertionError("Lag selection used a different supervision budget")
    if any(selection.get(key) is not False for key in
           ("known_lag_dictionary_used", "validation_or_eval_used", "oracle_or_manifest_used")):
        raise AssertionError("Lag selection information scope is not the train-only contract")
    candidates = list(range(16, 129))
    if selection.get("candidate_lags") != candidates or selection.get("candidates") != candidates:
        raise AssertionError("Lag selector used a different candidate grid")
    lags = selection["selected_lags"]
    if set(lags) != {"Y", "U", "V"}:
        raise AssertionError("Exactly one lag per named Y/U/V channel is required")
    for name in ("Y", "U", "V"):
        source_indices(lags[name])
        scores = shared.np.asarray(selection["scores"][name], dtype=shared.np.float64)
        correlations = shared.np.asarray(selection["pearson_correlations"][name], dtype=shared.np.float64)
        if scores.shape != (113,) or not shared.np.isfinite(scores).all() or correlations.shape != (113,):
            raise AssertionError("Incomplete or nonfinite train-lag score grid")
        if not shared.np.allclose(scores, correlations**2, rtol=1e-12, atol=1e-15):
            raise AssertionError("Stored lag scores are not squared Pearson correlation")
        if candidates[int(shared.np.argmax(scores))] != lags[name]:
            raise AssertionError("Selected lag differs from the sorted-grid first maximum")
    return {name: int(lags[name]) for name in ("Y", "U", "V")}


def study_inputs(args, root):
    data_path, lag_path = Path(args.data).resolve(), Path(args.lag_file).resolve()
    folder = root / "runs/peft_trainlag_v1/data"
    data_hash = shared.file_hash(data_path)
    selection = json.loads(lag_path.read_text(encoding="utf-8"))
    with shared.np.load(data_path, allow_pickle=False) as archive:
        metadata = json.loads(archive["manifest_json"].item())
        lags = validate_selection(selection, archive["context_train"], archive["target_train"], data_hash)
        if not shared.np.array_equal(archive["selected_lags"], [lags[name] for name in ("Y", "U", "V")]):
            raise AssertionError("Archive and separately locked lag selection disagree")
    corpus = metadata["corpus"]
    if metadata["condition"] != "Q00" or corpus not in (0, 1, 2) or metadata["base_seed"] != BASE_SEED:
        raise AssertionError("Use only the fresh, fixed-seed Q00 corpus0/1/2")
    if args.seed != 8100 + corpus:
        raise AssertionError("Keep the paired optimizer initialization and sampler seed")
    if data_path != folder / f"Q00_c{corpus}.npz" or lag_path != folder / f"Q00_c{corpus}_lags.json":
        raise AssertionError("Use the declared fresh dataset and its matching train-only lag file")
    manifest_path = folder / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    lag_hash = shared.file_hash(lag_path)
    if (not manifest["all_qc_passed"] or manifest["files"][data_path.name]["sha256"] != data_hash
            or manifest["lag_selection_files"][lag_path.name]["sha256"] != lag_hash):
        raise AssertionError("Dataset/lag hashes or QC differ from the generation manifest")
    sources = [Path(__file__).resolve(), Path(shared.__file__).resolve(), Path(ablation.__file__).resolve(),
               root / "experiments/peft_trainlag_v1/data.py"]
    sources.extend(root / "experiments/peft_adaptation_scope_v1" / name for name in ("modeling.py", "train.py", "data.py"))
    source_hashes = {path.relative_to(root).as_posix(): shared.file_hash(path) for path in sources}
    checkpoint = Path(args.checkpoint).resolve()
    weight_files = sorted(checkpoint.glob("*.safetensors"))
    if not weight_files:
        raise FileNotFoundError("Native checkpoint has no local safetensors weights")
    checkpoint_files = [checkpoint / "config.json", *weight_files]
    checkpoint_hashes = {path.name: shared.file_hash(path) for path in checkpoint_files}
    from chronos.chronos2.model import Chronos2Model

    native_folder = Path(inspect.getsourcefile(Chronos2Model)).resolve().parent
    native_sources = sorted(native_folder.glob("*.py"))
    paths = [data_path, lag_path, manifest_path, root / PLAN, *sources, *checkpoint_files, *native_sources]
    return {"lags": lags, "corpus": corpus, "data_sha256": data_hash, "lag_file_sha256": lag_hash,
            "lag_selection_input_sha256": selection["input_array_hash"], "source_hashes": source_hashes,
            "plan_sha256": shared.file_hash(root / PLAN), "checkpoint_hashes": checkpoint_hashes,
            "native_source_hashes": protected_hashes(native_sources), "protected_hashes": protected_hashes(paths)}


def finish_result(args, result, inputs, state, contract, started):
    output = Path(args.output)
    pending = json.loads((output / "result.json").read_text(encoding="utf-8"))
    if pending.get("completed") or pending.get("wrapped_completed") or not pending.get("inner_training_completed"):
        raise AssertionError("Inner training bypassed the wrapper completion gate")
    if not result.get("completed") or "model" not in state:
        raise AssertionError("The shared loop did not produce an audited model")
    if result["method"] != args.method or result["trainable_parameters"] != COUNTS[args.method]:
        raise AssertionError("Saved method or trainable parameter count differs from the contract")
    wanted_map = ablation.expected_map("ATTN_ONLY") if args.method == "ATTN_ONLY" else []
    if result["module_map"] != wanted_map or result["head_lr"] is not None or result["phase_transitions"]:
        raise AssertionError("Unexpected adapter map, extra head, or phased training")
    steps = (5 if args.smoke else 200) if args.method == "ATTN_ONLY" else 0
    samples = shared.np.random.default_rng(args.seed).integers(8 if args.smoke else 64, size=(steps, 8))
    sampler = hashlib.sha256(samples.astype(shared.np.int64).tobytes()).hexdigest()
    if result["steps_completed"] != steps or result["sampler_sha256"] != sampler:
        raise AssertionError("Sampler or training-budget contract changed")
    if args.method == "ATTN_ONLY" and result["cache_created_this_trial"]:
        raise AssertionError("Attention training must reuse the matching frozen F0 cache")
    if args.method == "ATTN_ONLY" and args.smoke:
        if len(state["first_B_gradients"]) != 96 or max(state["first_B_gradients"].values(), default=0) <= 0:
            raise AssertionError("S0 did not observe finite nonzero attention-LoRA B gradients")
    for name in ("zero_update_identity", "frozen_parameters_verified", "checkpoint_reload_verified"):
        if not result["audits"].get(name):
            raise AssertionError(f"Required native-loop audit missing: {name}")
    verify_hashes(inputs["protected_hashes"])
    verify_hashes(state["cache_hashes"])
    result.update({"completed": True, "wrapped_completed": True, "inner_training_completed": True,
                   "base_method": args.method, "method": report_method(args.method, args.input_mode),
                   "input_mode": args.input_mode, "selected_lags": inputs["lags"],
                   "lag_file_sha256": inputs["lag_file_sha256"],
                   "lag_selection_input_sha256": inputs["lag_selection_input_sha256"],
                   "wrapper_contract": contract, "wrapper_audits": state,
                   "wrapper_wall_seconds": time.perf_counter() - started})
    result["source_hashes"].update(inputs["source_hashes"])
    result["audits"]["input_scope"] = input_scope(args.input_mode, inputs["lags"])
    if not args.smoke:
        result["audits"]["information_audit"] = "S0 uses this same mode/lag-aware four-group encoder"
    shared.atomic_json(output / "result.json", result)
    return result


def run(args):
    started = time.perf_counter()
    validate_args(args)
    root = Path(__file__).resolve().parents[2]
    study = root / "runs/peft_trainlag_v1"
    output, cache = Path(args.output).resolve(), Path(args.cache).resolve()
    if not output.is_relative_to(study) or not cache.is_relative_to(study):
        raise ValueError("New trial and cache paths must stay inside runs/peft_trainlag_v1")
    if (output / "result.json").exists() or (output / "wrapper_contract.json").exists():
        raise FileExistsError("Preserve the existing trial and use a new output directory")
    inputs = study_inputs(args, root)
    output.mkdir(parents=True, exist_ok=True)
    contract = {"study": "peft_trainlag_v1", "kind": "Train-only lag selection followed by native attention adaptation",
                "method": report_method(args.method, args.input_mode), "base_method": args.method,
                "source_sha256": shared.file_hash(Path(__file__)), "source_hashes": inputs["source_hashes"],
                "native_source_hashes": inputs["native_source_hashes"], "plan_sha256": inputs["plan_sha256"],
                "data_sha256": inputs["data_sha256"], "lag_file_sha256": inputs["lag_file_sha256"],
                "lag_selection_input_sha256": inputs["lag_selection_input_sha256"],
                "checkpoint_hashes": inputs["checkpoint_hashes"], "protected_hashes": inputs["protected_hashes"],
                "corpus": inputs["corpus"], "input_scope": input_scope(args.input_mode, inputs["lags"]),
                "module_map": ablation.expected_map("ATTN_ONLY") if args.method == "ATTN_ONLY" else [],
                "trainable_parameters": COUNTS[args.method], "new_residual_head": False,
                "requested_arguments": vars(args).copy(),
                "common_training": {"steps": (5 if args.smoke else 200) if args.method == "ATTN_ONLY" else 0,
                                    "val_every": 1 if args.smoke else 40, "effective_groups": 8, "micro_groups": 4,
                                    "gradient_clip_norm": 1.0, "weight_decay": 0.0, "weight_dtype": "float32",
                                    "autocast": "bfloat16", "autocast_weight_cache": False, "tf32": False,
                                    "target": "Y only", "loss": "Unchanged shared.native_pinball",
                                    "new_training_loop": False}}
    shared.atomic_json(output / "wrapper_contract.json", contract)
    state = {"first_B_gradients": {}, "cache_hashes": {}}
    with registration(args, inputs, state):
        result = shared.run(argparse.Namespace(**vars(args)))
    return finish_result(args, result, inputs, state, contract, started)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--data", required=True)
    result.add_argument("--output", required=True)
    result.add_argument("--cache", required=True)
    result.add_argument("--method", choices=METHODS, required=True)
    result.add_argument("--input-mode", choices=INPUT_MODES, required=True)
    result.add_argument("--lag-file", required=True)
    result.add_argument("--lr", type=float, default=1e-4)
    result.add_argument("--seed", type=int, required=True)
    result.add_argument("--steps", type=int, default=200)
    result.add_argument("--val-every", type=int, default=40)
    result.add_argument("--checkpoint", default=str(shared.DEFAULT_CHECKPOINT))
    result.add_argument("--device", default="cuda")
    result.add_argument("--lp-fraction", type=float, default=.4)
    result.add_argument("--smoke", action="store_true")
    return result


def main():
    args = parser().parse_args()
    try:
        run(args)
    except Exception as error:
        output = Path(args.output)
        if output.exists() and output.resolve().is_relative_to(Path(__file__).resolve().parents[2] / "runs/peft_trainlag_v1"):
            shared.atomic_json(output / "failure.json", {"completed": False, "wrapped_completed": False,
                               "error_type": type(error).__name__, "error": str(error),
                               "traceback": traceback.format_exc(), "time": time.time()})
        raise


if __name__ == "__main__":
    main()
