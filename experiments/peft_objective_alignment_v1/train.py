"""Fixed-budget objective comparison using the frozen study12 numerical helpers."""

import argparse
import hashlib
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

from experiments.peft_external_gap_v1 import train as shared

native, legacy = shared.native, shared.legacy
file_hash, atomic_json, array_hash = shared.file_hash, shared.atomic_json, shared.array_hash
STUDY = "peft_objective_alignment_v1"
ARMS = ("NATIVE", "NORM_ALIGNED", "RAW_ALIGNED")
ROOT = Path(__file__).resolve().parents[2]


def calibration_indices(count):
    if count < 8:
        raise ValueError("The fixed gradient calibration needs eight training origins")
    return np.linspace(0, count - 1, 8).astype(int)


def gradient_vector(named):
    blocks = []
    for _, parameter in named:
        value = parameter.grad
        blocks.append(torch.zeros(parameter.numel(), dtype=torch.float32) if value is None
                      else value.detach().reshape(-1).float().cpu())
    vector = torch.cat(blocks).double()
    if not torch.isfinite(vector).all():
        raise FloatingPointError("Initial objective gradient is not finite")
    return vector


def gradient_summary(vectors):
    norms = {arm: float(torch.linalg.vector_norm(vectors[arm])) for arm in ARMS}
    if any(not math.isfinite(value) or value <= 0 for value in norms.values()):
        raise FloatingPointError("Every initial objective gradient norm must be finite and positive")
    multipliers = {arm: norms["NATIVE"] / norms[arm] for arm in ARMS}
    multipliers["NATIVE"] = 1.
    if any(not math.isfinite(value) or value <= 0 for value in multipliers.values()):
        raise FloatingPointError("The fixed objective multiplier is invalid")
    cosines = {f"{left}__{right}": float(torch.dot(vectors[left], vectors[right]) / (norms[left] * norms[right]))
               for index, left in enumerate(ARMS) for right in ARMS[index + 1:]}
    return {"norms": norms, "multipliers": multipliers, "cosines": cosines}


def clear_gradients(model):
    model.zero_grad(set_to_none=True)
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise AssertionError("Gradient calibration left parameter gradients behind")


def runtime_args(args, data):
    return argparse.Namespace(**vars(args), data=str(data), device="cuda", min_free_ram_gib=5.,
                              checkpoint=str(shared.DEFAULT_CHECKPOINT), method="OFF_LORA", seed=12000,
                              lr=1e-5 if args.smoke else 3e-5, steps=5 if args.smoke else 200,
                              val_every=5 if args.smoke else 40)


def new_output(output):
    output = Path(output).resolve()
    if not output.is_relative_to(ROOT / "runs" / STUDY):
        raise ValueError("Keep all new artifacts inside the objective-alignment study")
    if any((output / name).exists() for name in ("result.json", "trial_contract.json", "best_trainable.pt",
                                               "predictions.npz", "failure.json")):
        raise FileExistsError("Preserve existing complete and partial trial outputs")
    output.mkdir(parents=True, exist_ok=True)
    return output


def absolute(path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def study_state(args, runtime):
    from . import prepare

    contract = prepare.validate_contract(ROOT, absolute(args.contract))
    old = shared.source_state(ROOT, runtime)
    protected = {str(absolute(path)): digest for path, digest in contract["protected_hashes"].items()}
    protected.update(old["protected_hashes"])
    protected.update({str(absolute(value["path"])): value["sha256"] for value in contract["source_hashes"].values()})
    protected[str(absolute(contract["plan_path"]))] = contract["plan_sha256"]
    protected[str(absolute(args.contract))] = file_hash(absolute(args.contract))
    return contract, {**old, "source_hashes": contract["source_hashes"], "parent_source_hashes": old["source_hashes"], "protected_hashes": protected,
                      "plan_sha256": contract["plan_sha256"], "study_contract_sha256": file_hash(absolute(args.contract))}


def read_only_cache(panel, path, state):
    path = absolute(path)
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    contract = manifest["contract"]
    expected = {"study": shared.STUDY, "dataset": panel.dataset, "data_sha256": state["data_sha256"],
                "checkpoint_hashes": state["checkpoint_hashes"], "context": 336, "horizon": 48,
                "channels": panel.channels, "target_indices": panel.target_indices.tolist(),
                "stats_sha256": panel.stats_hash, "origins": {key: values.tolist() for key, values in panel.origins.items()},
                "smoke": panel.smoke, "micro_groups": 4, "weight_dtype": "float32", "autocast": "bfloat16",
                "autocast_weight_cache": False, "tf32": False, "dropout": 0.}
    if not manifest.get("completed") or any(contract.get(key) != value for key, value in expected.items()):
        raise AssertionError("The read-only study12 cache does not match the native data/information contract")
    if contract["native_source_hashes"] != state["native_source_hashes"]:
        raise AssertionError("The installed native source differs from the original feature producer")
    files = {str((path / name).resolve()): digest for name, digest in manifest["array_hashes"].items()}
    files[str(path / "manifest.json")] = file_hash(path / "manifest.json")
    shared.verify_hashes(files)
    signature = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    return legacy.FeatureCache(path, manifest), files, signature


def denominators(panel, expected):
    counts = np.isfinite(panel.targets("train")).sum(axis=(0, 2)).astype(np.int64)
    if counts.shape != (2,) or np.any(counts <= 0) or not np.array_equal(counts, np.asarray(expected)):
        raise AssertionError("Full-training target-cell denominators disagree with the frozen contract")
    return counts


def objective(arm, norm, target, loc, scale, quantiles, model, panel, counts):
    from . import losses

    if arm == "NATIVE":
        return native.native_pinball(norm, target, loc, scale, quantiles, model.use_arcsinh)
    return losses.compute_loss(arm, norm, target, loc, scale, quantiles, model.use_arcsinh,
                               target_indices=panel.target_indices, channel_count=panel.count_channels,
                               train_valid_counts=counts, train_origin_count=len(panel.origins["train"]),
                               train_target_scale=panel.fit_std[panel.target_indices])


def calibrate_gradients(model, panel, args, counts, quantiles):
    named = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    parameter_before = shared.parameter_hash(named)
    frozen_before = native.frozen_digest(model)[0]
    cpu_rng, cuda_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
    training_flags = [(module, module.training) for module in model.modules()]
    indices = calibration_indices(len(panel.origins["train"]))
    origins = panel.origins["train"][indices]
    vectors, loss_values, jacobians, crossings = {}, {}, [], []
    model.train()
    for arm in ARMS:
        clear_gradients(model)
        loss_values[arm] = 0.
        for offset in (0, 4):
            legacy.check_resources(args)
            context, target, groups = panel.batch(origins[offset:offset + 4], args.device)
            with legacy.precision(args):
                norm, _, loc, scale = shared.direct(model, context, groups)
                loss = objective(arm, norm, target, loc, scale, quantiles, model, panel, counts)
            if not torch.isfinite(loss):
                raise FloatingPointError("Initial objective loss is not finite")
            (loss / 2).backward()
            loss_values[arm] += float(loss.detach()) / 2
            if arm == "NATIVE":
                z = norm.detach().float().cpu().numpy().reshape(4, panel.count_channels, 21, 48)[:, panel.target_indices]
                s = scale.detach().float().cpu().numpy().reshape(4, panel.count_channels, 1)[:, panel.target_indices]
                jacobian = s[:, :, :, None].astype(np.float64) * (np.cosh(z.astype(np.float64)) if model.use_arcsinh else np.ones_like(z))
                jacobian /= panel.fit_std[panel.target_indices][None, :, None, None]
                valid = np.isfinite(target.detach().cpu().numpy().reshape(4, panel.count_channels, 48)[:, panel.target_indices])
                jacobians.append(jacobian[np.broadcast_to(valid[:, :, None, :], jacobian.shape)])
                crossings.append((np.diff(z, axis=2) < 0).reshape(-1))
        vectors[arm] = gradient_vector(named)
    clear_gradients(model)
    for module, training in training_flags:
        module.training = training
    rng_unchanged = torch.equal(cpu_rng, torch.get_rng_state()) and torch.equal(cuda_rng, torch.cuda.get_rng_state())
    parameters_unchanged = parameter_before == shared.parameter_hash(named) and frozen_before == native.frozen_digest(model)[0]
    if not rng_unchanged or not parameters_unchanged:
        raise AssertionError("Initial gradient calibration changed parameters or RNG state")
    jacobian = np.concatenate(jacobians)
    if not np.isfinite(jacobian).all():
        raise FloatingPointError("Initial raw pullback Jacobian is not finite")
    return {**gradient_summary(vectors), "origin_indices": indices.tolist(), "origins": origins.tolist(),
            "origin_sha256": array_hash(origins), "source_parameter_sha256": parameter_before,
            "source_frozen_sha256": frozen_before, "objective_losses": loss_values,
            "crossing_fraction": float(np.concatenate(crossings).mean()),
            "jacobian_summary": {"count": int(len(jacobian)), "min": float(jacobian.min()),
                                 "median": float(np.median(jacobian)), "p95": float(np.quantile(jacobian, .95)),
                                 "max": float(jacobian.max())},
            "rng_unchanged": True, "parameters_unchanged": True, "gradients_cleared": True,
            "optimizer_steps": 0, "gradient_space": "Concatenated float32 trainable gradients, CPU float64 norm/cosine; before clipping"}


def run(args):
    from . import prepare

    started = time.perf_counter()
    contract = prepare.validate_contract(ROOT, absolute(args.contract))
    dataset = contract["datasets"][args.dataset]
    runtime = runtime_args(args, absolute(dataset["fit_data_path"]))
    contract, state = study_state(args, runtime)
    jobs = contract["smoke_jobs" if args.smoke else "fit_jobs"]
    matches = [job for job in jobs if (job["dataset"], job["arm"]) == (args.dataset, args.arm)]
    if len(matches) != 1 or absolute(matches[0]["output"]) != absolute(args.output):
        raise AssertionError("Requested fit is not the exact frozen dataset/arm/output job")
    output = new_output(args.output)
    args._owns_output = True
    legacy.check_resources(runtime)
    panel = shared.Panel(runtime.data, "fit", args.smoke)
    if panel.dataset != args.dataset or state["data_sha256"] != dataset["fit_data_sha256"]:
        raise AssertionError("The requested dataset does not match its prepared fit archive")
    counts = denominators(panel, dataset["smoke_train_valid_counts" if args.smoke else "train_valid_counts"])
    atomic_json(output / "trial_contract.json", {**state, "study": STUDY, "arm": args.arm,
                "stage": "fit", "requested_arguments": vars(args), "runtime_arguments": vars(runtime),
                "dataset": panel.dataset, "origin_hashes": {key: array_hash(value) for key, value in panel.origins.items()},
                "train_valid_counts": counts.tolist()})
    base = shared.load_base(runtime)
    quantiles = np.asarray(base.chronos_config.quantiles, dtype=np.float64)
    if not np.array_equal(quantiles, panel.quantiles):
        raise AssertionError("Native and prepared quantiles differ")
    audits = shared.information_audit(base, panel, runtime) if args.smoke else {"information_audit": "Same train-only S0 source contract"}
    cache, cache_files, cache_hash = read_only_cache(panel, dataset["smoke_cache" if args.smoke else "cache"], state)
    model = shared.construct(base, "OFF_LORA", runtime.seed, panel.count_channels)
    named = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    initial_hash = shared.parameter_hash(named)
    frozen_before = native.frozen_digest(model)[0]
    identity_errors = []
    with torch.no_grad(), legacy.precision(runtime):
        for subset in (panel.origins["train"][:4], panel.origins["train"][-3:]):
            selected, _ = shared.padded_origins(subset)
            context, _, groups = panel.batch(selected, runtime.device)
            actual, _, _, _ = shared.direct(model, context, groups)
            identity_errors.append(float((actual - cache.batch(selected, runtime.device)[1]).abs().max()))
    if any(not math.isfinite(value) or value > 1e-5 for value in identity_errors):
        raise AssertionError(f"Native fixed-four-group identity failed: {identity_errors}")
    audits.update({"zero_update_identity": True, "zero_update_normalized_max_abs": max(identity_errors),
                   "padded_group_identity_checked": True, "initial_adaptation_sha256": initial_hash,
                   "frozen_before_sha256": frozen_before, "trainable_map_verified": True})
    parameters = [p for _, p in named]
    quantile_tensor = torch.as_tensor(quantiles, dtype=torch.float32, device=runtime.device)
    calibration = calibrate_gradients(model, panel, runtime, counts, quantile_tensor)
    multiplier = calibration["multipliers"][args.arm]
    atomic_json(output / "gradient_calibration.json", calibration)
    optimizer = torch.optim.AdamW(parameters, lr=runtime.lr, weight_decay=0., foreach=False)
    if {id(p) for group in optimizer.param_groups for p in group["params"]} != {id(p) for p in parameters}:
        raise AssertionError("Optimizer registration mismatch")
    samples = np.random.default_rng(runtime.seed).integers(len(panel.origins["train"]), size=(runtime.steps, 8))
    sampler_hash = array_hash(panel.origins["train"][samples])
    val_target = panel.targets("val")
    val_pred = shared.predict(model, panel, "val", runtime)
    val_metrics, _, _ = shared.scores(val_pred, val_target, panel.fit_std[panel.target_indices], quantiles)
    best, best_step = val_metrics["score"], 0
    checkpoint = output / "best_trainable.pt"
    legacy.save_trainable(model, checkpoint)
    history = [{"step": 0, "val_score": best}]
    shared.log(output, "validation", **history[-1])
    durations, gradient_norms, unscaled_losses, scaled_losses = [], [], [], []
    for step in range(1, runtime.steps + 1):
        legacy.check_resources(runtime)
        tick = time.perf_counter()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_loss, unscaled_loss = 0., 0.
        for offset in (0, 4):
            selected = panel.origins["train"][samples[step - 1, offset:offset + 4]]
            context, target, groups = panel.batch(selected, runtime.device)
            with legacy.precision(runtime):
                norm, _, loc, scale = shared.direct(model, context, groups)
                unscaled = objective(args.arm, norm, target, loc, scale, quantile_tensor, model, panel, counts)
                loss = unscaled if args.arm == "NATIVE" else unscaled * multiplier
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite masked objective loss")
            (loss / 2).backward()
            train_loss += float(loss.detach()) / 2
            unscaled_loss += float(unscaled.detach()) / 2
        gradient = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
        gradient_norms.append(float(gradient))
        optimizer.step()
        torch.cuda.synchronize()
        durations.append(time.perf_counter() - tick)
        unscaled_losses.append(unscaled_loss)
        scaled_losses.append(train_loss)
        if step == 1:
            audits["frozen_after_first_update_sha256"] = native.frozen_digest(model)[0]
            if audits["frozen_after_first_update_sha256"] != frozen_before:
                raise AssertionError("Frozen parameters changed after an update")
        if step % runtime.val_every == 0 or step == runtime.steps:
            val_pred = shared.predict(model, panel, "val", runtime)
            val_metrics, _, _ = shared.scores(val_pred, val_target, panel.fit_std[panel.target_indices], quantiles)
            if val_metrics["score"] < best:
                best, best_step = val_metrics["score"], step
                legacy.save_trainable(model, checkpoint)
            history.append({"step": step, "val_score": val_metrics["score"], "train_loss": train_loss,
                            "gradient_norm": float(gradient)})
            shared.log(output, "validation", **history[-1], best_step=best_step)
    before_restore = shared.parameter_hash(named)
    if max(gradient_norms) <= 0 or before_restore == initial_hash:
        raise AssertionError("No finite nonzero gradient/update was observed")
    frozen_after = native.frozen_digest(model)[0]
    if frozen_after != frozen_before:
        raise AssertionError("Frozen parameters changed during training")
    legacy.load_trainable(model, checkpoint, runtime.device)
    restored_hash = shared.parameter_hash(named)
    val_pred = shared.predict(model, panel, "val", runtime)
    val_metrics, loss_sums, valid_counts = shared.scores(val_pred, val_target, panel.fit_std[panel.target_indices], quantiles)
    if not np.isclose(val_metrics["score"], best, rtol=1e-6, atol=1e-8):
        raise AssertionError("Checkpoint restore does not reproduce the selected validation score")
    audits.update({"finite_nonzero_gradient_verified": True, "maximum_gradient_norm": max(gradient_norms),
                   "before_restore_adaptation_sha256": before_restore, "restored_adaptation_sha256": restored_hash,
                   "frozen_after_training_sha256": frozen_after, "frozen_parameters_verified": True,
                   "checkpoint_reload_verified": True, "checkpoint_includes_all_trainable_parameters": True,
                   "frozen_unchanged": True, "train_only_smoke": True if args.smoke else None,
                   "information_isolation": True if args.smoke else None, "gradient_nonzero": True,
                   "optimizer_steps": runtime.steps, "gradient_calibration_verified": True})
    np.savez_compressed(output / "predictions.npz", val_predictions=val_pred, val_target=val_target,
                        val_origins=panel.origins["val"], val_timestamps=panel.timestamps[panel.origins["val"]],
                        val_loss_sums=loss_sums, val_valid_counts=valid_counts, quantiles=quantiles,
                        target_indices=panel.target_indices, target_channels=np.asarray(panel.channels)[panel.target_indices])
    result = {"completed": False, "study": STUDY, "stage": "fit", "arm": args.arm, "method": "OFF_LORA",
              "dataset": panel.dataset, "seed": runtime.seed, "seed_index": 0, "lr": runtime.lr, "smoke": args.smoke,
              "steps_completed": runtime.steps, "best_step": best_step, "val_score": val_metrics["score"],
              "trainable": model.trainable_count, "trainable_parameters": model.trainable_count,
              "trainable_names": model.trainable_names, "module_map": model.module_map, "sampler_sha256": sampler_hash,
              "source_hashes": state["source_hashes"], "native_source_hashes": state["native_source_hashes"],
              "parent_source_hashes": state["parent_source_hashes"],
              "plan_sha256": state["plan_sha256"], "study_contract_sha256": state["study_contract_sha256"],
              "fit_data_sha256": state["data_sha256"], "data_sha256": state["data_sha256"],
              "checkpoint": state["checkpoint"], "native_checkpoint_hashes": state["checkpoint_hashes"],
              "checkpoint_sha256": file_hash(checkpoint), "predictions_sha256": file_hash(output / "predictions.npz"),
              "protected_hashes": state["protected_hashes"], "cache": str(cache.path.resolve()),
              "cache_created_this_trial": False, "cache_generation_seconds": cache.manifest["wall_seconds"],
              "cache_contract_sha256": cache_hash, "cache_array_hashes": cache_files,
              "channels": panel.channels, "target_indices": panel.target_indices.tolist(),
              "fit_mean": panel.fit_mean.tolist(), "fit_std": panel.fit_std.tolist(), "fit_median": panel.fit_median.tolist(),
              "stats_sha256": panel.stats_hash, "origin_counts": {key: len(value) for key, value in panel.origins.items()},
              "origin_hashes": {key: array_hash(value) for key, value in panel.origins.items()},
              "context": 336, "horizon": 48, "micro_groups": 4, "effective_groups": 8,
              "use_arcsinh": bool(model.use_arcsinh),
              "autocast": "bfloat16", "autocast_weight_cache": False, "weight_dtype": "float32", "tf32": False,
              "gradient_clip_norm": 1., "weight_decay": 0., "dropout": 0., "torch_threads": 2, "interop_threads": 1,
              "training_loss": args.arm, "loss_multiplier": multiplier, "train_valid_counts": counts.tolist(),
              "gradient_calibration": calibration, "gradient_calibration_sha256": file_hash(output / "gradient_calibration.json"),
              "gradient_norms": gradient_norms, "clipping_steps": [index + 1 for index, value in enumerate(gradient_norms) if value > 1.],
              "unscaled_training_losses": unscaled_losses, "scaled_training_losses": scaled_losses,
              "selection_metric": "Sorted raw quantiles, observed-cell mean within each target/train std, equal target macro",
              "holdout_file_opened": False, "validation": val_metrics, "validation_history": history, "audits": audits,
              "wall_seconds": time.perf_counter() - started, "optimizer_seconds_total": float(sum(durations)),
              "optimizer_seconds_per_step": float(np.mean(durations)), "peak_cuda_gib": torch.cuda.max_memory_allocated() / 2**30}
    if args.arm == "NATIVE":
        reference = absolute(dataset["smoke_reference_fit" if args.smoke else "native_reference_fit"])
        result["native_replay"] = verify_native_replay(result, output / "predictions.npz", reference)
        atomic_json(output / "native_replay.json", result["native_replay"])
        if not result["native_replay"]["passed"]:
            raise AssertionError(f"Native replay failed: {result['native_replay']['checks']}")
    else:
        result["native_replay"] = {"applicable": False, "reason": "Changed objective; study gate verifies its paired native arm"}
    shared.verify_hashes(state["protected_hashes"])
    shared.verify_hashes(cache_files)
    audits["source_unchanged"] = True
    result["wall_seconds"] = time.perf_counter() - started
    result["completed"] = True
    atomic_json(output / "result.json", result)
    shared.log(output, "fit_completed", arm=args.arm, dataset=panel.dataset, val_score=result["val_score"], best_step=best_step)
    return result


def verify_native_replay(result, predictions_path, reference):
    reference = Path(reference)
    original = json.loads((reference / "result.json").read_text(encoding="utf-8"))
    checks = {
        "initial_adaptation": result["audits"]["initial_adaptation_sha256"] == original["audits"]["initial_adaptation_sha256"],
        "sampler": result["sampler_sha256"] == original["sampler_sha256"],
        "validation_history": result["validation_history"] == original["validation_history"],
        "best_step": result["best_step"] == original["best_step"],
        "val_score": result["val_score"] == original["val_score"],
        "restored_adaptation": result["audits"]["restored_adaptation_sha256"] == original["audits"]["restored_adaptation_sha256"],
    }
    max_error = 0.
    with np.load(predictions_path, allow_pickle=False) as actual, np.load(reference / "predictions.npz", allow_pickle=False) as expected:
        for key in ("val_predictions", "val_target", "val_origins", "quantiles", "val_loss_sums", "val_valid_counts"):
            checks[key] = np.array_equal(actual[key], expected[key], equal_nan=True)
        if actual["val_predictions"].shape == expected["val_predictions"].shape:
            max_error = float(np.max(np.abs(actual["val_predictions"].astype(np.float64) - expected["val_predictions"])))
        else:
            max_error = None
    return {"passed": all(checks.values()), "reference_trial": str(reference.resolve()),
            "reference_result_sha256": file_hash(reference / "result.json"),
            "reference_predictions_sha256": file_hash(reference / "predictions.npz"),
            "checks": checks, "val_predictions_max_abs": max_error,
            "comparison": "Exact tensor hashes, scalar history and saved validation arrays; no serialization equality requirement"}


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--contract", required=True)
    cli.add_argument("--dataset", choices=("bike", "household"), required=True)
    cli.add_argument("--arm", choices=ARMS, required=True)
    cli.add_argument("--output", required=True)
    cli.add_argument("--smoke", action="store_true")
    return cli


def main():
    args = parser().parse_args()
    try:
        run(args)
    except Exception as error:
        output = Path(args.output).resolve()
        if getattr(args, "_owns_output", False) and output.is_relative_to(ROOT / "runs" / STUDY) and output.exists() and not (output / "result.json").exists():
            atomic_json(output / "failure.json", {"completed": False, "error_type": type(error).__name__,
                                                 "error": str(error), "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
