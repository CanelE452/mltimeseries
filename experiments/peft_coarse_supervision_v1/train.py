"""Monthly-only LoRA fitting with a fixed, previously selected ridge point head."""

import argparse
import math
import os
from pathlib import Path
import time
import traceback

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "2"

import numpy as np
import torch

from experiments.peft_adaptation_scope_v1.modeling import frozen_digest
from experiments.peft_adaptation_scope_v1.train import load_trainable, save_trainable
from experiments.peft_external_gap_v1.train import log
from .cache import (ROOT, STUDY, PADDED_HORIZON, Episodes, array_hash, file_hash, atomic_json,
                    read_json, parameter_hash, validate_stage, fresh_output, native_args,
                    load_base, load_verified_cache, check_resources, precision, validation_eligibility)
from .model import PointModel, coarse_loss


SEED = 2026090817
LR = 3e-5
ACCUMULATION = 4


def macro_score(losses, valid, sites, targets):
    losses, valid, sites, targets = map(np.asarray, (losses, valid, sites, targets))
    if any(value.shape != losses.shape for value in (valid, sites, targets)) or losses.ndim != 1:
        raise ValueError("Macro scoring needs one loss, validity flag, site and target per month")
    valid = valid.astype(bool)
    if not np.isfinite(losses[valid]).all():
        raise FloatingPointError("Nonfinite monthly scoring loss")
    site_scores, target_scores = {}, {}
    for site in np.unique(sites):
        values = []
        for target in np.unique(targets[sites == site]):
            selected = valid & (sites == site) & (targets == target)
            if not selected.any():
                raise ValueError(f"No valid monthly label for {site}/{target}")
            value = float(losses[selected].mean())
            values.append(value)
            target_scores[f"{site}/{target}"] = value
        site_scores[str(site)] = float(np.mean(values))
    return {"score": float(np.mean(list(site_scores.values()))), "site_scores": site_scores,
            "target_scores": target_scores, "valid_month_count": int(valid.sum())}


def sampler(n, steps, accumulation=ACCUMULATION, seed=SEED):
    rng = np.random.default_rng(seed)
    count = steps * accumulation
    sequence = np.concatenate([rng.permutation(n) for _ in range(math.ceil(count / n))])[:count]
    return sequence.reshape(steps, accumulation).astype(np.int64)


def cached_point(hidden, base_point, native_scale, weight, bias, horizon):
    with torch.autocast(device_type=hidden.device.type, enabled=False):
        residual = torch.nn.functional.linear(hidden.float(), weight.float(), bias.float()).flatten(1)
    return (base_point.float() + native_scale.reshape(-1, 1).float() * residual)[:, :horizon]


def gradient_summary(named_parameters):
    norms = {}
    for name, parameter in named_parameters:
        gradient = parameter.grad
        if gradient is not None and not torch.isfinite(gradient).all():
            raise FloatingPointError(f"Nonfinite gradient: {name}")
        norms[name] = 0. if gradient is None else float(gradient.float().norm())
    counts = {kind: sum(value > 0 for name, value in norms.items() if f".lora_{kind}." in name)
              for kind in ("A", "B")}
    if counts["B"] == 0:
        raise AssertionError("No LoRA B received a nonzero monthly gradient")
    return {"parameter_gradient_norms": norms, "nonzero_A_count": counts["A"], "nonzero_B_count": counts["B"]}


def predict(model, panel, indices, args):
    output = np.zeros((len(indices), PADDED_HORIZON), dtype=np.float32)
    with torch.no_grad():
        for row, index in enumerate(indices):
            check_resources(args)
            horizon = int(panel.arrays["horizon"][index])
            with precision(args):
                point = model(panel.context(index, args.device), torch.zeros(1, dtype=torch.long, device=args.device), horizon)
            output[row, :horizon] = point[0].float().cpu().numpy()
    if not np.isfinite(output).all():
        raise FloatingPointError("Nonfinite point forecast")
    return output


def score_predictions(predictions, panel, indices):
    indices = np.asarray(indices, dtype=np.int64)
    horizon = panel.arrays["horizon"][indices]
    means = np.array([predictions[i, :int(h)].astype(np.float64).mean() for i, h in enumerate(horizon)])
    losses = ((means - panel.arrays["total"][indices] / horizon) / panel.arrays["scale"][indices])**2
    score = macro_score(losses, panel.arrays["label_valid"][indices], panel.arrays["site"][indices],
                        panel.arrays["target_id"][indices])
    return score, losses


def verify_input_hashes(hashes):
    for path, expected in hashes.items():
        if file_hash(path) != expected:
            raise AssertionError(f"Training input artifact changed: {path}")


def run(args):
    started = time.perf_counter()
    contract_path, contract = validate_stage(args)
    contract_sha256 = file_hash(contract_path)
    output = Path(args.output).resolve()
    expected = Path(contract["paths"]["train_result"]).resolve().parent
    if (not args.smoke and output != expected) or (args.smoke and not output.is_relative_to(expected.parent / "smoke")):
        raise ValueError("Training output is outside its declared main or smoke directory")
    output = fresh_output(output)
    try:
        train = Episodes(contract["data"]["train"]["path"])
        if len(train) != 176 or not train.arrays["label_valid"].all():
            raise AssertionError("All 176 declared training monthly labels must be available")
        panels = {"train": train}
        if not args.smoke:
            panels["validation"] = Episodes(contract["data"]["validation"]["path"])
            validation_eligibility(panels["validation"])
        cached, cache_result = load_verified_cache(contract, contract_sha256, panels)
        head_path, ridge_result_path = map(Path, (contract["paths"]["ridge_head"], contract["paths"]["ridge_result"]))
        ridge_result = read_json(ridge_result_path)
        if (not ridge_result.get("completed") or ridge_result.get("head_weights_sha256") != file_hash(head_path)
                or ridge_result.get("contract_sha256") != contract_sha256
                or ridge_result.get("cache_sha256") != cache_result["cache_sha256"]):
            raise AssertionError("The fitted reference head is incomplete or its cache changed")
        with np.load(head_path, allow_pickle=False) as archive:
            weight, bias = archive["weight"].copy(), archive["bias"].copy()
        inputs = [Path(contract["data"][key]["path"]) for key in panels]
        inputs += [Path(contract["paths"]["cache"]), Path(contract["paths"]["cache_result"]), head_path, ridge_result_path]
        trial_inputs = {str(path.resolve()): file_hash(path) for path in inputs}
        settings = native_args(contract)
        torch.manual_seed(SEED)
        base = load_base(settings)
        first = int(np.flatnonzero(train.arrays["horizon"] == 744)[0])
        context, groups = train.context(first, settings.device), torch.zeros(1, dtype=torch.long, device=settings.device)
        reference = PointModel(base, "HEAD").load_point_head(weight, bias)
        with torch.no_grad(), precision(settings):
            reference_point = reference(context, groups, 744)
            cache_point = cached_point(torch.as_tensor(cached["train_hidden"][first:first + 1], device=settings.device),
                         torch.as_tensor(cached["train_base_point"][first:first + 1], device=settings.device),
                         torch.as_tensor(cached["train_native_scale"][first:first + 1], device=settings.device),
                         reference.point_head.weight, reference.point_head.bias, 744)
        model = PointModel(base, "ATTN_LORA_FIXED_HEAD", seed=SEED).load_point_head(weight, bias)
        del reference
        with torch.no_grad(), precision(settings):
            initial_point = model(context, groups, 744)
        scale = float(cached["train_native_scale"][first])
        errors = {"frozen_head_vs_lora": float((reference_point - initial_point).abs().max()) / scale,
                  "cache_vs_frozen_head": float((cache_point - reference_point).abs().max()) / scale}
        if max(errors.values()) > 1e-5:
            raise AssertionError(f"Zero-update native-scale identity failed: {errors}")
        del reference_point, cache_point, initial_point
        frozen_before = frozen_digest(model)[0]
        head_before = parameter_hash(model.point_head.named_parameters())
        adaptive = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
        initial_parameter_hashes = {name: parameter_hash([(name, parameter)]) for name, parameter in adaptive}
        initial_all_hash = parameter_hash(adaptive)
        steps = 3 if args.smoke else 200
        selected_panel = train if args.smoke else panels["validation"]
        selected_indices = np.array([first]) if args.smoke else np.arange(len(selected_panel))
        sequence = np.full((steps, ACCUMULATION), first, dtype=np.int64) if args.smoke else sampler(len(train), steps)
        best_path = output / "best_trainable.pt"
        best_step, history, gradient_norms, clipping_steps = 0, [], [], []
        best_predictions = predict(model, selected_panel, selected_indices, settings)
        best_score, _ = score_predictions(best_predictions, selected_panel, selected_indices)
        history.append({"step": 0, "val_loss": best_score["score"], "site_scores": best_score["site_scores"]})
        save_trainable(model, best_path)
        log(output, "validation", **history[-1])
        optimizer = torch.optim.AdamW([parameter for _, parameter in adaptive], lr=LR,
                                      betas=(.9, .999), eps=1e-8, weight_decay=0.)
        optimizer_started = time.perf_counter()
        first_gradient = None
        loss_history = []
        for step, batch in enumerate(sequence, 1):
            check_resources(settings)
            optimizer.zero_grad(set_to_none=True)
            batch_loss = 0.
            for index in batch:
                horizon = int(train.arrays["horizon"][index])
                with precision(settings):
                    point = model(train.context(index, settings.device), groups, horizon)
                    loss = coarse_loss(point, train.arrays["total"][index], horizon, train.arrays["scale"][index])
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite monthly loss at step {step}")
                (loss / ACCUMULATION).backward()
                batch_loss += float(loss.detach()) / ACCUMULATION
            if step == 1:
                first_gradient = gradient_summary(adaptive)
            norm = float(torch.nn.utils.clip_grad_norm_([parameter for _, parameter in adaptive], 1., error_if_nonfinite=True))
            gradient_norms.append(norm)
            if norm > 1.:
                clipping_steps.append(step)
            optimizer.step()
            loss_history.append(batch_loss)
            if step == 1 and frozen_digest(model)[0] != frozen_before:
                raise AssertionError("Frozen pretrained parameters or point head changed at the first update")
            if step == steps or (not args.smoke and step % 40 == 0):
                predictions = predict(model, selected_panel, selected_indices, settings)
                score, _ = score_predictions(predictions, selected_panel, selected_indices)
                history.append({"step": step, "val_loss": score["score"], "site_scores": score["site_scores"]})
                log(output, "validation", **history[-1])
                if score["score"] < best_score["score"]:
                    best_step, best_score, best_predictions = step, score, predictions.copy()
                    save_trainable(model, best_path)
        optimizer_seconds = time.perf_counter() - optimizer_started
        update_status = {name: parameter_hash([(name, value)]) != initial_parameter_hashes[name] for name, value in adaptive}
        if not any(update_status.values()):
            raise AssertionError("No LoRA parameter changed during training")
        if frozen_digest(model)[0] != frozen_before or parameter_hash(model.point_head.named_parameters()) != head_before:
            raise AssertionError("Frozen pretrained parameters or fitted head changed during training")
        load_trainable(model, best_path, settings.device)
        reloaded = predict(model, selected_panel, selected_indices, settings)
        reload_error = float(np.max(np.abs(reloaded - best_predictions)))
        if not np.array_equal(reloaded, best_predictions):
            raise AssertionError(f"Reloaded checkpoint did not reproduce the selected point forecasts: {reload_error}")
        restored_score, losses = score_predictions(reloaded, selected_panel, selected_indices)
        if restored_score != best_score:
            raise AssertionError("Selected validation score changed after checkpoint restoration")
        np.savez(output / "predictions.npz", validation_predictions=reloaded, validation_losses=losses,
                 validation_indices=selected_indices,
                 **{f"validation_{key}": selected_panel.arrays[key][selected_indices]
                    for key in ("horizon", "site", "target_id", "month", "total", "scale", "label_valid")})
        verify_input_hashes(trial_inputs)
        _, after = validate_stage(args)
        if after != contract or file_hash(contract_path) != contract_sha256:
            raise AssertionError("The training contract changed during fitting")
        result = {"completed": True, "study": STUDY, "stage": "train", "smoke": bool(args.smoke),
                  "method": "ATTN_LORA_FIXED_HEAD", "contract_sha256": contract_sha256,
                  "source_hashes": contract["source_hashes"], "trial_input_hashes": trial_inputs,
                  "head_weights_path": str(head_path.resolve()), "head_weights_sha256": file_hash(head_path),
                  "ridge_result_sha256": file_hash(ridge_result_path), "cache_sha256": cache_result["cache_sha256"],
                  "best_step": best_step, "best_val_loss": best_score["score"], "best_val_scores": best_score,
                  "best_trainable_path": str(best_path.resolve()), "best_trainable_sha256": file_hash(best_path),
                  "predictions_sha256": file_hash(output / "predictions.npz"), "trainable_count": model.trainable_count,
                  "seed": SEED, "lr": LR, "steps_completed": steps, "optimizer_steps": steps,
                  "accumulation": ACCUMULATION, "micro_groups": 1, "history": history,
                  "loss_history": loss_history, "gradient_norms": gradient_norms, "clipping_steps": clipping_steps,
                  "first_gradient": first_gradient, "parameter_updated": update_status,
                  "initial_adaptive_sha256": initial_all_hash, "restored_adaptive_sha256": parameter_hash(adaptive),
                  "initial_parameter_hashes": initial_parameter_hashes,
                  "module_map": model.module_map, "module_initialization_seeds": model.module_initialization_seeds,
                  "sampler_sha256": array_hash(sequence), "wall_seconds": time.perf_counter() - started,
                  "optimizer_and_validation_seconds": optimizer_seconds,
                  "peak_cuda_gib": torch.cuda.max_memory_allocated() / 1024**3, "evaluation_data_opened": False,
                  "audits": {"zero_update_identity": True, "zero_update_errors_native_scale": errors,
                             "finite_nonzero_gradient_verified": True, "frozen_unchanged": True,
                             "frozen_parameters_verified": True, "fixed_head_unchanged": True,
                             "frozen_sha256": frozen_before, "fixed_head_parameter_sha256": head_before,
                             "checkpoint_reload_verified": True, "checkpoint_reload_max_abs": reload_error,
                             "source_unchanged": True, "information_isolation": True,
                             "train_only_smoke": bool(args.smoke), "padding_excluded": True}}
        atomic_json(output / "result.json", result)
        return result
    except Exception as error:
        atomic_json(output / "result.json", {"completed": False, "study": STUDY, "stage": "train",
                    "smoke": bool(args.smoke), "contract_sha256": contract_sha256,
                    "error": str(error), "traceback": traceback.format_exc()})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract")
    parser.add_argument("--output", required=True)
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
