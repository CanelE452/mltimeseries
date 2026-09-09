"""Inference-only saved-candidate diagnostic; original selection gates stay frozen."""

import argparse
from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import time
import traceback

from experiments.peft_external_gap_v1 import train as original
from experiments.peft_selection_regret_v1 import prepare

import numpy as np
import torch


STUDY = "peft_selection_regret_v1"
PLAN = "_docs/notes/tsfm_topics/14_peft_selection_regret_plan_20260908.md"
PARENTS = {12: "peft_external_gap_v1", 13: "peft_temporal_replication_v1"}
CELLS = tuple(f"s{block}_{dataset}" for block in PARENTS for dataset in ("bike", "household"))
file_hash, atomic_json = original.file_hash, original.atomic_json
_spec = importlib.util.spec_from_file_location(__package__ + "._numerical_train", original.__file__)
_numerical = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_numerical)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def merged_hashes(root, *groups):
    result = {}
    for group in groups:
        for name, value in group.items():
            path = str((Path(root) / name).resolve())
            if path in result and result[path] != value:
                raise AssertionError(f"Conflicting protected hashes: {path}")
            result[path] = value
    return result


def numerical_identity():
    functions = ("load_base", "construct", "direct", "predict", "padded_origins")
    for name in functions:
        left, right = getattr(original, name).__code__, getattr(_numerical, name).__code__
        if left.co_code != right.co_code or left.co_consts != right.co_consts:
            raise AssertionError(f"Diagnostic changed the original numerical function: {name}")
    if _numerical is original or original.STUDY != PARENTS[12]:
        raise AssertionError("Original training module namespace was modified")
    return {name: "unchanged original source function" for name in functions}


@contextmanager
def capture_normalization(records):
    direct = _numerical.direct

    def captured(model, context, groups):
        if torch.is_grad_enabled():
            raise AssertionError("Diagnostic inference must not construct an autograd graph")
        values = direct(model, context, groups)
        norm, _, loc, scale = values
        records.append(tuple(x.detach().float().cpu().numpy().copy() for x in (norm, loc, scale)))
        return values

    _numerical.direct = captured
    try:
        yield
    finally:
        _numerical.direct = direct


def predict_with_normalization(model, panel, split, args):
    records = []
    with capture_normalization(records):
        prediction, unsorted = _numerical.predict(model, panel, split, args, return_unsorted=True)
    count = len(panel.origins[split])
    if len(records) != (count + 3) // 4:
        raise AssertionError("Each inference must use the original four-group direct path")
    captured = []
    for position in range(3):
        blocks = [record[position].reshape(4, panel.count_channels, *record[position].shape[1:]) for record in records]
        captured.append(np.concatenate(blocks)[:count, panel.target_indices])
    if captured[0].shape != prediction.shape or captured[1].shape != (count, 2, 1) or captured[2].shape != (count, 2, 1):
        raise AssertionError("Captured native normalization does not match the target map")
    if not all(np.isfinite(array).all() for array in (prediction, unsorted, *captured)) or np.any(captured[2] <= 0):
        raise FloatingPointError("Native forecast or normalization is nonfinite")
    return prediction, unsorted, *captured


def normalized_difference(left, right, loc, scale, use_arcsinh):
    if left.shape != right.shape or left.ndim != 4 or loc.shape != (*left.shape[:2], 1) or scale.shape != loc.shape:
        raise ValueError("Reference comparison requires matching [N,targets,Q,H] predictions and native loc/scale")
    if not all(np.isfinite(array).all() for array in (left, right, loc, scale)) or np.any(scale <= 0):
        raise ValueError("Reference comparison cannot ignore nonfinite predictions or invalid native scales")
    rows, quantiles, horizon = left.shape[0] * left.shape[1], left.shape[2], left.shape[3]
    values = [torch.as_tensor(np.asarray(x, dtype=np.float32).reshape(rows, quantiles, horizon)) for x in (left, right)]
    location = torch.as_tensor(np.asarray(loc, dtype=np.float32).reshape(rows, 1))
    scaling = torch.as_tensor(np.asarray(scale, dtype=np.float32).reshape(rows, 1))
    normalized = [original.native.raw_to_normalized(x, location, scaling, use_arcsinh) for x in values]
    return float((normalized[0] - normalized[1]).abs().max())


def compare_reference(arrays, reference_path, use_arcsinh, tolerance=1e-5):
    differences = {}
    with np.load(reference_path, allow_pickle=False) as reference:
        for name in ("quantiles", "target_indices", "target_channels"):
            if not np.array_equal(arrays[name], reference[name]):
                raise AssertionError(f"S0 reference {name} differs")
        for split in ("cal", "eval"):
            for name in ("origins", "timestamps", "target"):
                key = f"{split}_{name}"
                equal = np.array_equal(arrays[key], reference[key], equal_nan=True) if name == "target" else np.array_equal(arrays[key], reference[key])
                if not equal:
                    raise AssertionError(f"S0 reference {key} differs")
            for name in ("predictions", "unsorted_predictions"):
                key = f"{split}_{name}"
                error = normalized_difference(arrays[key], reference[key], arrays[f"{split}_loc"], arrays[f"{split}_scale"], use_arcsinh)
                if not np.isfinite(error) or error > tolerance:
                    raise AssertionError(f"S0 {key} normalized difference {error} exceeds {tolerance}")
                differences[key] = {"normalized_max_abs": error, "raw_max_abs": float(np.max(np.abs(arrays[key] - reference[key])))}
    return {"passed": True, "normalized_tolerance": tolerance, "comparisons": differences,
            "scope": "All C13/E83 origins, both target channels, 21 quantiles, all 48 leads; SORT and native unsorted",
            "normalization": "Both raw outputs inverted using the same freshly captured native loc/scale and arcsinh setting"}


def new_output(root, path):
    output = Path(path).resolve()
    if not output.is_relative_to(Path(root).resolve() / "runs" / STUDY):
        raise ValueError("Diagnostic outputs must stay within their own runs namespace")
    if any((output / name).exists() for name in ("result.json", "failure.json", "trial_contract.json", "predictions.npz")):
        raise FileExistsError("Preserve previous complete or partial diagnostic outputs")
    output.mkdir(parents=True, exist_ok=True)
    return output


def authorize_candidate(contract, cell_id, candidate_id, smoke=False):
    if set(contract["cells"]) != set(CELLS):
        raise AssertionError("Diagnostic requires the four pre-fixed original block/source cells")
    cell = contract["cells"][cell_id]
    expected = {(method, lr) for method, rates in original.RATES.items() for lr in rates}
    if not isinstance(cell["candidates"], dict) or any(key != value["candidate_id"] for key, value in cell["candidates"].items()):
        raise AssertionError("Candidates must be a mapping keyed by their canonical identifiers")
    candidates = list(cell["candidates"].values())
    if len(candidates) != 10 or {(c["method"], c["lr"]) for c in candidates} != expected:
        raise AssertionError("The candidate universe must be exactly the declared ten saved checkpoints")
    if any(c["seed"] != 12000 or c["cell_id"] != cell_id or c["dataset"] != cell["dataset"]
           or c["study"] != cell["study"] for c in candidates):
        raise AssertionError("Candidate identity differs from its original block/source/seed")
    if cell["study"] != PARENTS[cell["study_number"]] or cell_id != f"s{cell['study_number']}_{cell['dataset']}":
        raise AssertionError("Cell namespace and original study are inconsistent")
    matches = [c for c in candidates if c["candidate_id"] == candidate_id]
    if len(matches) != 1:
        raise AssertionError("Requested candidate is not uniquely present in the frozen universe")
    candidate = matches[0]
    if smoke:
        if (candidate_id != cell["smoke_candidate"] or candidate_id != cell["selectors"]["LORA_V"] or candidate["method"] != "OFF_LORA"
                or candidate["needs_forecast"] or not candidate["reuse_forecast"]
                or candidate["reuse_forecast"]["role"] != "OFF_LORA"):
            raise AssertionError("S0 replays only the original selected seed0 LoRA in each cell")
    elif not candidate["needs_forecast"] or candidate["reuse_forecast"] is not None:
        raise AssertionError("Production diagnostic generates only the seven non-reused candidates per cell")
    return cell, candidate


def validate_inputs(root, args):
    path = Path(args.contract).resolve()
    if path != root / "runs" / STUDY / "selection_contract.json":
        raise ValueError("Use the frozen selection contract in the new diagnostic namespace")
    contract = prepare.validate_contract(root, path)
    if contract.get("missing_optional_sources"):
        raise AssertionError("Freeze all five diagnostic source files before inference")
    if contract["plan_path"] != PLAN or contract["plan_sha256"] != file_hash(root / PLAN):
        raise AssertionError("Diagnostic plan changed after validation-only selection")
    source_records = contract["source_hashes"]
    source_hashes = {value["path"]: value["sha256"] for key, value in source_records.items() if key != "plan14"}
    for path_required in (Path(__file__).resolve(), Path(prepare.__file__).resolve()):
        if source_hashes.get(path_required.relative_to(root).as_posix()) != file_hash(path_required):
            raise AssertionError("Active diagnostic implementation is outside the frozen source contract")
    cell, candidate = authorize_candidate(contract, args.cell, args.candidate, args.smoke)
    fit_path = (root / candidate["fit_path"]).resolve()
    if not fit_path.is_relative_to(root / "runs" / cell["study"] / "trials"):
        raise ValueError("The diagnostic must restore an original production fit checkpoint")
    checkpoint = (root / candidate["checkpoint_path"]).resolve()
    if checkpoint != fit_path / "best_trainable.pt" or (root / candidate["fit_result_path"]).resolve() != fit_path / "result.json":
        raise AssertionError("Candidate adaptive checkpoint/result path mismatch")
    checks = {str(fit_path / "result.json"): candidate["fit_result_sha256"],
              str(checkpoint): candidate["checkpoint_sha256"],
              str(root / cell["fit_data_path"]): cell["fit_data_sha256"],
              str(root / cell["holdout_data_path"]): cell["holdout_data_sha256"],
              str(path): file_hash(path), str(root / PLAN): contract["plan_sha256"]}
    original.verify_hashes(checks)
    fit = read(fit_path / "result.json")
    guard = read(fit_path / "guard/status.json")
    if (not fit.get("completed") or fit.get("smoke") or fit.get("stage") != "fit" or not guard.get("completed")
            or guard["returncode"] != 0 or guard["reasons"]):
        raise AssertionError("Original fit and its resource guard must have completed successfully")
    if cell["study_number"] == 13 and not fit.get("wrapped_completed"):
        raise AssertionError("Original temporal wrapper completion gate did not pass")
    if any(fit[name] != candidate[name] for name in ("dataset", "method", "lr", "seed", "best_step")):
        raise AssertionError("Candidate identity does not match its frozen fit result")
    if (fit["checkpoint_sha256"] != candidate["checkpoint_sha256"]
            or fit["fit_data_sha256"] != cell["fit_data_sha256"]
            or fit["audits"]["restored_adaptation_sha256"] != candidate["restored_adaptation_sha256"]
            or fit["source_hashes"] != candidate["fit_source_hashes"]
            or fit["native_checkpoint_hashes"] != candidate["native_checkpoint_hashes"]
            or fit["protected_hashes"] != candidate["fit_protected_hashes"]):
        raise AssertionError("Stored candidate native/source/data/adaptive hashes changed")
    numerical_identity()
    producer = Path(original.__file__).resolve()
    if fit["source_hashes"].get(producer.relative_to(root).as_posix()) != file_hash(producer):
        raise AssertionError("Frozen fit does not use this exact numerical producer")
    native_path = Path(fit["checkpoint"]).resolve()
    if native_path.name != original.REVISION:
        raise AssertionError("Native revision differs from both original studies")
    native_hashes = {str(native_path / name): value for name, value in fit["native_checkpoint_hashes"].items()}
    protected = merged_hashes(root, contract["protected_hashes"], source_hashes, checks,
                             fit["protected_hashes"], fit["cache_array_hashes"], native_hashes,
                             {str(fit_path / "guard/status.json"): file_hash(fit_path / "guard/status.json")})
    if candidate["reuse_forecast"]:
        reuse = candidate["reuse_forecast"]
        protected = merged_hashes(root, protected, {reuse[name + "_path"]: reuse[name + "_sha256"] for name in ("result", "predictions")})
        reference = read(root / reuse["result_path"])
        if (not reference.get("completed") or not reference.get("model_unchanged")
                or reference["fit_result_sha256"] != candidate["fit_result_sha256"]
                or reference["fit_checkpoint_sha256"] != candidate["checkpoint_sha256"]
                or reference["predictions_sha256"] != reuse["predictions_sha256"]
                or reference["restored_adaptation_sha256"] != candidate["restored_adaptation_sha256"]):
            raise AssertionError("S0 reference is not the completed original forecast of this checkpoint")
    original.verify_hashes(protected)
    expected_holdout = root / "runs" / cell["study"] / "prepared" / f"{cell['dataset']}_holdout.npz"
    if (root / cell["holdout_data_path"]).resolve() != expected_holdout:
        raise ValueError("Use only the original frozen cell holdout archive")
    with np.load(root / cell["fit_data_path"], allow_pickle=False) as archive:
        manifest = json.loads(archive["manifest_json"].item())
    if manifest["holdout_archive_sha256"] != cell["holdout_data_sha256"]:
        raise AssertionError("Holdout data was not frozen by the original fit preparation")
    args.data, args.checkpoint = str(expected_holdout), str(native_path)
    return contract, cell, candidate, fit, protected, source_hashes


def run(args):
    root = Path(__file__).resolve().parents[2]
    output = new_output(root, args.output)
    started = time.perf_counter()
    atomic_json(output / "result.json", {"completed": False, "stage": "diagnostic_forecast", "cell_id": args.cell,
                                         "candidate_id": args.candidate, "smoke": args.smoke})
    try:
        contract, cell, candidate, fit, protected, source_hashes = validate_inputs(root, args)
        if not args.smoke and output != (root / candidate["new_forecast_output"]).resolve():
            raise ValueError("Production output must match the candidate's pre-fixed diagnostic path")
        if args.smoke and not output.is_relative_to(root / "runs" / STUDY / "smoke"):
            raise ValueError("S0 reference replays require their separate smoke namespace")
        panel = _numerical.Panel(args.data, "forecast")
        if (panel.dataset, panel.channels, panel.target_indices.tolist(), panel.stats_hash) != (
                fit["dataset"], fit["channels"], fit["target_indices"], fit["stats_sha256"]):
            raise AssertionError("Holdout context/targets/train-only statistics differ from original fitting")
        if {key: len(value) for key, value in panel.origins.items()} != {"cal": 13, "eval": 83}:
            raise AssertionError("Use the complete original C13/E83 origin panels")
        atomic_json(output / "trial_contract.json", {"stage": "diagnostic_forecast", "smoke": args.smoke,
                    "cell_id": args.cell, "candidate_id": args.candidate, "candidate": candidate,
                    "selection_contract_path": str(Path(args.contract).resolve()), "selection_contract_sha256": file_hash(args.contract),
                    "source_hashes": source_hashes, "protected_hashes": protected,
                    "scope": "Saved validation-best candidate inference only; no training and no change to original selection gates"})
        base = _numerical.load_base(args)
        quantiles = np.asarray(base.chronos_config.quantiles, dtype=np.float64)
        if not np.array_equal(quantiles, panel.quantiles):
            raise AssertionError("Native and original holdout quantile grids differ")
        model = _numerical.construct(base, candidate["method"], candidate["seed"], panel.count_channels)
        original.legacy.load_trainable(model, root / candidate["checkpoint_path"], args.device)
        adaptive = lambda: original.parameter_hash((n, p) for n, p in model.named_parameters() if p.requires_grad)
        loaded = adaptive()
        if loaded != candidate["restored_adaptation_sha256"]:
            raise AssertionError("Restored adaptive tensors do not match the stored best checkpoint")
        frozen_before = original.native.frozen_digest(model)[0]
        if frozen_before != fit["audits"]["frozen_after_training_sha256"]:
            raise AssertionError("Native frozen weights differ from the original fitted model")
        arrays = {"quantiles": quantiles, "target_indices": panel.target_indices,
                  "target_channels": np.asarray(panel.channels)[panel.target_indices]}
        metadata = {}
        for split in ("cal", "eval"):
            prediction, unsorted, norm, loc, scale = predict_with_normalization(model, panel, split, args)
            target = panel.targets(split)
            for name, value in {"predictions": prediction, "unsorted_predictions": unsorted,
                                "normalized_predictions": norm, "loc": loc, "scale": scale, "target": target,
                                "origins": panel.origins[split], "timestamps": panel.timestamps[panel.origins[split]]}.items():
                arrays[f"{split}_{name}"] = value
            metadata[split] = {"origins": len(panel.origins[split]), "target_shape": list(target.shape),
                               "predictions_shape": list(prediction.shape), "origin_sha256": original.array_hash(panel.origins[split]),
                               "target_sha256": original.array_hash(target), "encoder_calls": (len(target) + 3) // 4}
            original.log(output, "split_inference_completed", split=split, origins=len(target))
        frozen_after, adaptive_after = original.native.frozen_digest(model)[0], adaptive()
        if frozen_before != frozen_after or loaded != adaptive_after or any(p.grad is not None for p in model.parameters()):
            raise AssertionError("Inference changed model parameters or accumulated a gradient")
        comparison = compare_reference(arrays, root / candidate["reuse_forecast"]["predictions_path"], model.use_arcsinh) if args.smoke else None
        numerical_identity()
        np.savez_compressed(output / "predictions.npz", **arrays)
        original.verify_hashes(protected)
        result = {"completed": True, "stage": "diagnostic_forecast", "study": STUDY, "smoke": args.smoke,
                  "cell_id": args.cell, "candidate_id": args.candidate, "original_study": cell["study"],
                  "study_number": cell["study_number"], "dataset": panel.dataset, "method": candidate["method"],
                  "seed": candidate["seed"], "lr": candidate["lr"], "best_step": fit["best_step"],
                  "fit_trial": str((root / candidate["fit_path"]).resolve()), "fit_result_sha256": candidate["fit_result_sha256"],
                  "fit_checkpoint_sha256": candidate["checkpoint_sha256"], "fit_data_sha256": cell["fit_data_sha256"],
                  "holdout_data_sha256": cell["holdout_data_sha256"], "source_hashes": source_hashes,
                  "numerical_producer_path": str(Path(original.__file__).resolve()), "numerical_producer_sha256": file_hash(original.__file__),
                  "fit_source_hashes": fit["source_hashes"], "native_source_hashes": fit["native_source_hashes"],
                  "native_checkpoint": args.checkpoint, "native_checkpoint_hashes": fit["native_checkpoint_hashes"],
                  "plan_sha256": contract["plan_sha256"], "selection_contract_sha256": file_hash(args.contract),
                  "original_selection_sha256": cell["selection_json_sha256"], "original_study_contract_sha256": cell["study_contract_sha256"],
                  "fit_cache": fit["cache"], "cache_created_this_trial": False, "cache_used_for_forecast": False,
                  "channels": panel.channels, "target_indices": panel.target_indices.tolist(), "fit_std": panel.fit_std.tolist(),
                  "stats_sha256": panel.stats_hash, "trainable": model.trainable_count, "trainable_parameters": model.trainable_count,
                  "context": 336, "horizon": 48, "micro_groups": 4, "autocast": "bfloat16", "autocast_weight_cache": False,
                  "use_arcsinh": bool(model.use_arcsinh),
                  "weight_dtype": "float32", "tf32": False, "torch_threads": 2, "interop_threads": 1, "dropout": 0.,
                  "optimizer_steps": 0, "no_training": True, "model_unchanged": True, "checkpoint_reload_verified": True,
                  "restored_adaptation_sha256": loaded, "after_inference_adaptation_sha256": adaptive_after,
                  "frozen_before_sha256": frozen_before, "frozen_after_sha256": frozen_after,
                  "diagnostic_contract_verified_before_holdout_load": True,
                  "original_deployment_selection_gate_modified": False, "diagnostic_only": True,
                  "reference_comparison": comparison, "predictions_sha256": file_hash(output / "predictions.npz"),
                  "splits": metadata, "protected_hashes": protected,
                  "audits": {"source_unchanged": True, "weights_unchanged": True, "checkpoint_reload_verified": True,
                             "no_optimizer": True, "model_unchanged": True, "numerical_path_unchanged": True,
                             "reference_predictions_equal": True if args.smoke else None},
                  "wall_seconds": time.perf_counter() - started, "peak_cuda_gib": torch.cuda.max_memory_allocated() / 2**30}
        atomic_json(output / "result.json", result)
        original.log(output, "diagnostic_forecast_completed", cell_id=args.cell, candidate_id=args.candidate,
                     smoke=args.smoke, seconds=result["wall_seconds"])
        return result
    except Exception as error:
        failed = {"completed": False, "stage": "diagnostic_forecast", "cell_id": args.cell, "candidate_id": args.candidate,
                  "smoke": args.smoke, "error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc()}
        atomic_json(output / "result.json", failed)
        atomic_json(output / "failure.json", failed)
        raise


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--contract", required=True)
    value.add_argument("--cell", choices=CELLS, required=True)
    value.add_argument("--candidate", required=True)
    value.add_argument("--output", required=True)
    value.add_argument("--smoke", action="store_true")
    value.add_argument("--device", default="cuda")
    value.add_argument("--min-free-ram-gib", type=float, default=5.)
    return value


if __name__ == "__main__":
    run(parser().parse_args())
