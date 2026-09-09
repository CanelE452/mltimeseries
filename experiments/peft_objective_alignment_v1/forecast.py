"""Forecast the six fixed objective arms only after every fit is frozen."""

import argparse
import json
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from . import prepare
from .train import (ARMS, ROOT, STUDY, absolute, array_hash, atomic_json, file_hash,
                    legacy, native, new_output, runtime_args, shared, study_state, verify_native_replay)


def validate_selection(contract_path, selection_path, fit_trial, dataset, arm):
    contract = prepare.validate_contract(ROOT, absolute(contract_path))
    selection_path, fit_trial = absolute(selection_path), absolute(fit_trial)
    if selection_path != ROOT / "runs" / STUDY / "selection.json":
        raise ValueError("Use the single global selection file")
    selection = prepare.read(selection_path)
    expected = {(name, objective) for name in ("bike", "household") for objective in ARMS}
    entries = selection.get("selected", [])
    if not selection.get("completed") or not selection.get("global_choices_frozen") or selection.get("fit_trial_count") != 6:
        raise AssertionError("All six completed fits must be frozen before holdout access")
    if selection.get("study_contract_sha256") != file_hash(absolute(contract_path)):
        raise AssertionError("Selection and current study contract disagree")
    if len(entries) != 6 or {(entry["dataset"], entry["arm"]) for entry in entries} != expected:
        raise AssertionError("Selection must contain each dataset/objective pair exactly once")
    protected = {str(selection_path): file_hash(selection_path), str(absolute(contract_path)): file_hash(absolute(contract_path))}
    matches = []
    jobs = {(job["dataset"], job["arm"]): absolute(job["output"]) for job in contract["fit_jobs"]}
    for entry in entries:
        path = absolute(entry["path"])
        if path != jobs[(entry["dataset"], entry["arm"])]:
            raise AssertionError("Selected fit is not the canonical frozen job")
        if (entry["method"], entry["seed"], entry["lr"], entry["smoke"]) != ("OFF_LORA", 12000, 3e-5, False):
            raise AssertionError("Selection changed the common training procedure")
        for filename, key in (("result.json", "result_sha256"), ("best_trainable.pt", "checkpoint_sha256"),
                              ("predictions.npz", "predictions_sha256")):
            current = file_hash(path / filename)
            if current != entry[key]:
                raise AssertionError(f"Selected fit artifact changed: {path / filename}")
            protected[str(path / filename)] = current
        meta, guard = prepare.read(path / "result.json"), prepare.read(path / "guard/status.json")
        protected[str(path / "guard/status.json")] = file_hash(path / "guard/status.json")
        if not meta.get("completed") or meta.get("stage") != "fit" or meta.get("study") != STUDY or meta.get("smoke"):
            raise AssertionError("A selected production fit is incomplete or identifies another study")
        if not guard.get("completed") or guard.get("returncode") != 0 or guard.get("reasons"):
            raise AssertionError("A selected fit's guard did not complete successfully")
        if any(meta.get(key) != entry[key] for key in ("dataset", "arm", "method", "seed", "lr", "smoke", "best_step")):
            raise AssertionError("Selection and fit identity differ")
        if meta.get("steps_completed") != 200 or meta.get("holdout_file_opened") is not False:
            raise AssertionError("Fit did not obey the fixed update/information budget")
        if meta.get("source_hashes") != contract["source_hashes"] or meta.get("plan_sha256") != contract["plan_sha256"]:
            raise AssertionError("A selected fit was produced by a different source contract")
        if meta.get("study_contract_sha256") != selection["study_contract_sha256"]:
            raise AssertionError("A selected fit belongs to a different global contract")
        if entry["arm"] == "NATIVE":
            if not meta.get("native_replay", {}).get("passed"):
                raise AssertionError("Both native replays must pass before future predictions")
            replay = verify_native_replay(meta, path / "predictions.npz", absolute(contract["datasets"][entry["dataset"]]["native_reference_fit"]))
            if not replay["passed"]:
                raise AssertionError("Native replay no longer matches the protected original trial")
        shared.verify_hashes(meta["protected_hashes"])
        shared.verify_hashes(meta["cache_array_hashes"])
        if (entry["dataset"], entry["arm"], path) == (dataset, arm, fit_trial):
            matches.append((entry, meta))
    if len(matches) != 1:
        raise AssertionError("Requested forecast does not identify one frozen fit")
    return contract, matches[0][0], matches[0][1], protected


def run(args):
    started = time.perf_counter()
    contract, entry, fit_result, selection_hashes = validate_selection(
        args.contract, args.selection, args.fit_trial, args.dataset, args.arm)
    expected_output = ROOT / "runs" / STUDY / "forecasts" / args.dataset / args.arm
    if absolute(args.output) != expected_output:
        raise AssertionError("Use the canonical dataset/objective forecast output")
    dataset = contract["datasets"][args.dataset]
    runtime = runtime_args(args, absolute(dataset["holdout_data_path"]))
    contract, state = study_state(args, runtime)
    if state["source_hashes"] != fit_result["source_hashes"] or state["checkpoint_hashes"] != fit_result["native_checkpoint_hashes"]:
        raise AssertionError("Forecast source or native weights differ from fitting")
    if state["data_sha256"] != dataset["holdout_data_sha256"]:
        raise AssertionError("Holdout archive hash differs from the fixed data source")
    if absolute(runtime.checkpoint) != absolute(fit_result["checkpoint"]):
        raise AssertionError("Forecast must restore its fit's native revision")
    output = new_output(args.output)
    args._owns_output = True
    protected = {**state["protected_hashes"], **fit_result["protected_hashes"], **selection_hashes}
    atomic_json(output / "trial_contract.json", {**state, "study": STUDY, "stage": "forecast",
                "arm": args.arm, "selected_entry": entry, "protected_hashes": protected})
    # The constructor below is the first access to held-out target arrays.
    panel = shared.Panel(runtime.data, "forecast")
    with np.load(absolute(dataset["fit_data_path"]), allow_pickle=False) as archive:
        fit_manifest = json.loads(archive["manifest_json"].item())
    if fit_manifest["holdout_archive_sha256"] != state["data_sha256"]:
        raise AssertionError("Fit archive and holdout archive are not the original prepared pair")
    if (panel.dataset, panel.channels, panel.target_indices.tolist(), panel.stats_hash) != (
            fit_result["dataset"], fit_result["channels"], fit_result["target_indices"], fit_result["stats_sha256"]):
        raise AssertionError("Held-out channels or training statistics changed")
    base = shared.load_base(runtime)
    quantiles = np.asarray(base.chronos_config.quantiles, dtype=np.float64)
    if not np.array_equal(quantiles, panel.quantiles):
        raise AssertionError("Native and held-out quantile grids differ")
    model = shared.construct(base, "OFF_LORA", 12000, panel.count_channels)
    fit_trial = absolute(args.fit_trial)
    legacy.load_trainable(model, fit_trial / "best_trainable.pt", runtime.device)
    named = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    loaded_hash = shared.parameter_hash(named)
    if loaded_hash != fit_result["audits"]["restored_adaptation_sha256"]:
        raise AssertionError("Forecast adaptive tensors do not match the selected restored checkpoint")
    frozen_before = native.frozen_digest(model)[0]
    arrays = {"quantiles": quantiles, "target_indices": panel.target_indices,
              "target_channels": np.asarray(panel.channels)[panel.target_indices]}
    metadata = {}
    for split in ("cal", "eval"):
        prediction, unsorted = shared.predict(model, panel, split, runtime, return_unsorted=True)
        target = panel.targets(split)
        for name, value in {"predictions": prediction, "unsorted_predictions": unsorted, "target": target,
                            "origins": panel.origins[split], "timestamps": panel.timestamps[panel.origins[split]]}.items():
            arrays[f"{split}_{name}"] = value
        metadata[split] = {"origins": len(panel.origins[split]), "target_shape": list(target.shape),
                           "predictions_shape": list(prediction.shape), "origin_sha256": array_hash(panel.origins[split]),
                           "target_sha256": array_hash(target), "unsorted_crossing": float((np.diff(unsorted, axis=2) < 0).mean())}
    if frozen_before != native.frozen_digest(model)[0] or loaded_hash != shared.parameter_hash(named):
        raise AssertionError("Forecast inference changed model parameters")
    np.savez_compressed(output / "predictions.npz", **arrays)
    shared.verify_hashes(protected)
    shared.verify_hashes(fit_result["cache_array_hashes"])
    result = {"completed": True, "study": STUDY, "stage": "forecast", "smoke": False,
              "dataset": args.dataset, "arm": args.arm, "method": "OFF_LORA", "role": "OFF_LORA",
              "seed": 12000, "lr": 3e-5, "best_step": fit_result["best_step"], "fit_trial": str(fit_trial),
              "fit_result_sha256": entry["result_sha256"], "fit_checkpoint_sha256": entry["checkpoint_sha256"],
              "fit_data_sha256": fit_result["fit_data_sha256"], "holdout_data_sha256": state["data_sha256"],
              "source_hashes": state["source_hashes"], "parent_source_hashes": state["parent_source_hashes"],
              "native_source_hashes": state["native_source_hashes"], "native_checkpoint_hashes": state["checkpoint_hashes"],
              "plan_sha256": state["plan_sha256"], "selection_sha256": file_hash(absolute(args.selection)),
              "study_contract_sha256": state["study_contract_sha256"], "fit_cache": fit_result["cache"],
              "cache_created_this_trial": False, "channels": panel.channels, "target_indices": panel.target_indices.tolist(),
              "fit_std": panel.fit_std.tolist(), "stats_sha256": panel.stats_hash,
              "trainable": model.trainable_count, "trainable_parameters": model.trainable_count,
              "context": 336, "horizon": 48, "micro_groups": 4, "autocast": "bfloat16",
              "use_arcsinh": bool(model.use_arcsinh),
              "autocast_weight_cache": False, "weight_dtype": "float32", "tf32": False,
              "optimizer_steps": 0, "model_unchanged": True, "global_selection_verified_before_holdout_load": True,
              "checkpoint_reload_verified": True, "restored_adaptation_sha256": loaded_hash,
              "frozen_before_sha256": frozen_before, "frozen_after_sha256": native.frozen_digest(model)[0],
              "predictions_sha256": file_hash(output / "predictions.npz"), "splits": metadata,
              "protected_hashes": protected, "wall_seconds": time.perf_counter() - started,
              "peak_cuda_gib": torch.cuda.max_memory_allocated() / 2**30}
    atomic_json(output / "result.json", result)
    shared.log(output, "forecast_completed", dataset=args.dataset, arm=args.arm)
    return result


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--contract", required=True)
    cli.add_argument("--dataset", choices=("bike", "household"), required=True)
    cli.add_argument("--arm", choices=ARMS, required=True)
    cli.add_argument("--output", required=True)
    cli.add_argument("--selection", required=True)
    cli.add_argument("--fit-trial", required=True)
    cli.set_defaults(smoke=False)
    return cli


def main():
    args = parser().parse_args()
    try:
        run(args)
    except Exception as error:
        output = absolute(args.output)
        if getattr(args, "_owns_output", False) and output.is_relative_to(ROOT / "runs" / STUDY) and output.exists() and not (output / "result.json").exists():
            atomic_json(output / "failure.json", {"completed": False, "error_type": type(error).__name__,
                                                 "error": str(error), "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
