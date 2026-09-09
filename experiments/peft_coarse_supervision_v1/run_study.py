"""Serial guarded stages; freeze all choices before sealed evaluation."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
import time

from experiments.peft_adaptation_scope_v1.guard import run_guarded, StudyLock
from . import prepare
from .fetch import ROOT, STUDY, sha, read, write_once


def guard_ok(path):
    status = read(path)
    if not status.get("completed") or status.get("returncode") != 0 or status.get("reasons"):
        raise AssertionError(f"Guard did not complete successfully: {path}")
    return status


def verify_result(contract, stage):
    directory = ROOT / "runs" / STUDY / ("smoke/s0" if stage == "smoke" else stage)
    result = read(directory / "result.json")
    if not result.get("completed") or result.get("contract_sha256") != sha(ROOT / "runs" / STUDY / "study_contract.json"):
        raise AssertionError(f"Incomplete or foreign stage: {stage}")
    guard_ok(ROOT / "runs" / STUDY / "guards" / stage / "status.json")
    names = {"cache": [("cache.npz", "cache_sha256")],
             "ridge": [("head_weights.npz", "head_weights_sha256"), ("coarse_lift_weights.npz", "coarse_weights_sha256")],
             "smoke": [("best_trainable.pt", "best_trainable_sha256"), ("predictions.npz", "predictions_sha256")],
             "train": [("best_trainable.pt", "best_trainable_sha256"), ("predictions.npz", "predictions_sha256")],
             "forecast": [("predictions.npz", "predictions_sha256")]}
    for filename, key in names[stage]:
        if sha(directory / filename) != result[key]:
            raise AssertionError(f"Stage artifact changed: {stage}/{filename}")
    if stage in ("train", "smoke"):
        if bool(result["smoke"]) != (stage == "smoke") or result["steps_completed"] != (3 if stage == "smoke" else 200):
            raise AssertionError("Incorrect fitting procedure")
        prepare.verify(ROOT, result["trial_input_hashes"])
        for flag in ("zero_update_identity", "frozen_unchanged", "fixed_head_unchanged", "checkpoint_reload_verified"):
            if result["audits"][flag] is not True:
                raise AssertionError(f"Training audit failed: {flag}")
    if stage == "ridge" and result["cache_sha256"] != sha(contract["paths"]["cache"]):
        raise AssertionError("Ridge belongs to a different cache")
    return result


def verify_selection(contract):
    selection = read(contract["paths"]["selection"])
    if (not selection.get("completed") or not selection.get("all_choices_frozen_before_evaluation")
            or selection["contract_sha256"] != sha(ROOT / "runs" / STUDY / "study_contract.json")):
        raise AssertionError("Selection is incomplete or foreign")
    prepare.verify(ROOT, selection["input_hashes"])
    ridge, trained = verify_result(contract, "ridge"), verify_result(contract, "train")
    verify_result(contract, "smoke")
    if (selection["simple_policy"] != ridge["simple_policy"] or selection["best_step"] != trained["best_step"]
            or selection["best_trainable_path"] != trained["best_trainable_path"]):
        raise AssertionError("Selection differs from the completed training procedure")
    return selection


def freeze_selection(contract):
    if Path(contract["paths"]["selection"]).exists():
        return verify_selection(contract)
    ridge, trained = verify_result(contract, "ridge"), verify_result(contract, "train")
    verify_result(contract, "smoke")
    paths = [contract["paths"][key] for key in ("cache_result", "cache", "ridge_result", "ridge_head", "ridge_coarse", "train_result")]
    paths += [trained["best_trainable_path"], str(Path(contract["paths"]["train_result"]).parent / "predictions.npz")]
    selection = {"completed": True, "created_at_utc": datetime.now(timezone.utc).isoformat(),
                 "contract_sha256": sha(ROOT / "runs" / STUDY / "study_contract.json"),
                 "all_choices_frozen_before_evaluation": True,
                 "simple_policy": ridge["simple_policy"], "simple_validation_scores": ridge["perVscore"],
                 "selected_head_lambda": ridge["selected_head_lambda"], "selected_coarse_lambda": ridge["selected_coarse_lambda"],
                 "best_step": trained["best_step"], "best_val_loss": trained["best_val_loss"],
                 "best_trainable_path": trained["best_trainable_path"],
                 "input_hashes": {str(Path(path).resolve()): sha(path) for path in paths}}
    write_once(contract["paths"]["selection"], selection)
    return verify_selection(contract)


def run_stage(contract, stage):
    if stage in ("ridge", "smoke", "train"):
        verify_result(contract, "cache")
    if stage in ("smoke", "train"):
        verify_result(contract, "ridge")
    if stage == "train":
        verify_result(contract, "smoke")
    if stage == "forecast":
        freeze_selection(contract)
    study = ROOT / "runs" / STUDY
    output = study / ("smoke/s0" if stage == "smoke" else stage)
    if (output / "result.json").exists():
        return verify_result(contract, stage)
    guard = study / "guards" / stage
    if (output.exists() and any(output.iterdir())) or guard.exists():
        raise FileExistsError(f"Preserve partial stage before recovery: {stage}")
    command = [sys.executable, "-m", f"experiments.{STUDY}.{'train' if stage == 'smoke' else stage}",
               "--contract", str(study / "study_contract.json"), "--output", str(output)]
    if stage == "smoke":
        command.append("--smoke")
    timeout = {"cache": 600, "ridge": 120, "smoke": 120, "train": 900, "forecast": 300}[stage]
    print({"event": "start", "stage": stage, "time_limit_seconds": timeout}, flush=True)
    status = run_guarded(command, guard, ROOT, timeout_seconds=timeout, require_gpu=stage != "ridge")
    if not status.get("completed"):
        raise RuntimeError(f"Stage stopped: {stage}; preserve logs and partial output")
    result = verify_result(contract, stage)
    if prepare.validate() != contract:
        raise AssertionError("Core contract changed during stage")
    print({"event": "complete", "stage": stage, "guard_seconds": status["elapsed_seconds"]}, flush=True)
    return result


def analyze(contract):
    import numpy as np
    from .analysis import analyze_predictions

    selection = verify_selection(contract)
    forecast = verify_result(contract, "forecast")
    if forecast["selection_sha256"] != sha(contract["paths"]["selection"]) or forecast["evaluation_truth_opened"] is not False:
        raise AssertionError("Forecast was not sealed before analysis")
    prediction_path = Path(contract["paths"]["forecast_result"]).parent / "predictions.npz"
    with np.load(prediction_path, allow_pickle=False) as z:
        predicted = {key: z[key].copy() for key in z.files}
    with np.load(contract["data"]["evaluation_truth"]["path"], allow_pickle=False) as z:
        truth = {key: z[key].copy() for key in z.files}
    for key in ("horizon", "site", "target_id", "month"):
        if not np.array_equal(predicted[key], truth[key]):
            raise AssertionError(f"Evaluation identity mismatch: {key}")
    result = analyze_predictions({arm: predicted[arm] for arm in contract["arms"]}, truth["target"],
                               *(predicted[key] for key in ("horizon", "scale", "site", "target_id", "month")),
                               selection["simple_policy"])
    result.update({"completed": True, "contract_sha256": sha(ROOT / "runs" / STUDY / "study_contract.json"),
                   "selection_sha256": sha(contract["paths"]["selection"]),
                   "predictions_sha256": sha(prediction_path), "truth_sha256": contract["data"]["evaluation_truth"]["sha256"]})
    if prepare.validate() != contract:
        raise AssertionError("Contract changed during analysis")
    write_once(ROOT / "results" / STUDY / "metrics.json", result)
    print({"event": "analysis_complete", "decision": result["decision"]}, flush=True)
    return result


def run(stage):
    started = time.time()
    study = ROOT / "runs" / STUDY
    with StudyLock(study / ".runner.lock"):
        contract = prepare.validate()
        stages = ("cache", "ridge", "smoke", "train", "forecast", "analysis") if stage == "all" else (stage,)
        invocation = {"completed": False, "stage": stage, "started_at": started, "stages_completed": []}
        path = study / "invocations" / f"{time.time_ns()}.json"
        try:
            for current in stages:
                if current == "analysis":
                    guard = study / "guards" / "analysis"
                    result_path = ROOT / "results" / STUDY / "metrics.json"
                    if result_path.exists():
                        guard_ok(guard / "status.json")
                        analyze(contract)
                    else:
                        if guard.exists():
                            raise FileExistsError("Preserve partial analysis before recovery")
                        command = [sys.executable, "-m", f"experiments.{STUDY}.run_study", "--analysis-worker"]
                        status = run_guarded(command, guard, ROOT, timeout_seconds=120, require_gpu=False)
                        if not status.get("completed") or not read(result_path).get("completed"):
                            raise RuntimeError("Analysis did not complete")
                else:
                    run_stage(contract, current)
                invocation["stages_completed"].append(current)
            invocation["completed"] = True
        except Exception as error:
            invocation["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            invocation.update({"finished_at": time.time(), "wall_seconds": time.time() - started})
            write_once(path, invocation)
    return invocation


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all", "cache", "ridge", "smoke", "train", "forecast", "analysis"), default="all")
    parser.add_argument("--analysis-worker", action="store_true")
    args = parser.parse_args()
    print(analyze(prepare.validate()) if args.analysis_worker else run(args.stage))
