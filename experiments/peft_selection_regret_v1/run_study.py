"""Replay saved PEFT checkpoints with one guarded child and immutable selection."""

import argparse
import math
from pathlib import Path
import sys
import time

from experiments.peft_adaptation_scope_v1.guard import StudyLock, run_guarded
from experiments.peft_external_gap_v1.run_study import read, relative, sha, write
from experiments.peft_selection_regret_v1 import prepare


STUDY = prepare.STUDY


def verify_entry(root, contract, entry, smoke=False):
    cell = contract["cells"][entry["cell_id"]]
    candidate = cell["candidates"][entry["candidate_id"]]
    path = root / entry["path"]
    expected = (root / "runs" / STUDY / "smoke" / cell["cell_id"] / candidate["candidate_id"]
                if smoke else root / candidate["new_forecast_output"])
    if path.resolve() != expected.resolve() or entry["smoke"] is not smoke:
        raise AssertionError("Forecast path or phase changed")
    status = read(path / "guard/status.json")
    if not status.get("completed") or status.get("returncode") != 0 or status.get("reasons"):
        raise RuntimeError(f"Preserve failed or partial guard: {path}")
    result = read(path / "result.json")
    if (not result.get("completed") or result.get("smoke") is not smoke
            or result.get("stage") != "diagnostic_forecast" or result.get("study") != STUDY):
        raise AssertionError("Incomplete diagnostic forecast")
    for key in ("cell_id", "candidate_id", "dataset", "method", "lr", "seed", "best_step"):
        if result[key] != candidate[key]:
            raise AssertionError(f"Diagnostic identity mismatch: {key}")
    if (result["selection_contract_sha256"] != sha(root / "runs" / STUDY / "selection_contract.json")
            or result["fit_result_sha256"] != candidate["fit_result_sha256"]
            or result["fit_checkpoint_sha256"] != candidate["checkpoint_sha256"]
            or result["holdout_data_sha256"] != cell["holdout_data_sha256"]
            or result["restored_adaptation_sha256"] != candidate["restored_adaptation_sha256"]
            or result["predictions_sha256"] != sha(path / "predictions.npz")
            or entry["result_sha256"] != sha(path / "result.json")
            or entry["predictions_sha256"] != result["predictions_sha256"]):
        raise AssertionError("Forecast provenance changed")
    if (result["optimizer_steps"] != 0 or not result["no_training"] or not result["model_unchanged"]
            or result["original_deployment_selection_gate_modified"] is not False):
        raise AssertionError("Diagnostic must preserve weights and original selection")
    for key in ("source_unchanged", "weights_unchanged", "checkpoint_reload_verified",
                "no_optimizer", "model_unchanged", "numerical_path_unchanged"):
        if result["audits"].get(key) is not True:
            raise AssertionError(f"Forecast audit failed: {key}")
    if {split: value["origins"] for split, value in result["splits"].items()} != {"cal": 13, "eval": 83}:
        raise AssertionError("Forecast must cover the complete original C13/E83 panels")
    if smoke and (candidate["candidate_id"] != cell["smoke_candidate"]
                  or result["audits"].get("reference_predictions_equal") is not True):
        raise AssertionError("S0 must reproduce the original selected LoRA")
    if smoke:
        comparison = result.get("reference_comparison") or {}
        expected_keys = {f"{split}_{name}" for split in ("cal", "eval")
                         for name in ("predictions", "unsorted_predictions")}
        if (comparison.get("passed") is not True or comparison.get("normalized_tolerance") != 1e-5
                or set(comparison.get("comparisons", {})) != expected_keys):
            raise AssertionError("S0 requires the complete original prediction comparison")
        for value in comparison["comparisons"].values():
            error = value["normalized_max_abs"]
            if not math.isfinite(error) or error < 0 or error > 1e-5:
                raise AssertionError("S0 reference prediction difference exceeds the frozen tolerance")
    return result


def verify_smoke(root, contract):
    record = read(root / "runs" / STUDY / "smoke_completed.json")
    if (not record.get("completed") or record["selection_contract_sha256"] !=
            sha(root / "runs" / STUDY / "selection_contract.json")):
        raise AssertionError("This immutable contract must pass S0 before new forecasts")
    entries = record["trials"]
    if len(entries) != 4 or {e["cell_id"] for e in entries} != set(contract["cells"]):
        raise AssertionError("S0 requires all four original block/source cells")
    for entry in entries:
        verify_entry(root, contract, entry, smoke=True)
    return entries


def forecast_trial(root, contract, job, smoke, invocation, invocation_path):
    study = root / "runs" / STUDY
    path = root / job["output"]
    candidate = contract["cells"][job["cell_id"]]["candidates"][job["candidate_id"]]
    if (path / "result.json").exists():
        result = read(path / "result.json")
        if not result.get("completed"):
            raise RuntimeError(f"Preserve incomplete forecast before any retry: {path}")
        reused = True
    else:
        if path.exists() and any(path.iterdir()):
            raise RuntimeError(f"Preserve incomplete forecast before any retry: {path}")
        reused = False
        attempt = {"cell_id": job["cell_id"], "candidate_id": job["candidate_id"],
                   "path": relative(root, path), "started_at": time.time(), "completed": False}
        invocation["attempts"].append(attempt)
        write(invocation_path, invocation)
        write(study / "progress.json", {"state": "running", "phase": invocation["phase"],
                                       "current": relative(root, path), "updated_at": time.time()})
        print({"event": "start", "path": relative(root, path)}, flush=True)
        command = [sys.executable, "-m", f"experiments.{STUDY}.forecast", "--contract",
                   str(study / "selection_contract.json"), "--cell", job["cell_id"],
                   "--candidate", job["candidate_id"], "--output", str(path)]
        if smoke:
            command.append("--smoke")
        status = run_guarded(command, path / "guard", root, timeout_seconds=300)
        attempt.update({"finished_at": time.time(), "completed": status.get("completed", False),
                        "guard_status": status, "guard_status_sha256": sha(path / "guard/status.json")})
        write(invocation_path, invocation)
        if not status.get("completed") or status.get("returncode") != 0 or status.get("reasons"):
            raise RuntimeError(f"Forecast stopped; failed attempt preserved: {path}")
        result = read(path / "result.json")
    status = read(path / "guard/status.json")
    entry = {"cell_id": job["cell_id"], "candidate_id": candidate["candidate_id"],
             "path": relative(root, path), "smoke": smoke,
             "result_sha256": sha(path / "result.json"), "predictions_sha256": result["predictions_sha256"],
             "guard_seconds": status["elapsed_seconds"]}
    verify_entry(root, contract, entry, smoke)
    if reused:
        invocation["reused_completed_jobs"].append(entry)
        write(invocation_path, invocation)
    print({"event": "complete", "reused": reused, **entry}, flush=True)
    return entry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    study = root / "runs" / STUDY
    started = time.time()
    with StudyLock(study / ".runner.lock"):
        contract_path = study / "selection_contract.json"
        contract = prepare.validate_contract(root, contract_path)
        if contract.get("missing_optional_sources"):
            raise AssertionError("Freeze the complete implementation before S0")
        phase = "smoke" if args.smoke else "forecast"
        invocation_path = study / "invocations" / f"{time.time_ns()}.json"
        invocation = {"completed": False, "phase": phase, "started_at": started,
                      "selection_contract_sha256": sha(contract_path), "attempts": [], "reused_completed_jobs": []}
        write(invocation_path, invocation)
        try:
            if args.smoke:
                jobs = [{"cell_id": key, "candidate_id": cell["smoke_candidate"],
                         "output": relative(root, study / "smoke" / key / cell["smoke_candidate"])}
                        for key, cell in contract["cells"].items()]
            else:
                verify_smoke(root, contract)
                jobs = contract["unselected_forecast_jobs"]
            entries = []
            for job in jobs:
                entries.append(forecast_trial(root, contract, job, args.smoke, invocation, invocation_path))
                write(study / ("smoke_trials.json" if args.smoke else "forecasts.json"), entries)
            if prepare.validate_contract(root, contract_path) != contract:
                raise AssertionError("Selection contract changed during inference")
            done = {"completed": True, "selection_contract_sha256": sha(contract_path),
                    "invocation_wall_seconds": time.time() - started}
            if args.smoke:
                done.update({"trials": entries, "guard_seconds": sum(e["guard_seconds"] for e in entries)})
                target = study / "smoke_completed.json"
            else:
                done.update({"forecasts": len(entries), "reused_forecasts": len(contract["reuse_forecast_jobs"]),
                             "forecast_guard_seconds": sum(e["guard_seconds"] for e in entries), "training_runs": 0})
                target = study / "completed.json"
            if target.exists():
                existing = read(target)
                if any(existing.get(key) != value for key, value in done.items() if key != "invocation_wall_seconds"):
                    raise AssertionError("Existing completion record differs from verified work")
            else:
                write(target, done)
            if args.smoke:
                verify_smoke(root, contract)
            invocation["completed"] = True
            write(study / "progress.json", {"state": "completed", "phase": phase, "updated_at": time.time()})
        except Exception as error:
            invocation.update({"error_type": type(error).__name__, "error": str(error)})
            write(study / "progress.json", {"state": "failed", "phase": phase, "error": str(error), "updated_at": time.time()})
            raise
        finally:
            invocation.update({"finished_at": time.time(), "wall_seconds": time.time() - started})
            write(invocation_path, invocation)


if __name__ == "__main__":
    main()
