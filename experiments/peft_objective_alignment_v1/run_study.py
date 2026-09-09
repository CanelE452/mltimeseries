"""Run six controlled fits and six forecasts under the unchanged shared guard."""

import argparse
from pathlib import Path
import sys
import time

from experiments.peft_adaptation_scope_v1.guard import StudyLock, run_guarded
from experiments.peft_objective_alignment_v1 import prepare


STUDY = prepare.STUDY
read, write, sha, relative = prepare.read, prepare.write, prepare.sha, prepare.relative


def _identity(root, contract, entry, stage, smoke):
    path = root / entry["path"]
    phase = "smoke" if smoke else "trials" if stage == "fit" else "forecasts"
    if entry["dataset"] not in prepare.DATASETS or entry["arm"] not in prepare.ARMS:
        raise AssertionError("Unexpected objective procedure")
    if path.resolve() != root / "runs" / STUDY / phase / entry["dataset"] / entry["arm"]:
        raise AssertionError("Procedure path changed")
    status = read(path / "guard/status.json")
    if not status.get("completed") or status.get("returncode") != 0 or status.get("reasons"):
        raise AssertionError("Every referenced procedure requires a successful guard")
    result = read(path / "result.json")
    expected = {"completed": True, "study": STUDY, "stage": stage, "smoke": smoke,
                "dataset": entry["dataset"], "arm": entry["arm"], "method": "OFF_LORA",
                "seed": contract["settings"]["seed"], "lr": contract["settings"]["smoke_lr" if smoke else "lr"],
                "source_hashes": contract["source_hashes"], "plan_sha256": contract["plan_sha256"],
                "study_contract_sha256": sha(root / "runs" / STUDY / "study_contract.json")}
    if any(result.get(key) != value for key, value in expected.items()):
        raise AssertionError(f"Procedure identity or source contract changed: {path}")
    for key in ("dataset", "arm", "method", "seed", "lr", "smoke", "best_step"):
        if entry.get(key) != result.get(key):
            raise AssertionError(f"Completion entry differs from its procedure: {key}")
    if (entry["result_sha256"] != sha(path / "result.json") or
            entry["predictions_sha256"] != sha(path / "predictions.npz") or
            result["predictions_sha256"] != entry["predictions_sha256"] or
            entry["guard_seconds"] != status["elapsed_seconds"]):
        raise AssertionError("Procedure artifacts changed after completion")
    return path, result


def verify_fit(root, contract, entry, smoke=False):
    path, result = _identity(root, contract, entry, "fit", smoke)
    if (result["steps_completed"] != contract["settings"]["smoke_steps" if smoke else "steps"] or
            result["holdout_file_opened"] is not False or entry["checkpoint_sha256"] != sha(path / "best_trainable.pt") or
            result["checkpoint_sha256"] != entry["checkpoint_sha256"]):
        raise AssertionError("Fit budget, isolation or checkpoint changed")
    for flag in ("zero_update_identity", "checkpoint_reload_verified", "frozen_parameters_verified", "finite_nonzero_gradient_verified"):
        if result["audits"].get(flag) is not True:
            raise AssertionError(f"Fit failed a required audit: {flag}")
    if entry["arm"] == "NATIVE" and (result.get("native_replay") or {}).get("passed") is not True:
        raise AssertionError("Native replay must pass before new future performance is read")
    return result


def verify_forecast(root, contract, entry):
    path, result = _identity(root, contract, entry, "forecast", False)
    if (result.get("optimizer_steps") != 0 or result.get("model_unchanged") is not True or
            result.get("global_selection_verified_before_holdout_load") is not True or
            result.get("checkpoint_reload_verified") is not True):
        raise AssertionError("Forecast must use the frozen globally selected fits")
    selection_path = root / "runs" / STUDY / "selection.json"
    selection = verify_selection(root, contract)
    matches = [e for e in selection["selected"] if (e["dataset"], e["arm"]) == (entry["dataset"], entry["arm"])]
    if len(matches) != 1:
        raise AssertionError("Forecast has no unique selected fit")
    fit = matches[0]
    if (result["selection_sha256"] != sha(selection_path) or result["fit_result_sha256"] != fit["result_sha256"] or
            result["fit_checkpoint_sha256"] != fit["checkpoint_sha256"] or
            entry["fit_checkpoint_sha256"] != fit["checkpoint_sha256"] or entry["best_step"] != fit["best_step"]):
        raise AssertionError("Forecast fit provenance changed")
    return result


def verify_smoke(root, contract):
    done = read(root / "runs" / STUDY / "smoke_completed.json")
    entries = done["trials"]
    expected = {(d, a) for d in prepare.DATASETS for a in prepare.ARMS}
    if (not done.get("completed") or done["study_contract_sha256"] != sha(root / "runs" / STUDY / "study_contract.json") or
            len(entries) != 6 or {(e["dataset"], e["arm"]) for e in entries} != expected):
        raise AssertionError("All six train-only S0 procedures must pass first")
    for entry in entries:
        verify_fit(root, contract, entry, smoke=True)
    return entries


def verify_selection(root, contract):
    selection = read(root / "runs" / STUDY / "selection.json")
    entries = selection["selected"]
    expected = {(d, a) for d in prepare.DATASETS for a in prepare.ARMS}
    if (not selection.get("completed") or selection.get("global_choices_frozen") is not True or
            selection.get("fit_trial_count") != 6 or len(entries) != 6 or
            {(e["dataset"], e["arm"]) for e in entries} != expected or
            selection["study_contract_sha256"] != sha(root / "runs" / STUDY / "study_contract.json")):
        raise AssertionError("All six fits must be frozen before forecast")
    for entry in entries:
        verify_fit(root, contract, entry)
    return selection


def _complete_once(path, value, volatile=()):
    if path.exists():
        previous = read(path)
        if {k: v for k, v in previous.items() if k not in volatile} != {k: v for k, v in value.items() if k not in volatile}:
            raise AssertionError(f"Existing completion record changed: {path}")
    else:
        write(path, value)


def run_trial(root, contract, job, stage, smoke, invocation, invocation_path):
    study = root / "runs" / STUDY
    path = root / job["output"]
    reused = (path / "result.json").is_file()
    if reused:
        if not read(path / "result.json").get("completed"):
            raise RuntimeError(f"Preserve the incomplete procedure: {path}")
    else:
        if path.exists() and any(path.iterdir()):
            raise RuntimeError(f"Preserve the partial procedure before recovery: {path}")
        attempt = {"stage": stage, "smoke": smoke, "dataset": job["dataset"], "arm": job["arm"],
                   "path": relative(root, path), "started_at": time.time(), "completed": False}
        invocation["attempts"].append(attempt)
        write(invocation_path, invocation)
        write(study / "progress.json", {"state": "running", "stage": stage, "smoke": smoke,
                                       "current": relative(root, path), "updated_at": time.time()})
        command = [sys.executable, "-m", f"experiments.{STUDY}.{'train' if stage == 'fit' else 'forecast'}",
                   "--contract", str(study / "study_contract.json"), "--dataset", job["dataset"],
                   "--arm", job["arm"], "--output", str(path)]
        if smoke:
            command.append("--smoke")
        if stage == "forecast":
            command.extend(["--selection", str(study / "selection.json"), "--fit-trial", str(study / "trials" / job["dataset"] / job["arm"])])
        print({"event": "start", "stage": stage, "path": relative(root, path)}, flush=True)
        status = run_guarded(command, path / "guard", root, timeout_seconds=600)
        attempt.update({"finished_at": time.time(), "completed": status.get("completed", False),
                        "guard_status": status, "guard_status_sha256": sha(path / "guard/status.json")})
        write(invocation_path, invocation)
        if not status.get("completed") or status.get("returncode") != 0 or status.get("reasons"):
            raise RuntimeError(f"Procedure stopped; preserve the attempt: {path}")
    result, status = read(path / "result.json"), read(path / "guard/status.json")
    entry = {key: result[key] for key in ("dataset", "arm", "method", "seed", "lr", "smoke", "best_step")}
    entry.update({"path": relative(root, path), "result_sha256": sha(path / "result.json"),
                  "predictions_sha256": sha(path / "predictions.npz"), "guard_seconds": status["elapsed_seconds"]})
    key = "checkpoint_sha256" if stage == "fit" else "fit_checkpoint_sha256"
    entry[key] = result[key]
    if stage == "fit":
        verify_fit(root, contract, entry, smoke)
    else:
        verify_forecast(root, contract, entry)
    if reused:
        invocation["reused_completed_jobs"].append({"stage": stage, **entry})
        write(invocation_path, invocation)
    print({"event": "complete", "stage": stage, "reused": reused, "path": entry["path"]}, flush=True)
    return entry


def run(root, smoke=False):
    root = Path(root).resolve()
    study = root / "runs" / STUDY
    started = time.time()
    with StudyLock(study / ".runner.lock"):
        contract = prepare.validate_contract(root)
        digest = sha(study / "study_contract.json")
        invocation_path = study / "invocations" / f"{time.time_ns()}.json"
        invocation = {"completed": False, "phase": "smoke" if smoke else "production", "started_at": started,
                      "study_contract_sha256": digest, "attempts": [], "reused_completed_jobs": []}
        write(invocation_path, invocation)
        try:
            if not smoke:
                verify_smoke(root, contract)
            fits = []
            for job in contract["smoke_jobs" if smoke else "fit_jobs"]:
                fits.append(run_trial(root, contract, job, "fit", smoke, invocation, invocation_path))
                write(study / ("smoke_trials.json" if smoke else "trials.json"), fits)
            if smoke:
                done = {"completed": True, "study_contract_sha256": digest, "trials": fits,
                        "invocation_wall_seconds": time.time() - started, "guard_seconds": sum(e["guard_seconds"] for e in fits)}
                _complete_once(study / "smoke_completed.json", done, ("invocation_wall_seconds",))
                verify_smoke(root, contract)
            else:
                selection = {"completed": True, "global_choices_frozen": True, "fit_trial_count": 6,
                             "study_contract_sha256": digest, "selected": fits}
                _complete_once(study / "selection.json", selection)
                verify_selection(root, contract)
                forecasts = []
                for fit in fits:
                    job = {"dataset": fit["dataset"], "arm": fit["arm"],
                           "output": f"runs/{STUDY}/forecasts/{fit['dataset']}/{fit['arm']}"}
                    forecasts.append(run_trial(root, contract, job, "forecast", False, invocation, invocation_path))
                    write(study / "forecasts.json", forecasts)
                done = {"completed": True, "study_contract_sha256": digest, "fits": len(fits), "forecasts": len(forecasts),
                        "reused_forecasts": 2, "invocation_wall_seconds": time.time() - started,
                        "fit_guard_seconds": sum(e["guard_seconds"] for e in fits),
                        "forecast_guard_seconds": sum(e["guard_seconds"] for e in forecasts)}
                _complete_once(study / "completed.json", done, ("invocation_wall_seconds",))
            if prepare.validate_contract(root) != contract:
                raise AssertionError("Contract changed during the run")
            invocation["completed"] = True
            write(study / "progress.json", {"state": "completed", "phase": invocation["phase"], "updated_at": time.time()})
            return done
        except Exception as error:
            invocation.update({"error_type": type(error).__name__, "error": str(error)})
            write(study / "progress.json", {"state": "failed", "phase": invocation["phase"], "error": str(error), "updated_at": time.time()})
            raise
        finally:
            invocation.update({"finished_at": time.time(), "wall_seconds": time.time() - started})
            write(invocation_path, invocation)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    print(run(Path(__file__).resolve().parents[2], args.smoke))
