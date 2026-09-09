"""Serial guarded execution, audited reuse, and validation-only global selection."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

from experiments.peft_adaptation_scope_v1.guard import StudyLock, run_guarded
from .contract import ROOT, STUDY, digest, read_contract, read_json, relative, save_json

STUDY_ROOT = ROOT/"runs"/STUDY
CONTRACT = STUDY_ROOT/"study_contract.json"


def progress(**fields):
    value = {"updated_at_utc": datetime.now(timezone.utc).isoformat(), **fields}
    temporary = STUDY_ROOT/"progress.json.tmp"
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    os.replace(temporary, STUDY_ROOT/"progress.json")
    print(json.dumps(value), flush=True)


def job_name(arm, lr, seed):
    return f"{arm}/lr_{lr:g}_seed_{seed}"


def require_guard(path):
    status = read_json(path/"status.json")
    if not status.get("completed") or status.get("returncode") != 0 or status.get("reasons"):
        raise AssertionError(f"Guard did not complete successfully: {path}")
    return status


def verify_fit(path, guard, contract, dataset, arm, lr, seed, smoke):
    result = read_json(path/"result.json")
    expected = {"completed": True, "stage": "fit", "study": STUDY, "dataset": dataset,
                "arm": arm, "lr": lr, "seed": seed, "smoke": smoke,
                "contract_sha256": digest(CONTRACT), "holdout_file_opened": False,
                "steps_completed": contract["settings"]["smoke_steps" if smoke else "steps"]}
    if any(result.get(key) != value for key, value in expected.items()):
        raise AssertionError(f"Fit identity/budget differs: {path}")
    for field, filename in (("checkpoint_sha256", "best_trainable.pt"), ("Vpredictions_sha256", "Vpredictions.npz")):
        if result[field] != digest(path/filename):
            raise AssertionError(f"Fit artifact differs: {path/filename}")
    for flag in ("zero_update_identity", "trainable_map_verified", "optimizer_exact_parameter_set",
                 "finite_nonzero_gradient_verified", "checkpoint_reload_verified"):
        if result["audits"].get(flag) is not True:
            raise AssertionError(f"Missing required fit audit: {flag}")
    if arm == "FULL_FT":
        if result["audits"].get("full_model_update_scope_verified") is not True:
            raise AssertionError("Full FT scope not verified")
    elif result["audits"].get("frozen_parameters_verified") is not True:
        raise AssertionError("Frozen backbone was not preserved")
    status = require_guard(guard)
    return {"dataset": dataset, "arm": arm, "lr": lr, "seed": seed, "val_score": result["val_score"],
            "best_step": result["best_step"], "fit_dir": relative(path), "fit_result_sha256": digest(path/"result.json"),
            "checkpoint_sha256": result["checkpoint_sha256"], "guard_path": relative(guard),
            "guard_sha256": digest(guard/"status.json"), "guard_seconds": status["elapsed_seconds"]}


def launch(command, guard, *, gpu=True, timeout=1200):
    if guard.exists():
        raise FileExistsError(f"Preserve previous guard attempt: {guard}")
    status = run_guarded(command, guard, ROOT, timeout_seconds=timeout, require_gpu=gpu)
    if not status.get("completed"):
        raise RuntimeError(f"Child stopped; preserved logs at {guard}")


def fit_job(contract, dataset, arm, lr, seed, smoke):
    phase = "smoke" if smoke else "trials"
    suffix = Path(phase)/dataset/job_name(arm, lr, seed)
    path, guard = STUDY_ROOT/suffix, STUDY_ROOT/"guards"/suffix
    if not (path/"result.json").exists():
        if path.exists() and any(path.iterdir()):
            raise FileExistsError(f"Preserve partial fit: {path}")
        progress(state="running", stage=phase, current=relative(path))
        command = [sys.executable, "-m", f"experiments.{STUDY}.train", "--contract", str(CONTRACT),
                   "--dataset", dataset, "--arm", arm, "--lr", str(lr), "--seed", str(seed), "--output", str(path)]
        if smoke:
            command.append("--smoke")
        launch(command, guard, timeout=600 if smoke else 1200)
    return verify_fit(path, guard, contract, dataset, arm, lr, seed, smoke)


def select_trials(entries, datasets, grids, seeds):
    expected = {(d, a, lr, s) for d in datasets for a, rates in grids.items() for lr in rates for s in seeds}
    identities = [(e["dataset"], e["arm"], e["lr"], e["seed"]) for e in entries]
    if len(identities) != len(expected) or set(identities) != expected:
        raise AssertionError("Selection requires exactly the full registered fit grid")
    selected, choices = [], []
    for dataset in datasets:
        for arm, rates in grids.items():
            candidates = []
            for lr in rates:
                matching = [e for e in entries if (e["dataset"],e["arm"],e["lr"]) == (dataset,arm,lr)]
                mean = sum(e["val_score"] for e in matching)/len(seeds)
                candidates.append({"lr": lr, "mean_val_score": mean, "seed_scores": [e["val_score"] for e in matching]})
            best = min(candidates, key=lambda item: (item["mean_val_score"], rates.index(item["lr"])))
            selected.extend(e for e in entries if (e["dataset"],e["arm"],e["lr"]) == (dataset,arm,best["lr"]))
            choices.append({"dataset":dataset,"arm":arm,"selected_lr":best["lr"],"candidates":candidates,
                            "grid_boundary_selected":best["lr"] in (rates[0],rates[-1])})
    return selected, choices


def freeze_selection(contract, entries):
    selected, choices = select_trials(entries, contract["datasets"], contract["settings"]["lr_grids"], contract["settings"]["seeds"])
    baseline_dir = STUDY_ROOT/"baselines"
    require_guard(STUDY_ROOT/"guards/baseline_fit")
    baselines = {}
    for dataset in contract["datasets"]:
        folder = baseline_dir/dataset
        result, choice = read_json(folder/"result.json"), read_json(folder/"selection.json")
        if (not result["completed"] or not choice["completed"] or result["contract_sha256"] != digest(CONTRACT)
                or choice["model_sha256"] != digest(folder/"model.npz")
                or result["selection_sha256"] != digest(folder/"selection.json")
                or choice["V_predictions_sha256"] != digest(folder/"V_predictions.npz")):
            raise AssertionError("Simple baseline selection is incomplete or changed")
        baselines[dataset] = {"model_sha256": digest(folder/"model.npz"), "selection_sha256": digest(folder/"selection.json")}
    for dataset in contract["datasets"]:
        results = [read_json(ROOT/e["fit_dir"]/"result.json") for e in entries if e["dataset"] == dataset]
        if len({r["step0_predictions_sha256"] for r in results}) != 1:
            raise AssertionError("Registered neural trials have different initial validation predictions")
        for seed in contract["settings"]["seeds"]:
            if len({r["sampler_sha256"] for r in results if r["seed"] == seed}) != 1:
                raise AssertionError("Matched seeds do not share training origin schedules")
    value = {"completed": True, "global_choices_frozen": True, "contract_sha256": digest(CONTRACT),
             "fit_trial_count": len(entries), "selected": selected, "choices": choices, "all_trials": entries,
             "baselines": baselines, "C_or_E_used_for_selection": False}
    path = STUDY_ROOT/"selection.json"
    if path.exists():
        if read_json(path) != value:
            raise AssertionError("Previously frozen selection differs")
    else:
        save_json(path, value)
    return value


def forecast_job(contract, dataset, arm, fit):
    label = "F0" if fit is None else f"{arm}_seed_{fit['seed']}"
    suffix = Path("forecasts")/dataset/label
    path, guard = STUDY_ROOT/suffix, STUDY_ROOT/"guards"/suffix
    if not (path/"result.json").exists():
        progress(state="running", stage="forecast", current=relative(path))
        command = [sys.executable,"-m",f"experiments.{STUDY}.forecast","--contract",str(CONTRACT),
                   "--dataset",dataset,"--arm",arm,"--output",str(path)]
        if fit is not None:
            command.extend(["--fit-dir", str(ROOT/fit["fit_dir"])])
        launch(command,guard,timeout=600)
    status = require_guard(guard)
    result = read_json(path/"result.json")
    expected = {"study":STUDY,"stage":"forecast","dataset":dataset,"arm":arm,
                "seed":contract["settings"]["seeds"][0] if fit is None else fit["seed"],
                "fit_result_sha256":None if fit is None else fit["fit_result_sha256"],
                "checkpoint_sha256":None if fit is None else fit["checkpoint_sha256"]}
    if (any(result.get(key) != value for key,value in expected.items())
            or not result["completed"] or result["contract_sha256"] != digest(CONTRACT)
            or result["selection_sha256"] != digest(STUDY_ROOT/"selection.json")
            or result["optimizer_steps"] != 0 or not result["audits"]["model_unchanged"]):
        raise AssertionError("Forecast provenance/update audit differs")
    for letter in ("C","E"):
        if result[f"{letter}predictions_sha256"] != digest(path/f"{letter}predictions.npz"):
            raise AssertionError("Forecast artifact hash differs")
    return {"dataset":dataset,"arm":arm,"seed":result["seed"],"path":relative(path),
            "result_sha256":digest(path/"result.json"),"guard_path":relative(guard),"guard_seconds":status["elapsed_seconds"]}


def run(smoke):
    started = time.perf_counter()
    with StudyLock(STUDY_ROOT/".runner.lock"):
        contract = read_contract(CONTRACT)
        settings = contract["settings"]
        if not smoke:
            done = read_json(STUDY_ROOT/"smoke_completed.json")
            if not done["completed"] or done["contract_sha256"] != digest(CONTRACT) or len(done["trials"]) != 6:
                raise AssertionError("All six S0 trials must pass before production")
            for e in done["trials"]:
                verify_fit(ROOT/e["fit_dir"],ROOT/e["guard_path"],contract,e["dataset"],e["arm"],e["lr"],e["seed"],True)
        entries = []
        for dataset in contract["datasets"]:
            arms = ("FULL_FT","HEAD_ONLY","LORA") if smoke else settings["lr_grids"]
            for arm in arms:
                for lr in ([settings["smoke_lr"][arm]] if smoke else settings["lr_grids"][arm]):
                    for seed in (settings["seeds"][:1] if smoke else settings["seeds"]):
                        entries.append(fit_job(contract,dataset,arm,lr,seed,smoke))
                        progress(state="running",stage="smoke" if smoke else "fit",completed_fits=len(entries),total_fits=6 if smoke else 54)
        if smoke:
            path = STUDY_ROOT/"smoke_completed.json"
            value = {"completed":True,"contract_sha256":digest(CONTRACT),"trials":entries}
            if path.exists():
                if read_json(path) != value:
                    raise AssertionError("Existing S0 completion differs")
            else:
                save_json(path,value)
        else:
            if not (STUDY_ROOT/"baselines/fit_summary.json").exists():
                launch([sys.executable,"-m",f"experiments.{STUDY}.baselines","--contract",str(CONTRACT),
                        "--stage","fit","--output",str(STUDY_ROOT/"baselines")],STUDY_ROOT/"guards/baseline_fit",gpu=False,timeout=300)
            selection = freeze_selection(contract,entries)
            analysis_manifest = read_json(STUDY_ROOT/"analysis_contract.json")
            from .contract import verify_hashes
            verify_hashes(analysis_manifest["source_hashes"])
            if analysis_manifest["contract_sha256"] != digest(CONTRACT):
                raise AssertionError("Freeze analysis source before any holdout forecast")
            forecasts = []
            for dataset in contract["datasets"]:
                forecasts.append(forecast_job(contract,dataset,"F0",None))
                for entry in selection["selected"]:
                    if entry["dataset"] == dataset:
                        forecasts.append(forecast_job(contract,dataset,entry["arm"],entry))
            if not all((STUDY_ROOT/"baselines"/d/"forecast_result.json").exists() for d in contract["datasets"]):
                launch([sys.executable,"-m",f"experiments.{STUDY}.baselines","--contract",str(CONTRACT),
                        "--stage","forecast","--output",str(STUDY_ROOT/"baselines")],STUDY_ROOT/"guards/baseline_forecast",gpu=False,timeout=300)
            require_guard(STUDY_ROOT/"guards/baseline_forecast")
            path = STUDY_ROOT/"completed.json"
            if not path.exists():
                save_json(path,{"completed":True,"contract_sha256":digest(CONTRACT),"selection_sha256":digest(STUDY_ROOT/"selection.json"),"forecasts":forecasts})
        progress(state="completed",stage="smoke" if smoke else "production",invocation_seconds=time.perf_counter()-started)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke",action="store_true")
    args = parser.parse_args()
    try:
        run(args.smoke)
    except Exception as error:
        progress(state="failed",error=f"{type(error).__name__}: {error}")
        raise
