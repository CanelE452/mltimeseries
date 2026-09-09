"""Freeze choices before held-out forecasts; run one guarded GPU child at a time."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

from experiments.peft_adaptation_scope_v1.guard import StudyLock, run_guarded
from experiments.peft_trainlag_v1.run_study import contract as previous_contract


STUDY = "peft_external_gap_v1"
PLAN = "_docs/notes/tsfm_topics/12_peft_external_gap_plan_20260908.md"
DATASETS = ("bike", "household")
GRIDS = {"H_MLP": (1e-4, 3e-4, 1e-3), "H_FULL": (3e-5, 1e-4, 3e-4),
         "OFF_LORA": (1e-5, 3e-5, 1e-4)}
SEEDS = (12000, 12001, 12002)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def relative(root, path):
    return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")


def contract(root):
    previous = previous_contract(root)
    if previous != read(root / "runs/peft_trainlag_v1/study_contract.json"):
        raise AssertionError("Previous study contract changed")
    sources = dict(previous["sources"])
    for name in ("data.py", "raw.py", "train.py", "run_study.py"):
        path = root / "experiments" / STUDY / name
        sources[relative(root, path)] = sha(path)
    prepared = root / "runs" / STUDY / "prepared"
    required = [prepared / f"{dataset}_{part}.npz" for dataset in DATASETS for part in ("fit", "holdout")]
    if not all(path.is_file() for path in required):
        raise RuntimeError("Both datasets must finish preparation before S0")
    manifest = read(prepared / "manifest.json")
    expected_files = {f"{dataset}_{part}" for dataset in DATASETS for part in ("fit", "holdout")}
    if manifest.get("all_qc_passed") is not True or manifest["datasets"] != list(DATASETS) or set(manifest["files"]) != expected_files:
        raise AssertionError("Both fixed dataset QC gates must pass")
    for name, record in manifest["files"].items():
        if sha(prepared / f"{name}.npz") != record["sha256"]:
            raise AssertionError(f"Prepared manifest hash changed: {name}")
    artifacts = sorted(set(required + list(prepared.glob("*.json"))))
    return {"sources": sources, "plan_sha256": sha(root / PLAN),
            "previous_contract_sha256": sha(root / "runs/peft_trainlag_v1/study_contract.json"),
            "data_files": {relative(root, p): sha(p) for p in artifacts},
            "settings": {"datasets": list(DATASETS), "grids": {k: list(v) for k, v in GRIDS.items()},
                         "seeds": list(SEEDS), "steps": 200, "val_every": 40,
                         "expected_fit_trials": 28, "expected_adaptation_fits": 26,
                         "expected_selected": 14, "micro_groups": 4}}


def check_guard(path):
    value = read(path / "guard/status.json")
    if not value.get("completed") or value.get("returncode") != 0 or value.get("reasons"):
        raise RuntimeError(f"Preserve failed or partial guard: {path}")
    return value


def fit_trial(root, dataset, method, lr, seed, smoke=False):
    study = root / "runs" / STUDY
    path = study / ("smoke" if smoke else "trials") / dataset / method / f"lr_{lr:.0e}_seed_{seed}"
    data = study / "prepared" / f"{dataset}_fit.npz"
    steps = 0 if method == "F0" else (5 if smoke else 200)
    val_every = 5 if smoke else 40
    if (path / "result.json").exists():
        result, guard = read(path / "result.json"), check_guard(path)
    else:
        if path.exists() and any(path.iterdir()):
            raise RuntimeError(f"Preserve incomplete trial before retry: {path}")
        command = [sys.executable, "-m", f"experiments.{STUDY}.train", "fit", "--data", str(data),
                   "--output", str(path), "--cache", str(study / ("smoke_cache" if smoke else "cache") / dataset),
                   "--method", method, "--lr", str(lr), "--seed", str(seed),
                   "--steps", str(steps), "--val-every", str(val_every)]
        if smoke:
            command.append("--smoke")
        write(study / "progress.json", {"state": "running", "phase": "smoke" if smoke else "fit",
                                       "current": relative(root, path), "updated_at": time.time()})
        print(json.dumps({"event": "start", "path": relative(root, path)}), flush=True)
        guard = run_guarded(command, path / "guard", root, timeout_seconds=600)
        if not guard.get("completed"):
            raise RuntimeError(f"Trial stopped: {path}")
        result = read(path / "result.json")
    if not result.get("completed") or (result["dataset"], result["method"], result["lr"], result["seed"]) != (
            dataset, method, lr, seed):
        raise AssertionError("Incomplete or mismatched fit result")
    if result["fit_data_sha256"] != sha(data) or result["checkpoint_sha256"] != sha(path / "best_trainable.pt"):
        raise AssertionError("Fit data or checkpoint changed")
    entry = {"dataset": dataset, "method": method, "seed": seed, "lr": lr, "path": relative(root, path),
             "val_score": result["val_score"], "guard_seconds": guard["elapsed_seconds"],
             "result_sha256": sha(path / "result.json"), "checkpoint_sha256": result["checkpoint_sha256"]}
    print(json.dumps({"event": "complete", **entry}), flush=True)
    return entry


def verify_smoke(root, entries):
    expected = {(d, m) for d in DATASETS for m in ("F0", *GRIDS)}
    if len(entries) != 8 or {(e["dataset"], e["method"]) for e in entries} != expected:
        raise AssertionError("S0 requires two datasets and all four methods")
    for entry in entries:
        path = root / entry["path"]
        check_guard(path)
        result = read(path / "result.json")
        if not result["completed"] or sha(path / "result.json") != entry["result_sha256"]:
            raise AssertionError("S0 result changed")
        audit = result["audits"]
        for key in ("zero_update_identity", "frozen_unchanged", "checkpoint_reload_verified", "source_unchanged",
                    "train_only_smoke", "information_isolation"):
            if audit.get(key) is not True:
                raise AssertionError(f"S0 audit failed: {key}, {entry['path']}")
        if audit["zero_update_normalized_max_abs"] > 1e-5:
            raise AssertionError("S0 cache/direct zero identity mismatch")
        if entry["method"] != "F0" and (not audit.get("gradient_nonzero") or audit["optimizer_steps"] != 5):
            raise AssertionError("S0 adaptation did not train")


def forecast_trial(root, entry, selection_path):
    study = root / "runs" / STUDY
    path = study / "forecasts" / entry["dataset"] / f"{entry['role']}_seed_{entry['seed']}"
    if (path / "result.json").exists():
        result, guard = read(path / "result.json"), check_guard(path)
    else:
        if path.exists() and any(path.iterdir()):
            raise RuntimeError(f"Preserve incomplete forecast: {path}")
        command = [sys.executable, "-m", f"experiments.{STUDY}.train", "forecast", "--data",
                   str(study / "prepared" / f"{entry['dataset']}_holdout.npz"), "--fit-trial", str(root / entry["path"]),
                   "--selection", str(selection_path), "--output", str(path)]
        write(study / "progress.json", {"state": "running", "phase": "forecast",
                                       "current": relative(root, path), "updated_at": time.time()})
        print(json.dumps({"event": "forecast_start", "path": relative(root, path)}), flush=True)
        guard = run_guarded(command, path / "guard", root, timeout_seconds=300)
        if not guard.get("completed"):
            raise RuntimeError(f"Forecast stopped: {path}")
        result = read(path / "result.json")
    if not result.get("completed"):
        raise AssertionError("Incomplete forecast")
    if (result["dataset"], result["method"], result["role"], result["seed"], result["lr"]) != (
            entry["dataset"], entry["method"], entry["role"], entry["seed"], entry["lr"]):
        raise AssertionError("Forecast identity changed")
    if result["selection_sha256"] != sha(selection_path) or result["fit_result_sha256"] != entry["result_sha256"]:
        raise AssertionError("Forecast selection or fitting result changed")
    if result["fit_checkpoint_sha256"] != entry["checkpoint_sha256"] or result["predictions_sha256"] != sha(path / "predictions.npz"):
        raise AssertionError("Forecast checkpoint or predictions changed")
    if result["holdout_data_sha256"] != sha(study / "prepared" / f"{entry['dataset']}_holdout.npz"):
        raise AssertionError("Forecast used another holdout")
    return {**entry, "fit_path": entry["path"], "fit_result_sha256": entry["result_sha256"],
            "path": relative(root, path), "result_sha256": sha(path / "result.json"),
            "forecast_guard_seconds": guard["elapsed_seconds"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--fit-only", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    study = root / "runs" / STUDY
    started = time.time()
    with StudyLock(study / ".runner.lock"):
        frozen = contract(root)
        frozen_path = study / ("smoke_contract.json" if args.smoke else "study_contract.json")
        if frozen_path.exists() and read(frozen_path) != frozen:
            raise RuntimeError("Changed source/plan/data; preserve previous artifacts")
        write(frozen_path, frozen)
        if args.smoke:
            entries = [fit_trial(root, d, m, 0.0 if m == "F0" else GRIDS[m][0], SEEDS[0], True)
                       for d in DATASETS for m in ("F0", *GRIDS)]
            verify_smoke(root, entries)
            if contract(root) != frozen:
                raise AssertionError("S0 source contract changed")
            done = study / "smoke_completed.json"
            if not done.exists():
                write(done, {"completed": True, "trials": entries, "wall_seconds": time.time() - started})
        else:
            if read(study / "smoke_contract.json") != frozen:
                raise AssertionError("Current source contract must pass S0")
            verify_smoke(root, read(study / "smoke_completed.json")["trials"])
            trials, selected, choices = [], [], {}
            for dataset in DATASETS:
                f0 = fit_trial(root, dataset, "F0", 0.0, SEEDS[0])
                trials.append(f0)
                selected.append({**f0, "role": "F0"})
                candidates = []
                for method, rates in GRIDS.items():
                    for lr in rates:
                        entry = fit_trial(root, dataset, method, lr, SEEDS[0])
                        trials.append(entry)
                        candidates.append(entry)
                        write(study / "trials.json", trials)
                best = {"H": min((e for e in candidates if e["method"].startswith("H_")),
                                  key=lambda e: (e["val_score"], e["method"], e["lr"])),
                        "OFF_LORA": min((e for e in candidates if e["method"] == "OFF_LORA"),
                                         key=lambda e: (e["val_score"], e["lr"]))}
                choices[dataset] = best
                write(study / "development_selection.json", choices)
                for role, chosen in best.items():
                    selected.append({**chosen, "role": role})
                    for seed in SEEDS[1:]:
                        entry = fit_trial(root, dataset, chosen["method"], chosen["lr"], seed)
                        trials.append(entry)
                        selected.append({**entry, "role": role})
                        write(study / "trials.json", trials)
            if len(trials) != 28 or len(selected) != 14 or contract(root) != frozen:
                raise AssertionError("Incomplete fits or changed study contract")
            selection = {"completed": True, "global_choices_frozen": True,
                         "study_contract_sha256": sha(frozen_path), "fit_trial_count": 28,
                         "choices": choices, "selected": selected}
            selection_path = study / "selection.json"
            if selection_path.exists() and read(selection_path) != selection:
                raise AssertionError("Final selection changed")
            write(selection_path, selection)
            if not (study / "fit_completed.json").exists():
                write(study / "fit_completed.json", {"completed": True, "fit_trials": 28, "adaptation_fits": 26,
                           "frozen_cache_trials": 2, "fit_guard_seconds": sum(e["guard_seconds"] for e in trials),
                           "invocation_wall_seconds": time.time() - started})
            if not args.fit_only:
                forecasts = []
                for entry in selected:
                    forecasts.append(forecast_trial(root, entry, selection_path))
                    write(study / "forecasts.json", forecasts)
                if contract(root) != frozen or len(forecasts) != 14:
                    raise AssertionError("Forecasts incomplete or source contract changed")
                if not (study / "completed.json").exists():
                    write(study / "completed.json", {"completed": True, "fit_trials": 28, "forecasts": 14,
                               "fit_guard_seconds": sum(e["guard_seconds"] for e in trials),
                               "forecast_guard_seconds": sum(e["forecast_guard_seconds"] for e in forecasts),
                               "invocation_wall_seconds": time.time() - started})
        write(study / "progress.json", {"state": "completed", "phase": "smoke" if args.smoke else
                                       ("fit" if args.fit_only else "forecast"), "updated_at": time.time()})


if __name__ == "__main__":
    main()
