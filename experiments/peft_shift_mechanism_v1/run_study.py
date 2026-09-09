"""Sequential guarded, resumable study. Select hyperparameters on validation only."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

from experiments.peft_adaptation_scope_v1.guard import StudyLock, run_guarded


CONDITIONS = ("Q00", "Q10", "Q01", "Q11")
METHODS = ("H_LIN", "H_MLP", "OFF_LORA", "JOINT", "LP", "TIME", "GROUP")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def hash_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_trial(root, study, condition, corpus, method, lr, smoke=False):
    seed = 7100 + corpus
    name = f"{condition}_c{corpus}/{method}/lr_{lr:.0e}"
    output = study / ("smoke" if smoke else "trials") / name
    cache = study / ("smoke_cache" if smoke else "cache") / f"{condition}_c{corpus}"
    result_path = output / "result.json"
    guard_path = output / "guard/status.json"
    data_path = study / "data" / f"{condition}_c{corpus}.npz"
    if result_path.exists() and guard_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        guard = json.loads(guard_path.read_text(encoding="utf-8"))
        if result.get("completed") and guard.get("completed") and guard.get("returncode") == 0:
            if result.get("data_sha256") != hash_file(data_path):
                raise RuntimeError(f"Completed trial data changed: {name}")
            return {"condition": condition, "corpus": corpus, "method": method, "lr": lr,
                    "path": str(output.relative_to(root)).replace("\\", "/"),
                    "val_score": result["val_score"], "cached_completion": True}
    command = [sys.executable, "-m", "experiments.peft_shift_mechanism_v1.train",
               "--data", str(data_path),
               "--output", str(output), "--cache", str(cache), "--method", method,
               "--lr", str(lr), "--seed", str(seed), "--steps", "200", "--val-every", "40"]
    if smoke:
        command.append("--smoke")
    write_json(study / "progress.json", {"state": "running", "current": name, "smoke": smoke,
                                         "pid": os.getpid(), "updated_at": time.time()})
    print(json.dumps({"event": "start", "trial": name, "smoke": smoke}), flush=True)
    status = run_guarded(command, output / "guard", root, timeout_seconds=900)
    if not status["completed"]:
        raise RuntimeError(f"Trial stopped: {name}; see {output / 'guard/status.json'}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if not result["completed"] or result.get("data_sha256") != hash_file(data_path):
        raise RuntimeError(f"Incomplete result: {name}")
    print(json.dumps({"event": "complete", "trial": name, "val_score": result["val_score"],
                      "seconds": status["elapsed_seconds"]}), flush=True)
    return {"condition": condition, "corpus": corpus, "method": method, "lr": lr,
            "path": str(output.relative_to(root)).replace("\\", "/"), "val_score": result["val_score"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    study = root / "runs/peft_shift_mechanism_v1"
    study.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with StudyLock(study / ".runner.lock"):
        source_paths = [Path(__file__).parent / name for name in ("data.py", "train.py", "run_study.py")]
        source_paths.extend(root / "experiments/peft_adaptation_scope_v1" / name
                            for name in ("modeling.py", "guard.py", "train.py"))
        sources = {str(p.relative_to(root)).replace("\\", "/"): hash_file(p) for p in source_paths}
        plan = root / "_docs/notes/tsfm_topics/08_peft_shift_mechanism_plan_20260908.md"
        contract = {"sources": sources, "plan_hash": hash_file(plan), "steps": 200,
                    "methods": list(METHODS), "conditions": list(CONDITIONS), "corpora": 3,
                    "head_lr_grid": [.0003, .001], "lora_lr_grid": [.00003, .0001],
                    "data_hashes": {path.name: hash_file(path) for path in sorted((study / "data").glob("*.npz"))},
                    "data_manifest_sha256": hash_file(study / "data/manifest.json")}
        if len(contract["data_hashes"]) != 12:
            raise RuntimeError("Expected all 12 prepared datasets")
        contract_path = study / ("smoke_contract.json" if args.smoke else "study_contract.json")
        if contract_path.exists() and json.loads(contract_path.read_text(encoding="utf-8")) != contract:
            raise RuntimeError("Source/plan contract changed; preserve runs and explicitly resolve before resuming")
        write_json(contract_path, contract)
        if args.smoke:
            completed = []
            for method in ("F0", "H_LIN", "H_MLP", "OFF_LORA", "JOINT", "LP", "TIME", "GROUP"):
                lr = .001 if method in ("F0", "H_LIN", "H_MLP") else .0001
                completed.append(run_trial(root, study, "Q00", 0, method, lr, True))
            write_json(study / "smoke_completed.json", {"completed": True, "trials": completed,
                                                       "wall_seconds": time.time() - started})
        else:
            if not (study / "smoke_completed.json").exists():
                raise RuntimeError("Complete S0 before starting the study")
            smoke = json.loads((study / "smoke_completed.json").read_text(encoding="utf-8"))
            smoke_contract = json.loads((study / "smoke_contract.json").read_text(encoding="utf-8"))
            if not smoke.get("completed") or smoke_contract != contract:
                raise RuntimeError("S0 must pass under the current source/plan/data contract")
            if {item["method"] for item in smoke["trials"]} != {"F0", *METHODS} or len(smoke["trials"]) != 8:
                raise RuntimeError("Incomplete S0 method set")
            for item in smoke["trials"]:
                guard = json.loads((root / item["path"] / "guard/status.json").read_text(encoding="utf-8"))
                if not guard.get("completed") or guard.get("returncode") != 0:
                    raise RuntimeError("A current S0 guard did not complete")
            selected, choices = [], {}
            for condition in CONDITIONS:
                for corpus in range(3):
                    selected.append(run_trial(root, study, condition, corpus, "F0", .001))
                    for method in METHODS:
                        if corpus == 0:
                            grid = (.0003, .001) if method in ("H_LIN", "H_MLP") else (.00003, .0001)
                            candidates = [run_trial(root, study, condition, corpus, method, lr) for lr in grid]
                            chosen = min(candidates, key=lambda result: result["val_score"])
                            choices[f"{condition}/{method}"] = {"lr": chosen["lr"], "candidates": candidates}
                            write_json(study / "selection.json", choices)
                        else:
                            chosen = run_trial(root, study, condition, corpus, method, choices[f"{condition}/{method}"]["lr"])
                        selected.append(chosen)
                        write_json(study / "selected_partial.json", selected)
            write_json(study / "selected.json", selected)
            write_json(study / "completed.json", {"completed": True, "selected_count": len(selected),
                                                 "fit_count": 112, "f0_count": 12,
                                                 "wall_seconds": time.time() - started})
        write_json(study / "progress.json", {"state": "completed", "smoke": args.smoke,
                                             "wall_seconds": time.time() - started})


if __name__ == "__main__":
    main()
