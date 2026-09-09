"""Guarded Q00 retraining ablations with frozen references and source contracts."""

import argparse
import json
from pathlib import Path
import sys
import time

from experiments.peft_adaptation_scope_v1.guard import StudyLock, run_guarded
from experiments.peft_shift_mechanism_v1.run_study import hash_file, write_json


METHODS = ("OUT_ONLY", "ATTN_ONLY")
RATES = (3e-5, 1e-4)
STUDY = "peft_module_ablation_v1"
PARENT = "peft_shift_mechanism_v1"
PLAN = "_docs/notes/tsfm_topics/09_peft_module_ablation_plan_20260908.md"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def reference_paths(root):
    old = root / "runs" / PARENT
    selected = read(old / "selected.json")
    entries = [e for e in selected if e["condition"] == "Q00" and e["method"] in ("F0", "OFF_LORA")]
    if len(entries) != 6 or {(e["corpus"], e["method"]) for e in entries} != {
            (c, m) for c in range(3) for m in ("F0", "OFF_LORA")}:
        raise AssertionError("Expected exact six Q00 F0/BOTH references")
    refs = {}
    for entry in entries:
        c, method = entry["corpus"], entry["method"]
        lr = .001 if method == "F0" else 3e-5
        path = old / "trials" / f"Q00_c{c}" / method / f"lr_{lr:.0e}"
        if (root / entry["path"]).resolve() != path.resolve() or entry["lr"] != lr:
            raise AssertionError("Reference learning rate/path differs from the plan")
        result, guard = read(path / "result.json"), read(path / "guard/status.json")
        if not result["completed"] or not guard["completed"] or guard["returncode"] != 0 or guard["reasons"]:
            raise AssertionError("Reference is incomplete")
        if result["method"] != method or result["seed"] != 7100 + c or result["lr"] != lr:
            raise AssertionError("Reference identity differs")
        refs[(c, method)] = path
    return refs


def contract(root):
    old = root / "runs" / PARENT
    original = read(old / "study_contract.json")
    if not read(old / "completed.json")["completed"]:
        raise AssertionError("Parent study must be complete")
    for name, expected in original["sources"].items():
        if hash_file(root / name) != expected:
            raise AssertionError(f"Parent source changed: {name}")
    if hash_file(root / "_docs/notes/tsfm_topics/08_peft_shift_mechanism_plan_20260908.md") != original["plan_hash"]:
        raise AssertionError("Parent plan changed")
    if hash_file(old / "data/manifest.json") != original["data_manifest_sha256"]:
        raise AssertionError("Parent data manifest changed")
    refs = reference_paths(root)
    files = [old / name for name in ("study_contract.json", "completed.json", "selected.json", "selection.json")]
    for c in range(3):
        data = old / "data" / f"Q00_c{c}.npz"
        if hash_file(data) != original["data_hashes"][data.name]:
            raise AssertionError("Q00 data changed")
        files.append(data)
        for method in ("F0", "OFF_LORA"):
            p = refs[(c, method)]
            files.extend(p / name for name in ("result.json", "predictions.npz", "best_adaptation.pt", "guard/status.json"))
            if read(p / "result.json")["data_sha256"] != hash_file(data):
                raise AssertionError("Reference used different data")
        cache = Path(read(refs[(c, "F0")] / "result.json")["cache"])
        expected_parent = (old / "cache" / f"Q00_c{c}").resolve()
        if cache.parent.resolve() != expected_parent:
            raise AssertionError("Unexpected reference cache location")
        files.extend([cache / "manifest.json", *(cache / f"{split}_{name}.npy"
                      for split in ("train", "val", "eval") for name in ("hidden", "norm", "loc", "scale"))])
    old_smoke = old / "smoke/Q00_c0/OFF_LORA/lr_1e-04"
    files.extend(old_smoke / name for name in ("result.json", "predictions.npz", "guard/status.json"))
    sources = dict(original["sources"])
    for name in ("train.py", "run_study.py"):
        p = root / "experiments" / STUDY / name
        sources[str(p.relative_to(root)).replace("\\", "/")] = hash_file(p)
    return {"sources": sources, "plan_sha256": hash_file(root / PLAN),
            "references": {str(p.relative_to(root)).replace("\\", "/"): hash_file(p) for p in files},
            "methods": list(METHODS), "rates": list(RATES), "corpora": [0, 1, 2], "primary_lr": 3e-5,
            "steps": 200, "validation_interval": 40, "expected_new_fits": 12}


def trial(root, method, c, lr, smoke=False):
    study, old = root / "runs" / STUDY, root / "runs" / PARENT
    name = f"Q00_c{c}/{method}/lr_{lr:.0e}"
    output = study / ("smoke" if smoke else "trials") / name
    result_path = output / "result.json"
    expected_data = hash_file(old / "data" / f"Q00_c{c}.npz")
    if result_path.exists():
        result, guard = read(result_path), read(output / "guard/status.json")
        if not result.get("wrapped_completed") or not guard["completed"] or guard["returncode"] != 0:
            raise RuntimeError(f"Preserve and diagnose incomplete trial {name}")
    else:
        if output.exists() and any(output.iterdir()):
            raise RuntimeError(f"Preserve incomplete artifacts before retrying {name}")
        cache = study / "smoke_cache" / f"Q00_c{c}" if smoke else old / "cache" / f"Q00_c{c}"
        command = [sys.executable, "-m", f"experiments.{STUDY}.train", "--data",
                   str(old / "data" / f"Q00_c{c}.npz"), "--output", str(output), "--cache", str(cache),
                   "--method", method, "--lr", str(lr), "--seed", str(7100 + c),
                   "--steps", "200", "--val-every", "40"]
        if smoke:
            command.append("--smoke")
        write_json(study / "progress.json", {"state": "running", "current": name, "smoke": smoke,
                                              "updated_at": time.time()})
        print(json.dumps({"event": "start", "trial": name, "smoke": smoke}), flush=True)
        guard = run_guarded(command, output / "guard", root, timeout_seconds=300)
        if not guard["completed"]:
            raise RuntimeError(f"Failed guarded trial: {name}")
        result = read(result_path)
    if not result.get("wrapped_completed") or not result["completed"] or result["data_sha256"] != expected_data:
        raise AssertionError("Ablation wrapper/data verification failed")
    if (result["method"], result["seed"], result["lr"], result["smoke"]) != (method, 7100 + c, lr, smoke):
        raise AssertionError("Trial identity mismatch")
    if not smoke and result["cache_created_this_trial"]:
        raise AssertionError("Production ablations must reuse the original frozen cache")
    print(json.dumps({"event": "complete", "trial": name, "val_score": result["val_score"],
                      "seconds": guard["elapsed_seconds"]}), flush=True)
    return {"corpus": c, "method": method, "lr": lr, "path": str(output.relative_to(root)).replace("\\", "/"),
            "val_score": result["val_score"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    study = root / "runs" / STUDY
    study.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with StudyLock(study / ".runner.lock"):
        frozen = contract(root)
        cp = study / ("smoke_contract.json" if args.smoke else "study_contract.json")
        if cp.exists() and read(cp) != frozen:
            raise RuntimeError("Changed source/plan/reference contract; preserve existing artifacts")
        write_json(cp, frozen)
        if args.smoke:
            entries = [trial(root, method, 0, 1e-4, True) for method in (*METHODS, "OFF_LORA")]
            import numpy as np
            current = root / entries[-1]["path"]
            previous = root / "runs" / PARENT / "smoke/Q00_c0/OFF_LORA/lr_1e-04"
            a, b = read(current / "result.json"), read(previous / "result.json")
            for key in ("sampler_sha256", "module_initialization_seeds", "trainable_names", "best_step"):
                if a[key] != b[key]:
                    raise AssertionError(f"Wrapped BOTH differs from previous S0: {key}")
            if a["audits"]["initial_adaptation_sha256"] != b["audits"]["initial_adaptation_sha256"]:
                raise AssertionError("Wrapped BOTH initial tensors changed")
            with np.load(current / "predictions.npz") as x, np.load(previous / "predictions.npz") as y:
                for split in ("val", "eval"):
                    if not np.array_equal(x[split + "_predictions"], y[split + "_predictions"]):
                        raise AssertionError("Wrapped BOTH does not reproduce previous S0 predictions")
            for entry in entries:
                r = read(root / entry["path"] / "result.json")
                if r["sampler_sha256"] != b["sampler_sha256"]:
                    raise AssertionError("S0 minibatch sequence differs")
            smoke_done = study / "smoke_completed.json"
            if not smoke_done.exists():
                write_json(smoke_done, {"completed": True, "trials": entries,
                            "both_reproduces_parent_s0": True, "invocation_wall_seconds": time.time() - started,
                            "trial_guard_seconds": sum(read(root / e["path"] / "guard/status.json")["elapsed_seconds"] for e in entries)})
        else:
            smoke = read(study / "smoke_completed.json")
            if not smoke["completed"] or read(study / "smoke_contract.json") != frozen:
                raise RuntimeError("Current contract must first pass S0")
            if {e["method"] for e in smoke["trials"]} != {*METHODS, "OFF_LORA"} or len(smoke["trials"]) != 3:
                raise AssertionError("Incomplete S0 set")
            for e in smoke["trials"]:
                g = read(root / e["path"] / "guard/status.json")
                if not g["completed"] or g["returncode"] != 0:
                    raise AssertionError("S0 guard failed")
            entries, choices = [], {}
            for c in range(3):
                for method in METHODS:
                    candidates = [trial(root, method, c, lr) for lr in RATES]
                    entries.extend(candidates)
                    if c == 0:
                        choices[method] = {"lr": min(candidates, key=lambda e: e["val_score"])["lr"],
                                           "candidates": candidates}
                        write_json(study / "selection.json", choices)
                    write_json(study / "trials.json", entries)
            if contract(root) != frozen:
                raise AssertionError("Execution source/reference contract changed during training")
            done = study / "completed.json"
            if not done.exists():
                write_json(done, {"completed": True, "fit_count": len(entries),
                                 "invocation_wall_seconds": time.time() - started,
                                 "trial_guard_seconds": sum(read(root / e["path"] / "guard/status.json")["elapsed_seconds"] for e in entries)})
        write_json(study / "progress.json", {"state": "completed", "smoke": args.smoke,
                                              "wall_seconds": time.time() - started})


if __name__ == "__main__":
    main()
