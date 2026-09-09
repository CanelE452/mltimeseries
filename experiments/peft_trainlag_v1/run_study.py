"""Run the fixed train-only lag study sequentially under the shared resource guard."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

from experiments.peft_adaptation_scope_v1.guard import StudyLock, run_guarded
from experiments.peft_module_ablation_v1.run_study import contract as prior_contract
from experiments.peft_shift_mechanism_v1.run_study import hash_file, write_json


STUDY = "peft_trainlag_v1"
PLAN = "_docs/notes/tsfm_topics/10_peft_trainlag_plan_20260908.md"
RATES = (3e-5, 1e-4)
LABELS = {("raw", "F0"): "F0", ("raw", "ATTN_ONLY"): "ATTN",
          ("aligned", "F0"): "ALIGN_F0", ("aligned", "ATTN_ONLY"): "ALIGN_ATTN"}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def contract(root):
    previous = prior_contract(root)
    if previous != read(root / "runs/peft_module_ablation_v1/study_contract.json"):
        raise AssertionError("Previous completed study contract changed")
    folder = root / "runs" / STUDY / "data"
    manifest = read(folder / "manifest.json")
    if not manifest["all_qc_passed"] or manifest["train_corpora"] != [0, 1, 2]:
        raise AssertionError("Fixed data generation/QC must finish before training")
    if set(manifest["files"]) != {f"Q00_c{c}.npz" for c in range(3)} or set(manifest["lag_selection_files"]) != {
            f"Q00_c{c}_lags.json" for c in range(3)}:
        raise AssertionError("The manifest must contain exactly three data and lag files")
    for group in ("files", "lag_selection_files"):
        for name, item in manifest[group].items():
            if hash_file(folder / name) != item["sha256"]:
                raise AssertionError(f"Data generation manifest mismatch: {name}")
    data_files = [folder / "manifest.json"]
    for c in range(3):
        data_files.extend((folder / f"Q00_c{c}.npz", folder / f"Q00_c{c}_lags.json"))
    sources = dict(previous["sources"])
    for name in ("data.py", "raw.py", "train.py", "run_study.py"):
        path = root / "experiments" / STUDY / name
        sources[str(path.relative_to(root)).replace("\\", "/")] = hash_file(path)
    return {
        "sources": sources,
        "plan_sha256": hash_file(root / PLAN),
        "prior_contract_sha256": hashlib.sha256(json.dumps(previous, sort_keys=True).encode()).hexdigest(),
        "data_files": {str(path.relative_to(root)).replace("\\", "/"): hash_file(path) for path in data_files},
        "settings": {"rates": list(RATES), "corpora": [0, 1, 2], "steps": 200, "val_every": 40,
                     "optimizer_seeds": [8100, 8101, 8102], "expected_fits": 12, "expected_f0": 6,
                     "selected_fm_count": 12, "input_modes": ["raw", "aligned"]},
    }


def trial(root, c, mode, base_method, lr, smoke=False):
    study = root / "runs" / STUDY
    label = LABELS[(mode, base_method)]
    name = f"Q00_c{c}/{label}/lr_{lr:.0e}"
    output = study / ("smoke" if smoke else "trials") / name
    data = study / "data" / f"Q00_c{c}.npz"
    lag = study / "data" / f"Q00_c{c}_lags.json"
    result_path = output / "result.json"
    if result_path.exists():
        result, guard = read(result_path), read(output / "guard/status.json")
        if not result.get("wrapped_completed") or not guard.get("completed") or guard.get("returncode") != 0 or guard.get("reasons"):
            raise RuntimeError(f"Preserve incomplete trial and diagnose: {name}")
    else:
        if output.exists() and any(output.iterdir()):
            raise RuntimeError(f"Preserve existing partial output before any retry: {name}")
        cache = study / ("smoke_cache" if smoke else "cache") / f"Q00_c{c}" / mode
        command = [sys.executable, "-m", f"experiments.{STUDY}.train", "--data", str(data),
                   "--output", str(output), "--cache", str(cache), "--method", base_method,
                   "--input-mode", mode, "--lag-file", str(lag), "--lr", str(lr),
                   "--seed", str(8100 + c), "--steps", "200", "--val-every", "40"]
        if smoke:
            command.append("--smoke")
        write_json(study / "progress.json", {"state": "running", "current": name, "smoke": smoke,
                                            "updated_at": time.time()})
        print(json.dumps({"event": "start", "trial": name, "smoke": smoke}), flush=True)
        guard = run_guarded(command, output / "guard", root, timeout_seconds=300)
        if not guard["completed"]:
            raise RuntimeError(f"Guarded trial stopped: {name}")
        result = read(result_path)
    if not result.get("completed") or not result.get("wrapped_completed") or result["data_sha256"] != hash_file(data):
        raise AssertionError("Incomplete result or changed data")
    if (result["method"], result["base_method"], result["input_mode"], result["seed"], result["lr"], result["smoke"]) != (
            label, base_method, mode, 8100 + c, lr, smoke):
        raise AssertionError("Trial identity changed")
    print(json.dumps({"event": "complete", "trial": name, "val_score": result["val_score"],
                      "seconds": guard["elapsed_seconds"]}), flush=True)
    return {"condition": "Q00", "corpus": c, "method": label, "input_mode": mode, "lr": lr,
            "path": str(output.relative_to(root)).replace("\\", "/"), "val_score": result["val_score"]}


def verify_smoke(root, entries):
    if len(entries) != 4 or {entry["method"] for entry in entries} != set(LABELS.values()):
        raise AssertionError("S0 requires all four procedures")
    results = {}
    lag_path = root / "runs" / STUDY / "data/Q00_c0_lags.json"
    lags = read(lag_path)["selected_lags"]
    for entry in entries:
        path = root / entry["path"]
        guard = read(path / "guard/status.json")
        result = read(path / "result.json")
        if not guard["completed"] or guard["returncode"] != 0 or guard["reasons"] or not result.get("wrapped_completed"):
            raise AssertionError("S0 trial failed")
        if result["lag_file_sha256"] != hash_file(lag_path) or result["selected_lags"] != lags:
            raise AssertionError("S0 used a different lag selection")
        if not result["audits"]["zero_update_identity"] or result["audits"]["zero_update_normalized_max_abs"] > 1e-5:
            raise AssertionError("S0 cache/direct zero-update identity failed")
        results[entry["method"]] = result
    for frozen, tuned in (("F0", "ATTN"), ("ALIGN_F0", "ALIGN_ATTN")):
        if results[frozen]["cache"] != results[tuned]["cache"] or results[tuned]["cache_created_this_trial"]:
            raise AssertionError("S0 adaptation did not reuse its matching frozen-mode cache")
        if results[frozen]["val_score"] != results[tuned]["validation_history"][0]["val_score"]:
            raise AssertionError("S0 frozen and zero-update validation forecasts differ")
    a, b = results["ATTN"], results["ALIGN_ATTN"]
    for key in ("sampler_sha256", "module_initialization_seeds", "trainable_names"):
        if a[key] != b[key]:
            raise AssertionError(f"Raw/aligned attention S0 differ in {key}")
    if a["audits"]["initial_adaptation_sha256"] != b["audits"]["initial_adaptation_sha256"]:
        raise AssertionError("Raw/aligned attention initial tensors differ")


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
            raise RuntimeError("Source/plan/data contract changed; preserve existing artifacts")
        write_json(cp, frozen)
        if args.smoke:
            entries = []
            for mode in ("raw", "aligned"):
                entries.append(trial(root, 0, mode, "F0", .001, True))
                entries.append(trial(root, 0, mode, "ATTN_ONLY", 1e-4, True))
            verify_smoke(root, entries)
            done_path = study / "smoke_completed.json"
            done = {"completed": True, "trials": entries, "same_attention_initialization_and_sampler": True}
        else:
            smoke = read(study / "smoke_completed.json")
            if not smoke["completed"] or read(study / "smoke_contract.json") != frozen:
                raise RuntimeError("Current contract must first pass S0")
            verify_smoke(root, smoke["trials"])
            entries, selected, choices = [], [], {}
            for c in range(3):
                for mode in ("raw", "aligned"):
                    f0 = trial(root, c, mode, "F0", .001)
                    entries.append(f0)
                    selected.append(f0)
                    candidates = [trial(root, c, mode, "ATTN_ONLY", lr) for lr in RATES]
                    entries.extend(candidates)
                    label = LABELS[(mode, "ATTN_ONLY")]
                    if c == 0:
                        choices[label] = {"lr": min(candidates, key=lambda e: (e["val_score"], e["lr"]))["lr"],
                                          "candidates": candidates}
                        write_json(study / "selection.json", choices)
                    selected.append(next(e for e in candidates if e["lr"] == choices[label]["lr"]))
                    write_json(study / "trials.json", entries)
                    write_json(study / "selected.json", selected)
            if len(entries) != 18 or len(selected) != 12:
                raise AssertionError("Incomplete study grid")
            done_path = study / "completed.json"
            done = {"completed": True, "fit_count": 12, "f0_count": 6, "trial_count": 18}
        if contract(root) != frozen:
            raise AssertionError("Source/plan/data/reference changed during execution")
        done.update({"invocation_wall_seconds": time.time() - started,
                     "trial_guard_seconds": sum(read(root / e["path"] / "guard/status.json")["elapsed_seconds"] for e in entries)})
        if not done_path.exists():
            write_json(done_path, done)
        write_json(study / "progress.json", {"state": "completed", "smoke": args.smoke,
                                            "invocation_wall_seconds": time.time() - started})


if __name__ == "__main__":
    main()
