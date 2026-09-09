"""Freeze the objective diagnostic without changing its parent experiments."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


STUDY = "peft_objective_alignment_v1"
PLAN = "_docs/notes/tsfm_topics/15_peft_objective_alignment_plan_20260908.md"
ARMS = ("NATIVE", "NORM_ALIGNED", "RAW_ALIGNED")
DATASETS = ("bike", "household")
SOURCE_FILES = ("__init__.py", "prepare.py", "losses.py", "train.py", "forecast.py", "run_study.py", "analyse.py", "plot.py")
SETTINGS = {"seed": 12000, "lr": 3e-5, "steps": 200, "val_every": 40,
            "smoke_lr": 1e-5, "smoke_steps": 5, "smoke_val_every": 5,
            "method": "OFF_LORA", "context": 336, "horizon": 48, "quantiles": 21,
            "micro_groups": 4, "accumulation": 2, "gradient_clip": 1.,
            "initial_gradient_origins": "np.linspace(0,N_train-1,8).astype(int)",
            "primary_confidence": .975, "bootstrap_replicates": 4000,
            "bootstrap_seed": 2026090815, "primary_block_days": 7, "practical_threshold": .01}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(root, path):
    path = Path(path).resolve()
    try:
        return path.relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        return str(path)


def verify_protected_hashes(root, hashes):
    for path, expected in hashes.items():
        if sha(Path(root) / path) != expected:
            raise AssertionError(f"Protected artifact changed: {path}")


def merge(root, destination, values):
    for path, digest in values.items():
        key = relative(root, Path(root) / path)
        if key in destination and destination[key] != digest:
            raise AssertionError(f"Conflicting source history: {key}")
        destination[key] = digest


def add(root, destination, path, expected=None):
    path = Path(root) / path
    actual = sha(path)
    if expected is not None and actual != expected:
        raise AssertionError(f"Parent provenance changed: {path}")
    merge(root, destination, {str(path): actual})
    return actual


def _collect_parent(root):
    prior = root / "runs/peft_selection_regret_v1/selection_contract.json"
    result = root / "results/peft_selection_regret_v1"
    contract, verification, audit = read(prior), read(result / "verification.json"), read(result / "final_audit.json")
    if not (contract.get("completed") and verification.get("passed") and verification.get("completed") and audit.get("passed")):
        raise AssertionError("Study14 must be complete and audited")
    protected = {}
    add(root, protected, prior, verification["selection_contract_sha256"])
    add(root, protected, result / "verification.json", audit["verification_sha256"])
    add(root, protected, result / "final_audit.json")
    add(root, protected, "_docs/notes/tsfm_topics/14_peft_selection_regret_results_20260908.md", audit["report_sha256"])
    add(root, protected, contract["plan_path"], contract["plan_sha256"])
    for group in (contract["protected_hashes"], verification["artifact_hashes"], verification["analysis_sources"]):
        merge(root, protected, group)
    for entry in contract["source_hashes"].values():
        add(root, protected, entry["path"], entry["sha256"])
    for filename, digest in verification["output_hashes"].items():
        add(root, protected, result / filename, digest)
    manifest_path = result / "figures/plot_manifest.json"
    add(root, protected, manifest_path, audit["plot_manifest_sha256"])
    plot = read(manifest_path)
    merge(root, protected, plot["input_hashes"])
    for filename, digest in plot["figure_hashes"].items():
        add(root, protected, result / "figures" / filename, digest)
    add(root, protected, result / "windows_events.json", audit["windows_event_audit_sha256"])
    for folder in (root / "runs/peft_selection_regret_v1/invocations", root / "runs/peft_selection_regret_v1/analysis_guard"):
        for path in sorted(folder.glob("*")):
            if path.is_file():
                add(root, protected, path)
    return protected


def _datasets(root, protected):
    output, checkpoint = {}, None
    parent = root / "runs/peft_external_gap_v1"
    for dataset in DATASETS:
        fit_path, holdout_path = [parent / "prepared" / f"{dataset}_{stage}.npz" for stage in ("fit", "holdout")]
        refs = {"native_reference_fit": parent / "trials" / dataset / "OFF_LORA/lr_3e-05_seed_12000",
                "smoke_reference_fit": parent / "smoke" / dataset / "OFF_LORA/lr_1e-05_seed_12000",
                "f0_forecast": parent / "forecasts" / dataset / "F0_seed_12000"}
        records = {}
        for key, path in refs.items():
            record = read(path / "result.json")
            guard = read(path / "guard/status.json")
            if not record.get("completed") or not guard.get("completed") or guard["returncode"] != 0 or guard["reasons"]:
                raise AssertionError(f"Incomplete parent reference: {path}")
            if record["dataset"] != dataset or record["seed"] != SETTINGS["seed"]:
                raise AssertionError("Parent reference identity mismatch")
            for name in ("result.json", "trial_contract.json", "predictions.npz"):
                add(root, protected, path / name)
            if key != "f0_forecast":
                add(root, protected, path / "best_trainable.pt", record["checkpoint_sha256"])
                merge(root, protected, record["cache_array_hashes"])
                if record["method"] != "OFF_LORA" or record["lr"] != (1e-5 if key.startswith("smoke") else 3e-5):
                    raise AssertionError("Native replay reference uses a different procedure")
            elif record["method"] != "F0":
                raise AssertionError("Reused baseline is not F0")
            merge(root, protected, record["protected_hashes"])
            for log in sorted((path / "guard").glob("*")):
                if log.is_file():
                    add(root, protected, log)
            records[key] = record
        with np.load(fit_path, allow_pickle=False) as z:
            if any(key in z for key in ("cal_origins", "eval_origins")):
                raise AssertionError("Fit archive includes future origin partitions")
            train, val, indices = z["train_origins"], z["val_origins"], z["target_indices"]
            if (len(train), len(val), len(indices), int(z["context"]), int(z["horizon"])) != (63, 13, 2, 336, 48):
                raise AssertionError("Study12 fit split contract changed")
            targets = z["target_values"]
            counts = lambda origins: np.isfinite(np.stack([targets[o:o + 48, indices].T for o in origins])).sum(axis=(0, 2)).tolist()
            valid, smoke_valid = counts(train), counts(train[:8])
            if min(*valid, *smoke_valid) <= 0:
                raise AssertionError("Every target needs observed optimization labels")
            item = {"train_origin_count": len(train), "train_valid_counts": valid,
                    "smoke_train_valid_counts": smoke_valid, "channels": z["channels"].tolist(),
                    "target_indices": indices.tolist(), "fit_std": z["fit_std"].tolist(),
                    "initial_gradient_origins": train[np.linspace(0, len(train) - 1, 8).astype(int)].tolist()}
        for name, path in (("fit_data", fit_path), ("holdout_data", holdout_path)):
            item[name + "_path"] = relative(root, path)
            item[name + "_sha256"] = add(root, protected, path)
        item.update({key: relative(root, path) for key, path in refs.items()})
        item["cache"] = relative(root, records["native_reference_fit"]["cache"])
        item["smoke_cache"] = relative(root, records["smoke_reference_fit"]["cache"])
        current_checkpoint = str(Path(records["native_reference_fit"]["checkpoint"]).resolve())
        if checkpoint is not None and current_checkpoint != checkpoint:
            raise AssertionError("Native checkpoint differs between sources")
        checkpoint = current_checkpoint
        output[dataset] = item
    return output, checkpoint


def build_contract(root):
    root = Path(root).resolve()
    protected = _collect_parent(root)
    datasets, checkpoint = _datasets(root, protected)
    sources = {}
    for name in SOURCE_FILES:
        path = root / "experiments" / STUDY / name
        sources[name] = {"path": relative(root, path), "sha256": sha(path)}
    verify_protected_hashes(root, protected)
    jobs = lambda phase: [{"dataset": dataset, "arm": arm, "output": f"runs/{STUDY}/{phase}/{dataset}/{arm}"}
                          for dataset in DATASETS for arm in ARMS]
    return {"completed": True, "study": STUDY, "plan_path": PLAN, "plan_sha256": sha(root / PLAN),
            "checkpoint": checkpoint, "settings": dict(SETTINGS), "arms": list(ARMS),
            "source_hashes": sources, "protected_hashes": protected, "datasets": datasets,
            "smoke_jobs": jobs("smoke"), "fit_jobs": jobs("trials"),
            "scope": "Exploratory reuse of study12; same seed and fixed LR; existing raw-loss control, not a novel method"}


def validate_contract(root, contract_path=None):
    root = Path(root).resolve()
    path = Path(contract_path) if contract_path is not None else root / "runs" / STUDY / "study_contract.json"
    if path.resolve() != root / "runs" / STUDY / "study_contract.json":
        raise ValueError("Use the single canonical objective diagnostic contract")
    saved = read(path)
    if saved != build_contract(root):
        raise AssertionError("The frozen objective diagnostic contract changed")
    return saved


def run(root):
    root = Path(root).resolve()
    path = root / "runs" / STUDY / "study_contract.json"
    contract = build_contract(root)
    if path.exists():
        if read(path) != contract:
            raise FileExistsError("Preserve the previous objective diagnostic contract")
    else:
        write(path, contract)
    return contract


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    result = run(Path(__file__).resolve().parents[2])
    print(json.dumps({"completed": True, "study": STUDY, "parent_artifacts": len(result["protected_hashes"])}))
