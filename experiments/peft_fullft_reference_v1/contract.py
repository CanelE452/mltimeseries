"""Freeze study20 inputs and procedures; verify the long parent chain only at gates."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STUDY = "peft_fullft_reference_v1"
PLAN = ROOT / "_docs/notes/tsfm_topics/20_peft_fullft_reference_plan_20260909.md"
CHECKPOINT = Path(r"C:\Users\User\.cache\huggingface\hub\models--amazon--chronos-2\snapshots\29ec3766d36d6f73f0696f85560a422f50e8498c")
SETTINGS = {"steps": 200, "val_every": 40, "effective_batch": 8, "micro_batch": 4,
            "rank": 8, "alpha": 16, "threads": 2, "seeds": [20000, 20001, 20002],
            "smoke_steps": 5, "smoke_lr": {"HEAD_ONLY": 3e-5, "LORA": 1e-5, "FULL_FT": 1e-6},
            "lr_grids": {"HEAD_ONLY": [3e-5, 1e-4, 3e-4], "LORA": [1e-5, 3e-5, 1e-4], "FULL_FT": [1e-6, 3e-6, 1e-5]},
            "context": 336, "horizon": 48, "stride": 24, "origin_counts": [90, 30, 20, 80],
            "bootstrap_replicates": 4000, "bootstrap_seed": 2026090920, "block_days": 7, "confidence": .95}


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            value.update(block)
    return value.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def relative(path):
    path = Path(path).resolve()
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)


def verify_hashes(values):
    for name, expected in values.items():
        if digest(ROOT/name) != expected:
            raise AssertionError(f"Frozen artifact changed: {name}")


def parent_hashes():
    old_path = ROOT/"runs/peft_revision_entry_v1/cpu_contract.json"
    old = read_json(old_path)
    audit_path = ROOT/"results/peft_revision_entry_v1/final_audit.json"
    audit = read_json(audit_path)
    editorial_path = ROOT/"results/peft_revision_entry_v1/final_audit_editorial.json"
    editorial = read_json(editorial_path)
    if not audit["passed"] or not editorial["passed"] or digest(audit_path) != editorial["prior_final_audit_sha256"]:
        raise AssertionError("Study19 completion evidence differs")
    output = {}
    for mapping in (old["protected_hashes"], old["source_hashes"], old["input_hashes"], audit["artifact_hashes"]):
        for name, value in mapping.items():
            key = relative(ROOT/name)
            if key in output and output[key] != value:
                raise AssertionError(f"Conflicting parent chain: {key}")
            output[key] = value
    report = ROOT/"_docs/notes/tsfm_topics/19_peft_revision_entry_results_20260908.md"
    if digest(report) != editorial["report_sha256"]:
        raise AssertionError("Study19 final report changed")
    for path in (old_path, audit_path, editorial_path, report, ROOT/"results/peft_revision_entry_v1/report_before_editorial.md"):
        output[relative(path)] = digest(path)
    verify_hashes(output)
    return output


def read_contract(path, verify="core"):
    path = Path(path).resolve()
    if path != ROOT/"runs"/STUDY/"study_contract.json":
        raise ValueError("Use the canonical study20 contract")
    saved = read_json(path)
    if not saved["completed"] or saved["settings"] != SETTINGS or saved["study"] != STUDY:
        raise AssertionError("Frozen study identity/settings differ")
    verify_hashes(saved["core_hashes"])
    if verify == "full":
        verify_hashes(saved["protected_hashes"])
        verify_hashes(saved["holdout_hashes"])
        verify_hashes(saved["raw_hashes"])
    elif verify != "core":
        raise ValueError("Verification must be core or full")
    return saved


def freeze():
    path = ROOT/"runs"/STUDY/"study_contract.json"
    if path.exists():
        return read_contract(path, verify="full")
    prepared = ROOT/"runs"/STUDY/"prepared"
    manifest = read_json(prepared/"manifest.json")
    datasets = manifest["datasets"]
    if set(datasets) != {"bike", "household"} or not manifest["all_qc_passed"]:
        raise AssertionError("Require both predetermined datasets")
    core = {relative(PLAN): digest(PLAN), relative(prepared/"manifest.json"): digest(prepared/"manifest.json")}
    sources = []
    for name in ("__init__.py", "contract.py", "data.py", "model.py", "train.py", "forecast.py", "baselines.py", "run_study.py"):
        sources.append(ROOT/"experiments"/STUDY/name)
    for namespace in ("peft_external_gap_v1", "peft_adaptation_scope_v1", "peft_temporal_replication_v1"):
        sources.extend((ROOT/"experiments"/namespace).glob("*.py"))
    native_root = Path(next(iter(importlib.util.find_spec("chronos").submodule_search_locations)))
    sources.extend(native_root.glob("*.py"))
    sources.extend((native_root/"chronos2").glob("*.py"))
    for source in sources:
        core[relative(source)] = digest(source)
    for source in CHECKPOINT.iterdir():
        if source.is_file() and source.suffix in (".json", ".safetensors"):
            core[relative(source)] = digest(source)
    holdout, raw = {}, {}
    for item in datasets.values():
        if item["origin_counts"] != {"train":90,"val":30,"cal":20,"eval":80}:
            raise AssertionError("Prepared origin counts differ from registered protocol")
        fit, future = ROOT/item["fit_data_path"], ROOT/item["holdout_data_path"]
        if digest(fit) != item["fit_data_sha256"] or digest(future) != item["holdout_data_sha256"]:
            raise AssertionError("Prepared input hash differs")
        core[relative(fit)] = digest(fit)
        holdout[relative(future)] = digest(future)
        raw[relative(ROOT/item["raw_source_path"])] = item["raw_sha256"]
    verify_hashes(raw)
    saved = {"completed": True, "study": STUDY, "created_at_utc": datetime.now(timezone.utc).isoformat(),
             "checkpoint": str(CHECKPOINT), "settings": SETTINGS, "datasets": datasets,
             "plan_path": relative(PLAN), "plan_sha256": digest(PLAN),
             "core_hashes": core, "holdout_hashes": holdout, "raw_hashes": raw, "protected_hashes": parent_hashes(),
             "pretraining_overlap": "UNKNOWN", "holdout_selection": "Next unused local temporal block, fixed by timestamps"}
    save_json(path, saved)
    return saved


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-full", action="store_true")
    args = parser.parse_args()
    result = read_contract(ROOT/"runs"/STUDY/"study_contract.json", "full") if args.verify_full else freeze()
    print(json.dumps({"completed": True, "protected": len(result["protected_hashes"]), "core": len(result["core_hashes"])}))
