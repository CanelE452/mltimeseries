"""Freeze data, executable sources, native model files and prior evidence."""

import argparse
import importlib.metadata
import inspect
from pathlib import Path
import sys

from .fetch import ROOT, STUDY, sha, read, write_once

PLAN = "_docs/notes/tsfm_topics/17_coarse_supervision_learning_plan_20260908.md"
REVISION = "29ec3766d36d6f73f0696f85560a422f50e8498c"
FILES = ("__init__.py", "fetch.py", "data_qc.py", "data.py", "ridge.py", "model.py",
         "cache.py", "train.py", "analysis.py", "prepare.py", "forecast.py", "run_study.py")


def verify(root, hashes):
    for filename, expected in hashes.items():
        if sha(Path(root) / filename) != expected:
            raise AssertionError(f"Frozen artifact changed: {filename}")


def build(root=ROOT):
    from chronos.chronos2 import Chronos2Model
    from . import data

    root = Path(root).resolve()
    entry = root / "data_external/bdg2_coarse_supervision_v1"
    parent = read(entry / "contract.json")
    protected = dict(parent["protected_hashes"])
    protected.update(parent["source_hashes"])
    protected[parent["plan_path"]] = parent["plan_sha256"]
    for filename in ("contract.json", "qc.json", "independent_audit.json", "receipts/metadata.json",
                     "receipts/electricity.json", "raw/metadata.csv", "raw/electricity.csv"):
        path = entry / filename
        protected[str(path)] = sha(path)
    prepared = data.validate(root)
    if not prepared.get("completed") or not prepared["metadata"].get("completed"):
        raise AssertionError("Prepared data did not pass structural coverage gates")
    for filename in ("data_build_contract.json",):
        path = root / "runs" / STUDY / "data" / filename
        protected[str(path)] = sha(path)
    build_contract = read(path)
    verify(root, build_contract["source_hashes"])
    verify(root, build_contract["input_hashes"])
    if build_contract["plan_sha256"] != sha(root / PLAN):
        raise AssertionError("Learning plan changed since data preparation")
    paths = dict(prepared["paths"])
    stage = root / "runs" / STUDY
    paths.update({"ridge_coarse": str(stage / "ridge/coarse_lift_weights.npz"),
                  "train_result": str(stage / "train/result.json"),
                  "selection": str(stage / "selection.json"),
                  "forecast_result": str(stage / "forecast/result.json")})
    sources = {f"experiments/{STUDY}/{name}": sha(root / "experiments" / STUDY / name) for name in FILES}
    for path in sorted((root / "experiments" / STUDY / "tests").glob("test_*.py")):
        sources[path.relative_to(root).as_posix()] = sha(path)
    checkpoint = Path.home() / ".cache/huggingface/hub/models--amazon--chronos-2/snapshots" / REVISION
    runtime_files = list(Path(inspect.getsourcefile(Chronos2Model)).parent.glob("*.py"))
    runtime_files += [checkpoint / "config.json", *checkpoint.glob("*.safetensors"), Path(sys.executable)]
    for package in ("peft_external_gap_v1", "peft_adaptation_scope_v1"):
        runtime_files += list((root / "experiments" / package).glob("*.py"))
    runtime = {str(p.resolve()): sha(p) for p in runtime_files}
    versions = {name: importlib.metadata.version(name) for name in ("torch", "numpy", "transformers", "peft", "chronos-forecasting")}
    verify(root, protected)
    return {"completed": True, "study": STUDY, "plan_path": PLAN, "plan_sha256": sha(root / PLAN),
            "checkpoint": str(checkpoint), "checkpoint_revision": REVISION,
            "source_hashes": sources, "protected_hashes": protected, "runtime_hashes": runtime,
            "runtime_versions": versions, "python": sys.version, "data": prepared["data"], "paths": paths,
            "scope": "Monthly-only necessity screen; standard LoRA is not a new method",
            "arms": ["PROFILE", "F0", "COARSE_LIFT", "FROZEN_HEAD", "ATTN_LORA"]}


def validate(root=ROOT, contract_path=None):
    root = Path(root).resolve()
    path = Path(contract_path or root / "runs" / STUDY / "study_contract.json").resolve()
    if path != root / "runs" / STUDY / "study_contract.json":
        raise ValueError("Use the canonical study contract")
    contract = read(path)
    if not contract.get("completed") or contract.get("study") != STUDY or contract.get("checkpoint_revision") != REVISION:
        raise AssertionError("Incomplete or foreign study contract")
    for key in ("source_hashes", "protected_hashes", "runtime_hashes"):
        verify(root, contract[key])
    if sha(root / contract["plan_path"]) != contract["plan_sha256"]:
        raise AssertionError("Learning plan changed")
    for entry in contract["data"].values():
        if sha(entry["path"]) != entry["sha256"]:
            raise AssertionError(f"Prepared data changed: {entry['path']}")
    if contract["python"] != sys.version or any(importlib.metadata.version(k) != v for k, v in contract["runtime_versions"].items()):
        raise AssertionError("Execution environment changed")
    return contract


def run(root=ROOT):
    contract = build(root)
    write_once(Path(root) / "runs" / STUDY / "study_contract.json", contract)
    validate(root)
    return {"completed": True, "protected_count": len(contract["protected_hashes"]),
            "source_count": len(contract["source_hashes"]), "runtime_count": len(contract["runtime_hashes"])}


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(run())
