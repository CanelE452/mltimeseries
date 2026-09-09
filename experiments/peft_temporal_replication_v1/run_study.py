"""Repeat the fixed real-source procedure on the next nonoverlapping time block."""

import importlib.util
from pathlib import Path

from experiments.peft_external_gap_v1 import run_study as parent


STUDY = "peft_temporal_replication_v1"
PLAN = "_docs/notes/tsfm_topics/13_peft_temporal_replication_plan_20260908.md"
DATASETS, GRIDS, SEEDS = parent.DATASETS, parent.GRIDS, parent.SEEDS
read, sha, write, relative = parent.read, parent.sha, parent.write, parent.relative

_spec = importlib.util.spec_from_file_location(__package__ + "._shared_runner", parent.__file__)
_shared = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_shared)
_shared.STUDY, _shared.PLAN = STUDY, PLAN


def contract(root):
    root = Path(root).resolve()
    previous = parent.contract(root)
    previous_path = root / "runs/peft_external_gap_v1/study_contract.json"
    if previous != read(previous_path):
        raise AssertionError("The original real-source study contract changed")
    verified_path = root / "results/peft_external_gap_v1/verification.json"
    verified = read(verified_path)
    if not verified.get("passed") or not verified.get("completed"):
        raise AssertionError("Temporal replication requires the completed original analysis")
    protected = dict(verified["artifact_hashes"])
    protected.update(verified["analysis_sources"])
    protected.update({str(root / "results/peft_external_gap_v1" / name): value
                      for name, value in verified["output_hashes"].items()})
    for path in (verified_path, previous_path, root / parent.PLAN,
                 root / "runs/peft_external_gap_v1/selection.json"):
        protected[str(path)] = sha(path)
    final_audit_path = root / "results/peft_external_gap_v1/final_audit.json"
    final_audit = read(final_audit_path)
    report_path = root / "_docs/notes/tsfm_topics/12_peft_external_gap_results_20260908.md"
    if (not final_audit.get("passed") or sha(report_path) != final_audit["report_sha256"]
            or sha(verified_path) != final_audit["verification_sha256"]):
        raise AssertionError("The original report or final audit changed")
    protected[str(final_audit_path)] = sha(final_audit_path)
    protected[str(report_path)] = final_audit["report_sha256"]
    parent_artifacts = {}
    for name, expected in protected.items():
        path = (root / name).resolve()
        if sha(path) != expected:
            raise AssertionError(f"Original study artifact changed: {path}")
        parent_artifacts[relative(root, path)] = expected
    sources = dict(previous["sources"])
    for name in ("data.py", "raw.py", "train.py", "run_study.py"):
        path = root / "experiments" / STUDY / name
        sources[relative(root, path)] = sha(path)
    prepared = root / "runs" / STUDY / "prepared"
    required = [prepared / f"{dataset}_{part}.npz" for dataset in DATASETS for part in ("fit", "holdout")]
    if not all(path.is_file() for path in required):
        raise RuntimeError("Both new fixed time blocks must finish preparation before S0")
    manifest = read(prepared / "manifest.json")
    expected_files = {f"{dataset}_{part}" for dataset in DATASETS for part in ("fit", "holdout")}
    if (manifest.get("all_qc_passed") is not True or manifest["datasets"] != list(DATASETS)
            or set(manifest["files"]) != expected_files):
        raise AssertionError("Both fixed temporal-block QC gates must pass")
    for name, record in manifest["files"].items():
        if sha(prepared / f"{name}.npz") != record["sha256"]:
            raise AssertionError(f"Prepared manifest hash changed: {name}")
    artifacts = sorted(set(required + list(prepared.glob("*.json"))))
    return {"sources": sources, "plan_sha256": sha(root / PLAN),
            "previous_contract_sha256": sha(previous_path), "parent_artifacts": parent_artifacts,
            "data_files": {relative(root, path): sha(path) for path in artifacts},
            "settings": {**previous["settings"], "temporal_block_index": 1, "offset_days": 190,
                         "fresh_native_initialization": True, "repeat_full_selection_grid": True}}


_shared.contract = contract
verify_smoke = _shared.verify_smoke


def main():
    _shared.main()


if __name__ == "__main__":
    main()
