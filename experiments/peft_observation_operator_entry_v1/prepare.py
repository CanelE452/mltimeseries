"""Freeze a small numerical audit while preserving completed study15 evidence."""

from pathlib import Path

from experiments.peft_objective_alignment_v1 import prepare as parent


ROOT = Path(__file__).resolve().parents[2]
STUDY = "peft_observation_operator_entry_v1"
PLAN = "_docs/notes/tsfm_topics/16_observation_operator_entry_plan_20260908.md"
SOURCES = ("__init__.py", "prepare.py", "model.py", "run.py")
read, write, sha = parent.read, parent.write, parent.sha


def build(root=ROOT):
    root = Path(root).resolve()
    parent.validate_contract(root)
    folder = root / "results/peft_objective_alignment_v1"
    audit = read(folder / "final_audit.json")
    verification = read(folder / "verification.json")
    if not audit["passed"] or not verification["passed"]:
        raise AssertionError("Complete study15 before its next candidate entry audit")
    protected = dict(verification["artifact_hashes"])
    for path, expected in ((folder / "verification.json", audit["verification_sha256"]),
                           (folder / "independent_audit.json", audit["independent_audit_sha256"]),
                           (folder / "windows_events.json", audit["windows_event_audit_sha256"]),
                           (folder / "figures/plot_manifest.json", audit["plot_manifest_sha256"]),
                           (root / "_docs/notes/tsfm_topics/15_peft_objective_alignment_results_20260908.md", audit["report_sha256"])):
        if sha(path) != expected:
            raise AssertionError(f"Study15 completion evidence changed: {path}")
        protected[str(path)] = expected
    protected[str(folder / "final_audit.json")] = sha(folder / "final_audit.json")
    for path, expected in verification["output_hashes"].items():
        protected[str(folder / path)] = expected
    plot = read(folder / "figures/plot_manifest.json")
    protected.update(plot["input_hashes"])
    protected.update(audit["analysis_guard_artifact_hashes"])
    for path, expected in plot["figure_hashes"].items():
        protected[str(folder / "figures" / path)] = expected
    normalized = {}
    parent.merge(root, normalized, protected)
    parent.verify_protected_hashes(root, normalized)
    return {"completed": True, "study": STUDY, "plan_path": PLAN, "plan_sha256": sha(root / PLAN),
            "source_hashes": {name: {"path": f"experiments/{STUDY}/{name}", "sha256": sha(root / "experiments" / STUDY / name)} for name in SOURCES},
            "protected_hashes": normalized,
            "settings": {"history": 24, "horizon": 4, "origin": 23, "a": .8, "innovation_variance": .36,
                         "base_sensor_variance": .04, "canonical_report_variance": .0025,
                         "seed": 2026090816, "max_error": 1e-10, "device": "cpu", "new_training_updates": 0},
            "scope": "Known linear Gaussian entry veto and numerical correctness only; no new PEFT method or empirical forecasting advantage"}


def validate(root=ROOT):
    root = Path(root).resolve()
    contract = read(root / "runs" / STUDY / "contract.json")
    if contract != build(root):
        raise AssertionError("The fixed CPU audit contract changed")
    return contract


def run(root=ROOT):
    root = Path(root).resolve()
    value = build(root)
    path = root / "runs" / STUDY / "contract.json"
    if path.exists():
        if read(path) != value:
            raise AssertionError("Preserve the existing different CPU audit contract")
    else:
        write(path, value)
    return value


if __name__ == "__main__":
    result = run()
    print({"completed": True, "protected_artifacts": len(result["protected_hashes"])})
