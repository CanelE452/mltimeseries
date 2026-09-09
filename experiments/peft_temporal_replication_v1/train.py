"""Repeat the unchanged real-panel numerical loop in an isolated study namespace."""

from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import time
import traceback

from experiments.peft_external_gap_v1 import train as parent


STUDY = "peft_temporal_replication_v1"
PLAN = "_docs/notes/tsfm_topics/13_peft_temporal_replication_plan_20260908.md"
PARENT_STUDY = "peft_external_gap_v1"
PARENT_PLAN = "_docs/notes/tsfm_topics/12_peft_external_gap_plan_20260908.md"
file_hash, atomic_json = parent.file_hash, parent.atomic_json

_spec = importlib.util.spec_from_file_location(__package__ + "._shared_train", parent.__file__)
_shared = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_shared)
_shared.STUDY, _shared.PLAN = STUDY, PLAN
_original_source_state = _shared.source_state


def __getattr__(name):
    return getattr(_shared, name)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def absolute_hashes(root, values):
    result = {}
    for name, digest in values.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root):
            raise AssertionError("Study provenance must point inside the shared project")
        result[str(path)] = digest
    return result


def merge_hashes(*groups):
    result = {}
    for group in groups:
        for path, digest in group.items():
            if path in result and result[path] != digest:
                raise AssertionError(f"Conflicting frozen hashes for {path}")
            result[path] = digest
    return result


def frozen_provenance(root, smoke=False):
    root = Path(root).resolve()
    study = root / "runs" / STUDY
    contract_path = study / ("smoke_contract.json" if smoke else "study_contract.json")
    frozen = read(contract_path)
    prior = root / "runs" / PARENT_STUDY
    prior_contract_path = prior / "study_contract.json"
    prior_verification_path = root / "results" / PARENT_STUDY / "verification.json"
    if frozen["previous_contract_sha256"] != file_hash(prior_contract_path):
        raise AssertionError("Temporal replication must reference the unchanged original real-source contract")
    declared_parent = absolute_hashes(root, frozen["parent_artifacts"])
    if str(prior_verification_path) not in declared_parent:
        raise AssertionError("The new contract must freeze the original successful verification")
    parent.verify_hashes(declared_parent)
    verified, previous = read(prior_verification_path), read(prior_contract_path)
    if not verified.get("passed") or not verified.get("completed"):
        raise AssertionError("The original real-source result has not passed verification")
    if verified["study_contract_sha256"] != frozen["previous_contract_sha256"]:
        raise AssertionError("Original verification and original study contract disagree")
    required_parent = merge_hashes(
        absolute_hashes(root, verified["artifact_hashes"]),
        absolute_hashes(root, verified["analysis_sources"]),
        {str(root / "results" / PARENT_STUDY / name): digest for name, digest in verified["output_hashes"].items()},
        {str(prior_contract_path): frozen["previous_contract_sha256"],
         str(prior / "selection.json"): verified["selection_sha256"]})
    if any(declared_parent.get(path) != digest for path, digest in required_parent.items()):
        raise AssertionError("New contract omitted or changed an originally verified artifact")
    protected = merge_hashes(
        declared_parent, absolute_hashes(root, previous["sources"]),
        absolute_hashes(root, previous["data_files"]), absolute_hashes(root, frozen["sources"]),
        {str(root / PARENT_PLAN): previous["plan_sha256"], str(root / PLAN): frozen["plan_sha256"],
         str(contract_path): file_hash(contract_path)})
    selection = read(prior / "selection.json")
    f0_entries = [entry for entry in selection["selected"] if entry["role"] == "F0"]
    if {entry["dataset"] for entry in f0_entries} != {"bike", "household"} or len(f0_entries) != 2:
        raise AssertionError("Both original production F0 caches must remain available")
    for entry in f0_entries:
        result_path = (root / entry["path"] / "result.json").resolve()
        if declared_parent.get(str(result_path)) != entry["result_sha256"]:
            raise AssertionError("Original F0 result is not in the frozen verification")
        meta = read(result_path)
        protected = merge_hashes(protected, absolute_hashes(root, meta["cache_array_hashes"]))
    parent.verify_hashes(protected)
    return frozen, protected, contract_path


def source_state(root, args):
    root = Path(root).resolve()
    state = _original_source_state(root, args)
    frozen, protected, contract_path = frozen_provenance(root, getattr(args, "smoke", False))
    paths = [Path(__file__).resolve(), root / "experiments" / PARENT_STUDY / "data.py"]
    state["source_hashes"].update({path.relative_to(root).as_posix(): file_hash(path) for path in paths})
    if any(frozen["sources"].get(path) != digest for path, digest in state["source_hashes"].items()):
        raise AssertionError("Numerical producer or temporal consumer source is outside the frozen contract")
    data_name = Path(args.data).resolve().relative_to(root).as_posix()
    if frozen["data_files"].get(data_name) != state["data_sha256"] or state["plan_sha256"] != frozen["plan_sha256"]:
        raise AssertionError("Use the fixed new-block data and plan")
    state["protected_hashes"] = merge_hashes(state["protected_hashes"], protected,
                                              {str(path): file_hash(path) for path in paths})
    state["wrapper_contract"] = {
        "study": STUDY, "temporal_block_index": 1, "offset_days": 190,
        "source_sha256": file_hash(__file__), "numerical_producer_source_sha256": file_hash(parent.__file__),
        "numerical_producer_path": str(Path(parent.__file__).resolve()),
        "study_contract_path": str(contract_path), "study_contract_sha256": file_hash(contract_path),
        "previous_contract_sha256": frozen["previous_contract_sha256"],
        "previous_verification_sha256": file_hash(root / "results" / PARENT_STUDY / "verification.json"),
        "fresh_native_initialization": True, "same_full_selection_rule": True,
        "source_execution": "Private module instance of the unchanged original numerical source",
        "parent_artifacts": frozen["parent_artifacts"],
    }
    return state


def validate_selection(root, selection_path, fit_trial):
    root, selection_path, fit_trial = Path(root).resolve(), Path(selection_path).resolve(), Path(fit_trial).resolve()
    study = root / "runs" / STUDY
    if selection_path != study / "selection.json" or not fit_trial.is_relative_to(study / "trials"):
        raise ValueError("Forecasts require this temporal study's final selection and production trial")
    selection = read(selection_path)
    if not selection.get("completed") or not selection.get("global_choices_frozen") or selection.get("fit_trial_count") != 28:
        raise AssertionError("All 28 fits and both dataset choices must be frozen before holdout inference")
    contract_path = study / "study_contract.json"
    if selection["study_contract_sha256"] != file_hash(contract_path):
        raise AssertionError("Temporal selection and study contract disagree")
    entries = selection["selected"]
    expected = {(dataset, role, seed) for dataset in ("bike", "household") for role in ("F0", "H", "OFF_LORA")
                for seed in ((12000,) if role == "F0" else (12000, 12001, 12002))}
    if len(entries) != 14 or {(e["dataset"], e["role"], e["seed"]) for e in entries} != expected:
        raise AssertionError("Temporal selection must contain all fourteen fixed procedures")
    if set(selection["choices"]) != {"bike", "household"}:
        raise AssertionError("Both sources must retain the full selection rule")
    for entry in entries:
        if entry["role"] == "F0":
            if entry["method"] != "F0" or entry["lr"] != 0:
                raise AssertionError("F0 is the fixed native baseline")
        else:
            chosen = selection["choices"][entry["dataset"]][entry["role"]]
            if (entry["method"], entry["lr"]) != (chosen["method"], chosen["lr"]):
                raise AssertionError("Additional seeds changed the selected family or learning rate")
            if entry["role"] == "H" and entry["method"] not in ("H_MLP", "H_FULL"):
                raise AssertionError("Only the declared output-adaptation families may be selected as H")
            if entry["role"] == "OFF_LORA" and entry["method"] != "OFF_LORA":
                raise AssertionError("The standard 97-projection LoRA map must remain unchanged")
        path = (root / entry["path"]).resolve()
        if not path.is_relative_to(study / "trials") or file_hash(path / "result.json") != entry["result_sha256"]:
            raise AssertionError("Selected temporal fit path/result changed")
        if file_hash(path / "best_trainable.pt") != entry["checkpoint_sha256"]:
            raise AssertionError("Selected temporal checkpoint changed")
        meta, guard = read(path / "result.json"), read(path / "guard/status.json")
        if (not meta.get("completed") or not meta.get("wrapped_completed") or meta.get("smoke")
                or not guard.get("completed") or guard.get("returncode") != 0 or guard.get("reasons")):
            raise AssertionError("A selected fit, wrapper or guard is incomplete")
        if any(meta[key] != entry[key] for key in ("dataset", "method", "seed", "lr")):
            raise AssertionError("Selected temporal identity does not match its saved fit")
    matches = [entry for entry in entries if (root / entry["path"]).resolve() == fit_trial]
    if len(matches) != 1:
        raise AssertionError("The requested temporal checkpoint is not uniquely selected")
    frozen, protected, contract_path = frozen_provenance(root)
    protected = merge_hashes(protected, absolute_hashes(root, frozen["data_files"]),
                              {str(selection_path): file_hash(selection_path), str(contract_path): file_hash(contract_path)})
    parent.verify_hashes(protected)
    return matches[0], protected


@contextmanager
def registration(output, states):
    original_atomic, original_state = _shared.atomic_json, _shared.source_state
    result_path = (Path(output) / "result.json").resolve()

    def pending_json(path, value):
        if Path(path).resolve() == result_path:
            value = {**value, "inner_training_completed": bool(value.get("completed")),
                     "completed": False, "wrapped_completed": False}
        return original_atomic(path, value)

    def capture_state(root, args):
        state = source_state(root, args)
        states.append(state)
        return state

    try:
        _shared.atomic_json, _shared.source_state = pending_json, capture_state
        yield
    finally:
        _shared.atomic_json, _shared.source_state = original_atomic, original_state


def run(args):
    started = time.perf_counter()
    output, states = Path(args.output).resolve(), []
    with registration(output, states):
        result = (_shared.run_fit if args.command == "fit" else _shared.run_forecast)(args)
        pending = read(output / "result.json")
        if (len(states) != 1 or not result.get("completed") or pending.get("completed")
                or pending.get("wrapped_completed") or not pending.get("inner_training_completed")):
            raise AssertionError("The numerical loop bypassed the temporal wrapper completion gate")
        parent.verify_hashes(states[0]["protected_hashes"])
        parent.verify_hashes(result["protected_hashes"])
        if result["source_hashes"] != states[0]["source_hashes"]:
            raise AssertionError("Completed numerical result lost the wrapper/producer provenance")
        if args.command == "fit":
            parent.verify_hashes(result["cache_array_hashes"])
            if not result["audits"].get("zero_update_identity") or not result["audits"].get("checkpoint_reload_verified"):
                raise AssertionError("Inherited numerical/checkpoint audits failed")
        elif not result.get("checkpoint_reload_verified") or not result.get("model_unchanged"):
            raise AssertionError("Selected-checkpoint inference audit failed")
        result.update({"completed": True, "inner_training_completed": True, "wrapped_completed": True,
                       "wrapper_contract": states[0]["wrapper_contract"],
                       "wrapper_audits": {"parent_verified_artifacts_unchanged": True,
                                          "parent_production_caches_unchanged": True,
                                          "numerical_source_unchanged": True, "consumer_source_unchanged": True,
                                          "isolated_module_namespace": True, "completion_gate_passed": True},
                       "wrapper_wall_seconds": time.perf_counter() - started})
        atomic_json(output / "result.json", result)
    return result


_shared.source_state = source_state
_shared.validate_selection = validate_selection


def main():
    args = _shared.parser().parse_args()
    try:
        run(args)
    except Exception as error:
        output = Path(args.output).resolve()
        root = Path(__file__).resolve().parents[2]
        finished = (output / "result.json").is_file() and read(output / "result.json").get("completed")
        if output.is_relative_to(root / "runs" / STUDY) and output.is_dir() and not finished:
            atomic_json(output / "failure.json", {"completed": False, "stage": args.command,
                        "error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
