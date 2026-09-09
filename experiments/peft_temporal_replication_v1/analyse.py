"""Verify the unchanged procedure on block 13 and report blocks 12/13 separately."""

import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_key] = "2"

import numpy as np


STUDY = "peft_temporal_replication_v1"
PARENT_STUDY = "peft_external_gap_v1"
BOOTSTRAP_SEED = 2026090813
REFERENCE_ANALYSIS_SHA256 = "e0d32f04674a45d02df58ce27eb4f432ea88f576780db3bfb554c9e90e5c702a"
REFERENCE_PLOT_SHA256 = "881094853811a6e23c84428c64d1e449023d3a7be7458a6f8c167d60a09d59af"


def load_reference(name, filename, expected_sha256):
    path = Path(__file__).resolve().parents[1] / PARENT_STUDY / filename
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
        raise AssertionError("The verified original numerical/reporting source changed")
    spec = importlib.util.spec_from_file_location(__package__ + "." + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_core = load_reference("_shared_analysis", "analyse.py", REFERENCE_ANALYSIS_SHA256)
_core.STUDY, _core.BOOTSTRAP_SEED = STUDY, BOOTSTRAP_SEED
_original_weights = _core.moving_block_weights
read_json, write_json, sha256_file = _core.read_json, _core.write_json, _core.sha256_file


def moving_block_weights(n, block, replicates=4000, seed=BOOTSTRAP_SEED):
    # Changing a module global does not change a previously bound default argument.
    return _original_weights(n, block, replicates=replicates, seed=seed)


_core.moving_block_weights = moving_block_weights


def __getattr__(name):
    return getattr(_core, name)


def verify_wrapper(root, meta, frozen, checked):
    root = Path(root).resolve()
    if any(meta.get(key) is not True for key in ("completed", "inner_training_completed", "wrapped_completed")):
        raise AssertionError("The temporal numerical loop and wrapper must both complete")
    wrapper = meta["wrapper_contract"]
    consumer = root / "experiments" / STUDY / "train.py"
    producer = root / "experiments" / PARENT_STUDY / "train.py"
    contract_path = root / "runs" / STUDY / "study_contract.json"
    previous_verification = root / "results" / PARENT_STUDY / "verification.json"
    expected = {"study": STUDY, "temporal_block_index": 1, "offset_days": 190,
                "fresh_native_initialization": True, "same_full_selection_rule": True,
                "source_sha256": sha256_file(consumer),
                "numerical_producer_source_sha256": sha256_file(producer),
                "study_contract_sha256": sha256_file(contract_path),
                "previous_contract_sha256": frozen["previous_contract_sha256"],
                "previous_verification_sha256": sha256_file(previous_verification),
                "parent_artifacts": frozen["parent_artifacts"]}
    for key, value in expected.items():
        if wrapper.get(key) != value:
            raise AssertionError(f"Temporal wrapper contract mismatch: {key}")
    if (Path(wrapper["numerical_producer_path"]).resolve() != producer
            or Path(wrapper["study_contract_path"]).resolve() != contract_path):
        raise AssertionError("Temporal producer or study contract path changed")
    for key in ("parent_verified_artifacts_unchanged", "parent_production_caches_unchanged",
                "numerical_source_unchanged", "consumer_source_unchanged",
                "isolated_module_namespace", "completion_gate_passed"):
        if meta["wrapper_audits"].get(key) is not True:
            raise AssertionError(f"Temporal wrapper audit did not pass: {key}")
    protected = {(root / name).resolve(): value for name, value in meta["protected_hashes"].items()}
    required = {**frozen["parent_artifacts"], str(consumer): sha256_file(consumer),
                str(producer): sha256_file(producer), str(contract_path): sha256_file(contract_path)}
    if any(protected.get((root / name).resolve()) != digest for name, digest in required.items()):
        raise AssertionError("Temporal result omitted a protected parent, source or contract hash")
    _core.verify_file_hashes(root, required, checked)


_original_verify_fit, _original_verify_forecast = _core.verify_fit, _core.verify_forecast


def verify_fit(root, entry, panel, frozen, checked):
    meta = _original_verify_fit(root, entry, panel, frozen, checked)
    verify_wrapper(root, meta, frozen, checked)
    cache = Path(meta["cache"]).resolve()
    if not cache.is_relative_to(Path(root).resolve() / "runs" / STUDY):
        raise AssertionError("A new-block fit cannot reuse an original-block feature cache")
    return meta


def verify_forecast(root, entry, fit, panel, frozen, selection_sha, contract_sha, checked):
    meta, arrays = _original_verify_forecast(root, entry, fit, panel, frozen, selection_sha, contract_sha, checked)
    verify_wrapper(root, meta, frozen, checked)
    return meta, arrays


def verify_temporal_layout(dataset, metadata, timestamps, previous_metadata, previous_timestamps):
    from .data import EXPECTED_BOUNDARIES, VERSION

    current = np.asarray(timestamps).astype("datetime64[s]")
    previous = np.asarray(previous_timestamps).astype("datetime64[s]")
    boundaries = metadata["contract"]["boundaries"]
    expected = EXPECTED_BOUNDARIES[dataset]
    if metadata["version"] != VERSION or boundaries != expected:
        raise AssertionError("Temporal block differs from the fixed calendar plan")
    for values in (current, previous):
        if len(values) != 190 * 24 or np.any(np.diff(values) != np.timedelta64(1, "h")):
            raise AssertionError("Both declared blocks must contain exactly 190 complete hourly days")
    if (previous[-1] + np.timedelta64(1, "h") != current[0]
            or current[0] != np.datetime64(expected["precontext"][0], "s")
            or current[-1] + np.timedelta64(1, "h") != np.datetime64(expected["eval"][1], "s")):
        raise AssertionError("The new precontext must begin exactly after the old evaluation ends")
    old_bounds = previous_metadata["contract"]["boundaries"]
    if (np.datetime64(old_bounds["precontext"][0], "s") != previous[0]
            or np.datetime64(old_bounds["eval"][1], "s") != current[0]):
        raise AssertionError("Original declared calendar and timestamp archive disagree")
    replication = metadata["source"]["temporal_replication"]
    required = {"replication_index": 1, "offset_days_from_first_complete_day": 190, "block_days": 190,
                "raw_timestep_overlap_with_previous_block": 0,
                "replication_block": [expected["precontext"][0], expected["eval"][1]],
                "previous_block": [old_bounds["precontext"][0], old_bounds["eval"][1]]}
    if any(replication.get(key) != value for key, value in required.items()):
        raise AssertionError("Temporal replication metadata disagrees with the actual calendar")
    if (replication["source_end_index_exclusive"] - replication["source_start_index"] != len(current)
            or metadata["source"]["source_sha256"] != previous_metadata["source"]["source_sha256"]):
        raise AssertionError("Replication must use the same source and exact new 190-day crop")
    return {"previous_boundaries": old_bounds, "replication_boundaries": boundaries,
            "same_source": True, "raw_timestamp_overlap": 0, "new_time_block_not_new_source": True}


_original_load_panels, _original_load_raw = _core.load_panels, _core.load_raw
_temporal_audits = {}


def load_panels(root, dataset, checked):
    panels = _original_load_panels(root, dataset, checked)
    old_path = Path(root) / "runs" / PARENT_STUDY / "prepared" / f"{dataset}_holdout.npz"
    with np.load(old_path, allow_pickle=False) as old:
        old_meta = json.loads(str(old["manifest_json"].item()))
        old_timestamps = old["timestamps"]
    _temporal_audits[dataset] = verify_temporal_layout(dataset, panels[1].metadata, panels[1].timestamps,
                                                      old_meta, old_timestamps)
    return panels


def load_raw(root, dataset, *panels):
    values = _original_load_raw(root, dataset, *panels)
    meta = values[0]
    for prefix, study in (("wrapper", STUDY), ("delegated_raw", PARENT_STUDY)):
        source = Path(root).resolve() / "experiments" / study / "raw.py"
        if (Path(meta[f"{prefix}_source_path"]).resolve() != source
                or meta[f"{prefix}_source_sha256"] != sha256_file(source)):
            raise AssertionError("RAW temporal wrapper and original numerical producer must be unchanged")
    if meta.get("study") != STUDY:
        raise AssertionError("RAW output was not produced for the new time block")
    return values


_core.verify_fit, _core.verify_forecast = verify_fit, verify_forecast
_core.load_panels, _core.load_raw = load_panels, load_raw


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def summarize_block(rows, effects, number):
    expected = {(source, role, seed, procedure) for source in ("bike", "household")
                for role in ("F0", "H", "OFF_LORA", "RAW")
                for seed in ((12000,) if role == "F0" else ((-1,) if role == "RAW" else (12000, 12001, 12002)))
                for procedure in ("SORT", "QCAL")}
    keys = [(r["source"], r["role"], int(r["seed"]), r["procedure"]) for r in rows]
    if len(rows) != 32 or set(keys) != expected:
        raise AssertionError("Each temporal block must retain its own complete 32-row result")
    if (effects.get("completed") is not True or effects["bootstrap_seed"] != 2026090800 + number
            or effects["primary_confidence_each"] != .975 or effects["primary_block_days"] != 7
            or effects["bootstrap_replicates"] != 4000 or set(effects["sources"]) != {"bike", "household"}):
        raise AssertionError("Each block must retain its own fixed inferential contract")
    sources = {}
    for source in ("bike", "household"):
        means, seed_scores = {}, {}
        for procedure in ("SORT", "QCAL"):
            seed_scores[procedure] = {}
            for role in ("F0", "H", "OFF_LORA", "RAW"):
                selected = sorted((r for r in rows if (r["source"], r["role"], r["procedure"]) == (source, role, procedure)),
                                  key=lambda row: int(row["seed"]))
                seed_scores[procedure][role] = [float(r["score"]) for r in selected]
            means[procedure] = {role: float(np.mean(values)) for role, values in seed_scores[procedure].items()}
        if any(not np.isfinite(v) or v <= 0 for values in means.values() for v in values.values()):
            raise AssertionError("Block comparison requires finite positive proper scores")
        selected_effect = effects["sources"][source]
        for procedure, key, confidence in (("QCAL", "primary", .975), ("SORT", "sort_effect", .95)):
            scores = seed_scores[procedure]
            deltas = (np.asarray(scores["H"]) - np.asarray(scores["OFF_LORA"])) / scores["F0"][0]
            value = selected_effect[key]
            if (value["confidence"] != confidence or value["block_days"] != 7
                    or not np.allclose(deltas, value["seed_values"], rtol=1e-10, atol=1e-12)
                    or not np.isclose(deltas.mean(), value["value"], rtol=1e-10, atol=1e-12)):
                raise AssertionError("A separate block effect disagrees with its own scores")
        sources[source] = {"score_means": means, "seed_scores": seed_scores,
                           "primary": selected_effect["primary"], "sort_effect": selected_effect["sort_effect"],
                           "raw_veto": selected_effect["raw_veto"],
                           "selected_head_family": selected_effect["selected_head_family"],
                           "selected_head_lr": selected_effect["selected_head_lr"],
                           "selected_lora_lr": selected_effect["selected_lora_lr"]}
    return {"study_number": number, "sources": sources, "overall_gate": effects["overall_gate"],
            "bootstrap_seed": effects["bootstrap_seed"], "selections_refitted_within_this_block": True}


def build_comparison(root, temporal_audits):
    root = Path(root).resolve()
    blocks, inputs = {}, {}
    for number, study in ((12, PARENT_STUDY), (13, STUDY)):
        folder = root / "results" / study
        paths = [folder / "selected_results.csv", folder / "effects.json"]
        if number == 12:
            verification = read_json(folder / "verification.json")
            if verification.get("passed") is not True or verification.get("completed") is not True:
                raise AssertionError("The original temporal block must have a passed verification")
            for path in paths:
                if sha256_file(path) != verification["output_hashes"][path.name]:
                    raise AssertionError("An original comparison input changed after verification")
            inputs[str(folder / "verification.json")] = sha256_file(folder / "verification.json")
        inputs.update({str(path): sha256_file(path) for path in paths})
        blocks[str(number)] = summarize_block(read_csv(paths[0]), read_json(paths[1]), number)
    if set(temporal_audits) != {"bike", "household"}:
        raise AssertionError("Both fixed calendar blocks require an independent timestamp audit")
    return {"completed": True, "blocks": blocks, "calendar_audits": temporal_audits,
            "input_hashes": inputs, "pooled_estimate_computed": False, "pooled_confidence_interval_computed": False,
            "cross_block_difference_test_computed": False,
            "scope": "Same two sources, next nonoverlapping time block; separate selections, calibration, scores and conditional intervals. No new-source replication or pooled significance.",
            "interpretation": "The original block's inconclusive source remains inconclusive; a later positive result does not revise it. Vanilla LoRA benefit alone does not establish a need for a new PEFT method."}


def run(root):
    root = Path(root).resolve()
    own_path = Path(__file__).resolve()
    own_sha = sha256_file(own_path)
    _temporal_audits.clear()
    original_write = _core.write_json

    def verified_write(path, value):
        if Path(path).name == "effects.json":
            value.update({"study": STUDY, "temporal_block_index": 1,
                          "scope": "Next fixed time block of the same two real sources. Separate conditional temporal bootstrap; no pooling of blocks 12/13 and no whole selection/training uncertainty."})
        if Path(path).name == "verification.json":
            frozen = read_json(root / "runs" / STUDY / "study_contract.json")
            _core.verify_file_hashes(root, frozen["parent_artifacts"], set())
            if sha256_file(own_path) != own_sha:
                raise AssertionError("Temporal analysis wrapper changed during execution")
            comparison = build_comparison(root, dict(_temporal_audits))
            comparison_path = Path(path).parent / "temporal_comparison.json"
            original_write(comparison_path, comparison)
            value["output_hashes"][comparison_path.name] = sha256_file(comparison_path)
            value["analysis_sources"][str(own_path)] = own_sha
            value.update({"wrapped_completed": True, "temporal_block_index": 1,
                          "bootstrap_seed": BOOTSTRAP_SEED, "parent_artifacts_unchanged": True,
                          "parent_artifacts": frozen["parent_artifacts"], "temporal_audits": dict(_temporal_audits),
                          "comparison_without_pooling": True})
            value["checks"].extend(["isolated unchanged numerical analysis with new bootstrap default seed",
                                    "both temporal trainer completion/protection gates and new-block caches",
                                    "same raw sources; adjacent 190-day blocks with no timestamp overlap",
                                    "separate original/replication estimates; no pooled CI or cross-block significance test"])
        return original_write(path, value)

    _core.write_json = verified_write
    try:
        return _core.run(root)
    finally:
        _core.write_json = original_write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    run(parser.parse_args().root)


if __name__ == "__main__":
    main()
