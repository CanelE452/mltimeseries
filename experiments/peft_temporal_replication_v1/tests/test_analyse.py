import copy
import csv
import importlib
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.peft_temporal_replication_v1 import analyse, data


def test_private_numerical_module_and_new_default_seed_leave_original_unchanged():
    original = importlib.import_module("experiments.peft_external_gap_v1.analyse")
    assert analyse._core is not original
    assert analyse._core.__package__ == "experiments.peft_temporal_replication_v1"
    assert original.STUDY == "peft_external_gap_v1"
    assert original.BOOTSTRAP_SEED == 2026090812
    actual = analyse._core.moving_block_weights(83, 7, replicates=29)
    expected = original.moving_block_weights(83, 7, replicates=29, seed=2026090813)
    assert np.array_equal(actual, expected)
    assert not np.array_equal(actual, original.moving_block_weights(83, 7, replicates=29))
    assert np.array_equal(actual.sum(axis=1), np.full(29, 83))


def test_reused_numerical_functions_match_original_with_missing_targets():
    original = importlib.import_module("experiments.peft_external_gap_v1.analyse")
    rng = np.random.default_rng(5)
    target = rng.normal(size=(19, 2, 4))
    target[::2, 0, 1] = np.nan
    q = np.array([.1, .5, .9])
    prediction = np.sort(rng.normal(size=(19, 2, 3, 4)), axis=2)
    scale = np.array([2., 7.])
    left = analyse.score_prediction(prediction, target, scale, q)
    right = original.score_prediction(prediction, target, scale, q)
    assert all(np.array_equal(a, b) for a, b in zip(left, right))
    assert np.array_equal(analyse.qcal_offsets(prediction, target, q), original.qcal_offsets(prediction, target, q))


def temporal_fixture():
    bounds = copy.deepcopy(data.EXPECTED_BOUNDARIES["bike"])
    current_start = np.datetime64(bounds["precontext"][0], "s")
    current = current_start + np.arange(190 * 24) * np.timedelta64(1, "h")
    previous = current - np.timedelta64(190, "D")
    old_bounds = {key: [str(np.datetime64(value, "s") - np.timedelta64(190, "D")) for value in interval]
                  for key, interval in bounds.items()}
    replication = {"replication_index": 1, "offset_days_from_first_complete_day": 190,
                   "block_days": 190, "raw_timestep_overlap_with_previous_block": 0,
                   "replication_block": [bounds["precontext"][0], bounds["eval"][1]],
                   "previous_block": [old_bounds["precontext"][0], old_bounds["eval"][1]],
                   "source_start_index": 4560, "source_end_index_exclusive": 9120}
    metadata = {"version": data.VERSION, "contract": {"boundaries": bounds},
                "source": {"source_sha256": "source", "temporal_replication": replication}}
    old = {"contract": {"boundaries": old_bounds}, "source": {"source_sha256": "source"}}
    return metadata, current, old, previous


def test_adjacent_actual_timestamp_blocks_pass_and_overlap_is_rejected():
    meta, timestamps, old, previous = temporal_fixture()
    result = analyse.verify_temporal_layout("bike", meta, timestamps, old, previous)
    assert result["raw_timestamp_overlap"] == 0
    assert result["new_time_block_not_new_source"] is True
    with pytest.raises(AssertionError, match="after the old evaluation"):
        analyse.verify_temporal_layout("bike", meta, timestamps, old, previous + np.timedelta64(1, "h"))


@pytest.mark.parametrize("change", ["calendar", "source", "offset", "length"])
def test_temporal_layout_rejects_stale_metadata_or_different_source(change):
    meta, timestamps, old, previous = temporal_fixture()
    if change == "calendar":
        meta["contract"]["boundaries"]["eval"][0] = "2011-10-25T00:00:00"
    elif change == "source":
        old["source"]["source_sha256"] = "other-source"
    elif change == "offset":
        meta["source"]["temporal_replication"]["offset_days_from_first_complete_day"] = 191
    else:
        timestamps = timestamps[:-1]
    with pytest.raises(AssertionError):
        analyse.verify_temporal_layout("bike", meta, timestamps, old, previous)


def wrapper_fixture(root):
    paths = [root / "experiments" / study / "train.py" for study in (analyse.STUDY, analyse.PARENT_STUDY)]
    paths += [root / "runs" / analyse.STUDY / "study_contract.json",
              root / "results" / analyse.PARENT_STUDY / "verification.json"]
    for index, path in enumerate(paths):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(index), encoding="utf-8")
    consumer, producer, contract, previous = paths
    parent_artifacts = {str(previous): analyse.sha256_file(previous)}
    frozen = {"previous_contract_sha256": "previous-contract", "parent_artifacts": parent_artifacts}
    wrapper = {"study": analyse.STUDY, "temporal_block_index": 1, "offset_days": 190,
               "fresh_native_initialization": True, "same_full_selection_rule": True,
               "source_sha256": analyse.sha256_file(consumer),
               "numerical_producer_source_sha256": analyse.sha256_file(producer),
               "numerical_producer_path": str(producer), "study_contract_path": str(contract),
               "study_contract_sha256": analyse.sha256_file(contract),
               "previous_contract_sha256": "previous-contract", "previous_verification_sha256": analyse.sha256_file(previous),
               "parent_artifacts": parent_artifacts}
    audits = {key: True for key in ("parent_verified_artifacts_unchanged", "parent_production_caches_unchanged",
              "numerical_source_unchanged", "consumer_source_unchanged", "isolated_module_namespace", "completion_gate_passed")}
    meta = {"completed": True, "inner_training_completed": True, "wrapped_completed": True,
            "wrapper_contract": wrapper, "wrapper_audits": audits,
            "protected_hashes": {str(p): analyse.sha256_file(p) for p in paths}}
    return meta, frozen


def test_wrapper_requires_completion_and_original_cache_preservation_audit(tmp_path):
    meta, frozen = wrapper_fixture(tmp_path)
    analyse.verify_wrapper(tmp_path, meta, frozen, set())
    broken = copy.deepcopy(meta)
    broken["wrapped_completed"] = False
    with pytest.raises(AssertionError, match="both complete"):
        analyse.verify_wrapper(tmp_path, broken, frozen, set())
    broken = copy.deepcopy(meta)
    broken["wrapper_audits"]["parent_production_caches_unchanged"] = False
    with pytest.raises(AssertionError, match="parent_production_caches"):
        analyse.verify_wrapper(tmp_path, broken, frozen, set())
    broken = copy.deepcopy(meta)
    broken["protected_hashes"].pop(next(iter(frozen["parent_artifacts"])))
    with pytest.raises(AssertionError, match="omitted a protected"):
        analyse.verify_wrapper(tmp_path, broken, frozen, set())


def result_fixture(number):
    rows, sources = [], {}
    for source in ("bike", "household"):
        h, lora = ([1., 1.01, .99], [.9, .91, .89]) if number == 12 else ([1., 1.01, .99], [1.1, 1.11, 1.09])
        scores = {"F0": [1.], "H": h, "OFF_LORA": lora, "RAW": [1.2]}
        for role, values in scores.items():
            seeds = [-1] if role == "RAW" else list(range(12000, 12000 + len(values)))
            for seed, score in zip(seeds, values):
                for procedure in ("SORT", "QCAL"):
                    rows.append({"source": source, "role": role, "seed": seed, "procedure": procedure, "score": score})
        delta = (np.array(h) - np.array(lora)).tolist()
        effect = {"confidence": .975, "block_days": 7, "value": float(np.mean(delta)), "seed_values": delta,
                  "ci": [float(np.mean(delta)) - .02, float(np.mean(delta)) + .02], "decision": "fixture"}
        sources[source] = {"primary": effect, "sort_effect": {**effect, "confidence": .95}, "raw_veto": {"applies": False},
                           "selected_head_family": "H_FULL", "selected_head_lr": 3e-5, "selected_lora_lr": 1e-4}
    effects = {"completed": True, "bootstrap_seed": 2026090800 + number, "primary_confidence_each": .975,
               "primary_block_days": 7, "bootstrap_replicates": 4000, "sources": sources, "overall_gate": f"block_{number}"}
    return rows, effects


def test_separate_comparison_retains_opposite_effects_without_pooling(tmp_path):
    for number, study in ((12, analyse.PARENT_STUDY), (13, analyse.STUDY)):
        folder = tmp_path / "results" / study
        folder.mkdir(parents=True)
        rows, effects = result_fixture(number)
        with (folder / "selected_results.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        analyse.write_json(folder / "effects.json", effects)
        analyse.write_json(folder / "verification.json", {"passed": True, "completed": True,
                           "output_hashes": {name: analyse.sha256_file(folder / name) for name in ("selected_results.csv", "effects.json")}})
    comparison = analyse.build_comparison(tmp_path, {"bike": {}, "household": {}})
    assert comparison["blocks"]["12"]["sources"]["bike"]["primary"]["value"] > 0
    assert comparison["blocks"]["13"]["sources"]["bike"]["primary"]["value"] < 0
    assert comparison["blocks"]["12"]["overall_gate"] == "block_12"
    assert comparison["blocks"]["13"]["overall_gate"] == "block_13"
    assert comparison["pooled_estimate_computed"] is False
    assert comparison["pooled_confidence_interval_computed"] is False
    assert comparison["cross_block_difference_test_computed"] is False


def test_comparison_rejects_partial_duplicate_or_mislabelled_bootstrap():
    rows, effects = result_fixture(13)
    for altered in (rows[:-1], rows[:-1] + rows[:1]):
        with pytest.raises(AssertionError, match="complete 32-row"):
            analyse.summarize_block(altered, effects, 13)
    effects["bootstrap_seed"] = 2026090812
    with pytest.raises(AssertionError, match="inferential contract"):
        analyse.summarize_block(rows, effects, 13)


def test_production_gate_fails_before_opening_holdout_and_restores_private_writer(tmp_path, monkeypatch):
    study = tmp_path / "runs" / analyse.STUDY
    study.mkdir(parents=True)
    analyse.write_json(study / "completed.json", {"completed": False})
    analyse.write_json(study / "fit_completed.json", {"completed": False})
    def forbidden_load(*args, **kwargs):
        pytest.fail("An incomplete production run opened a numerical archive")
    monkeypatch.setattr(analyse.np, "load", forbidden_load)
    before = analyse._core.write_json
    with pytest.raises(AssertionError, match="all 28 fits"):
        analyse.run(tmp_path)
    assert analyse._core.write_json is before
    assert not (tmp_path / "results").exists()


def test_existing_partial_result_is_preserved(tmp_path):
    output = tmp_path / "results" / analyse.STUDY
    output.mkdir(parents=True)
    previous = output / "effects.json"
    previous.write_text("partial", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Preserve previous"):
        analyse.run(tmp_path)
    assert previous.read_text(encoding="utf-8") == "partial"
