import copy
from datetime import datetime, timedelta, timezone
import json
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.peft_selection_regret_v1 import analyse


def test_selection_harm_sign_and_f0_ratio_with_unequal_observation_counts():
    counts = np.tile([21., 2100.], (83, 1))
    baseline = counts * [1., 100.]
    selected = baseline + counts * [.1, 0.]
    effect = analyse.paired_effect(selected, baseline, baseline, counts)
    expected = .05 / 50.5
    assert effect["value"] == pytest.approx(expected)
    assert effect["ci"] == pytest.approx([expected, expected])
    assert effect["confidence"] == .9875
    assert effect["bootstrap_seed"] == 2026090814
    assert effect["requested_resamples"] == effect["valid_resamples"] == 4000


def test_identical_selected_candidate_has_zero_regret_for_every_block_draw():
    counts = np.ones((83, 2))
    f0 = np.arange(1, 167).reshape(83, 2)
    value = analyse.paired_effect(f0, f0, f0, counts)
    assert value["value"] == 0
    assert value["ci"] == [0, 0]
    assert value["decision"] == "conditional_practical_equivalence"


def test_independent_index_bootstrap_matches_weighted_implementation():
    rng = np.random.default_rng(44)
    counts = rng.integers(1, 35, (83, 2)).astype(float)
    f0 = counts * rng.uniform(.1, 2, (83, 2))
    low = counts * rng.uniform(.1, 2, (83, 2))
    selected = counts * rng.uniform(.1, 2, (83, 2))
    actual = analyse.paired_effect(selected, low, f0, counts)
    rng = np.random.default_rng(2026090814)
    starts = rng.integers(0, 77, (4000, 12))
    indices = (starts[:, :, None] + np.arange(7)).reshape(4000, -1)[:, :83]
    sampled_counts = counts[indices].sum(axis=1)
    delta = ((selected[indices].sum(axis=1) - low[indices].sum(axis=1)) / sampled_counts).mean(axis=1)
    denominator = (f0[indices].sum(axis=1) / sampled_counts).mean(axis=1)
    expected = np.quantile(delta / denominator, [.00625, .99375])
    assert actual["ci"] == pytest.approx(expected, abs=1e-14)


def test_nonoverlapping_subsets_preserve_requested_calendar_block_scale():
    counts = np.ones((83, 2))
    baseline = np.ones((83, 2))
    result = analyse.comparison_effects(baseline * 1.03, baseline, baseline, counts)
    for parity, size in (("even", 42), ("odd", 41)):
        for days, block_origins, span in ((3, 2, 4), (7, 4, 8), (14, 7, 14)):
            effect = result["nonoverlapping"][parity][str(days)]
            assert effect["origin_count"] == size
            assert effect["stride_days"] == 2
            assert effect["requested_block_days"] == days
            assert effect["block_origins"] == block_origins
            assert effect["target_span_days"] == span
            assert effect["value"] == pytest.approx(.03)


def gate_fixture():
    sources = {"12_bike": "bike", "13_bike": "bike", "12_household": "household", "13_household": "household"}
    effects, scores = {}, {}
    for cell in sources:
        opposite = cell == "13_bike"
        ci, value = ([-.04, -.02], -.03) if opposite else ([.02, .04], .03)
        effect = {"daily": {"7": {"ci": ci, "value": value}},
                  "nonoverlapping": {parity: {"7": {"value": value}} for parity in ("even", "odd")}}
        recent = {"daily": {"7": {"ci": [.01, .05], "value": .04}},
                  "nonoverlapping": {parity: {"7": {"value": .04}} for parity in ("even", "odd")}}
        effects[cell] = {"SORT": {"LORA_V_vs_FIXED_LOW": effect, "LORA_RECENT7_vs_FIXED_LOW": recent}}
        scores[cell] = {rule: 1.1 for rule in analyse.SELECTORS}
        scores[cell].update({"F0": 1.1, "LORA_V": 1. if opposite else 1.03, "FIXED_LOW": 1.03 if opposite else 1.})
    return effects, scores, sources


def test_gate_requires_cross_source_repetition_and_reverse_harm():
    effects, scores, sources = gate_fixture()
    assert analyse.selection_gate(effects, scores, sources)["decision"] == "enter_bounded_selection_mechanism_research"
    for cell in ("12_household", "13_household"):
        effects[cell]["SORT"]["LORA_V_vs_FIXED_LOW"]["daily"]["7"]["ci"][0] = .005
    assert analyse.selection_gate(effects, scores, sources)["decision"] == "close_current_B_branch"
    effects, scores, sources = gate_fixture()
    effects["13_bike"]["SORT"]["LORA_V_vs_FIXED_LOW"]["daily"]["7"]["ci"][1] = -.005
    assert analyse.selection_gate(effects, scores, sources)["opposite_fixed_low_harm_cells"] == []
    assert analyse.selection_gate(effects, scores, sources)["decision"] == "close_current_B_branch"


def test_simple_rule_veto_blocks_entry_without_claiming_equivalence():
    effects, scores, sources = gate_fixture()
    for cell in scores:
        scores[cell]["ALL_RECENT7"] = min(scores[cell]["LORA_V"], scores[cell]["FIXED_LOW"]) + .005
    result = analyse.selection_gate(effects, scores, sources)
    assert result["decision"] == "close_current_B_branch"
    assert result["simple_rule_veto"] is True
    assert "ALL_RECENT7" in result["simple_rule_veto_rules"]
    assert result["simple_rule_comparison_is_descriptive"] is True


def test_recent7_and_both_nonoverlap_signs_must_fail_in_same_positive_cells():
    effects, scores, sources = gate_fixture()
    for cell in ("12_household", "13_household"):
        effects[cell]["SORT"]["LORA_RECENT7_vs_FIXED_LOW"]["nonoverlapping"]["odd"]["7"]["value"] = 0
    result = analyse.selection_gate(effects, scores, sources)
    assert result["decision"] == "close_current_B_branch"
    assert result["harm_repeats_across_both_sources"] is False


def test_hindsight_oracles_keep_family_and_all_universes_without_confidence_interval():
    candidates = [{"candidate_id": "F0", "method": "F0"}]
    for method in ("OFF_LORA", "H_FULL", "H_MLP"):
        candidates.extend({"candidate_id": f"{method}_{i}", "method": method} for i in range(3))
    scores = {c["candidate_id"]: 1.2 for c in candidates}
    scores.update({"F0": 1., "OFF_LORA_0": .9, "H_FULL_0": .8})
    selectors = {"F0": "F0", "FIXED_LOW": "OFF_LORA_0", "LORA_V": "OFF_LORA_1",
                 "LORA_RECENT7": "OFF_LORA_2", "HEAD_V": "H_MLP_0", "ALL_V": "H_FULL_1", "ALL_RECENT7": "F0"}
    result = analyse.oracle_diagnostics(scores, candidates, selectors, 1.)
    assert result["oracles"]["LORA"]["candidate_id"] == "OFF_LORA_0"
    assert result["oracles"]["HEAD"]["candidate_id"] == result["oracles"]["ALL"]["candidate_id"] == "H_FULL_0"
    assert result["selectors"]["FIXED_LOW"]["family_regret_over_f0"] == 0
    assert result["selectors"]["FIXED_LOW"]["all_regret_over_f0"] == pytest.approx(.1)
    assert result["confidence_intervals_computed"] is False


def write_guard(path, index=1, smoke=False, completed=True):
    guard = path / "guard"
    guard.mkdir(parents=True, exist_ok=True)
    start = datetime(2026, 9, 8, tzinfo=timezone.utc) + timedelta(minutes=index)
    status = {"completed": completed, "returncode": 0 if completed else 1,
              "reasons": [] if completed else ["git_process_count_above_limit"],
              "state": "completed" if completed else "safety_stop", "guard_pid": index,
              "started_at": start.isoformat(), "finished_at": (start + timedelta(seconds=1 if completed else 6.125)).isoformat(),
              "elapsed_seconds": 1. if completed else 6.125,
              "command": ["python", "-m", f"experiments.{analyse.STUDY}.forecast"] + (["--smoke"] if smoke else [])}
    analyse.write_json(guard / "status.json", status)
    sample = {"timestamp": start.isoformat(), "available_ram_gib": 10., "available_commit_gib": 10.,
              "child_tree_rss_gib": .5, "git_process_count": 0 if completed else 56, "gpus": None}
    (guard / "resource_log.jsonl").write_text(json.dumps(sample) + "\n" + json.dumps({"event": "finish"}) + "\n", encoding="utf-8")
    return status


def reused_forecast_fixture(root, number=12):
    study = "peft_external_gap_v1" if number == 12 else "peft_temporal_replication_v1"
    fit_path = root / "runs" / study / "trials/bike/F0/lr_0e+00_seed_12000"
    path = root / "runs" / study / "forecasts/bike/F0_seed_12000"
    fit_path.mkdir(parents=True)
    path.mkdir(parents=True)
    source = root / "native_source.py"
    source.write_text("verified native source", encoding="utf-8")
    native = {str(source): analyse.sha(source)}
    analyse.write_json(fit_path / "result.json", {"native_source_hashes": native})
    q = np.concatenate(([.01], np.arange(.05, 1, .05), [.99]))
    origins = {"cal": 336+np.arange(13)*24, "eval": 672+np.arange(83)*24}
    timestamps = np.datetime64("2020-01-01T00:00:00") + np.arange(3000)*np.timedelta64(1, "h")
    targets = {split: np.ones((len(values), 2, 48), np.float32) for split, values in origins.items()}
    targets["eval"][0, 0, 0] = np.nan
    panel = SimpleNamespace(target_indices=np.array([0, 1]), channels=["y0", "y1", "x0", "x1", "x2"],
                            fit_std=np.ones(5), quantiles=q, stats_hash="stats", origins=origins,
                            timestamps=timestamps, targets=lambda split: targets[split])
    arrays = {"quantiles": q, "target_indices": panel.target_indices, "target_channels": np.array(["y0", "y1"])}
    for split in origins:
        pred = np.broadcast_to(q[None, None, :, None], (len(origins[split]), 2, 21, 48)).astype(np.float32).copy()
        arrays.update({f"{split}_predictions": pred, f"{split}_unsorted_predictions": pred.copy(),
                       f"{split}_target": targets[split], f"{split}_origins": origins[split], f"{split}_timestamps": timestamps[origins[split]]})
    np.savez_compressed(path / "predictions.npz", **arrays)
    meta = {"completed": True, "stage": "forecast", "dataset": "bike", "method": "F0", "seed": 12000, "lr": 0.,
            "fit_result_sha256": analyse.sha(fit_path / "result.json"), "fit_checkpoint_sha256": "checkpoint",
            "fit_data_sha256": "fitdata", "holdout_data_sha256": "holdout", "restored_adaptation_sha256": "parameters",
            "checkpoint_reload_verified": True, "model_unchanged": True, "optimizer_steps": 0,
            "predictions_sha256": analyse.sha(path / "predictions.npz"), "selection_sha256": "old-selection",
            "stats_sha256": panel.stats_hash, "target_indices": [0, 1], "channels": panel.channels,
            "protected_hashes": native, "source_hashes": native}
    if number == 13:
        meta["wrapped_completed"] = True
    assert "native_source_hashes" not in meta
    analyse.write_json(path / "result.json", meta)
    analyse.write_json(path / "trial_contract.json", {"completed": True})
    write_guard(path)
    candidate = {"candidate_id": f"s{number}_bike_F0", "dataset": "bike", "method": "F0", "seed": 12000, "lr": 0.,
                 "needs_forecast": False, "fit_result_path": str(fit_path / "result.json"),
                 "fit_result_sha256": meta["fit_result_sha256"], "checkpoint_sha256": "checkpoint",
                 "restored_adaptation_sha256": "parameters",
                 "reuse_forecast": {"path": str(path), "result_sha256": analyse.sha(path / "result.json"),
                                    "predictions_sha256": meta["predictions_sha256"]}}
    cell = {"study_number": number, "fit_data_sha256": "fitdata", "holdout_data_sha256": "holdout", "selection_json_sha256": "old-selection"}
    return cell, candidate, path, panel, meta, arrays


@pytest.mark.parametrize("number", [12, 13])
def test_actual_reused_schema_without_native_source_field_loads_npz_and_checks_fit_native_source(tmp_path, number):
    cell, candidate, path, panel, expected, original_arrays = reused_forecast_fixture(tmp_path, number)
    meta, arrays = analyse.verify_forecast(tmp_path, cell, candidate, path, panel, {}, set(), {})
    assert "native_source_hashes" not in meta
    assert np.array_equal(arrays["eval_target"], original_arrays["eval_target"], equal_nan=True)
    (tmp_path / "native_source.py").write_text("changed native source", encoding="utf-8")
    with pytest.raises(AssertionError, match="hash"):
        analyse.verify_forecast(tmp_path, cell, candidate, path, panel, {}, set(), {})


def test_reused_forecast_cannot_impute_missing_target_even_if_prediction_hash_is_updated(tmp_path):
    cell, candidate, path, panel, meta, arrays = reused_forecast_fixture(tmp_path)
    arrays["eval_target"] = arrays["eval_target"].copy()
    arrays["eval_target"][0, 0, 0] = 1.
    np.savez_compressed(path / "predictions.npz", **arrays)
    meta["predictions_sha256"] = analyse.sha(path / "predictions.npz")
    analyse.write_json(path / "result.json", meta)
    candidate["reuse_forecast"].update({"result_sha256": analyse.sha(path / "result.json"), "predictions_sha256": meta["predictions_sha256"]})
    with pytest.raises(AssertionError, match="missing mask"):
        analyse.verify_forecast(tmp_path, cell, candidate, path, panel, {}, set(), {})


def test_attempt_inventory_counts_preserved_failure_and_does_not_report_zero_safety_stops(tmp_path):
    study = tmp_path / "runs" / analyse.STUDY
    study.mkdir(parents=True)
    analyse.write_json(study / "selection_contract.json", {"frozen": True})
    attempts = []
    for index in range(33):
        smoke, completed = index >= 29, index != 28
        path = study / ("smoke" if smoke else "failures" if not completed else "forecasts") / str(index)
        status = write_guard(path, index, smoke=smoke, completed=completed)
        attempts.append({"guard_status": status})
    invocation = study / "invocations/1.json"
    invocation.parent.mkdir()
    analyse.write_json(invocation, {"selection_contract_sha256": analyse.sha(study / "selection_contract.json"),
                       "finished_at": 10., "wall_seconds": 10., "attempts": attempts})
    audit = analyse.attempt_audit(tmp_path, {})
    production = audit["phases"]["production"]
    assert production["successful_attempts"] == 28
    assert production["attempts"] == 29
    assert production["unsuccessful_guard_seconds"] == 6.125
    assert production["all_attempt_guard_seconds"] == 34.125
    assert audit["phases"]["s0"]["attempts"] == 4
    resources = analyse.summarize_attempt_resources([r for r in audit["attempt_records"] if not r["smoke"]], "fixture")
    assert resources["max_git_process_count"] == 56
    assert resources["safety_stops"] == resources["guard_failures"] == 1
    assert resources["max_gpu_memory_used_mib"] is None


def test_production_gate_precedes_future_archive_loading(tmp_path, monkeypatch):
    study = tmp_path / "runs" / analyse.STUDY
    study.mkdir(parents=True)
    analyse.write_json(study / "completed.json", {"completed": False})
    def forbidden(*args, **kwargs):
        pytest.fail("Incomplete production opened a numerical archive")
    monkeypatch.setattr(analyse.np, "load", forbidden)
    with pytest.raises(AssertionError, match="All 28"):
        analyse.run(tmp_path)
    assert not (tmp_path / "results").exists()
