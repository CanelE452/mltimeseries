import copy
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.peft_trainlag_v1 import analyse, data


def entries_fixture(root):
    trials = []
    for c, method, lr in sorted(analyse.expected_trial_keys()):
        trials.append({"condition": "Q00", "corpus": c, "method": method,
                       "input_mode": "aligned" if method.startswith("ALIGN_") else "raw", "lr": lr,
                       "path": f"runs/{analyse.STUDY}/trials/Q00_c{c}/{method}/lr_{lr:.0e}",
                       "val_score": .5 + (lr if c == 0 else -lr)})
    choices = {method: {"lr": 3e-5, "candidates": [row.copy() for row in trials if row["corpus"] == 0
                                                   and row["method"] == method]}
               for method in analyse.TRAINED}
    selected = [row.copy() for row in trials if row["method"] not in analyse.TRAINED or row["lr"] == 3e-5]
    return trials, selected, choices


def test_exact_grid_and_corpus0_lr_selection(tmp_path):
    trials, selected, choices = entries_fixture(tmp_path)
    assert len(analyse.verify_entries(trials, selected, choices, tmp_path)) == 18
    with pytest.raises(AssertionError, match="18 unique"):
        analyse.verify_entries(trials[:-1], selected, choices, tmp_path)
    with pytest.raises(AssertionError, match="18 unique"):
        analyse.verify_entries(trials[:-1] + [trials[0]], selected, choices, tmp_path)
    bad = copy.deepcopy(choices)
    bad["ATTN"]["candidates"][0]["val_score"] += .1
    with pytest.raises(AssertionError, match="exactly reproduce"):
        analyse.verify_entries(trials, selected, bad, tmp_path)
    bad = copy.deepcopy(choices)
    bad["ATTN"]["lr"] = 1e-4
    with pytest.raises(AssertionError, match="validation argmin"):
        analyse.verify_entries(trials, selected, bad, tmp_path)


def test_selected_repetitions_cannot_reselect_using_their_own_validation(tmp_path):
    trials, selected, choices = entries_fixture(tmp_path)
    bad = [row for row in selected if not (row["corpus"] == 1 and row["method"] == "ATTN")]
    bad.append(next(row for row in trials if row["corpus"] == 1 and row["method"] == "ATTN" and row["lr"] == 1e-4))
    with pytest.raises(AssertionError, match="fixed corpus0"):
        analyse.verify_entries(trials, bad, choices, tmp_path)
    with pytest.raises(AssertionError, match="12 selected"):
        analyse.verify_entries(trials, selected[:-1], choices, tmp_path)


def test_ratio_bootstrap_recomputes_shared_denominator_and_preserves_corpus_pairing():
    numerator = np.array([[1., 3.], [2., 6.], [3., 9.]])
    denominator = np.array([[2., 12.], [4., 24.], [6., 36.]])
    weights = np.array([[1., 0.], [0., 1.], [.5, .5]])
    result = analyse.effect_ratio(numerator, denominator, weights)
    expected_draws = np.array([.5, .25, 2 / 7])
    np.testing.assert_allclose(result["ci"], np.quantile(expected_draws, [.0125, .9875]))
    assert result["value"] == pytest.approx(2 / 7)
    assert result["value"] != pytest.approx((numerator / denominator).mean())
    np.testing.assert_allclose(result["corpus_values"], [2 / 7] * 3)


def test_primary_sign_family_confidence_and_strict_practical_gate():
    losses = {}
    for c in range(3):
        for method, value in {"F0": 1., "ATTN": .9, "ALIGN_F0": .8, "ALIGN_ATTN": .7}.items():
            losses[(c, method, 3e-5 if method in analyse.TRAINED else .001)] = np.full(6, value)
    family = analyse.contrast_family(losses, {method: 3e-5 for method in analyse.TRAINED}, np.ones((5, 6)) / 6)
    for effect in family["effects"].values():
        assert effect["value"] == pytest.approx(.1)
        assert effect["confidence"] == .975
        assert effect["decision"] == "repeated_practical_improvement"
    assert analyse.classify_effect({"ci": [-.02, .02], "corpus_values": [0., 0., 0.]}) == "inconclusive"
    assert analyse.classify_effect({"ci": [-.009, .009], "corpus_values": [1., -1., 0.]}) == "conditional_practical_equivalence"
    assert analyse.classify_effect({"ci": [.011, .02], "corpus_values": [.03, .04, -.01]}) == "inconclusive"


def test_train_selector_replay_rejects_unselected_candidate_score_or_input_change():
    arrays = data.build_archive(0)
    selection = data.select_lags_from_train(arrays["context_train"], arrays["target_train"])
    selection["data_sha256"] = "fixture-data"
    analyse.verify_lag_selection(arrays, selection, "fixture-data")
    tampered = copy.deepcopy(selection)
    tampered["scores"]["U"][0] += 1e-6
    with pytest.raises(AssertionError, match="replay mismatch"):
        analyse.verify_lag_selection(arrays, tampered, "fixture-data")
    tampered = {**arrays, "context_train": arrays["context_train"].copy()}
    tampered["context_train"][0, 0, 0] += 1
    with pytest.raises(AssertionError, match="hash"):
        analyse.verify_lag_selection(tampered, selection, "fixture-data")
    bad_features = {**arrays, "raw_features_eval": arrays["raw_features_eval"].copy()}
    bad_features["raw_features_eval"][0, 0, 0] += 1
    with pytest.raises(AssertionError, match="past-index"):
        analyse.verify_lag_selection(bad_features, selection, "fixture-data")


def test_alignment_scope_rejects_original_future_observation():
    lags = {"Y": 32, "U": 48, "V": 64}
    scope = {"input_mode": "aligned", "selected_lags": lags, "Y_past_unchanged": True,
             "Y_future_mask": 0, "additional_original_future_observations": 0, "Y_lag_used_by_FM": False,
             "driver_alignment": {name: {"lag": lag, "retained_past_source_range": [0, 255-lag],
                 "future_source_indices": (256 + np.arange(16) - lag).tolist(), "padding": "NaN",
                 "original_recent_values_dropped": lag-16} for name, lag in lags.items() if name != "Y"}}
    analyse.verify_input_scope(scope, "aligned", lags)
    scope["driver_alignment"]["U"]["future_source_indices"][-1] = 256
    with pytest.raises(AssertionError, match="past indices"):
        analyse.verify_input_scope(scope, "aligned", lags)


def test_partial_completed_flag_is_not_enough_to_allow_evaluation(tmp_path, monkeypatch):
    study = tmp_path / "runs" / analyse.STUDY
    study.mkdir(parents=True)
    (study / "completed.json").write_text(json.dumps({"completed": True, "trial_count": 17, "fit_count": 12, "f0_count": 5}))
    monkeypatch.setattr(analyse, "runner_contract", lambda root: pytest.fail("Must reject incomplete study before reading artifacts"))
    with pytest.raises(AssertionError, match="all 18 GPU"):
        analyse.load_completed_study(tmp_path)


def test_guard_finish_row_is_excluded_and_failure_artifact_blocks_success(tmp_path):
    guard = tmp_path / "guard"
    guard.mkdir()
    sample = {"timestamp": "2026-09-08T00:00:00Z", "available_ram_gib": 10., "available_commit_gib": 11.,
              "child_tree_rss_gib": 1., "git_process_count": 2,
              "gpus": [{"memory_used_mib": 1500, "temperature_c": 50}]}
    (guard / "status.json").write_text(json.dumps({"completed": True, "returncode": 0, "reasons": []}))
    (guard / "resource_log.jsonl").write_text(json.dumps(sample) + "\n" + json.dumps({"event": "finish"}) + "\n")
    assert analyse.validate_guard(tmp_path)["completed"]
    assert analyse.resource_samples([tmp_path]) == [sample]
    (tmp_path / "failure.json").write_text("{}")
    with pytest.raises(AssertionError, match="Failure artifact"):
        analyse.validate_guard(tmp_path)


def test_saved_prediction_loss_is_recomputed_and_string_ids_remain_exact(tmp_path):
    q = np.array([.1, .5, .9], dtype=np.float32)
    target = np.arange(8, dtype=np.float32).reshape(2, 4)
    prediction = np.repeat(target[:, None], 3, axis=1)
    arrays, saved = {"quantiles": q}, {"quantiles": q}
    for split in ("val", "eval"):
        ids = np.array([f"trainlag_{split}_a", f"trainlag_{split}_b"])
        arrays.update({f"target_{split}": target, f"episode_ids_{split}": ids})
        saved.update({f"{split}_predictions": prediction, f"{split}_target": target,
                      f"{split}_episode_ids": ids, f"{split}_episode_losses": np.zeros(2)})
    np.savez(tmp_path / "predictions.npz", **saved)
    analyse.verify_prediction(tmp_path, arrays, {"val_score": 0., "eval_score": 0.})
    saved["eval_predictions"] = prediction + 1
    np.savez(tmp_path / "predictions.npz", **saved)
    with pytest.raises(AssertionError, match="episode losses"):
        analyse.verify_prediction(tmp_path, arrays, {"val_score": 0., "eval_score": 0.})


def test_own_checkpoints_may_differ_and_must_match_their_own_hash(tmp_path):
    for name, content in (("raw", b"raw trained adapter"), ("aligned", b"aligned trained adapter")):
        path = tmp_path / name
        path.mkdir()
        checkpoint = path / "best_adaptation.pt"
        checkpoint.write_bytes(content)
        analyse.validate_adaptation_checkpoint_file(path, {"checkpoint_sha256": analyse.sha256_file(checkpoint)})
    with pytest.raises(AssertionError, match="checkpoint hash"):
        analyse.validate_adaptation_checkpoint_file(tmp_path / "aligned", {
            "checkpoint_sha256": analyse.sha256_file(tmp_path / "raw/best_adaptation.pt")})


def test_raw_reproduction_reuse_and_no_partial_overwrite(tmp_path):
    study = tmp_path / "runs" / analyse.STUDY
    data.generate(study / "data")
    for name in ("raw.py", "data.py"):
        source = tmp_path / "experiments" / analyse.STUDY / name
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes((Path(analyse.__file__).parent / name).read_bytes())
    plan = tmp_path / analyse.PLAN
    plan.parent.mkdir(parents=True)
    plan.write_text("Temporary fixture plan; no real study data used")
    arrays, lag, data_path, lag_path = analyse.load_data(study, 0)
    first = analyse.load_or_fit_raw(tmp_path, 0, arrays, lag, data_path, lag_path)
    second = analyse.load_or_fit_raw(tmp_path, 0, arrays, lag, data_path, lag_path)
    np.testing.assert_array_equal(first[2]["eval_predictions"], second[2]["eval_predictions"])
    (study / "raw/Q00_c1").mkdir()
    (study / "raw/Q00_c1/partial.log").write_text("preserve this")
    with pytest.raises(RuntimeError, match="automatic overwrite"):
        analyse.load_or_fit_raw(tmp_path, 1, arrays, lag, data_path, lag_path)
    (first[0] / "predictions.npz").write_bytes(b"changed")
    with pytest.raises(AssertionError, match="artifact changed"):
        analyse.load_or_fit_raw(tmp_path, 0, arrays, lag, data_path, lag_path)


def test_reporting_integration_keeps_hpo_rows_out_of_selected_effects(tmp_path, monkeypatch):
    """Exercise saved prediction/RAW/report aggregation with a CPU-only wrapper stand-in."""
    study = tmp_path / "runs" / analyse.STUDY
    data.generate(study / "data")
    for name in ("raw.py", "data.py"):
        source = tmp_path / "experiments" / analyse.STUDY / name
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes((Path(analyse.__file__).parent / name).read_bytes())
    plan = tmp_path / analyse.PLAN
    plan.parent.mkdir(parents=True)
    plan.write_text("CPU integration fixture")
    trials, selected, choices = entries_fixture(tmp_path)
    datasets = [analyse.load_data(study, c)[0] for c in range(3)]
    metadata = {}
    for entry in trials:
        key = analyse.trial_key(entry)
        c, method, lr = key
        arrays = datasets[c]
        path = tmp_path / entry["path"]
        (path / "guard").mkdir(parents=True)
        shifts = {"F0": 1., "ATTN": .8, "ALIGN_F0": .4, "ALIGN_ATTN": .2}
        prediction = {}
        for split in ("val", "eval"):
            # The unused larger LR looks better on validation but much worse on evaluation.
            offset = shifts[method] + ((-.1 if split == "val" else 3.) if lr == 1e-4 else 0.)
            values = arrays[f"oracle_quantiles_{split}"].astype(np.float64) + offset
            prediction.update({f"{split}_predictions": values, f"{split}_target": arrays[f"target_{split}"],
                f"{split}_episode_ids": arrays[f"episode_ids_{split}"],
                f"{split}_episode_losses": analyse.episode_loss(values, arrays[f"target_{split}"], arrays["quantiles"])})
        np.savez(path / "predictions.npz", quantiles=arrays["quantiles"], **prediction)
        meta = {"val_score": float(prediction["val_episode_losses"].mean()),
                "eval_score": float(prediction["eval_episode_losses"].mean()), "wall_seconds": 1.,
                "wrapper_wall_seconds": 1.1, "peak_cuda_gib": .1, "best_step": 40,
                "trainable_parameters": 1179648 if method in analyse.TRAINED else 0,
                "sampler_sha256": f"paired-{c}", "module_initialization_seeds": {}, "trainable_names": [],
                "audits": {"initial_adaptation_sha256": f"initial-{c}"},
                "wrapper_audits": {"model": {"module_a_initial_sha256": {}, "module_b_initial_sha256": {}}}}
        metadata[key] = meta
        (path / "result.json").write_text(json.dumps(meta))
        (path / "best_adaptation.pt").write_bytes(f"fixture-{key}".encode())
        sample = {"timestamp": "2026-09-08T00:00:00Z", "available_ram_gib": 10.,
                  "available_commit_gib": 11., "child_tree_rss_gib": 1., "git_process_count": 2,
                  "gpus": [{"memory_used_mib": 1500, "temperature_c": 50}]}
        (path / "guard/resource_log.jsonl").write_text(json.dumps(sample) + "\n" + json.dumps({"event": "finish"}) + "\n")
    # Choose rates from the simulated corpus0 validation only, not evaluation.
    for entry in trials:
        entry["val_score"] = metadata[analyse.trial_key(entry)]["val_score"]
        method = entry["method"]
        f0_method = "ALIGN_F0" if method.startswith("ALIGN_") else "F0"
        metadata[analyse.trial_key(entry)]["validation_history"] = [{"step": 0,
            "val_score": metadata[(entry["corpus"], f0_method, .001)]["val_score"]}]
    for method in analyse.TRAINED:
        candidates = [entry.copy() for entry in trials if entry["corpus"] == 0 and entry["method"] == method]
        choices[method] = {"lr": min(candidates, key=lambda row: (row["val_score"], row["lr"]))["lr"], "candidates": candidates}
    selected = [entry.copy() for entry in trials if entry["method"] not in analyse.TRAINED
                or entry["lr"] == choices[entry["method"]]["lr"]]
    lookup = analyse.verify_entries(trials, selected, choices, tmp_path)
    completed = {"trial_guard_seconds": 18., "invocation_wall_seconds": 20.}
    monkeypatch.setattr(analyse, "load_completed_study", lambda root: (study, completed, {}, lookup, selected, choices))
    monkeypatch.setattr(analyse, "runner_contract", lambda root: {})
    monkeypatch.setattr(analyse, "verify_fm_metadata", lambda root, path, entry, *args:
                        (metadata[analyse.trial_key(entry)], {"elapsed_seconds": 1.}))
    verification = analyse.run(tmp_path)
    assert verification["passed"] and verification["all_rows"] == 24 and verification["selected_rows"] == 18
    output = tmp_path / "results" / analyse.STUDY
    effects = analyse.read_json(output / "effects.json")
    assert effects["primary"]["lr_by_method"] == {method: choices[method]["lr"] for method in analyse.TRAINED}
    assert effects["fixed_lr_descriptive"]["lr_by_method"] == {method: 3e-5 for method in analyse.TRAINED}
    assert effects["raw_oracle_diagnostic"]["confidence"] == .95
    assert analyse.read_json(output / "resource_summary.json")["sample_count"] == 18
    assert len(analyse.read_json(output / "costs.json")["gpu_trials"]) == 18
