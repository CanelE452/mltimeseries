import copy
from pathlib import Path

import numpy as np
import pytest

from experiments.peft_external_gap_v1 import analyse


def test_score_equal_target_weight_and_observed_cells():
    target = np.array([[[1., 3.], [20., np.nan]], [[5., np.nan], [np.nan, np.nan]]])
    prediction = np.zeros((2, 2, 1, 2))
    score, sums, counts = analyse.score_prediction(prediction, target, [1., 10.], [.5])
    assert score == pytest.approx((3. + 2.) / 2)
    np.testing.assert_equal(counts, [[2, 1], [1, 0]])
    assert analyse.macro_score(sums, counts) == score


def test_score_rejects_silent_target_drop_and_bad_shape():
    target = np.array([[[1., 3.], [np.nan, np.nan]]])
    with pytest.raises(ValueError, match="Every fixed target"):
        analyse.score_prediction(np.zeros((1, 2, 1, 2)), target, [1., 1.], [.5])
    with pytest.raises(ValueError, match="Expected predictions"):
        analyse.score_prediction(np.zeros((1, 1, 1, 2)), target, [1., 1.], [.5])


def test_qcal_fits_only_provided_calibration_cells_and_ignores_nan():
    target = np.array([[[2., 4.]], [[6., np.nan]]])
    prediction = np.zeros((2, 1, 3, 2))
    q = np.array([.1, .5, .9])
    offsets = analyse.qcal_offsets(prediction, target, q)
    np.testing.assert_allclose(offsets, [[2.4, 4., 5.6]])
    heldout = np.full((1, 1, 3, 2), 100.)
    first = analyse.apply_qcal(heldout, offsets)
    np.testing.assert_allclose(first[0, 0, :, 0], [102.4, 104., 105.6])
    np.testing.assert_array_equal(offsets, analyse.qcal_offsets(prediction, target, q))


def test_quantile_sort_after_offsets_handles_crossing():
    prediction = np.array([[[[3.], [2.], [1.]]]])
    actual = analyse.apply_qcal(prediction, np.array([[.5, .2, .1]]))
    np.testing.assert_allclose(actual[0, 0, :, 0], [1.1, 2.2, 3.5])


def test_bootstrap_pairs_targets_and_seeds_and_recomputes_ratio():
    counts = np.array([[1., 100.], [100., 1.], [2., 4.]])
    f0 = np.array([[1., 500.], [300., 1.], [4., 12.]])
    left = np.broadcast_to(f0 * 1.2, (3, 3, 2)).copy()
    right = np.broadcast_to(f0, (3, 3, 2)).copy()
    weights = np.array([[3., 0., 0.], [0., 3., 0.], [0., 0., 3.], [1., 1., 1.]])
    effect = analyse.paired_effect(left, right, f0, counts, weights)
    assert effect["value"] == pytest.approx(.2)
    np.testing.assert_allclose(effect["ci"], [.2, .2])
    np.testing.assert_allclose(effect["seed_values"], [.2, .2, .2])
    assert effect["decision"] == "conditional_practical_improvement"


def test_bootstrap_does_not_treat_seed_axis_as_independent_days():
    counts = np.ones((4, 2))
    f0 = np.ones((4, 2))
    left = np.ones((3, 4, 2))
    right = left - np.array([.01, .02, -.03])[:, None, None]
    weights = analyse.moving_block_weights(4, 2, replicates=20)
    effect = analyse.paired_effect(left, right, f0, counts, weights)
    assert effect["value"] == pytest.approx(0, abs=1e-15)
    np.testing.assert_allclose(effect["ci"], [0, 0], atol=1e-15)
    assert effect["decision"] == "conditional_practical_equivalence"


def test_draw_missing_an_entire_target_is_reported_not_silently_reweighted():
    counts = np.array([[1., 0.], [0., 1.]])
    f0 = counts.copy()
    left = np.broadcast_to(2 * f0, (3, 2, 2))
    right = np.broadcast_to(f0, (3, 2, 2))
    weights = np.array([[2., 0.], [0., 2.], [1., 1.]])
    effect = analyse.paired_effect(left, right, f0, counts, weights)
    assert effect["discarded_resamples"] == 2
    assert effect["valid_resamples"] == 1
    assert effect["value"] == 1.


def test_moving_blocks_have_fixed_total_count_and_deterministic_draws():
    weights = analyse.moving_block_weights(83, 7, 100)
    np.testing.assert_array_equal(weights.sum(axis=1), np.full(100, 83))
    np.testing.assert_array_equal(weights, analyse.moving_block_weights(83, 7, 100))


def test_horizon_diagnostic_keeps_fully_missing_lead_without_failing_total_score():
    target = np.array([[[1., np.nan], [2., 4.]], [[3., np.nan], [4., 6.]]])
    prediction = np.zeros((2, 2, 3, 2))
    detail = analyse.score_details(prediction, target, [1., 2.], [.1, .5, .9])
    assert detail["valid_cells"] == [2, 4]
    assert detail["horizon_target_loss_counts"][1][0] == 0
    assert np.isfinite(detail["score"])


def metadata_fixture(root):
    trials, selected, choices = [], [], {}
    def entry(dataset, method, seed, lr, score):
        return {"dataset": dataset, "method": method, "seed": seed, "lr": lr,
                "path": str(Path("runs") / analyse.STUDY / "trials" / dataset / method / f"lr_{lr:.0e}_seed_{seed}"),
                "val_score": score, "guard_seconds": 1., "result_sha256": "a" * 64, "checkpoint_sha256": "b" * 64}
    for dataset in analyse.DATASETS:
        f0 = entry(dataset, "F0", 12000, 0., 1.)
        trials.append(f0)
        selected.append({**f0, "role": "F0"})
        candidates = []
        for method, rates in analyse.GRIDS.items():
            for i, lr in enumerate(rates):
                value = entry(dataset, method, 12000, lr, .8 + .05 * i)
                candidates.append(value)
                trials.append(value)
        choices[dataset] = {}
        for role, allowed in (("H", ("H_MLP", "H_FULL")), ("OFF_LORA", ("OFF_LORA",))):
            chosen = min((e for e in candidates if e["method"] in allowed),
                         key=lambda e: (e["val_score"], e["method"], e["lr"]))
            choices[dataset][role] = chosen
            selected.append({**chosen, "role": role})
            for seed in (12001, 12002):
                repeat = entry(dataset, chosen["method"], seed, chosen["lr"], .81)
                trials.append(repeat)
                selected.append({**repeat, "role": role})
    selection = {"completed": True, "global_choices_frozen": True, "fit_trial_count": 28,
                 "choices": choices, "selected": selected}
    forecasts = [{**e, "fit_path": e["path"], "fit_result_sha256": e["result_sha256"],
                  "path": str(Path("runs") / analyse.STUDY / "forecasts" / e["dataset"] / f"{e['role']}_seed_{e['seed']}"),
                  "result_sha256": "c" * 64, "forecast_guard_seconds": 2.} for e in selected]
    return trials, selection, choices, forecasts, root


def test_complete_metadata_and_exact_selection_pass(tmp_path):
    actual = analyse.verify_entries(*metadata_fixture(tmp_path))
    assert [len(value) for value in actual] == [28, 14, 14]


@pytest.mark.parametrize("field", ["trials", "selected", "forecasts"])
def test_partial_artifacts_cannot_be_reported_passed(tmp_path, field):
    arguments = list(metadata_fixture(tmp_path))
    if field == "trials":
        arguments[0].pop()
    elif field == "selected":
        arguments[1]["selected"].pop()
    else:
        arguments[3].pop()
    with pytest.raises(AssertionError):
        analyse.verify_entries(*arguments)


def test_choice_must_reproduce_all_candidate_argmin(tmp_path):
    arguments = copy.deepcopy(metadata_fixture(tmp_path))
    wrong = next(e for e in arguments[0] if e["dataset"] == "bike" and e["method"] == "H_MLP")
    arguments[2]["bike"]["H"] = wrong
    arguments[1]["choices"] = arguments[2]
    with pytest.raises(AssertionError, match="argmin"):
        analyse.verify_entries(*arguments)


def test_forecast_cannot_repoint_checkpoint_fit(tmp_path):
    arguments = metadata_fixture(tmp_path)
    arguments[3][1]["fit_path"] = arguments[3][0]["fit_path"]
    with pytest.raises(AssertionError, match="own fit"):
        analyse.verify_entries(*arguments)


def test_terminal_resource_row_is_not_a_measurement(tmp_path):
    import json
    folder = tmp_path / "guard"
    folder.mkdir()
    sample = {"timestamp": "x", "available_ram_gib": 10, "available_commit_gib": 10,
              "child_tree_rss_gib": 1, "git_process_count": 2, "gpus": []}
    (folder / "resource_log.jsonl").write_text(json.dumps(sample) + '\n{"event":"finish"}\n')
    assert analyse.resource_samples([tmp_path]) == [sample]


def test_training_score_recomputes_exact_masked_multitarget_estimand():
    from experiments.peft_external_gap_v1 import train
    rng = np.random.default_rng(100)
    target = rng.normal(size=(7, 2, 48)).astype(np.float32)
    target[1:4, 0, 2:30] = np.nan
    target[4:, 1, 10:] = np.nan
    prediction = rng.normal(size=(7, 2, 21, 48)).astype(np.float32)
    q = np.linspace(.01, .99, 21)
    scale = np.array([.3, 40.])
    native_score, native_sums, native_counts = train.scores(prediction, target, scale, q)
    value, sums, counts = analyse.score_prediction(np.sort(prediction, axis=2), target, scale, q)
    assert value == pytest.approx(native_score["score"], abs=1e-14)
    np.testing.assert_allclose(sums, native_sums, rtol=1e-14, atol=1e-14)
    np.testing.assert_array_equal(counts, native_counts)


def prepared_fixture(root):
    from datetime import datetime, timedelta
    from experiments.peft_external_gap_v1 import data
    source = root / "fixture_source.txt"
    source.write_text("Synthetic periodic CPU fixture; no real held-out data", encoding="utf-8")
    hours = 200 * 24
    t = np.arange(hours, dtype=np.float64)
    values = np.stack([2 + np.sin(t / 24), 4 + np.cos(t / 28),
                       .5 + np.sin(t / 168), 10 + np.cos(t / 168), .2 + np.sin(t / 14)], axis=1).astype(np.float32)
    values[400:405, 0] = np.nan
    values[2000:2007, 1] = np.nan
    values[10, 2] = np.nan
    timestamps = [datetime(2020, 1, 1) + timedelta(hours=int(hour)) for hour in range(hours)]
    data.write_panel_archives("bike", timestamps, values, list(data.BIKE_SPEC.channels), [0, 1],
                              root / "runs" / analyse.STUDY / "prepared",
                              {"source_path": str(source), "source_sha256": analyse.sha256_file(source)})


def test_real_data_api_prefix_scaling_masks_and_raw_replay_on_cpu_fixture(tmp_path):
    prepared_fixture(tmp_path)
    panels = analyse.load_panels(tmp_path, "bike", set())
    meta, predictions, path, audit = analyse.load_raw(tmp_path, "bike", *panels)
    assert meta["completed"] is True
    assert audit["normal_equations_verified"] is True
    assert predictions["eval_predictions"].shape == (83, 2, 21, 48)
    previous_hash = analyse.sha256_file(path / "result.json")
    _, _, _, repeated = analyse.load_raw(tmp_path, "bike", *panels)
    assert repeated["created_this_analysis"] is False
    assert analyse.sha256_file(path / "result.json") == previous_hash


def test_analysis_refuses_incomplete_study_before_opening_holdout(tmp_path):
    study = tmp_path / "runs" / analyse.STUDY
    analyse.write_json(study / "completed.json", {"completed": True, "fit_trials": 28, "forecasts": 13})
    analyse.write_json(study / "fit_completed.json", {"completed": True, "fit_trials": 28, "adaptation_fits": 26, "frozen_cache_trials": 2})
    with pytest.raises(AssertionError, match="14 completed forecasts"):
        analyse.run(tmp_path)


def test_separate_raw_guard_directory_is_verified_directly(tmp_path):
    sample = {"timestamp": "x", "available_ram_gib": 10, "available_commit_gib": 10,
              "child_tree_rss_gib": 1, "git_process_count": 1, "gpus": []}
    import json
    path = tmp_path / "raw_guards" / "bike"
    path.mkdir(parents=True)
    analyse.write_json(path / "status.json", {"completed": True, "returncode": 0, "reasons": [], "elapsed_seconds": 2})
    (path / "resource_log.jsonl").write_text(json.dumps(sample) + '\n{"event":"finish"}\n')
    assert analyse.validate_guard(path, guard_directory=True)["elapsed_seconds"] == 2
    assert analyse.summarize_resources([path], "fixture", guard_directory=True)["sample_count"] == 1


def test_cpu_null_gpu_record_and_mixed_gpu_records_preserve_missingness(tmp_path):
    import json
    cpu = {"timestamp": "2026-09-08T06:32:55.785781+00:00", "available_ram_gib": 16.008975982666016,
           "available_commit_gib": 13.555068969726562, "child_tree_rss_gib": 0.0,
           "git_process_count": 0, "gpus": None}
    cpu_path, gpu_path = tmp_path / "cpu", tmp_path / "gpu"
    cpu_path.mkdir()
    gpu_path.mkdir()
    (cpu_path / "resource_log.jsonl").write_text(json.dumps(cpu) + '\n{"event":"finish"}\n')
    cpu_summary = analyse.summarize_resources([cpu_path], "CPU only", guard_directory=True)
    assert cpu_summary["sample_count"] == 1
    assert cpu_summary["max_gpu_memory_used_mib"] is None
    assert cpu_summary["max_gpu_temperature_c"] is None
    gpu = {"timestamp": "2026-09-08T06:19:42.963106+00:00", "available_ram_gib": 15.7947998046875,
           "available_commit_gib": 13.496238708496094, "child_tree_rss_gib": 0.0,
           "git_process_count": 0, "gpus": [{"index": 0, "memory_used_mib": 1301.0, "temperature_c": 47.0}]}
    (gpu_path / "resource_log.jsonl").write_text(json.dumps(gpu) + '\n{"event":"finish"}\n')
    mixed = analyse.summarize_resources([cpu_path, gpu_path], "Mixed", guard_directory=True)
    assert mixed["sample_count"] == 2
    assert mixed["max_gpu_memory_used_mib"] == 1301.0
    assert mixed["max_gpu_temperature_c"] == 47.0
    assert mixed["min_available_ram_gib"] == 15.7947998046875
