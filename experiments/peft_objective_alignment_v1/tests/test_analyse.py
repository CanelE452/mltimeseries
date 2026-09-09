import json
import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.peft_objective_alignment_v1 import analyse as a


def test_ratio_recomputes_denominator_and_target_macro():
    n = 40
    counts = np.column_stack([np.arange(1, n+1), np.ones(n)*3])
    f0 = counts * np.column_stack([np.linspace(1, 3, n), np.linspace(4, 2, n)])
    right = f0*.9
    left = right + counts*np.column_stack([np.linspace(.02, .3, n), np.zeros(n)])
    result = a.paired_effect(left, right, f0, counts)
    weights = a.numerical.moving_block_weights(n, 7, 4000, 2026090815)
    draw = ((weights@(left-right))/(weights@counts)).mean(axis=1) / ((weights@f0)/(weights@counts)).mean(axis=1)
    np.testing.assert_allclose(result["ci"], np.quantile(draw, [.0125, .9875]), rtol=0, atol=1e-15)
    assert result["value"] == pytest.approx((a.numerical.macro_score(left, counts)-a.numerical.macro_score(right, counts))/a.numerical.macro_score(f0, counts))
    assert result["bootstrap_seed"] == 2026090815


def test_constant_proportional_effect_and_parity_contract():
    counts = np.ones((83, 2))*21*48
    f0 = counts.copy()
    effects = a.comparison_effects(f0*1.03, f0, f0, counts)
    np.testing.assert_allclose(effects["daily"]["7"]["ci"], [.03, .03])
    assert effects["nonoverlapping"]["even"]["7"]["origin_count"] == 42
    assert effects["nonoverlapping"]["odd"]["7"]["origin_count"] == 41
    assert effects["nonoverlapping"]["even"]["7"]["block_origins"] == 4
    assert effects["nonoverlapping"]["even"]["7"]["target_span_days"] == 8


def test_gate_requires_both_sources_and_strict_threshold():
    def source(low):
        return {"SORT": {"NORM_ALIGNED_vs_RAW_ALIGNED": {"daily": {"7": {"ci": [low, .04]}}}}}
    value = {"bike": source(.011), "household": source(.01)}
    assert a.alignment_gate(value)["decision"] == "CLOSE_CURRENT_OBJECTIVE_ALIGNMENT_SCREEN"
    value["household"] = source(.0101)
    assert a.alignment_gate(value)["decision"] == "RAW_ALIGNMENT_PRACTICALLY_RELEVANT"
    assert not a.alignment_gate(value)["new_method_demonstrated"]
    with pytest.raises(AssertionError):
        a.alignment_gate({"bike": source(.1)})


def test_masked_target_macro_not_global_micro_average():
    target = np.zeros((2, 2, 2))
    target[0, 1, 1] = np.nan
    quantiles = np.array([.1, .5, .9])
    prediction = np.ones((2, 2, 3, 2))
    prediction[:, 1] = 10
    score, _, counts = a.numerical.score_prediction(prediction, target, np.ones(2), quantiles)
    assert score == pytest.approx(5.5)
    assert counts.sum(axis=0).tolist() == [12, 9]


def test_mixed_cpu_gpu_resource_and_failure_accounting(tmp_path):
    records = []
    for i, gpus in enumerate((None, [{"memory_used_mib": 1700, "temperature_c": 48}])):
        folder = tmp_path / str(i)
        folder.mkdir()
        sample = {"timestamp": str(i), "available_ram_gib": 12-i, "available_commit_gib": 10,
                  "child_tree_rss_gib": 1., "git_process_count": 56 if i else 0, "gpus": gpus}
        (folder/"resource_log.jsonl").write_text(json.dumps(sample)+"\n")
        records.append({"guard_path": str(folder), "completed": not i, "state": "safety_stop" if i else "completed"})
    result = a.summarize_attempt_resources(records, "fixture")
    assert result["max_gpu_memory_used_mib"] == 1700
    assert result["max_git_process_count"] == 56
    assert result["guard_failures"] == result["safety_stops"] == 1
    assert a.summarize_attempt_resources(records[:1], "cpu")["max_gpu_memory_used_mib"] is None


def entry_fixture():
    trials = [{"dataset": dataset, "arm": arm, "method": "OFF_LORA", "seed": 12000,
               "lr": 3e-5, "smoke": False, "guard_seconds": 2.} for dataset in a.DATASETS for arm in a.ARMS]
    selection = {"completed": True, "global_choices_frozen": True, "fit_trial_count": 6,
                 "study_contract_sha256": "fixed", "selected": copy.deepcopy(trials)}
    done = {"completed": True, "fits": 6, "forecasts": 6, "reused_forecasts": 2,
            "study_contract_sha256": "fixed", "fit_guard_seconds": 12., "forecast_guard_seconds": 12.}
    return trials, copy.deepcopy(trials), selection, done


@pytest.mark.parametrize("change", ("missing_fit", "duplicate_forecast", "wrong_lr", "selection_changed", "not_completed", "cost_omitted"))
def test_incomplete_or_unpaired_job_ledgers_cannot_pass(change):
    trials, forecasts, selection, done = entry_fixture()
    a.verify_entry_sets(trials, forecasts, selection, done, "fixed")
    if change == "missing_fit":
        trials.pop()
    elif change == "duplicate_forecast":
        forecasts[-1] = forecasts[0]
    elif change == "wrong_lr":
        forecasts[0]["lr"] = 1e-4
    elif change == "selection_changed":
        selection["selected"][0]["lr"] = 1e-4
    elif change == "not_completed":
        done["completed"] = False
    else:
        done["fit_guard_seconds"] = 10.
    with pytest.raises(AssertionError):
        a.verify_entry_sets(trials, forecasts, selection, done, "fixed")


def test_unfinished_production_rejected_before_panel_or_holdout_access(tmp_path, monkeypatch):
    study = tmp_path/"runs"/a.STUDY
    study.mkdir(parents=True)
    a.write_json(study/"completed.json", {"completed": False, "fits": 6, "forecasts": 5})
    from experiments.peft_external_gap_v1 import train
    monkeypatch.setattr(train, "Panel", lambda *args, **kwargs: pytest.fail("No panel may be opened before the completion gate"))
    with pytest.raises(AssertionError, match="finish"):
        a.run(tmp_path)


def calibration_fixture(path):
    panel = SimpleNamespace(origins={"train": np.arange(63)*2+48}, target_values=np.ones((240, 2)), target_indices=np.array([0, 1]))
    indices = np.linspace(0, 62, 8).astype(int)
    origins = panel.origins["train"][indices]
    value = {"origin_indices": indices.tolist(), "origins": origins.tolist(), "origin_sha256": a.numerical.array_hash(origins),
             "source_parameter_sha256": "initial", "source_frozen_sha256": "frozen", "optimizer_steps": 0,
             "rng_unchanged": True, "parameters_unchanged": True, "gradients_cleared": True,
             "norms": {"NATIVE": 2., "NORM_ALIGNED": 1., "RAW_ALIGNED": 4.},
             "multipliers": {"NATIVE": 1., "NORM_ALIGNED": 2., "RAW_ALIGNED": .5},
             "objective_losses": {arm: .3 for arm in a.ARMS},
             "cosines": {"NATIVE__NORM_ALIGNED": .9, "NATIVE__RAW_ALIGNED": .7, "NORM_ALIGNED__RAW_ALIGNED": .8},
             "jacobian_summary": {"count": 8*2*48*21, "min": 1., "median": 2., "p95": 3., "max": 4.}, "crossing_fraction": .01}
    a.write_json(path/"gradient_calibration.json", value)
    meta = {"gradient_calibration": value, "gradient_calibration_sha256": a.sha(path/"gradient_calibration.json"),
            "arm": "RAW_ALIGNED", "steps_completed": 3, "gradient_norms": [.5, 1., 2.], "clipping_steps": [3],
            "loss_multiplier": .5, "unscaled_training_losses": [.3, .2, .1], "scaled_training_losses": [.15, .1, .05],
            "audits": {"initial_adaptation_sha256": "initial", "frozen_before_sha256": "frozen", "maximum_gradient_norm": 2.}}
    return meta, panel


@pytest.mark.parametrize("change", ("scale", "origins", "jacobian_count", "clipping", "rng"))
def test_gradient_calibration_reconstructs_fixed_train_only_norms_and_clipping(tmp_path, change):
    meta, panel = calibration_fixture(tmp_path)
    a.verify_calibration(meta, panel, tmp_path)
    if change == "scale":
        meta["gradient_calibration"]["multipliers"]["RAW_ALIGNED"] = 1.
    elif change == "origins":
        meta["gradient_calibration"]["origins"][-1] += 24
    elif change == "jacobian_count":
        meta["gradient_calibration"]["jacobian_summary"]["count"] -= 21
    elif change == "clipping":
        meta["clipping_steps"] = [2, 3]
    else:
        meta["gradient_calibration"]["rng_unchanged"] = False
    a.write_json(tmp_path/"gradient_calibration.json", meta["gradient_calibration"])
    meta["gradient_calibration_sha256"] = a.sha(tmp_path/"gradient_calibration.json")
    with pytest.raises(AssertionError):
        a.verify_calibration(meta, panel, tmp_path)


def test_all_attempts_include_preserved_failure_and_cpu_null_gpu(tmp_path):
    study = tmp_path/"runs"/a.STUDY
    study.mkdir(parents=True)
    a.write_json(study/"study_contract.json", {"fixture": True})
    contract_sha = a.sha(study/"study_contract.json")
    invocations = {phase: {"phase": phase, "study_contract_sha256": contract_sha, "completed": True,
                  "started_at": 0., "finished_at": 100., "wall_seconds": 100., "attempts": [], "reused_completed_jobs": []}
                   for phase in ("smoke", "production")}
    jobs = [("fit", True, True)]*6 + [("fit", False, True)]*6 + [("forecast", False, True)]*6 + [("forecast", False, False), ("analysis_cpu", False, False)]
    for index, (stage, smoke, completed) in enumerate(jobs):
        directory = study/"preserved_attempts"/str(index)/"guard"
        directory.mkdir(parents=True)
        module = "train" if stage == "fit" else "forecast" if stage == "forecast" else "analyse"
        status = {"command": ["python", "-m", f"experiments.{a.STUDY}.{module}"] + (["--smoke"] if smoke else []),
                  "guard_pid": index, "started_at": "2026-09-08T00:00:00+00:00", "finished_at": "2026-09-08T00:00:05+00:00",
                  "elapsed_seconds": 5., "completed": completed, "state": "completed" if completed else "safety_stop",
                  "returncode": 0 if completed else 1, "reasons": [] if completed else ["fixture stop"]}
        a.write_json(directory/"status.json", status)
        sample = {"timestamp": "2026-09-08T00:00:01+00:00", "available_ram_gib": 10., "available_commit_gib": 9.,
                  "child_tree_rss_gib": 1., "git_process_count": 56 if index == 18 else 0,
                  "gpus": None if stage == "analysis_cpu" else [{"memory_used_mib": 1400, "temperature_c": 44}]}
        (directory/"resource_log.jsonl").write_text(json.dumps(sample)+"\n")
        if stage != "analysis_cpu":
            invocations["smoke" if smoke else "production"]["attempts"].append({"guard_status": status, "guard_status_sha256": a.sha(directory/"status.json")})
    (study/"invocations").mkdir()
    for phase, value in invocations.items():
        a.write_json(study/"invocations"/f"{phase}.json", value)
    result = a.attempt_audit(tmp_path, {})
    assert result["phases"]["production_forecast"]["all_attempt_guard_seconds"] == 35.
    assert result["phases"]["production_forecast"]["unsuccessful_attempts"] == 1
    assert result["phases"]["completed_analysis_cpu"]["all_attempt_guard_seconds"] == 5.
    summary = a.summarize_attempt_resources(result["attempt_records"], "fixture")
    assert summary["max_git_process_count"] == 56 and summary["guard_failures"] == 2
    invocations["production"]["attempts"].pop()
    a.write_json(study/"invocations"/"production.json", invocations["production"])
    with pytest.raises(AssertionError, match="represented"):
        a.attempt_audit(tmp_path, {})
