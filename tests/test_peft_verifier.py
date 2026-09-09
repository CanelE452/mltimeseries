import copy
import csv

import pytest

from experiments.peft_adaptation_scope_v1.verify_results import (
    verify_optimization_boundaries,
    verify_selected_report,
)


@pytest.fixture
def report_fixture(tmp_path):
    methods = ["AFF", "H_LIN", "H_MLP", "H_FULL", "OFF_LORA", "FULL"]
    contract = {"panels": ["ettm2", "jena"], "seeds": [0, 1, 2], "steps": 500,
                "grids": {method: [0.001, 0.01, 0.1] for method in methods}}
    trials, selection, rows, effects, raw_selected = [], {}, [], [], {}

    def add_row(panel, method, seed, metric, source):
        rows.append({"panel": panel, "method": method, "seed": str(seed),
                     "lr": str(metric.get("lr", metric.get("selected_ridge"))),
                     "selected_step": str(metric["selected_step"]), "source_path": str(source),
                     "val_score": str(metric["val_score"]), "eval_score": str(metric["eval_score"])})

    for panel in contract["panels"]:
        f0 = {"panel": panel, "method": "F0", "seed": 0, "lr": 0.0, "selected_step": 0,
              "val_score": 1.0, "eval_score": 1.0, "name": f"{panel}_F0_lr0_seed0"}
        trials.append(f0)
        add_row(panel, "F0", 0, f0, tmp_path / "trials" / f0["name"])
        raw = {"selected_ridge": 10.0, "selected_step": 0, "val_score": 0.9, "eval_score": 0.9}
        source = tmp_path / "raw_rescue" / panel
        raw_selected[panel] = {"choice": {"source_path": str(source)}, "metric": raw}
        add_row(panel, "RAW_VARX_RIDGE", 0, raw, source)
        for method_index, method in enumerate(methods):
            for seed in contract["seeds"]:
                grid = contract["grids"][method] if seed == 0 else [0.01]
                for lr in grid:
                    metric = {"panel": panel, "method": method, "seed": seed, "lr": lr,
                              "selected_step": 200, "val_score": 0.7 + method_index * 0.01 + (0.05 if lr != 0.01 else 0),
                              "eval_score": 0.6 + method_index * 0.01 + seed * 0.001,
                              "name": f"{panel}_{method}_lr{lr:g}_seed{seed}"}
                    trials.append(metric)
                    if lr == 0.01:
                        add_row(panel, method, seed, metric, tmp_path / "trials" / metric["name"])
                        if seed == 0:
                            selection[f"{panel}/{method}"] = {"lr": lr, "val_score": metric["val_score"]}
        for internal in ["OFF_LORA", "FULL"]:
            head_index, internal_index = methods.index("H_LIN"), methods.index(internal)
            delta = (head_index - internal_index) * 0.01
            effects.append({"panel": panel, "internal": internal, "head_selected_on_validation": "H_LIN",
                            "paired_seed_ids": [0, 1, 2], "paired_seed_count": 3,
                            "seed_deltas": [delta] * 3, "delta_over_F0": delta})
    return {"rows": rows, "effects": effects, "contract": contract, "selection": selection,
            "trials": trials, "raw_selected": raw_selected, "run_root": tmp_path}


def test_complete_report_has_exactly_40_rows_and_four_effects(report_fixture):
    result = verify_selected_report(**report_fixture)
    assert result["selected_csv_rows"] == 40 and result["effects_verified"] == 4
    assert result["paired_seed_ids_verified"] == [0, 1, 2]


def test_partial_csv_fails_even_when_all_62_trial_metrics_exist(report_fixture, tmp_path):
    assert len(report_fixture["trials"]) == 62
    partial = [row for row in report_fixture["rows"] if row["seed"] == "0"]
    # Model the actual stale --partial artifact, preserving both RAW rows.
    csv_path = tmp_path / "selected_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(partial[0]))
        writer.writeheader()
        writer.writerows(partial)
    with csv_path.open(encoding="utf-8", newline="") as handle:
        report_fixture["rows"] = list(csv.DictReader(handle))
    with pytest.raises(AssertionError, match="Selected CSV keys differ"):
        verify_selected_report(**report_fixture)


def test_duplicate_selected_key_fails(report_fixture):
    report_fixture["rows"].append(copy.deepcopy(report_fixture["rows"][0]))
    with pytest.raises(AssertionError, match="Duplicate selected CSV key"):
        verify_selected_report(**report_fixture)


@pytest.mark.parametrize("field,value,pattern", [
    ("lr", "9.0", "LR mismatch"),
    ("source_path", "unexpected/source", "source mismatch"),
    ("val_score", "99.0", "val_score mismatch"),
    ("eval_score", "99.0", "eval_score mismatch"),
])
def test_stale_neural_selected_row_fails(report_fixture, field, value, pattern):
    row = next(row for row in report_fixture["rows"] if row["method"] == "H_LIN" and row["seed"] == "1")
    row[field] = value
    with pytest.raises(AssertionError, match=pattern):
        verify_selected_report(**report_fixture)


def test_missing_effect_fails(report_fixture):
    report_fixture["effects"].pop()
    with pytest.raises(AssertionError, match="Missing or extra effect keys"):
        verify_selected_report(**report_fixture)


def test_incomplete_effect_seed_pairing_fails(report_fixture):
    report_fixture["effects"][0]["paired_seed_ids"] = [0, 2]
    report_fixture["effects"][0]["paired_seed_count"] = 2
    with pytest.raises(AssertionError, match="Incomplete seed pairing"):
        verify_selected_report(**report_fixture)


def test_stale_effect_values_fail(report_fixture):
    report_fixture["effects"][0]["delta_over_F0"] = 100
    with pytest.raises(AssertionError, match="Stale mean effect"):
        verify_selected_report(**report_fixture)


def test_boundary_report_includes_repeat_seed_budget(report_fixture):
    trials = report_fixture["trials"]
    repeat = next(trial for trial in trials if trial["panel"] == "ettm2" and trial["method"] == "H_LIN" and trial["seed"] == 1)
    repeat["selected_step"] = 500
    boundaries = verify_optimization_boundaries(trials, report_fixture["contract"], report_fixture["selection"])
    boundary = next(item for item in boundaries if item["panel"] == "ettm2" and item["method"] == "H_LIN")
    assert not boundary["step_at_budget"]
    assert boundary["any_seed_at_budget"]
    assert boundary["selected_seeds"][1] == {"seed": 1, "selected_step": 500, "step_at_budget": True}


def test_verifier_checks_executed_grid_against_saved_contract(report_fixture):
    report_fixture["contract"]["grids"]["AFF"] = [1.0, 2.0, 3.0]
    with pytest.raises(AssertionError, match="Executed LR grid differs from contract"):
        verify_optimization_boundaries(report_fixture["trials"], report_fixture["contract"], report_fixture["selection"])
