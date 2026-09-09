import importlib
import importlib.util
import json
import math

import numpy as np
import pytest


MODULE = "experiments.peft_coarse_supervision_v1.analysis"
ARMS = ("F0", "PROFILE", "COARSE_LIFT", "FROZEN_HEAD", "ATTN_LORA")


def fixture():
    site = np.repeat(["Eagle", "Lamb"], 24)
    target_id = np.tile(np.repeat([f"target{i}" for i in range(8)], 3), 2)
    month = np.tile(["2017-04", "2017-05", "2017-06"], 16)
    truth = np.full((48, 744), 10.0)
    return {
        "predictions": {arm: truth + 1.0 for arm in ARMS},
        "truth": truth,
        "horizon": np.full(48, 744),
        "scale": np.ones(48),
        "site": site,
        "target_id": target_id,
        "month": month,
        "simple_policy": "FROZEN_HEAD",
    }


def analyze(data):
    return importlib.import_module(MODULE).analyze_predictions(**data)


def test_public_function_exists():
    assert importlib.util.find_spec(MODULE) is not None, "Pure analysis module is not implemented"
    assert callable(importlib.import_module(MODULE).analyze_predictions)


def test_error_decomposition_and_constant_lift_preserve_pattern():
    data = fixture()
    data["scale"][:] = 2.0
    signal = np.tile([-2.0, 2.0], 372)
    data["predictions"]["F0"] = data["truth"] + signal + 3.0
    data["predictions"]["COARSE_LIFT"] = data["predictions"]["F0"] + 7.0
    result = analyze(data)
    base = result["cells"][0]["arms"]["F0"]
    lift = result["cells"][0]["arms"]["COARSE_LIFT"]
    assert base["mse"] == pytest.approx(3.25)
    assert base["level_mse"] == pytest.approx(2.25)
    assert base["pattern_mse"] == pytest.approx(1.0)
    assert base["mse"] == pytest.approx(base["level_mse"] + base["pattern_mse"])
    assert lift["pattern_mse"] == pytest.approx(base["pattern_mse"])
    assert lift["level_mse"] == pytest.approx(25.0)
    assert result["decomposition_max_abs_error"] < 1e-12
    json.dumps(result, allow_nan=False)


def test_validity_uses_raw_truth_and_horizon_with_exact_eighty_percent_boundary():
    data = fixture()
    data["horizon"][:] = 100
    data["truth"][0, :10] = -1.0
    data["truth"][0, 10:20] = np.nan
    data["truth"][1, :21] = np.nan
    for values in data["predictions"].values():
        values[:, 100:] = np.nan
        values[0, :20] = np.inf
        values[1, :21] = np.nan
    result = analyze(data)
    assert result["cells"][0]["count"] == 80
    assert result["cells"][0]["eligible"] is True
    assert result["cells"][1]["count"] == 79
    assert result["cells"][1]["eligible"] is False
    assert result["sites"]["Eagle"]["eligible_cell_count"] == 23
    assert result["sites"]["Eagle"]["data_gate_passed"] is True


def test_macro_gives_equal_target_weight_despite_missing_months():
    data = fixture()
    data["truth"][:2] = np.nan
    data["predictions"]["F0"][2] = 13.0
    result = analyze(data)
    assert result["sites"]["Eagle"]["arms"]["F0"]["mse"] == pytest.approx((9 + 7) / 8)
    assert result["pooled"]["arms"]["F0"]["mse"] == pytest.approx(1.5)


def test_missing_entire_target_is_data_insufficiency_not_method_failure():
    data = fixture()
    data["truth"][:3] = np.nan
    result = analyze(data)
    assert result["sites"]["Eagle"]["eligible_cell_count"] == 21
    assert result["sites"]["Eagle"]["targets_without_eligible_cells"] == ["target0"]
    assert result["decision"] == "INSUFFICIENT_EVALUATION_DATA"
    assert result["bootstrap"]["performed"] is False
    assert result["novel_method_demonstrated"] is False


def test_target_cluster_ci_matches_exact_binomial_toy_not_month_resampling():
    data = fixture()
    data["predictions"]["ATTN_LORA"][:12] = 10.0
    result = analyze(data)
    distribution = np.cumsum([math.comb(8, k) / 256 for k in range(9)])
    lower = int(np.searchsorted(distribution, 0.025)) / 16
    upper = int(np.searchsorted(distribution, 0.975)) / 16
    assert result["effects"]["pooled"]["point"] == pytest.approx(0.25)
    assert result["effects"]["pooled"]["ci95"] == pytest.approx([lower, upper])
    assert result["effects"]["sites"]["Eagle"]["ci95"] == pytest.approx([2 * lower, 2 * upper])
    assert result["effects"]["sites"]["Lamb"]["ci95"] == [0.0, 0.0]
    assert result["bootstrap"]["replicates"] == 4000
    assert result["bootstrap"]["seed"] == 2026090817


def test_progress_requires_pattern_gain_in_both_sites():
    data = fixture()
    alternating = np.tile([-1.0, 1.0], 372)
    data["predictions"]["FROZEN_HEAD"] = data["truth"] + 2.0 + alternating
    data["predictions"]["ATTN_LORA"] = data["truth"] + 0.5 * alternating
    result = analyze(data)
    assert result["decision"] == "PROCEED_TO_METHOD_DEVELOPMENT"
    assert result["progress_gate"]["passed"] is True
    assert result["novel_method_demonstrated"] is False
    data["predictions"]["ATTN_LORA"][24:] = data["truth"][24:] + alternating
    result = analyze(data)
    assert result["decision"] == "CLOSE_CURRENT_COARSE_SUPERVISION_SCREEN"
    assert result["progress_gate"]["site_pattern_better_than_head"]["Lamb"] is False


@pytest.mark.parametrize("problem", ["duplicate_cell", "missing_arm", "nonfinite_prediction", "wrong_policy"])
def test_invalid_inputs_fail_instead_of_changing_paired_population(problem):
    data = fixture()
    if problem == "duplicate_cell":
        data["month"][0] = data["month"][1]
    elif problem == "missing_arm":
        del data["predictions"]["PROFILE"]
    elif problem == "nonfinite_prediction":
        data["predictions"]["ATTN_LORA"][0, 0] = np.nan
    else:
        data["simple_policy"] = "ATTN_LORA"
    with pytest.raises(ValueError):
        analyze(data)
