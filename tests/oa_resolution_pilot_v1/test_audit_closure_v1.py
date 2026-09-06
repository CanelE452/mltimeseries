"""A01-A15: the audit's own regression tests.

None of these fits a model. They check that the audit's claims hold against the
stored artifacts, that the fairness properties the pilot asserted are still true,
and that the two semantic bugs the audit found cannot come back.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
ORIG = REPO / "results" / "oa_resolution_pilot_v1"
AUDIT = ORIG / "audit_closure_v1"

pytestmark = pytest.mark.skipif(not AUDIT.exists(),
                                reason="audit artifacts not built; run scripts/audit_oa_resolution_pilot_v1.py")


def load(name: str) -> dict:
    return json.loads((AUDIT / name).read_text(encoding="utf-8"))


# --------------------------------------------------------------- A01 immutability


def test_A01_original_artifacts_unchanged():
    imm = load("original_artifact_immutability.json")
    assert imm["changed"] == []
    assert imm["status"] == "ORIGINAL_ARTIFACTS_UNCHANGED"

    manifest = load("original_artifact_manifest.json")["files"]
    import hashlib

    for rel, rec in manifest.items():
        h = hashlib.sha256((ORIG / rel).read_bytes()).hexdigest()
        assert h == rec["sha256"], rel


# ------------------------------------------------------------------ A02 arithmetic


def test_A02_metrics_recomputation_matches_stored_contrasts():
    arith = load("arithmetic_revalidation.json")
    assert arith["status"] == "ARITHMETIC_REPRODUCTION_OK"
    assert arith["max_abs_delta_pp"] <= arith["tolerance_pp"]


def test_A02_primary_contrast_is_still_near_zero_and_negative():
    """The audit must not have quietly moved the headline number."""
    arith = load("arithmetic_revalidation.json")
    om = arith["roles"]["unseen_interpolation"]["O_vs_M"]
    assert om["recomputed_pct"] == pytest.approx(om["stored_pct"], abs=1e-10)
    assert -1.0 < om["recomputed_pct"] < 0.0
    assert om["n_cells"] == 8


# ------------------------------------------------------------- A03/A04 M/O parity


def test_A03_M_and_O_initial_state_identical_under_one_seed():
    from experiments.oa_resolution_pilot_v1 import model as M

    def state(arm):
        torch.manual_seed(2026090601)
        return M.build(arm).state_dict()

    a, b = state("M"), state("O")
    assert a.keys() == b.keys()
    for k in a:
        assert torch.equal(a[k], b[k]), k


def test_A04_M_and_O_parameter_counts_equal():
    from experiments.oa_resolution_pilot_v1 import model as M

    assert M.parameter_count(M.build("M")) == M.parameter_count(M.build("O"))

    fm = pd.read_csv(ORIG / "fit_manifest.csv")
    counts = fm[fm.arm.isin(["M", "O"])]["n_parameters"].unique()
    assert len(counts) == 1


def test_A03_time_basis_is_input_not_parameter():
    """The thing that differs between M and O must not be a trainable tensor."""
    from experiments.oa_resolution_pilot_v1.operators import observation_support, token_features

    sup = observation_support(6, "INTERVAL_MEAN")
    fm = token_features(6, "INTERVAL_MEAN", "M")
    fo = token_features(6, "INTERVAL_MEAN", "O")
    assert isinstance(fo["time_basis"], np.ndarray)
    assert not np.allclose(fm["time_basis"], fo["time_basis"])
    assert np.array_equal(fm["width"], fo["width"])
    assert np.array_equal(fm["support"], fo["support"]) and len(sup) == len(fm["op_id"])


# ------------------------------------------------------------------- A05 schedule


def test_A05_schedule_sha_identical_across_arms():
    fm = pd.read_csv(ORIG / "fit_manifest.csv")
    per = fm.groupby(["dataset", "model_seed"])["schedule_sha"].nunique()
    assert (per == 1).all(), per.to_dict()
    assert load("audit_verdict.json")["train_schedule_fairness"] == "OK"


# ------------------------------------------------------- A06 unseen r stayed unseen


def test_A06_unseen_resolutions_absent_from_training_and_selection():
    from experiments.oa_resolution_pilot_v1.train import CONDITION_CYCLE, TRAIN_R

    trained = {r for r, _ in CONDITION_CYCLE}
    assert trained == set(TRAIN_R) == {2, 4, 8}
    assert 3 not in trained and 6 not in trained

    spec = json.loads((ORIG / "execution_spec.json").read_text(encoding="utf-8"))
    assert spec["train_report_intervals_r"] == [2, 4, 8]
    assert spec["unseen_interpolation_r"] == [3, 6]
    assert "TRAINING resolutions only" in spec["checkpoint_selection"]

    # the validation loop that feeds checkpoint selection iterates the same cycle
    import inspect

    from experiments.oa_resolution_pilot_v1 import train as T

    src = inspect.getsource(T.validation_loss)
    assert "CONDITION_CYCLE" in src


# --------------------------------------------------- A07/A08 FlowState semantics


@pytest.mark.parametrize("r", [2, 3, 4, 6, 8, 12])
def test_A07_native_end_bin_cannot_define_hourly_mean(r):
    from experiments.oa_resolution_pilot_v1.references import can_form_hourly_mean_from_native

    ok, _ = can_form_hourly_mean_from_native("END_BIN", r)
    assert ok is False


def test_A07_end_bin_counterexample_is_arithmetically_real():
    """A perfect END_BIN forecaster is biased against the hourly-mean target."""
    truth = np.arange(1.0, 7.0)
    assert truth.mean() == 3.5
    for r, expected in ((2, 4.0), (3, 4.5), (6, 6.0)):
        end_bin = truth[r - 1 :: r]
        assert end_bin.mean() == pytest.approx(expected)
        assert end_bin.mean() != pytest.approx(truth.mean())


def test_A08_native_interval_mean_tiles_hour_correctly():
    from experiments.oa_resolution_pilot_v1.references import can_form_hourly_mean_from_native

    truth = np.arange(1.0, 7.0)
    for r in (2, 3, 6):
        ok, _ = can_form_hourly_mean_from_native("INTERVAL_MEAN", r)
        assert ok is True
        blocks = truth.reshape(6 // r, r).mean(axis=1)
        assert blocks.mean() == pytest.approx(truth.mean())
    for r in (4, 8, 12):
        assert can_form_hourly_mean_from_native("INTERVAL_MEAN", r)[0] is False


def test_A07_audit_marks_historical_end_bin_rows_not_comparable():
    fs = load("flowstate_reference_semantic_audit.json")
    for ds, entry in fs["FLOWSTATE_NATIVE_RATE"].items():
        assert entry["END_BIN"]["status"] == "INVALID_FOR_CORE_HOURLY_MEAN_METRIC"
        for cell in entry["END_BIN"]["cells"]:
            assert cell["note"] == "HISTORICAL_NUMBER_NOT_COMPARABLE"
    assert "TARGET_VALIDATION_TUNED_PREPROCESSOR" in \
        fs["FLOWSTATE_RESAMPLED"]["information_condition"]


# --------------------------------------------------------------- A09 r = 12 table


def test_A09_r12_table_joins_on_exact_keys():
    t = pd.read_csv(AUDIT / "r12_seed_table_corrected.csv")
    assert len(t) == 2 * 2 * 2                      # dataset x operation x seed
    assert set(t["r"]) == {12}
    assert set(t.columns) >= {"dataset", "operation", "r", "seed", "R", "M", "O"}

    full = pd.concat([
        pd.DataFrame(np.load(REPO / "runs/oa_resolution_pilot_v1/errors" /
                             f"errors_{ds}_{arm}_{seed}.npy")).assign(dataset=ds)
        for ds in ("jena", "uci") for arm in ("R", "M", "O")
        for seed in (2026090601, 2026090602)], ignore_index=True)
    for c in ("split", "operation", "arm"):
        full[c] = full[c].astype(str)
    full = full[(full.split == "test") & (full.r == 12)]
    full["primary"] = 0.5 * (full.SE10 / full.count10) + 0.5 * (full.SE60 / full.count60)
    ref = full.groupby(["dataset", "operation", "seed", "arm"])["primary"].mean()

    for _, row in t.iterrows():
        for arm in ("R", "M", "O"):
            key = (row["dataset"], row["operation"], row["seed"], arm)
            assert row[arm] == pytest.approx(ref[key], abs=1e-12), key


def test_A09_status_table_r_column_came_from_the_wrong_operation():
    """Locks in the bug the audit found, so a regenerated table cannot repeat it."""
    prov = load("r12_effect_decomposition.json")["status_md_r_column_provenance"]
    assert prov["r_column_wrong"] is True
    assert prov["m_and_o_columns_correct"] is True
    assert prov["scope"] == "REPORTING_CELL_MAPPING_BUG_ONLY"
    for seed, c in prov["per_seed"].items():
        assert c["printed_R_matches_END_BIN"] is True
        assert c["printed_R_matches_INTERVAL_MEAN"] is False


def test_A09_aggregate_metrics_were_not_affected():
    d = load("r12_effect_decomposition.json")["metrics_csv_vs_per_seed_mean"]
    assert d["aggregate_is_consistent"] is True
    assert d["max_abs_delta"] < 1e-9


# ------------------------------------------------------------ A10 raw error truth


def test_A10_raw_error_git_status_reported_accurately():
    inv = load("raw_error_inventory.json")
    assert inv["status"] == "LOCAL_12_RAW_ERRORS_PRESENT_GIT_UNTRACKED"
    assert inv["n_git_tracked"] == 0
    assert inv["n_local"] == 12
    assert all(f["sha256"] for f in inv["files"])
    assert "runs/" in (inv["gitignore_rule"] or "")


# --------------------------------------------------------------- A11 bootstrap


def test_A11_bootstrap_reproduces_from_sufficient_statistics():
    b = load("bootstrap_revalidation.json")
    assert b["status"] == "BOOTSTRAP_REPRODUCED"
    assert b["max_abs_delta_pp"] <= b["tolerance_pp"]
    for key in ("O_vs_M_unseen_interpolation", "O_vs_R_unseen_interpolation",
                "M_vs_R_unseen_interpolation"):
        assert b[key]["recomputed"]["draws"] == 1000


def test_A11_sufficient_stats_are_committable_and_complete():
    s = pd.read_csv(AUDIT / "bootstrap_block_sufficient_stats.csv")
    assert set(s.columns) == {"dataset", "block_id", "operation", "r", "arm",
                              "primary_sum", "n_keys"}
    assert set(s["arm"]) == {"R", "M", "O"}
    assert set(s["r"]) == {3, 6, 12}
    assert (s["n_keys"] > 0).all()
    assert (AUDIT / "bootstrap_block_sufficient_stats.csv").stat().st_size < 2_000_000


def test_A11_bootstrap_name_corrected():
    b = load("bootstrap_revalidation.json")["naming_correction"]
    assert "moving-block" not in b["correct_name"].lower()
    assert "time-block" in b["correct_name"] or "cluster" in b["correct_name"]


# ------------------------------------------------------------------ A12 verdict


def test_A12_original_scientific_decision_untouched():
    v = load("audit_verdict.json")
    assert v["scientific_decision_original"] == "INCONCLUSIVE"
    assert json.loads((ORIG / "verdict.json").read_text(encoding="utf-8"))[
        "scientific_decision"] == "INCONCLUSIVE"
    assert v["audit_recommendation_current_implementation"] == \
        "STOP_SCALING_CURRENT_INTERVAL_INTEGRATED_FOURIER_O"
    assert v["broader_topic_status"] == "OPEN_NOT_DIRECTLY_TESTED"
    assert v["core_model_fits_in_audit"] == 0
    for banned in ("NO_GO", "GO", "SCREEN_NEGATIVE", "PROMISING"):
        assert v["scientific_decision_original"] != banned


# ---------------------------------------------------------- A13 no training here


def test_A13_audit_script_does_not_train():
    src = (REPO / "scripts" / "audit_oa_resolution_pilot_v1.py").read_text(encoding="utf-8")
    for banned in ("train_arm", "backward(", "optim.", "AdamW", "build_schedule",
                   "select_lambda", "from experiments.oa_resolution_pilot_v1 import report",
                   "from experiments.oa_resolution_pilot_v1.report"):
        assert banned not in src, banned
    assert "import torch" not in src


# ----------------------------------------------------- A14/A15 report discipline


def test_A14_status_is_backed_by_audit_artifacts():
    status = (AUDIT / "STATUS.md").read_text(encoding="utf-8")
    v = load("audit_verdict.json")
    assert "ORIGINAL SCIENTIFIC DECISION:\nINCONCLUSIVE" in status
    assert v["audit_recommendation_current_implementation"] in status
    assert v["broader_topic_status"] in status
    for required in ("original_artifact_manifest.json", "arithmetic_revalidation.json",
                     "bootstrap_revalidation.json", "raw_error_inventory.json",
                     "r12_seed_table_corrected.csv",
                     "flowstate_reference_semantic_audit.json"):
        assert (AUDIT / required).exists(), required


def test_A15_no_causal_or_superiority_overclaim_in_audit_text():
    recon = load("reconstruction_interpretation_audit.json")
    assert "caused" in recon["withdrawn_statement"]
    assert "consistent with" in recon["allowed_statement"]

    mech = load("mechanism_interpretation_audit.json")
    assert "does not show" in mech["what_it_does_not_show"]
    assert len(mech["direct_counterexamples"]) >= 1

    width = load("width_interpretation_audit.json")
    assert width["withdrawn_statement"] == "The model does not use interval width."

    status = (AUDIT / "STATUS.md").read_text(encoding="utf-8").lower()
    for banned in ("state of the art", "sota", "outperforms flowstate", "beat flowstate",
                   "mechanism is proven", "caused the forecasting reversal"):
        assert banned not in status, banned


def test_A15_r12_effect_is_reported_as_one_cell():
    d = load("r12_effect_decomposition.json")
    assert len(d["all_four_cells"]) == 4
    assert d["macro_all_pct"] > 1.0
    assert d["macro_excluding_dominant_pct"] < 0.0     # sign flips without that cell
    assert d["dominant_cell"]["dataset"] == "jena"
    assert d["dominant_cell"]["operation"] == "INTERVAL_MEAN"


def test_A15_wrong_support_counterexample_is_recorded():
    mech = load("mechanism_interpretation_audit.json")
    ce = mech["direct_counterexamples"]
    assert any(c["r"] == 8 and c["dataset"] == "jena" and c["operation"] == "INTERVAL_MEAN"
               for c in ce)
    for c in ce:
        assert c["wrong_support_sensitivity_pct"] > 1.0
        assert c["O_vs_M_pct"] <= 0.0


# ------------------------------------- A16 STATUS numbers must match the artifacts


def _status_text() -> str:
    return (AUDIT / "STATUS.md").read_text(encoding="utf-8")


def test_A16_width_bound_in_status_matches_the_artifact():
    """The pilot quoted a bound taken over averaged cells, which read tighter than the
    diagnostic supports. The audit's number is pinned to the artifact instead."""
    w = load("width_interpretation_audit.json")
    bound = w["max_abs_sensitivity_pct_interval_mean"]
    assert bound == pytest.approx(0.08794644286012065, abs=1e-12)

    text = _status_text()
    section = text[text.index("## 17. Width diagnostic"):text.index("## 18.")]

    # the supported claim must carry the artifact's bound, not the understated one
    supported = section[section.index("Supported:"):section.index("Withdrawn:")]
    claim = supported[:supported.index(chr(10) * 2)]
    assert f"{bound:.3f} %" in claim
    assert "0.016 %" not in claim

    # the smaller figure may only appear where it is explained or withdrawn
    withdrawn = text[text.index("## 20. Claims weakened or withdrawn"):text.index("## 21.")]
    assert "0.016 %" in withdrawn
    assert "0.016 %" in section and "maximum" in section


@pytest.mark.parametrize("claim,path", [
    ("-0.049 %", ("arithmetic_revalidation.json", "unseen_interpolation", "O_vs_M")),
    ("-0.182 %", ("arithmetic_revalidation.json", "seen", "O_vs_M")),
])
def test_A16_macro_claims_match_recomputation(claim, path):
    name, role, pair = path
    v = load(name)["roles"][role][pair]["recomputed_pct"]
    assert f"{v:+.3f} %".replace("+", "") in claim or f"{v:.3f} %" == claim.lstrip("+")
    assert claim in _status_text()


def test_A16_reproduction_tolerances_quoted_exactly():
    arith = load("arithmetic_revalidation.json")["max_abs_delta_pp"]
    boot = load("bootstrap_revalidation.json")["max_abs_delta_pp"]
    text = _status_text()
    assert f"{arith:.2e}" in text
    assert f"{boot:.2e}" in text


def test_A16_r12_numbers_match_the_decomposition():
    d = load("r12_effect_decomposition.json")
    text = _status_text()
    assert f"{d['macro_all_pct']:+.2f} %".replace("+", "+") in text
    assert f"{d['macro_excluding_dominant_pct']:.2f} %" in text
    assert f"{d['dominant_cell']['ri_pct']:+.2f} %" in text


def test_A16_counterexample_numbers_match():
    ce = load("mechanism_interpretation_audit.json")["direct_counterexamples"][0]
    text = _status_text()
    assert f"{ce['wrong_support_sensitivity_pct']:+.2f} %" in text
    assert f"{ce['O_vs_M_pct']:.2f} %" in text
