"""Guards for HQ-TOKEN-PILOT-v1-AUDIT-CLOSURE.

These are not tests of the pilot. They check that the audit stayed inside its contract:
it changed nothing, it reproduced the numbers rather than restating them, and it did not
quietly reintroduce any of the five over-claims it was written to withdraw.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
import re

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(ROOT, "results", "hq_token_pilot_v1")
AUD = os.path.join(RES, "audit_closure_v1")
SCRIPTS = os.path.join(ROOT, "scripts")

pytestmark = pytest.mark.skipif(
    not os.path.isdir(AUD), reason="audit has not been run in this checkout"
)


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


@pytest.fixture(scope="module")
def audit():
    return {n[:-5]: load(os.path.join(AUD, n))
            for n in os.listdir(AUD) if n.endswith(".json")}


def test_audit01_original_artifacts_are_byte_identical(audit):
    """T_AUDIT01: every pre-audit artifact still hashes to what it did before."""
    before = audit["original_artifact_manifest"]["captured_before_audit"]
    after = audit["original_artifact_immutability"]["hashes_after"]
    assert audit["original_artifact_immutability"]["status"] == "ORIGINAL_ARTIFACTS_UNCHANGED"
    assert before == after
    for name, digest in before.items():
        p = os.path.join(RES, name)
        if digest is None:
            continue
        assert os.path.exists(p), f"{name} disappeared"
        assert sha256_file(p) == digest, f"{name} changed on disk after the audit"


def test_audit02_contrasts_recompute_from_metrics(audit):
    """T_AUDIT02: recomputed relative improvements match contrasts.json."""
    m = {}
    with open(os.path.join(RES, "metrics.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            m[(r["arm"], r["dataset"], int(r["horizon"]), int(r["model_seed"]))] = float(r["MSE"])
    orig = load(os.path.join(RES, "contrasts.json"))
    seeds = sorted({k[3] for k in m if k[0] == "C"} & {k[3] for k in m if k[0] == "I"})
    for d in ("ETTm2", "weather", "electricity"):
        for h in (96, 336):
            ec = np.mean([m[("C", d, h, s)] for s in seeds])
            ei = np.mean([m[("I", d, h, s)] for s in seeds])
            ri = 100.0 * (1.0 - ec / ei)
            assert abs(ri - orig["primary"]["cells"][f"{d}|{h}"]) <= 1e-10
    assert audit["arithmetic_revalidation"]["status"] == "ARITHMETIC_REPRODUCTION_OK"
    assert audit["arithmetic_revalidation"]["max_abs_difference_pct_points"] <= 1e-10


def test_audit03_arms_share_the_training_schedule():
    """T_AUDIT03: I, C and R consume the same sample schedule at a given dataset and seed."""
    core = os.path.join(ROOT, "runs", "hq_token_pilot_v1", "core")
    if not os.path.isdir(core):
        pytest.skip("raw run artifacts not present in this checkout")
    rows = list(csv.DictReader(open(os.path.join(AUD, "schedule_fairness.csv"), encoding="utf-8")))
    assert rows, "schedule fairness table is empty"
    for r in rows:
        assert r["all_identical"] == "True", f"schedule mismatch at {r['dataset']} {r['model_seed']}"


def test_audit04_capacity_control_is_not_promoted_to_joint_success(audit):
    """T_AUDIT04: an H=96 win cannot be reported as solving the joint objective."""
    cap = audit["capacity_control_audit"]
    joint = cap["joint_equal_weight_mean_mse"]
    assert cap["JOINT_HORIZON_TASK_SOLVED"] == (joint["C"] < joint["I"])
    if joint["C"] >= joint["I"]:
        assert cap["JOINT_HORIZON_TASK_SOLVED"] is False
        text = json.dumps(cap).lower()
        assert "successfully learned horizon-conditioned forecasting" not in text
        assert "completely excluded" not in text
        assert cap["what_this_does_not_support"]


def test_audit05_paired_seed_variability_is_not_claimed(audit):
    """T_AUDIT05: an I-only seed study cannot stand in for the paired C minus I spread."""
    s = audit["seed_uncertainty_audit"]
    assert s["paired_CI_effect_seed_variability_estimated"] is False
    assert s["core_pair_seed_count"] == 2
    assert "only for I" in s["reason"]
    assert s["explicitly_withdrawn_statement"]


def test_audit06_no_artifact_calls_the_compression_gap_an_upper_bound(audit):
    """T_AUDIT06: no audit field asserts DENSE minus U bounds the achievable gain."""
    gap = audit["compression_gap_audit"]
    assert gap["is_an_upper_bound"] is False
    assert "OBSERVED_DENSE_VS_UNIFORM_COMPRESSION_GAP" in gap
    for name, doc in audit.items():
        flat = json.dumps(doc)
        for m in re.finditer(r'"([A-Za-z_]*upper_bound[A-Za-z_]*)"\s*:\s*(true|false)', flat):
            field, value = m.group(1), m.group(2)
            if value == "true":
                assert "NOT" in field.upper(), f"{name}.{field} asserts an upper bound"
        assert "maximum headroom" not in flat.lower()


def test_audit07_original_scientific_decision_is_preserved(audit):
    """T_AUDIT07: the pre-registered decision is carried through unchanged."""
    original = load(os.path.join(RES, "verdict.json"))["scientific_decision"]
    assert original == "INCONCLUSIVE"
    assert audit["audit_verdict"]["scientific_decision_original"] == original
    for forbidden in ("SCREEN_NEGATIVE", "NO_GO", "GO"):
        assert audit["audit_verdict"].get("scientific_decision") != forbidden


def test_audit08_implementation_and_topic_verdicts_are_separate(audit):
    """T_AUDIT08: the implementation recommendation is not the topic status."""
    v = audit["audit_verdict"]
    assert v["audit_recommendation_current_implementation"] == "STOP_SCALING_CURRENT_FIXED_NEIGHBORHOOD_C"
    assert v["broader_topic_status"] == "OPEN_NOT_DIRECTLY_TESTED"
    assert v["audit_recommendation_current_implementation"] != v["broader_topic_status"]
    assert set(v["meaning"]) == {"STOP_SCALING_CURRENT_FIXED_NEIGHBORHOOD_C",
                                 "OPEN_NOT_DIRECTLY_TESTED"}


def test_audit09_audit_scripts_launch_no_training(audit):
    """T_AUDIT09: nothing in the audit path can start a core fit."""
    assert audit["audit_verdict"]["no_model_fit_performed"] is True
    forbidden = [
        r"experiments\.hq_token_pilot_v1\.run",
        r"\bfrom\s+experiments\.hq_token_pilot_v1\s+import\s+train\b",
        r"\bT\.fit\(", r"\bstage_core\(", r"\bstage_noise_floor\(",
        r"\bstage_capacity_check\(", r"\.backward\(", r"\boptim\b",
    ]
    for name in ("audit_hq_token_pilot_v1.py", "audit_status_hq_token_pilot_v1.py"):
        src = open(os.path.join(SCRIPTS, name), encoding="utf-8").read()
        for pat in forbidden:
            assert not re.search(pat, src), f"{name} matches forbidden pattern {pat}"
    # the micro-profile may import the model definition, but must not train
    mp = open(os.path.join(SCRIPTS, "audit_microprofile_hq_token_pilot_v1.py"), encoding="utf-8").read()
    assert ".backward(" not in mp and "optim" not in mp
    assert "torch.no_grad" in mp


def test_audit10_status_is_regenerable_from_the_audit_artifacts():
    """T_AUDIT10: STATUS.md is a function of the stored audit files, not of memory."""
    path = os.path.join(SCRIPTS, "audit_status_hq_token_pilot_v1.py")
    spec = importlib.util.spec_from_file_location("audit_status", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    regenerated = mod.build(AUD)
    with open(os.path.join(AUD, "STATUS.md"), encoding="utf-8") as f:
        stored = f.read()
    assert regenerated == stored, "STATUS.md does not match what the audit artifacts produce"
    for line in ("HQ-TOKEN-PILOT-v1 AUDIT RESULT",
                 "ORIGINAL SCIENTIFIC DECISION: INCONCLUSIVE",
                 "CURRENT IMPLEMENTATION RECOMMENDATION: STOP_SCALING_CURRENT_FIXED_NEIGHBORHOOD_C",
                 "BROADER TOPIC: OPEN_NOT_DIRECTLY_TESTED"):
        assert line in stored
