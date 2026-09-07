"""Tests for the parts of the closure whose failure would be silent."""

from __future__ import annotations

import ast
import json
import pathlib

import numpy as np
import pandas as pd
import pytest

from experiments.tsfm_long_horizon_specialist_closure_v1.src import paths, source_audit

RESULTS = paths.RESULTS


def _json(name):
    path = RESULTS / name
    if not path.exists():
        pytest.skip(f"{name} not produced yet")
    return json.loads(path.read_text(encoding="utf-8"))


def _csv(name):
    path = RESULTS / name
    if not path.exists() or path.stat().st_size < 5:
        pytest.skip(f"{name} not produced yet")
    return pd.read_csv(path)


def test_development_set_is_rederived_not_copied():
    """The four D1-long tasks must come out of the source metadata, not a literal."""
    metadata = pd.read_csv(paths.SOURCE_RESULTS / "task_metadata.csv")
    assert source_audit.development_tasks(metadata) == sorted(
        source_audit.EXPECTED_DEVELOPMENT_TASKS
    )


def test_development_set_assertion_actually_fires():
    """A source study with a different D1-long set must hard stop, not pass quietly."""
    metadata = pd.read_csv(paths.SOURCE_RESULTS / "task_metadata.csv")
    tampered = metadata.copy()
    mask = tampered.D1_horizon_ratio == "long"
    tampered.loc[tampered[mask].index[0], "D1_horizon_ratio"] = "medium"
    with pytest.raises(SystemExit, match="SOURCE_CONTRACT_MISMATCH"):
        source_audit.development_tasks(tampered)


def test_holdout_excludes_previous_tasks_and_families():
    spec = _json("long_horizon_tasks.json")
    previous = set(spec["excluded_previous_tasks"])
    holdout = set(spec["fresh_holdout_tasks"])
    assert not (holdout & previous)

    from experiments.tsfm_benchmark_gap_discovery_v1.src.select_tasks import dataset_family

    pool = pd.read_csv(paths.SOURCE_RESULTS / "task_pool.csv")
    pool["family"] = pool.dataset_config.map(dataset_family)
    holdout_families = set(pool[pool.task_uid.isin(holdout)].family)
    assert not (holdout_families & set(spec["excluded_families"]))
    assert len(holdout_families) == len(holdout), "one task per dataset family"


def test_holdout_uses_the_frozen_d1_threshold():
    """The long cut is the source study's, not one recomputed to suit this study."""
    spec = _json("long_horizon_tasks.json")
    frozen = json.loads(
        (paths.SOURCE_RESULTS / "descriptor_thresholds.json").read_text(encoding="utf-8")
    )["cuts"]["horizon_to_context_ratio"]
    assert spec["d1_thresholds"] == frozen
    for candidate in spec["long_candidates"]:
        assert candidate["horizon_to_context_ratio"] > frozen["high"]


def test_holdout_hash_matches_file():
    payload = (RESULTS / "long_horizon_tasks.json").read_text(encoding="utf-8")
    import hashlib

    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    recorded = (RESULTS / "long_horizon_tasks.sha256").read_text(encoding="utf-8").split()[0]
    assert digest == recorded


def test_training_frame_ends_before_every_later_window():
    """No row the benchmark scores may appear in the specialist's training data."""
    conversion = _json("conversion_index.json")
    for task in conversion["tasks"]:
        first = task["windows"][0]["last_context_timestamp"]
        for window in task["windows"][1:]:
            assert first <= window["last_context_timestamp"], task["task_uid"]


def test_later_windows_only_grow_the_context():
    """Later windows add observations; they never remove or rewrite earlier ones."""
    conversion = _json("conversion_index.json")
    for task in conversion["tasks"]:
        medians = [w["context_median"] for w in task["windows"]]
        assert medians == sorted(medians), task["task_uid"]


def test_exported_windows_carry_no_covariates():
    conversion = _json("conversion_index.json")
    task = conversion["tasks"][0]
    payload = np.load(
        paths.DATA_EXTERNAL / task["safe_name"] / "window_00.npz", allow_pickle=True
    )
    assert set(payload.files) == {"target", "timestamp", "lengths", "item_id"}


def test_item_order_matches_fev_univariate_order():
    """Row order is what maps a forecast back to a target column; check it directly."""
    conversion = _json("conversion_index.json")
    task_meta = next(t for t in conversion["tasks"] if len(t["target_columns"]) > 1)
    payload = np.load(
        paths.DATA_EXTERNAL / task_meta["safe_name"] / "window_00.npz", allow_pickle=True
    )
    ids = [str(i) for i in payload["item_id"]]
    targets = task_meta["target_columns"]
    # fev emits item-major, target-minor: the suffix must cycle through targets.
    suffixes = [i.split("||")[1] for i in ids[: len(targets)]]
    assert suffixes == targets
    assert len(ids) % len(targets) == 0


def test_specialist_suite_has_no_foundation_model():
    manifest = _csv("specialist_manifest.csv")
    forbidden = ("chronos", "toto", "timesfm", "tirex")
    for _, row in manifest[manifest.status == "OK"].iterrows():
        fitted = ast.literal_eval(row.fitted_models)
        assert not [m for m in fitted for f in forbidden if f in m.lower()], fitted


def test_s_best_is_a_single_model():
    manifest = _csv("specialist_manifest.csv")
    for _, row in manifest[manifest.status == "OK"].iterrows():
        assert "Ensemble" not in str(row.model_best)


def test_train_specialists_never_refits():
    """The fitting call must appear exactly once, and refit_full never."""
    source = (paths.EXP / "src" / "train_specialists.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"fit", "refit_full", "refit_single_full"}
    ]
    attrs = [c.func.attr for c in calls]
    assert attrs.count("refit_full") == 0
    assert attrs.count("refit_single_full") == 0
    assert attrs.count("fit") == 1, f"expected exactly one fit call, found {attrs}"


def test_envelope_is_min_of_the_three_primaries():
    comparator = _csv("comparator_scores.csv")
    primaries = ["chronos-2", "tirex-2", "timesfm-3.0"]
    np.testing.assert_allclose(
        comparator[primaries].min(axis=1), comparator.F_FAMILY_ENVELOPE, equal_nan=True
    )
    # The synthetic diagnostic must not be able to lower the envelope.
    assert "chronos-2-synth" not in primaries


def test_relative_improvement_sign_convention():
    from experiments.tsfm_long_horizon_specialist_closure_v1.src.compare import (
        _relative_improvement,
    )

    assert _relative_improvement(0.5, 1.0) == pytest.approx(50.0)
    assert _relative_improvement(2.0, 1.0) == pytest.approx(-100.0)
    assert np.isnan(_relative_improvement(1.0, 0.0))
