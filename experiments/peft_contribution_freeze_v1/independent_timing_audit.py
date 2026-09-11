"""Independent CPU-only audit for Study31 timing replay and resource-matched analysis.

This script deliberately avoids torch and the Study31 analyse/policy modules.  It is
meant to run only after the corresponding timing replay or analysis artifacts exist.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import traceback
from typing import Any, Iterable

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "runs/peft_contribution_freeze_v1"
RESULTS = ROOT / "results/peft_contribution_freeze_v1"
MATCHED = RESULTS / "resource_matched"
REPLAY_AUDIT = RUN / "independent_timing_replay_audit.json"
ANALYSIS_AUDIT = RUN / "independent_timing_analysis_audit.json"
ARMS = ("FULL", "ES2", "FIXED_FREEZE", "CONTRIB_FREEZE")
NONFULL_ARMS = ("ES2", "FIXED_FREEZE", "CONTRIB_FREEZE")
TOL = 1e-10
COST_FIELDS = {"fit_seconds", "fit_savings_percent", "original_fit_seconds"}
RESULT_SCIENCE_FIELDS = (
    "completed",
    "job",
    "best_score",
    "best_step",
    "steps",
    "cap",
    "stop_reason",
    "frozen_step",
    "adapter_updates",
    "frozen_tail_verified",
    "toggle_checks",
    "trainable",
    "snapshot_names",
    "initial_head_hash",
    "identity_error",
    "frozen_hash",
    "frozen_verified",
    "checkpoint_sha256",
    "replay_verified",
    "contract_sha256",
    "holdout_opened",
    "sample_index_sha256",
)
SUMMARY_STABLE_FIELDS = (
    "F0",
    "gates",
    "independent_source_count",
    "seeds",
    "cells_not_independent",
    "new_method_claim",
    "closest_prior_AFLoRA_implemented",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def rel(path: Path | str) -> str:
    p = Path(path).resolve()
    try:
        return p.relative_to(ROOT).as_posix()
    except ValueError:
        return str(p)


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_once(path: Path, payload: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    return sha256_file(path)


def fail_once(path: Path, mode: str, exc: BaseException) -> None:
    if path.exists():
        return
    payload = {
        "completed": False,
        "mode": mode,
        "created_utc": utc_now(),
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback": traceback.format_exc(),
    }
    try:
        write_once(path, payload)
    except FileExistsError:
        pass


def assert_no_existing_output(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing audit output: {rel(path)}")


def ensure(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def assert_close(name: str, actual: float, expected: float, tol: float = TOL) -> float:
    if not math.isfinite(float(actual)) or not math.isfinite(float(expected)):
        ensure(actual == expected, f"{name}: nonfinite mismatch {actual!r} != {expected!r}")
        return 0.0
    err = abs(float(actual) - float(expected))
    if err > tol:
        raise AssertionError(f"{name}: {actual!r} != {expected!r}, abs_error={err}")
    return err


def parse_scalar(value: str) -> Any:
    if value == "":
        return None
    if value == "None":
        return None
    if value == "True":
        return True
    if value == "False":
        return False
    text = value.strip()
    if text.startswith("{") or text.startswith("["):
        return json.loads(text.replace("'", '"'))
    try:
        if any(mark in text for mark in (".", "e", "E")):
            return float(text)
        return int(text)
    except ValueError:
        return value


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        rows = []
        for row in reader:
            rows.append({key: parse_scalar(value) for key, value in row.items()})
    return rows


def row_key(row: dict[str, Any]) -> tuple[str, str, int, str]:
    return (str(row["dataset"]), str(row["condition"]), int(row["seed"]), str(row["arm"]))


def plan_job_key(entry: dict[str, Any]) -> tuple[str, str, int, str]:
    job = entry["job"]
    return (str(job["dataset"]), str(job["condition"]), int(job["seed"]), str(job["arm"]))


def duplicated(values: Iterable[Any]) -> list[Any]:
    seen = set()
    dup = []
    for value in values:
        if value in seen and value not in dup:
            dup.append(value)
        seen.add(value)
    return dup


def validate_timing_contract() -> dict[str, Any]:
    contract_path = RUN / "timing_contract.json"
    contract = read_json(contract_path)
    source_checks = {}
    input_checks = {}
    for item, expected in contract.get("source_hashes", {}).items():
        actual = sha256_file(ROOT / item)
        source_checks[item] = actual == expected
    for item, expected in contract.get("input_hashes", {}).items():
        actual = sha256_file(ROOT / item)
        input_checks[item] = actual == expected
    bad_sources = [item for item, ok in source_checks.items() if not ok]
    bad_inputs = [item for item, ok in input_checks.items() if not ok]
    if bad_sources or bad_inputs:
        raise AssertionError(f"timing contract hash mismatch: sources={bad_sources}, inputs={bad_inputs}")
    keys = [entry["key"] for entry in contract["jobs"]]
    ensure(len(keys) == 13, f"timing contract should contain 13 jobs, got {len(keys)}")
    ensure(not duplicated(keys), f"timing contract duplicate keys: {duplicated(keys)}")
    ensure(contract.get("E_started_at_creation") is False, "timing contract must be sealed before E")
    return {
        "path": rel(contract_path),
        "sha256": sha256_file(contract_path),
        "job_count": len(keys),
        "source_hash_checks": source_checks,
        "input_hash_checks": input_checks,
    }


def guard_ok(path: Path, label: str) -> dict[str, Any]:
    status = read_json(path)
    ensure(status.get("completed") is True, f"{label}: guard not completed")
    ensure(status.get("returncode") == 0, f"{label}: returncode is not zero")
    ensure(status.get("reasons") == [], f"{label}: guard reasons not empty")
    ensure(status.get("state") == "completed", f"{label}: guard state is not completed")
    return {
        "path": rel(path),
        "elapsed_seconds": float(status["elapsed_seconds"]),
        "returncode": status.get("returncode"),
    }


def drop_utc_fields(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: drop_utc_fields(child)
            for key, child in value.items()
            if "utc" not in key.lower()
        }
    if isinstance(value, list):
        return [drop_utc_fields(item) for item in value]
    return value


def compare_result_science(original: dict[str, Any], replay: dict[str, Any], label: str) -> dict[str, Any]:
    mismatches = []
    checked = []
    for field in RESULT_SCIENCE_FIELDS:
        ensure(field in original, f"{label}: original result missing {field}")
        ensure(field in replay, f"{label}: replay result missing {field}")
        checked.append(field)
        if original[field] != replay[field]:
            mismatches.append(field)
    if mismatches:
        raise AssertionError(f"{label}: result science field mismatch {mismatches}")
    original_history = drop_utc_fields(original.get("history"))
    replay_history = drop_utc_fields(replay.get("history"))
    ensure(original_history == replay_history, f"{label}: history differs after dropping utc fields")
    return {"fields_checked": checked, "history_equal_utc_excluded": True}


def array_equal_nan(left: np.ndarray, right: np.ndarray) -> bool:
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    if np.issubdtype(left.dtype, np.floating) or np.issubdtype(left.dtype, np.complexfloating):
        return bool(np.array_equal(left, right, equal_nan=True))
    return bool(np.array_equal(left, right))


def compare_npz(left: Path, right: Path, label: str) -> dict[str, Any]:
    with np.load(left, allow_pickle=False) as a, np.load(right, allow_pickle=False) as b:
        keys_a = sorted(a.files)
        keys_b = sorted(b.files)
        ensure(keys_a == keys_b, f"{label}: npz keys differ {keys_a} != {keys_b}")
        checked = []
        for key in keys_a:
            left_arr = np.asarray(a[key])
            right_arr = np.asarray(b[key])
            if not array_equal_nan(left_arr, right_arr):
                raise AssertionError(f"{label}: array mismatch for key={key}, left={left_arr.shape}/{left_arr.dtype}, right={right_arr.shape}/{right_arr.dtype}")
            checked.append({"array": key, "shape": list(left_arr.shape), "dtype": str(left_arr.dtype)})
    return {"file": left.name, "arrays_checked": checked}


def compare_all_prediction_npz(original_out: Path, replay_out: Path, label: str) -> dict[str, Any]:
    original_files = sorted(path.name for path in original_out.glob("point_*.npz")) + ["val_predictions.npz"]
    replay_files = sorted(path.name for path in replay_out.glob("point_*.npz")) + ["val_predictions.npz"]
    ensure(original_files == replay_files, f"{label}: point/val npz file set mismatch")
    details = []
    arrays_checked = 0
    for name in original_files:
        detail = compare_npz(original_out / name, replay_out / name, f"{label}/{name}")
        arrays_checked += len(detail["arrays_checked"])
        details.append(detail)
    return {"npz_files_checked": len(original_files), "arrays_checked": arrays_checked, "files": details}


def replay_rows_by_key(timing_completed: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = timing_completed.get("rows", [])
    keys = [row["key"] for row in rows]
    ensure(len(keys) == 13, f"timing_completed should contain 13 rows, got {len(keys)}")
    ensure(not duplicated(keys), f"timing_completed duplicate keys: {duplicated(keys)}")
    return {row["key"]: row for row in rows}


def run_replay_audit(output: Path = REPLAY_AUDIT) -> dict[str, Any]:
    assert_no_existing_output(output)
    contract_status = validate_timing_contract()
    contract = read_json(RUN / "timing_contract.json")
    timing_completed_path = RUN / "timing_completed.json"
    timing_completed = read_json(timing_completed_path)
    ensure(timing_completed.get("completed") is True, "timing_completed is not completed")
    assert_close(
        "timing_completed contract sha",
        0.0 if timing_completed.get("contract_sha256") == contract_status["sha256"] else 1.0,
        0.0,
        0.0,
    )
    replay_by_key = replay_rows_by_key(timing_completed)
    contract_keys = [entry["key"] for entry in contract["jobs"]]
    ensure(set(replay_by_key) == set(contract_keys), "timing_completed keys do not exactly match timing_contract jobs")

    row_details = []
    total_npz_files = 0
    total_arrays = 0
    for entry in contract["jobs"]:
        key = entry["key"]
        replay_row = replay_by_key[key]
        replay_key = replay_row["replay_key"]
        original_dir = RUN / key
        replay_dir = RUN / replay_key
        original_guard = guard_ok(original_dir / "guard/status.json", f"original/{key}")
        replay_guard = guard_ok(replay_dir / "guard/status.json", f"replay/{key}")
        assert_close(f"{key}: replay row fit_seconds", replay_row["fit_seconds"], replay_guard["elapsed_seconds"])

        original_result = read_json(original_dir / "output/result.json")
        replay_result = read_json(replay_dir / "output/result.json")
        ensure(original_result.get("job") == replay_result.get("job") == entry["job"], f"{key}: job mismatch")
        science = compare_result_science(original_result, replay_result, key)
        ensure(sha256_file(original_dir / "output/best.pt") == original_result["checkpoint_sha256"], f"{key}: original best.pt sha mismatch")
        ensure(sha256_file(replay_dir / "output/best.pt") == original_result["checkpoint_sha256"], f"{key}: replay best.pt sha mismatch")
        npz = compare_all_prediction_npz(original_dir / "output", replay_dir / "output", key)
        total_npz_files += npz["npz_files_checked"]
        total_arrays += npz["arrays_checked"]
        row_details.append({
            "key": key,
            "replay_key": replay_key,
            "original_guard": original_guard,
            "replay_guard": replay_guard,
            "result_science": science,
            "checkpoint_sha256": original_result["checkpoint_sha256"],
            "npz_files_checked": npz["npz_files_checked"],
            "npz_arrays_checked": npz["arrays_checked"],
        })

    payload = {
        "completed": True,
        "mode": "replay",
        "created_utc": utc_now(),
        "contract_status": contract_status,
        "timing_completed_path": rel(timing_completed_path),
        "timing_completed_sha256": sha256_file(timing_completed_path),
        "counts": {
            "contract_jobs": len(contract_keys),
            "timing_completed_rows": len(replay_by_key),
            "duplicate_contract_keys": 0,
            "duplicate_timing_completed_keys": 0,
            "npz_files_checked": total_npz_files,
            "npz_arrays_checked": total_arrays,
        },
        "exact_key_set_match": True,
        "all_original_and_replay_guards_exit0": True,
        "all_histories_equal_utc_excluded": True,
        "all_prediction_npz_arrays_equal_nan": True,
        "rows": row_details,
    }
    payload["output_sha256"] = write_once(output, payload)
    return payload


def keyed_rows(rows: list[dict[str, Any]], label: str) -> dict[tuple[str, str, int, str], dict[str, Any]]:
    keys = [row_key(row) for row in rows]
    ensure(len(keys) == len(set(keys)), f"{label}: duplicate row keys: {duplicated(keys)}")
    return {row_key(row): row for row in rows}


def numeric_equal_or_exact(name: str, actual: Any, expected: Any, tol: float = TOL) -> float:
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)) and not isinstance(actual, bool) and not isinstance(expected, bool):
        return assert_close(name, float(actual), float(expected), tol)
    ensure(actual == expected, f"{name}: {actual!r} != {expected!r}")
    return 0.0


def compare_raw_matched_rows(raw_rows: list[dict[str, Any]], matched_rows: list[dict[str, Any]]) -> dict[str, Any]:
    raw = keyed_rows(raw_rows, "raw metrics")
    matched = keyed_rows(matched_rows, "matched metrics")
    ensure(len(raw) == 48, f"raw metrics should contain 48 rows, got {len(raw)}")
    ensure(len(matched) == 48, f"matched metrics should contain 48 rows, got {len(matched)}")
    ensure(set(raw) == set(matched), "raw/matched row key sets differ")
    stable_fields = sorted((set(raw_rows[0]) & set(matched_rows[0])) - COST_FIELDS)
    max_error = 0.0
    mismatched_fields = []
    for key in sorted(raw):
        for field in stable_fields:
            try:
                err = numeric_equal_or_exact(f"{key}/{field}", matched[key][field], raw[key][field])
                max_error = max(max_error, err)
            except AssertionError:
                mismatched_fields.append((key, field))
    if mismatched_fields:
        raise AssertionError(f"raw/matched quality field mismatches: {mismatched_fields[:8]} total={len(mismatched_fields)}")
    original_time_errors = []
    for key in sorted(raw):
        if "original_fit_seconds" in matched[key]:
            original_time_errors.append(assert_close(f"{key}/original_fit_seconds", matched[key]["original_fit_seconds"], raw[key]["fit_seconds"]))
    return {
        "rows": 48,
        "stable_fields_checked": stable_fields,
        "stable_field_max_abs_error": max_error,
        "original_fit_seconds_max_abs_error": max(original_time_errors) if original_time_errors else None,
    }


def recompute_cell_fields(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keyed = keyed_rows(rows, "matched metrics")
    regret_errors = []
    savings_errors = []
    for key, row in keyed.items():
        ds, condition, seed, _arm = key
        full = keyed[(ds, condition, seed, "FULL")]
        regret = 100.0 * (float(row["E_score"]) - float(full["E_score"])) / float(row["F0_score"])
        saving = 100.0 * (1.0 - float(row["fit_seconds"]) / float(full["fit_seconds"]))
        regret_errors.append(assert_close(f"{key}/regret_pct_F0", row["regret_pct_F0"], regret))
        savings_errors.append(assert_close(f"{key}/fit_savings_percent", row["fit_savings_percent"], saving))
    return {
        "cell_regret_max_abs_error": max(regret_errors),
        "cell_savings_max_abs_error": max(savings_errors),
    }


def source_names_from_rows(rows: list[dict[str, Any]]) -> list[str]:
    return sorted({str(row["dataset"]) for row in rows})


def recompute_metrics(rows: list[dict[str, Any]], plan: dict[str, Any]) -> list[dict[str, Any]]:
    datasets = source_names_from_rows(rows)
    metrics = []
    for seed in plan["seeds"]:
        base = [row for row in rows if int(row["seed"]) == int(seed) and row["arm"] == "FULL"]
        ensure(len(base) == len(datasets) * 3, f"seed {seed}: unexpected FULL base size {len(base)}")
        base_seconds = sum(float(row["fit_seconds"]) for row in base)
        for arm in NONFULL_ARMS:
            group = [row for row in rows if int(row["seed"]) == int(seed) and row["arm"] == arm]
            ensure(len(group) == len(datasets) * 3, f"seed {seed}/{arm}: unexpected group size {len(group)}")
            sources = {
                ds: float(np.mean([float(row["regret_pct_F0"]) for row in group if row["dataset"] == ds]))
                for ds in datasets
            }
            regret = float(np.mean([float(row["regret_pct_F0"]) for row in group]))
            seconds = sum(float(row["fit_seconds"]) for row in group)
            savings = 100.0 * (1.0 - seconds / base_seconds)
            gates = plan["gates"]
            quality = regret <= gates["macro_regret_max_pct_F0"] and max(sources.values()) <= gates["source_regret_max_pct_F0"]
            metrics.append({
                "seed": int(seed),
                "arm": arm,
                "macro_regret_pct_F0": regret,
                "source_regret_pct_F0": sources,
                "fit_seconds": seconds,
                "fit_savings_percent": savings,
                "quality_passed": bool(quality),
                "cost_passed": bool(savings > gates["savings_min_percent"]),
                "gate_passed": bool(quality and savings > gates["savings_min_percent"]),
                "freeze_count": sum(row["frozen_step"] is not None for row in group),
                "stopped_before_cap": sum(int(row["steps"]) < (180 if row["condition"] == "FULL90" else 60) for row in group),
            })
    return metrics


def recompute_dominance(metrics: list[dict[str, Any]], plan: dict[str, Any]) -> list[dict[str, Any]]:
    dominance = []
    for seed in plan["seeds"]:
        candidate = next(row for row in metrics if row["seed"] == int(seed) and row["arm"] == "CONTRIB_FREEZE")
        for arm in ("ES2", "FIXED_FREEZE"):
            baseline = next(row for row in metrics if row["seed"] == int(seed) and row["arm"] == arm)
            no_worse = baseline["macro_regret_pct_F0"] <= candidate["macro_regret_pct_F0"] and baseline["fit_seconds"] <= candidate["fit_seconds"]
            strict = baseline["macro_regret_pct_F0"] < candidate["macro_regret_pct_F0"] or baseline["fit_seconds"] < candidate["fit_seconds"]
            dominance.append({
                "seed": int(seed),
                "baseline": arm,
                "dominates_candidate_macro_quality_cost": bool(no_worse and strict),
            })
    return dominance


def metric_key(row: dict[str, Any]) -> tuple[int, str]:
    return (int(row["seed"]), str(row["arm"]))


def dominance_key(row: dict[str, Any]) -> tuple[int, str]:
    return (int(row["seed"]), str(row["baseline"]))


def compare_metric_dicts(actual: list[dict[str, Any]], expected: list[dict[str, Any]], label: str) -> dict[str, Any]:
    actual_map = {metric_key(row): row for row in actual}
    expected_map = {metric_key(row): row for row in expected}
    ensure(set(actual_map) == set(expected_map), f"{label}: metric keys differ")
    max_error = 0.0
    for key in sorted(actual_map):
        left = actual_map[key]
        right = expected_map[key]
        for field in ("macro_regret_pct_F0", "fit_seconds", "fit_savings_percent"):
            max_error = max(max_error, assert_close(f"{label}/{key}/{field}", left[field], right[field]))
        for field in ("quality_passed", "cost_passed", "gate_passed", "freeze_count", "stopped_before_cap"):
            ensure(left[field] == right[field], f"{label}/{key}/{field}: {left[field]!r} != {right[field]!r}")
        ensure(set(left["source_regret_pct_F0"]) == set(right["source_regret_pct_F0"]), f"{label}/{key}: source keys differ")
        for source in left["source_regret_pct_F0"]:
            max_error = max(max_error, assert_close(f"{label}/{key}/source/{source}", left["source_regret_pct_F0"][source], right["source_regret_pct_F0"][source]))
    return {"metric_count": len(actual_map), "max_abs_error": max_error}


def compare_dominance(actual: list[dict[str, Any]], expected: list[dict[str, Any]], label: str) -> dict[str, Any]:
    actual_map = {dominance_key(row): row for row in actual}
    expected_map = {dominance_key(row): row for row in expected}
    ensure(set(actual_map) == set(expected_map), f"{label}: dominance keys differ")
    for key in sorted(actual_map):
        field = "dominates_candidate_macro_quality_cost"
        ensure(actual_map[key][field] == expected_map[key][field], f"{label}/{key}/{field} mismatch")
    return {"dominance_count": len(actual_map)}


def validate_timing_completed_for_analysis(contract_status: dict[str, Any]) -> dict[str, Any]:
    contract = read_json(RUN / "timing_contract.json")
    replay = read_json(RUN / "timing_completed.json")
    ensure(replay.get("completed") is True, "timing_completed is not completed")
    ensure(replay.get("contract_sha256") == contract_status["sha256"], "timing_completed contract sha mismatch")
    replay_by_key = replay_rows_by_key(replay)
    contract_keys = {entry["key"] for entry in contract["jobs"]}
    ensure(set(replay_by_key) == contract_keys, "timing_completed rows do not exactly cover contract jobs")
    for key, row in replay_by_key.items():
        ensure(row.get("checkpoint_exact_match") is True, f"{key}: checkpoint_exact_match false")
        status = guard_ok(RUN / row["replay_key"] / "guard/status.json", f"timing_replay/{key}")
        assert_close(f"{key}: replay fit_seconds", row["fit_seconds"], status["elapsed_seconds"])
    return {"timing_completed_rows": len(replay_by_key), "exact_contract_key_set": True}


def validate_cost_substitution(raw_rows: list[dict[str, Any]], matched_rows: list[dict[str, Any]]) -> dict[str, Any]:
    contract = read_json(RUN / "timing_contract.json")
    replay = read_json(RUN / "timing_completed.json")
    replay_by_key = replay_rows_by_key(replay)
    contract_keys = {entry["key"] for entry in contract["jobs"]}
    plan = read_json(RUN / "plan.json")
    plan_key_for_row = {plan_job_key(entry): entry["key"] for entry in plan["jobs"]}
    raw = keyed_rows(raw_rows, "raw metrics")
    matched = keyed_rows(matched_rows, "matched metrics")
    replay_count = 0
    original_count = 0
    max_error = 0.0
    for key in sorted(matched):
        plan_key = plan_key_for_row[key]
        original_status = read_json(RUN / plan_key / "guard/status.json")
        max_error = max(max_error, assert_close(f"{plan_key}: raw guard time", raw[key]["fit_seconds"], original_status["elapsed_seconds"]))
        if "original_fit_seconds" in matched[key]:
            max_error = max(max_error, assert_close(f"{plan_key}: matched original time", matched[key]["original_fit_seconds"], original_status["elapsed_seconds"]))
        if plan_key in contract_keys:
            replay_count += 1
            replay_row = replay_by_key[plan_key]
            replay_status = read_json(RUN / replay_row["replay_key"] / "guard/status.json")
            max_error = max(max_error, assert_close(f"{plan_key}: replay substituted time", matched[key]["fit_seconds"], replay_row["fit_seconds"]))
            max_error = max(max_error, assert_close(f"{plan_key}: replay status time", matched[key]["fit_seconds"], replay_status["elapsed_seconds"]))
        else:
            original_count += 1
            max_error = max(max_error, assert_close(f"{plan_key}: original fallback time", matched[key]["fit_seconds"], raw[key]["fit_seconds"]))
    ensure(replay_count == 13, f"expected 13 replay-substituted rows, got {replay_count}")
    ensure(original_count == 35, f"expected 35 original-time rows, got {original_count}")
    return {"replay_substituted_rows": replay_count, "original_time_rows": original_count, "max_abs_error": max_error}


def compare_summary_stable_fields(raw_summary: dict[str, Any], matched_summary: dict[str, Any]) -> dict[str, Any]:
    checked = []
    for field in SUMMARY_STABLE_FIELDS:
        if field in raw_summary and field in matched_summary:
            ensure(raw_summary[field] == matched_summary[field], f"summary stable field mismatch: {field}")
            checked.append(field)
    raw_metrics = raw_summary.get("metrics", [])
    matched_metrics = matched_summary.get("metrics", [])
    ensure(len(raw_metrics) == len(matched_metrics) == 6, "summary metric count mismatch")
    for raw_row in raw_metrics:
        matched_row = next(row for row in matched_metrics if int(row["seed"]) == int(raw_row["seed"]) and row["arm"] == raw_row["arm"])
        for field in ("macro_regret_pct_F0", "source_regret_pct_F0", "quality_passed", "freeze_count", "stopped_before_cap"):
            ensure(raw_row[field] == matched_row[field], f"summary quality field mismatch {raw_row['seed']}/{raw_row['arm']}/{field}")
    return {"stable_summary_fields_checked": checked, "quality_metric_rows_checked": 6}


def run_analysis_audit(output: Path = ANALYSIS_AUDIT) -> dict[str, Any]:
    assert_no_existing_output(output)
    contract_status = validate_timing_contract()
    completed = read_json(RUN / "completed.json")
    ensure(completed.get("completed") is True, "main run completed.json is not completed")
    timing_status = validate_timing_completed_for_analysis(contract_status)

    raw_summary_path = RESULTS / "summary.json"
    matched_summary_path = MATCHED / "summary.json"
    raw_metrics_path = RESULTS / "metrics.csv"
    matched_metrics_path = MATCHED / "metrics.csv"
    for path in (raw_summary_path, matched_summary_path, raw_metrics_path, matched_metrics_path):
        ensure(path.exists(), f"required analysis artifact missing: {rel(path)}")

    raw_summary = read_json(raw_summary_path)
    matched_summary = read_json(matched_summary_path)
    raw_rows = read_csv_rows(raw_metrics_path)
    matched_rows = read_csv_rows(matched_metrics_path)
    plan = read_json(RUN / "plan.json")

    row_compare = compare_raw_matched_rows(raw_rows, matched_rows)
    stable_summary = compare_summary_stable_fields(raw_summary, matched_summary)
    cell = recompute_cell_fields(matched_rows)
    recomputed_metrics = recompute_metrics(matched_rows, plan)
    metric_compare = compare_metric_dicts(matched_summary["metrics"], recomputed_metrics, "resource_matched_summary")
    recomputed_dominance = recompute_dominance(recomputed_metrics, plan)
    dominance_compare = compare_dominance(matched_summary["dominance"], recomputed_dominance, "resource_matched_summary")
    candidate_gate = all(row["gate_passed"] for row in recomputed_metrics if row["arm"] == "CONTRIB_FREEZE")
    ensure(bool(matched_summary["candidate_gate_passed"]) == bool(candidate_gate), "candidate gate mismatch")
    cost = validate_cost_substitution(raw_rows, matched_rows)

    payload = {
        "completed": True,
        "mode": "analysis",
        "created_utc": utc_now(),
        "contract_status": contract_status,
        "timing_status": timing_status,
        "inputs": {
            "raw_summary": {"path": rel(raw_summary_path), "sha256": sha256_file(raw_summary_path)},
            "matched_summary": {"path": rel(matched_summary_path), "sha256": sha256_file(matched_summary_path)},
            "raw_metrics": {"path": rel(raw_metrics_path), "sha256": sha256_file(raw_metrics_path)},
            "matched_metrics": {"path": rel(matched_metrics_path), "sha256": sha256_file(matched_metrics_path)},
        },
        "counts": {"raw_rows": len(raw_rows), "matched_rows": len(matched_rows), "summary_metrics": len(matched_summary["metrics"]), "dominance_rows": len(matched_summary["dominance"])},
        "raw_vs_resource_matched_quality_fields": row_compare,
        "raw_vs_resource_matched_summary_quality_fields": stable_summary,
        "cell_recompute": cell,
        "resource_matched_metric_recompute": metric_compare,
        "resource_matched_dominance_recompute": dominance_compare,
        "cost_substitution": cost,
        "candidate_gate_passed": bool(candidate_gate),
    }
    payload["output_sha256"] = write_once(output, payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Independent Study31 timing audit")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--replay", action="store_true", help="audit timing replay outputs after timing_completed.json exists")
    mode.add_argument("--analysis", action="store_true", help="audit raw vs resource-matched analysis after both summaries exist")
    args = parser.parse_args(argv)
    output = REPLAY_AUDIT if args.replay else ANALYSIS_AUDIT
    label = "replay" if args.replay else "analysis"
    try:
        payload = run_replay_audit(output) if args.replay else run_analysis_audit(output)
    except BaseException as exc:
        fail_once(output, label, exc)
        raise
    print(json.dumps({"completed": True, "mode": label, "output": rel(output), "output_sha256": payload["output_sha256"]}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
