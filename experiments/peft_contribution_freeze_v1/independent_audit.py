"""Independent CPU-only numerical audit for Study31.

This file deliberately avoids torch and the Study31 policy/analyse modules.
It reads completed artifacts only and writes one audit JSON per mode.
"""

from __future__ import annotations

import argparse
import hashlib
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import traceback
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "runs/peft_contribution_freeze_v1"
RESULTS = ROOT / "results/peft_contribution_freeze_v1"
ARMS = ("FULL", "ES2", "FIXED_FREEZE", "CONTRIB_FREEZE")
NONFULL_ARMS = ("ES2", "FIXED_FREEZE", "CONTRIB_FREEZE")
DATASETS = ("bdg2", "jena")
CONDITIONS = ("FULL90", "SPREAD30", "RECENT30")
SEEDS = (27000, 27001)
TOL = 1e-10


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        arr = np.ascontiguousarray(array)
        digest.update(str((arr.shape, arr.dtype)).encode("utf-8"))
        digest.update(arr.tobytes())
    return digest.hexdigest()


def read_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_once(path: Path, payload: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    return sha256_file(path)


def rel(path: Path | str) -> str:
    path = Path(path).resolve()
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def validate_contract(audit_contract_path: Path) -> dict:
    contract = read_json(audit_contract_path)
    self_path = Path(__file__).resolve()
    checks = {
        "independent_audit_py": sha256_file(self_path) == contract["independent_audit_py_sha256"],
        "run_contract": sha256_file(RUN / "contract.json") == contract["run_contract_sha256"],
        "plan": sha256_file(RUN / "plan.json") == contract["plan_sha256"],
    }
    for item in contract.get("input_hashes", []):
        path = ROOT / item["path"]
        checks[f"input:{item['path']}"] = sha256_file(path) == item["sha256"]
    if not all(checks.values()):
        bad = [name for name, ok in checks.items() if not ok]
        raise AssertionError(f"independent audit contract mismatch: {bad}")
    return {"path": str(audit_contract_path.resolve()), "sha256": sha256_file(audit_contract_path), "checks": checks}


def npz_score_loop(path: Path | str) -> dict:
    path = Path(path)
    with np.load(path, allow_pickle=False) as z:
        prediction = np.asarray(z["prediction"], dtype=np.float64)
        target = np.asarray(z["target"], dtype=np.float64)
        scale = np.asarray(z["scale"], dtype=np.float64)
        quantiles = np.asarray(z["quantiles"], dtype=np.float64)
    return score_arrays_loop(prediction, target, scale, quantiles)


def score_arrays_loop(prediction: np.ndarray, target: np.ndarray, scale: np.ndarray, quantiles: np.ndarray) -> dict:
    prediction = np.sort(np.asarray(prediction, dtype=np.float64), axis=2)
    target = np.asarray(target, dtype=np.float64)
    scale = np.asarray(scale, dtype=np.float64)
    quantiles = np.asarray(quantiles, dtype=np.float64)
    if prediction.ndim != 4 or target.ndim != 3:
        raise ValueError("prediction must be [N,C,Q,H] and target [N,C,H]")
    if prediction.shape[0] != target.shape[0] or prediction.shape[1] != target.shape[1] or prediction.shape[3] != target.shape[2]:
        raise ValueError("prediction/target shapes do not align")
    if prediction.shape[1] != 2 or prediction.shape[2] != len(quantiles) or scale.shape != (2,):
        raise ValueError("Study31 scoring expects two targets and the archive quantile grid")
    if not np.isfinite(prediction).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("nonfinite prediction or invalid scale")

    n, channels, q_count, horizon = prediction.shape
    per_channel_quantile_loss = np.zeros((channels, q_count), dtype=np.float64)
    valid_counts = np.zeros(channels, dtype=np.int64)
    for i in range(n):
        for c in range(channels):
            for h in range(horizon):
                y = target[i, c, h]
                if not np.isfinite(y):
                    continue
                valid_counts[c] += 1
                for qi, q in enumerate(quantiles):
                    error = y - prediction[i, c, qi, h]
                    left = q * error
                    right = (q - 1.0) * error
                    per_channel_quantile_loss[c, qi] += 2.0 * (left if left >= right else right)
    if np.any(valid_counts == 0):
        raise ValueError(f"empty target channel in score: {valid_counts.tolist()}")
    scaled = per_channel_quantile_loss / valid_counts[:, None] / scale[:, None]
    score = float(scaled.mean())
    return {
        "score": score,
        "valid_counts_by_channel": valid_counts.astype(int).tolist(),
        "scale": scale.tolist(),
        "quantiles": quantiles.tolist(),
        "per_channel_quantile_scaled_loss": scaled.tolist(),
        "prediction_shape": list(map(int, prediction.shape)),
        "target_shape": list(map(int, target.shape)),
    }


def contribution_plateau_independent(history: list[dict], minimum_step: int, fraction: float = 0.25) -> tuple[bool, list]:
    points = [h for h in history if "contribution_halves" in h]
    if len(points) < 4 or points[-1]["step"] < minimum_step:
        return False, []
    slopes = [
        [
            (b["contribution_halves"][j] - a["contribution_halves"][j]) / (b["step"] - a["step"])
            for j in range(2)
        ]
        for a, b in zip(points, points[1:])
    ]
    thresholds = [fraction * max(0.0, *[s[j] for s in slopes[:-1]]) for j in range(2)]
    return all(slopes[-1][j] <= thresholds[j] for j in range(2)), [slopes[-1], thresholds]


def early_stop_independent(history: list[dict], patience: int = 2) -> bool:
    best = history[0]["score"]
    stale = 0
    for h in history[1:]:
        if h["score"] < best:
            best = h["score"]
            stale = 0
        else:
            stale += 1
    return stale >= patience


def nested_max_abs_error(left: Any, right: Any) -> float:
    left_arr = np.asarray(left, dtype=np.float64)
    right_arr = np.asarray(right, dtype=np.float64)
    if left_arr.shape != right_arr.shape:
        return float("inf")
    if left_arr.size == 0:
        return 0.0
    return float(np.max(np.abs(left_arr - right_arr)))


def job_lookup(plan: dict) -> dict[tuple, dict]:
    return {
        (e["job"]["dataset"], e["job"]["condition"], e["job"]["arm"], e["job"]["seed"]): e
        for e in plan["jobs"]
    }


def output_dir_for_entry(entry: dict) -> Path:
    return RUN / entry["key"] / "output"


def result_for_entry(entry: dict) -> dict:
    path = output_dir_for_entry(entry) / "result.json"
    if not path.exists():
        raise FileNotFoundError(f"completed result missing: {rel(path)}")
    result = read_json(path)
    if not result.get("completed"):
        raise AssertionError(f"result is not completed: {rel(path)}")
    return result


def load_prediction_array(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as z:
        return np.asarray(z["prediction"])


def load_point_score(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"point archive missing: {rel(path)}")
    return npz_score_loop(path)


def score_slice(path: Path, row_slice: slice) -> dict:
    with np.load(path, allow_pickle=False) as z:
        prediction = np.asarray(z["prediction"])[row_slice]
        target = np.asarray(z["target"])[row_slice]
        scale = np.asarray(z["scale"])
        quantiles = np.asarray(z["quantiles"])
    return score_arrays_loop(prediction, target, scale, quantiles)


def score_off_slice(path: Path, row_slice: slice) -> dict:
    with np.load(path, allow_pickle=False) as z:
        prediction = np.asarray(z["off_prediction"])[row_slice]
        target = np.asarray(z["target"])[row_slice]
        scale = np.asarray(z["scale"])
        quantiles = np.asarray(z["quantiles"])
    return score_arrays_loop(prediction, target, scale, quantiles)


def hash_point_prediction(path: Path, key: str = "prediction") -> str:
    with np.load(path, allow_pickle=False) as z:
        return array_sha256(np.asarray(z[key]))


def point_files_for_result(out: Path, history: list[dict]) -> list[Path]:
    return [out / f"point_{int(h['step'])}.npz" for h in history]


def expected_sample_hash(job: dict, steps_done: int) -> str:
    schedule_steps = 3 if False else None
    del schedule_steps
    cap = 180 if job["condition"] == "FULL90" else 60
    samples = np.random.default_rng(job["seed"]).integers(len(job["training_rows"]), size=(cap, 8))
    return hashlib.sha256(samples[:steps_done].tobytes()).hexdigest()


def expected_best(history: list[dict]) -> tuple[float, int]:
    best = float(history[0]["score"])
    best_step = int(history[0]["step"])
    for record in history[1:]:
        score = float(record["score"])
        if score < best:
            best = score
            best_step = int(record["step"])
    return best, best_step


def expected_freeze_step(job: dict, history: list[dict], smoke: bool = False) -> tuple[int | None, list[dict]]:
    if job["arm"] not in ("FIXED_FREEZE", "CONTRIB_FREEZE") and not smoke:
        return None, []
    cap = 3 if smoke else (180 if job["condition"] == "FULL90" else 60)
    minimum = cap // 3
    decisions = []
    frozen = None
    prefix: list[dict] = []
    for record in history:
        prefix.append(record)
        step = int(record["step"])
        if step == 0 or step >= cap or frozen is not None:
            continue
        should = False
        details: list = []
        if smoke and step == 1:
            should = True
        elif job["arm"] == "FIXED_FREEZE":
            should = step >= minimum
        elif job["arm"] == "CONTRIB_FREEZE":
            should, details = contribution_plateau_independent(prefix, minimum)
            if "plateau_decision" in record and bool(record["plateau_decision"]) != bool(should):
                raise AssertionError(f"plateau decision mismatch at step {step}")
            if "slope_details" in record:
                if nested_max_abs_error(record["slope_details"], details) > TOL:
                    raise AssertionError(f"slope detail mismatch at step {step}")
        decisions.append({"step": step, "should_freeze": bool(should), "details": details})
        if should:
            frozen = step
            break
    return frozen, decisions


def expected_es_stop_step(job: dict, history: list[dict]) -> tuple[int, str, list[dict]]:
    cap = 180 if job["condition"] == "FULL90" else 60
    if job["arm"] != "ES2":
        return int(history[-1]["step"]), "cap", []
    prefix = [history[0]]
    checks = []
    for record in history[1:]:
        prefix.append(record)
        stop = early_stop_independent(prefix)
        checks.append({"step": int(record["step"]), "would_stop": bool(stop)})
        if stop:
            return int(record["step"]), "validation_patience2", checks
    return cap, "cap", checks


def fit_entry_audit(entry: dict, plan: dict, full_points: dict[tuple, dict[int, str]]) -> dict:
    job = entry["job"]
    out = output_dir_for_entry(entry)
    result = result_for_entry(entry)
    if result["job"] != job:
        raise AssertionError(f"job mismatch for {entry['key']}")
    if result["contract_sha256"] != sha256_file(RUN / "contract.json"):
        raise AssertionError(f"contract hash mismatch for {entry['key']}")
    if sha256_file(out / "best.pt") != result["checkpoint_sha256"]:
        raise AssertionError(f"checkpoint hash mismatch for {entry['key']}")

    history = result["history"]
    if history[0]["step"] != 0:
        raise AssertionError(f"history does not start at step 0: {entry['key']}")
    point_score_errors = []
    contribution_errors = []
    point_hashes: dict[int, str] = {}
    for record, point_path in zip(history, point_files_for_result(out, history)):
        step = int(record["step"])
        score_info = load_point_score(point_path)
        point_score_errors.append(abs(score_info["score"] - float(record["score"])))
        point_hashes[step] = hash_point_prediction(point_path)
        with np.load(point_path, allow_pickle=False) as z:
            has_off = "off_prediction" in z.files
        if "off_score" in record:
            if not has_off:
                raise AssertionError(f"missing off_prediction for contribution point {entry['key']} step {step}")
            off_info = npz_score_loop_with_key(point_path, "off_prediction")
            contribution_errors.append(abs(off_info["score"] - float(record["off_score"])))
            initial = float(history[0]["score"])
            on_halves = [score_slice(point_path, s)["score"] for s in (slice(0, 14), slice(16, 30))]
            off_halves = [score_off_slice(point_path, s)["score"] for s in (slice(0, 14), slice(16, 30))]
            recomputed_halves = [100.0 * (b - a) / initial for a, b in zip(on_halves, off_halves)]
            contribution_errors.extend(abs(a - b) for a, b in zip(recomputed_halves, record["contribution_halves"]))
            recomputed_total = 100.0 * (off_info["score"] - score_info["score"]) / initial
            contribution_errors.append(abs(recomputed_total - float(record["contribution"])))
        elif has_off:
            raise AssertionError(f"unexpected off_prediction without off_score for {entry['key']} step {step}")

    best_score, best_step = expected_best(history)
    val_score = npz_score_loop(out / "val_predictions.npz")["score"]
    best_errors = [
        abs(best_score - float(result["best_score"])),
        abs(val_score - float(result["best_score"])),
        float(best_step != int(result["best_step"])),
    ]

    expected_stop, expected_reason, es_checks = expected_es_stop_step(job, history)
    if job["arm"] == "ES2":
        if int(result["steps"]) != expected_stop or result["stop_reason"] != expected_reason:
            raise AssertionError(f"ES2 stop mismatch for {entry['key']}: expected {expected_stop}/{expected_reason}")
    elif result["stop_reason"] != "cap":
        raise AssertionError(f"non-ES arm stopped unexpectedly: {entry['key']}")

    expected_freeze, freeze_decisions = expected_freeze_step(job, history)
    if result["frozen_step"] != expected_freeze:
        raise AssertionError(f"freeze step mismatch for {entry['key']}: expected {expected_freeze}, got {result['frozen_step']}")

    prefix_compare = {"compared_steps": [], "mismatches": []}
    if job["arm"] != "FULL":
        full_key = (job["dataset"], job["condition"], job["seed"])
        full_by_step = full_points[full_key]
        limit = result["frozen_step"] if result["frozen_step"] is not None else int(result["steps"])
        for step, digest in point_hashes.items():
            if step <= limit and step in full_by_step:
                prefix_compare["compared_steps"].append(step)
                if digest != full_by_step[step]:
                    prefix_compare["mismatches"].append(step)
        if prefix_compare["mismatches"]:
            raise AssertionError(f"FULL prefix prediction mismatch for {entry['key']}: {prefix_compare['mismatches']}")

    expected_sample = expected_sample_hash(job, int(result["steps"]))
    if expected_sample != result["sample_index_sha256"]:
        raise AssertionError(f"sample RNG hash mismatch for {entry['key']}")

    snapshot_names = result["snapshot_names"]
    if not any("lora_" in name for name in snapshot_names):
        raise AssertionError(f"snapshot names do not include LoRA parameters: {entry['key']}")
    if result["trainable"] != 1768949:
        raise AssertionError(f"trainable count mismatch for {entry['key']}")
    if result["frozen_step"] is not None and not result["frozen_tail_verified"]:
        raise AssertionError(f"frozen tail not verified by execution result: {entry['key']}")

    return {
        "key": entry["key"],
        "dataset": job["dataset"],
        "condition": job["condition"],
        "arm": job["arm"],
        "seed": job["seed"],
        "steps": result["steps"],
        "best_step": result["best_step"],
        "frozen_step": result["frozen_step"],
        "stop_reason": result["stop_reason"],
        "toggle_checks": result["toggle_checks"],
        "adapter_updates": result["adapter_updates"],
        "point_score_max_abs_error": max(point_score_errors) if point_score_errors else 0.0,
        "contribution_max_abs_error": max(contribution_errors) if contribution_errors else 0.0,
        "best_selection_errors": best_errors,
        "sample_index_sha256_verified": True,
        "initial_head_hash": result["initial_head_hash"],
        "initial_prediction_sha256": point_hashes[0],
        "frozen_backbone_hash": result["frozen_hash"],
        "trainable": result["trainable"],
        "snapshot_name_count": len(snapshot_names),
        "lora_snapshot_name_count": sum("lora_" in name for name in snapshot_names),
        "snapshot_names_include_lora": True,
        "checkpoint_sha256": result["checkpoint_sha256"],
        "checkpoint_lora_storage_note": "best.pt is not opened by this CPU audit; result snapshot_names include LoRA entries and fit.py execution asserts set(state)==names",
        "prefix_vs_full": prefix_compare,
        "freeze_policy_decisions": freeze_decisions,
        "es2_policy_checks": es_checks,
    }


def npz_score_loop_with_key(path: Path, prediction_key: str) -> dict:
    with np.load(path, allow_pickle=False) as z:
        prediction = np.asarray(z[prediction_key], dtype=np.float64)
        target = np.asarray(z["target"], dtype=np.float64)
        scale = np.asarray(z["scale"], dtype=np.float64)
        quantiles = np.asarray(z["quantiles"], dtype=np.float64)
    return score_arrays_loop(prediction, target, scale, quantiles)


def audit_smoke(plan: dict) -> dict:
    candidates = [
        e for e in plan["jobs"]
        if e["job"]["arm"] == "CONTRIB_FREEZE" and e["job"]["condition"] == "RECENT30"
    ]
    if not candidates:
        raise AssertionError("no smoke source job found")
    job = candidates[0]["job"]
    out = RUN / "smoke" / "output"
    result_path = out / "result.json"
    if not result_path.exists():
        raise FileNotFoundError(f"smoke result missing: {rel(result_path)}")
    result = read_json(result_path)
    if result["job"] != job or not result.get("completed"):
        raise AssertionError("smoke result/job mismatch")
    history = result["history"]
    point_errors = []
    contribution_errors = []
    for record in history:
        point_path = out / f"point_{int(record['step'])}.npz"
        score_info = load_point_score(point_path)
        point_errors.append(abs(score_info["score"] - float(record["score"])))
        with np.load(point_path, allow_pickle=False) as z:
            has_off = "off_prediction" in z.files
        if "off_score" in record:
            if not has_off:
                raise AssertionError(f"smoke missing off_prediction at step {record['step']}")
            off_info = npz_score_loop_with_key(point_path, "off_prediction")
            contribution_errors.append(abs(off_info["score"] - float(record["off_score"])))
            initial = float(history[0]["score"])
            on_halves = [score_slice(point_path, s)["score"] for s in (slice(0, 14), slice(16, 30))]
            off_halves = [score_off_slice(point_path, s)["score"] for s in (slice(0, 14), slice(16, 30))]
            contribution_errors.extend(
                abs(100.0 * (b - a) / initial - recorded)
                for a, b, recorded in zip(on_halves, off_halves, record["contribution_halves"])
            )
            contribution_errors.append(abs(100.0 * (off_info["score"] - score_info["score"]) / initial - float(record["contribution"])))
        elif has_off:
            raise AssertionError(f"smoke unexpected off_prediction at step {record['step']}")
    expected_freeze, decisions = expected_freeze_step(job, history, smoke=True)
    if expected_freeze != 1 or result["frozen_step"] != 1 or not result["frozen_tail_verified"]:
        raise AssertionError("smoke lifecycle freeze check failed")
    sample_verified = expected_sample_hash(job, int(result["steps"])) == result["sample_index_sha256"]
    if not sample_verified:
        raise AssertionError("smoke sample RNG hash mismatch")
    return {
        "completed": True,
        "job": job,
        "steps": result["steps"],
        "frozen_step": result["frozen_step"],
        "toggle_checks": result["toggle_checks"],
        "point_score_max_abs_error": max(point_errors),
        "contribution_max_abs_error": max(contribution_errors) if contribution_errors else 0.0,
        "force_freeze_decisions": decisions,
        "trainable": result["trainable"],
        "snapshot_name_count": len(result["snapshot_names"]),
        "sample_index_sha256_verified": True,
    }


def same_freeze_pair_checks(plan: dict, rows: list[dict]) -> list[dict]:
    lookup = job_lookup(plan)
    checks = []
    for ds in DATASETS:
        for condition in CONDITIONS:
            for seed in SEEDS:
                fixed = next(row for row in rows if (row["dataset"], row["condition"], row["seed"], row["arm"]) == (ds, condition, seed, "FIXED_FREEZE"))
                contrib = next(row for row in rows if (row["dataset"], row["condition"], row["seed"], row["arm"]) == (ds, condition, seed, "CONTRIB_FREEZE"))
                if fixed["frozen_step"] != contrib["frozen_step"]:
                    continue
                fixed_entry = lookup[(ds, condition, "FIXED_FREEZE", seed)]
                contrib_entry = lookup[(ds, condition, "CONTRIB_FREEZE", seed)]
                fixed_out = output_dir_for_entry(fixed_entry)
                contrib_out = output_dir_for_entry(contrib_entry)
                fixed_history = result_for_entry(fixed_entry)["history"]
                contrib_history = result_for_entry(contrib_entry)["history"]
                fixed_steps = {int(record["step"]) for record in fixed_history}
                contrib_steps = {int(record["step"]) for record in contrib_history}
                common_steps = sorted(fixed_steps & contrib_steps)
                mismatches = []
                for step in common_steps:
                    left = hash_point_prediction(fixed_out / f"point_{step}.npz")
                    right = hash_point_prediction(contrib_out / f"point_{step}.npz")
                    if left != right:
                        mismatches.append(step)
                left_val = hash_point_prediction(fixed_out / "val_predictions.npz")
                right_val = hash_point_prediction(contrib_out / "val_predictions.npz")
                if mismatches or left_val != right_val:
                    raise AssertionError(f"same-freeze FIXED/CONTRIB trajectory mismatch: {ds}/{condition}/s{seed}")
                checks.append({
                    "dataset": ds,
                    "condition": condition,
                    "seed": seed,
                    "frozen_step": fixed["frozen_step"],
                    "common_point_steps": common_steps,
                    "on_prediction_trajectory_identical": True,
                    "selected_val_prediction_identical": True,
                })
    return checks


def run_fit_audit(audit_contract_path: Path, output: Path) -> dict:
    contract_status = validate_contract(audit_contract_path)
    plan = read_json(RUN / "plan.json")
    if len(plan["jobs"]) != 48:
        raise AssertionError(f"expected 48 jobs, got {len(plan['jobs'])}")
    missing = [e["key"] for e in plan["jobs"] if not (RUN / e["key"] / "output" / "result.json").exists()]
    if missing:
        raise FileNotFoundError(f"fit results incomplete; missing {len(missing)} result files")

    full_points: dict[tuple, dict[int, str]] = {}
    for entry in plan["jobs"]:
        job = entry["job"]
        if job["arm"] != "FULL":
            continue
        result = result_for_entry(entry)
        full_points[(job["dataset"], job["condition"], job["seed"])] = {
            int(record["step"]): hash_point_prediction(output_dir_for_entry(entry) / f"point_{int(record['step'])}.npz")
            for record in result["history"]
        }

    rows = [fit_entry_audit(entry, plan, full_points) for entry in plan["jobs"]]
    smoke = audit_smoke(plan)
    same_freeze_checks = same_freeze_pair_checks(plan, rows)
    frozen_hashes = sorted({row["frozen_backbone_hash"] for row in rows})
    snapshot_name_counts = sorted({row["snapshot_name_count"] for row in rows})
    initial_head_groups: dict[str, list[str]] = {}
    initial_pred_groups: dict[str, list[str]] = {}
    for row in rows:
        group = f"{row['dataset']}/{row['condition']}/s{row['seed']}"
        initial_head_groups.setdefault(group, []).append(row["initial_head_hash"])
        initial_pred_groups.setdefault(group, []).append(row["initial_prediction_sha256"])
    group_mismatches = {
        "initial_head_hash": [group for group, values in initial_head_groups.items() if len(set(values)) != 1],
        "initial_prediction_sha256": [group for group, values in initial_pred_groups.items() if len(set(values)) != 1],
    }
    if group_mismatches["initial_head_hash"] or group_mismatches["initial_prediction_sha256"]:
        raise AssertionError(f"initial prefix mismatch across arms: {group_mismatches}")
    max_errors = {
        "point_score": max(row["point_score_max_abs_error"] for row in rows),
        "contribution": max(row["contribution_max_abs_error"] for row in rows),
        "best_selection": max(max(row["best_selection_errors"]) for row in rows),
        "smoke_point_score": smoke["point_score_max_abs_error"],
        "smoke_contribution": smoke["contribution_max_abs_error"],
    }
    if any(value > TOL for value in max_errors.values()):
        raise AssertionError(f"fit audit numerical mismatch: {max_errors}")
    payload = {
        "completed": True,
        "mode": "fit",
        "created_utc": utc_now(),
        "contract_status": contract_status,
        "counts": {"fit_jobs": len(rows), "smoke_jobs": 1},
        "max_abs_errors": max_errors,
        "frozen_backbone_hash_unique_count": len(frozen_hashes),
        "frozen_backbone_hashes": frozen_hashes,
        "snapshot_name_counts": snapshot_name_counts,
        "lora_snapshot_name_counts": sorted({row["lora_snapshot_name_count"] for row in rows}),
        "trainable_counts": sorted({row["trainable"] for row in rows}),
        "initial_hash_groups_verified": True,
        "same_freeze_fixed_vs_contrib_checks": same_freeze_checks,
        "rows": rows,
        "smoke": smoke,
    }
    payload["output_sha256"] = write_once(output, payload)
    return payload


def archive_result_score(path: Path, result_path: Path, label: str) -> dict:
    score_info = npz_score_loop(path)
    result = read_json(result_path)
    key = "score"
    if key not in result:
        raise AssertionError(f"{label} result has no score")
    error = abs(score_info["score"] - float(result[key]))
    return {"label": label, "path": rel(path), "result_path": rel(result_path), "score": score_info["score"], "result_score": result[key], "abs_error": error, "score_detail": score_info}


def guard_seconds(path: Path) -> float:
    status = read_json(path / "guard" / "status.json")
    return float(status["elapsed_seconds"])


def run_forecast_audit(audit_contract_path: Path, output: Path) -> dict:
    contract_status = validate_contract(audit_contract_path)
    plan = read_json(RUN / "plan.json")
    completed = read_json(RUN / "completed.json")
    if not completed.get("completed"):
        raise AssertionError("run completed.json is not completed")
    if len(plan["jobs"]) != 48:
        raise AssertionError(f"expected 48 jobs, got {len(plan['jobs'])}")
    missing = []
    for ds in DATASETS:
        if not (RUN / "f0" / ds / "output" / "predictions.npz").exists():
            missing.append(f"f0/{ds}")
    for entry in plan["jobs"]:
        path = RUN / entry["key"].replace("fits/", "eval/") / "output" / "predictions.npz"
        if not path.exists():
            missing.append(entry["key"].replace("fits/", "eval/"))
    if missing:
        raise FileNotFoundError(f"forecast results incomplete: {len(missing)} missing")

    f0 = {}
    archive_checks = []
    for ds in DATASETS:
        check = archive_result_score(RUN / "f0" / ds / "output" / "predictions.npz", RUN / "f0" / ds / "output" / "result.json", f"f0/{ds}")
        archive_checks.append(check)
        f0[ds] = check["score"]

    rows = []
    for entry in plan["jobs"]:
        job = entry["job"]
        fit_path = RUN / entry["key"]
        eval_path = RUN / entry["key"].replace("fits/", "eval/")
        eval_check = archive_result_score(eval_path / "output" / "predictions.npz", eval_path / "output" / "result.json", entry["key"].replace("fits/", "eval/"))
        archive_checks.append(eval_check)
        fit_result = read_json(fit_path / "output" / "result.json")
        val_score_info = npz_score_loop(fit_path / "output" / "val_predictions.npz")
        val_check = {
            "label": entry["key"] + "/val",
            "path": rel(fit_path / "output" / "val_predictions.npz"),
            "result_path": rel(fit_path / "output" / "result.json"),
            "score": val_score_info["score"],
            "result_score": fit_result["best_score"],
            "abs_error": abs(val_score_info["score"] - float(fit_result["best_score"])),
            "score_detail": val_score_info,
        }
        archive_checks.append(val_check)
        rows.append({
            "dataset": job["dataset"],
            "condition": job["condition"],
            "seed": job["seed"],
            "arm": job["arm"],
            "E_score": eval_check["score"],
            "F0_score": f0[job["dataset"]],
            "fit_seconds": guard_seconds(fit_path),
            "eval_seconds": guard_seconds(eval_path),
            "steps": fit_result["steps"],
            "adapter_updates": fit_result["adapter_updates"],
            "frozen_step": fit_result["frozen_step"],
            "best_step": fit_result["best_step"],
            "best_V_score": fit_result["best_score"],
            "toggle_checks": fit_result["toggle_checks"],
        })

    for row in rows:
        full = next(r for r in rows if r["arm"] == "FULL" and all(r[k] == row[k] for k in ("dataset", "condition", "seed")))
        row["regret_pct_F0"] = 100.0 * (row["E_score"] - full["E_score"]) / row["F0_score"]
        row["fit_savings_percent"] = 100.0 * (1.0 - row["fit_seconds"] / full["fit_seconds"])

    metrics = []
    for seed in plan["seeds"]:
        base = [r for r in rows if r["seed"] == seed and r["arm"] == "FULL"]
        base_seconds = sum(r["fit_seconds"] for r in base)
        for arm in NONFULL_ARMS:
            group = [r for r in rows if r["seed"] == seed and r["arm"] == arm]
            sources = {ds: float(np.mean([r["regret_pct_F0"] for r in group if r["dataset"] == ds])) for ds in DATASETS}
            regret = float(np.mean([r["regret_pct_F0"] for r in group]))
            seconds = sum(r["fit_seconds"] for r in group)
            savings = 100.0 * (1.0 - seconds / base_seconds)
            gates = plan["gates"]
            quality = regret <= gates["macro_regret_max_pct_F0"] and max(sources.values()) <= gates["source_regret_max_pct_F0"]
            metrics.append({
                "seed": seed,
                "arm": arm,
                "macro_regret_pct_F0": regret,
                "source_regret_pct_F0": sources,
                "fit_seconds": seconds,
                "fit_savings_percent": savings,
                "quality_passed": bool(quality),
                "cost_passed": bool(savings > gates["savings_min_percent"]),
                "gate_passed": bool(quality and savings > gates["savings_min_percent"]),
                "freeze_count": sum(r["frozen_step"] is not None for r in group),
                "stopped_before_cap": sum(r["steps"] < (180 if r["condition"] == "FULL90" else 60) for r in group),
            })

    dominance = []
    for seed in plan["seeds"]:
        candidate = next(r for r in metrics if r["seed"] == seed and r["arm"] == "CONTRIB_FREEZE")
        for arm in ("ES2", "FIXED_FREEZE"):
            baseline = next(r for r in metrics if r["seed"] == seed and r["arm"] == arm)
            no_worse = baseline["macro_regret_pct_F0"] <= candidate["macro_regret_pct_F0"] and baseline["fit_seconds"] <= candidate["fit_seconds"]
            strict = baseline["macro_regret_pct_F0"] < candidate["macro_regret_pct_F0"] or baseline["fit_seconds"] < candidate["fit_seconds"]
            dominance.append({"seed": seed, "baseline": arm, "dominates_candidate_macro_quality_cost": bool(no_worse and strict)})

    root_summary_compare = None
    root_summary_path = RESULTS / "summary.json"
    if root_summary_path.exists():
        root_summary = read_json(root_summary_path)
        metric_errors = []
        for metric in metrics:
            other = next(r for r in root_summary["metrics"] if r["seed"] == metric["seed"] and r["arm"] == metric["arm"])
            metric_errors.append(abs(other["macro_regret_pct_F0"] - metric["macro_regret_pct_F0"]))
            metric_errors.append(abs(other["fit_seconds"] - metric["fit_seconds"]))
            metric_errors.append(abs(other["fit_savings_percent"] - metric["fit_savings_percent"]))
            for ds in DATASETS:
                metric_errors.append(abs(other["source_regret_pct_F0"][ds] - metric["source_regret_pct_F0"][ds]))
        root_summary_compare = {
            "path": rel(root_summary_path),
            "summary_sha256": sha256_file(root_summary_path),
            "F0_max_abs_error": max(abs(root_summary["F0"][ds] - f0[ds]) for ds in DATASETS),
            "metrics_max_abs_error": max(metric_errors) if metric_errors else 0.0,
            "candidate_gate_match": bool(root_summary["candidate_gate_passed"] == all(r["gate_passed"] for r in metrics if r["arm"] == "CONTRIB_FREEZE")),
        }

    max_archive_error = max(check["abs_error"] for check in archive_checks)
    if max_archive_error > TOL:
        raise AssertionError(f"forecast audit score mismatch: {max_archive_error}")
    payload = {
        "completed": True,
        "mode": "forecast",
        "created_utc": utc_now(),
        "contract_status": contract_status,
        "counts": {"f0_forecasts": 2, "eval_forecasts": 48, "paired_full_regrets": 48, "arm_seed_gate_metrics": 6},
        "F0": f0,
        "rows": rows,
        "metrics": metrics,
        "dominance": dominance,
        "candidate_gate_passed": all(r["gate_passed"] for r in metrics if r["arm"] == "CONTRIB_FREEZE"),
        "archive_score_max_abs_error": max_archive_error,
        "archive_checks": archive_checks,
        "root_summary_compare": root_summary_compare,
    }
    payload["output_sha256"] = write_once(output, payload)
    return payload


def guarded_write_failure(output: Path, mode: str, exc: BaseException) -> None:
    if output.exists():
        return
    payload = {
        "completed": False,
        "mode": mode,
        "created_utc": utc_now(),
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "traceback": traceback.format_exc(),
    }
    write_once(output, payload)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-contract", type=Path, default=RUN / "independent_audit_contract.json")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--fit", action="store_true")
    mode.add_argument("--forecast", action="store_true")
    args = parser.parse_args(argv)
    if args.fit:
        output = RUN / "independent_fit_audit.json"
        selected_mode = "fit"
        runner = run_fit_audit
    else:
        output = RUN / "independent_forecast_audit.json"
        selected_mode = "forecast"
        runner = run_forecast_audit
    try:
        payload = runner(args.audit_contract, output)
    except Exception as exc:
        guarded_write_failure(output, selected_mode, exc)
        raise
    print(json.dumps({
        "completed": True,
        "mode": selected_mode,
        "output": str(output.resolve()),
        "output_sha256": payload["output_sha256"],
    }, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
