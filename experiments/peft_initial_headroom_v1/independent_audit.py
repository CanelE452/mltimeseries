"""Independent CPU-only audit for Study34 initial-headroom diagnostics.

The audit reads JSON/NPZ outputs after the guarded run finishes. It avoids
torch and does not import the study's fit.py or analyse.py; equations are
implemented locally from the frozen artifact schema.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN = ROOT / "runs/peft_initial_headroom_v1"
DEFAULT_RESULTS = ROOT / "results/peft_initial_headroom_v1"
PREVIOUS_PLAN = ROOT / "runs/peft_decision_transfer_v1/plan.json"
FAMILIES = ("F0", "HEAD", "WIDE", "JOINT", "CORRECTION")
NEURAL_FAMILIES = ("HEAD", "WIDE", "JOINT")
CONDITIONS = ("FULL90", "SPREAD30")
SEEDS = (29000, 29001)
GATE_BAND = 0.25


def sha256(path: Path | str) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(path: Path | str) -> str:
    path = Path(path).resolve()
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def read_json(path: Path | str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json_x(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    return sha256(path)


def parse_dt(value: str) -> datetime:
    out = datetime.fromisoformat(value)
    if out.tzinfo is None:
        out = out.replace(tzinfo=timezone.utc)
    return out.astimezone(timezone.utc)


def file_mtime_utc(path: Path) -> datetime:
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)


def assert_close(actual: float, expected: float, label: str, tol: float = 1e-10) -> None:
    if abs(float(actual) - float(expected)) > tol:
        raise AssertionError(f"{label}: {actual!r} != {expected!r} within {tol}")


def verify_hashes(plan: dict[str, Any]) -> dict[str, Any]:
    if len(plan["source_hashes"]) != 46 or len(plan["input_hashes"]) != 5:
        raise AssertionError(f"unexpected plan hash counts: source={len(plan['source_hashes'])} input={len(plan['input_hashes'])}")
    for section in ("source_hashes", "input_hashes"):
        for path, expected in plan[section].items():
            actual = sha256(ROOT / path)
            if actual != expected:
                raise AssertionError(f"{section} mismatch {path}: {actual} != {expected}")

    previous = read_json(PREVIOUS_PLAN)
    missing_sources = [path for path in previous["source_hashes"] if path not in plan["source_hashes"]]
    changed_sources = [
        path for path, expected in previous["source_hashes"].items()
        if path in plan["source_hashes"] and plan["source_hashes"][path] != expected
    ]
    if missing_sources or changed_sources:
        raise AssertionError({"missing_previous_sources": missing_sources, "changed_previous_sources": changed_sources})
    previous_hash_mismatches = []
    for section in ("source_hashes", "input_hashes"):
        for path, expected in previous[section].items():
            actual = sha256(ROOT / path)
            if actual != expected:
                previous_hash_mismatches.append({"section": section, "path": path, "actual": actual, "expected": expected})
    if previous_hash_mismatches:
        raise AssertionError({"previous_plan_file_hash_mismatches": previous_hash_mismatches})
    previous_inputs_present = [path for path in previous["input_hashes"] if path in plan["input_hashes"]]
    return {
        "source_hashes": len(plan["source_hashes"]),
        "input_hashes": len(plan["input_hashes"]),
        "previous33_sources_preserved_in_study34_plan": len(previous["source_hashes"]),
        "previous33_inputs_still_hash_valid": len(previous["input_hashes"]),
        "previous33_inputs_present_in_study34_plan": len(previous_inputs_present),
        "previous33_inputs_not_inherited_by_study34_plan": len(previous["input_hashes"]) - len(previous_inputs_present),
    }


def pinball_components(prediction: np.ndarray, target: np.ndarray, quantiles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    prediction = np.sort(prediction.astype(np.float64), axis=2)
    target = target.astype(np.float64)
    if prediction.ndim != 4 or target.ndim != 3:
        raise AssertionError("prediction/target rank mismatch")
    if prediction.shape[0] != target.shape[0] or prediction.shape[1] != target.shape[1] or prediction.shape[3] != target.shape[2]:
        raise AssertionError("prediction/target shape mismatch")
    q = quantiles.astype(np.float64)[None, None, :, None]
    error = target[:, :, None, :] - prediction
    valid = np.isfinite(target)
    numerator = np.where(valid[:, :, None, :], 2.0 * np.maximum(q * error, (q - 1.0) * error), 0.0).sum(axis=3)
    counts = valid.sum(axis=2).astype(np.float64)
    return numerator, counts


def score_arrays(prediction: np.ndarray, target: np.ndarray, quantiles: np.ndarray, scale: np.ndarray) -> float:
    numerator, counts = pinball_components(prediction, target, quantiles)
    scale = scale.astype(np.float64)
    if np.any(counts.sum(axis=0) <= 0) or np.any(~np.isfinite(scale)) or np.any(scale <= 0):
        raise AssertionError("empty target count or bad scale")
    return float((numerator.sum(axis=0) / counts.sum(axis=0)[:, None] / scale[:, None]).mean())


def score_npz(path: Path | str) -> float:
    with np.load(path, allow_pickle=False) as z:
        return score_arrays(z["prediction"], z["target"], z["quantiles"], z["scale"])


def load_npz_core(path: Path | str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return (
            z["prediction"].astype(np.float64),
            z["target"].astype(np.float64),
            z["quantiles"].astype(np.float64),
            z["scale"].astype(np.float64),
        )


def correction_candidates() -> list[dict[str, Any]]:
    result = [{"name": "identity", "kind": "identity", "shrinkage": 0.0}]
    for shrinkage in (0.25, 0.5, 1.0):
        result.append({"name": f"affine_s{shrinkage:g}", "kind": "affine", "shrinkage": shrinkage})
    for shrinkage in (0.25, 0.5, 1.0):
        result.append({"name": f"bias_s{shrinkage:g}", "kind": "bias", "shrinkage": shrinkage})
    return result


def correction_parameters(train_npz: Path) -> dict[str, Any]:
    prediction, target, quantiles, scale = load_npz_core(train_npz)
    prediction = np.sort(prediction, axis=2)
    median_index = int(np.argmin(np.abs(quantiles - 0.5)))
    median = prediction[:, :, median_index, :]
    slopes, intercepts, biases, counts = [], [], [], []
    ridge = 1e-6
    for channel in range(target.shape[1]):
        x = median[:, channel].reshape(-1)
        y = target[:, channel].reshape(-1)
        mask = np.isfinite(x) & np.isfinite(y)
        counts.append(int(mask.sum()))
        if mask.sum() < 2:
            slopes.append(1.0)
            intercepts.append(0.0)
            biases.append(0.0)
            continue
        x = x[mask] / scale[channel]
        y = y[mask] / scale[channel]
        slope = float(np.clip(np.mean((x - x.mean()) * (y - y.mean())) / (np.var(x) + ridge), 0.25, 4.0))
        intercept = float(np.mean(y - slope * x) * scale[channel])
        bias = float(np.mean(y - x) * scale[channel])
        slopes.append(slope)
        intercepts.append(intercept)
        biases.append(bias)
    return {"median_quantile_index": median_index, "ridge": ridge, "slope_clip": [0.25, 4.0], "slopes": slopes, "intercepts": intercepts, "biases": biases, "fit_counts": counts}


def apply_correction(prediction: np.ndarray, params: dict[str, Any], candidate: dict[str, Any]) -> np.ndarray:
    corrected = np.array(prediction, dtype=np.float64, copy=True)
    shrink = float(candidate["shrinkage"])
    if candidate["kind"] == "identity":
        return corrected
    if candidate["kind"] == "affine":
        slopes = np.asarray(params["slopes"], dtype=np.float64)
        intercepts = np.asarray(params["intercepts"], dtype=np.float64)
        for channel in range(corrected.shape[1]):
            corrected[:, channel] = corrected[:, channel] + shrink * (slopes[channel] * corrected[:, channel] + intercepts[channel] - corrected[:, channel])
    elif candidate["kind"] == "bias":
        biases = np.asarray(params["biases"], dtype=np.float64)
        for channel in range(corrected.shape[1]):
            corrected[:, channel] = corrected[:, channel] + shrink * biases[channel]
    else:
        raise AssertionError(candidate)
    return np.sort(corrected, axis=2)


def choose_correction(train_npz: Path, val_npz: Path) -> dict[str, Any]:
    params = correction_parameters(train_npz)
    prediction, target, quantiles, scale = load_npz_core(val_npz)
    scored = []
    for order, candidate in enumerate(correction_candidates()):
        corrected = apply_correction(prediction, params, candidate)
        scored.append({**candidate, "order": order, "V": score_arrays(corrected, target, quantiles, scale)})
    selected = min(scored, key=lambda item: (item["V"], item["order"]))
    return {"parameters": params, "candidates": scored, "selected": selected}


def compare_nested_floats(actual: Any, expected: Any, label: str, tol: float = 1e-10) -> None:
    if isinstance(expected, dict):
        if set(actual) != set(expected):
            raise AssertionError(f"{label} keys differ: {set(actual)} != {set(expected)}")
        for key in expected:
            compare_nested_floats(actual[key], expected[key], f"{label}.{key}", tol)
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            raise AssertionError(f"{label} length differs")
        for i, value in enumerate(expected):
            compare_nested_floats(actual[i], value, f"{label}[{i}]", tol)
    elif isinstance(expected, float):
        assert_close(float(actual), expected, label, tol)
    else:
        if actual != expected:
            raise AssertionError(f"{label}: {actual!r} != {expected!r}")


def job_cell(job: dict[str, Any]) -> str:
    return f"{job['dataset']}/{job['condition']}/s{job['seed']}"


def audit_fit_jobs(run: Path, plan: dict[str, Any], plan_hash: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    cells: dict[str, dict[str, Any]] = {}
    fit_hashes: dict[str, str] = {}
    paired: dict[str, list[tuple[str, str, str, np.ndarray, float]]] = {}
    family_hashes: dict[str, dict[str, str]] = {}
    max_score_error = 0.0
    checkpoint_hashes = 0
    for index, entry in enumerate(plan["jobs"]):
        job = entry["job"]
        out = run / "fit" / entry["key"] / "output"
        result_path = out / "result.json"
        result = read_json(result_path)
        if not result.get("completed") or result["job"] != job:
            raise AssertionError(f"bad fit result {entry['key']}")
        if result["plan_sha256"] != plan_hash or result.get("D_opened") is not False:
            raise AssertionError(f"fit result plan/D flag mismatch {entry['key']}")
        if not result.get("frozen_verified") or not result.get("restore_exact"):
            raise AssertionError(f"fit invariant failed {entry['key']}")
        if sha256(out / "best.pt") != result["checkpoint_sha256"]:
            raise AssertionError(f"checkpoint hash mismatch {entry['key']}")
        checkpoint_hashes += 1
        best = min(result["history"], key=lambda point: (point["V"], point["step"]))
        if (best["V"], best["step"]) != (result["best_V"], result["best_step"]):
            raise AssertionError(f"best history mismatch {entry['key']}")
        selected_score = score_npz(out / "selected_val.npz")
        f0_score = score_npz(out / "F0_val.npz")
        max_score_error = max(max_score_error, abs(selected_score - result["best_V"]), abs(f0_score - result["initial_V"]))
        fit_hashes[entry["key"]] = sha256(result_path)
        cell = job_cell(job)
        with np.load(out / "F0_val.npz", allow_pickle=False) as z:
            initial_prediction = z["prediction"].copy()
        paired.setdefault(cell, []).append((job["family"], result["initial_head_hash"], result["sample_sha256"], initial_prediction, result["initial_V"]))
        cells.setdefault(cell, {"dataset": job["dataset"], "condition": job["condition"], "seed": job["seed"], "families": {}})
        current = cells[cell]["families"].get(job["family"])
        candidate = {
            "index": index,
            "key": entry["key"],
            "recipe": job["recipe"],
            "best_V": result["best_V"],
            "best_step": result["best_step"],
            "selected_hash": result["selected_hash"],
            "initial_V": result["initial_V"],
            "trajectory_seconds": result["trajectory_seconds"],
            "trainable": result["trainable"],
        }
        if current is None or (candidate["best_V"], candidate["best_step"], candidate["recipe"]) < (current["best_V"], current["best_step"], current["recipe"]):
            cells[cell]["families"][job["family"]] = candidate
        family_hashes.setdefault(cell, {})[job["family"]] = result["initial_head_hash"]

    by_cell_recipe = {(job_cell(entry["job"]), entry["job"]["family"], entry["job"]["recipe"]): index for index, entry in enumerate(plan["jobs"])}
    f0_equal_cells = 0
    head_joint_hash_cells = 0
    sample_hash_cells = 0
    for cell, items in paired.items():
        if len(items) != 12:
            raise AssertionError(f"{cell} expected 12 fit jobs, got {len(items)}")
        if len({item[2] for item in items}) == 1:
            sample_hash_cells += 1
        head_joint_hashes = {item[1] for item in items if item[0] in ("HEAD", "JOINT")}
        if len(head_joint_hashes) == 1:
            head_joint_hash_cells += 1
        reference = items[0][3]
        if all(np.array_equal(item[3], reference) for item in items) and len({item[4] for item in items}) == 1:
            f0_equal_cells += 1
        else:
            raise AssertionError(f"{cell} F0_val predictions or initial V differ across 12 jobs")
        if set(cells[cell]["families"]) != set(NEURAL_FAMILIES):
            raise AssertionError(f"{cell} missing neural family")
        if family_hashes[cell]["HEAD"] != family_hashes[cell]["JOINT"]:
            raise AssertionError(f"{cell} HEAD/JOINT initial hash differs")
        f0_index = by_cell_recipe[(cell, "HEAD", 0)]
        f0_entry = plan["jobs"][f0_index]
        correction = choose_correction(run / "fit" / f0_entry["key"] / "output/F0_train.npz", run / "fit" / f0_entry["key"] / "output/F0_val.npz")
        cells[cell]["f0_index"] = cells[cell]["families"]["HEAD"]["index"]
        cells[cell]["f0_key"] = cells[cell]["families"]["HEAD"]["key"]
        cells[cell]["correction"] = correction

    if max_score_error > 1e-10:
        raise AssertionError(f"fit score max error {max_score_error}")
    return cells, fit_hashes, {
        "fit_jobs": len(plan["jobs"]),
        "checkpoint_hashes_checked": checkpoint_hashes,
        "selected_val_and_F0_val_max_abs_error": max_score_error,
        "cells_with_12_equal_F0_predictions": f0_equal_cells,
        "cells_with_equal_HEAD_JOINT_initial_head_hash": head_joint_hash_cells,
        "cells_with_equal_sample_hash": sample_hash_cells,
    }


def compare_selection_seal(run: Path, plan: dict[str, Any], plan_hash: str, cells: dict[str, Any], fit_hashes: dict[str, str]) -> dict[str, Any]:
    seal = read_json(run / "selection_sealed.json")
    if seal["plan_sha256"] != plan_hash or seal.get("D_opened") is not False or seal.get("fit_only_selection") is not True:
        raise AssertionError("selection seal exposure/hash markers mismatch")
    compare_nested_floats(seal["fit_result_hashes"], fit_hashes, "fit_result_hashes", tol=0.0)
    forecast_indices = sorted({cells[cell]["families"][family]["index"] for cell in cells for family in NEURAL_FAMILIES})
    if seal["forecast_indices"] != forecast_indices or len(forecast_indices) != 24:
        raise AssertionError("forecast index set mismatch")
    for cell, payload in cells.items():
        sealed_cell = seal["cells"][cell]
        for key in ("dataset", "condition", "seed", "f0_index", "f0_key"):
            if sealed_cell[key] != payload[key]:
                raise AssertionError(f"{cell}.{key} mismatch")
        for family in NEURAL_FAMILIES:
            compare_nested_floats(sealed_cell["families"][family], payload["families"][family], f"{cell}.{family}")
        compare_nested_floats(sealed_cell["correction"], payload["correction"], f"{cell}.correction")
    return {
        "selection_cells": len(seal["cells"]),
        "fit_result_hashes": len(seal["fit_result_hashes"]),
        "forecast_indices": len(seal["forecast_indices"]),
        "correction_cells_recomputed": len(cells),
    }


def load_family_prediction(run: Path, cell: dict[str, Any], family: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, str]:
    if family == "F0":
        key, name = cell["f0_key"], "F0"
    elif family == "CORRECTION":
        key, name = cell["f0_key"], "F0"
    else:
        key, name = cell["families"][family]["key"], "selected"
    prediction, target, quantiles, scale = load_npz_core(run / "forecast" / key / "output" / f"{name}.npz")
    if family == "CORRECTION":
        prediction = apply_correction(prediction, cell["correction"]["parameters"], cell["correction"]["selected"])
    score = score_arrays(prediction, target, quantiles, scale)
    return prediction, target, quantiles, scale, score, key


def recompute_metrics(run: Path, plan: dict[str, Any], seal: dict[str, Any], plan_hash: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    loss_records: list[dict[str, Any]] = []
    numerators: list[np.ndarray] = []
    counts: list[np.ndarray] = []
    scales: list[np.ndarray] = []
    score_errors = []
    forecast_result_checks = 0
    for cell_key, cell in sorted(seal["cells"].items()):
        f0_score = None
        for family in FAMILIES:
            prediction, target, quantiles, scale, score, forecast_key = load_family_prediction(run, cell, family)
            numerator, count = pinball_components(prediction, target, quantiles)
            recomputed = float((numerator.sum(axis=0) / count.sum(axis=0)[:, None] / scale[:, None]).mean())
            score_errors.append(abs(score - recomputed))
            if family in NEURAL_FAMILIES:
                result = read_json(run / "forecast" / forecast_key / "output/result.json")
                if result["plan_sha256"] != plan_hash or result["job"] != plan["jobs"][cell["families"][family]["index"]]["job"]:
                    raise AssertionError(f"forecast result mismatch {forecast_key}")
                assert_close(result["D_score"], score, f"forecast D_score {forecast_key}")
                if result["selected_hash"] != cell["families"][family]["selected_hash"]:
                    raise AssertionError(f"forecast selected hash mismatch {forecast_key}")
                forecast_result_checks += 1
            if family == "F0":
                f0_score = score
            if f0_score is None:
                raise AssertionError("F0 must be scored before other families")
            row = {
                "dataset": cell["dataset"],
                "condition": cell["condition"],
                "seed": cell["seed"],
                "family": family,
                "D_score": score,
                "F0_score": f0_score,
                "D_over_F0": score / f0_score,
                "improvement_pct_F0": 100.0 * (f0_score - score) / f0_score,
                "forecast_key": forecast_key,
            }
            if family in NEURAL_FAMILIES:
                row.update({
                    "recipe": cell["families"][family]["recipe"],
                    "best_V": cell["families"][family]["best_V"],
                    "best_step": cell["families"][family]["best_step"],
                    "trainable": cell["families"][family]["trainable"],
                })
            elif family == "CORRECTION":
                selected = cell["correction"]["selected"]
                row.update({
                    "recipe": selected["name"],
                    "best_V": selected["V"],
                    "best_step": 0,
                    "trainable": {"identity": 0, "bias": 2, "affine": 4}[selected["kind"]],
                })
            else:
                row.update({"recipe": "identity", "best_V": None, "best_step": 0, "trainable": 0})
            rows.append(row)
            loss_records.append({k: row[k] for k in ("dataset", "condition", "seed", "family")})
            numerators.append(numerator)
            counts.append(count)
            scales.append(scale)
    if len(rows) != 40 or forecast_result_checks != 24:
        raise AssertionError(f"expected 40 metric rows and 24 forecast result checks, got {len(rows)} and {forecast_result_checks}")
    max_error = max(score_errors)
    if max_error > 1e-10:
        raise AssertionError(f"D score max error {max_error}")
    return rows, loss_records, np.stack(numerators), np.stack(counts), np.stack(scales), {"rows": len(rows), "forecast_result_checks": forecast_result_checks, "raw_score_max_abs_error": max_error}


def compute_gates(plan: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    by = {(r["dataset"], r["condition"], r["seed"], r["family"]): r for r in rows}
    g1_cells = []
    for dataset in sorted(plan["data"]):
        for condition in CONDITIONS:
            seed_values = []
            for seed in SEEDS:
                joint = by[(dataset, condition, seed, "JOINT")]
                comparator = min(by[(dataset, condition, seed, fam)]["D_score"] for fam in ("F0", "HEAD", "WIDE", "CORRECTION"))
                seed_values.append(100.0 * (comparator - joint["D_score"]) / joint["F0_score"])
            g1_cells.append({"dataset": dataset, "condition": condition, "seed_values_pct_F0": seed_values, "passes": all(v > GATE_BAND for v in seed_values)})
    family_means = {family: float(np.mean([r["D_over_F0"] for r in rows if r["family"] == family])) for family in FAMILIES}
    best_global = min(family_means, key=family_means.get)
    best_source = {}
    for dataset in sorted(plan["data"]):
        means = {family: float(np.mean([r["D_over_F0"] for r in rows if r["family"] == family and r["dataset"] == dataset])) for family in FAMILIES}
        best_source[dataset] = min(means, key=means.get)
    oracle_rows = []
    seed_global, seed_source = {}, {}
    for seed in SEEDS:
        seed_key = str(seed)
        seed_global[seed_key] = min(FAMILIES, key=lambda f: np.mean([r["D_over_F0"] for r in rows if r["seed"] == seed and r["family"] == f]))
        seed_source[seed_key] = {dataset: min(FAMILIES, key=lambda f: np.mean([r["D_over_F0"] for r in rows if r["seed"] == seed and r["dataset"] == dataset and r["family"] == f])) for dataset in plan["data"]}
    for dataset in sorted(plan["data"]):
        for condition in CONDITIONS:
            for seed in SEEDS:
                candidates = [by[(dataset, condition, seed, family)] for family in FAMILIES]
                oracle = min(candidates, key=lambda r: (r["D_score"], FAMILIES.index(r["family"])))
                seed_key = str(seed)
                global_row = by[(dataset, condition, seed, seed_global[seed_key])]
                source_row = by[(dataset, condition, seed, seed_source[seed_key][dataset])]
                f0 = by[(dataset, condition, seed, "F0")]["D_score"]
                oracle_rows.append({
                    "dataset": dataset,
                    "condition": condition,
                    "seed": seed,
                    "oracle_family": oracle["family"],
                    "oracle_gain_vs_global_pct_F0": 100.0 * (global_row["D_score"] - oracle["D_score"]) / f0,
                    "oracle_gain_vs_source_pct_F0": 100.0 * (source_row["D_score"] - oracle["D_score"]) / f0,
                })
    g2_seed = {}
    for seed in SEEDS:
        items = [row for row in oracle_rows if row["seed"] == seed]
        g2_seed[str(seed)] = {
            "mean_gain_vs_global_pct_F0": float(np.mean([row["oracle_gain_vs_global_pct_F0"] for row in items])),
            "mean_gain_vs_source_pct_F0": float(np.mean([row["oracle_gain_vs_source_pct_F0"] for row in items])),
        }
    return {
        "family_mean_D_over_F0": family_means,
        "gates": {
            "G1_useful_initial_lora_gap": any(item["passes"] for item in g1_cells),
            "G1_cells": g1_cells,
            "G2_controller_headroom": all(value["mean_gain_vs_global_pct_F0"] > GATE_BAND and value["mean_gain_vs_source_pct_F0"] > GATE_BAND for value in g2_seed.values()),
            "G2_seed_summary": g2_seed,
            "best_global_hindsight_family": best_global,
            "best_source_hindsight_family": best_source,
            "G2_per_seed_global_family": seed_global,
            "G2_per_seed_source_family": seed_source,
            "G3_descriptive_only": True,
        },
        "oracle_rows": oracle_rows,
    }


def conditional_intervals(rows: list[dict[str, Any]], loss_records: list[dict[str, Any]], numerators: np.ndarray, counts: np.ndarray, scales: np.ndarray) -> dict[str, Any]:
    rng = np.random.default_rng(29344)
    draws = {}
    for dataset in sorted({row["dataset"] for row in rows}):
        starts = rng.integers(0, 20, size=(2000, 10))
        indices = np.stack([starts, (starts + 1) % 20], axis=-1).reshape(2000, 20)
        draws[dataset] = np.stack([np.bincount(index, minlength=20) for index in indices])
    row_index = {(row["dataset"], row["condition"], row["seed"], row["family"]): i for i, row in enumerate(rows)}

    def boot_score(row: dict[str, Any]) -> np.ndarray:
        i = row_index[(row["dataset"], row["condition"], row["seed"], row["family"])]
        weights = draws[row["dataset"]]
        num = np.einsum("bo,ocq->bcq", weights, numerators[i])
        den = weights @ counts[i]
        if np.any(den <= 0):
            raise AssertionError("bootstrap draw has empty target count")
        return (num / den[:, :, None] / scales[i][None, :, None]).mean(axis=(1, 2))

    result = {}
    for comparator in ("HEAD", "WIDE", "CORRECTION"):
        values = []
        for row in rows:
            if row["family"] != "JOINT":
                continue
            other = next(r for r in rows if r["dataset"] == row["dataset"] and r["condition"] == row["condition"] and r["seed"] == row["seed"] and r["family"] == comparator)
            f0 = next(r for r in rows if r["dataset"] == row["dataset"] and r["condition"] == row["condition"] and r["seed"] == row["seed"] and r["family"] == "F0")
            values.append(100.0 * (boot_score(other) - boot_score(row)) / boot_score(f0))
        result[f"JOINT_minus_{comparator}_pct_F0"] = np.quantile(np.mean(values, axis=0), [0.05, 0.95]).tolist()
    return {"interval": 0.9, "draws": 2000, "block_origins": 2, "scope": "Conditional on these two diagnostic periods; seeds and conditions share target labels.", "intervals": result, "loss_record_count": len(loss_records)}


def trajectory(plan: dict[str, Any], run: Path, seal: dict[str, Any]) -> list[dict[str, Any]]:
    forecasted = set(seal["forecast_indices"])
    rows = []
    for index, entry in enumerate(plan["jobs"]):
        result = read_json(run / "fit" / entry["key"] / "output/result.json")
        rows.append({
            "key": entry["key"],
            "dataset": entry["job"]["dataset"],
            "condition": entry["job"]["condition"],
            "seed": entry["job"]["seed"],
            "family": entry["job"]["family"],
            "recipe": entry["job"]["recipe"],
            "initial_V": result["initial_V"],
            "best_V": result["best_V"],
            "best_step": result["best_step"],
            "steps": result["steps"],
            "trajectory_seconds": result["trajectory_seconds"],
            "seconds": result["seconds"],
            "gradient_norm_min": result["gradient_norm_min"],
            "gradient_norm_max": result["gradient_norm_max"],
            "clipped_steps": result["clipped_steps"],
            "forecasted": index in forecasted,
        })
    return rows


def compare_main_outputs(results: Path, rows: list[dict[str, Any]], trajectory_rows: list[dict[str, Any]], numerators: np.ndarray, counts: np.ndarray, scales: np.ndarray, gates_payload: dict[str, Any], intervals: dict[str, Any]) -> dict[str, Any]:
    summary = read_json(results / "summary.json")
    if not summary.get("completed") or summary.get("rows") != 40 or summary.get("cells") != 8:
        raise AssertionError("main summary completion/count mismatch")
    compare_nested_floats(summary["family_mean_D_over_F0"], gates_payload["family_mean_D_over_F0"], "family_mean_D_over_F0")
    compare_nested_floats(summary["gates"], gates_payload["gates"], "gates")
    compare_nested_floats(summary["oracle_rows"], gates_payload["oracle_rows"], "oracle_rows")
    compare_nested_floats(summary["conditional_intervals"], intervals, "conditional_intervals")
    with (results / "metrics.csv").open("r", encoding="utf-8", newline="") as stream:
        metric_rows = list(csv.DictReader(stream))
    if len(metric_rows) != 40:
        raise AssertionError(f"metrics.csv row count {len(metric_rows)}")
    expected_rows = {(r["dataset"], r["condition"], str(r["seed"]), r["family"]): r for r in rows}
    for row in metric_rows:
        key = (row["dataset"], row["condition"], row["seed"], row["family"])
        expected = expected_rows[key]
        for field, value in row.items():
            if field in {"dataset", "condition", "family", "forecast_key"}:
                if str(expected[field]) != value:
                    raise AssertionError(f"metrics mismatch {key}.{field}")
            elif field == "recipe":
                if str(expected[field]) != value:
                    raise AssertionError(f"metrics mismatch {key}.recipe")
            elif value == "":
                if expected[field] is not None:
                    raise AssertionError(f"metrics mismatch {key}.{field}")
            else:
                assert_close(float(expected[field]), float(value), f"metrics {key}.{field}")
    with (results / "trajectory.csv").open("r", encoding="utf-8", newline="") as stream:
        actual_trajectory = list(csv.DictReader(stream))
    if len(actual_trajectory) != 96:
        raise AssertionError(f"trajectory.csv row count {len(actual_trajectory)}")
    with np.load(results / "per_example_losses.npz", allow_pickle=False) as z:
        np.testing.assert_allclose(z["pinball_numerator"], numerators, rtol=0, atol=0)
        np.testing.assert_allclose(z["valid_target_count"], counts, rtol=0, atol=0)
        np.testing.assert_allclose(z["scale"], scales, rtol=0, atol=0)
        np.testing.assert_allclose(z["F0"], np.asarray([r["F0_score"] for r in rows]), rtol=0, atol=0)
        keys = [json.loads(str(value)) for value in z["row_keys"]]
        if keys != [{k: r[k] for k in ("dataset", "condition", "seed", "family")} for r in rows]:
            raise AssertionError("per_example_losses row_keys mismatch")
    return {
        "summary_sha256": sha256(results / "summary.json"),
        "metrics_sha256": sha256(results / "metrics.csv"),
        "trajectory_sha256": sha256(results / "trajectory.csv"),
        "per_example_losses_sha256": sha256(results / "per_example_losses.npz"),
        "metrics_rows_checked": len(metric_rows),
        "trajectory_rows_checked": len(actual_trajectory),
    }


def check_guard(path: Path) -> dict[str, Any]:
    status = read_json(path)
    if not status.get("completed") or status.get("returncode") != 0 or status.get("reasons") not in ([], None):
        raise AssertionError(f"guard failed: {path}")
    return status


def audit_guards_chronology(run: Path, plan: dict[str, Any], seal: dict[str, Any]) -> dict[str, Any]:
    guard_count = 0
    smoke_finishes = []
    for family in ("HEAD", "WIDE", "JOINT"):
        status = check_guard(run / f"smoke/{family}/guard/status.json")
        smoke_finishes.append(parse_dt(status["finished_at"]))
        guard_count += 1
    fit_finishes = []
    for entry in plan["jobs"]:
        status = check_guard(run / "fit" / entry["key"] / "guard/status.json")
        fit_finishes.append(parse_dt(status["finished_at"]))
        guard_count += 1
    forecast_starts, forecast_finishes, forecast_records = [], [], []
    for index in seal["forecast_indices"]:
        entry = plan["jobs"][index]
        status_path = run / "forecast" / entry["key"] / "guard/status.json"
        status = check_guard(status_path)
        started_at = parse_dt(status["started_at"])
        finished_at = parse_dt(status["finished_at"])
        forecast_starts.append(started_at)
        forecast_finishes.append(finished_at)
        forecast_records.append({
            "key": entry["key"],
            "path": rel(status_path),
            "started_at": started_at,
            "finished_at": finished_at,
            "finished_at_raw": status["finished_at"],
            "mtime_utc": file_mtime_utc(status_path),
        })
        guard_count += 1
    completed_path = run / "completed.json"
    completed = read_json(completed_path)
    completed_start = parse_dt(completed["finished_utc"])
    completed_end_exclusive = completed_start + timedelta(seconds=1)
    completed_mtime = file_mtime_utc(completed_path)
    last_forecast = max(forecast_records, key=lambda item: item["finished_at"])
    completed_is_second_precision = "." not in completed["finished_utc"]
    exact_completed_order = last_forecast["finished_at"] <= completed_start
    second_floor_interval_order = (
        completed_is_second_precision
        and completed_start <= last_forecast["finished_at"] < completed_end_exclusive
        and completed_start <= completed_mtime < completed_end_exclusive
        and last_forecast["finished_at"] <= completed_mtime
    )
    checks = {
        "all_smoke_finished_before_any_fit_end": max(smoke_finishes) <= max(fit_finishes),
        "all_96_fits_finished_before_selection_seal": max(fit_finishes) <= parse_dt(seal["sealed_utc"]),
        "selection_seal_before_any_D_forecast": parse_dt(seal["sealed_utc"]) <= min(forecast_starts),
        "all_D_forecasts_finished_before_completed": exact_completed_order or second_floor_interval_order,
    }
    if guard_count != 123:
        raise AssertionError(f"expected 123 guards, got {guard_count}")
    if not all(checks.values()):
        raise AssertionError(f"chronology checks failed: {checks}")
    chronology_evidence = {
        "completed_finished_utc_raw": completed["finished_utc"],
        "completed_marker_precision": "second_floor_interval" if completed_is_second_precision else "subsecond_or_unknown",
        "completed_marker_interval_start_utc": completed_start.isoformat(),
        "completed_marker_interval_end_exclusive_utc": completed_end_exclusive.isoformat(),
        "completed_json_mtime_utc": completed_mtime.isoformat(),
        "last_forecast_key": last_forecast["key"],
        "last_forecast_status_path": last_forecast["path"],
        "last_forecast_finished_at_raw": last_forecast["finished_at_raw"],
        "last_forecast_finished_at_utc": last_forecast["finished_at"].isoformat(),
        "last_forecast_status_mtime_utc": last_forecast["mtime_utc"].isoformat(),
        "exact_completed_order": exact_completed_order,
        "second_floor_interval_order": second_floor_interval_order,
        "last_forecast_seconds_after_completed_marker_start": (last_forecast["finished_at"] - completed_start).total_seconds(),
        "completed_file_mtime_seconds_after_last_forecast": (completed_mtime - last_forecast["finished_at"]).total_seconds(),
        "interpretation": "completed.finished_utc is stored after replace(microsecond=0), so this audit treats a second-precision value t as the interval [t, t+1s) only for the final completed-marker boundary and requires the completed file mtime to be after the last forecast guard.",
    }
    return {"guards_checked": guard_count, "checks": checks, "chronology_evidence": chronology_evidence}


def run_audit(run: Path, results: Path) -> dict[str, Any]:
    output_path = run / "independent_audit.json"
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    plan_path = run / "plan.json"
    plan = read_json(plan_path)
    plan_hash = sha256(plan_path)
    completed = read_json(run / "completed.json")
    if not completed.get("completed") or completed.get("plan_sha256") != plan_hash:
        raise AssertionError("run completion/plan hash mismatch")
    hash_audit = verify_hashes(plan)
    cells, fit_hashes, fit_audit = audit_fit_jobs(run, plan, plan_hash)
    selection_audit = compare_selection_seal(run, plan, plan_hash, cells, fit_hashes)
    seal = read_json(run / "selection_sealed.json")
    rows, loss_records, numerators, counts, scales, forecast_audit = recompute_metrics(run, plan, seal, plan_hash)
    gates_payload = compute_gates(plan, rows)
    intervals = conditional_intervals(rows, loss_records, numerators, counts, scales)
    trajectory_rows = trajectory(plan, run, seal)
    main_comparison = compare_main_outputs(results, rows, trajectory_rows, numerators, counts, scales, gates_payload, intervals)
    guard_audit = audit_guards_chronology(run, plan, seal)
    payload = {
        "completed": True,
        "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "script_sha256_before_write": sha256(Path(__file__)),
        "plan_sha256": plan_hash,
        "hash_audit": hash_audit,
        "fit_audit": fit_audit,
        "selection_audit": selection_audit,
        "forecast_metric_audit": forecast_audit,
        "rows": len(rows),
        "family_mean_D_over_F0": gates_payload["family_mean_D_over_F0"],
        "gates": gates_payload["gates"],
        "conditional_intervals": intervals,
        "main_output_comparison": main_comparison,
        "guard_chronology_audit": guard_audit,
        "claim_limits": [
            "development-only diagnostic; no final-test labels prepared or opened",
            "two source periods only; seeds and conditions share labels",
            "same raw source families as prior work; checked target-label periods do not overlap known local exposures",
        ],
    }
    payload["independent_audit_sha256"] = write_json_x(output_path, payload)
    return payload


def record_failure(run: Path, exc: BaseException) -> None:
    target = run / "independent_audit_failure.json"
    if target.exists():
        index = 2
        while (run / f"independent_audit_failure_{index:02d}.json").exists():
            index += 1
        target = run / f"independent_audit_failure_{index:02d}.json"
    write_json_x(target, {
        "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "script_sha256": sha256(Path(__file__)),
        "error_type": type(exc).__name__,
        "error_message": str(exc),
    })


def self_test() -> dict[str, Any]:
    prediction = np.array([[[[0.0, 1.0], [1.0, 2.0]]]], dtype=np.float64)
    target = np.array([[[0.5, 1.5]]], dtype=np.float64)
    score = score_arrays(prediction, target, np.array([0.25, 0.75]), np.array([1.0]))
    assert abs(score - 0.25) < 1e-12
    candidates = correction_candidates()
    assert [c["name"] for c in candidates] == ["identity", "affine_s0.25", "affine_s0.5", "affine_s1", "bias_s0.25", "bias_s0.5", "bias_s1"]
    return {"self_test": True, "script_sha256": sha256(Path(__file__)), "families": list(FAMILIES)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()
    if args.run == args.self_test:
        parser.error("choose exactly one of --run or --self-test")
    try:
        payload = run_audit(args.run_dir, args.results_dir) if args.run else self_test()
    except BaseException as exc:
        if args.run:
            record_failure(args.run_dir, exc)
        raise
    print(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
