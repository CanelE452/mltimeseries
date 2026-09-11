"""Analyse the Study34 initial-headroom diagnostic."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "runs/peft_initial_headroom_v1"
OUT = ROOT / "results/peft_initial_headroom_v1"
FAMILIES = ("F0", "HEAD", "WIDE", "JOINT", "CORRECTION")
NEURAL_FAMILIES = ("HEAD", "WIDE", "JOINT")
CONDITIONS = ("FULL90", "SPREAD30")
SEEDS = (29000, 29001)
GATE_BAND = 0.25


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def pinball_components(prediction: np.ndarray, target: np.ndarray, quantiles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    prediction = np.sort(prediction.astype(np.float64), axis=2)
    target = target.astype(np.float64)
    error = target[:, :, None, :] - prediction
    q = quantiles[None, None, :, None].astype(np.float64)
    valid = np.isfinite(target)
    numerator = np.where(valid[:, :, None, :], 2.0 * np.maximum(q * error, (q - 1.0) * error), 0.0).sum(axis=3)
    counts = valid.sum(axis=2).astype(np.float64)
    return numerator, counts


def score_arrays(prediction: np.ndarray, target: np.ndarray, quantiles: np.ndarray, scale: np.ndarray) -> float:
    numerator, counts = pinball_components(prediction, target, quantiles)
    if not np.isfinite(numerator).all() or np.any(counts.sum(axis=0) <= 0) or np.any(scale <= 0):
        raise ValueError("empty target count in score")
    return float((numerator.sum(axis=0) / counts.sum(axis=0)[:, None] / scale[:, None]).mean())


def score_npz(path: Path) -> float:
    with np.load(path, allow_pickle=False) as z:
        return score_arrays(z["prediction"], z["target"], z["quantiles"], z["scale"])


def correction_parameters(train_npz: Path) -> dict:
    with np.load(train_npz, allow_pickle=False) as z:
        prediction = np.sort(z["prediction"].astype(np.float64), axis=2)
        target = z["target"].astype(np.float64)
        quantiles = z["quantiles"].astype(np.float64)
        scale = z["scale"].astype(np.float64)
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
        slope = float(np.clip(np.mean((x-x.mean())*(y-y.mean())) / (np.var(x)+ridge), 0.25, 4.0))
        intercept = float(np.mean(y - slope * x) * scale[channel])
        bias = float(np.mean(y - x) * scale[channel])
        slopes.append(slope)
        intercepts.append(intercept)
        biases.append(bias)
    return {
        "median_quantile_index": median_index,
        "ridge": ridge,
        "slope_clip": [0.25, 4.0],
        "slopes": slopes,
        "intercepts": intercepts,
        "biases": biases,
        "fit_counts": counts,
    }


def apply_correction(prediction: np.ndarray, params: dict, candidate: dict) -> np.ndarray:
    corrected = np.array(prediction, dtype=np.float64, copy=True)
    shrink = float(candidate["shrinkage"])
    if candidate["kind"] == "identity":
        return corrected
    if candidate["kind"] == "affine":
        slopes = np.asarray(params["slopes"], dtype=np.float64)
        intercepts = np.asarray(params["intercepts"], dtype=np.float64)
        for channel in range(corrected.shape[1]):
            corrected[:, channel] = corrected[:, channel] + shrink * (
                slopes[channel] * corrected[:, channel] + intercepts[channel] - corrected[:, channel]
            )
    elif candidate["kind"] == "bias":
        biases = np.asarray(params["biases"], dtype=np.float64)
        for channel in range(corrected.shape[1]):
            corrected[:, channel] = corrected[:, channel] + shrink * biases[channel]
    else:
        raise ValueError(candidate)
    return np.sort(corrected, axis=2)


def correction_candidates() -> list[dict]:
    result = [{"name": "identity", "kind": "identity", "shrinkage": 0.0}]
    for shrinkage in (0.25, 0.5, 1.0):
        result.append({"name": f"affine_s{shrinkage:g}", "kind": "affine", "shrinkage": shrinkage})
    for shrinkage in (0.25, 0.5, 1.0):
        result.append({"name": f"bias_s{shrinkage:g}", "kind": "bias", "shrinkage": shrinkage})
    return result


def choose_correction(train_npz: Path, val_npz: Path) -> dict:
    params = correction_parameters(train_npz)
    with np.load(val_npz, allow_pickle=False) as z:
        prediction = z["prediction"].astype(np.float64)
        target = z["target"].astype(np.float64)
        quantiles = z["quantiles"].astype(np.float64)
        scale = z["scale"].astype(np.float64)
    scored = []
    for order, candidate in enumerate(correction_candidates()):
        corrected = apply_correction(prediction, params, candidate)
        scored.append(candidate | {"order": order, "V": score_arrays(corrected, target, quantiles, scale)})
    selected = min(scored, key=lambda item: (item["V"], item["order"]))
    return {"parameters": params, "candidates": scored, "selected": selected}


def job_cell(job: dict) -> str:
    return f"{job['dataset']}/{job['condition']}/s{job['seed']}"


def summarize_sealed_fit(plan: dict) -> dict:
    cells: dict[str, dict] = {}
    fit_hashes: dict[str, str] = {}
    forecast_indices: set[int] = set()
    family_hashes: dict[str, dict[str, str]] = {}
    paired = {}

    for index, entry in enumerate(plan["jobs"]):
        job = entry["job"]
        result_path = RUN / "fit" / entry["key"] / "output/result.json"
        result = read_json(result_path)
        if not result["completed"] or result["job"] != job:
            raise AssertionError(entry["key"])
        assert result['plan_sha256'] == sha256(RUN/'plan.json')
        assert result['frozen_verified'] and result['restore_exact'] and not result['D_opened']
        folder = result_path.parent
        assert sha256(folder/'best.pt') == result['checkpoint_sha256']
        selected = min(result['history'], key=lambda p: (p['V'], p['step']))
        assert (selected['V'], selected['step']) == (result['best_V'], result['best_step'])
        assert abs(score_npz(folder/'selected_val.npz')-result['best_V']) < 1e-12
        fit_hashes[entry["key"]] = sha256(result_path)
        cell = job_cell(job)
        with np.load(folder/'F0_val.npz') as z:
            initial = z['prediction'].copy()
        paired.setdefault(cell, []).append((job['family'], result['initial_head_hash'], result['sample_sha256'], initial))
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
        if current is None or (candidate["best_V"], candidate["best_step"], candidate["recipe"]) < (
            current["best_V"],
            current["best_step"],
            current["recipe"],
        ):
            cells[cell]["families"][job["family"]] = candidate
        family_hashes.setdefault(cell, {})[job["family"]] = result["initial_head_hash"]

    by_cell_recipe: dict[tuple[str, str, int], int] = {}
    for index, entry in enumerate(plan["jobs"]):
        job = entry["job"]
        by_cell_recipe[(job_cell(job), job["family"], job["recipe"])] = index

    for cell, payload in cells.items():
        assert len(paired[cell]) == 12
        assert len({item[1] for item in paired[cell] if item[0] in ('HEAD', 'JOINT')}) == 1
        assert len({item[2] for item in paired[cell]}) == 1
        for item in paired[cell][1:]:
            np.testing.assert_array_equal(item[3], paired[cell][0][3])
        if set(payload["families"]) != set(NEURAL_FAMILIES):
            raise AssertionError(f"missing family in {cell}")
        if family_hashes[cell]["HEAD"] != family_hashes[cell]["JOINT"]:
            raise AssertionError(f"HEAD/JOINT initial head mismatch in {cell}")
        for family in NEURAL_FAMILIES:
            forecast_indices.add(payload["families"][family]["index"])
        f0_index = by_cell_recipe[(cell, "HEAD", 0)]
        f0_entry = plan["jobs"][f0_index]
        f0_dir = RUN / "fit" / f0_entry["key"] / "output"
        payload["f0_index"] = payload['families']['HEAD']['index']
        payload["f0_key"] = payload['families']['HEAD']['key']
        payload["correction"] = choose_correction(f0_dir / "F0_train.npz", f0_dir / "F0_val.npz")

    return {
        "cells": cells,
        "fit_result_hashes": fit_hashes,
        "forecast_indices": sorted(forecast_indices),
        "fit_only_selection": True,
        "D_opened": False,
    }


def sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _forecast_dir(key: str) -> Path:
    return RUN / "forecast" / key / "output"


def load_family_prediction(cell: dict, family: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, str]:
    if family == "F0":
        key = cell["f0_key"]
        name = "F0"
    elif family == "CORRECTION":
        key = cell["f0_key"]
        name = "F0"
    else:
        key = cell["families"][family]["key"]
        name = "selected"
    with np.load(_forecast_dir(key) / f"{name}.npz", allow_pickle=False) as z:
        prediction = z["prediction"].astype(np.float64)
        target = z["target"].astype(np.float64)
        quantiles = z["quantiles"].astype(np.float64)
        scale = z["scale"].astype(np.float64)
    if family == "CORRECTION":
        prediction = apply_correction(prediction, cell["correction"]["parameters"], cell["correction"]["selected"])
    score = score_arrays(prediction, target, quantiles, scale)
    return prediction, target, quantiles, scale, score, key


def conditional_intervals(rows: list[dict], loss_records: list[dict], numerators: np.ndarray, counts: np.ndarray, scales: np.ndarray) -> dict:
    rng = np.random.default_rng(29344)
    draws = {}
    for dataset in sorted({row["dataset"] for row in rows}):
        starts = rng.integers(0, 20, size=(2000, 10))
        indices = np.stack([starts, (starts + 1) % 20], axis=-1).reshape(2000, 20)
        draws[dataset] = np.stack([np.bincount(index, minlength=20) for index in indices])
    row_index = {(row["dataset"], row["condition"], row["seed"], row["family"]): i for i, row in enumerate(rows)}

    def boot_score(row: dict) -> np.ndarray:
        i = row_index[(row["dataset"], row["condition"], row["seed"], row["family"])]
        weights = draws[row["dataset"]]
        num = np.einsum("bo,ocq->bcq", weights, numerators[i])
        den = weights @ counts[i]
        return (num / den[:, :, None] / scales[i][None, :, None]).mean(axis=(1, 2))

    result = {}
    for comparator in ("HEAD", "WIDE", "CORRECTION"):
        values = []
        for row in rows:
            if row["family"] != "JOINT":
                continue
            other = next(
                r for r in rows
                if r["dataset"] == row["dataset"]
                and r["condition"] == row["condition"]
                and r["seed"] == row["seed"]
                and r["family"] == comparator
            )
            f0 = next(
                r for r in rows
                if r["dataset"] == row["dataset"]
                and r["condition"] == row["condition"]
                and r["seed"] == row["seed"]
                and r["family"] == "F0"
            )
            values.append(100.0 * (boot_score(other) - boot_score(row)) / boot_score(f0))
        merged = np.mean(values, axis=0)
        result[f"JOINT_minus_{comparator}_pct_F0"] = np.quantile(merged, [0.05, 0.95]).tolist()
    return {
        "interval": 0.9,
        "draws": 2000,
        "block_origins": 2,
        "scope": "Conditional on these two diagnostic periods; seeds and conditions share target labels.",
        "intervals": result,
        "loss_record_count": len(loss_records),
    }


def analyse() -> dict:
    completed = read_json(RUN / "completed.json")
    if not completed["completed"]:
        raise AssertionError("run is not complete")
    plan = read_json(RUN / "plan.json")
    seal = read_json(RUN / "selection_sealed.json")
    if seal["plan_sha256"] != sha256(RUN / "plan.json"):
        raise AssertionError("selection seal does not match plan")

    rows: list[dict] = []
    loss_records: list[dict] = []
    numerators: list[np.ndarray] = []
    counts: list[np.ndarray] = []
    scales: list[np.ndarray] = []
    raw_errors = []
    for cell_key, cell in sorted(seal["cells"].items()):
        f0_score = None
        for family in FAMILIES:
            prediction, target, quantiles, scale, score, forecast_key = load_family_prediction(cell, family)
            numerator, count = pinball_components(prediction, target, quantiles)
            recomputed = float((numerator.sum(axis=0) / count.sum(axis=0)[:, None] / scale[:, None]).mean())
            raw_errors.append(abs(score - recomputed))
            if family == "F0":
                f0_score = score
            assert f0_score is not None
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
                row.update({
                    "recipe": cell["correction"]["selected"]["name"],
                    "best_V": cell["correction"]["selected"]["V"],
                    "best_step": 0,
                    "trainable": {'identity': 0, 'bias': 2, 'affine': 4}[cell['correction']['selected']['kind']],
                })
            else:
                row.update({"recipe": "identity", "best_V": None, "best_step": 0, "trainable": 0})
            rows.append(row)
            loss_records.append({k: row[k] for k in ("dataset", "condition", "seed", "family")})
            numerators.append(numerator)
            counts.append(count)
            scales.append(scale)

    by = {(r["dataset"], r["condition"], r["seed"], r["family"]): r for r in rows}
    g1_cells = []
    for dataset in sorted(plan["data"]):
        for condition in CONDITIONS:
            seed_values = []
            for seed in SEEDS:
                joint = by[(dataset, condition, seed, "JOINT")]
                comparator = min(by[(dataset, condition, seed, fam)]["D_score"] for fam in ("F0", "HEAD", "WIDE", "CORRECTION"))
                value = 100.0 * (comparator - joint["D_score"]) / joint["F0_score"]
                seed_values.append(value)
            g1_cells.append({"dataset": dataset, "condition": condition, "seed_values_pct_F0": seed_values, "passes": all(v > GATE_BAND for v in seed_values)})
    g1 = any(item["passes"] for item in g1_cells)

    family_means = {
        family: float(np.mean([r["D_over_F0"] for r in rows if r["family"] == family]))
        for family in FAMILIES
    }
    best_global = min(family_means, key=family_means.get)
    best_source = {}
    for dataset in sorted(plan["data"]):
        source_means = {
            family: float(np.mean([r["D_over_F0"] for r in rows if r["family"] == family and r["dataset"] == dataset]))
            for family in FAMILIES
        }
        best_source[dataset] = min(source_means, key=source_means.get)

    oracle_rows = []
    seed_global, seed_source = {}, {}
    for seed in SEEDS:
        seed_global[seed] = min(FAMILIES, key=lambda f: np.mean([r['D_over_F0'] for r in rows if r['seed'] == seed and r['family'] == f]))
        seed_source[seed] = {ds: min(FAMILIES, key=lambda f: np.mean([r['D_over_F0'] for r in rows if r['seed'] == seed and r['dataset'] == ds and r['family'] == f])) for ds in plan['data']}
    for dataset in sorted(plan["data"]):
        for condition in CONDITIONS:
            for seed in SEEDS:
                candidates = [by[(dataset, condition, seed, family)] for family in FAMILIES]
                oracle = min(candidates, key=lambda r: (r["D_score"], FAMILIES.index(r["family"])))
                global_row = by[(dataset, condition, seed, seed_global[seed])]
                source_row = by[(dataset, condition, seed, seed_source[seed][dataset])]
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
    g2 = all(
        value["mean_gain_vs_global_pct_F0"] > GATE_BAND and value["mean_gain_vs_source_pct_F0"] > GATE_BAND
        for value in g2_seed.values()
    )

    trajectory = []
    for index, entry in enumerate(plan["jobs"]):
        result = read_json(RUN / "fit" / entry["key"] / "output/result.json")
        trajectory.append({
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
            "forecasted": index in set(seal["forecast_indices"]),
        })

    numerators_array = np.stack(numerators)
    counts_array = np.stack(counts)
    scales_array = np.stack(scales)
    summary = {
        "completed": True,
        "analysed_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "plan_sha256": sha256(RUN / "plan.json"),
        "selection_sha256": sha256(RUN / "selection_sealed.json"),
        "cells": 8,
        "families": list(FAMILIES),
        "rows": len(rows),
        "raw_score_max_abs_error": max(raw_errors),
        "family_mean_D_over_F0": family_means,
        "gates": {
            "G1_useful_initial_lora_gap": g1,
            "G1_cells": g1_cells,
            "G2_controller_headroom": g2,
            "G2_seed_summary": g2_seed,
            "best_global_hindsight_family": best_global,
            "best_source_hindsight_family": best_source,
            "G2_per_seed_global_family": seed_global,
            "G2_per_seed_source_family": seed_source,
            "G3_descriptive_only": True,
        },
        "oracle_rows": oracle_rows,
        "conditional_intervals": conditional_intervals(rows, loss_records, numerators_array, counts_array, scales_array),
        "trajectory": trajectory,
        "limitations": [
            "development-only diagnostic; no final-test labels prepared or opened",
            "two source periods only; seeds and conditions share labels",
            "same raw source families as prior work; target-label periods are nonoverlapping under checked local histories",
            "equal trainable count does not equal equal function class or equal optimization difficulty",
        ],
    }

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (OUT / "trajectory.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(trajectory[0]))
        writer.writeheader()
        writer.writerows(trajectory)
    np.savez_compressed(
        OUT / "per_example_losses.npz",
        pinball_numerator=numerators_array,
        valid_target_count=counts_array,
        scale=scales_array,
        F0=np.asarray([r['F0_score'] for r in rows]),
        row_keys=np.asarray([json.dumps(record, sort_keys=True) for record in loss_records]),
    )
    write_json(OUT / "summary.json", summary)
    write_json(OUT / "selection_sealed.json", seal)
    plot(rows, summary)
    return summary


def plot(rows: list[dict], summary: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = []
    values = []
    colors = []
    for item in summary["gates"]["G1_cells"]:
        labels.extend([f"{item['dataset']} {item['condition']} s{seed}" for seed in SEEDS])
        values.extend(item["seed_values_pct_F0"])
        colors.extend(["#007e87" if v > GATE_BAND else "#b54d2b" for v in item["seed_values_pct_F0"]])
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.bar(range(len(values)), values, color=colors)
    ax.axhline(GATE_BAND, color="black", linestyle="--", linewidth=1, label="G1 threshold")
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("JOINT gain over best simple comparator (%F0)")
    ax.set_title("Initial LoRA headroom gate on sealed diagnostic origins")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "01_g1_initial_lora_gap.png", dpi=180)
    plt.close(fig)

    family_order = list(FAMILIES)
    cell_order = [f"{dataset}/{condition}/s{seed}" for dataset in sorted({r["dataset"] for r in rows}) for condition in CONDITIONS for seed in SEEDS]
    matrix = np.full((len(family_order), len(cell_order)), np.nan)
    for i, family in enumerate(family_order):
        for j, cell in enumerate(cell_order):
            dataset, condition, seed_text = cell.split("/")
            seed = int(seed_text[1:])
            row = next(r for r in rows if r["dataset"] == dataset and r["condition"] == condition and r["seed"] == seed and r["family"] == family)
            matrix[i, j] = row["D_over_F0"]
    fig, ax = plt.subplots(figsize=(13, 4.8))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis_r")
    ax.set_yticks(range(len(family_order)))
    ax.set_yticklabels(family_order)
    ax.set_xticks(range(len(cell_order)))
    ax.set_xticklabels(cell_order, rotation=35, ha="right")
    ax.set_title("Selected diagnostic loss by family (D/F0, lower is better)")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", fontsize=7, color="white" if matrix[i, j] < np.nanmean(matrix) else "black")
    fig.tight_layout()
    fig.savefig(OUT / "02_family_heatmap.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.2))
    for family in NEURAL_FAMILIES:
        subset = [r for r in summary["trajectory"] if r["family"] == family]
        ax.scatter([r["trajectory_seconds"] for r in subset], [r["best_V"] / r["initial_V"] for r in subset], label=family, s=28, alpha=0.8)
    ax.set_xlabel("Fit trajectory seconds")
    ax.set_ylabel("Best V / initial V")
    ax.set_title("Validation adaptation response and cost")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "03_validation_cost.png", dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seal-only", action="store_true")
    args = parser.parse_args()
    plan = read_json(RUN / "plan.json")
    if args.seal_only:
        assert not (RUN/'selection_sealed.json').exists()
        payload = summarize_sealed_fit(plan)
        payload["sealed_utc"] = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        payload["plan_sha256"] = sha256(RUN / "plan.json")
        write_json(RUN / "selection_sealed.json", payload)
        print(json.dumps({"sealed": True, "forecast_indices": payload["forecast_indices"]}, indent=2), flush=True)
        return 0
    summary = analyse()
    print(json.dumps({
        "completed": summary["completed"],
        "G1": summary["gates"]["G1_useful_initial_lora_gap"],
        "G2": summary["gates"]["G2_controller_headroom"],
        "best_global": summary["gates"]["best_global_hindsight_family"],
        "family_mean_D_over_F0": summary["family_mean_D_over_F0"],
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
