"""V2 month leave-one-out sensitivity for Hospital alpha selection.

Reads completed Hospital V2 gate artifacts only. It does not train a model,
inspect E, tune a new rule, or modify source run outputs.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "runs" / "hospital_shared_strength_v1_run2"
PREP = ROOT / "runs" / "hospital_shared_strength_v1_prepared"
OUT = ROOT / "results" / "peft_mechanism_diagnostics_v1" / "hospital_strength"
POLICY = RUN / "stages" / "select_gates" / "attempt_01" / "output" / "policies.json"
FORECAST_GATE = RUN / "stages" / "forecast_gate" / "attempt_01" / "output"
GATE_PANEL = PREP / "gate.npz"


COLORS = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "gray": "#737373",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def tied_choice(loss: np.ndarray, grid: np.ndarray, preferred: float, tol: float = 1e-12) -> int:
    loss = np.asarray(loss, dtype=np.float64)
    grid = np.asarray(grid, dtype=np.float64)
    if loss.shape != grid.shape or loss.ndim != 1 or not np.isfinite(loss).all():
        raise ValueError("Invalid grid losses")
    ties = np.flatnonzero(loss <= loss.min() + tol)
    return int(min(ties, key=lambda j: (round(abs(float(grid[j]) - preferred), 14), float(grid[j]))))


def combine(f0: np.ndarray, lora: np.ndarray, alpha: float) -> np.ndarray:
    return f0 + float(alpha) * (lora - f0)


def monthly_loss_grid(
    f0: np.ndarray,
    lora: np.ndarray,
    target: np.ndarray,
    scale: np.ndarray,
    quantiles: np.ndarray,
    grid: np.ndarray,
) -> np.ndarray:
    if f0.shape != lora.shape or f0.ndim != 4 or f0.shape[1] != 1:
        raise ValueError("Expected V2 prediction axes [series, 1, quantile, 12]")
    if target.shape != (f0.shape[0], 12):
        raise ValueError("Expected target axes [series, 12]")
    if scale.shape != (f0.shape[0],) or quantiles.shape != (f0.shape[2],):
        raise ValueError("Scale or quantile axes differ")
    if not all(np.isfinite(x).all() for x in (f0, lora, target, scale, quantiles)):
        raise ValueError("Nonfinite V2 data")
    if np.any(scale <= 0) or np.any(np.diff(quantiles) <= 0):
        raise ValueError("Invalid scale or quantiles")
    if np.any(np.diff(f0, axis=2) < -1e-10) or np.any(np.diff(lora, axis=2) < -1e-10):
        raise ValueError("Predictions must be sorted before convex combination")

    losses = []
    y = target[:, None, :]  # [series, 1, month]
    q = quantiles[None, :, None]
    for alpha in grid:
        pred = combine(f0[:, 0], lora[:, 0], float(alpha))  # [series, quantile, month]
        err = y - pred
        loss = 2.0 * np.maximum(q * err, (q - 1.0) * err) / scale[:, None, None]
        losses.append(loss.mean(axis=1))  # [series, month]
    return np.stack(losses, axis=2)  # [series, month, alpha]


def select_alpha(loss_grid: np.ndarray, grid: np.ndarray, tol: float = 1e-12) -> tuple[float, np.ndarray, np.ndarray]:
    global_idx = tied_choice(loss_grid.mean(axis=0), grid, 0.0, tol)
    global_alpha = float(grid[global_idx])
    indices = np.array([tied_choice(row, grid, global_alpha, tol) for row in loss_grid], dtype=np.int64)
    return global_alpha, grid[indices], indices


def histogram(values: np.ndarray, grid: np.ndarray) -> dict[str, int]:
    return {f"{alpha:.1f}": int(np.sum(np.isclose(values, alpha, rtol=0, atol=1e-12))) for alpha in grid}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    policy = read_json(POLICY)

    with np.load(GATE_PANEL, allow_pickle=False) as panel:
        values = panel["values"].astype(np.float64)
        series_ids = panel["series_ids"]
        scale = panel["scale"].astype(np.float64)
        target = values[:, 48:60].astype(np.float64)
    if target.shape != (len(series_ids), 12):
        raise AssertionError("V2 target must be exactly 12 held-out gate months")

    with np.load(FORECAST_GATE / "f0.npz", allow_pickle=False) as f:
        f0_prediction = f["prediction"].astype(np.float64)
        f0_unsorted = f["unsorted_prediction"].astype(np.float64)
        f0_ids = f["series_ids"]
        origins = f["origins"]
        quantiles = f["quantiles"].astype(np.float64)
        f0_scale = f["scale"].astype(np.float64)
        f0_stage = str(f["stage"])

    if f0_stage != "gate" or not np.array_equal(origins, np.array([48])):
        raise AssertionError("F0 forecast is not the frozen V2 gate forecast")
    if not np.array_equal(f0_prediction, np.sort(f0_unsorted, axis=2)):
        raise AssertionError("F0 sorted prediction does not match unsorted archive")
    if not np.array_equal(series_ids, f0_ids) or not np.array_equal(scale, f0_scale):
        raise AssertionError("Gate panel and F0 forecast axes differ")

    rows: list[dict] = []
    seed_summaries: list[dict] = []
    month_summaries_by_seed: dict[int, list[dict]] = {}
    source_hashes = {
        "gate_panel": sha256(GATE_PANEL),
        "policies": sha256(POLICY),
        "f0_gate": sha256(FORECAST_GATE / "f0.npz"),
    }

    for policy_item in policy["policies"]:
        seed = int(policy_item["seed"])
        strength_path = Path(policy_item["strengths_path"])
        source_hashes[f"strengths_{seed}"] = sha256(strength_path)
        source_hashes[f"lora_gate_{seed}"] = sha256(FORECAST_GATE / f"lora_{seed}.npz")

        with np.load(strength_path, allow_pickle=False) as s:
            grid = s["grid"].astype(np.float64)
            original_alpha = s["alpha"].astype(np.float64)
            original_global_alpha = float(s["global_alpha"])
            original_loss_grid = s["v2_loss_grid"].astype(np.float64)
            strength_ids = s["series_ids"]
            strength_scale = s["scale"].astype(np.float64)
            strength_quantiles = s["quantiles"].astype(np.float64)

        with np.load(FORECAST_GATE / f"lora_{seed}.npz", allow_pickle=False) as l:
            lora_prediction = l["prediction"].astype(np.float64)
            lora_unsorted = l["unsorted_prediction"].astype(np.float64)
            lora_ids = l["series_ids"]
            lora_origins = l["origins"]
            lora_quantiles = l["quantiles"].astype(np.float64)
            lora_scale = l["scale"].astype(np.float64)
            lora_stage = str(l["stage"])
            lora_checkpoint = str(l["checkpoint_sha256"])

        if not np.allclose(grid, np.linspace(0, 1, 11), rtol=0, atol=1e-14):
            raise AssertionError("Unexpected alpha grid")
        if lora_stage != "gate" or not np.array_equal(lora_origins, np.array([48])):
            raise AssertionError("LoRA forecast is not a V2 gate forecast")
        if not np.array_equal(lora_prediction, np.sort(lora_unsorted, axis=2)):
            raise AssertionError("LoRA sorted prediction does not match unsorted archive")
        if str(policy_item["checkpoint_sha256"]) != lora_checkpoint:
            raise AssertionError("LoRA gate forecast checkpoint differs from policy")
        for name, ids in {"strength": strength_ids, "lora": lora_ids}.items():
            if not np.array_equal(series_ids, ids):
                raise AssertionError(f"{name} series order differs")
        for name, other_scale in {"strength": strength_scale, "lora": lora_scale}.items():
            if not np.array_equal(scale, other_scale):
                raise AssertionError(f"{name} scale differs")
        if not np.array_equal(quantiles, lora_quantiles) or not np.array_equal(quantiles, strength_quantiles):
            raise AssertionError("Quantile grid differs")

        month_alpha_loss = monthly_loss_grid(f0_prediction, lora_prediction, target, scale, quantiles, grid)
        full12_loss_grid = month_alpha_loss.mean(axis=1)
        full_global_alpha, full_alpha, _ = select_alpha(full12_loss_grid, grid)
        loss_grid_max_abs_diff = float(np.max(np.abs(full12_loss_grid - original_loss_grid)))
        if loss_grid_max_abs_diff > 1e-12:
            raise AssertionError(f"Recomputed full V2 loss grid differs: {loss_grid_max_abs_diff}")
        if abs(full_global_alpha - original_global_alpha) > 1e-12:
            raise AssertionError("Recomputed full global alpha differs")
        if not np.array_equal(full_alpha, original_alpha):
            raise AssertionError("Recomputed full individual alpha differs")
        if abs(float(policy_item["global_alpha"]) - original_global_alpha) > 1e-12:
            raise AssertionError("Policy global alpha differs from strengths")
        if histogram(original_alpha, grid) != {str(k): int(v) for k, v in policy_item["alpha_histogram"].items()}:
            raise AssertionError("Policy alpha histogram differs from strengths")

        changed_counts = np.zeros(len(series_ids), dtype=np.int64)
        endpoint_flip_counts = np.zeros(len(series_ids), dtype=np.int64)
        abs_delta_sum = np.zeros(len(series_ids), dtype=np.float64)
        max_abs_delta = np.zeros(len(series_ids), dtype=np.float64)
        lomo_values = np.empty((len(series_ids), 12), dtype=np.float64)
        month_summaries: list[dict] = []

        for leave_month in range(12):
            lomo_loss_grid = (month_alpha_loss.sum(axis=1) - month_alpha_loss[:, leave_month, :]) / 11.0
            lomo_global_alpha, lomo_alpha, _ = select_alpha(lomo_loss_grid, grid)
            delta = lomo_alpha - original_alpha
            abs_delta = np.abs(delta)
            changed = abs_delta > 1e-12
            endpoint_flip = ((original_alpha == 0.0) & (lomo_alpha == 1.0)) | (
                (original_alpha == 1.0) & (lomo_alpha == 0.0)
            )

            changed_counts += changed.astype(np.int64)
            endpoint_flip_counts += endpoint_flip.astype(np.int64)
            abs_delta_sum += abs_delta
            max_abs_delta = np.maximum(max_abs_delta, abs_delta)
            lomo_values[:, leave_month] = lomo_alpha

            month_summaries.append(
                {
                    "leave_month_offset_0based": leave_month,
                    "calendar_year": 2004,
                    "calendar_month_1based": leave_month + 1,
                    "absolute_month_1based": 49 + leave_month,
                    "lomo_global_alpha": float(lomo_global_alpha),
                    "changed_count": int(changed.sum()),
                    "changed_fraction": float(changed.mean()),
                    "mean_abs_delta_alpha": float(abs_delta.mean()),
                    "median_abs_delta_alpha": float(np.median(abs_delta)),
                    "max_abs_delta_alpha": float(abs_delta.max()),
                    "endpoint_flip_count": int(endpoint_flip.sum()),
                    "endpoint_flip_fraction": float(endpoint_flip.mean()),
                    "lomo_alpha_histogram": histogram(lomo_alpha, grid),
                }
            )

        month_summaries_by_seed[seed] = month_summaries
        mean_abs_delta = abs_delta_sum / 12.0
        changed_fraction_by_series = changed_counts / 12.0
        endpoint_flip_fraction_by_series = endpoint_flip_counts / 12.0

        for i, sid in enumerate(series_ids):
            rows.append(
                {
                    "seed": seed,
                    "series_id": str(sid),
                    "full12_alpha": float(original_alpha[i]),
                    "lomo_alpha_mean": float(lomo_values[i].mean()),
                    "lomo_alpha_std": float(lomo_values[i].std(ddof=0)),
                    "lomo_alpha_min": float(lomo_values[i].min()),
                    "lomo_alpha_max": float(lomo_values[i].max()),
                    "changed_months": int(changed_counts[i]),
                    "changed_month_fraction": float(changed_fraction_by_series[i]),
                    "mean_abs_delta_alpha": float(mean_abs_delta[i]),
                    "max_abs_delta_alpha": float(max_abs_delta[i]),
                    "endpoint_flip_months": int(endpoint_flip_counts[i]),
                    "endpoint_flip_month_fraction": float(endpoint_flip_fraction_by_series[i]),
                }
            )

        all_lomo = lomo_values.reshape(-1)
        seed_summaries.append(
            {
                "seed": seed,
                "series_count": int(len(series_ids)),
                "full12_global_alpha": float(original_global_alpha),
                "lomo_global_alpha_histogram": {
                    f"{alpha:.1f}": int(sum(np.isclose(m["lomo_global_alpha"], alpha, rtol=0, atol=1e-12) for m in month_summaries))
                    for alpha in grid
                },
                "full12_alpha_histogram": histogram(original_alpha, grid),
                "all_lomo_alpha_histogram": histogram(all_lomo, grid),
                "full12_v2_loss_grid_max_abs_diff_vs_strengths": loss_grid_max_abs_diff,
                "full12_recomputed_alpha_matches_strengths": True,
                "full12_recomputed_global_alpha_matches_policy": True,
                "series_any_changed_fraction": float(np.mean(changed_counts > 0)),
                "series_all_unchanged_fraction": float(np.mean(changed_counts == 0)),
                "series_any_endpoint_flip_fraction": float(np.mean(endpoint_flip_counts > 0)),
                "series_mean_changed_month_fraction": float(changed_fraction_by_series.mean()),
                "series_median_changed_month_fraction": float(np.median(changed_fraction_by_series)),
                "all_series_month_changed_fraction": float(changed_counts.sum() / (len(series_ids) * 12)),
                "all_series_month_mean_abs_delta_alpha": float(mean_abs_delta.mean()),
                "series_median_mean_abs_delta_alpha": float(np.median(mean_abs_delta)),
                "all_series_month_endpoint_flip_fraction": float(endpoint_flip_counts.sum() / (len(series_ids) * 12)),
                "max_abs_delta_alpha": float(max_abs_delta.max()),
                "month_summaries": month_summaries,
            }
        )

    csv_path = OUT / "month_sensitivity.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "completed": True,
        "diagnostic": "V2 finite 12-month leave-one-month-out alpha selection sensitivity",
        "scope": "Uses saved V2 gate forecasts and 2004 target months only; no E read, no new model, no new alpha rule.",
        "not_claims": [
            "not an uncertainty interval",
            "not an independent-year replication",
            "not a new shrinkage or selection method",
            "not allowed to pick a LOMO rule using E",
        ],
        "target": {
            "panel": "runs/hospital_shared_strength_v1_prepared/gate.npz",
            "origin_zero_based": 48,
            "target_slice_zero_based": [48, 60],
            "calendar_year": 2004,
            "calendar_months_1based": list(range(1, 13)),
        },
        "selection_rule": {
            "grid": [float(x) for x in np.linspace(0, 1, 11)],
            "global_tie_preferred_alpha": 0.0,
            "individual_tie_preferred_alpha": "selected global alpha, matching hospital_pilot.gates.select_strengths",
            "tie_tolerance": 1e-12,
        },
        "verification": {
            "series_ids_match": True,
            "scale_match": True,
            "quantiles_match": True,
            "full12_loss_grid_matches_strengths_for_all_seeds": True,
            "full12_alpha_matches_strengths_for_all_seeds": True,
            "policy_histogram_matches_strengths_for_all_seeds": True,
        },
        "seed_summaries": seed_summaries,
        "series_csv": "results/peft_mechanism_diagnostics_v1/hospital_strength/month_sensitivity.csv",
        "figure": "results/peft_mechanism_diagnostics_v1/hospital_strength/month_sensitivity.png",
        "source_hashes": source_hashes,
    }
    with (OUT / "month_sensitivity.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2, allow_nan=False)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), layout="constrained")
    months = np.arange(1, 13)
    for idx, seed_summary in enumerate(seed_summaries):
        seed = seed_summary["seed"]
        color = COLORS["blue"] if idx == 0 else COLORS["orange"]
        month_summary = seed_summary["month_summaries"]
        axes[0, 0].plot(
            months,
            [m["changed_fraction"] * 100 for m in month_summary],
            marker="o",
            color=color,
            label=f"seed {seed}",
        )
        axes[0, 1].plot(
            months,
            [m["mean_abs_delta_alpha"] for m in month_summary],
            marker="o",
            color=color,
            label=f"seed {seed}",
        )
        axes[1, 0].plot(
            months,
            [m["endpoint_flip_fraction"] * 100 for m in month_summary],
            marker="o",
            color=color,
            label=f"seed {seed}",
        )

    axes[0, 0].set_title("Changed alpha after leaving out one V2 month")
    axes[0, 0].set_xlabel("Left-out 2004 month")
    axes[0, 0].set_ylabel("Series changed (%)")
    axes[0, 0].set_xticks(months)
    axes[0, 0].legend(frameon=False)

    axes[0, 1].set_title("Mean |delta alpha| by left-out month")
    axes[0, 1].set_xlabel("Left-out 2004 month")
    axes[0, 1].set_ylabel("Mean |delta alpha|")
    axes[0, 1].set_xticks(months)
    axes[0, 1].legend(frameon=False)

    axes[1, 0].set_title("Endpoint flips: 0 <-> 1")
    axes[1, 0].set_xlabel("Left-out 2004 month")
    axes[1, 0].set_ylabel("Series endpoint-flipped (%)")
    axes[1, 0].set_xticks(months)
    axes[1, 0].legend(frameon=False)

    ax = axes[1, 1]
    width = 0.038
    grid = np.linspace(0, 1, 11)
    for idx, seed_summary in enumerate(seed_summaries):
        seed = seed_summary["seed"]
        color = COLORS["blue"] if idx == 0 else COLORS["orange"]
        full_hist = np.array([seed_summary["full12_alpha_histogram"][f"{a:.1f}"] for a in grid], dtype=float)
        lomo_hist = np.array([seed_summary["all_lomo_alpha_histogram"][f"{a:.1f}"] for a in grid], dtype=float) / 12.0
        shift = (-0.5 if idx == 0 else 0.5) * width
        ax.bar(grid + shift, full_hist, width=width, color=color, alpha=0.28, label=f"full seed {seed}")
        ax.plot(grid, lomo_hist, color=color, marker="s", linewidth=1.3, label=f"LOMO avg seed {seed}")
    ax.set_title("Full-12 alpha histogram vs average LOMO histogram")
    ax.set_xlabel("alpha")
    ax.set_ylabel("Series count")
    ax.set_xticks(grid)
    ax.legend(frameon=False, fontsize=8, ncols=2)

    fig.suptitle("Hospital V2 month sensitivity only: saved 2004 gate forecasts, no E", fontsize=13)
    fig.savefig(OUT / "month_sensitivity.png", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(
        json.dumps(
            {
                "completed": True,
                "json": str((OUT / "month_sensitivity.json").relative_to(ROOT)),
                "csv": str(csv_path.relative_to(ROOT)),
                "png": str((OUT / "month_sensitivity.png").relative_to(ROOT)),
                "seed_summaries": [
                    {
                        "seed": s["seed"],
                        "changed_fraction": s["all_series_month_changed_fraction"],
                        "mean_abs_delta": s["all_series_month_mean_abs_delta_alpha"],
                        "endpoint_flip_fraction": s["all_series_month_endpoint_flip_fraction"],
                        "series_any_changed_fraction": s["series_any_changed_fraction"],
                    }
                    for s in seed_summaries
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
