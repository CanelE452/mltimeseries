"""Follow-up diagnostics for the PEFT phenomenon-to-mechanism roadmap.

This script reads completed experiment artifacts only. It does not train,
select new hyperparameters, or modify the original runs.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "peft_mechanism_diagnostics_v1"
DOC = OUT / "saved_artifact_snapshot.md"
REFERENCE = OUT / "reference" / "summary.json"
MODULE = ROOT / "results" / "peft_module_ablation_v1"
HOSPITAL_RUN = ROOT / "runs" / "hospital_shared_strength_v1_run2"
HOSPITAL_PREP = ROOT / "runs" / "hospital_shared_strength_v1_prepared"

OKABE = {
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "gray": "#737373",
    "black": "#000000",
}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def corr(a, b) -> float | None:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    keep = np.isfinite(a) & np.isfinite(b)
    if keep.sum() < 3:
        return None
    aa, bb = a[keep], b[keep]
    if np.std(aa) == 0 or np.std(bb) == 0:
        return None
    return float(np.corrcoef(aa, bb)[0, 1])


def rankdata(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and x[order[j]] == x[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2.0 + 1.0
        i = j
    return ranks


def spearman(a, b) -> float | None:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    keep = np.isfinite(a) & np.isfinite(b)
    if keep.sum() < 3:
        return None
    return corr(rankdata(a[keep]), rankdata(b[keep]))


def tied_argmin(row: np.ndarray, grid: np.ndarray, preferred: float, tol: float = 1e-12) -> int:
    row = np.asarray(row, dtype=float)
    ties = np.flatnonzero(row <= np.min(row) + tol)
    return int(min(ties, key=lambda j: (round(abs(float(grid[j]) - preferred), 14), float(grid[j]))))


def combine(f0: np.ndarray, lora: np.ndarray, alpha) -> np.ndarray:
    a = np.asarray(alpha, dtype=float)
    if a.ndim == 0:
        a = np.full(f0.shape[0], float(a))
    if a.shape != (f0.shape[0],):
        raise ValueError("Expected one alpha per series")
    return f0 + a[:, None, None, None] * (lora - f0)


def per_series_window(pred: np.ndarray, target: np.ndarray, scale: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
    pred = np.asarray(pred, dtype=float)
    target = np.asarray(target, dtype=float)
    scale = np.asarray(scale, dtype=float)
    q = np.asarray(quantiles, dtype=float)
    err = target[:, :, None, :] - pred
    loss = 2.0 * np.maximum(q[None, None, :, None] * err, (q[None, None, :, None] - 1.0) * err)
    return (loss / scale[:, None, None, None]).mean(axis=(2, 3))


def pct_improve(reference: np.ndarray, candidate: np.ndarray, denom: np.ndarray) -> float:
    return float(100.0 * (np.mean(reference) - np.mean(candidate)) / np.mean(denom))


def pct_harm(reference: np.ndarray, candidate: np.ndarray, denom: np.ndarray) -> float:
    return float(100.0 * (np.mean(candidate) - np.mean(reference)) / np.mean(denom))


def load_reference_summary() -> dict:
    if not REFERENCE.exists():
        raise FileNotFoundError(f"Missing R1 reference diagnostic: {REFERENCE}")
    return read_json(REFERENCE)


def build_layer_scope() -> dict:
    out = OUT / "layer_scope"
    out.mkdir(parents=True, exist_ok=True)
    effects = read_json(MODULE / "effects.json")
    costs = read_json(MODULE / "costs.json")
    verification = read_json(MODULE / "verification.json")

    rows = []
    with (MODULE / "selected_results.csv").open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            row["score"] = float(row["score"])
            row["lr"] = float(row["lr"])
            row["corpus"] = int(row["corpus"])
            row["trainable"] = int(row["trainable"])
            row["wall"] = float(row["wall"])
            row["best_step"] = int(row["best_step"])
            rows.append(row)

    primary_lr = float(effects["primary"]["lr_by_method"]["ATTN_ONLY"])
    primary = [
        r
        for r in rows
        if r["method"] == "F0" or math.isclose(r["lr"], primary_lr, rel_tol=0.0, abs_tol=1e-12)
    ]
    by_method = {}
    for method in ("F0", "BOTH", "OUT_ONLY", "ATTN_ONLY"):
        selected = [r for r in primary if r["method"] == method]
        if not selected:
            raise AssertionError(f"No selected rows for {method} at primary lr {primary_lr}")
        by_method[method] = {
            "mean_score": float(np.mean([r["score"] for r in selected])),
            "scores": [r["score"] for r in selected],
            "trainable": int(selected[0]["trainable"]),
            "mean_wall_seconds": float(np.mean([r["wall"] for r in selected])),
            "best_steps": [r["best_step"] for r in selected],
        }

    both_params = by_method["BOTH"]["trainable"]
    attn_params = by_method["ATTN_ONLY"]["trainable"]
    out_params = by_method["OUT_ONLY"]["trainable"]
    f0_score = by_method["F0"]["mean_score"]
    summary = {
        "completed": True,
        "source": "results/peft_module_ablation_v1",
        "new_training": 0,
        "primary_lr": primary_lr,
        "verification_passed": bool(verification["passed"]),
        "method_summary": by_method,
        "attention_deletion_penalty_pctF0": float(effects["primary"]["attention_deletion_penalty"]["value"] * 100.0),
        "output_deletion_penalty_pctF0": float(effects["primary"]["output_deletion_penalty"]["value"] * 100.0),
        "attention_deletion_decision": effects["primary"]["attention_deletion_penalty"]["decision"],
        "output_deletion_decision": effects["primary"]["output_deletion_penalty"]["decision"],
        "attn_only_saves_params_pct_of_both": float(100.0 * (both_params - attn_params) / both_params),
        "out_only_params_pct_of_both": float(100.0 * out_params / both_params),
        "attn_only_improvement_pctF0": float(100.0 * (f0_score - by_method["ATTN_ONLY"]["mean_score"]) / f0_score),
        "both_improvement_pctF0": float(100.0 * (f0_score - by_method["BOTH"]["mean_score"]) / f0_score),
        "out_only_improvement_pctF0": float(100.0 * (f0_score - by_method["OUT_ONLY"]["mean_score"]) / f0_score),
        "new_run_resource_summary": costs,
        "limits": [
            "Q00 synthetic/development setting only",
            "module deletion, not a layer-wise placement rule",
            "parameter count reduction does not imply end-to-end speed reduction",
            "OUT_ONLY has far lower capacity than BOTH; capacity and location are not fully separated",
        ],
        "source_hashes": {
            "effects.json": sha256(MODULE / "effects.json"),
            "costs.json": sha256(MODULE / "costs.json"),
            "verification.json": sha256(MODULE / "verification.json"),
            "selected_results.csv": sha256(MODULE / "selected_results.csv"),
        },
    }

    with (out / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, allow_nan=False)

    plt.rcParams.update(
        {
            "font.family": "Malgun Gothic",
            "font.size": 10,
            "axes.unicode_minus": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
        }
    )
    methods = ["F0", "OUT_ONLY", "ATTN_ONLY", "BOTH"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    ax = axes[0]
    scores = [by_method[m]["mean_score"] for m in methods]
    colors = [OKABE["gray"], OKABE["orange"], OKABE["blue"], OKABE["green"]]
    ax.bar(np.arange(len(methods)), scores, color=colors, alpha=0.82)
    for i, method in enumerate(methods):
        vals = by_method[method]["scores"]
        ax.scatter(np.full(len(vals), i) + np.linspace(-0.08, 0.08, len(vals)), vals, color="black", s=18, zorder=3)
        ax.text(i, scores[i] + 0.004, f"{scores[i]:.4f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(np.arange(len(methods)), methods)
    ax.set_ylabel("Raw mean 2-pinball loss")
    ax.set_title("R2: attention path carries the Q00 gain")
    ax.set_ylim(min(scores) * 0.97, max(scores) * 1.04)

    ax = axes[1]
    params = [by_method[m]["trainable"] for m in methods]
    gains = [
        0.0,
        summary["out_only_improvement_pctF0"],
        summary["attn_only_improvement_pctF0"],
        summary["both_improvement_pctF0"],
    ]
    sizes = [40 + 120 * (p / both_params if both_params else 0) for p in params]
    ax.scatter(params, gains, s=sizes, c=colors, edgecolors="black", linewidths=0.7)
    label_offsets = {
        "F0": (8, 8),
        "OUT_ONLY": (8, 20),
        "ATTN_ONLY": (9, 10),
        "BOTH": (9, -15),
    }
    for x, y, m in zip(params, gains, methods):
        ax.annotate(m, (x, y), xytext=label_offsets[m], textcoords="offset points")
    ax.axhline(0, color=OKABE["gray"], linestyle="--", linewidth=1)
    ax.set_xlabel("Trainable parameters")
    ax.set_ylabel("F0 대비 개선율 (%)")
    ax.set_title("R2: 성능 단서는 있지만 비용 이득은 작음")
    fig.suptitle("Module-scope diagnostic from completed Q00 ablation; no new training", fontsize=12)
    for ext in ("png", "svg"):
        fig.savefig(out / f"scope_cost_diagnostic.{ext}", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return summary


def build_hospital_strength() -> dict:
    out = OUT / "hospital_strength"
    out.mkdir(parents=True, exist_ok=True)
    policy = read_json(HOSPITAL_RUN / "stages" / "select_gates" / "attempt_01" / "output" / "policies.json")
    hospital_summary = read_json(HOSPITAL_RUN / "stages" / "analyse" / "attempt_01" / "output" / "summary.json")
    published = read_json(ROOT / "results" / "hospital_shared_strength_v1" / "publication_verification.json")

    with np.load(HOSPITAL_PREP / "eval.npz", allow_pickle=False) as panel:
        values = panel["values"].astype(float)
        panel_ids = panel["series_ids"]
    target = np.stack([values[:, 60:72], values[:, 72:84]], axis=1)

    with np.load(HOSPITAL_RUN / "stages" / "forecast_eval" / "attempt_01" / "output" / "f0.npz", allow_pickle=False) as f:
        f0_pred = f["prediction"].astype(float)
        q = f["quantiles"].astype(float)
        scale = f["scale"].astype(float)
        series_ids = f["series_ids"]
    if not np.array_equal(series_ids, panel_ids):
        raise AssertionError("Prepared panel and forecast series order differ")
    f0_eval = per_series_window(f0_pred, target, scale, q)

    losses_archive = np.load(HOSPITAL_RUN / "stages" / "analyse" / "attempt_01" / "output" / "losses.npz", allow_pickle=False)
    source_hashes = {
        "policies.json": sha256(HOSPITAL_RUN / "stages" / "select_gates" / "attempt_01" / "output" / "policies.json"),
        "losses.npz": sha256(HOSPITAL_RUN / "stages" / "analyse" / "attempt_01" / "output" / "losses.npz"),
        "eval.npz": sha256(HOSPITAL_PREP / "eval.npz"),
        "published_publication_verification.json": sha256(ROOT / "results" / "hospital_shared_strength_v1" / "publication_verification.json"),
    }

    grid_summaries = []
    rows = []
    for p in policy["policies"]:
        seed = int(p["seed"])
        strength_path = Path(p["strengths_path"])
        source_hashes[strength_path.relative_to(ROOT).as_posix()] = sha256(strength_path)
        with np.load(strength_path, allow_pickle=False) as s:
            grid = s["grid"].astype(float)
            alpha = s["alpha"].astype(float)
            global_alpha = float(s["global_alpha"])
            v2_loss_grid = s["v2_loss_grid"].astype(float)
            strength_ids = s["series_ids"]
        if not np.array_equal(strength_ids, series_ids):
            raise AssertionError("Strength and forecast series order differ")
        global_idx = int(np.flatnonzero(np.isclose(grid, global_alpha, atol=1e-12))[0])
        chosen_idx = np.array([int(np.flatnonzero(np.isclose(grid, a, atol=1e-12))[0]) for a in alpha], dtype=int)

        with np.load(HOSPITAL_RUN / "stages" / "forecast_eval" / "attempt_01" / "output" / f"lora_{seed}.npz", allow_pickle=False) as l:
            lora_pred = l["prediction"].astype(float)
            lora_ids = l["series_ids"]
        if not np.array_equal(lora_ids, series_ids):
            raise AssertionError("Lora forecast series order differs")

        eval_grids = []
        for a in grid:
            eval_grids.append(per_series_window(combine(f0_pred, lora_pred, float(a)), target, scale, q))
        eval_loss_grid = np.stack(eval_grids, axis=2)  # series, eval window, alpha
        eval_mean_grid = eval_loss_grid.mean(axis=1)
        eval_global = eval_mean_grid[:, global_idx]
        eval_individual = eval_mean_grid[np.arange(len(series_ids)), chosen_idx]
        eval_oracle_idx = np.array([tied_argmin(row, grid, global_alpha) for row in eval_mean_grid], dtype=int)
        eval_oracle = eval_mean_grid[np.arange(len(series_ids)), eval_oracle_idx]
        v2_global = v2_loss_grid[:, global_idx]
        v2_best = v2_loss_grid[np.arange(len(series_ids)), chosen_idx]
        v2_second = []
        for row in v2_loss_grid:
            ordered = np.sort(row)
            v2_second.append(float(ordered[1] - ordered[0]))
        v2_second = np.asarray(v2_second)
        v2_direction = v2_loss_grid[:, 0] - v2_loss_grid[:, -1]
        eval_direction = eval_mean_grid[:, 0] - eval_mean_grid[:, -1]
        chosen_eval_rank = []
        for i, row in enumerate(eval_mean_grid):
            chosen_eval_rank.append(int(1 + np.sum(row < row[chosen_idx[i]] - 1e-12)))
        chosen_eval_rank = np.asarray(chosen_eval_rank)

        e_window_effects = []
        for w, origin in enumerate((60, 72)):
            window_grid = eval_loss_grid[:, w, :]
            window_oracle_idx = np.array([tied_argmin(row, grid, global_alpha) for row in window_grid], dtype=int)
            e_window_effects.append(
                {
                    "origin": int(origin),
                    "year": int(2000 + origin // 12),
                    "individual_vs_global_pctF0": pct_improve(eval_loss_grid[:, w, global_idx], eval_loss_grid[np.arange(len(series_ids)), w, chosen_idx], f0_eval[:, w]),
                    "oracle_vs_global_pctF0_posthoc": pct_improve(eval_loss_grid[:, w, global_idx], window_grid[np.arange(len(series_ids)), window_oracle_idx], f0_eval[:, w]),
                    "exact_alpha_match_fraction_posthoc": float(np.mean(alpha == grid[window_oracle_idx])),
                }
            )

        seed_summary = {
            "seed": seed,
            "global_alpha": global_alpha,
            "v2_f0_mean": float(np.mean(v2_loss_grid[:, 0])),
            "v2_global_mean": float(np.mean(v2_global)),
            "v2_individual_best_mean": float(np.mean(v2_best)),
            "v2_individual_over_global_pctF0_optimistic": pct_improve(v2_global, v2_best, v2_loss_grid[:, 0]),
            "v2_median_second_best_gap_pctF0": float(100.0 * np.median(v2_second) / np.mean(v2_loss_grid[:, 0])),
            "v2_series_with_gap_lt_0p1pctF0": float(np.mean((v2_second / np.mean(v2_loss_grid[:, 0]) * 100.0) < 0.1)),
            "eval_individual_vs_global_pctF0": pct_improve(eval_global, eval_individual, f0_eval),
            "eval_oracle_vs_global_pctF0_posthoc": pct_improve(eval_global, eval_oracle, f0_eval),
            "eval_chosen_rank_mean_of_11": float(np.mean(chosen_eval_rank)),
            "eval_chosen_rank_top1_fraction": float(np.mean(chosen_eval_rank == 1)),
            "v2_alpha_vs_eval_oracle_alpha_spearman": spearman(alpha, grid[eval_oracle_idx]),
            "v2_lora_direction_vs_eval_direction_spearman": spearman(v2_direction, eval_direction),
            "v2_lora_direction_sign_agreement_with_eval": float(np.mean(np.sign(v2_direction) == np.sign(eval_direction))),
            "endpoint_flip_fraction_0_to_1_or_1_to_0": float(
                np.mean(((alpha == 0.0) & (grid[eval_oracle_idx] == 1.0)) | ((alpha == 1.0) & (grid[eval_oracle_idx] == 0.0)))
            ),
            "future_oracle_alpha_histogram_posthoc": {f"{a:.1f}": int(np.sum(grid[eval_oracle_idx] == a)) for a in grid},
            "v2_alpha_histogram": {f"{a:.1f}": int(np.sum(alpha == a)) for a in grid},
            "eval_window_effects": e_window_effects,
        }
        grid_summaries.append(seed_summary)

        for i, sid in enumerate(series_ids):
            rows.append(
                {
                    "series_id": str(sid),
                    "seed": seed,
                    "alpha_v2": float(alpha[i]),
                    "alpha_eval_oracle_posthoc": float(grid[eval_oracle_idx[i]]),
                    "v2_gap_global_minus_chosen": float(v2_global[i] - v2_best[i]),
                    "v2_second_best_gap": float(v2_second[i]),
                    "eval_regret_individual_minus_global": float(eval_individual[i] - eval_global[i]),
                    "eval_oracle_gain_global_minus_oracle": float(eval_global[i] - eval_oracle[i]),
                    "eval_chosen_rank_of_11": int(chosen_eval_rank[i]),
                    "v2_f0_minus_lora_loss": float(v2_direction[i]),
                    "eval_f0_minus_lora_loss": float(eval_direction[i]),
                }
            )

    csv_path = out / "hospital_alpha_series.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    aggregate_individual_vs_global = float(np.mean([x["eval_individual_vs_global_pctF0"] for x in grid_summaries]))
    aggregate_oracle_vs_global = float(np.mean([x["eval_oracle_vs_global_pctF0_posthoc"] for x in grid_summaries]))
    summary = {
        "completed": True,
        "source": "runs/hospital_shared_strength_v1_run2",
        "new_training": 0,
        "published_main_effect_pctF0_individual_over_global": hospital_summary["main_effect_pctF0"],
        "published_matching_effect_pctF0_individual_over_shuffled": hospital_summary["matching_effect_pctF0"],
        "publication_verification_completed": bool(published["completed"]),
        "seed_summaries": grid_summaries,
        "aggregate_eval_individual_vs_global_pctF0": aggregate_individual_vs_global,
        "aggregate_eval_oracle_vs_global_pctF0_posthoc": aggregate_oracle_vs_global,
        "interpretation": {
            "v2_selection_optimism": "Individual alpha improves V2 by construction but fails to transfer to E.",
            "future_oracle_status": "E oracle alpha is posthoc diagnostic, not a valid method result.",
            "likely_primary_cause": "selection noise and temporal instability dominate this direct per-series alpha rule",
            "method_implication": "No automatic method test: first establish historically predictable heterogeneity; oracle gain alone does not identify noise, shift, or stable heterogeneity.",
        },
        "limits": [
            "E1/E2 were already inspected; all E oracle analyses are descriptive only",
            "767 series are related series in one panel, not 767 independent hospitals",
            "two training seeds are not enough to decompose optimizer noise from series heterogeneity",
            "no new model training or new test split is created here",
        ],
        "source_hashes": source_hashes,
    }
    with (out / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, allow_nan=False)

    plt.rcParams.update(
        {
            "font.family": "Malgun Gothic",
            "font.size": 10,
            "axes.unicode_minus": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    grid = np.linspace(0, 1, 11)
    ax = axes[0, 0]
    for p in policy["policies"]:
        seed = int(p["seed"])
        ax.plot(grid, p["global_validation_curve"], marker="o", label=f"seed {seed}")
    ax.set_xlabel("alpha")
    ax.set_ylabel("V2 mean loss")
    ax.set_title("GLOBAL은 두 seed 모두 alpha=1")
    ax.legend(frameon=False)

    ax = axes[0, 1]
    width = 0.038
    for j, seed_summary in enumerate(grid_summaries):
        vals = [seed_summary["v2_alpha_histogram"][f"{a:.1f}"] for a in grid]
        oracle = [seed_summary["future_oracle_alpha_histogram_posthoc"][f"{a:.1f}"] for a in grid]
        shift = (-0.5 if j == 0 else 0.5) * width
        ax.bar(grid + shift, vals, width=width, color=OKABE["blue"], alpha=0.35 if j == 0 else 0.65, label=f"V2 chosen {seed_summary['seed']}")
        ax.plot(grid, oracle, color=OKABE["vermillion"], alpha=0.55 if j == 0 else 0.9, marker="s", linewidth=1, label=f"E oracle {seed_summary['seed']}")
    ax.set_xlabel("alpha")
    ax.set_ylabel("series count")
    ax.set_title("V2 선택 분포와 E 사후 oracle 분포")
    ax.legend(frameon=False, fontsize=8, ncols=2)

    ax = axes[1, 0]
    for seed_summary in grid_summaries:
        seed_rows = [r for r in rows if r["seed"] == seed_summary["seed"]]
        ax.scatter(
            [100.0 * r["v2_gap_global_minus_chosen"] / seed_summary["v2_f0_mean"] for r in seed_rows],
            [100.0 * r["eval_regret_individual_minus_global"] / float(np.mean(f0_eval)) for r in seed_rows],
            s=10,
            alpha=0.35,
            label=f"seed {seed_summary['seed']}",
        )
    ax.axhline(0, color=OKABE["gray"], linestyle="--", linewidth=1)
    ax.axvline(0, color=OKABE["gray"], linestyle="--", linewidth=1)
    ax.set_xlabel("V2에서 individual이 global보다 좋아 보인 정도 (% V2 F0)")
    ax.set_ylabel("E에서 individual 손해 (+) / 이득 (-), % E F0")
    ax.set_title("V2 local 이득은 E 손익을 잘 예측하지 못함")
    ax.legend(frameon=False)

    ax = axes[1, 1]
    x = np.arange(len(grid_summaries))
    v2_gain = [s["v2_individual_over_global_pctF0_optimistic"] for s in grid_summaries]
    e_gain = [s["eval_individual_vs_global_pctF0"] for s in grid_summaries]
    e_oracle = [s["eval_oracle_vs_global_pctF0_posthoc"] for s in grid_summaries]
    ax.bar(x - 0.22, v2_gain, width=0.2, color=OKABE["sky"], label="V2 optimistic")
    ax.bar(x, e_gain, width=0.2, color=OKABE["blue"], label="E transfer")
    ax.bar(x + 0.22, e_oracle, width=0.2, color=OKABE["vermillion"], label="E oracle posthoc")
    ax.axhline(0, color=OKABE["gray"], linestyle="--", linewidth=1)
    ax.set_xticks(x, [f"seed {s['seed']}" for s in grid_summaries])
    ax.set_ylabel("GLOBAL 대비 개선율 (% F0)")
    ax.set_title("선택하면 V2는 좋아지지만 E로 전달되지 않음")
    ax.legend(frameon=False)

    fig.suptitle("R3 Hospital alpha-strength diagnostic; descriptive reuse of completed artifacts", fontsize=13)
    for ext in ("png", "svg"):
        fig.savefig(out / f"hospital_alpha_diagnostic.{ext}", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    losses_archive.close()
    return summary


def build_overall_report(reference: dict, layer: dict, hospital: dict) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {
        "completed": True,
        "created_at_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        "domain_context": {
            "domain": "time-series foundation model adaptation",
            "downstream_decision": "choose a thesis-worthy PEFT mechanism rather than tune another isolated result",
            "quality_bar": "exploratory mechanism diagnosis; not independent publication confirmation",
            "leakage_constraints": [
                "past-only features for R1",
                "V2-only alpha selection for R3; E oracle only descriptive",
                "do not count overlapping windows or related series as independent domains",
            ],
        },
        "r1": {
            "bike_head_lora_gain_pctF0_by_seed": reference["datasets"]["bike"]["gain_head_lora_pct_f0_by_seed"],
            "household_head_lora_gain_pctF0_by_seed": reference["datasets"]["household"]["gain_head_lora_pct_f0_by_seed"],
            "bike_positive_origin_fraction": reference["datasets"]["bike"]["positive_origin_fraction"],
            "household_positive_origin_fraction": reference["datasets"]["household"]["positive_origin_fraction"],
            "bike_feature_mean": reference["datasets"]["bike"]["feature_mean"],
            "household_feature_mean": reference["datasets"]["household"]["feature_mean"],
            "bike_feature_gain_corr": reference["datasets"]["bike"]["descriptive_within_dataset_correlations"],
            "household_feature_gain_corr": reference["datasets"]["household"]["descriptive_within_dataset_correlations"],
            "interpretation": "Bike is more structured and shows repeated LoRA-over-head gain, but within-dataset raw complexity correlations are weak; matched head/probe controls remain necessary.",
        },
        "r2": {
            "attn_only_improvement_pctF0": layer["attn_only_improvement_pctF0"],
            "both_improvement_pctF0": layer["both_improvement_pctF0"],
            "attn_only_saves_params_pct_of_both": layer["attn_only_saves_params_pct_of_both"],
            "attention_deletion_penalty_pctF0": layer["attention_deletion_penalty_pctF0"],
            "output_deletion_penalty_pctF0": layer["output_deletion_penalty_pctF0"],
            "interpretation": "Attention module scope is promising as a mechanism clue, but not yet an efficient layer/rank selection method.",
        },
        "r3": {
            "published_individual_over_global_pctF0": hospital["published_main_effect_pctF0_individual_over_global"],
            "aggregate_eval_individual_vs_global_pctF0": hospital["aggregate_eval_individual_vs_global_pctF0"],
            "aggregate_eval_oracle_vs_global_pctF0_posthoc": hospital["aggregate_eval_oracle_vs_global_pctF0_posthoc"],
            "seed_summaries": hospital["seed_summaries"],
            "interpretation": "Positive selected-V2 gain and negative E transfer; sampling noise, temporal shift, and absent stable heterogeneity remain unresolved.",
        },
        "next_decision": [
            "Run a matched frozen-MLP vs LoRA+same-MLP control for Bike/Household before claiming backbone necessity.",
            "If R1 survives, test pre-declared attention layer groups with random/fixed budget-matched controls.",
            "For Hospital, require longer historical replication of predictable heterogeneity before any shrinkage or group adapter experiment.",
        ],
    }
    with (OUT / "followup_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, allow_nan=False)

    bike = reference["datasets"]["bike"]
    household = reference["datasets"]["household"]
    hseeds = hospital["seed_summaries"]
    doc = f"""# PEFT 현상에서 원인으로: R1-R3 진단 결과

2026-09-10. 이 문서는 사용자가 요청한 `1, 2, 3` 순차 진단을 저장된 실험 산출물로 실행한 결과다. 새 학습은 하지 않았고, 기존 run과 checkpoint는 읽기만 했다. 품질 기준은 논문 확증이 아니라 **탐색적 원인 진단**이다.

## 도메인과 누수 기준

도메인은 시계열 foundation model의 target-domain PEFT다. 다운스트림 결정은 "어떤 adapter를 만들까"가 아니라 **논문 주제로 밀 수 있는 부족 원인을 고를지**다. 그래서 이번 산출물은 세 가지 누수를 분리한다.

- R1: 과거 context에서 계산한 특성만 사용했지만, 이미 본 E 구간과 gain의 상관은 탐색적이다.
- R2: Q00 합성 개발 결과라 layer 일반화 주장이 아니다.
- R3: 2005/2006 E oracle은 사후 진단이며, 새 선택 규칙의 test가 아니다.

```mermaid
flowchart LR
  A[현상: LoRA 이득 차이] --> B[R1: 어떤 데이터 구조에서 내부 적응 이득이 반복되나]
  B -->|head 대조 후에도 남음| C[R2: 어느 적응 위치가 필요한가]
  B -->|head로 설명됨| D[새 PEFT 방법 중단]
  H[현상: Hospital individual alpha 실패] --> I[R3: 선택 잡음 / 시간 전이 / 안정 이질성]
  I -->|안정 이질성 확인| J[수축 또는 group 강도]
  I -->|전달 안 됨| K[개별 선택 분기 닫기]
```

## R1. Bike와 Household 차이

[확인] 저장된 Study20 예측을 seed, origin, horizon, 과거 context 특성으로 다시 분해했다. Bike의 head 대비 LoRA 추가 이득은 seed별 `{bike['gain_head_lora_pct_f0_by_seed'][0]:.3f}`, `{bike['gain_head_lora_pct_f0_by_seed'][1]:.3f}`, `{bike['gain_head_lora_pct_f0_by_seed'][2]:.3f}`%F0이고, 80개 evaluation origin 중 양수 비율은 `{100*bike['positive_origin_fraction']:.1f}%`다. Household는 `{household['gain_head_lora_pct_f0_by_seed'][0]:.3f}`, `{household['gain_head_lora_pct_f0_by_seed'][1]:.3f}`, `{household['gain_head_lora_pct_f0_by_seed'][2]:.3f}`%F0이고 양수 비율은 `{100*household['positive_origin_fraction']:.1f}%`다.

[확인] Bike는 평균 spectral entropy `{bike['feature_mean']['spectral_entropy']:.3f}`, 24시간 ACF `{bike['feature_mean']['acf24']:.3f}`, 168시간 ACF `{bike['feature_mean']['acf168']:.3f}`로 반복 구조가 강했다. Household는 entropy `{household['feature_mean']['spectral_entropy']:.3f}`, 24시간 ACF `{household['feature_mean']['acf24']:.3f}`, 168시간 ACF `{household['feature_mean']['acf168']:.3f}`로 더 불규칙했다.

하지만 [확인] 각 데이터셋 내부에서 과거 spectral entropy와 origin별 LoRA 이득의 상관은 Bike `{bike['descriptive_within_dataset_correlations']['spectral_entropy']:.3f}`, Household `{household['descriptive_within_dataset_correlations']['spectral_entropy']:.3f}`로 약했다. 즉 "복잡도 하나가 크면 LoRA가 된다"는 단순 문장은 지금 증거와 맞지 않는다. 더 그럴듯한 문장은 **recoverable temporal structure가 남는 데이터에서 내부 적응 이득이 반복될 수 있다**다.

![R1 저장 결과 진단](../../../../results/peft_mechanism_diagnostics_v1/reference/reference_diagnostic.png)

다음 R1 실험은 matched head control이어야 한다. Frozen+MLP와 LoRA+같은 MLP를 같은 출력 형식, 같은 선택 예산으로 비교해야 head 용량/최적화 설명을 걷어낼 수 있다.

## R2. 일부 모듈만으로 충분한가

[확인] Q00 module deletion 결과에서 ATTN_ONLY는 F0 대비 `{layer['attn_only_improvement_pctF0']:.3f}%` 개선했고 BOTH는 `{layer['both_improvement_pctF0']:.3f}%` 개선했다. Attention 삭제 비용은 `{layer['attention_deletion_penalty_pctF0']:.3f}%F0`, output projection 삭제 비용은 `{layer['output_deletion_penalty_pctF0']:.3f}%F0`다.

해석은 꽤 차갑다. attention 경로가 중요하다는 단서는 살아 있다. 그런데 ATTN_ONLY가 BOTH 대비 줄인 학습 파라미터는 `{layer['attn_only_saves_params_pct_of_both']:.3f}%`뿐이다. 그래서 지금 단계에서 "효율적 layer PEFT"라고 부르면 약하다. R2는 **위치/범위 선택 문제**로 남겨야 한다.

![R2 module scope 진단](../../../../results/peft_mechanism_diagnostics_v1/layer_scope/scope_cost_diagnostic.png)

다음 R2 실험은 R1이 통과한 뒤에만 의미가 있다. 앞/중간/뒤 attention layer group을 미리 고정하고, random 위치와 같은 총 파라미터 예산 대조를 같이 둬야 한다.

## R3. Hospital individual alpha는 왜 실패했나

[확인] 완료 결과의 주효과는 INDIVIDUAL이 GLOBAL보다 `{hospital['published_main_effect_pctF0_individual_over_global']:.3f}%F0` 나빴다는 것이다. 이번 추가 진단에서 V2에서는 individual 선택이 seed별로 global보다 `{hseeds[0]['v2_individual_over_global_pctF0_optimistic']:.3f}`, `{hseeds[1]['v2_individual_over_global_pctF0_optimistic']:.3f}%F0` 좋아 보였다. 그런데 E에서는 각각 `{hseeds[0]['eval_individual_vs_global_pctF0']:.3f}`, `{hseeds[1]['eval_individual_vs_global_pctF0']:.3f}%F0`로 손해였다.

[확인] E를 사후 oracle로 고르면 GLOBAL 대비 `{hospital['aggregate_eval_oracle_vs_global_pctF0_posthoc']:.3f}%F0`의 여지는 있다. 그러나 이것은 test를 보고 고른 상한이다. 방법 결과가 아니다. V2 alpha와 E oracle alpha의 Spearman 상관은 seed별 `{hseeds[0]['v2_alpha_vs_eval_oracle_alpha_spearman']:.3f}`, `{hseeds[1]['v2_alpha_vs_eval_oracle_alpha_spearman']:.3f}`이고, 선택 alpha가 E oracle top1인 비율은 `{100*hseeds[0]['eval_chosen_rank_top1_fraction']:.1f}%`, `{100*hseeds[1]['eval_chosen_rank_top1_fraction']:.1f}%`다.

![R3 Hospital alpha 진단](../../../../results/peft_mechanism_diagnostics_v1/hospital_strength/hospital_alpha_diagnostic.png)

[미확정] **12개월 V2 선택 잡음, 시간 전이, 안정적인 계열별 이질성 부족을 현재 자료로 구분하지 못한다.** 사후 oracle 이득은 잡음만 있어도 생길 수 있으므로 안정적 개인화 가능성의 증거가 아니다. 과거로 예측 가능한 차이를 별도로 확보할 때만 단순 수축을 검토한다.

## 지금 잡을 주제

가장 좋은 주제 문장은 아직 "새 adapter를 제안한다"가 아니다. 지금은 다음처럼 잡는 게 맞다.

> 시계열 foundation model PEFT에서 target 데이터의 원시 복잡도보다, frozen representation이 남긴 회수 가능한 시간 구조와 그 구조가 위치한 적응 범위를 진단해 PEFT 필요성과 배치를 결정한다.

이 문장은 R1과 R2를 하나로 묶는다. R3는 별도 보조 결과로, per-series personalization이 얼마나 쉽게 validation noise에 먹히는지 보여준다.

## 다음 실행 순서

1. Bike/Household에서 Frozen+MLP vs LoRA+같은 MLP matched control을 먼저 실행한다.
2. 그 차이가 남으면 attention layer group을 사전 지정해서 R2를 실데이터로 옮긴다.
3. Hospital은 이미 본 E를 새 test로 쓰지 않는다. 더 긴 과거 반복에서 안정적 차이가 있는지 먼저 검사하고, 그 근거 없이 수축이나 그룹 adapter를 실행하지 않는다.

이번 산출물: [R1 summary](../../../../results/peft_mechanism_diagnostics_v1/reference/summary.json), [R2 summary](../../../../results/peft_mechanism_diagnostics_v1/layer_scope/summary.json), [R3 summary](../../../../results/peft_mechanism_diagnostics_v1/hospital_strength/summary.json), [통합 summary](../../../../results/peft_mechanism_diagnostics_v1/followup_summary.json).
"""
    DOC.parent.mkdir(parents=True, exist_ok=True)
    doc = '> Historical saved-artifact CPU snapshot only. New controlled training in runs/peft_mechanism_diagnostics_v1 is reported separately.\n\n' + doc
    DOC.write_text(doc.replace('../../../../results/', '../../results/'), encoding="utf-8")
    return summary


def main() -> None:
    start = time.perf_counter()
    reference = load_reference_summary()
    layer = build_layer_scope()
    hospital = build_hospital_strength()
    summary = build_overall_report(reference, layer, hospital)
    summary["seconds"] = time.perf_counter() - start
    with (OUT / "followup_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, allow_nan=False)
    print(
        json.dumps(
            {
                "completed": True,
                "seconds": summary["seconds"],
                "doc": str(DOC.relative_to(ROOT)),
                "r2_attn_only_saves_params_pct": layer["attn_only_saves_params_pct_of_both"],
                "r3_eval_individual_vs_global_pctF0": hospital["aggregate_eval_individual_vs_global_pctF0"],
                "r3_eval_oracle_vs_global_pctF0_posthoc": hospital["aggregate_eval_oracle_vs_global_pctF0_posthoc"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
