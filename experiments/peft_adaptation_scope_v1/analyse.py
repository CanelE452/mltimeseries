"""Descriptive, paired analysis of the two-panel development screen."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT = Path(__file__).resolve().parents[2]
RUN = PROJECT / "runs/peft_adaptation_scope_v1"
RESULT = PROJECT / "results/peft_adaptation_scope_v1"
METHODS = ["RAW_VARX_RIDGE", "F0", "AFF", "H_LIN", "H_MLP", "H_FULL", "OFF_LORA", "FULL"]


def select_raw_result(panel):
    candidates = []
    for family in ["raw", "raw_rescue"]:
        folder = RUN / family / panel
        if family == "raw_rescue" and not folder.exists():
            continue
        metric = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
        if not metric.get("completed") or not np.isfinite(metric["val_score"]):
            raise RuntimeError(f"Incomplete or invalid RAW candidate: {folder}")
        candidates.append({"source_path": str(folder.resolve()), "raw_rescue_used": family == "raw_rescue",
                           "val_score": metric["val_score"], "selected_ridge": metric["selected_ridge"],
                           "penalty_grid": metric["rawknobs"]["lambda_grid"],
                           "wall_seconds": metric["wall_seconds"]})
    # Stable tie handling retains the original run; evaluation never enters this decision.
    chosen = min(candidates, key=lambda candidate: candidate["val_score"])
    union_grid = sorted({value for candidate in candidates for value in candidate["penalty_grid"]})
    return {**chosen, "penaltyboundary": chosen["selected_ridge"] in [min(union_grid), max(union_grid)],
            "all_evaluated_penalties": union_grid, "source_candidate_count": len(chosen["penalty_grid"]),
            "total_candidates_executed": sum(len(candidate["penalty_grid"]) for candidate in candidates),
            "total_search_wall_seconds": sum(candidate["wall_seconds"] for candidate in candidates),
            "selection_policy": "Minimum validation scaled 2-pinball across preserved original/rescue runs; original wins ties",
            "candidates": candidates}


def score_prediction(prediction, target, scale, quantiles):
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    scale = np.asarray(scale, dtype=np.float64)
    quantiles = np.asarray(quantiles, dtype=np.float64)
    if not np.isfinite(prediction).all():
        raise FloatingPointError("Nonfinite forecast in analysis")
    error = target[:, :, None, :] - prediction
    q = quantiles[None, None, :, None]
    loss = 2 * np.maximum(q * error, (q - 1) * error) / scale[None, :, None, None]
    valid = np.isfinite(target)
    sums = np.where(valid[:, :, None, :], loss, 0.0).sum(axis=(2, 3))
    counts = valid.sum(axis=2) * len(quantiles)
    return float(macro_score(sums, counts)), sums, counts


def macro_score(sums, counts):
    """Sufficient statistics end in (origin, channel); all leading axes are preserved."""
    totals = np.asarray(sums).sum(axis=-2)
    denominators = np.asarray(counts).sum(axis=-2)
    per_channel = np.divide(totals, denominators,
                            out=np.full(totals.shape, np.nan), where=denominators > 0)
    # Every fixed target must be represented; silently dropping a target changes the estimand.
    return np.mean(per_channel, axis=-1)


def paired_block_ci(difference_sums, counts, baseline_sums, block, seed=20260908):
    rng = np.random.default_rng(seed)
    days = difference_sums.shape[-2]
    starts = rng.integers(0, days, size=(4000, int(np.ceil(days / block))))
    indices = ((starts[:, :, None] + np.arange(block)) % days).reshape(4000, -1)[:, :days]
    sampled_counts = counts[indices]
    numerator = macro_score(difference_sums[:, indices, :], sampled_counts).mean(axis=0)
    denominator = macro_score(baseline_sums[indices], sampled_counts)
    valid = np.isfinite(numerator) & np.isfinite(denominator) & (denominator > 0)
    sampled = numerator[valid] / denominator[valid]
    if not len(sampled):
        raise RuntimeError("No bootstrap draw contains every fixed target and a positive F0 score")
    return {"ci95": np.quantile(sampled, [0.025, 0.975]).tolist(),
            "valid_resamples": int(valid.sum()), "requested_resamples": len(indices),
            "discarded_resamples": int((~valid).sum()),
            "denominator": "F0 score recomputed on the same resampled blocks",
            "missing_target_policy": "Discard draws lacking all observations of any fixed target"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--partial", action="store_true")
    args = parser.parse_args()
    selection_file = RUN / "selection.json"
    selection = json.loads(selection_file.read_text(encoding="utf-8")) if selection_file.exists() else {}
    rows, effects, full_metrics, raw_selections = [], [], [], {}
    panel_method_statistics = {}
    for panel in ["ettm2", "jena"]:
        data_file = RUN / "prepared" / f"{panel}.npz"
        if not data_file.exists():
            continue
        with np.load(data_file) as data:
            horizon = int(data["horizon"])
            target = data["values"][data["eval_origins"][:, None] + np.arange(horizon)[None, :]].transpose(0, 2, 1)
            scale = data["fit_std"]
            eval_origins = data["eval_origins"]
            val_origins = data["val_origins"]
            val_target = data["values"][val_origins[:, None] + np.arange(horizon)[None, :]].transpose(0, 2, 1)
        for method in METHODS:
            is_raw = method == "RAW_VARX_RIDGE"
            key = f"{panel}/{method}"
            if method != "F0" and not is_raw and key not in selection:
                continue
            lr = 0 if method == "F0" or is_raw else selection[key]["lr"]
            raw_choice = None
            if is_raw:
                if args.partial and not (RUN / "raw" / panel / "metrics.json").exists():
                    continue
                raw_choice = select_raw_result(panel)
                raw_selections[panel] = raw_choice
            trial_statistics = {}
            for seed in ([0] if method == "F0" or is_raw else [0, 1, 2]):
                trial = Path(raw_choice["source_path"]) if is_raw else RUN / "trials" / f"{panel}_{method}_lr{lr:g}_seed{seed}"
                metrics_file = trial / "metrics.json"
                if not metrics_file.exists():
                    if args.partial:
                        continue
                    raise RuntimeError(f"Missing selected trial: {trial}")
                metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
                if not metrics.get("completed"):
                    raise RuntimeError(f"Incomplete selected trial: {trial}")
                if is_raw:
                    lr = metrics["selected_ridge"]
                with np.load(trial / "predictions.npz") as predictions:
                    pred = predictions["eval_pred"]
                    quantiles = predictions["quantiles"]
                    if not np.array_equal(predictions["eval_origins"], eval_origins):
                        raise AssertionError(f"Evaluation origin order mismatch: {trial}")
                    if is_raw:
                        if not np.array_equal(predictions["val_origins"], val_origins):
                            raise AssertionError(f"RAW validation origin order mismatch: {trial}")
                        raw_val_score, _, _ = score_prediction(predictions["val_pred"], val_target, scale, quantiles)
                        if not np.isclose(raw_val_score, metrics["val_score"], rtol=2e-5, atol=2e-6):
                            raise AssertionError(f"RAW validation metric reconciliation failed: {trial}")
                    score, loss_sums, loss_counts = score_prediction(pred, target, scale, quantiles)
                    median_idx = int(np.argmin(np.abs(quantiles - 0.5)))
                    low_idx, high_idx = [int(np.argmin(np.abs(quantiles - v))) for v in [0.1, 0.9]]
                    valid = np.isfinite(target)
                    coverage = float(np.mean(((target >= pred[:, :, low_idx]) & (target <= pred[:, :, high_idx]))[valid]))
                    crossing = float(np.mean(np.diff(pred, axis=2) < 0))
                    median_mae = float(np.nanmean(np.nanmean(np.abs(target - pred[:, :, median_idx]), axis=(0, 2))))
                    proxy_mse = float(np.nanmean(np.nanmean((target - np.mean(pred, axis=2)) ** 2, axis=(0, 2))))
                if not np.isclose(score, metrics["eval_score"], rtol=2e-5, atol=2e-6):
                    raise AssertionError(f"Independent metric reconciliation failed: {trial}: {score} != {metrics['eval_score']}")
                rows.append({"panel": panel, "method": method, "seed": seed, "lr": lr,
                             "source_path": str(trial.resolve()),
                             "raw_rescue_used": raw_choice["raw_rescue_used"] if is_raw else None,
                             "penaltyboundary": raw_choice["penaltyboundary"] if is_raw else None,
                             "raw_source_candidate_count": raw_choice["source_candidate_count"] if is_raw else None,
                             "raw_unique_penalty_count": len(raw_choice["all_evaluated_penalties"]) if is_raw else None,
                             "raw_total_candidates_executed": raw_choice["total_candidates_executed"] if is_raw else None,
                             "raw_total_search_wall_seconds": raw_choice["total_search_wall_seconds"] if is_raw else None,
                             "lr_definition": "ridge regularization lambda" if is_raw else "optimizer learning rate",
                             "selected_step": metrics["selected_step"], "val_score": metrics["val_score"],
                             "eval_score": score, "coverage80": coverage, "crossing_rate": crossing,
                             "median_mae_raw_macro": median_mae, "quantile_average_proxy_mse": proxy_mse,
                             "proxy_mse_definition": "Arithmetic mean of 21 quantiles; not an identified conditional mean",
                             "trainable_parameters": metrics.get("trainable_parameters"),
                             "ridge_coefficient_count": metrics.get("coefficient_count"),
                             "empirical_quantile_offset_count": metrics.get("calibration_offset_count"),
                             "parameter_count_definition": "RAW closed-form coefficients/intercepts and empirical offsets are reported separately; no optimizer trainable-parameter count" if is_raw else "Registered trainable neural-network scalars",
                             "wall_seconds": metrics.get("wall_seconds"),
                             "wall_seconds_scope": f"Selected RAW source run: {raw_choice['source_candidate_count']} penalties including OOF calibration; all original/rescue costs recorded separately" if is_raw else "One selected trial; cache-generation inclusion is recorded in trial metadata",
                             "peak_vram_gib": metrics.get("peak_vram_gib")})
                trial_statistics[seed] = {"sums": loss_sums, "counts": loss_counts,
                                          "score": score, "metric_score": metrics["eval_score"]}
                full_metrics.append({"trial": str(trial.relative_to(RUN)) if is_raw else trial.name,
                                     "raw_selection": raw_choice, **metrics})
            if trial_statistics:
                panel_method_statistics[(panel, method)] = trial_statistics
        head_candidates = [m for m in ["H_LIN", "H_MLP", "H_FULL"] if f"{panel}/{m}" in selection]
        if not head_candidates or (panel, "F0") not in panel_method_statistics:
            continue
        head = min(head_candidates, key=lambda m: selection[f"{panel}/{m}"]["val_score"])
        baseline = panel_method_statistics[(panel, "F0")][0]
        base = baseline["score"]
        if not np.isfinite(base) or base <= 0:
            raise RuntimeError(f"Cannot normalize effects by F0 score {base}: {panel}")
        for internal in ["OFF_LORA", "FULL"]:
            if (panel, head) not in panel_method_statistics or (panel, internal) not in panel_method_statistics:
                continue
            head_statistics = panel_method_statistics[(panel, head)]
            internal_statistics = panel_method_statistics[(panel, internal)]
            paired_seeds = sorted(set(head_statistics) & set(internal_statistics))
            if not paired_seeds:
                continue
            counts = baseline["counts"]
            for seed in paired_seeds:
                if not (np.array_equal(head_statistics[seed]["counts"], counts)
                        and np.array_equal(internal_statistics[seed]["counts"], counts)):
                    raise AssertionError(f"Methods must use the same target availability: {panel}/{seed}")
            difference_sums = np.stack([head_statistics[seed]["sums"] - internal_statistics[seed]["sums"]
                                        for seed in paired_seeds])
            seed_deltas = macro_score(difference_sums, counts) / base
            metric_deltas = np.array([(head_statistics[seed]["metric_score"] - internal_statistics[seed]["metric_score"])
                                      / baseline["metric_score"] for seed in paired_seeds])
            if not np.allclose(seed_deltas, metric_deltas, rtol=2e-5, atol=2e-6):
                raise AssertionError(f"Point effects differ from stored primary score differences: {panel}/{internal}")
            bootstrap = {str(b): paired_block_ci(difference_sums, counts, baseline["sums"], b)
                         for b in [1, 3, 7]}
            effects.append({"panel": panel, "head_selected_on_validation": head, "internal": internal,
                            "paired_seed_count": len(paired_seeds), "paired_seed_ids": paired_seeds,
                            "delta_over_F0": float(seed_deltas.mean()), "seed_deltas": seed_deltas.tolist(),
                            "point_effect_metric_reconciliation": True,
                            "seed_aggregation": "Arithmetic mean of separately evaluated seed losses; no prediction ensemble",
                            "conditional_dayblock_ci95": {key: value["ci95"] for key, value in bootstrap.items()},
                            "bootstrap_details": bootstrap,
                            "uncertainty_scope": "Same circular evaluation-day blocks resample every method, channel and seed. Recompute each channel's loss sum / valid count and the F0 denominator. Conditional on these trained seeds; two development panels are not independent-domain evidence."})
    if not rows:
        raise RuntimeError("No selected, completed results to analyse")
    RESULT.mkdir(parents=True, exist_ok=True)
    with (RESULT / "selected_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (RESULT / "effects.json").write_text(json.dumps(effects, indent=2), encoding="utf-8")
    (RESULT / "selected_trial_details.json").write_text(json.dumps(full_metrics, indent=2), encoding="utf-8")
    (RESULT / "raw_selection.json").write_text(json.dumps(raw_selections, indent=2), encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for axis, panel in zip(axes, ["ettm2", "jena"]):
        available = [method for method in METHODS if any(r["panel"] == panel and r["method"] == method for r in rows)]
        values = [np.array([r["eval_score"] for r in rows if r["panel"] == panel and r["method"] == method]) for method in available]
        for pos, (method, value) in enumerate(zip(available, values)):
            axis.bar(pos, value.mean(), color="#087e8b" if method in ["OFF_LORA", "FULL"] else "#99a4b3", width=0.65)
            axis.scatter(np.full(len(value), pos), value, color="#16212d", s=20, zorder=3)
        labels = ["RAW ridge" if method == "RAW_VARX_RIDGE" else method for method in available]
        axis.set_xticks(np.arange(len(available)), labels, rotation=35, ha="right")
        axis.set_title(panel.upper() + " · development evaluation")
        axis.set_ylabel("Fit-scaled 2-pinball (lower is better)")
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
    fig.savefig(RESULT / "adaptation_comparison.png", dpi=180)
    fig.savefig(RESULT / "adaptation_comparison.pdf")
    plt.close(fig)
    print(json.dumps({"selected_trials": len(rows), "effects": effects, "output": str(RESULT)}, indent=2))


if __name__ == "__main__":
    main()
