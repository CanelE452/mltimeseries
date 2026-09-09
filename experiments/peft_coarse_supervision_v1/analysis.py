"""Pure, sealed-evaluation metrics for the fixed coarse-supervision screen."""

import numpy as np


ARMS = ("F0", "PROFILE", "COARSE_LIFT", "FROZEN_HEAD", "ATTN_LORA")
SIMPLE_ARMS = ARMS[:-1]
METRICS = ("mse", "level_mse", "pattern_mse", "mae")
BOOTSTRAP_REPLICATES = 4000
BOOTSTRAP_SEED = 2026090817


def _average(records):
    if not records:
        return {arm: {metric: None for metric in METRICS} for arm in ARMS}
    return {
        arm: {
            metric: float(np.mean([record[arm][metric] for record in records]))
            for metric in METRICS
        }
        for arm in ARMS
    }


def _effect(arms, simple_policy):
    denominator = arms["F0"]["mse"]
    if denominator is None or denominator <= 0:
        return None
    return float((arms[simple_policy]["mse"] - arms["ATTN_LORA"]["mse"]) / denominator)


def _interval(numerator, denominator):
    valid = np.isfinite(numerator) & np.isfinite(denominator) & (denominator > 0)
    count = int(valid.sum())
    if count != BOOTSTRAP_REPLICATES:
        return None, count
    return np.quantile(numerator / denominator, [0.025, 0.975]).tolist(), count


def analyze_predictions(predictions, truth, horizon, scale, site, target_id, month, simple_policy):
    """Analyze all 48 fixed target/month cells without files, fitting, or mutation.

    MSE, MAE and the level/pattern decomposition use errors divided by the
    supplied positive training annual mean. Only finite, nonnegative truth
    within each horizon defines observation validity. Forecast failures never
    remove observations. Site scores weight targets equally and eligible months
    equally within each target; the pooled score weights the two sites equally.
    Month labels identify cells; calendar/date provenance belongs to the caller.
    """
    if set(predictions) != set(ARMS):
        raise ValueError(f"Exactly these prediction arms are required: {ARMS}")
    if simple_policy not in SIMPLE_ARMS:
        raise ValueError("simple_policy must be fixed to one of the four simple arms")
    truth = np.asarray(truth, dtype=np.float64)
    if truth.shape != (48, 744):
        raise ValueError("The fixed evaluation requires truth shape (48, 744)")
    predictions = {arm: np.asarray(predictions[arm], dtype=np.float64) for arm in ARMS}
    if any(values.shape != truth.shape for values in predictions.values()):
        raise ValueError("Every prediction arm must match truth shape")
    horizon = np.asarray(horizon)
    scale = np.asarray(scale, dtype=np.float64)
    site = np.asarray(site).astype(str)
    target_id = np.asarray(target_id).astype(str)
    month = np.asarray(month).astype(str)
    if any(values.shape != (48,) for values in (horizon, scale, site, target_id, month)):
        raise ValueError("All metadata arrays must have shape (48,)")
    if not np.all(np.isfinite(horizon)) or not np.all(horizon == np.floor(horizon)):
        raise ValueError("Horizons must be finite integers")
    horizon = horizon.astype(np.int64)
    if np.any(horizon < 1) or np.any(horizon > truth.shape[1]):
        raise ValueError("Horizons must be between 1 and 744")
    if not np.all(np.isfinite(scale) & (scale > 0)):
        raise ValueError("Training annual-mean scales must be positive and finite")
    site_names = sorted(set(site.tolist()))
    month_names = set(month.tolist())
    if len(site_names) != 2 or len(month_names) != 3:
        raise ValueError("Exactly two sites and three month labels are required")
    identities = list(zip(site.tolist(), target_id.tolist(), month.tolist()))
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate site/target/month cell")
    target_names = {}
    for name in site_names:
        targets = sorted(set(target_id[site == name].tolist()))
        if len(targets) != 8:
            raise ValueError("Each site must retain all eight prespecified targets")
        target_names[name] = targets
        for target in targets:
            indices = (site == name) & (target_id == target)
            if set(month[indices].tolist()) != month_names or int(indices.sum()) != 3:
                raise ValueError("Each target must have the same three month cells")
            if not np.all(scale[indices] == scale[indices][0]):
                raise ValueError("A target must use the same training scale in every month")

    valid = (np.arange(744)[None, :] < horizon[:, None]) & np.isfinite(truth) & (truth >= 0)
    for arm, values in predictions.items():
        if not np.all(np.isfinite(values[valid])):
            raise ValueError(f"Nonfinite prediction on observed support: {arm}")
    cells = []
    decomposition_max_abs_error = 0.0
    for index, (name, target, month_label) in enumerate(identities):
        mask = valid[index]
        count = int(mask.sum())
        arms = {}
        for arm in ARMS:
            if count == 0:
                arms[arm] = {metric: None for metric in METRICS}
                continue
            error = (predictions[arm][index, mask] - truth[index, mask]) / scale[index]
            mean_error = float(np.mean(error))
            mse = float(np.mean(error ** 2))
            level = mean_error ** 2
            pattern = float(np.mean((error - mean_error) ** 2))
            arms[arm] = {"mse": mse, "level_mse": level, "pattern_mse": pattern,
                         "mae": float(np.mean(np.abs(error)))}
            if not all(np.isfinite(value) for value in arms[arm].values()):
                raise ValueError(f"Nonfinite normalized metric: {arm}, cell {index}")
            difference = abs(mse - level - pattern)
            decomposition_max_abs_error = max(decomposition_max_abs_error, difference)
            if difference > 1e-10 * max(1.0, mse):
                raise ArithmeticError("Mean-error/pattern decomposition failed")
        cells.append({"site": name, "target_id": target, "month": month_label,
                      "horizon": int(horizon[index]), "scale": float(scale[index]),
                      "count": count, "valid_fraction": count / int(horizon[index]),
                      "eligible": 5 * count >= 4 * int(horizon[index]), "arms": arms})

    sites = {}
    target_metrics = {}
    for name in site_names:
        site_cells = [cell for cell in cells if cell["site"] == name]
        eligible = [cell for cell in site_cells if cell["eligible"]]
        per_target = {}
        for target in target_names[name]:
            target_cells = [cell["arms"] for cell in eligible if cell["target_id"] == target]
            if target_cells:
                per_target[target] = _average(target_cells)
        absent = [target for target in target_names[name] if target not in per_target]
        target_metrics[name] = per_target
        sites[name] = {
            "cell_count": len(site_cells), "target_count": len(target_names[name]),
            "eligible_cell_count": len(eligible),
            "valid_hour_count": sum(cell["count"] for cell in site_cells),
            "eligible_valid_hour_count": sum(cell["count"] for cell in eligible),
            "targets_without_eligible_cells": absent,
            "data_gate_passed": len(eligible) >= 16 and not absent,
            "arms": _average(list(per_target.values())),
        }
    data_gate = all(record["data_gate_passed"] for record in sites.values())
    all_sites_have_scores = all(record["arms"]["F0"]["mse"] is not None for record in sites.values())
    pooled = {
        "site_count": len(site_names),
        "eligible_cell_count": sum(record["eligible_cell_count"] for record in sites.values()),
        "valid_hour_count": sum(record["valid_hour_count"] for record in sites.values()),
        "eligible_valid_hour_count": sum(record["eligible_valid_hour_count"] for record in sites.values()),
        "arms": _average([sites[name]["arms"] for name in site_names] if all_sites_have_scores else []),
    }
    effects = {
        "sites": {name: {"point": _effect(sites[name]["arms"], simple_policy),
                         "ci95": None, "bootstrap_valid_replicates": 0} for name in site_names},
        "pooled": {"point": _effect(pooled["arms"], simple_policy),
                   "ci95": None, "bootstrap_valid_replicates": 0},
        "definition": "(simple_policy_mse - ATTN_LORA_mse) / F0_mse",
        "simple_policy": simple_policy,
    }
    if data_gate:
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        resampled = []
        simple_index = ARMS.index(simple_policy)
        lora_index = ARMS.index("ATTN_LORA")
        for name in site_names:
            scores = np.array([[target_metrics[name][target][arm]["mse"] for arm in ARMS]
                               for target in target_names[name]])
            indices = rng.integers(0, len(scores), size=(BOOTSTRAP_REPLICATES, len(scores)))
            means = scores[indices].mean(axis=1)
            interval, count = _interval(means[:, simple_index] - means[:, lora_index], means[:, 0])
            effects["sites"][name].update(ci95=interval, bootstrap_valid_replicates=count)
            resampled.append(means)
        pooled_replicates = np.mean(resampled, axis=0)
        interval, count = _interval(pooled_replicates[:, simple_index] - pooled_replicates[:, lora_index],
                                    pooled_replicates[:, 0])
        effects["pooled"].update(ci95=interval, bootstrap_valid_replicates=count)

    point_gates = {name: effects["sites"][name]["point"] is not None
                  and effects["sites"][name]["point"] >= 0.01 for name in site_names}
    pattern_gates = {}
    for name in site_names:
        arms = sites[name]["arms"]
        lora_pattern, head_pattern = arms["ATTN_LORA"]["pattern_mse"], arms["FROZEN_HEAD"]["pattern_mse"]
        pattern_gates[name] = lora_pattern is not None and head_pattern is not None and lora_pattern < head_pattern
    pooled_interval = effects["pooled"]["ci95"]
    positive_interval = pooled_interval is not None and pooled_interval[0] > 0
    proceed = data_gate and all(point_gates.values()) and all(pattern_gates.values()) and positive_interval
    decision = ("INSUFFICIENT_EVALUATION_DATA" if not data_gate else
                "PROCEED_TO_METHOD_DEVELOPMENT" if proceed else
                "CLOSE_CURRENT_COARSE_SUPERVISION_SCREEN")
    return {
        "completed": True, "cells": cells, "sites": sites, "pooled": pooled,
        "effects": effects, "data_gate_passed": data_gate,
        "decomposition_max_abs_error": decomposition_max_abs_error,
        "macro_definition": "Eligible months equal within target; targets equal within site; sites equal pooled",
        "normalization": "Error divided by target training annual-mean scale",
        "bootstrap": {
            "performed": data_gate, "replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED,
            "confidence": 0.95, "unit": "target within site, retaining all eligible months together",
            "paired_across_arms": True, "site_order": site_names,
            "scope": "Conditional target-cluster uncertainty; excludes training-seed and temporal-generalization uncertainty",
            "zero_denominator_policy": "Return null CI if any replicate has undefined F0 normalization",
        },
        "progress_gate": {"passed": proceed, "site_effect_at_least_one_percent": point_gates,
                          "site_pattern_better_than_head": pattern_gates,
                          "pooled_ci_lower_positive": positive_interval},
        "decision": decision, "novel_method_demonstrated": False,
        "interpretation": "Exploratory necessity screen; repeated topic-search error is not controlled",
    }
