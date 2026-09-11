"""CPU-only audit for the capacity-matched head and short-response probe."""
import os
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "2"

import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "runs/peft_capacity_probe_v1"
OUT = ROOT / "results/peft_capacity_probe_v1"
PERIODS = ("P0", "P1")
DATASETS = ("bike", "household")
CONDITIONS = ("FULL90", "SPREAD30", "RECENT30")
SEEDS = {"P0": (25000, 25001), "P1": (26000, 26001)}
RULES = ("PROBE_1", "PROBE_2")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def rel(path):
    return Path(path).relative_to(ROOT).as_posix()


def assert_same_path(left, right):
    if Path(left).resolve() != Path(right).resolve():
        raise AssertionError(f"path mismatch: {left} != {right}")


def verify_plan(plan):
    for path, digest in {**plan["source_hashes"], **plan["original_input_hashes"], **plan["preserved_inputs"]}.items():
        if sha(ROOT / path) != digest:
            raise AssertionError(f"hash mismatch: {path}")


def read_forecast_npz(path):
    with np.load(path, allow_pickle=False) as z:
        key = "prediction" if "prediction" in z.files else "predictions"
        item = {
            "prediction": z[key].copy(),
            "target": z["target"].copy(),
            "origins": z["origins"].copy() if "origins" in z.files else None,
            "quantiles": z["quantiles"].copy() if "quantiles" in z.files else None,
            "scale": z["scale"].copy() if "scale" in z.files else None,
        }
    return item


def score(prediction, target, scale, quantiles, rows=None):
    pred = np.asarray(prediction, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    if rows is not None:
        pred = pred[rows]
        y = y[rows]
    scale = np.asarray(scale, dtype=np.float64)
    q = np.asarray(quantiles, dtype=np.float64)
    if pred.shape != (len(y), y.shape[1], len(q), y.shape[2]):
        raise AssertionError(f"score shape mismatch: prediction={pred.shape}, target={y.shape}, quantiles={q.shape}")
    if not np.isfinite(pred).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise AssertionError("invalid prediction or scale")
    if np.any(np.diff(pred, axis=2) < -1e-12):
        raise AssertionError("non-monotone quantile predictions")
    valid = np.isfinite(y)
    counts = valid.sum(axis=(0, 2)).astype(np.float64)
    if np.any(counts <= 0):
        raise AssertionError("empty target count")
    error = np.where(valid[:, :, None, :], y[:, :, None, :] - pred, 0.0)
    loss = 2.0 * np.maximum(q[None, None, :, None] * error, (q[None, None, :, None] - 1.0) * error)
    return float((loss.sum(axis=(0, 3)) / counts[:, None] / scale[:, None]).mean())


def reference_scale_quantiles(plan, period, dataset):
    spec = plan["references"][period]["datasets"][dataset]
    with np.load(ROOT / spec["holdout_data_path"], allow_pickle=False) as z:
        return z["fit_std"][z["target_indices"]].copy(), z["quantiles"].copy()


def load_f0(plan):
    f0, meta, errors = {}, {}, []
    for period in PERIODS:
        f0[period], meta[period] = {}, {}
        for dataset in DATASETS:
            path = ROOT / plan["f0_paths"][period][dataset]
            item = read_forecast_npz(path)
            scale, quantiles = item["scale"], item["quantiles"]
            if scale is None or quantiles is None:
                scale, quantiles = reference_scale_quantiles(plan, period, dataset)
            value = score(item["prediction"], item["target"], scale, quantiles)
            record_path = path.with_name("result.json")
            if record_path.exists():
                record = load_json(record_path)
                if "score" in record:
                    errors.append(abs(value - record["score"]))
            f0[period][dataset] = value
            meta[period][dataset] = {
                "target": item["target"],
                "origins": item["origins"],
                "scale": scale,
                "quantiles": quantiles,
                "path": rel(path),
            }
    return f0, meta, errors


def old_root(period):
    return ROOT / ("runs/peft_optimization_control_v1" if period == "P0" else "runs/peft_overlap_transfer_v1")


def old_eval_key(period, fit_key):
    replacement = "eval/EXPOSURE/" if period == "P0" else "eval/"
    return fit_key.replace("fits/", replacement, 1)


def load_eval_score(output, expected_job, expected_meta, expected_regime="EXPOSURE"):
    record = load_json(output / "result.json")
    if not record["completed"] or record["job"] != expected_job or record.get("regime") != expected_regime:
        raise AssertionError(f"invalid eval record: {output}")
    item = read_forecast_npz(output / "predictions.npz")
    if item["origins"] is not None and expected_meta["origins"] is not None:
        np.testing.assert_equal(item["origins"], expected_meta["origins"])
    np.testing.assert_equal(item["target"], expected_meta["target"])
    if item["scale"] is not None:
        np.testing.assert_allclose(item["scale"], expected_meta["scale"], rtol=0, atol=0)
    if item["quantiles"] is not None:
        np.testing.assert_allclose(item["quantiles"], expected_meta["quantiles"], rtol=0, atol=0)
    value = score(item["prediction"], item["target"], expected_meta["scale"], expected_meta["quantiles"])
    return record, value, abs(value - record["score"])


def guard_status(path):
    return load_json(path / "guard/status.json")


def guard_elapsed(path):
    status = guard_status(path)
    if not status["completed"] or status.get("reasons"):
        raise AssertionError(f"guard not cleanly completed: {path}")
    return float(status["elapsed_seconds"])


def parse_time(value):
    return datetime.fromisoformat(value).timestamp()


def history_score(result, step):
    matches = [h for h in result["history"] if h["step"] == step]
    if len(matches) != 1:
        raise AssertionError(f"missing history step {step}")
    return float(matches[0]["score"])


def prefix_wall_seconds(run_path, result, step):
    status = guard_status(run_path)
    if not status["completed"] or status.get("reasons"):
        raise AssertionError(f"guard not cleanly completed: {run_path}")
    matches = [h for h in result["history"] if h["step"] == step]
    if len(matches) != 1 or "utc" not in matches[0]:
        raise AssertionError(f"missing UTC checkpoint for {run_path} step {step}")
    delta = parse_time(matches[0]["utc"]) - parse_time(status["started_at"])
    elapsed = float(status["elapsed_seconds"])
    if delta < -1e-6 or delta - elapsed > 1e-6:
        raise AssertionError(f"prefix checkpoint outside guard clock: {run_path} step {step} delta={delta} elapsed={elapsed}")
    return float(delta)


def assert_fit_result(path, job, contract_path, trainable=1768949):
    result = load_json(path / "output/result.json")
    if not result["completed"] or result["job"] != job:
        raise AssertionError(f"invalid fit result: {path}")
    if result["contract_sha256"] != sha(contract_path):
        raise AssertionError(f"contract hash mismatch: {path}")
    if result["trainable"] != trainable or result["identity_error"] >= 1e-5:
        raise AssertionError(f"fit identity/count mismatch: {path}")
    if not result["frozen_verified"] or result["holdout_opened"]:
        raise AssertionError(f"fit leakage/frozen check failed: {path}")
    return result


def validate_wide_selection(plan):
    wide0 = plan["wide_p0_jobs"]
    selected0 = load_json(RUN / "wide_p0_selection.json")
    p1_jobs = load_json(RUN / "p1_jobs.json")
    selected_keys = {e["key"] for e in selected0}
    recipe_means, recipes = {}, {}
    for dataset in DATASETS:
        for condition in CONDITIONS:
            group = [e for e in wide0 if (e["job"]["dataset"], e["job"]["condition"]) == (dataset, condition)]
            if len(group) != 4:
                raise AssertionError(f"unexpected P0 WIDE group size: {dataset}/{condition}")
            means = {}
            for recipe in (0, 1):
                scores = []
                for entry in group:
                    if entry["job"]["recipe"] != recipe:
                        continue
                    result = assert_fit_result(RUN / entry["key"], entry["job"], RUN / entry["contract"])
                    scores.append(result["regimes"]["EXPOSURE"]["best_score"])
                if len(scores) != 2:
                    raise AssertionError(f"missing recipe scores: {dataset}/{condition}/r{recipe}")
                means[str(recipe)] = float(np.mean(scores))
            chosen = min((0, 1), key=lambda r: (means[str(r)], -r))
            recipes[(dataset, condition)] = chosen
            recipe_means[f"P0/{dataset}/{condition}"] = {"means": means, "chosen_recipe": chosen}
            expected = {e["key"] for e in group if e["job"]["recipe"] == chosen}
            if not expected <= selected_keys:
                raise AssertionError(f"P0 WIDE selection mismatch: {dataset}/{condition}")
    for entry in p1_jobs:
        job = entry["job"]
        if job["recipe"] != recipes[(job["dataset"], job["condition"])]:
            raise AssertionError(f"P1 transferred recipe mismatch: {entry['key']}")
    selections = load_json(RUN / "selection.json")
    expected = {e["key"] for e in selected0 + p1_jobs}
    actual = {e["key"] for e in selections}
    if actual != expected or len(selections) != 24:
        raise AssertionError("sealed WIDE selection mismatch")
    return selections, selected0, p1_jobs, recipe_means


def load_old_selected(period, arm):
    root = old_root(period)
    selection = load_json(root / "selection.json")
    out = {}
    for entry in selection:
        if period == "P0" and entry.get("regime") != "EXPOSURE":
            continue
        if entry["job"]["arm"] != arm:
            continue
        job = entry["job"]
        key = (root / entry["key"]).relative_to(ROOT).as_posix()
        eval_key = (root / old_eval_key(period, entry["key"])).relative_to(ROOT).as_posix()
        out[(period, job["dataset"], job["condition"], job["seed"])] = {**entry, "fit_key": key, "eval_key": eval_key}
    return out


def compare_target_meta(item, meta):
    if item["origins"] is not None and meta["origins"] is not None:
        np.testing.assert_equal(item["origins"], meta["origins"])
    np.testing.assert_equal(item["target"], meta["target"])
    if item["scale"] is not None:
        np.testing.assert_allclose(item["scale"], meta["scale"], rtol=0, atol=0)
    if item["quantiles"] is not None:
        np.testing.assert_allclose(item["quantiles"], meta["quantiles"], rtol=0, atol=0)


def point_score(path, meta, rows=None):
    item = read_forecast_npz(path)
    compare_target_meta(item, meta)
    return score(item["prediction"], item["target"], meta["scale"], meta["quantiles"], rows=rows)


def average_ranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        rank = (i + j - 1) / 2.0
        for k in range(i, j):
            ranks[order[k]] = rank
        i = j
    return np.asarray(ranks, dtype=np.float64)


def corr(x, y, rank=False):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if rank:
        x, y = average_ranks(x), average_ranks(y)
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def sign(value):
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_capacity(capacity_rows, summary):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    colors = {"wide_minus_all_pct_F0": "#0072B2", "mlp_minus_all_pct_F0": "#999999"}
    for row_idx, period in enumerate(PERIODS):
        for col_idx, dataset in enumerate(DATASETS):
            ax = axes[row_idx, col_idx]
            for offset, key, label in [(-0.18, "mlp_minus_all_pct_F0", "small MLP - ALL"), (0.18, "wide_minus_all_pct_F0", "WIDE - ALL")]:
                means, dots = [], []
                for condition in CONDITIONS:
                    vals = [r[key] for r in capacity_rows if r["period"] == period and r["dataset"] == dataset and r["condition"] == condition and r[key] != ""]
                    means.append(float(np.mean(vals)))
                    dots.append(vals)
                xs = np.arange(len(CONDITIONS)) + offset
                ax.bar(xs, means, width=0.32, label=label, color=colors[key], alpha=0.88)
                for x, vals in zip(xs, dots):
                    ax.scatter([x] * len(vals), vals, color="black", s=18, zorder=3)
            ax.axhline(0, color="black", linewidth=0.8)
            if period == "P1" and dataset == "bike":
                ax.axhline(1.0, color="red", linestyle="--", linewidth=1.0, label="P1 Bike FULL90 gate 1%")
            ax.set_title(f"{period} {dataset}: positive means ALL lower loss")
            ax.set_xticks(np.arange(len(CONDITIONS)), CONDITIONS)
            ax.set_ylabel("(candidate loss - ALL loss) / F0 (%)")
            ax.grid(axis="y", alpha=0.25)
            ax.legend(fontsize=8)
    fig.suptitle("Equal trainable-count WIDE head vs existing LoRA+MLP, exposed development E")
    fig.savefig(OUT / "01_capacity_effects.png", dpi=180)
    plt.close(fig)


def plot_probe(probe_rows, probe_summary):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    markers = {"bike": "o", "household": "s"}
    colors = {"P0": "#999999", "P1": "#0072B2"}
    for ax, endpoint in zip(axes, (1, 2)):
        rows = [r for r in probe_rows if r["endpoint_index"] == endpoint]
        for period in PERIODS:
            for dataset in DATASETS:
                vals = [r for r in rows if r["period"] == period and r["dataset"] == dataset]
                ax.scatter([r["probe_g_pct_F0"] for r in vals], [r["final_wide_minus_all_pct_F0"] for r in vals],
                           label=f"{period} {dataset}", marker=markers[dataset], color=colors[period], s=42, alpha=0.85)
        lims = ax.get_xlim() + ax.get_ylim()
        low, high = min(lims), max(lims)
        ax.plot([low, high], [low, high], color="gray", linestyle=":", linewidth=1)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlim(low, high)
        ax.set_ylim(low, high)
        text = []
        for period in PERIODS:
            key = f"{period}/endpoint{endpoint}"
            item = probe_summary["probe_alignment"][key]
            text.append(f"{period}: sign={item['sign_agreement_rate']:.2f}, spearman={item['spearman_cell_mean']}")
        ax.text(0.02, 0.98, "\n".join(text), transform=ax.transAxes, va="top", ha="left",
                fontsize=9, bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8})
        ax.set_title(f"Probe endpoint {endpoint}: raw V response vs final E gap")
        ax.set_xlabel("raw V: (WIDE - ALL) / F0V (%)")
        ax.set_ylabel("final E: (WIDE - ALL) / F0E (%)")
        ax.grid(alpha=0.25)
    axes[1].legend(fontsize=8, loc="lower right")
    fig.suptitle("Short adaptation response is descriptive only; endpoints fixed before exposed E scoring")
    fig.savefig(OUT / "02_probe_alignment.png", dpi=180)
    plt.close(fig)


def plot_cost(cost_rows, cost_summary):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for endpoint, color in [(1, "#56B4E9"), (2, "#D55E00")]:
        rows = [r for r in cost_rows if r["endpoint_index"] == endpoint]
        xs = np.array([endpoint - 0.15, endpoint + 0.15])
        axes[0].bar(endpoint, np.mean([r["macro_regret_pct_F0"] for r in rows]), color=color, alpha=0.85)
        axes[0].scatter(xs, [r["macro_regret_pct_F0"] for r in rows], color="black", s=25, zorder=3)
        axes[1].bar(endpoint, np.mean([r["savings_percent"] for r in rows]), color=color, alpha=0.85)
        axes[1].scatter(xs, [r["savings_percent"] for r in rows], color="black", s=25, zorder=3)
    axes[0].axhline(0.25, color="red", linestyle="--", label="macro regret gate")
    axes[1].axhline(5.0, color="red", linestyle="--", label="savings gate")
    for ax in axes:
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks([1, 2], ["endpoint 1", "endpoint 2"])
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("macro regret vs ALWAYS_ALL (% F0)")
    axes[1].set_ylabel("optimistic reuse savings vs ALWAYS_ALL (%)")
    axes[0].set_title("Quality cost of g_k > 0 selector")
    axes[1].set_title("P1 lower-bound counterfactual cost")
    fig.suptitle("Probe selector cost uses guard elapsed and prefix UTC minus guard start; switch overhead excluded")
    fig.savefig(OUT / "03_p1_probe_cost_tradeoff.png", dpi=180)
    plt.close(fig)


def main():
    plan = load_json(RUN / "plan.json")
    analysis_contract = load_json(RUN / "analysis_contract.json")
    current_analysis_sha = sha(__file__)
    if analysis_contract.get("analysis_sha256") != current_analysis_sha:
        raise AssertionError("analysis contract hash mismatch")
    if analysis_contract.get("plan_sha256") != sha(RUN / "plan.json"):
        raise AssertionError("analysis contract plan hash mismatch")
    if "context_input_hashes" not in analysis_contract:
        raise AssertionError("analysis contract missing context_input_hashes")
    for path, digest in analysis_contract["context_input_hashes"].items():
        if sha(ROOT / path) != digest:
            raise AssertionError(f"analysis context hash mismatch: {path}")
    analysis_contract_created = parse_time(analysis_contract["created_utc"])
    done = load_json(RUN / "completed.json")
    if not done["completed"] or done["selection_sha256"] != sha(RUN / "selection.json"):
        raise AssertionError("run is incomplete or selection hash mismatch")
    verify_plan(plan)
    selections, selected0, p1_jobs, recipe_means = validate_wide_selection(plan)
    if done.get("forecasts") != 24 or done.get("wide_fits") != 36 or done.get("short_fits") != 24:
        raise AssertionError("unexpected completed run counts")

    f0, f0_meta, record_errors = load_f0(plan)
    old_all = {(e["period"], e["job"]["dataset"], e["job"]["condition"], e["job"]["seed"]): e for e in plan["old_all"]}
    old_mlp = {}
    old_mlp_hashes = {}
    for period in PERIODS:
        for key, entry in load_old_selected(period, "MLP").items():
            old_mlp[key] = entry
            for suffix in ("output/result.json", "output/predictions.npz"):
                path = ROOT / entry["eval_key"] / suffix
                if path.exists():
                    old_mlp_hashes[rel(path)] = sha(path)

    capacity_rows, capacity_map, fit_index = [], {}, {}
    for entry in selections:
        period, job, key = entry["period"], entry["job"], entry["key"]
        dataset, condition, seed = job["dataset"], job["condition"], job["seed"]
        wide_fit_path = RUN / key
        wide_fit = assert_fit_result(wide_fit_path, job, RUN / entry["contract"])
        if set(wide_fit["regimes"]) != {"EXPOSURE"} or wide_fit["regimes"]["EXPOSURE"] != entry["chosen"]:
            raise AssertionError(f"WIDE selection mismatch: {key}")
        if sha(wide_fit_path / "output/EXPOSURE.pt") != entry["chosen"]["checkpoint_sha256"]:
            raise AssertionError(f"WIDE checkpoint hash mismatch: {key}")
        wide_eval_path = RUN / key.replace("fits/", "eval/", 1) / "output"
        wide_eval, wide_score, err = load_eval_score(wide_eval_path, job, f0_meta[period][dataset])
        record_errors.append(err)
        assert_same_path(wide_eval["fit"], wide_fit_path / "output")

        all_entry = old_all[(period, dataset, condition, seed)]
        all_job = all_entry["job"]
        all_fit_path = ROOT / all_entry["key"]
        all_fit = load_json(all_fit_path / "output/result.json")
        if not all_fit["completed"] or not all_fit["frozen_verified"] or all_fit["holdout_opened"]:
            raise AssertionError(f"invalid old ALL fit: {all_entry['key']}")
        all_eval, all_score, err = load_eval_score(ROOT / all_entry["eval_key"] / "output", all_job, f0_meta[period][dataset])
        record_errors.append(err)

        mlp_score = ""
        mlp_minus_all = ""
        mlp_entry = old_mlp.get((period, dataset, condition, seed))
        if mlp_entry is not None:
            _, mlp_score_value, err = load_eval_score(ROOT / mlp_entry["eval_key"] / "output", mlp_entry["job"], f0_meta[period][dataset])
            record_errors.append(err)
            mlp_score = mlp_score_value
            mlp_minus_all = 100.0 * (mlp_score_value - all_score) / f0[period][dataset]

        wide_minus_all = 100.0 * (wide_score - all_score) / f0[period][dataset]
        row = {
            "period": period, "dataset": dataset, "condition": condition, "seed": seed,
            "wide_recipe": job["recipe"], "wide_lr": job["head_lr"], "wide_best_step": entry["chosen"]["best_step"],
            "f0_score": f0[period][dataset], "wide_score": wide_score, "all_score": all_score,
            "mlp_score": mlp_score, "wide_minus_all_pct_F0": wide_minus_all,
            "mlp_minus_all_pct_F0": mlp_minus_all,
            "wide_gain_vs_f0_pct": 100.0 * (f0[period][dataset] - wide_score) / f0[period][dataset],
            "all_gain_vs_f0_pct": 100.0 * (f0[period][dataset] - all_score) / f0[period][dataset],
            "wide_fit_guard_seconds": guard_elapsed(wide_fit_path),
            "all_fit_guard_seconds": guard_elapsed(all_fit_path),
            "wide_eval_guard_seconds": guard_elapsed(RUN / key.replace("fits/", "eval/", 1)),
        }
        capacity_rows.append(row)
        capacity_map[(period, dataset, condition, seed)] = row
        fit_index[(period, dataset, condition, seed, "WIDE")] = {"path": wide_fit_path, "result": wide_fit}
        fit_index[(period, dataset, condition, seed, "ALL_FULL")] = {"path": all_fit_path, "result": all_fit}

    capacity_effects = {}
    for period in PERIODS:
        capacity_effects[period] = {}
        for dataset in DATASETS:
            capacity_effects[period][dataset] = {}
            for condition in CONDITIONS:
                vals = [capacity_map[(period, dataset, condition, seed)]["wide_minus_all_pct_F0"] for seed in SEEDS[period]]
                capacity_effects[period][dataset][condition] = {
                    "by_seed": vals,
                    "mean": float(np.mean(vals)),
                    "both_positive": all(v > 0 for v in vals),
                }
    gate_vals = capacity_effects["P1"]["bike"]["FULL90"]["by_seed"]
    capacity_gate = {
        "passed": all(v > 0 for v in gate_vals) and float(np.mean(gate_vals)) >= plan["capacity_gate"]["mean_min_pct_F0"],
        "P1_bike_FULL90_by_seed": gate_vals,
        "P1_bike_FULL90_mean": float(np.mean(gate_vals)),
    }

    probe_rows = []
    for short_entry in plan["short_jobs"]:
        period, job, condition = short_entry["period"], short_entry["job"], short_entry["job"]["condition"]
        dataset, seed = job["dataset"], job["seed"]
        short_path = RUN / short_entry["key"]
        short_result = assert_fit_result(short_path, job, RUN / short_entry["contract"])
        original = load_json(ROOT / short_entry["old_key"] / "output/result.json")
        for h in short_result["history"]:
            if abs(float(h["score"]) - history_score(original, h["step"])) >= 1e-10:
                raise AssertionError(f"short ALL score mismatch: {short_entry['key']} step {h['step']}")
        if period == "P0":
            matches = [
                e for e in plan["wide_p0_jobs"]
                if (
                    e["job"]["dataset"], e["job"]["condition"], e["job"]["seed"], e["job"]["recipe"]
                ) == (dataset, condition, seed, 0)
            ]
            if len(matches) != 1:
                raise AssertionError(f"missing P0 recipe0 WIDE probe fit: {dataset}/{condition}/s{seed}")
            wide_probe_entry = matches[0]
            wide_key = wide_probe_entry["key"]
            expected_key = f"fits/P0/{dataset}/{condition}/r0_s{seed}"
            if wide_key != expected_key:
                raise AssertionError(f"P0 WIDE probe key mismatch: {wide_key} != {expected_key}")
            wide_result = assert_fit_result(RUN / wide_key, wide_probe_entry["job"], RUN / wide_probe_entry["contract"])
        else:
            wide_key = next(e["key"] for e in p1_jobs if (e["job"]["dataset"], e["job"]["condition"], e["job"]["seed"]) == (dataset, condition, seed))
            wide_info = fit_index[(period, dataset, condition, seed, "WIDE")]
            if wide_info["path"] != RUN / wide_key:
                raise AssertionError(f"WIDE probe path mismatch: {wide_key}")
            wide_result = wide_info["result"]
        meta = f0_meta[period][dataset]
        budgets = plan["probe_budgets"][condition]
        point0 = read_forecast_npz(short_path / "output/point_0.npz")
        if len(point0["target"]) < 30:
            raise AssertionError("validation point archive must have 30 origins")
        front = np.arange(0, 14)
        back = np.arange(16, 30)
        base_val_meta = {"target": point0["target"], "origins": None, "scale": point0["scale"], "quantiles": point0["quantiles"]}
        f0_v = point_score(short_path / "output/point_0.npz", base_val_meta)
        for endpoint_index, step in enumerate(budgets, start=1):
            point_meta = read_forecast_npz(short_path / f"output/point_{step}.npz")
            val_meta = {"target": point_meta["target"], "origins": None, "scale": point_meta["scale"], "quantiles": point_meta["quantiles"]}
            all_v = point_score(short_path / f"output/point_{step}.npz", val_meta)
            wide_v = point_score(RUN / wide_key / f"output/point_{step}.npz", val_meta)
            all_point_error = abs(all_v - history_score(short_result, step))
            wide_point_error = abs(wide_v - history_score(wide_result, step))
            record_errors.extend([all_point_error, wide_point_error])
            if all_point_error >= 1e-10:
                raise AssertionError(f"short ALL point score mismatch: {short_entry['key']} step {step}")
            if wide_point_error >= 1e-10:
                raise AssertionError(f"WIDE point score mismatch: {wide_key} step {step}")
            f0_front = point_score(short_path / "output/point_0.npz", val_meta, rows=front)
            f0_back = point_score(short_path / "output/point_0.npz", val_meta, rows=back)
            all_front = point_score(short_path / f"output/point_{step}.npz", val_meta, rows=front)
            all_back = point_score(short_path / f"output/point_{step}.npz", val_meta, rows=back)
            wide_front = point_score(RUN / wide_key / f"output/point_{step}.npz", val_meta, rows=front)
            wide_back = point_score(RUN / wide_key / f"output/point_{step}.npz", val_meta, rows=back)
            gk = 100.0 * (wide_v - all_v) / f0_v
            front_gk = 100.0 * (wide_front - all_front) / f0_front
            back_gk = 100.0 * (wide_back - all_back) / f0_back
            final_g = capacity_map[(period, dataset, condition, seed)]["wide_minus_all_pct_F0"]
            probe_rows.append({
                "period": period, "dataset": dataset, "condition": condition, "seed": seed,
                "endpoint_index": endpoint_index, "step": step,
                "update_fraction": step / (180.0 if condition == "FULL90" else 60.0),
                "f0_v_score": f0_v, "wide_v_score": wide_v, "all_v_score": all_v,
                "probe_g_pct_F0": gk, "final_wide_minus_all_pct_F0": final_g,
                "same_sign": (gk == 0 and final_g == 0) or (gk > 0 and final_g > 0) or (gk < 0 and final_g < 0),
                "absolute_error_pct_F0": abs(gk - final_g),
                "front_g_pct_F0": front_gk, "back_g_pct_F0": back_gk,
                "front_back_abs_gap_pct_F0": abs(front_gk - back_gk),
                "front_back_same_sign": (front_gk == 0 and back_gk == 0) or (front_gk > 0 and back_gk > 0) or (front_gk < 0 and back_gk < 0),
                "wide_prefix_wall_seconds": prefix_wall_seconds(RUN / wide_key, wide_result, step),
                "all_prefix_wall_seconds": prefix_wall_seconds(short_path, short_result, step),
                "all_short_guard_seconds": guard_elapsed(short_path),
            })

    alignment = {}
    for period in PERIODS:
        for endpoint_index in (1, 2):
            rows = [r for r in probe_rows if r["period"] == period and r["endpoint_index"] == endpoint_index]
            cell_probe, cell_final = [], []
            probe_seed_disagreements = []
            final_seed_disagreements = []
            for dataset in DATASETS:
                for condition in CONDITIONS:
                    vals = [r for r in rows if r["dataset"] == dataset and r["condition"] == condition]
                    cell_probe.append(float(np.mean([r["probe_g_pct_F0"] for r in vals])))
                    cell_final.append(float(np.mean([r["final_wide_minus_all_pct_F0"] for r in vals])))
                    probe_signs = {sign(r["probe_g_pct_F0"]) for r in vals}
                    final_signs = {sign(r["final_wide_minus_all_pct_F0"]) for r in vals}
                    if len(probe_signs) > 1:
                        probe_seed_disagreements.append({
                            "dataset": dataset, "condition": condition,
                            "sign_by_seed": {str(r["seed"]): sign(r["probe_g_pct_F0"]) for r in vals},
                        })
                    if len(final_signs) > 1:
                        final_seed_disagreements.append({
                            "dataset": dataset, "condition": condition,
                            "sign_by_seed": {str(r["seed"]): sign(r["final_wide_minus_all_pct_F0"]) for r in vals},
                        })
            final_positive_count = sum(sign(r["final_wide_minus_all_pct_F0"]) > 0 for r in rows)
            final_negative_count = sum(sign(r["final_wide_minus_all_pct_F0"]) < 0 for r in rows)
            final_zero_count = len(rows) - final_positive_count - final_negative_count
            alignment[f"{period}/endpoint{endpoint_index}"] = {
                "sign_agreement_rate": float(np.mean([r["same_sign"] for r in rows])),
                "always_all_sign_agreement_rate": float(final_positive_count / len(rows)),
                "always_wide_sign_agreement_rate": float(final_negative_count / len(rows)),
                "final_positive_label_count": int(final_positive_count),
                "final_negative_label_count": int(final_negative_count),
                "final_zero_label_count": int(final_zero_count),
                "total_label_count": len(rows),
                "probe_seed_direction_disagreement_count": len(probe_seed_disagreements),
                "final_seed_direction_disagreement_count": len(final_seed_disagreements),
                "probe_seed_direction_disagreements": probe_seed_disagreements,
                "final_seed_direction_disagreements": final_seed_disagreements,
                "front_back_sign_agreement_rate": float(np.mean([r["front_back_same_sign"] for r in rows])),
                "mean_abs_error_pct_F0": float(np.mean([r["absolute_error_pct_F0"] for r in rows])),
                "mean_front_back_abs_gap_pct_F0": float(np.mean([r["front_back_abs_gap_pct_F0"] for r in rows])),
                "pearson_cell_mean": corr(cell_probe, cell_final, rank=False),
                "spearman_cell_mean": corr(cell_probe, cell_final, rank=True),
                "cell_mean_probe": cell_probe,
                "cell_mean_final": cell_final,
            }

    cost_rows = []
    for endpoint_index in (1, 2):
        for seed in SEEDS["P1"]:
            cell_rows = []
            for dataset in DATASETS:
                for condition in CONDITIONS:
                    probe = next(r for r in probe_rows if (r["period"], r["dataset"], r["condition"], r["seed"], r["endpoint_index"]) == ("P1", dataset, condition, seed, endpoint_index))
                    final = capacity_map[("P1", dataset, condition, seed)]
                    choose_all = probe["probe_g_pct_F0"] > plan["probe_rule_threshold"]
                    all_full = final["all_fit_guard_seconds"]
                    wide_full = final["wide_fit_guard_seconds"]
                    selected_cost = all_full + probe["wide_prefix_wall_seconds"] if choose_all else wide_full + probe["all_prefix_wall_seconds"]
                    regret = 0.0 if choose_all else final["wide_minus_all_pct_F0"]
                    cell_rows.append({"dataset": dataset, "condition": condition, "chosen_arm": "ALL" if choose_all else "WIDE",
                                      "regret_pct_F0": regret, "selected_cost_seconds": selected_cost,
                                      "all_full_seconds": all_full})
            macro = float(np.mean([r["regret_pct_F0"] for r in cell_rows]))
            source = {dataset: float(np.mean([r["regret_pct_F0"] for r in cell_rows if r["dataset"] == dataset])) for dataset in DATASETS}
            selected_seconds = float(sum(r["selected_cost_seconds"] for r in cell_rows))
            all_seconds = float(sum(r["all_full_seconds"] for r in cell_rows))
            savings = 100.0 * (1.0 - selected_seconds / all_seconds)
            gates = plan["cost_gate"]
            quality = macro <= gates["macro_regret_max_pct_F0"] and max(source.values()) <= gates["source_regret_max_pct_F0"]
            cost = savings > gates["savings_min_percent"]
            cost_rows.append({"endpoint_index": endpoint_index, "seed": seed,
                              "macro_regret_pct_F0": macro,
                              "bike_source_regret_pct_F0": source["bike"],
                              "household_source_regret_pct_F0": source["household"],
                              "selected_cost_seconds": selected_seconds,
                              "all_full_cost_seconds": all_seconds,
                              "savings_percent": savings,
                              "quality_passed": quality, "cost_passed": cost, "passed": quality and cost,
                              "chosen_arms": ";".join(f"{r['dataset']}/{r['condition']}={r['chosen_arm']}" for r in cell_rows)})
    cost_summary = {}
    for endpoint_index in (1, 2):
        rows = [r for r in cost_rows if r["endpoint_index"] == endpoint_index]
        cost_summary[f"endpoint{endpoint_index}"] = {
            "quality_both": all(r["quality_passed"] for r in rows),
            "cost_both": all(r["cost_passed"] for r in rows),
            "passed_both": all(r["passed"] for r in rows),
            "macro_regret_by_seed": [r["macro_regret_pct_F0"] for r in rows],
            "savings_by_seed": [r["savings_percent"] for r in rows],
        }

    independent_score_max_abs_error = max(record_errors) if record_errors else 0.0
    if independent_score_max_abs_error >= 1e-10:
        raise AssertionError(f"independent NPZ score max error too large: {independent_score_max_abs_error}")

    starts = []
    for p in (RUN / "eval").glob("**/guard/status.json"):
        starts.append(parse_time(load_json(p)["started_at"]))
    if starts and (RUN / "selection.json").stat().st_mtime > min(starts):
        raise AssertionError("selection was not sealed before new WIDE E forecasts")
    if starts and analysis_contract_created > min(starts) + 1e-6:
        raise AssertionError("analysis contract was not sealed before new WIDE E forecasts")

    statuses = [load_json(p) for p in RUN.glob("**/guard/status.json")]
    if len(statuses) != plan["expected_guard_jobs"] or not all(s["completed"] and not s.get("reasons") for s in statuses):
        raise AssertionError("guard job count/status mismatch")
    guard_seconds = [float(s["elapsed_seconds"]) for s in statuses]
    raw_resources = []
    for path in RUN.glob("**/guard/resource_log.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                raw_resources.append(json.loads(line))
    resources = [r for r in raw_resources if "available_ram_gib" in r]
    required_resource_keys = {"timestamp", "available_ram_gib", "available_commit_gib", "child_tree_rss_gib", "git_process_count", "gpus"}
    gpu = []
    for sample in resources:
        missing = required_resource_keys - set(sample)
        if missing:
            raise AssertionError(f"resource sample missing keys: {sorted(missing)}")
        devices = sample.get("gpus")
        if devices is None:
            continue
        if not isinstance(devices, list):
            raise AssertionError("resource sample gpus must be a list or null")
        for device in devices:
            for key in ("memory_used_mib", "temperature_c"):
                if key not in device:
                    raise AssertionError(f"resource GPU sample missing {key}")
            gpu.append(device)
    resource_summary = {
        "guard_jobs": len(statuses),
        "resource_samples": len(resources),
        "total_guard_seconds": float(sum(guard_seconds)),
        "wide_fit_guard_seconds": float(sum(guard_elapsed(RUN / e["key"]) for e in plan["wide_p0_jobs"] + p1_jobs)),
        "short_all_guard_seconds": float(sum(guard_elapsed(RUN / e["key"]) for e in plan["short_jobs"])),
        "wide_eval_guard_seconds": float(sum(guard_elapsed(RUN / e["key"].replace("fits/", "eval/", 1)) for e in selections)),
        "smoke_guard_seconds": guard_elapsed(RUN / "smoke/WIDE"),
        "minimum_ram_gib": min((r["available_ram_gib"] for r in resources if "available_ram_gib" in r), default=None),
        "minimum_commit_gib": min((r["available_commit_gib"] for r in resources if r.get("available_commit_gib") is not None), default=None),
        "maximum_gpu_mib": max((g["memory_used_mib"] for g in gpu), default=None),
        "maximum_temperature_c": max((g["temperature_c"] for g in gpu), default=None),
    }

    summary = {
        "completed": True,
        "fresh_test": False,
        "periods": {"P0": "peft_optimization_control_v1 exposed development E", "P1": "peft_overlap_transfer_v1 exposed development E"},
        "independent_score_max_abs_error": independent_score_max_abs_error,
        "f0_scores": f0,
        "capacity_effects": capacity_effects,
        "capacity_gate": capacity_gate,
        "wide_p0_recipe_means": recipe_means,
        "probe_alignment": alignment,
        "p1_probe_rule_cost": cost_summary,
        "resources": resource_summary,
        "old_small_head_context_hashes": old_mlp_hashes,
        "analysis_contract": analysis_contract,
        "analysis_sha256": current_analysis_sha,
        "run_completed": done,
        "limitations": [
            "Both E periods are already exposed development evaluations, not independent tests.",
            "WIDE and LoRA+MLP have equal trainable counts but different function classes and FLOPs.",
            "Small-head context rows are descriptive and are not used for gates.",
            "Probe response is descriptive; no trained threshold, regression, p-value or confidence interval.",
            "P1 optimistic reuse cost excludes pause/resume/switch overhead and is not realized campaign savings.",
        ],
    }

    OUT.mkdir(parents=True, exist_ok=False)
    save_json(OUT / "summary.json", summary)
    write_csv(OUT / "capacity_metrics.csv", capacity_rows, [
        "period", "dataset", "condition", "seed", "wide_recipe", "wide_lr", "wide_best_step",
        "f0_score", "wide_score", "all_score", "mlp_score", "wide_minus_all_pct_F0",
        "mlp_minus_all_pct_F0", "wide_gain_vs_f0_pct", "all_gain_vs_f0_pct",
        "wide_fit_guard_seconds", "all_fit_guard_seconds", "wide_eval_guard_seconds",
    ])
    write_csv(OUT / "probe_metrics.csv", probe_rows, [
        "period", "dataset", "condition", "seed", "endpoint_index", "step", "update_fraction",
        "f0_v_score", "wide_v_score", "all_v_score", "probe_g_pct_F0",
        "final_wide_minus_all_pct_F0", "same_sign", "absolute_error_pct_F0",
        "front_g_pct_F0", "back_g_pct_F0", "front_back_abs_gap_pct_F0",
        "front_back_same_sign", "wide_prefix_wall_seconds", "all_prefix_wall_seconds",
        "all_short_guard_seconds",
    ])
    write_csv(OUT / "p1_probe_rule_cost.csv", cost_rows, [
        "endpoint_index", "seed", "macro_regret_pct_F0", "bike_source_regret_pct_F0",
        "household_source_regret_pct_F0", "selected_cost_seconds", "all_full_cost_seconds",
        "savings_percent", "quality_passed", "cost_passed", "passed", "chosen_arms",
    ])
    plot_capacity(capacity_rows, summary)
    plot_probe(probe_rows, summary)
    plot_cost(cost_rows, summary)
    print(json.dumps({
        "completed": True,
        "capacity_gate": capacity_gate,
        "independent_score_max_abs_error": summary["independent_score_max_abs_error"],
        "p1_probe_rule_cost": cost_summary,
        "outputs": [rel(p) for p in sorted(OUT.iterdir())],
    }, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
