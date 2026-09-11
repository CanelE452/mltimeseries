"""Post-hoc descriptive diagnostics for Study35 completion.

Reads existing Study35 outputs only.  It does not train, forecast, download, or
modify frozen run artifacts.  The fixed result files are created with exclusive
open so an existing diagnostic is never overwritten.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "runs" / "peft_head_convergence_v1"
OUT = ROOT / "results" / "peft_head_convergence_v1"
JSON_OUT = OUT / "descriptive_decomposition.json"
FIG_VD = OUT / "06_validation_development.png"
FIG_ORIGIN = OUT / "07_target_origin_diagnostics.png"
FAMILIES = ("HEAD", "WIDE", "JOINT")
COLORS = {"HEAD": "#cc6633", "WIDE": "#3269aa", "JOINT": "#087e77"}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def no_overwrite_targets() -> None:
    existing = [p for p in (JSON_OUT, FIG_VD, FIG_ORIGIN) if p.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite existing diagnostic outputs: {existing}")


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def eq_nan(a: np.ndarray, b: np.ndarray) -> bool:
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        return False
    if a.dtype.kind in "fc" or b.dtype.kind in "fc":
        return bool(np.array_equal(a, b, equal_nan=True))
    return bool(np.array_equal(a, b))


def pinball_components(prediction: np.ndarray, target: np.ndarray, quantiles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    prediction = np.sort(prediction.astype(np.float64), axis=2)
    target = target.astype(np.float64)
    quantiles = quantiles.astype(np.float64)
    error = target[:, :, None, :] - prediction
    q = quantiles[None, None, :, None]
    valid = np.isfinite(target)
    numerator = np.where(valid[:, :, None, :], 2.0 * np.maximum(q * error, (q - 1.0) * error), 0.0).sum(axis=3)
    counts = valid.sum(axis=2).astype(np.float64)
    return numerator, counts


def score_arrays(prediction: np.ndarray, target: np.ndarray, quantiles: np.ndarray, scale: np.ndarray) -> float:
    numerator, counts = pinball_components(prediction, target, quantiles)
    if np.any(counts.sum(axis=0) <= 0) or np.any(scale <= 0):
        raise ValueError("empty target count or nonpositive scale")
    return float((numerator.sum(axis=0) / counts.sum(axis=0)[:, None] / scale[:, None]).mean())


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key].copy() for key in z.files}


def load_metrics() -> list[dict]:
    with (OUT / "metrics.csv").open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    parsed = []
    for row in rows:
        item = dict(row)
        item["seed"] = int(item["seed"])
        item["D_score"] = float(item["D_score"])
        item["F0_score"] = float(item["F0_score"])
        item["D_over_F0"] = float(item["D_over_F0"])
        item["step"] = int(item["step"])
        item["V"] = None if item["V"] in ("", "None") else float(item["V"])
        item["recipe"] = None if item["recipe"] in ("", "None") else int(item["recipe"])
        parsed.append(item)
    return parsed


def metric_index(rows: list[dict]) -> dict[tuple[str, str, str], dict]:
    return {(row["cell"], row["budget"], row["family"]): row for row in rows}


def prepared_eval_reference(plan: dict, dataset: str) -> dict:
    spec = plan["data"][dataset]
    holdout = load_npz(ROOT / spec["holdout_path"])
    origins = holdout["eval_origins"][:: plan["eval_stride"]]
    target_indices = holdout["target_indices"].astype(np.int64)
    horizon = int(holdout["horizon"])
    target = np.stack([holdout["target_values"][int(origin) : int(origin) + horizon][:, target_indices].T for origin in origins])
    timestamps = holdout["timestamps"]
    channels = holdout["channels"]
    return {
        "origins": origins,
        "target": target,
        "quantiles": holdout["quantiles"],
        "scale": holdout["fit_std"][target_indices],
        "origin_timestamps": [str(timestamps[int(origin)]) for origin in origins],
        "target_columns": [str(channels[int(i)]) for i in target_indices],
    }


def validation_development_rows(plan: dict, selection: dict, rows: list[dict]) -> list[dict]:
    metrics = metric_index(rows)
    output = []
    for cell, cell_payload in sorted(selection["cells"].items()):
        for family in FAMILIES:
            selected = cell_payload["L720"][family]
            result = read_json(RUN / "fit" / selected["key"] / "output" / "result.json")
            metric = metrics[(cell, "L720", family)]
            history_by_step = {int(item["step"]): item for item in result["history"]}
            selected_history = history_by_step[int(selected["step"])]
            initial_train = float(result["history"][0]["train_score"])
            selected_train = float(selected_history["train_score"])
            output.append(
                {
                    "cell": cell,
                    "dataset": metric["dataset"],
                    "seed": int(metric["seed"]),
                    "family": family,
                    "recipe": int(selected["recipe"]),
                    "selected_step": int(selected["step"]),
                    "initial_V": float(result["initial_V"]),
                    "selected_V": float(selected["V"]),
                    "validation_gain_pct_initial": 100.0 * (float(result["initial_V"]) - float(selected["V"])) / float(result["initial_V"]),
                    "initial_train_score": initial_train,
                    "selected_train_score": selected_train,
                    "train_gain_pct_initial": 100.0 * (initial_train - selected_train) / initial_train,
                    "D_score": float(metric["D_score"]),
                    "F0_score": float(metric["F0_score"]),
                    "development_gain_pct_F0": 100.0 * (float(metric["F0_score"]) - float(metric["D_score"])) / float(metric["F0_score"]),
                    "D_over_F0": float(metric["D_over_F0"]),
                    "selected_key": selected["key"],
                }
            )
    return output


def decompose_joint_vs_f0(plan: dict, selection: dict, rows: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    metrics = metric_index(rows)
    target_rows = []
    origin_rows = []
    cell_rows = []
    for cell, cell_payload in sorted(selection["cells"].items()):
        dataset = cell_payload["dataset"]
        ref = prepared_eval_reference(plan, dataset)
        joint = cell_payload["L720"]["JOINT"]
        head = cell_payload["L720"]["HEAD"]
        joint_npz = load_npz(RUN / "forecast" / joint["key"] / "L720" / "output" / "selected.npz")
        f0_npz = load_npz(RUN / "forecast" / head["key"] / "L720" / "output" / "F0.npz")
        for label, item in (("JOINT", joint_npz), ("F0", f0_npz)):
            for key in ("target", "quantiles", "scale", "origins"):
                expected = ref["origins"] if key == "origins" else ref[key]
                if not eq_nan(item[key], expected):
                    raise AssertionError(f"{cell} {label} {key} does not match prepared holdout eval reference")
        if not (eq_nan(joint_npz["target"], f0_npz["target"]) and eq_nan(joint_npz["quantiles"], f0_npz["quantiles"]) and eq_nan(joint_npz["scale"], f0_npz["scale"]) and np.array_equal(joint_npz["origins"], f0_npz["origins"])):
            raise AssertionError(f"{cell} JOINT and F0 forecast payloads are not aligned")

        j_num, j_count = pinball_components(joint_npz["prediction"], joint_npz["target"], joint_npz["quantiles"])
        f_num, f_count = pinball_components(f0_npz["prediction"], f0_npz["target"], f0_npz["quantiles"])
        if not np.array_equal(j_count, f_count):
            raise AssertionError(f"{cell} JOINT/F0 valid target counts differ")
        scale = joint_npz["scale"].astype(np.float64)
        quantiles = joint_npz["quantiles"].astype(np.float64)
        channel_count = len(scale)
        q_count = len(quantiles)
        total_counts = j_count.sum(axis=0)
        if np.any(total_counts <= 0):
            raise AssertionError(f"{cell} empty target denominator")
        j_score = score_arrays(joint_npz["prediction"], joint_npz["target"], joint_npz["quantiles"], scale)
        f_score = score_arrays(f0_npz["prediction"], f0_npz["target"], f0_npz["quantiles"], scale)
        metric_joint = metrics[(cell, "L720", "JOINT")]
        metric_f0 = metrics[(cell, "L720", "F0")]
        if abs(j_score - metric_joint["D_score"]) > 1e-10 or abs(f_score - metric_f0["D_score"]) > 1e-10:
            raise AssertionError(f"{cell} recomputed JOINT/F0 scores disagree with metrics.csv")
        macro_gap = j_score - f_score
        per_target_contrib = []
        for target_index, target_name in enumerate(ref["target_columns"]):
            j_t = float((j_num[:, target_index, :].sum(axis=0) / total_counts[target_index] / scale[target_index]).mean())
            f_t = float((f_num[:, target_index, :].sum(axis=0) / total_counts[target_index] / scale[target_index]).mean())
            contribution = (j_t - f_t) / channel_count
            row = {
                "cell": cell,
                "dataset": dataset,
                "seed": int(cell_payload["seed"]),
                "target_index": target_index,
                "target": target_name,
                "joint_target_score": j_t,
                "f0_target_score": f_t,
                "target_score_gap_joint_minus_f0": j_t - f_t,
                "macro_gap_contribution": contribution,
                "macro_gap_contribution_pct_F0": 100.0 * contribution / f_score,
            }
            target_rows.append(row)
            per_target_contrib.append(contribution)

        origin_contrib = ((j_num - f_num) / total_counts[None, :, None] / scale[None, :, None]).sum(axis=(1, 2)) / (channel_count * q_count)
        if abs(float(origin_contrib.sum()) - macro_gap) > 1e-10:
            raise AssertionError(f"{cell} origin decomposition does not sum to macro gap")
        if abs(float(sum(per_target_contrib)) - macro_gap) > 1e-10:
            raise AssertionError(f"{cell} target decomposition does not sum to macro gap")
        positive = origin_contrib > 0
        timestamps = ref["origin_timestamps"]
        for i, value in enumerate(origin_contrib):
            origin_rows.append(
                {
                    "cell": cell,
                    "dataset": dataset,
                    "seed": int(cell_payload["seed"]),
                    "origin_order": i,
                    "origin_index": int(joint_npz["origins"][i]),
                    "origin_timestamp": timestamps[i],
                    "macro_gap_contribution": float(value),
                    "macro_gap_contribution_pct_F0": 100.0 * float(value) / f_score,
                    "joint_worse_than_f0": bool(value > 0),
                }
            )
        worst = np.argsort(origin_contrib)[-5:][::-1]
        cell_rows.append(
            {
                "cell": cell,
                "dataset": dataset,
                "seed": int(cell_payload["seed"]),
                "f0_score": f_score,
                "joint_score": j_score,
                "macro_gap_joint_minus_f0": macro_gap,
                "macro_gap_pct_F0": 100.0 * macro_gap / f_score,
                "target_contribution_sum": float(sum(per_target_contrib)),
                "origin_contribution_sum": float(origin_contrib.sum()),
                "origin_deterioration_count": int(positive.sum()),
                "origin_count": int(len(origin_contrib)),
                "worst_origins": [
                    {
                        "origin_order": int(i),
                        "origin_timestamp": timestamps[int(i)],
                        "macro_gap_contribution_pct_F0": 100.0 * float(origin_contrib[int(i)]) / f_score,
                    }
                    for i in worst
                ],
            }
        )
    return target_rows, origin_rows, cell_rows


def plot_validation_development(rows: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 6.2), constrained_layout=True)
    markers = {"bmra": "o", "jena": "s"}
    xs = [row["validation_gain_pct_initial"] for row in rows]
    ys = [row["development_gain_pct_F0"] for row in rows]
    manual_offsets = {
        ("jena", 30000, "HEAD"): (-62, -10),
        ("jena", 30001, "WIDE"): (8, 10),
        ("bmra", 30000, "HEAD"): (-86, -18),
        ("bmra", 30000, "WIDE"): (-86, 12),
        ("bmra", 30000, "JOINT"): (-74, 4),
        ("bmra", 30001, "HEAD"): (-78, -4),
        ("bmra", 30001, "WIDE"): (-78, 4),
        ("bmra", 30001, "JOINT"): (6, 4),
    }
    for family in FAMILIES:
        subset = [row for row in rows if row["family"] == family]
        for row in subset:
            ax.scatter(
                row["validation_gain_pct_initial"],
                row["development_gain_pct_F0"],
                s=80,
                marker=markers[row["dataset"]],
                color=COLORS[family],
                edgecolor="black",
                linewidth=0.7,
                alpha=0.9,
            )
            key = (row["dataset"], row["seed"], family)
            offset = manual_offsets.get(key, (6, 4))
            ha = "right" if offset[0] < 0 else "left"
            ax.annotate(
                f"{family[0]} {row['cell'].replace('/s', ' s')}",
                (row["validation_gain_pct_initial"], row["development_gain_pct_F0"]),
                textcoords="offset points",
                xytext=offset,
                fontsize=7,
                ha=ha,
            )
    ax.axhline(0, color="black", linewidth=1)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlim(min(0.0, min(xs) - 0.35), max(xs) + 0.55)
    ax.set_ylim(min(ys) - 0.9, max(0.5, max(ys) + 0.9))
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("Selected L720 validation improvement vs initial V (%)")
    ax.set_ylabel("Development D improvement vs F0 (%)")
    ax.set_title("Study35 selected models: validation improvement did not transfer to D")
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS[f], markeredgecolor="black", label=f, markersize=8)
        for f in FAMILIES
    ] + [
        plt.Line2D([0], [0], marker=markers[d], color="black", linestyle="", label=d, markersize=7)
        for d in ("bmra", "jena")
    ]
    ax.legend(handles=handles, loc="best", fontsize=8)
    fig.savefig(FIG_VD, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_target_origin(target_rows: list[dict], origin_rows: list[dict]) -> None:
    cells = sorted({row["cell"] for row in target_rows})
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), constrained_layout=True)
    ax = axes[0]
    width = 0.35
    target_names_by_dataset = {}
    for row in target_rows:
        target_names_by_dataset.setdefault(row["dataset"], {})[row["target_index"]] = row["target"]
    for target_index in (0, 1):
        vals = []
        for cell in cells:
            row = next(item for item in target_rows if item["cell"] == cell and item["target_index"] == target_index)
            vals.append(row["macro_gap_contribution_pct_F0"])
        x = np.arange(len(cells)) + (target_index - 0.5) * width
        bmra_name = target_names_by_dataset.get("bmra", {}).get(target_index, "?")
        jena_name = target_names_by_dataset.get("jena", {}).get(target_index, "?")
        ax.bar(x, vals, width=width, label=f"target {target_index}: BMRA {bmra_name}; Jena {jena_name}")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(np.arange(len(cells)))
    ax.set_xticklabels([cell.replace("/s", " s") for cell in cells], rotation=15, ha="right")
    ax.set_ylabel("Contribution to JOINT-F0 macro gap (%F0)")
    ax.set_title("Target-level contributions; positive means JOINT worse than F0")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[1]
    for cell in cells:
        subset = [row for row in origin_rows if row["cell"] == cell]
        subset.sort(key=lambda row: row["origin_order"])
        dataset = subset[0]["dataset"]
        seed = subset[0]["seed"]
        label = f"{dataset} s{seed} ({subset[0]['origin_timestamp'][:10]}..{subset[-1]['origin_timestamp'][:10]})"
        ax.plot(
            [row["origin_order"] for row in subset],
            [row["macro_gap_contribution_pct_F0"] for row in subset],
            marker="o",
            linewidth=1.4,
            markersize=4,
            label=label,
        )
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(np.arange(0, 20, 2))
    ax.set_xlabel("Eval origin order 0..19, using prepared eval_origins[::4]")
    ax.set_ylabel("Origin contribution to JOINT-F0 macro gap (%F0)")
    ax.set_title("Origin-level deterioration pattern; positive means JOINT worse than F0")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.savefig(FIG_ORIGIN, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-plots", action="store_true", help="preserve JSON and regenerate only the two explanatory PNGs")
    args = parser.parse_args()
    if args.refresh_plots:
        payload = read_json(JSON_OUT)
        plot_validation_development(payload["validation_development_rows"])
        plot_target_origin(payload["joint_vs_f0_target_contributions"], payload["joint_vs_f0_origin_rows"])
        for png in (FIG_VD, FIG_ORIGIN):
            if not png.exists() or png.stat().st_size <= 0:
                raise AssertionError(f"plot was not created: {png}")
        print(json.dumps({"refreshed": [FIG_VD.as_posix(), FIG_ORIGIN.as_posix()], "json_preserved": JSON_OUT.as_posix()}, indent=2), flush=True)
        return 0

    no_overwrite_targets()
    plan = read_json(RUN / "plan.json")
    completed = read_json(RUN / "completed.json")
    if not completed.get("completed") or completed.get("stage_B_executed"):
        raise AssertionError("expected completed Study35 run with Stage B skipped")
    selection = read_json(RUN / "selection_A.json")
    summary_a = read_json(OUT / "summary_A.json")
    rows = load_metrics()
    val_rows = validation_development_rows(plan, selection, rows)
    target_rows, origin_rows, cell_rows = decompose_joint_vs_f0(plan, selection, rows)
    plot_validation_development(val_rows)
    plot_target_origin(target_rows, origin_rows)
    for png in (FIG_VD, FIG_ORIGIN):
        if not png.exists() or png.stat().st_size <= 0:
            raise AssertionError(f"plot was not created: {png}")
    family_means = {}
    for family in FAMILIES:
        vals = [row["D_over_F0"] for row in rows if row["budget"] == "L720" and row["family"] == family]
        family_means[family] = float(np.mean(vals))
    consistency = {
        "all_selected_validation_gain_positive": all(row["validation_gain_pct_initial"] > 0 for row in val_rows),
        "all_selected_train_gain_positive": all(row["train_gain_pct_initial"] > 0 for row in val_rows),
        "all_selected_development_gain_negative": all(row["development_gain_pct_F0"] < 0 for row in val_rows),
        "max_abs_target_sum_minus_macro_gap": float(max(abs(row["target_contribution_sum"] - row["macro_gap_joint_minus_f0"]) for row in cell_rows)),
        "max_abs_origin_sum_minus_macro_gap": float(max(abs(row["origin_contribution_sum"] - row["macro_gap_joint_minus_f0"]) for row in cell_rows)),
        "png_outputs": {path.name: {"exists": path.exists(), "bytes": path.stat().st_size if path.exists() else 0} for path in (FIG_VD, FIG_ORIGIN)},
    }
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "posthoc descriptive diagnostic on existing Study35 outputs only",
        "limitations": [
            "post-hoc descriptive decomposition; not used for model selection",
            "D is a development diagnostic, not an independent final test",
            "two seeds per source share the same target labels/origins, so seed rows are dependent views of the same source period",
            "loss decomposition localizes where JOINT worsened versus F0 but does not establish shift causality",
        ],
        "run_completed": completed,
        "G1": summary_a["gates"]["G1"],
        "L720_family_mean_D_over_F0": family_means,
        "validation_development_rows": val_rows,
        "joint_vs_f0_target_contributions": target_rows,
        "joint_vs_f0_origin_rows": origin_rows,
        "joint_vs_f0_cell_summary": cell_rows,
        "consistency": consistency,
        "outputs": {
            "validation_development_png": FIG_VD.as_posix(),
            "target_origin_diagnostics_png": FIG_ORIGIN.as_posix(),
        },
    }
    with JSON_OUT.open("x", encoding="utf-8") as stream:
        json.dump(jsonable(payload), stream, indent=2, ensure_ascii=False, allow_nan=False)
    print(json.dumps({"written": [JSON_OUT.as_posix(), FIG_VD.as_posix(), FIG_ORIGIN.as_posix()], "family_means": family_means, "consistency": consistency}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
