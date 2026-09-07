"""Assemble the raw result matrix and the normalised scores.

Section 14 asks for the raw matrix before any interpretation, so this module
writes benchmark_results.csv exactly as measured, then adds only the two
normalisations the metric contract defines.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import paths
from .models import REGISTRY

NAIVE = "seasonal-naive"
SPECIALIST = "linear-ar-specialist"
BASELINES = (NAIVE, SPECIALIST)


def load_rows() -> pd.DataFrame:
    rows = []
    for path in sorted(paths.PRED_CACHE.glob("*.json")):
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    if not rows:
        raise SystemExit("no cached results found - run run_models first")
    return pd.DataFrame(rows)


def load_origins(model: str, track: str, task_uid: str):
    safe = task_uid.replace("::", "__").replace("/", "-")
    path = paths.PRED_CACHE / f"{model}__track{track}__{safe}__origins.npz"
    if not path.exists():
        return None
    payload = np.load(path, allow_pickle=True)
    keys = np.array([str(k) for k in payload["origin_key"]])
    windows = np.array([int(k.split("::")[0][1:]) for k in keys])
    return payload["origin_loss"].astype(np.float64), keys, windows


def recompute_reconciliation(results: pd.DataFrame) -> pd.DataFrame:
    """Re-derive the per-origin aggregate from the stored arrays.

    Written here rather than trusted from the run row so that a change to the
    aggregation rule takes effect without re-running any model.
    """
    from . import scoring

    results = results.copy()
    for index, row in results.iterrows():
        if row.get("status") != "OK":
            continue
        loaded = load_origins(row.model, row.track, row.task_uid)
        if loaded is None:
            continue
        loss, _, windows = loaded
        aggregate = scoring.task_aggregate(loss, windows)
        results.loc[index, "origin_aggregate"] = aggregate
        results.loc[index, "origin_reconciliation_gap"] = scoring.reconcile(
            aggregate, float(row.native_score)
        )
    return results


def normalise(results: pd.DataFrame) -> pd.DataFrame:
    """Add relative_to_naive, within-task rank and regret against the best deployable model.

    Ranks and regret are computed inside a (track, task) cell only, never across
    information conditions.
    """
    results = results.copy()
    results["relative_to_naive"] = np.nan
    results["task_rank"] = np.nan
    results["regret_vs_best_deployable_pct"] = np.nan
    results["best_deployable_model"] = None

    for (track, task_uid), group in results.groupby(["track", "task_uid"]):
        ok = group[group.status == "OK"]
        if ok.empty:
            continue
        naive_rows = ok[ok.model == NAIVE]
        if len(naive_rows) == 1 and np.isfinite(naive_rows.native_score.iloc[0]):
            naive_score = float(naive_rows.native_score.iloc[0])
            if naive_score > 0:
                results.loc[ok.index, "relative_to_naive"] = ok.native_score / naive_score
        results.loc[ok.index, "task_rank"] = ok.native_score.rank(method="min")

        deployable = ok[ok.model_role != "oracle"]
        if not deployable.empty:
            best_index = deployable.native_score.idxmin()
            best_score = float(deployable.native_score.loc[best_index])
            results.loc[ok.index, "best_deployable_model"] = deployable.model.loc[best_index]
            if best_score > 0:
                results.loc[ok.index, "regret_vs_best_deployable_pct"] = (
                    100.0 * (ok.native_score - best_score) / best_score
                )
    return results


def attach_task_metadata(results: pd.DataFrame) -> pd.DataFrame:
    pool = pd.read_csv(paths.RESULTS / "task_pool.csv")
    selected = json.loads((paths.RESULTS / "selected_tasks.json").read_text(encoding="utf-8"))
    split = {task: "discovery" for task in selected["discovery_tasks"]}
    split.update({task: "confirmation" for task in selected["confirmation_tasks"]})
    keep = [
        "task_uid",
        "domain",
        "freq",
        "freq_bucket",
        "horizon",
        "num_windows",
        "seasonality",
        "median_length",
        "n_series",
        "n_targets",
        "n_known_cov",
        "n_past_cov",
        "is_multivariate",
        "has_known_cov",
        "has_past_cov",
        "horizon_to_median_length",
        "dataset_config",
    ]
    merged = results.merge(pool[keep], on="task_uid", how="left")
    merged["split"] = merged.task_uid.map(split)
    audit = pd.read_csv(paths.RESULTS / "model_audit.csv")
    families = dict(zip(audit.model_id, audit.architecture_family))
    contamination = dict(zip(audit.model_id, audit.contamination_status_fev_bench))
    merged["architecture_family"] = merged.model.map(families).fillna("baseline")
    merged["contamination_status"] = merged.model.map(contamination).fillna("NOT_APPLICABLE")
    merged["model_revision"] = merged.model.map(dict(zip(audit.model_id, audit.resolved_revision)))
    return merged


def main() -> None:
    results = attach_task_metadata(normalise(recompute_reconciliation(load_rows())))
    ordered = [
        "benchmark",
        "track",
        "split",
        "task_uid",
        "dataset_config",
        "domain",
        "freq",
        "freq_bucket",
        "model",
        "model_role",
        "architecture_family",
        "model_revision",
        "contamination_status",
        "native_metric",
        "native_score",
        "relative_to_naive",
        "task_rank",
        "regret_vs_best_deployable_pct",
        "best_deployable_model",
        "context_mode",
        "actual_context_min",
        "actual_context_median",
        "actual_context_max",
        "future_covariates_used",
        "multivariate_mode",
        "n_windows",
        "n_targets",
        "n_origins",
        "origin_aggregate",
        "origin_reconciliation_gap",
        "inference_seconds",
        "peak_memory_mb",
        "status",
    ]
    present = [column for column in ordered if column in results.columns]
    rest = [column for column in results.columns if column not in present]
    results = results[present + rest].sort_values(["track", "split", "task_uid", "model"])
    out = paths.RESULTS / "benchmark_results.csv"
    results.to_csv(out, index=False)
    print(f"wrote {out}  ({len(results)} rows)")

    baseline_rows = results[results.model.isin(BASELINES)]
    baseline_rows.to_csv(paths.RESULTS / "baseline_results.csv", index=False)

    failures = results[results.status != "OK"]
    if len(failures):
        print(f"\n{len(failures)} non-OK rows:")
        print(failures[["track", "task_uid", "model", "status"]].to_string(index=False))
    worst_gap = results.origin_reconciliation_gap.max()
    print(f"\nworst origin/official reconciliation gap: {worst_gap:.3e}")
    for track, group in results[results.status == "OK"].groupby("track"):
        print(f"\n=== TRACK {track} ===")
        pivot = group.pivot_table(
            index="task_uid", columns="model", values="relative_to_naive", aggfunc="first"
        )
        print(pivot.round(3).to_string())


if __name__ == "__main__":
    main()
