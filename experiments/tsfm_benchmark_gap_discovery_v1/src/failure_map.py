"""Where do the primary models lose ground, and do they lose it together?

For each descriptor bucket and each model: the median relative error against the
naive anchor, the median regret against the best deployable model in the task,
and how the bucket compares with that model's own overall median. A condition is
a common-failure candidate only if the thresholds registered in study.yaml are
met - Section 16.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import yaml

from . import paths
from .descriptors import DESCRIPTOR_LABELS
from .models import DIAGNOSTIC_MODELS, PRIMARY_MODELS

DESCRIPTOR_COLUMNS = {
    "D1": "D1_horizon_ratio",
    "D2": "D2_frequency",
    "D3": "D3_dimensionality",
    "D4": "D4_future_covariates",
    "D5": "D5_past_covariates",
    "D6": "D6_zero_fraction",
    "D7": "D7_missingness",
    "D8": "D8_shift",
    "D9": "D9_seasonal_strength",
}


def thresholds() -> dict:
    config = yaml.safe_load((paths.CONFIGS / "study.yaml").read_text(encoding="utf-8"))
    return config["thresholds"]["failure_gate"]


def build(track: str = "U", split: str = "discovery") -> tuple[pd.DataFrame, pd.DataFrame]:
    results = pd.read_csv(paths.RESULTS / "benchmark_results.csv")
    metadata = pd.read_csv(paths.RESULTS / "task_metadata.csv")
    results = results[
        (results.track == track) & (results.split == split) & (results.status == "OK")
    ]
    merged = results.merge(
        metadata[["task_uid", *DESCRIPTOR_COLUMNS.values()]], on="task_uid", how="left"
    )

    overall = (
        merged[merged.model.isin(PRIMARY_MODELS)]
        .groupby("model")
        .agg(
            overall_median_relative=("relative_to_naive", "median"),
            overall_median_regret=("regret_vs_best_deployable_pct", "median"),
            n_tasks=("task_uid", "nunique"),
        )
    )

    rows = []
    models = PRIMARY_MODELS + DIAGNOSTIC_MODELS
    for descriptor, column in DESCRIPTOR_COLUMNS.items():
        for bucket, bucket_rows in merged.groupby(column):
            tasks = sorted(bucket_rows.task_uid.unique())
            for model in models:
                model_rows = bucket_rows[bucket_rows.model == model]
                if model_rows.empty:
                    continue
                reference = merged[(merged.model == model) & (merged[column] != bucket)]
                bucket_median = float(model_rows.relative_to_naive.median())
                reference_median = (
                    float(reference.relative_to_naive.median()) if len(reference) else np.nan
                )
                own_overall = (
                    float(overall.overall_median_relative.get(model, np.nan))
                    if model in overall.index
                    else np.nan
                )
                gap_pct = (
                    100.0 * (bucket_median - reference_median) / reference_median
                    if np.isfinite(reference_median) and reference_median > 0
                    else np.nan
                )
                rows.append(
                    {
                        "track": track,
                        "split": split,
                        "descriptor": descriptor,
                        "descriptor_label": DESCRIPTOR_LABELS[descriptor],
                        "bucket": bucket,
                        "model": model,
                        "model_role": "primary" if model in PRIMARY_MODELS else "diagnostic",
                        "n_tasks_in_bucket": len(tasks),
                        "tasks": json.dumps(tasks),
                        "median_relative_to_naive": bucket_median,
                        "reference_median_relative_to_naive": reference_median,
                        "condition_gap_pct": gap_pct,
                        "median_regret_pct": float(model_rows.regret_vs_best_deployable_pct.median()),
                        "own_overall_median_relative": own_overall,
                        "worse_than_own_overall": bool(bucket_median > own_overall)
                        if np.isfinite(own_overall)
                        else None,
                        "wins_in_bucket": int((model_rows.task_rank == 1).sum()),
                    }
                )
    failure_map = pd.DataFrame(rows)

    gate = thresholds()
    candidates = []
    for (descriptor, bucket), group in failure_map[
        failure_map.model_role == "primary"
    ].groupby(["descriptor", "bucket"]):
        n_tasks = int(group.n_tasks_in_bucket.iloc[0])
        affected = group[group.condition_gap_pct >= gate["min_condition_gap_pct"]]
        families = sorted(
            set(
                pd.read_csv(paths.RESULTS / "model_audit.csv")
                .set_index("model_id")
                .loc[affected.model.tolist(), "architecture_family"]
            )
        ) if len(affected) else []
        median_regret = float(group.median_regret_pct.median())
        passes = (
            n_tasks >= gate["min_discovery_affected_tasks"]
            and len(families) >= gate["min_model_families"]
            and float(affected.condition_gap_pct.median() if len(affected) else np.nan)
            >= gate["min_condition_gap_pct"]
            and median_regret >= gate["min_regret_vs_comparator_pct"]
        )
        candidates.append(
            {
                "descriptor": descriptor,
                "descriptor_label": DESCRIPTOR_LABELS[descriptor],
                "bucket": bucket,
                "n_tasks_in_bucket": n_tasks,
                "affected_models": json.dumps(sorted(affected.model.tolist())),
                "n_affected_models": len(affected),
                "affected_families": json.dumps(families),
                "n_affected_families": len(families),
                "median_condition_gap_pct": float(affected.condition_gap_pct.median())
                if len(affected)
                else np.nan,
                "median_regret_pct": median_regret,
                "gate_min_tasks": n_tasks >= gate["min_discovery_affected_tasks"],
                "gate_min_families": len(families) >= gate["min_model_families"],
                "gate_min_gap": bool(
                    len(affected)
                    and float(affected.condition_gap_pct.median()) >= gate["min_condition_gap_pct"]
                ),
                "gate_min_regret": median_regret >= gate["min_regret_vs_comparator_pct"],
                "passes_failure_gate": bool(passes),
                "verdict": "CANDIDATE" if passes else ("CASE_ONLY" if n_tasks < gate["min_discovery_affected_tasks"] else "NOT_A_CANDIDATE"),
            }
        )
    return failure_map, pd.DataFrame(candidates)


def raw_task_effects(track: str = "U", split: str = "discovery") -> pd.DataFrame:
    """Per-task relative errors, so small buckets are read as effects, not as a CI.

    Section 26 asks for raw task effects rather than a formal interval when the
    number of tasks in a condition is small, which it is throughout here.
    """
    results = pd.read_csv(paths.RESULTS / "benchmark_results.csv")
    metadata = pd.read_csv(paths.RESULTS / "task_metadata.csv")
    results = results[
        (results.track == track) & (results.split == split) & (results.status == "OK")
    ]
    merged = results.merge(
        metadata[["task_uid", *DESCRIPTOR_COLUMNS.values()]], on="task_uid", how="left"
    )
    return merged.pivot_table(
        index=[*DESCRIPTOR_COLUMNS.values(), "task_uid"],
        columns="model",
        values="relative_to_naive",
        aggfunc="first",
    ).reset_index()


def write_markdown(failure_map: pd.DataFrame, candidates: pd.DataFrame, gate: dict) -> str:
    lines = [
        "# Failure map - TRACK U, discovery split",
        "",
        "Numbers first. `relative_to_naive` is the task's native SQL divided by the",
        "SeasonalNaive SQL on the same task, so 1.0 is the naive anchor and lower is better.",
        "`condition_gap_pct` compares a model's median inside the bucket with its median on",
        "every discovery task outside the bucket; positive means worse inside.",
        "",
        "## Candidate gate",
        "",
        f"A condition is promoted only with at least {gate['min_discovery_affected_tasks']} discovery tasks, "
        f"at least {gate['min_model_families']} affected architecture families, a median condition gap of "
        f"at least {gate['min_condition_gap_pct']}%, and a median regret against the best deployable model "
        f"of at least {gate['min_regret_vs_comparator_pct']}%.",
        "",
        candidates.sort_values(
            ["passes_failure_gate", "median_condition_gap_pct"], ascending=[False, False]
        )
        .drop(columns=["affected_models", "affected_families"])
        .to_markdown(index=False),
        "",
        "## Per-bucket detail",
        "",
    ]
    lines += [
        "Bucket sizes here are three to eight tasks. Section 26 asks for the raw task effects at",
        "that size rather than an interval built from them, so every per-task number is listed",
        "below the bucket medians.",
        "",
    ]
    for descriptor in sorted(failure_map.descriptor.unique()):
        subset = failure_map[failure_map.descriptor == descriptor]
        label = subset.descriptor_label.iloc[0]
        lines += [f"### {descriptor} - {label}", ""]
        pivot = subset.pivot_table(
            index=["bucket", "n_tasks_in_bucket"],
            columns="model",
            values="median_relative_to_naive",
        )
        lines += [pivot.round(3).to_markdown(), ""]
    return "\n".join(lines)


def main() -> None:
    failure_map, candidates = build()
    failure_map.to_csv(paths.RESULTS / "failure_map.csv", index=False)
    candidates.to_csv(paths.RESULTS / "failure_candidates_gate.csv", index=False)
    markdown = write_markdown(failure_map, candidates, thresholds())
    (paths.RESULTS / "failure_map.md").write_text(markdown, encoding="utf-8")
    print(f"wrote {paths.RESULTS / 'failure_map.csv'} ({len(failure_map)} rows)")
    passing = candidates[candidates.passes_failure_gate]
    print(f"\nconditions passing the failure gate: {len(passing)}")
    if len(passing):
        print(
            passing[
                [
                    "descriptor",
                    "bucket",
                    "n_tasks_in_bucket",
                    "n_affected_models",
                    "n_affected_families",
                    "median_condition_gap_pct",
                    "median_regret_pct",
                ]
            ].to_string(index=False)
        )
    else:
        print("NO_STRONG_GAP_FOUND at the failure-gate stage")
    print("\nTop conditions by median condition gap:")
    print(
        candidates.sort_values("median_condition_gap_pct", ascending=False)
        .head(10)[
            [
                "descriptor",
                "bucket",
                "n_tasks_in_bucket",
                "n_affected_families",
                "median_condition_gap_pct",
                "median_regret_pct",
                "verdict",
            ]
        ]
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
