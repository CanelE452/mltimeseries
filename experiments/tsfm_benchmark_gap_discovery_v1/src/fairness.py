"""The information-condition table (Section 27).

One row per (model, track, benchmark), stating exactly what that model was given
and whether its number may be placed beside the others in a pooled ranking.
"""

from __future__ import annotations

import pandas as pd

from . import paths
from .baselines import BASELINES
from .models import REGISTRY
from .tracks import track_description


def build() -> pd.DataFrame:
    results = pd.read_csv(paths.RESULTS / "benchmark_results.csv")
    audit = pd.read_csv(paths.RESULTS / "model_audit.csv").set_index("model_id")
    rows = []
    for (model, track), group in results[results.status == "OK"].groupby(["model", "track"]):
        is_baseline = model in BASELINES
        info = audit.loc[model] if model in audit.index else None
        trained_here = model == "linear-ar-specialist"
        contamination = (
            info.contamination_status_fev_bench if info is not None else "NOT_APPLICABLE"
        )
        comparability = "DIRECT"
        reason = "same targets, same windows, same metric, same information condition"
        if trained_here:
            comparability = "PARTIAL"
            reason = (
                "fitted on each task's visible history, so it is a supervised specialist and not "
                "zero-shot; compared as a headroom anchor, never ranked against zero-shot models "
                "as if the information condition were identical"
            )
        elif model == "chronos-2-synth":
            comparability = "PARTIAL"
            reason = (
                "synthetic-only pretraining makes it a contamination anchor rather than a "
                "leaderboard entry; kept out of the primary gap map"
            )
        rows.append(
            {
                "benchmark": "fev-bench",
                "model": model,
                "track": track,
                "track_information_condition": track_description(track),
                "role": REGISTRY.get(model, {}).get("role", "baseline"),
                "target_info": "all target columns of the task"
                if track != "U"
                else "one target column at a time",
                "past_covariates_used": False,
                "future_covariates_used": bool(group.future_covariates_used.iloc[0]),
                "known_columns_used": group.get("known_columns_used", pd.Series(["[]"])).iloc[0],
                "known_columns_skipped": group.get("known_columns_skipped", pd.Series(["[]"])).iloc[0],
                "multivariate": track == "M",
                "context": f"{group.context_mode.iloc[0]}; median {int(group.actual_context_median.median())} steps",
                "adaptation": "task-fitted" if trained_here else "zero-shot, no fine-tuning",
                "target_training_data_used": "visible history of the same task"
                if trained_here
                else "none",
                "benchmark_specific_checkpoint": (
                    info.benchmark_specific_checkpoint if info is not None else None
                ),
                "contamination_status": contamination,
                "metric": group.native_metric.iloc[0],
                "evaluator": "fev native",
                "comparability_status": comparability,
                "comparability_reason": reason,
                "n_tasks": int(group.task_uid.nunique()),
            }
        )
    return pd.DataFrame(rows).sort_values(["track", "model"])


def main() -> None:
    matrix = build()
    matrix.to_csv(paths.RESULTS / "fairness_matrix.csv", index=False)
    print(f"wrote {paths.RESULTS / 'fairness_matrix.csv'} ({len(matrix)} rows)")
    print(
        matrix[
            [
                "model",
                "track",
                "adaptation",
                "future_covariates_used",
                "contamination_status",
                "comparability_status",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
