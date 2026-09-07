"""The information-condition table (Section 28).

The training regimes here are deliberately different. The zero-shot models see a
task for the first time at inference; the specialists are fitted on that task's
own history. That asymmetry is the question, not a flaw: if allowing task
specialisation does not expose headroom, then the headroom is not there to take.
What the table has to make impossible is confusing the two.
"""

from __future__ import annotations

import json

import pandas as pd

from . import paths

PRIMARY = ["chronos-2", "tirex-2", "timesfm-3.0"]


def build() -> pd.DataFrame:
    source = json.loads((paths.RESULTS / "SOURCE_STUDY.json").read_text(encoding="utf-8"))
    env = json.loads((paths.RESULTS / "specialist_env.json").read_text(encoding="utf-8"))
    conversion = json.loads((paths.RESULTS / "conversion_index.json").read_text(encoding="utf-8"))
    quantiles = conversion["tasks"][0]["quantile_levels"]

    rows = []
    for model in PRIMARY:
        rows.append(
            {
                "comparator": model,
                "pretrained": True,
                "task_trained": False,
                "training_cutoff": "not applicable (zero-shot)",
                "target_info": "one univariate target series at a time",
                "covariates": "none",
                "rolling_parameter_updates": False,
                "context_available": "the window's full legal past",
                "evaluation_windows": "every window the task declares",
                "metric": "fev native SQL",
                "quantiles": str(quantiles),
                "contamination_status": source["primary_model_contamination"].get(model),
                "comparison_role": "F_FIXED" if model == "timesfm-3.0" else "envelope member",
            }
        )
    rows.append(
        {
            "comparator": "chronos-2-synth",
            "pretrained": True,
            "task_trained": False,
            "training_cutoff": "not applicable (zero-shot)",
            "target_info": "one univariate target series at a time",
            "covariates": "none",
            "rolling_parameter_updates": False,
            "context_available": "the window's full legal past",
            "evaluation_windows": "every window the task declares",
            "metric": "fev native SQL",
            "quantiles": str(quantiles),
            "contamination_status": "SYNTH_ONLY_DIAGNOSTIC",
            "comparison_role": "diagnostic only, excluded from the envelope and every gate",
        }
    )
    specialists = [("S_ENSEMBLE", "primary strong specialist"), ("S_BEST", "interpretation only")]
    specialists += [
        (row["requested"], "ensemble component")
        for row in env.get("model_availability", [])
        if row.get("available")
    ]
    for name, role in specialists:
        rows.append(
            {
                "comparator": name,
                "pretrained": False,
                "task_trained": True,
                "training_cutoff": "the task's first evaluation cutoff; fitted once, then frozen",
                "target_info": "one univariate target series at a time",
                "covariates": "none",
                "rolling_parameter_updates": False,
                "context_available": "the window's full legal past, as inference input only",
                "evaluation_windows": "every window the task declares",
                "metric": "fev native SQL",
                "quantiles": str(quantiles),
                "contamination_status": "not applicable (no pretraining corpus)",
                "comparison_role": role,
            }
        )
    for name, role in (
        ("linear-ar-specialist", "weak supervised baseline from the source study"),
        ("seasonal-naive", "naive anchor"),
    ):
        rows.append(
            {
                "comparator": name,
                "pretrained": False,
                "task_trained": name == "linear-ar-specialist",
                "training_cutoff": "refitted inside each window from that window's visible history"
                if name == "linear-ar-specialist"
                else "not applicable",
                "target_info": "one univariate target series at a time",
                "covariates": "none",
                "rolling_parameter_updates": name == "linear-ar-specialist",
                "context_available": "the window's full legal past",
                "evaluation_windows": "every window the task declares",
                "metric": "fev native SQL",
                "quantiles": str(quantiles),
                "contamination_status": "not applicable",
                "comparison_role": role,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    matrix = build()
    matrix.to_csv(paths.RESULTS / "fairness_matrix.csv", index=False)
    print(f"wrote {paths.RESULTS / 'fairness_matrix.csv'} ({len(matrix)} rows)")
    print(
        matrix[
            ["comparator", "pretrained", "task_trained", "rolling_parameter_updates", "comparison_role"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
