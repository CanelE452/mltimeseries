"""Score the surviving candidates (Section 23).

Seven components, 33 points. A hard fail is not outscorable: a candidate that
trips one is dropped whatever its total.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import paths

MAX_REPORTED = 3


def _band(value: float, edges: list[float], points: list[int]) -> int:
    if not np.isfinite(value):
        return 0
    for edge, point in zip(edges, points):
        if value < edge:
            return point
    return points[-1]


def score_candidates() -> pd.DataFrame:
    spec = json.loads((paths.RESULTS / "candidate_probe_spec.json").read_text(encoding="utf-8"))
    candidates = {c["candidate_id"]: c for c in spec["candidates"]}
    if not candidates:
        return pd.DataFrame()
    headroom = pd.read_csv(paths.RESULTS / "headroom_table.csv").set_index("candidate_id")
    confirmation_path = paths.RESULTS / "confirmation_results.csv"
    confirmation = (
        pd.read_csv(confirmation_path).set_index("candidate_id")
        if confirmation_path.exists() and confirmation_path.stat().st_size > 10
        else pd.DataFrame()
    )
    literature_path = paths.RESULTS / "literature_audit.json"
    literature = (
        json.loads(literature_path.read_text(encoding="utf-8")) if literature_path.exists() else {}
    )

    rows = []
    for candidate_id, candidate in candidates.items():
        head = headroom.loc[candidate_id] if candidate_id in headroom.index else None
        confirm = (
            confirmation.loc[candidate_id]
            if len(confirmation) and candidate_id in confirmation.index
            else None
        )
        novelty = literature.get(candidate_id, {})
        novelty_status = novelty.get("novelty_status", "UNCLEAR")

        severity = float(candidate["severity"]["median_condition_gap_pct"])
        families = len(candidate["affected_families"])
        h_oracle = float(head.H_oracle_pct) if head is not None else np.nan
        h_residual = float(head.H_residual_pct) if head is not None else np.nan
        confirmed = bool(confirm.confirmed) if confirm is not None else False

        component = {
            "A_failure_severity": _band(severity, [5, 10, 15, 25, 40], [1, 2, 3, 4, 5]),
            "B_cross_model_consistency": min(5, max(0, (families - 1) * 2 + 1)),
            "C_oracle_headroom": _band(h_oracle, [3, 5, 10, 20, 35], [0, 2, 3, 4, 5]),
            "D_residual_after_simple_fix": _band(h_residual, [1.5, 3, 6, 12, 25], [0, 2, 3, 4, 5]),
            "E_holdout_confirmation": (
                5
                if confirmed
                else (
                    sum(
                        int(bool(confirm[f]))
                        for f in ("C1_direction_same", "C2_two_families_positive", "C3_severity")
                    )
                    if confirm is not None
                    else 0
                )
            ),
            "F_novelty_space": {
                "OPEN_SPACE": 5,
                "ADJACENT_CROWDED": 3,
                "UNCLEAR": 2,
                "DIRECTLY_OWNED": 0,
            }.get(novelty_status, 2),
            "G_feasibility": novelty.get("feasibility_points", 2),
        }
        total = sum(component.values())

        hard_fails = []
        if novelty_status == "DIRECTLY_OWNED":
            hard_fails.append("DIRECTLY_OWNED")
        if confirm is not None and not confirmed:
            hard_fails.append("CONFIRMATION_FAILED")
        if confirm is None:
            hard_fails.append("CONFIRMATION_NOT_RUN")
        if np.isfinite(h_oracle) and h_oracle < 3:
            hard_fails.append("ORACLE_HEADROOM_BELOW_3PCT")
        if candidate["n_affected_tasks"] < 2:
            hard_fails.append("ONLY_ONE_TASK")
        if families < 2:
            hard_fails.append("ONLY_ONE_MODEL_FAMILY")
        if head is not None and bool(head.simple_baseline_solves):
            hard_fails.append("SIMPLE_BASELINE_SOLVES")

        rows.append(
            {
                "candidate_id": candidate_id,
                "observed_condition": candidate["observed_condition"],
                "n_affected_tasks": candidate["n_affected_tasks"],
                "n_affected_families": families,
                "severity_pct": severity,
                "H_oracle_pct": h_oracle,
                "R_simple": float(head.R_simple) if head is not None else np.nan,
                "H_residual_pct": h_residual,
                "confirmed": confirmed,
                "novelty_status": novelty_status,
                **component,
                "total_score": total,
                "max_score": 33,
                "hard_fails": json.dumps(hard_fails),
                "eliminated": bool(hard_fails),
            }
        )
    return pd.DataFrame(rows).sort_values(["eliminated", "total_score"], ascending=[True, False])


def main() -> None:
    ranking = score_candidates()
    ranking.to_csv(paths.RESULTS / "candidate_ranking.csv", index=False)
    if ranking.empty:
        print("no candidates to rank")
        return
    print(
        ranking[
            [
                "candidate_id",
                "n_affected_tasks",
                "n_affected_families",
                "severity_pct",
                "H_oracle_pct",
                "R_simple",
                "H_residual_pct",
                "confirmed",
                "novelty_status",
                "total_score",
                "eliminated",
                "hard_fails",
            ]
        ].to_string(index=False)
    )
    survivors = ranking[~ranking.eliminated]
    print(f"\nsurviving candidates: {len(survivors)} (reporting at most {MAX_REPORTED})")


if __name__ == "__main__":
    main()
