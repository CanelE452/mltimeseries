"""Write verdict.json and STATUS.md in the order Section 35 requires."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import paths
from .models import DIAGNOSTIC_MODELS, PRIMARY_MODELS


def _read_csv(name: str) -> pd.DataFrame:
    path = paths.RESULTS / name
    if not path.exists() or path.stat().st_size < 5:
        return pd.DataFrame()
    return pd.read_csv(path)


def _read_json(name: str):
    path = paths.RESULTS / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def decide_verdict(
    gate: pd.DataFrame, headroom: pd.DataFrame, confirmation: pd.DataFrame, ranking: pd.DataFrame
) -> tuple[str, str, dict | None]:
    integrity = _read_csv("integrity_checks.csv")
    if len(integrity) and (integrity.result == "FAIL").any():
        failed = integrity[integrity.result == "FAIL"].check.tolist()
        return "NOT_EVALUATED", f"integrity checks failed: {failed}", None
    if gate.empty or not gate.passes_failure_gate.any():
        return (
            "NO_STRONG_GAP_FOUND",
            "no descriptor bucket met the registered failure gate on the discovery split",
            None,
        )
    if headroom.empty:
        return "NO_STRONG_GAP_FOUND", "candidates existed but no probe produced headroom", None
    if headroom.simple_baseline_solves.any() and not headroom.passes_headroom_screen.any():
        return (
            "CHARACTERIZATION_ONLY",
            "every candidate is absorbed by a simple deployable fix",
            None,
        )
    if not headroom.passes_headroom_screen.any():
        return (
            "CHARACTERIZATION_ONLY",
            "no candidate cleared the headroom screen",
            None,
        )
    survivors = ranking[~ranking.eliminated] if len(ranking) else pd.DataFrame()
    if survivors.empty:
        return (
            "CHARACTERIZATION_ONLY",
            "candidates were characterised but every one tripped a hard fail",
            None,
        )
    top = survivors.iloc[0]
    return "GAP_CANDIDATE_READY", "a candidate cleared every gate", top.to_dict()


def build_verdict() -> dict:
    results = _read_csv("benchmark_results.csv")
    selected = _read_json("selected_tasks.json")
    gate = _read_csv("failure_candidates_gate.csv")
    headroom = _read_csv("headroom_table.csv")
    confirmation = _read_csv("confirmation_results.csv")
    ranking = _read_csv("candidate_ranking.csv")
    spec = _read_json("candidate_probe_spec.json")

    token, reason, top = decide_verdict(gate, headroom, confirmation, ranking)
    top_candidate = None
    if top is not None:
        candidate_id = top["candidate_id"]
        confirm_row = (
            confirmation[confirmation.candidate_id == candidate_id]
            if len(confirmation)
            else pd.DataFrame()
        )
        head_row = headroom[headroom.candidate_id == candidate_id]
        top_candidate = {
            "id": candidate_id,
            "condition": top["observed_condition"],
            "affected_models": next(
                (c["affected_models"] for c in spec["candidates"] if c["candidate_id"] == candidate_id),
                [],
            ),
            "discovery_failure_gap_pct": float(top["severity_pct"]),
            "confirmation_gap_pct": float(confirm_row.median_gap_pct.iloc[0])
            if len(confirm_row)
            else None,
            "oracle_headroom_pct": float(head_row.H_oracle_pct.iloc[0]) if len(head_row) else None,
            "simple_recovery_fraction": float(head_row.R_simple.iloc[0]) if len(head_row) else None,
            "residual_headroom_pct": float(head_row.H_residual_pct.iloc[0]) if len(head_row) else None,
            "novelty_status": top["novelty_status"],
            "score": float(top["total_score"]),
        }

    integrity = _read_csv("integrity_checks.csv")
    ok = results[results.status == "OK"] if len(results) else pd.DataFrame()
    return {
        "experiment": "TSFM-BENCHMARK-GAP-DISCOVERY-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "execution_status": "COMPLETED" if len(ok) else "NOT_RUN",
        "final_token": token,
        "final_token_reason": reason,
        "benchmark": "fev-bench",
        "primary_models": PRIMARY_MODELS,
        "diagnostic_models": DIAGNOSTIC_MODELS,
        "tracks_run": sorted(ok.track.unique().tolist()) if len(ok) else [],
        "discovery_tasks": len(selected["discovery_tasks"]),
        "confirmation_tasks": len(selected["confirmation_tasks"]),
        "confirmation_opened": bool(len(ok) and (ok.split == "confirmation").any()),
        "n_gap_candidates": int(gate.passes_failure_gate.sum()) if len(gate) else 0,
        "top_candidate": top_candidate,
        "integrity": {
            "task_selection_frozen": bool(
                len(integrity) and (integrity[integrity.check == "A01"].result == "PASS").all()
            ),
            "confirmation_untouched_until_freeze": bool(
                len(integrity) and (integrity[integrity.check == "A11"].result == "PASS").all()
            ),
            "information_conditions_fair": bool(
                len(integrity)
                and (integrity[integrity.check.isin(["A03", "A04", "A18"])].result == "PASS").all()
            ),
            "metric_contract_valid": bool(
                len(integrity) and (integrity[integrity.check == "A06"].result == "PASS").all()
            ),
            "checks_passed": int((integrity.result == "PASS").sum()) if len(integrity) else 0,
            "checks_total": int(len(integrity)),
        },
        "notes": [],
    }


def main() -> None:
    verdict = build_verdict()
    (paths.RESULTS / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    print(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    main()
