"""Execute the frozen probe plan and compute headroom.

The probe spec must already be hashed. Every probe result is tagged deployable or
diagnostic, and the headroom arithmetic in Section 19 is applied only to the
tasks a candidate actually names.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import yaml

from . import paths, probes, uncertainty
from .models import PRIMARY_MODELS


def _check_frozen() -> dict:
    payload = (paths.RESULTS / "candidate_probe_spec.json").read_text(encoding="utf-8")
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    recorded = (paths.RESULTS / "candidate_probe_spec.sha256").read_text(encoding="utf-8").split()[0]
    if digest != recorded:
        raise SystemExit("HARD STOP: candidate_probe_spec.json changed after being frozen")
    return json.loads(payload)


def _aggregate(loss: np.ndarray, windows: np.ndarray, window_subset: set[int] | None) -> float:
    """Mean over windows of the per-window mean, matching fev's task aggregation."""
    from . import scoring

    if window_subset is not None:
        mask = np.isin(windows, list(window_subset))
        loss, windows = loss[mask], windows[mask]
    if loss.size == 0 or np.all(np.isnan(loss)):
        return float("nan")
    return scoring.task_aggregate(loss, windows)


def run(track: str, tasks: list[str]) -> tuple[pd.DataFrame, dict]:
    rows = []
    raw: dict[str, dict[str, tuple]] = {}
    baseline_model, aggregates = probes.strongest_model(track, tasks, PRIMARY_MODELS)
    print(f"baseline comparator for this task set: {baseline_model} ({aggregates})", flush=True)
    for task_uid in tasks:
        bank = probes.run_bank(track, task_uid, PRIMARY_MODELS, baseline_model, aggregates)
        deployable_windows = None
        for probe in bank:
            if probe["deployable"] and probe["probe"] != "BASELINE_strongest_foundation_model":
                finite = probe["windows"][np.isfinite(probe["origin_loss"])]
                subset = set(finite.tolist())
                deployable_windows = subset if deployable_windows is None else deployable_windows & subset
        raw[task_uid] = {
            probe["probe"]: (probe["origin_loss"], probe["windows"]) for probe in bank
        }
        for probe in bank:
            rows.append(
                {
                    "track": track,
                    "task_uid": task_uid,
                    "probe": probe["probe"],
                    "deployable": probe["deployable"],
                    "aggregate_all_windows": _aggregate(
                        probe["origin_loss"], probe["windows"], None
                    ),
                    "aggregate_common_windows": _aggregate(
                        probe["origin_loss"], probe["windows"], deployable_windows
                    ),
                    "n_origins": int(np.isfinite(probe["origin_loss"]).sum()),
                    "n_common_windows": len(deployable_windows) if deployable_windows else 0,
                    "detail": json.dumps(probe["detail"], default=str),
                }
            )
    return pd.DataFrame(rows), raw


def headroom(probe_results: pd.DataFrame, candidates: list[dict], track: str) -> pd.DataFrame:
    """Per candidate: oracle headroom, simple recovery, residual (Section 19)."""
    rows = []
    pivot = probe_results.pivot_table(
        index="task_uid", columns="probe", values="aggregate_common_windows", aggfunc="first"
    )
    simple_probes = [
        c
        for c in (
            "P1_validation_selected_model",
            "P3_affine_calibration",
            "P4_supervised_specialist",
            "P5_covariate_ablation",
            "P6_quantile_average_ensemble",
        )
        if c in pivot.columns
    ]
    for candidate in candidates:
        tasks = [t for t in candidate["affected_tasks"] if t in pivot.index]
        if not tasks:
            continue
        subset = pivot.loc[tasks]
        baseline = float(subset["BASELINE_strongest_foundation_model"].mean())
        oracle = float(subset["P0_cross_model_origin_oracle"].mean())
        # A simple fix only counts if it is defined on every task of the candidate.
        # Averaging one over the subset where it happens to exist would compare it
        # against a baseline averaged over a different set of tasks.
        available = [p for p in simple_probes if subset[p].notna().all()]
        skipped = [p for p in simple_probes if p not in available]
        if not available:
            continue
        per_simple = {p: float(subset[p].mean()) for p in available}
        best_simple_name = min(per_simple, key=per_simple.get)
        simple = per_simple[best_simple_name]

        h_oracle = 100.0 * (baseline - oracle) / baseline if baseline > 0 else np.nan
        denominator = baseline - oracle
        r_simple = (baseline - simple) / denominator if denominator > 0 else np.nan
        h_residual = 100.0 * (simple - oracle) / baseline if baseline > 0 else np.nan
        rows.append(
            {
                "candidate_id": candidate["candidate_id"],
                "track": track,
                "n_tasks": len(tasks),
                "baseline_loss": baseline,
                "baseline_model": "strongest primary foundation model per task",
                "oracle_loss": oracle,
                "oracle_is_deployable": False,
                "simple_loss": simple,
                "best_simple_fix": best_simple_name,
                "per_simple_fix_loss": json.dumps(per_simple),
                "simple_fixes_skipped_incomplete": json.dumps(skipped),
                "H_oracle_pct": h_oracle,
                "R_simple": r_simple,
                "H_residual_pct": h_residual,
                "n_affected_families": len(candidate["affected_families"]),
            }
        )
    return pd.DataFrame(rows)


def screen(headroom_table: pd.DataFrame) -> pd.DataFrame:
    config = yaml.safe_load((paths.CONFIGS / "study.yaml").read_text(encoding="utf-8"))
    screen_cfg = config["thresholds"]["headroom_screen"]
    stop_cfg = config["thresholds"]["simple_fix_stop"]
    table = headroom_table.copy()
    table["passes_oracle_headroom"] = table.H_oracle_pct >= screen_cfg["min_oracle_headroom_pct"]
    table["passes_simple_recovery"] = table.R_simple < screen_cfg["max_simple_recovery_fraction"]
    table["passes_residual"] = table.H_residual_pct >= screen_cfg["min_residual_pct"]
    table["passes_families"] = table.n_affected_families >= screen_cfg["min_model_families"]
    table["simple_baseline_solves"] = (
        (table.R_simple >= stop_cfg["solved_recovery_fraction"])
        | (table.H_residual_pct <= stop_cfg["solved_residual_pct"])
    )
    table["passes_headroom_screen"] = (
        table.passes_oracle_headroom
        & table.passes_simple_recovery
        & table.passes_residual
        & table.passes_families
        & ~table.simple_baseline_solves
    )
    table["screen_verdict"] = np.where(
        table.passes_headroom_screen,
        "PROCEED_TO_CONFIRMATION",
        np.where(table.simple_baseline_solves, "SIMPLE_BASELINE_SOLVES", "CHARACTERIZATION_ONLY"),
    )
    return table


def main() -> None:
    spec = _check_frozen()
    candidates = spec["candidates"]
    if not candidates:
        print("no candidates were frozen; nothing to probe")
        pd.DataFrame().to_csv(paths.RESULTS / "probe_results.csv", index=False)
        pd.DataFrame().to_csv(paths.RESULTS / "headroom_table.csv", index=False)
        return
    tasks = sorted({t for candidate in candidates for t in candidate["affected_tasks"]})
    print(f"probing {len(tasks)} tasks for {len(candidates)} candidates")
    probe_results, raw = run(spec["track"], tasks)
    probe_results.to_csv(paths.RESULTS / "probe_results.csv", index=False)
    table = screen(headroom(probe_results, candidates, spec["track"]))

    intervals = {}
    for _, row in table.iterrows():
        candidate = next(c for c in candidates if c["candidate_id"] == row.candidate_id)
        subset = {t: raw[t] for t in candidate["affected_tasks"] if t in raw}
        if subset and row.best_simple_fix in next(iter(subset.values())):
            intervals[row.candidate_id] = uncertainty.headroom_interval(
                subset, row.best_simple_fix
            )
    (paths.RESULTS / "headroom_intervals.json").write_text(
        json.dumps(intervals, indent=2), encoding="utf-8"
    )
    for candidate_id, interval in intervals.items():
        mask = table.candidate_id == candidate_id
        for key in ("H_oracle_pct", "R_simple", "H_residual_pct"):
            if key in interval:
                table.loc[mask, f"{key}_ci_lower"] = interval[key]["ci_lower"]
                table.loc[mask, f"{key}_ci_upper"] = interval[key]["ci_upper"]
    table.to_csv(paths.RESULTS / "headroom_table.csv", index=False)
    print("\nprobe aggregates (common windows):")
    print(
        probe_results.pivot_table(
            index="task_uid", columns="probe", values="aggregate_common_windows"
        )
        .round(4)
        .to_string()
    )
    print("\nheadroom:")
    print(
        table[
            [
                "candidate_id",
                "n_tasks",
                "baseline_loss",
                "oracle_loss",
                "simple_loss",
                "best_simple_fix",
                "H_oracle_pct",
                "R_simple",
                "H_residual_pct",
                "screen_verdict",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
