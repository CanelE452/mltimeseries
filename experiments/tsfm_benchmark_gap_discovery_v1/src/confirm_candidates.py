"""Open the holdout split and test whether the candidates reproduce.

Runs only after candidate_probe_spec.json is frozen. The condition definitions,
model set, baselines, metric and probes are the discovery ones, unchanged;
Section 21 asks confirmation to reproduce the direction and a real magnitude, not
the exact discovery thresholds.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import yaml

from . import paths, run_probes
from .descriptors import assign_buckets, build_task_metadata
from .failure_map import DESCRIPTOR_COLUMNS
from .models import PRIMARY_MODELS


def _frozen_spec() -> dict:
    payload = (paths.RESULTS / "candidate_probe_spec.json").read_text(encoding="utf-8")
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    recorded = (paths.RESULTS / "candidate_probe_spec.sha256").read_text(encoding="utf-8").split()[0]
    if digest != recorded:
        raise SystemExit("HARD STOP: candidate spec changed before confirmation")
    return json.loads(payload)


def _confirmation_metadata() -> pd.DataFrame:
    """Confirmation descriptors, bucketed with the discovery cut points."""
    cuts = json.loads(
        (paths.RESULTS / "descriptor_thresholds.json").read_text(encoding="utf-8")
    )["cuts"]
    metadata = pd.read_csv(paths.RESULTS / "task_metadata.csv")
    if "D1_horizon_ratio" not in metadata.columns:
        metadata, _ = assign_buckets(build_task_metadata(), cuts)
    return metadata


def condition_gap(results: pd.DataFrame, metadata: pd.DataFrame, descriptor: str, bucket: str) -> dict:
    column = DESCRIPTOR_COLUMNS[descriptor]
    merged = results.merge(metadata[["task_uid", column]], on="task_uid", how="inner")
    inside = merged[merged[column] == bucket]
    outside = merged[merged[column] != bucket]
    per_model = {}
    for model in PRIMARY_MODELS:
        model_inside = inside[inside.model == model].relative_to_naive
        model_outside = outside[outside.model == model].relative_to_naive
        if model_inside.empty or model_outside.empty:
            per_model[model] = np.nan
            continue
        inside_median = float(model_inside.median())
        outside_median = float(model_outside.median())
        per_model[model] = (
            100.0 * (inside_median - outside_median) / outside_median
            if outside_median > 0
            else np.nan
        )
    positive = [m for m, v in per_model.items() if np.isfinite(v) and v > 0]
    return {
        "per_model_gap_pct": per_model,
        "n_models_positive": len(positive),
        "median_gap_pct": float(np.nanmedian(list(per_model.values()))),
        "tasks_inside": sorted(inside.task_uid.unique().tolist()),
    }


def main() -> None:
    spec = _frozen_spec()
    candidates = spec["candidates"]
    if not candidates:
        print("no candidates frozen; confirmation not opened")
        pd.DataFrame().to_csv(paths.RESULTS / "confirmation_results.csv", index=False)
        return

    config = yaml.safe_load((paths.CONFIGS / "study.yaml").read_text(encoding="utf-8"))
    gate = config["thresholds"]["confirmation"]
    results = pd.read_csv(paths.RESULTS / "benchmark_results.csv")
    confirmation = results[
        (results.split == "confirmation") & (results.track == spec["track"]) & (results.status == "OK")
    ]
    if confirmation.empty:
        raise SystemExit("confirmation split has no results - run run_models --split confirmation")
    metadata = _confirmation_metadata()
    metadata = metadata[metadata.split == "confirmation"]

    audit = pd.read_csv(paths.RESULTS / "model_audit.csv").set_index("model_id")
    rows = []
    for candidate in candidates:
        descriptor, bucket = candidate["descriptor"], candidate["bucket"]
        observed = condition_gap(confirmation, metadata, descriptor, bucket)
        tasks = observed["tasks_inside"]
        if tasks:
            probe_rows, _ = run_probes.run(spec["track"], tasks)
            table = run_probes.headroom(
                probe_rows, [{**candidate, "affected_tasks": tasks}], spec["track"]
            )
        else:
            probe_rows, table = pd.DataFrame(), pd.DataFrame()

        families = sorted(
            {
                audit.loc[m, "architecture_family"]
                for m, v in observed["per_model_gap_pct"].items()
                if np.isfinite(v) and v > 0 and m in audit.index
            }
        )
        h_oracle = float(table.H_oracle_pct.iloc[0]) if len(table) else np.nan
        h_residual = float(table.H_residual_pct.iloc[0]) if len(table) else np.nan
        r_simple = float(table.R_simple.iloc[0]) if len(table) else np.nan
        checks = {
            "C1_direction_same": bool(observed["median_gap_pct"] > 0),
            "C2_two_families_positive": len(families) >= 2,
            "C3_severity": bool(observed["median_gap_pct"] >= gate["min_severity_pct"]),
            "C4_oracle_headroom": bool(h_oracle >= gate["min_oracle_headroom_pct"]),
            "C5_simple_residual": bool(h_residual >= gate["min_residual_pct"]),
        }
        rows.append(
            {
                "candidate_id": candidate["candidate_id"],
                "descriptor": descriptor,
                "bucket": bucket,
                "n_confirmation_tasks": len(tasks),
                "confirmation_tasks": json.dumps(tasks),
                "per_model_gap_pct": json.dumps(observed["per_model_gap_pct"]),
                "n_models_positive": observed["n_models_positive"],
                "positive_families": json.dumps(families),
                "median_gap_pct": observed["median_gap_pct"],
                "H_oracle_pct": h_oracle,
                "R_simple": r_simple,
                "H_residual_pct": h_residual,
                **checks,
                "confirmed": all(checks.values()),
                "verdict": "CONFIRMED" if all(checks.values()) else "DISCOVERY_ONLY",
            }
        )
        if len(probe_rows):
            probe_rows.assign(split="confirmation", candidate_id=candidate["candidate_id"]).to_csv(
                paths.RESULTS / f"confirmation_probes_{candidate['candidate_id']}.csv", index=False
            )

    table = pd.DataFrame(rows)
    table.to_csv(paths.RESULTS / "confirmation_results.csv", index=False)
    print(table.drop(columns=["confirmation_tasks", "per_model_gap_pct"]).to_string(index=False))


if __name__ == "__main__":
    main()
