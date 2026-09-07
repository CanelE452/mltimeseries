"""Pin the source study and re-derive its D1-long development set from its files.

Section 1 forbids hardcoding a SHA from conversation, so the source branch tip is
read from the remote at run time. The four development tasks are likewise
re-derived from the source `task_metadata.csv`; the names the instruction lists
are used only as an assertion, and a mismatch is a hard stop.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone

import pandas as pd

from . import paths

# Asserted, never assumed: these are what the source study is expected to hold.
EXPECTED_DEVELOPMENT_TASKS = [
    "fevbench::ETT_1D::ETT_1D",
    "fevbench::LOOP_SEATTLE_1D::LOOP_SEATTLE_1D",
    "fevbench::australian_tourism::australian_tourism",
    "fevbench::hermes::hermes",
]

REQUIRED_SOURCE_FILES = [
    "selected_tasks.json",
    "selected_tasks.sha256",
    "task_metadata.csv",
    "task_pool.csv",
    "benchmark_results.csv",
    "failure_candidates_gate.csv",
    "model_audit.csv",
    "benchmark_audit.json",
    "descriptor_thresholds.json",
    "STATUS.md",
    "verdict.json",
]


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=paths.REPO, capture_output=True, text=True
    ).stdout.strip()


def source_remote_sha() -> str:
    line = git("ls-remote", "origin", f"refs/heads/{paths.SOURCE_BRANCH}")
    if not line:
        raise SystemExit(
            f"SOURCE_CONTRACT_MISMATCH: origin has no branch {paths.SOURCE_BRANCH}"
        )
    return line.split()[0]


def file_digest(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def development_tasks(metadata: pd.DataFrame) -> list[str]:
    """Every discovery task the source study bucketed as D1 == long."""
    subset = metadata[(metadata.split == "discovery") & (metadata.D1_horizon_ratio == "long")]
    tasks = sorted(subset.task_uid.tolist())
    if tasks != sorted(EXPECTED_DEVELOPMENT_TASKS):
        raise SystemExit(
            "SOURCE_CONTRACT_MISMATCH: D1-long discovery tasks re-derived from "
            f"task_metadata.csv are {tasks}, expected {sorted(EXPECTED_DEVELOPMENT_TASKS)}"
        )
    return tasks


def build() -> dict:
    missing = [f for f in REQUIRED_SOURCE_FILES if not (paths.SOURCE_RESULTS / f).exists()]
    if missing:
        raise SystemExit(f"SOURCE_CONTRACT_MISMATCH: source files missing {missing}")

    metadata = pd.read_csv(paths.SOURCE_RESULTS / "task_metadata.csv")
    tasks = development_tasks(metadata)

    gate = pd.read_csv(paths.SOURCE_RESULTS / "failure_candidates_gate.csv")
    d1_long = gate[(gate.descriptor == "D1") & (gate.bucket == "long")]
    if len(d1_long) != 1:
        raise SystemExit("SOURCE_CONTRACT_MISMATCH: no unique D1/long row in the source gate table")
    d1_long = d1_long.iloc[0]

    verdict = json.loads((paths.SOURCE_RESULTS / "verdict.json").read_text(encoding="utf-8"))
    benchmark = json.loads((paths.SOURCE_RESULTS / "benchmark_audit.json").read_text(encoding="utf-8"))
    thresholds = json.loads(
        (paths.SOURCE_RESULTS / "descriptor_thresholds.json").read_text(encoding="utf-8")
    )
    selected = json.loads((paths.SOURCE_RESULTS / "selected_tasks.json").read_text(encoding="utf-8"))
    audit = pd.read_csv(paths.SOURCE_RESULTS / "model_audit.csv")

    return {
        "experiment": "TSFM-LONG-HORIZON-SPECIALIST-CLOSURE-v1",
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo": "CanelE452/mltimeseries",
        "source_branch": paths.SOURCE_BRANCH,
        "source_remote_sha": source_remote_sha(),
        "source_local_base_sha": git("rev-parse", "HEAD"),
        "source_results_root": "results/tsfm_benchmark_gap_discovery_v1",
        "source_benchmark": benchmark["primary_benchmark"]["name"],
        "source_benchmark_commit": benchmark["primary_benchmark"]["pinned_commit"],
        "source_benchmark_tasks_sha256": benchmark["primary_benchmark"]["task_definition_sha256"],
        "selected_tasks_sha256": (paths.SOURCE_RESULTS / "selected_tasks.sha256")
        .read_text(encoding="utf-8")
        .split()[0],
        "source_final_token": verdict["final_token"],
        "source_d1_long_effect": {
            "median_condition_gap_pct": float(d1_long.median_condition_gap_pct),
            "n_tasks_in_bucket": int(d1_long.n_tasks_in_bucket),
            "n_affected_families": int(d1_long.n_affected_families),
            "verdict": d1_long.verdict,
        },
        "source_d1_long_regret": {
            "median_regret_pct": float(d1_long.median_regret_pct),
            "gate_min_regret_pct": 8.0,
            "why_it_failed_the_gate": (
                "The condition effect was large and shared across all three families, but the "
                "median regret against the best evaluated deployable estimator was far below the "
                "registered 8% threshold: on these tasks no other evaluated model did better."
            ),
        },
        "d1_horizon_ratio_thresholds": thresholds["cuts"]["horizon_to_context_ratio"],
        "d1_threshold_source": "results/tsfm_benchmark_gap_discovery_v1/descriptor_thresholds.json",
        "development_tasks": tasks,
        "previously_used_tasks": sorted(
            set(selected["discovery_tasks"]) | set(selected["confirmation_tasks"])
        ),
        "primary_model_revisions": {
            row.model_id: row.resolved_revision
            for _, row in audit[audit.role == "primary"].iterrows()
        },
        "primary_model_contamination": {
            row.model_id: row.contamination_status_fev_bench
            for _, row in audit[audit.role == "primary"].iterrows()
        },
        "files_read": {
            name: file_digest(paths.SOURCE_RESULTS / name) for name in REQUIRED_SOURCE_FILES
        },
    }


def main() -> None:
    payload = build()
    out = paths.RESULTS / "SOURCE_STUDY.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    print(f"source branch tip: {payload['source_remote_sha']}")
    print(f"source verdict:    {payload['source_final_token']}")
    print(f"D1/long effect:    {payload['source_d1_long_effect']}")
    print(f"D1/long regret:    {payload['source_d1_long_regret']['median_regret_pct']:.2f}%")
    print("development tasks:")
    for task in payload["development_tasks"]:
        print("  ", task)
    print(f"model revisions:   {payload['primary_model_revisions']}")


if __name__ == "__main__":
    main()
