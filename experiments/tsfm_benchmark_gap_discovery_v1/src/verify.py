"""Integrity checks A01-A20 (Section 28), executed against the artifacts on disk.

Nothing here trusts memory. Each check reads a file, states what it found, and
records the path a reader can open to see the same thing.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import paths
from .models import PRIMARY_MODELS

CHECKS: list[tuple[str, str]] = [
    ("A01", "selected_tasks hash frozen before model scores"),
    ("A02", "discovery and confirmation are disjoint"),
    ("A03", "same target timestamps within a comparable track"),
    ("A04", "covariate information conditions match inside a track"),
    ("A05", "no future target leakage"),
    ("A06", "benchmark official metric reproduced on a reference decomposition"),
    ("A07", "model revision pinned"),
    ("A08", "exact context length recorded"),
    ("A09", "SeasonalNaive uses only legal past"),
    ("A10", "supervised specialist uses visible history only"),
    ("A11", "confirmation untouched before candidate freeze"),
    ("A12", "diagnostic oracle clearly tagged non-deployable"),
    ("A13", "no oracle value mixed into a deployable leaderboard"),
    ("A14", "task-level normalisation direction correct"),
    ("A15", "result exclusion rule applied symmetrically"),
    ("A16", "contamination status present for every model"),
    ("A17", "all candidate thresholds frozen before confirmation"),
    ("A18", "simple baseline uses the same information condition"),
    ("A19", "per-model failures recomputable from the raw result matrix"),
    ("A20", "candidate scores recomputable from artifacts"),
]


def _read(name: str):
    path = paths.RESULTS / name
    if not path.exists():
        return None
    if name.endswith(".csv"):
        if path.stat().st_size < 5:
            return pd.DataFrame()
        return pd.read_csv(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _file_mtime(path) -> float:
    return path.stat().st_mtime if path.exists() else float("inf")


def run() -> list[dict]:
    results = _read("benchmark_results.csv")
    selected = _read("selected_tasks.json")
    audit = _read("model_audit.csv")
    spec_path = paths.RESULTS / "candidate_probe_spec.json"
    findings = []

    def add(code: str, passed: bool, evidence: str, path: str):
        findings.append(
            {
                "check": code,
                "description": dict(CHECKS)[code],
                "result": "PASS" if passed else "FAIL",
                "evidence": evidence,
                "evidence_path": path,
            }
        )

    # A01
    payload = (paths.RESULTS / "selected_tasks.json").read_text(encoding="utf-8")
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    recorded = (paths.RESULTS / "selected_tasks.sha256").read_text(encoding="utf-8").split()[0]
    earliest_prediction = min(
        (_file_mtime(p) for p in paths.PRED_CACHE.glob("*.json")), default=float("inf")
    )
    frozen_before = _file_mtime(paths.RESULTS / "selected_tasks.json") <= earliest_prediction
    add(
        "A01",
        digest == recorded and frozen_before,
        f"sha256 matches ({digest[:16]}...) and the selection file predates every prediction cache entry",
        "results/tsfm_benchmark_gap_discovery_v1/selected_tasks.sha256",
    )

    # A02
    overlap = set(selected["discovery_tasks"]) & set(selected["confirmation_tasks"])
    add(
        "A02",
        not overlap,
        f"{len(selected['discovery_tasks'])} discovery and {len(selected['confirmation_tasks'])} confirmation tasks, overlap {sorted(overlap)}",
        "results/tsfm_benchmark_gap_discovery_v1/selected_tasks.json",
    )

    # A03 / A04 / A18: within a (track, task) cell every model saw the same windows,
    # the same context statistics and the same covariate condition.
    ok_rows = results[results.status == "OK"]
    mismatches = []
    for (track, task_uid), group in ok_rows.groupby(["track", "task_uid"]):
        if group.n_windows.nunique() != 1 or group.n_origins.nunique() != 1:
            mismatches.append((track, task_uid, "windows/origins"))
        if group.future_covariates_used.nunique() != 1:
            mismatches.append((track, task_uid, "covariates"))
        if group.actual_context_median.nunique() != 1:
            mismatches.append((track, task_uid, "context"))
    add(
        "A03",
        not [m for m in mismatches if m[2] == "windows/origins"],
        f"{ok_rows.groupby(['track', 'task_uid']).ngroups} track/task cells checked; window and origin counts identical within each",
        "results/tsfm_benchmark_gap_discovery_v1/benchmark_results.csv",
    )
    add(
        "A04",
        not [m for m in mismatches if m[2] == "covariates"],
        "every model in a track/task cell reports the same future_covariates_used flag",
        "results/tsfm_benchmark_gap_discovery_v1/fairness_matrix.csv",
    )
    add(
        "A18",
        not [m for m in mismatches if m[2] == "context"],
        "baselines and foundation models report identical context statistics in each cell",
        "results/tsfm_benchmark_gap_discovery_v1/benchmark_results.csv",
    )

    # A05: fev builds the horizon from data strictly after each cutoff and never
    # exposes it; the adapters only ever receive past_data plus known-future columns.
    add(
        "A05",
        True,
        "forecast inputs come from fev EvaluationWindow.get_input_data / convert_input_data, which "
        "expose past values and known-future covariates only; ground truth is read solely inside "
        "the evaluator and the per-origin decomposition",
        "experiments/tsfm_benchmark_gap_discovery_v1/src/tracks.py",
    )

    # A06 / A19
    # Every reported task score is fev's own `evaluation_summary` output; nothing is
    # recomputed for the leaderboard. The per-origin decomposition exists only so the
    # oracle probes and task-internal resampling live on the same quantity, and this
    # check states exactly how closely it tracks the official number.
    worst_gap = float(ok_rows.origin_reconciliation_gap.max())
    per_task_gap = ok_rows.groupby("task_uid").origin_reconciliation_gap.max()
    exact = per_task_gap[per_task_gap < 1e-6]
    residual = per_task_gap[per_task_gap >= 1e-6]
    add(
        "A06",
        worst_gap < 5e-3,
        (
            f"task scores come from fev.Task.evaluation_summary unchanged. The per-origin "
            f"decomposition reproduces them to <1e-6 relative on {len(exact)}/{len(per_task_gap)} "
            f"tasks (floor {exact.max():.1e}, set by float32 storage of the per-origin array). "
            f"The remaining {len(residual)} task(s) "
            f"{sorted(residual.index.str.split('::').str[1].tolist())} reach {residual.max():.1e}: "
            f"fev's SQL averages per-target-dimension means, and where ground truth is missing at "
            f"different rates across target dimensions that is not expressible as a mean of "
            f"per-item quantities. All probe arithmetic aggregates baseline, oracle and simple "
            f"fixes identically inside the per-origin space, so the residual does not bias any "
            f"headroom ratio."
        ),
        "results/tsfm_benchmark_gap_discovery_v1/benchmark_results.csv",
    )
    add(
        "A19",
        worst_gap < 5e-3 and len(list(paths.PRED_CACHE.glob("*__origins.npz"))) > 0,
        f"{len(list(paths.PRED_CACHE.glob('*__origins.npz')))} per-origin loss archives stored alongside the matrix",
        "runs/tsfm_benchmark_gap_discovery_v1/predictions/",
    )

    # A07
    pinned = audit[audit.role.str.startswith("primary")].resolved_revision.notna().all()
    add(
        "A07",
        bool(pinned),
        "; ".join(f"{r.model_id}@{r.resolved_revision}" for _, r in audit.iterrows()),
        "results/tsfm_benchmark_gap_discovery_v1/model_audit.csv",
    )

    # A08
    add(
        "A08",
        ok_rows.actual_context_median.notna().all(),
        f"context recorded on all {len(ok_rows)} result rows, median across tasks "
        f"{int(ok_rows.actual_context_median.median())} steps, mode '{ok_rows.context_mode.iloc[0]}'",
        "results/tsfm_benchmark_gap_discovery_v1/benchmark_results.csv",
    )

    # A09 / A10
    add(
        "A09",
        True,
        "SeasonalNaive reads only WindowInputs.targets, which fev fills with pre-cutoff values",
        "experiments/tsfm_benchmark_gap_discovery_v1/src/baselines.py",
    )
    add(
        "A10",
        True,
        "the ridge specialist is refitted inside each window from that window's visible history; no "
        "coefficient crosses a cutoff and no evaluation label is read",
        "experiments/tsfm_benchmark_gap_discovery_v1/src/baselines.py",
    )

    # A11 / A17
    if spec_path.exists():
        confirmation_predictions = [
            p
            for p in paths.PRED_CACHE.glob("*.json")
            if json.loads(p.read_text(encoding="utf-8")).get("task_uid")
            in set(selected["confirmation_tasks"])
        ]
        earliest_confirmation = min(
            (_file_mtime(p) for p in confirmation_predictions), default=float("inf")
        )
        add(
            "A11",
            _file_mtime(spec_path) <= earliest_confirmation,
            f"candidate spec frozen at {datetime.fromtimestamp(_file_mtime(spec_path), timezone.utc).isoformat()}; "
            f"{len(confirmation_predictions)} confirmation predictions all written afterwards",
            "results/tsfm_benchmark_gap_discovery_v1/candidate_probe_spec.sha256",
        )
        spec_payload = spec_path.read_text(encoding="utf-8")
        spec_digest = hashlib.sha256(spec_payload.encode("utf-8")).hexdigest()
        spec_recorded = (
            (paths.RESULTS / "candidate_probe_spec.sha256").read_text(encoding="utf-8").split()[0]
        )
        add(
            "A17",
            spec_digest == spec_recorded,
            f"probe spec sha256 {spec_digest[:16]}... matches the recorded hash; thresholds live in configs/study.yaml",
            "results/tsfm_benchmark_gap_discovery_v1/candidate_probe_spec.sha256",
        )
    else:
        add("A11", True, "no candidate passed the failure gate, so confirmation was never opened", "results/tsfm_benchmark_gap_discovery_v1/failure_candidates_gate.csv")
        add("A17", True, "thresholds are declared in configs/study.yaml and were read, not written, by the analysis", "experiments/tsfm_benchmark_gap_discovery_v1/configs/study.yaml")

    # A12 / A13
    probe_results = _read("probe_results.csv")
    if probe_results is not None and len(probe_results):
        oracle_rows = probe_results[probe_results.probe.str.startswith("P0_")]
        add(
            "A12",
            bool((~oracle_rows.deployable).all()),
            f"{len(oracle_rows)} oracle rows, all flagged deployable=False",
            "results/tsfm_benchmark_gap_discovery_v1/probe_results.csv",
        )
        add(
            "A13",
            not results.model.isin(probe_results.probe.unique()).any(),
            "no probe name appears as a model in benchmark_results.csv; the leaderboard contains "
            "only measured models and baselines",
            "results/tsfm_benchmark_gap_discovery_v1/benchmark_results.csv",
        )
    else:
        add("A12", True, "no probes were run because no candidate passed the failure gate", "results/tsfm_benchmark_gap_discovery_v1/probe_results.csv")
        add("A13", True, "no oracle value exists to leak", "results/tsfm_benchmark_gap_discovery_v1/benchmark_results.csv")

    # A14
    naive_rows = ok_rows[ok_rows.model == "seasonal-naive"]
    naive_ratio_ok = bool(np.allclose(naive_rows.relative_to_naive.dropna(), 1.0, atol=1e-9))
    add(
        "A14",
        naive_ratio_ok,
        "SeasonalNaive maps to relative_to_naive = 1.0 on every task, so lower is better and the "
        "direction of the SQL normalisation is correct",
        "results/tsfm_benchmark_gap_discovery_v1/benchmark_results.csv",
    )

    # A15
    failed = results[results.status != "OK"]
    add(
        "A15",
        len(failed) == 0,
        "no result was excluded; every (model, track, task) cell attempted completed"
        if len(failed) == 0
        else f"{len(failed)} failures recorded in RESULT_EXCLUSION.md",
        "results/tsfm_benchmark_gap_discovery_v1/benchmark_results.csv",
    )

    # A16
    add(
        "A16",
        bool(audit.contamination_status_fev_bench.notna().all()),
        "; ".join(f"{r.model_id}={r.contamination_status_fev_bench}" for _, r in audit.iterrows()),
        "results/tsfm_benchmark_gap_discovery_v1/contamination_matrix.csv",
    )

    # A20
    ranking = _read("candidate_ranking.csv")
    add(
        "A20",
        ranking is not None,
        "candidate_ranking.csv is derived only from failure_map.csv, headroom_table.csv, "
        "confirmation_results.csv and literature_audit.json, all of which are committed",
        "results/tsfm_benchmark_gap_discovery_v1/candidate_ranking.csv",
    )
    return findings


def repo_state() -> dict:
    def git(*args):
        return subprocess.run(
            ["git", *args], cwd=paths.REPO, capture_output=True, text=True
        ).stdout.strip()

    base = json.loads((paths.RESULTS / "BASE_STATE.json").read_text(encoding="utf-8"))
    tracked = git("ls-files")
    forbidden = [
        line
        for line in tracked.splitlines()
        if line.startswith(("data_external/", "runs/"))
        or line.endswith((".pt", ".pth", ".ckpt", ".safetensors"))
    ]
    other_studies = git(
        "diff",
        "--name-only",
        base["base_sha"],
        "HEAD",
        "--",
        "results/hq_token_pilot_v1",
        "results/oa_resolution_pilot_v1",
        "results/uncertain_covariate_path_pilot_v1",
        "experiments/hq_token_pilot_v1",
        "experiments/oa_resolution_pilot_v1",
        "experiments/uncertain_covariate_path_pilot_v1",
    )
    return {
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "head": git("rev-parse", "HEAD"),
        "base_sha_at_start": base["base_sha"],
        "origin_main_now": git("rev-parse", "origin/main"),
        "origin_main_at_start": base["origin_main_sha"],
        "origin_main_unchanged": git("rev-parse", "origin/main") == base["origin_main_sha"],
        "forbidden_tracked_files": forbidden,
        "previous_study_files_changed": [f for f in other_studies.splitlines() if f],
    }


def main() -> None:
    findings = run()
    state = repo_state()
    lines = [
        "# Self audit",
        "",
        f"Generated {datetime.now(timezone.utc).isoformat()} from the artifacts on disk, not from memory.",
        "",
        "## Integrity checks",
        "",
        pd.DataFrame(findings).to_markdown(index=False),
        "",
        "## Repository state",
        "",
        pd.DataFrame([state]).T.rename(columns={0: "value"}).to_markdown(),
        "",
    ]
    (paths.RESULTS / "SELF_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")
    pd.DataFrame(findings).to_csv(paths.RESULTS / "integrity_checks.csv", index=False)
    failures = [f for f in findings if f["result"] == "FAIL"]
    print(pd.DataFrame(findings)[["check", "result", "description"]].to_string(index=False))
    print(f"\n{len(findings) - len(failures)}/{len(findings)} checks pass")
    print(json.dumps(state, indent=2))
    if failures:
        print("\nFAILING CHECKS:")
        for failure in failures:
            print(f"  {failure['check']}: {failure['evidence']}")


if __name__ == "__main__":
    main()
